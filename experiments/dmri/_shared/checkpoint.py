"""Load public reference weights or a saved B3S run with strict EMA checks."""

import hashlib
from contextlib import closing
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

from prism.config import build_artifact_config, load_artifact_config
from prism.config.aliases import resolve_class_path
from prism.dmri.simulators.models import (
    Ball3StickSharedDiffusivity,
    MultiShellBall3StickSharedDiffusivity,
)
from prism.simulators.config import resolve_simulator_model
from prism.train.build_model import build_model
from prism.train.build_simulator import build_simulator
from prism.train.checkpointing import CheckpointManager
from prism.train.utils import load_cfg, load_checkpoint, select_checkpoint_params

from .acquisition import protocol_metadata
from .storage import canonical_hash, file_manifest, file_sha256

HF_REPO = "manugloeck/dmri-pretrained"
MODEL_NAME = "msb3s_2_4_6_128"
HF_REVISION = "017561ffbf77b109c9fe6c3d9c7d48831780ddc3"
CHECKPOINT_STEPS = {"best": 1520000, "latest": 2280000}


LEGACY_TARGETS = {
    "experiments.dmri_paper.runtime.PaperDMRIModel": "experiments.dmri._shared.model.PaperDMRIModel",
    "experiments.dmri_paper.acquisition.random_ukb_acquisition": "experiments.dmri._shared.acquisition.random_ukb_acquisition",
    "experiments.dmri_paper.acquisition.random_hcp_acquisition": "experiments.dmri._shared.acquisition.random_hcp_acquisition",
    "experiments.dmri.calibration.model.PaperDMRIModel": "experiments.dmri._shared.model.PaperDMRIModel",
    "experiments.dmri.calibration.acquisition.random_ukb_acquisition": "experiments.dmri._shared.acquisition.random_ukb_acquisition",
    "experiments.dmri.calibration.acquisition.random_hcp_acquisition": "experiments.dmri._shared.acquisition.random_hcp_acquisition",
}


def _physical_parameter_shapes(sim_type):
    instance = sim_type.from_theta(np.zeros(sim_type.theta_dim, dtype=np.float32))

    def shapes(component):
        return {name: list(np.shape(value)) for name, value in component.params.items()}

    return {
        "model_fractions": list(instance.model_fractions.shape),
        "shared": shapes(instance.shared_parameter),
        "model_components": [
            shapes(component) for component in instance.model_compartments
        ],
        "noise_components": [
            shapes(component) for component in instance.noise_compartments
        ],
    }


def apply_ema(checkpoint, model):
    """Restore strict EMA parameters and verify every finite parameter leaf."""
    params = select_checkpoint_params(checkpoint, "params_ema")
    nnx.update(model, params)
    model.eval()
    actual = nnx.state(model, nnx.Param)
    if jax.tree.structure(actual) != jax.tree.structure(params):
        raise ValueError("Loaded parameter structure differs from the EMA checkpoint")
    digest = hashlib.sha256()
    parameter_count = 0
    for index, (value, expected) in enumerate(
        zip(jax.tree.leaves(actual), jax.tree.leaves(params), strict=True)
    ):
        array, reference = np.asarray(value), np.asarray(expected)
        if not np.array_equal(array, reference) or not np.isfinite(array).all():
            raise ValueError(f"Loaded parameter leaf {index} is not finite, exact EMA")
        digest.update(f"{index}:{array.dtype}:{array.shape}\n".encode())
        digest.update(array.tobytes(order="C"))
        parameter_count += array.size
    return {
        "parameter_count": parameter_count,
        "parameter_content_sha256": digest.hexdigest(),
    }


def load_public_model(checkpoint_path, which="best"):
    """Return model, simulator class, two generators, and JSON-safe metadata.

    ``checkpoint_path`` is a local bundle root containing ``config.yaml`` and
    ``checkpoints/``. The caller stages and verifies the pinned Hub revision;
    this loader does not download anything. Simulation uses the saved training
    configuration, including both acquisition schemes and the mask prior.
    """
    if which not in CHECKPOINT_STEPS:
        raise ValueError("which must be 'best' or 'latest'")
    path = Path(checkpoint_path).expanduser().resolve()
    config_path = path / "config.yaml"
    saved_config = OmegaConf.load(config_path)
    cfg = load_cfg(path)
    sim_type, simulators = build_simulator(cfg)
    checkpoint, model, restored_sim_type = load_checkpoint(path, which=which)
    step = int(checkpoint["step"])
    if step != CHECKPOINT_STEPS[which]:
        raise ValueError(
            f"Expected public {which} step {CHECKPOINT_STEPS[which]}, restored {step}"
        )
    if sim_type is not restored_sim_type or sim_type is not model.cfg.simulator:
        raise ValueError("Saved simulation config differs from the checkpoint model")
    if len(simulators) != 2:
        raise ValueError("Expected the two saved public-checkpoint acquisition schemes")
    weights = apply_ema(checkpoint, model)
    metadata = {
        "checkpoint_path": str(path),
        "expected_source": {
            "repo_id": HF_REPO,
            "model_name": MODEL_NAME,
            "revision": HF_REVISION,
        },
        "which": which,
        "selected_step": step,
        "val_loss": (
            float(checkpoint["val_loss"])
            if checkpoint.get("val_loss") is not None
            else None
        ),
        "params_name": "params_ema",
        "loaded_parameters_exactly_equal_ema": True,
        "parameters_finite": True,
        "parameter_count": weights["parameter_count"],
        "parameter_content_sha256": weights["parameter_content_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "saved_config": OmegaConf.to_container(saved_config, resolve=False),
        "simulator_config": OmegaConf.to_container(cfg.simulator, resolve=False),
        "simulator_class": f"{sim_type.__module__}.{sim_type.__qualname__}",
        "theta_dim": sim_type.theta_dim,
        "mask_dim": sim_type.num_compartments(),
        "model_components": [component.__name__ for component in sim_type.model_types],
        "noise_components": [component.__name__ for component in sim_type.noise_types],
        "fraction_prior": np.asarray(sim_type.fraction_prior).tolist(),
        "physical_parameter_shapes": _physical_parameter_shapes(sim_type),
        "training_seed": OmegaConf.select(cfg, "run.seed"),
        "model_init_seed": OmegaConf.select(cfg, "model.init_seed"),
        "theta_sampler_defaults": {
            "sample_method": "ode",
            "num_steps": 64,
            "last_euler_step": True,
        },
    }
    return model, sim_type, simulators, metadata


def portable_config(config):
    """Map known historical import targets in memory, retaining all settings."""
    replacements = {}

    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str) and value in LEGACY_TARGETS:
            replacements[value] = LEGACY_TARGETS[value]
            return LEGACY_TARGETS[value]
        return value

    plain = OmegaConf.to_container(config, resolve=False)
    return OmegaConf.create(visit(plain)), replacements


def load_run_model(checkpoint_path, which="latest", expected_step=None):
    """Load a local run/snapshot, recording its actual weights and settings.

    No public-checkpoint identity or completed training budget is implied.
    Historical paper adapters preserve their diffusion and acquisition choices.
    """
    if which not in ("latest", "best"):
        raise ValueError("which must be latest or best")
    path = Path(checkpoint_path).expanduser().resolve()
    if not (path / "checkpoints").is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {path}")
    saved = load_cfg(path)
    cfg, targets = portable_config(saved)
    artifact = load_artifact_config(path, cfg=cfg)
    if artifact is not None:
        artifact, artifact_targets = portable_config(artifact)
        targets.update(artifact_targets)
        # Older artifacts name the simulator under a former package root
        # (the public msb3s_2_4_6_128 says ``dmri.simulators...``); compare
        # the class it resolves to rather than the spelling.
        model_class = OmegaConf.select(artifact, "simulator.model_class")
        if model_class is not None:
            current = resolve_class_path(model_class)
            if current != model_class:
                targets[model_class] = current
                artifact.simulator.model_class = current
        if artifact != build_artifact_config(cfg):
            raise ValueError("Artifact differs from the saved model/simulator config")
    sim_type = resolve_simulator_model(cfg)
    if sim_type not in (
        Ball3StickSharedDiffusivity,
        MultiShellBall3StickSharedDiffusivity,
    ):
        raise ValueError("The fixed-mask calibration protocol requires a B3S simulator")
    if sim_type.num_compartments() != 5 or sim_type.theta_dim not in (11, 12):
        raise ValueError("Unexpected B3S component or parameter layout")
    configured_type, simulators = build_simulator(cfg)
    if configured_type is not sim_type or len(simulators) != 2:
        raise ValueError("Expected two matching short/long acquisition generators")
    model = build_model(cfg, sim_type)
    with closing(
        CheckpointManager(str(path / "checkpoints"), continue_training=True)
    ) as manager:
        latest = manager.get_latest_step()
        best = manager.best_manager.latest_step()
        selected = best if which == "best" else latest
        if selected is None:
            raise FileNotFoundError(f"No {which} checkpoint found")
        if expected_step is not None and selected != expected_step:
            raise ValueError(
                f"Expected selected step {expected_step}, found {selected}"
            )
        state = manager.restore_parameters(
            step=selected,
            params=nnx.state(model, nnx.Param),
            from_best=which == "best",
            partial_restore=False,
        )
    if state is None or int(state["step"]) != selected:
        raise ValueError("Stored step differs from the selected checkpoint")
    weights = apply_ema(state, model)
    selected_dir = path / "checkpoints"
    if which == "best":
        selected_dir /= "best"
    selected_dir /= str(selected)
    decoder = model.inference_decoder
    metadata = {
        "checkpoint_path": str(path),
        "which": which,
        "selected_step": selected,
        "latest_saved_step": latest,
        "selected_best_checkpoint_step": best,
        "val_loss": float(state["val_loss"])
        if state.get("val_loss") is not None
        else None,
        "params_name": "params_ema",
        "loaded_parameters_exactly_equal_ema": True,
        "parameters_finite": True,
        **weights,
        "selected_checkpoint_sha256": canonical_hash(
            file_manifest(selected_dir, allow_file_symlinks=True)
        ),
        "config_representation": "Normalized saved training config; interpolation strings retained",
        "saved_config": OmegaConf.to_container(saved, resolve=False),
        "saved_config_sha256": canonical_hash(
            OmegaConf.to_container(saved, resolve=False)
        ),
        "legacy_import_targets": targets,
        "acquisition_adapter": protocol_metadata()
        if any(
            target in OmegaConf.to_yaml(cfg, resolve=False)
            for target in LEGACY_TARGETS.values()
            if ".acquisition." in target
        )
        else None,
        "simulator_class": f"{sim_type.__module__}.{sim_type.__qualname__}",
        "theta_dim": sim_type.theta_dim,
        "diffusion_defaults": {
            "t_min": float(decoder.schedule.t_min),
            "t_max": float(decoder.schedule.t_max),
        },
        "snapshot_metadata_sha256": {
            name: file_sha256(path / name)
            for name in ("snapshot.json", "training-complete.json", "artifact.yaml")
            if (path / name).is_file()
        },
    }
    return model, sim_type, simulators, metadata
