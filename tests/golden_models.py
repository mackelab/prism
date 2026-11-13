"""Golden model construction and outputs from saved configurations.

Saved runs name their classes in ``config.yaml``/``artifact.yaml``. These
references must keep resolving to the same parameter tree, initialization,
simulated data and network outputs when code moves. Cases cover published
v1 dMRI configs, artifacts with dotted class paths, the frozen Figure 4b
architectures and every training preset.

Regenerate only when a change is meant to alter models:

    uv run python tests/golden_models.py
"""

import copy
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

from prism.config import load_artifact_config, normalize_config
from prism.simulators.config import resolve_simulator_model
from prism.train.build_model import build_model
from prism.train.build_simulator import build_simulator
from prism.train.utils import load_cfg

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_models.json"
CONF_ROOT = Path(__file__).resolve().parents[1] / "conf"
NUM_SAMPLES = 2
NUM_STEPS = 4
LOSS_KWARGS = {
    "weight_by_complexity": False,
    "label_smoothing": 0.01,
    "use_loss_mask": True,
    "cut_off_tsm": 0.1,
    "permute_order": False,
}


def _param_leaves(model):
    state = nnx.state(model, nnx.Param)
    leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    return [(jax.tree_util.keystr(path), np.asarray(leaf)) for path, leaf in leaves]


def param_digest(model):
    """Return a hash of parameter paths and shapes plus per-leaf sums."""
    leaves = _param_leaves(model)
    layout = "\n".join(f"{path}:{leaf.shape}:{leaf.dtype}" for path, leaf in leaves)
    return {
        "layout_sha256": hashlib.sha256(layout.encode()).hexdigest(),
        "num_leaves": len(leaves),
        "num_params": int(sum(leaf.size for _, leaf in leaves)),
        "leaf_sums": [float(leaf.astype(np.float64).sum()) for _, leaf in leaves],
        "leaf_sq_sums": [
            float((leaf.astype(np.float64) ** 2).sum()) for _, leaf in leaves
        ],
    }


def build_saved_run(run_dir, files):
    """Write saved run files and rebuild them the way checkpoint loading does."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        OmegaConf.save(OmegaConf.create(content), run_dir / name)
    cfg = load_cfg(run_dir)
    artifact = load_artifact_config(run_dir, cfg=cfg)
    build_cfg = (
        OmegaConf.create({
            "model": OmegaConf.to_container(artifact.model, resolve=False),
            "simulator": OmegaConf.to_container(artifact.simulator, resolve=False),
        })
        if artifact is not None
        else cfg
    )
    sim_type = resolve_simulator_model(build_cfg)
    model = build_model(build_cfg, sim_type)
    model.eval()
    return cfg, sim_type, model


def _leaf_sums(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return [float(np.asarray(leaf, dtype=np.float64).sum()) for leaf in leaves]


def model_outputs(cfg, model):
    """Simulate a small batch and evaluate every public inference method."""
    _, generators = build_simulator(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), NUM_SAMPLES)
    batch = jax.vmap(generators[0])(keys)
    loss = model.loss_fn(jax.random.PRNGKey(1), **batch, **LOSS_KWARGS)

    outputs = {
        "batch_sums": _leaf_sums(batch),
        "loss": np.asarray(loss).tolist(),
        "log_prob_mask": [],
        "sample_mask": [],
        "sample_theta": [],
        "log_prob_theta": [],
    }
    sample_keys = jax.random.split(jax.random.PRNGKey(2), NUM_SAMPLES)
    for index in range(NUM_SAMPLES):
        item = jax.tree_util.tree_map(lambda value, i=index: value[i], batch)
        inputs = {"x": item["x"]}
        if "acq" in item:
            inputs["acq"] = item["acq"]
        mask_key, theta_key = jax.random.split(sample_keys[index])
        outputs["log_prob_mask"].append(
            float(
                model.log_prob_mask(
                    item["model_mask"], mask_prior=item["mask_prior"], **inputs
                )
            )
        )
        outputs["sample_mask"].append(
            np.asarray(
                model.sample_mask(mask_key, mask_prior=item["mask_prior"], **inputs),
                dtype=np.float64,
            ).tolist()
        )
        outputs["sample_theta"].append(
            np.asarray(
                model.sample_theta(
                    theta_key,
                    model_mask=item["model_mask"],
                    num_steps=NUM_STEPS,
                    **inputs,
                )
            ).tolist()
        )
        outputs["log_prob_theta"].append(
            float(
                jnp.sum(
                    model.log_prob_theta(
                        item["theta"],
                        model_mask=item["model_mask"],
                        num_steps=NUM_STEPS,
                        **inputs,
                    )
                )
            )
        )
    return outputs


def compose_preset(name):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(CONF_ROOT), version_base=None):
            cfg = compose("train", overrides=[f"+experiment/train={name}"])
    finally:
        GlobalHydra.instance().clear()
    return normalize_config(cfg, "train")


def preset_names():
    return sorted(
        path.stem for path in (CONF_ROOT / "experiment" / "train").glob("*.yaml")
    )


def preset_digest(name):
    """Architecture of a preset: simulator and parameter paths, shapes, dtypes.

    Shapes come from ``nnx.eval_shape``, so full-size presets are never
    initialized; the saved-run cases pin initial values and outputs.
    """
    cfg = compose_preset(name)
    sim_type = resolve_simulator_model(cfg)
    model = nnx.eval_shape(lambda: build_model(cfg, sim_type))
    leaves, _ = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    layout = "\n".join(
        f"{jax.tree_util.keystr(path)}:{tuple(leaf.shape)}:{leaf.dtype}"
        for path, leaf in leaves
    )
    return {
        "simulator": sim_type.__name__,
        "layout_sha256": hashlib.sha256(layout.encode()).hexdigest(),
        "num_leaves": len(leaves),
        "num_params": int(sum(np.prod(leaf.shape, dtype=int) for _, leaf in leaves)),
    }


# ---------------------------------------------------------------------------
# Fixture generation


def _reduce_model(model_cfg):
    """Shrink a saved model config so the golden forward passes stay fast."""
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg["model_dim"] = 16
    for section in ("embedding_net", "inference_net", "model_selection_net"):
        params = model_cfg[section].setdefault("params", {})
        params["num_layers"] = 1
        for key in ("use_flash_attention", "use_flash_cross_attention"):
            if key in params:
                params[key] = False
        if "attn_size" in params:
            params["attn_size"] = 8
        if "num_heads" in params:
            params["num_heads"] = 2
        if "time_embed_dim" in params:
            params["time_embed_dim"] = 16
        if "prior_params_embed_dim" in params:
            params["prior_params_embed_dim"] = 16
    return model_cfg


def _hf_config(name):
    root = Path.home() / ".cache/huggingface/hub/models--manugloeck--dmri-pretrained"
    (path,) = root.glob(f"snapshots/*/{name}/config.yaml")
    saved = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    return {
        "schema_version": saved.get("schema_version", 2),
        "simulator": saved["simulator"],
        "model": _reduce_model(saved["model"]),
    }


def _preset_artifact(name):
    from prism.config import build_artifact_config

    cfg = compose_preset(name)
    artifact = OmegaConf.to_container(build_artifact_config(cfg), resolve=True)
    artifact["model"] = _reduce_model(artifact["model"])
    config = {
        "schema_version": 2,
        "simulator": OmegaConf.to_container(cfg.simulator, resolve=True),
        "model": artifact["model"],
    }
    return {"config.yaml": config, "artifact.yaml": artifact}


def saved_run_cases():
    """Saved-run files exactly as older code wrote them (class names included)."""
    smoke_artifact = {
        "artifact_version": 1,
        "config_schema_version": 2,
        "parameter_tree_version": 1,
        "model": {
            "embedding_net": {
                "name": "prism.nn.embedding_net.TransformerEmbeddingConfig",
                "cls": "prism.nn.embedding_net.SBMIEmbeddingTransformer",
                "params": {
                    "input_dim": 2,
                    "num_layers": 1,
                    "num_heads": 2,
                    "widening_factor": 2,
                    "attn_size": 8,
                    "positional_encoding": False,
                    "dropout_rate": 0.0,
                    "out_norm": True,
                    "use_flash_attention": False,
                },
            },
            "inference_net": {
                "name": "DMRIThetaInferenceConfig",
                "params": {
                    "num_layers": 1,
                    "num_heads": 2,
                    "widening_factor": 2,
                    "attn_size": 8,
                    "time_embed_dim": 16,
                    "use_flash_cross_attention": False,
                },
            },
            "model_selection_net": {
                "name": "DMRIModelSelectionAmortizedPriorConfig",
                "params": {
                    "num_layers": 1,
                    "num_heads": 2,
                    "widening_factor": 2,
                    "attn_size": 8,
                    "prior_params_embed_dim": 16,
                    "mask_prior_dim": 1,
                    "use_flash_cross_attention": False,
                },
            },
            "init_seed": 0,
            "name": "DMRIInferenceModelConfigMaskPriorAmortized",
            "tokenizer_cls": "prism.nn.symbolic_tokenizer.SymbolicTokenizer",
            "model_dim": 16,
            "use_attention_mask": True,
            "inference_loss_type": "v",
        },
        "simulator": {
            "model_class": "prism.simulators.symbolic.SimplePolynomial",
            "mask_prior": {"alpha": 1.0, "beta": 1.0},
        },
    }
    smoke_config = {
        "schema_version": 2,
        "simulator": {
            "model_class": "prism.simulators.symbolic.SimplePolynomial",
            "num_points": 32,
            "x_min": -1.0,
            "x_max": 1.0,
            "mask_prior": {"alpha": 1.0, "beta": 1.0},
        },
        "model": smoke_artifact["model"],
    }
    return {
        "dmri_v1_ppp": {"config.yaml": _hf_config("msb3s_2_4_6_128")},
        "dmri_v1_pp": {"config.yaml": _hf_config("b3s_2_4_6_64")},
        "symbolic_artifact": {
            "config.yaml": smoke_config,
            "artifact.yaml": smoke_artifact,
        },
        "symbolic_paper_artifact": _preset_artifact("symbolic_sbmi"),
        "symbolic_prior_artifact": _preset_artifact("symbolic15"),
    }


def generate(tmp_root):
    golden = {"saved_runs": {}, "presets": {}}
    for name, files in saved_run_cases().items():
        cfg, _, model = build_saved_run(Path(tmp_root) / name, files)
        golden["saved_runs"][name] = {
            "files": files,
            "params": param_digest(model),
            "outputs": model_outputs(cfg, model),
        }
        print(f"saved run {name}: {golden['saved_runs'][name]['params']['num_params']}")
    for name in preset_names():
        golden["presets"][name] = preset_digest(name)
        print(f"preset {name}: {golden['presets'][name]['num_params']}")
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    import tempfile

    jax.config.update("jax_platform_name", "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        generate(tmp)
