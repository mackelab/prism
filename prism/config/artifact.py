from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# Bound as a module, not by name: prism.simulators.config imports
# prism.config.aliases, which runs this package's __init__ first.
import prism.simulators.config as simulator_config
from prism.config.migrate import CONFIG_SCHEMA_VERSION, normalize_config

ARTIFACT_VERSION = 1


def build_artifact_config(cfg: DictConfig | dict[str, Any]) -> DictConfig:
    """Extract the stable model-construction subset of a training config."""
    canonical = normalize_config(cfg, "train")
    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "parameter_tree_version": 1,
        "model": OmegaConf.to_container(canonical.model, resolve=False),
        "simulator": simulator_config.simulator_model_config(canonical),
    }
    return OmegaConf.create(artifact)


def load_artifact_config(
    path: str | Path, *, cfg: DictConfig | dict[str, Any] | None = None
) -> DictConfig | None:
    """Load an artifact, filling omitted v1 construction fields from saved config.

    Early v1 artifacts stored only the simulator class. A saved training config
    can recover their input dimension and mask-prior overrides. Explicit artifact
    fields take precedence, and the artifact on disk is never changed.
    """
    artifact_path = Path(path) / "artifact.yaml"
    if not artifact_path.exists():
        return None
    artifact = OmegaConf.load(artifact_path)
    version = artifact.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {version!r}; expected {ARTIFACT_VERSION}"
        )
    if cfg is not None:
        needs_input_dim = "input_dim" not in artifact.simulator
        needs_mask_prior = (
            "mask_prior" not in artifact.simulator
            and not simulator_config.mask_prior_overrides(artifact.simulator)
        )
        if needs_input_dim or needs_mask_prior:
            saved = normalize_config(cfg, "train")
            recovered = {}
            if needs_input_dim and "input_dim" in saved.simulator:
                recovered["input_dim"] = saved.simulator.input_dim
            if needs_mask_prior:
                prior = simulator_config.mask_prior_overrides(saved.simulator)
                if prior or "mask_prior" in saved.simulator:
                    recovered["mask_prior"] = prior
            if recovered and (
                simulator_config.resolve_simulator_model(saved)
                is not simulator_config.resolve_simulator_model(artifact)
            ):
                raise ValueError(
                    "Cannot recover omitted artifact settings from a different simulator"
                )
            artifact.simulator.update(recovered)
    return artifact
