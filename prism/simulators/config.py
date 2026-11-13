from typing import Any

from hydra.utils import get_class
from omegaconf import DictConfig, ListConfig, OmegaConf

from prism.config.aliases import LEGACY_SIMULATOR_MODULE, resolve_class_path
from prism.simulators.base import Simulator, missing_simulator_members


def _simulator_cfg(cfg):
    return cfg.simulator if hasattr(cfg, "simulator") else cfg


def resolve_simulator_model(cfg) -> type[Simulator]:
    """Resolve the simulator class from new or frozen config."""
    simulator_cfg = _simulator_cfg(cfg)
    model_class = simulator_cfg.get("model_class")
    legacy_name = OmegaConf.select(simulator_cfg, "sim_type.name")
    legacy_path = None
    if legacy_name is not None:
        legacy_path = (
            legacy_name
            if "." in legacy_name
            else f"{LEGACY_SIMULATOR_MODULE}.{legacy_name}"
        )
    if model_class is None:
        model_class = legacy_path
    if model_class is None:
        raise ValueError(
            "simulator.model_class must name an importable simulator class; "
            "choose a preset (+experiment/train=NAME) or set simulator=NAME"
        )

    resolved = get_class(resolve_class_path(model_class))
    if (
        legacy_path is not None
        and get_class(resolve_class_path(legacy_path)) is not resolved
    ):
        raise ValueError(
            "Conflicting simulator settings 'model_class' and 'sim_type.name'"
        )
    missing = missing_simulator_members(resolved)
    if not isinstance(resolved, type) or missing:
        raise TypeError(
            f"simulator.model_class {model_class!r} does not implement the "
            f"simulator interface; missing {missing}"
        )
    return resolved


def simulator_model_config(cfg) -> dict[str, Any]:
    """Return the simulator settings used to construct an inference model."""
    model_class = resolve_simulator_model(cfg)
    simulator_cfg = _simulator_cfg(cfg)
    spec: dict[str, Any] = {
        "model_class": f"{model_class.__module__}.{model_class.__qualname__}"
    }
    if "input_dim" in simulator_cfg:
        spec["input_dim"] = simulator_cfg.input_dim
    prior = mask_prior_overrides(simulator_cfg)
    if prior or "mask_prior" in simulator_cfg:
        spec["mask_prior"] = prior
    return spec


def _plain(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def simulator_option(simulator_cfg, new_name, legacy_name, default=None):
    new_value = simulator_cfg.get(new_name)
    legacy_value = simulator_cfg.get(legacy_name)
    if new_value is not None and legacy_value is not None and new_value != legacy_value:
        raise ValueError(
            f"Conflicting simulator settings {new_name!r} and {legacy_name!r}"
        )
    if new_value is not None:
        return new_value
    if legacy_value is not None:
        return legacy_value
    return default


def mask_prior_overrides(simulator_cfg) -> dict[str, Any]:
    """Read generic overrides while retaining frozen prior field aliases."""
    overrides = dict(_plain(simulator_cfg.get("mask_prior")) or {})
    aliases = {
        "prior_mask_alpha": "alpha",
        "prior_mask_beta": "beta",
        "prior_mask_p0": "p0",
        "prior_mask_u_alpha": "u_alpha",
        "prior_mask_u_beta": "u_beta",
    }
    for old_name, new_name in aliases.items():
        legacy_value = simulator_cfg.get(old_name)
        if legacy_value is None:
            continue
        if new_name in overrides and overrides[new_name] != legacy_value:
            raise ValueError(
                f"Conflicting simulator mask-prior settings {old_name!r} and "
                f"mask_prior.{new_name}"
            )
        overrides.setdefault(new_name, legacy_value)
    return overrides
