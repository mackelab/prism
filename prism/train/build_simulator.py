from __future__ import annotations

from omegaconf import DictConfig

from prism.simulators.config import (
    mask_prior_overrides,
    resolve_simulator_model,
    simulator_option,
)


def _simulator_cfg(cfg):
    return cfg.simulator if hasattr(cfg, "simulator") else cfg


def build_simulator(cfg: DictConfig):
    """Build the simulator class and its training-data generators."""
    simulator_cfg = _simulator_cfg(cfg)
    model_class = resolve_simulator_model(simulator_cfg)
    mask_prior = model_class.create_mask_prior(**mask_prior_overrides(simulator_cfg))
    posterior_score = bool(
        simulator_option(
            simulator_cfg,
            "posterior_score",
            "with_posterior_score",
            False,
        )
    )
    generators = model_class.training_generators(
        simulator_cfg, mask_prior=mask_prior, posterior_score=posterior_score
    )
    return model_class, generators
