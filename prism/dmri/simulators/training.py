"""Training data for dMRI simulators: acquisitions from config and batches."""

from functools import partial
from typing import Any

import jax
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from prism.config.aliases import resolve_class_path
from prism.dmri.simulators.acquisition_scheme import (
    random_advanced_reasearch_acquisition_scheme,
    random_clinical_acquisition,
    random_hardi_acquisition,
    random_hcp_acquisition,
    random_hcp_large_acquisition,
    random_ssfp_acquisition,
)

# This map is only a reader for frozen named configs. New acquisition factories
# are imported directly from their Hydra targets and require no registration.
_LEGACY_ACQUISITION_FACTORIES = {
    "clinical": random_clinical_acquisition,
    "hardi": random_hardi_acquisition,
    "advanced_research": random_advanced_reasearch_acquisition_scheme,
    "hcp": random_hcp_acquisition,
    "hcp_large": random_hcp_large_acquisition,
    "ssfp": random_ssfp_acquisition,
}


def _plain(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _legacy_acquisition_factories(simulator_cfg):
    acquisition_cfg = simulator_cfg.get("acquisition_scheme")
    if acquisition_cfg is None:
        return None
    schemes = OmegaConf.select(acquisition_cfg, "params.schemes")
    if schemes is None:
        schemes = acquisition_cfg.get("schemes")
    if schemes is None:
        schemes = [
            {
                "name": acquisition_cfg.get("name"),
                "params": acquisition_cfg.get("params", {}),
            }
        ]

    factories = []
    for scheme in schemes:
        name = scheme.get("name")
        try:
            factory = _LEGACY_ACQUISITION_FACTORIES[name]
        except KeyError as error:
            raise ValueError(f"Unknown legacy acquisition scheme {name!r}") from error
        factories.append(partial(factory, **(_plain(scheme.get("params")) or {})))
    return factories


def _simulator_cfg(cfg):
    return cfg.simulator if hasattr(cfg, "simulator") else cfg


def _current_target(acquisition):
    """Point ``_target_`` paths written by older configs at current code."""
    acquisition = _plain(acquisition)
    if isinstance(acquisition, dict) and "_target_" in acquisition:
        acquisition = {
            **acquisition,
            "_target_": resolve_class_path(acquisition["_target_"]),
        }
    return acquisition


def resolve_acquisition_factories(cfg) -> list[Any]:
    """Resolve direct acquisition targets or frozen named acquisition configs."""
    simulator_cfg = _simulator_cfg(cfg)
    acquisitions = simulator_cfg.get("acquisitions")
    legacy_cfg = simulator_cfg.get("acquisition_scheme")
    if acquisitions is not None and legacy_cfg is not None:
        raise ValueError(
            "Conflicting simulator settings 'acquisitions' and 'acquisition_scheme'"
        )
    if acquisitions is None:
        legacy = _legacy_acquisition_factories(simulator_cfg)
        if legacy is not None:
            return legacy
        raise ValueError("simulator.acquisitions must contain at least one factory")

    factories = [
        instantiate(_current_target(acquisition)) for acquisition in acquisitions
    ]
    if not factories or not all(callable(factory) for factory in factories):
        raise TypeError(
            "Every simulator acquisition must instantiate to a callable; "
            "set _partial_: true on acquisition functions"
        )
    return factories


def make_simulation_generator(
    model_class,
    acquisition_factory,
    mask_prior,
    posterior_score,
):
    def simulator(rng, mask_prior_hyperparameter=None, model_mask=None):
        rng0, rng1, rng2, rng3 = jax.random.split(rng, 4)
        acquisition = acquisition_factory(rng0)
        theta = jax.random.normal(rng1, shape=(model_class.theta_dim,))
        rng_prior, rng_mask = jax.random.split(rng2)
        if mask_prior_hyperparameter is None:
            p_mask = mask_prior.sample_hyperparameters(rng_prior)
        else:
            p_mask = mask_prior_hyperparameter
        if model_mask is None:
            model_mask = mask_prior.sample_model_mask(rng_mask, p_mask)

        output = {
            "mask_prior": p_mask,
            "model_mask": model_mask,
            "theta": theta,
            "acq": acquisition,
        }
        if not posterior_score:
            model = model_class.from_theta(theta, model_mask=model_mask)
            output["x"] = model.signal(acquisition, rng=rng3)
            return output

        def posterior_potential(value):
            model = model_class.from_theta(value, model_mask=model_mask)
            observed = jax.lax.stop_gradient(model.signal(acquisition, rng=rng3))
            potential = model.log_likelihood(acquisition, observed).sum()
            potential += jax.scipy.stats.norm.logpdf(value, 0, 1).sum(-1)
            return potential, observed

        score, observed = jax.grad(posterior_potential, has_aux=True)(theta)
        output["x"] = observed
        output["target_score"] = score
        return output

    return simulator


def training_generators(model_class, simulator_cfg, mask_prior, posterior_score):
    """Return one sample generator per configured acquisition distribution."""
    return [
        make_simulation_generator(
            model_class, acquisition_factory, mask_prior, posterior_score
        )
        for acquisition_factory in resolve_acquisition_factories(simulator_cfg)
    ]
