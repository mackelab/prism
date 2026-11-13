"""Regression against K=50 outputs executed from the frozen upstream source."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from prism.simulators.mask_prior import BetaBernoulliMaskPrior
from prism.symbolic.paper.model_specs import (
    SymbolicModelV0,
    SymbolicModelV50,
)
from prism.train.build_simulator import build_simulator

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURES / "symbolic_k50_source.npz", allow_pickle=False) as data:
        return dict(data)


@pytest.fixture(scope="module")
def metadata():
    return json.loads((FIXTURES / "symbolic_k50_source.json").read_text())


@pytest.fixture(scope="module")
def k50_generator():
    cfg = OmegaConf.create({
        "simulator": OmegaConf.load(REPO / "conf/simulator/symbolic_50.yaml")
    })
    model, generators = build_simulator(cfg)
    assert model is SymbolicModelV50
    return generators[0]


def test_k50_components_order_bounds_and_dimensions_match_source(reference, metadata):
    components = SymbolicModelV50.model_types + SymbolicModelV50.noise_types
    assert [component.__name__ for component in components] == metadata["components"]
    assert [component.expression_str for component in components] == metadata[
        "expressions"
    ]
    assert len(SymbolicModelV50.model_types) == 50
    assert len(SymbolicModelV50.noise_types) == 8
    assert SymbolicModelV50.num_compartments() == 58
    assert SymbolicModelV50.theta_dim == metadata["theta_dim"] == 115
    np.testing.assert_array_equal(SymbolicModelV50.split_idx(), reference["split_dims"])
    for bound in ("param_low", "param_high"):
        actual = np.concatenate([getattr(component, bound) for component in components])
        np.testing.assert_array_equal(actual, reference[bound])
    assert isinstance(SymbolicModelV50.create_mask_prior(), BetaBernoulliMaskPrior)
    assert SymbolicModelV0.theta_dim == 28


def test_k50_bounded_prior_transforms_match_source(reference):
    model = SymbolicModelV50.from_theta(jnp.asarray(reference["transform_theta"]))
    components = model.model_compartments + model.noise_compartments
    actual_params = jnp.concatenate([
        component.params_array for component in components
    ])
    np.testing.assert_array_equal(actual_params, reference["bounded_params"])
    np.testing.assert_allclose(
        model.theta, reference["roundtrip_theta"], atol=2e-6, rtol=2e-6
    )
    np.testing.assert_array_equal(
        jax.vmap(SymbolicModelV50.theta_mask)(
            jnp.asarray(reference["component_masks"])
        ),
        reference["theta_mask"],
    )


def test_k50_each_expression_noise_and_likelihood_match_source(reference, metadata):
    theta = jnp.asarray(reference["transform_theta"])
    masks = jnp.asarray(reference["component_masks"])
    grid = jnp.asarray(reference["grid"])
    observed = jnp.asarray(reference["observed"])
    key = jax.random.PRNGKey(metadata["component_noise_seed"])
    signals = jax.vmap(
        lambda mask: SymbolicModelV50.from_theta(theta, model_mask=mask).evaluate(
            grid, key
        )
    )(masks)
    likelihoods = jax.vmap(
        lambda mask: SymbolicModelV50.from_theta(theta, model_mask=mask).log_likelihood(
            grid, observed
        )
    )(masks)
    np.testing.assert_allclose(
        signals, reference["component_signals"], atol=2e-5, rtol=2e-6
    )
    np.testing.assert_allclose(
        likelihoods, reference["log_likelihood"], atol=2e-4, rtol=2e-6
    )


def test_k50_generator_rng_and_observation_blocks_match_source(
    reference, k50_generator
):
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(reference["seeds"]))
    actual = jax.vmap(k50_generator)(keys)
    assert actual["x"].shape == (4, 50, 10)
    assert actual["theta"].shape == (4, 115)
    assert actual["model_mask"].shape == (4, 58)
    assert actual["mask_prior"].shape == (4, 1)
    for name in ("theta", "model_mask", "mask_prior"):
        np.testing.assert_array_equal(actual[name], reference[name])
    np.testing.assert_allclose(actual["x"], reference["x"], atol=2e-5, rtol=2e-6)


def test_k50_fixed_prior_conditioning_matches_source(
    reference, metadata, k50_generator
):
    key = jax.random.PRNGKey(metadata["fixed_hyper_seed"])
    actual = jax.vmap(
        lambda hyper: k50_generator(key, mask_prior_hyperparameter=hyper)
    )(jnp.asarray(reference["fixed_hyper"]))
    for name in ("theta", "model_mask", "mask_prior"):
        np.testing.assert_array_equal(actual[name], reference[f"fixed_{name}"])
    np.testing.assert_allclose(actual["x"], reference["fixed_x"], atol=2e-5, rtol=2e-6)


def test_k50_fixed_mask_keeps_source_parameter_and_noise_draws(
    reference, k50_generator
):
    mask = jnp.asarray(reference["model_mask"][-1])
    actual = k50_generator(
        jax.random.PRNGKey(int(reference["seeds"][-1])), model_mask=mask
    )
    for name in ("theta", "model_mask", "mask_prior"):
        np.testing.assert_array_equal(actual[name], reference[name][-1])
    np.testing.assert_allclose(actual["x"], reference["x"][-1], atol=2e-5, rtol=2e-6)
