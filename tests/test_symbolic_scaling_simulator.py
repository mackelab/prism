"""K15/K30 regression against independently executed, pinned native source."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_module

from prism.simulators.mask_prior import BetaBernoulliMaskPrior
from prism.symbolic.paper.mask_prior import SymbolicMaskPrior
from prism.symbolic.paper.model_specs import (
    SymbolicModelV0,
    SymbolicModelV15,
    SymbolicModelV30,
    SymbolicModelV80,
    SymbolicModelV100,
)
from prism.train.build_simulator import build_simulator

FIXTURES = Path(__file__).parent / "fixtures"
# K=15/30 live in the original fixture, K=80/100 in a second recorded fixture.
FIXTURE_NAMES = ("symbolic_scaling_source", "symbolic_scaling_large_source")


@pytest.fixture(scope="module")
def reference():
    merged = {}
    for name in FIXTURE_NAMES:
        with np.load(FIXTURES / f"{name}.npz", allow_pickle=False) as data:
            merged.update(dict(data))
    return merged


@pytest.fixture(scope="module")
def metadata():
    records = [
        json.loads((FIXTURES / f"{name}.json").read_text()) for name in FIXTURE_NAMES
    ]
    merged = dict(records[0])
    merged["variants"] = {}
    for record in records:
        merged["variants"].update(record["variants"])
    return merged


@pytest.fixture(
    scope="module",
    params=[
        (15, SymbolicModelV15, 28, 5),
        (30, SymbolicModelV30, 73, 8),
        (80, SymbolicModelV80, 187, 8),
        (100, SymbolicModelV100, 223, 8),
    ],
    ids=["k15", "k30", "k80", "k100"],
)
def variant(request, reference, metadata):
    k, model_class, theta_dim, num_noises = request.param
    prefix = f"k{k}_"
    arrays = {
        name.removeprefix(prefix): value
        for name, value in reference.items()
        if name.startswith(prefix)
    }
    return k, model_class, theta_dim, num_noises, arrays, metadata["variants"][str(k)]


@pytest.fixture(scope="module")
def generator(variant):
    k, model_class, *_ = variant
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(config_name=None, overrides=[f"+simulator=symbolic_{k}"])
    actual_model, generators = build_simulator(cfg)
    assert actual_model is model_class
    assert len(generators) == 1
    return generators[0]


def test_scaling_fixture_provenance(metadata):
    assert metadata["variants"]["15"]["source_class"] == "SymbolicModelV0"
    for k in (30, 80, 100):
        assert metadata["variants"][str(k)]["source_class"] == f"SymbolicModelV{k}"
    for name in FIXTURE_NAMES:
        record = json.loads((FIXTURES / f"{name}.json").read_text())
        digest = hashlib.sha256((FIXTURES / f"{name}.npz").read_bytes()).hexdigest()
        assert digest == record["fixture_sha256"]


def test_scaling_components_order_bounds_and_dimensions_match_source(variant):
    k, model_class, theta_dim, num_noises, arrays, source = variant
    components = model_class.model_types + model_class.noise_types
    assert [component.__name__ for component in components] == source["components"]
    assert [component.expression_str for component in components] == source[
        "expressions"
    ]
    assert len(model_class.model_types) == source["num_models"] == k
    assert len(model_class.noise_types) == source["num_noises"] == num_noises
    assert model_class.num_compartments() == k + num_noises
    assert model_class.theta_dim == source["theta_dim"] == theta_dim
    np.testing.assert_array_equal(model_class.split_idx(), arrays["split_dims"])
    for bound in ("param_low", "param_high"):
        actual = np.concatenate([getattr(component, bound) for component in components])
        np.testing.assert_array_equal(actual, arrays[bound])
    prior = model_class.create_mask_prior()
    assert isinstance(prior, BetaBernoulliMaskPrior)
    assert prior.alpha == prior.beta == prior.min_active_models == 1


def test_scaling_k15_keeps_figure4b_components_and_separate_prior():
    assert SymbolicModelV15.model_types == SymbolicModelV0.model_types
    assert SymbolicModelV15.noise_types == SymbolicModelV0.noise_types
    assert SymbolicModelV0.mask_prior_cls is SymbolicMaskPrior
    assert SymbolicModelV15.mask_prior_cls is BetaBernoulliMaskPrior
    assert SymbolicModelV0.theta_dim == SymbolicModelV15.theta_dim == 28


def test_scaling_bounded_prior_transforms_and_theta_masks_match_source(variant):
    _, model_class, _, _, arrays, _ = variant
    model = model_class.from_theta(jnp.asarray(arrays["transform_theta"]))
    components = model.model_compartments + model.noise_compartments
    bounded_params = jnp.concatenate([
        component.params_array for component in components
    ])
    np.testing.assert_array_equal(bounded_params, arrays["bounded_params"])
    assert np.all(bounded_params >= arrays["param_low"])
    assert np.all(bounded_params <= arrays["param_high"])
    np.testing.assert_allclose(
        model.theta, arrays["roundtrip_theta"], atol=2e-6, rtol=2e-6
    )
    np.testing.assert_array_equal(
        jax.vmap(model_class.theta_mask)(jnp.asarray(arrays["component_masks"])),
        arrays["theta_mask"],
    )


def test_scaling_each_expression_noise_and_likelihood_match_source(variant, metadata):
    _, model_class, _, _, arrays, _ = variant
    theta = jnp.asarray(arrays["transform_theta"])
    masks = jnp.asarray(arrays["component_masks"])
    grid = jnp.asarray(arrays["grid"])
    observed = jnp.asarray(arrays["observed"])
    key = jax.random.PRNGKey(metadata["component_noise_seed"])
    signals = jax.vmap(
        lambda mask: model_class.from_theta(theta, model_mask=mask).evaluate(grid, key)
    )(masks)
    likelihoods = jax.vmap(
        lambda mask: model_class.from_theta(theta, model_mask=mask).log_likelihood(
            grid, observed
        )
    )(masks)
    np.testing.assert_allclose(
        signals, arrays["component_signals"], atol=2e-5, rtol=2e-6
    )
    np.testing.assert_allclose(
        likelihoods, arrays["log_likelihood"], atol=2e-4, rtol=2e-6
    )


def test_scaling_generator_rng_and_observation_blocks_match_source(
    variant, reference, generator
):
    k, _, theta_dim, num_noises, arrays, _ = variant
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(reference["seeds"]))
    actual = jax.vmap(generator)(keys)
    assert actual["x"].shape == (4, 50, 10)
    assert actual["theta"].shape == (4, theta_dim)
    assert actual["model_mask"].shape == (4, k + num_noises)
    assert actual["mask_prior"].shape == (4, 1)
    assert np.isfinite(actual["x"]).all()
    assert np.all(actual["model_mask"][:, :k].sum(-1) >= 1)
    np.testing.assert_array_equal(actual["model_mask"][:, k:].sum(-1), 1)
    for name in ("theta", "model_mask", "mask_prior"):
        np.testing.assert_array_equal(actual[name], arrays[name])
    np.testing.assert_allclose(actual["x"], arrays["x"], atol=2e-5, rtol=2e-6)


def test_scaling_fixed_prior_endpoints_match_source(
    variant, reference, metadata, generator
):
    _, _, _, _, arrays, _ = variant
    key = jax.random.PRNGKey(metadata["fixed_hyper_seed"])
    actual = jax.vmap(lambda hyper: generator(key, mask_prior_hyperparameter=hyper))(
        jnp.asarray(reference["fixed_hyper"])
    )
    np.testing.assert_array_equal(
        reference["fixed_hyper"], np.array([[0.1], [0.9]], dtype=np.float32)
    )
    for name in ("theta", "model_mask", "mask_prior"):
        np.testing.assert_array_equal(actual[name], arrays[f"fixed_{name}"])
    np.testing.assert_allclose(actual["x"], arrays["fixed_x"], atol=2e-5, rtol=2e-6)


def test_scaling_fixed_mask_keeps_source_parameter_and_noise_draws(
    variant, reference, generator
):
    _, _, _, _, arrays, _ = variant
    actual = generator(
        jax.random.PRNGKey(int(reference["seeds"][-1])),
        model_mask=jnp.asarray(arrays["model_mask"][-1]),
    )
    for name in ("theta", "model_mask", "mask_prior"):
        np.testing.assert_array_equal(actual[name], arrays[name][-1])
    np.testing.assert_allclose(actual["x"], arrays["x"][-1], atol=2e-5, rtol=2e-6)
