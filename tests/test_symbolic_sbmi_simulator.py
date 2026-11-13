"""Numerical regression against the separately executed frozen Figure 4b source."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from prism.symbolic.paper import mask_prior as prior_module
from prism.symbolic.paper.mask_prior import SymbolicMaskPrior
from prism.symbolic.paper.model_specs import SymbolicModelV0
from prism.symbolic.simulators import SimplePolynomial
from prism.train.build_simulator import build_simulator

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURES / "symbolic_sbmi_source.npz", allow_pickle=False) as data:
        return dict(data)


@pytest.fixture(scope="module")
def paper_generator():
    cfg = OmegaConf.create({
        "simulator": OmegaConf.load(REPO / "conf/simulator/symbolic_sbmi.yaml")
    })
    model, generators = build_simulator(cfg)
    assert model is SymbolicModelV0
    return generators[0]


def test_paper_components_and_bounds_match_frozen_source(reference):
    metadata = json.loads((FIXTURES / "symbolic_sbmi_source.json").read_text())
    components = SymbolicModelV0.model_types + SymbolicModelV0.noise_types
    assert [component.__name__ for component in components] == metadata["components"]
    assert SymbolicModelV0.theta_dim == 28
    assert len(SymbolicModelV0.model_types) == 15
    assert len(SymbolicModelV0.noise_types) == 5
    for bound in ("param_low", "param_high"):
        actual = np.concatenate([getattr(component, bound) for component in components])
        np.testing.assert_array_equal(actual, reference[bound])
    assert SimplePolynomial.theta_dim == 3
    assert SymbolicModelV0.model_types[2] is not SimplePolynomial.model_types[1]


def test_paper_mask_assets_are_unchanged():
    metadata = json.loads((FIXTURES / "symbolic_sbmi_source.json").read_text())
    directory = Path(prior_module.__file__).parent / "data"
    for name, expected in metadata["assets_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    with pytest.raises(ValueError, match="15 model and 5 noise"):
        SymbolicMaskPrior(2, 1)


def test_paper_rng_order_and_observation_blocks_match_frozen_source(
    reference, paper_generator
):
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(reference["seeds"]))
    actual = jax.vmap(paper_generator)(keys)
    assert actual["mask_prior"] is None
    assert actual["x"].shape == (4, 50, 10)
    for name in ("theta", "model_mask"):
        np.testing.assert_array_equal(actual[name], reference[name])
    np.testing.assert_allclose(actual["x"], reference["x"], atol=2e-5, rtol=2e-6)


def test_paper_components_noise_and_likelihood_match_frozen_source(reference):
    theta = jnp.asarray(reference["theta"][0])
    masks = jnp.asarray(reference["component_masks"])
    grid = jnp.arange(0.0, 10.0, 0.02)
    observed = jnp.asarray(reference["x"][0]).reshape(-1)
    signals = jax.vmap(
        lambda mask: SymbolicModelV0.from_theta(theta, model_mask=mask).evaluate(
            grid, jax.random.PRNGKey(23)
        )
    )(masks)
    likelihoods = jax.vmap(
        lambda mask: SymbolicModelV0.from_theta(theta, model_mask=mask).log_likelihood(
            grid, observed
        )
    )(masks)
    np.testing.assert_allclose(
        signals, reference["component_signals"], atol=2e-5, rtol=2e-6
    )
    np.testing.assert_allclose(
        likelihoods, reference["log_likelihood"], atol=2e-4, rtol=2e-6
    )
    np.testing.assert_array_equal(
        jax.vmap(SymbolicModelV0.theta_mask)(masks), reference["theta_mask"]
    )


def test_paper_generator_accepts_a_fixed_mask(reference, paper_generator):
    mask = jnp.asarray(reference["model_mask"][0])
    actual = paper_generator(jax.random.PRNGKey(42), model_mask=mask)
    np.testing.assert_array_equal(actual["model_mask"], mask)
    np.testing.assert_array_equal(actual["theta"], reference["theta"][-1])


@pytest.mark.parametrize(
    "override, message",
    [
        ({"input_dim": 0}, "input_dim"),
        ({"x_step": 0}, "x_step"),
        ({"posterior_score": True}, "posterior scores"),
        ({"observation_format": "unknown"}, "observation_format"),
    ],
)
def test_invalid_paper_observation_config_is_rejected(override, message):
    # A small existing model exercises configuration checks without loading tables.
    cfg = OmegaConf.create({
        "model_class": "prism.symbolic.simulators.SimplePolynomial",
        "observation_format": "signal_blocks",
        "x_min": 0.0,
        "x_max": 10.0,
        "x_step": 0.02,
        "input_dim": 10,
        **override,
    })
    with pytest.raises(ValueError, match=message):
        build_simulator(cfg)
