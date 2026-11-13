"""Separated calibration heads must preserve all combined-evaluator statistics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.dmri._shared.metrics import (
    make_model_sbc_fn,
    make_sbc_fn,
    make_theta_sbc_fn,
)


class FakeModel:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.tokenizer = SimpleNamespace(
            simulator=SimpleNamespace(theta_mask=lambda mask: mask)
        )
        self.sampler_settings = []

    def sample_mask(self, key, *, acq, x, mask_prior):
        return jax.random.bernoulli(key, mask_prior[0], (self.dimension,))

    def log_prob_mask(self, mask, *, acq, x, mask_prior):
        p = mask_prior[0]
        return jnp.where(mask, jnp.log(p), jnp.log1p(-p))

    def sample_theta(self, key, *, acq, x, model_mask, **kwargs):
        self.sampler_settings.append(kwargs)
        return jax.random.normal(key, (self.dimension,)) + x


def observation(dimension=3):
    return {
        "acq": jnp.zeros(1),
        "x": jnp.linspace(-0.1, 0.2, dimension),
        "mask_prior": jnp.array([0.5]),
        "model_mask": jnp.arange(dimension) % 2 == 0,
        "theta": jnp.linspace(-0.3, 0.1, dimension),
    }


def test_batched_statistics_preserve_per_trial_keys():
    batch = jax.tree.map(lambda value: jnp.stack([value] * 8), observation())
    keys = jax.random.split(jax.random.key(93), 8)
    combined = jax.jit(
        jax.vmap(make_sbc_fn(FakeModel(), mask_samples=7, theta_samples=9))
    )
    masks = jax.jit(jax.vmap(make_model_sbc_fn(FakeModel(), mask_samples=7)))
    thetas = jax.jit(jax.vmap(make_theta_sbc_fn(FakeModel(), theta_samples=9)))
    expected = combined(batch, keys)
    actual = {**masks(batch, keys), **thetas(batch, keys)}
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_model_only_factory_never_calls_parameter_head():
    class MaskOnlyModel(FakeModel):
        def sample_theta(self, *args, **kwargs):
            raise AssertionError("Parameter sampling is not part of model SBC")

    data = observation()
    data.pop("theta")
    result = make_model_sbc_fn(MaskOnlyModel())(data, jax.random.key(1))
    assert set(result) == {"model_pit"}
    assert np.isfinite(result["model_pit"])


def test_theta_only_factory_never_calls_model_head_and_keeps_sampler_settings():
    class ThetaOnlyModel(FakeModel):
        def sample_mask(self, *args, **kwargs):
            raise AssertionError("Model sampling is not part of parameter SBC")

        def log_prob_mask(self, *args, **kwargs):
            raise AssertionError("Model likelihood is not part of parameter SBC")

    model = ThetaOnlyModel()
    data = observation()
    data.pop("mask_prior")
    result = make_theta_sbc_fn(model, theta_samples=5, num_steps=19, t_min=0.02)(
        data, jax.random.key(2)
    )
    assert set(result) == {"theta_sum_pit", "theta_pit", "theta_mask"}
    assert model.sampler_settings == [
        {
            "sample_method": "ode",
            "num_steps": 19,
            "last_euler_step": False,
            "t_min": 0.02,
        }
    ]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, np.finfo(np.float32).max])
def test_nonfinite_or_overflowing_log_probabilities_invalidate_model_pit(invalid):
    class InvalidModel(FakeModel):
        def log_prob_mask(self, *args, **kwargs):
            return jnp.full(3, invalid)

    result = make_model_sbc_fn(InvalidModel(), mask_samples=2)(
        observation(), jax.random.key(3)
    )
    assert np.isnan(result["model_pit"])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, np.finfo(np.float32).max])
def test_nonfinite_or_overflowing_active_samples_invalidate_parameter_pits(invalid):
    class InvalidModel(FakeModel):
        def sample_theta(self, *args, **kwargs):
            return jnp.full(3, invalid)

    result = make_theta_sbc_fn(InvalidModel(), theta_samples=2)(
        observation(), jax.random.key(4)
    )
    assert np.isnan(result["theta_sum_pit"])
    assert np.isnan(result["theta_pit"]).all()


def test_nonfinite_active_truth_invalidates_parameter_statistics():
    data = observation()
    data["theta"] = data["theta"].at[0].set(jnp.nan)
    result = make_theta_sbc_fn(FakeModel(), theta_samples=3)(data, jax.random.key(5))
    assert np.isnan(result["theta_sum_pit"])
    assert np.isnan(result["theta_pit"]).all()


def test_inactive_nonfinite_values_do_not_change_active_statistics():
    class InactiveNanModel(FakeModel):
        def sample_theta(self, *args, **kwargs):
            return super().sample_theta(*args, **kwargs).at[1].set(jnp.nan)

    data = observation()
    key = jax.random.key(6)
    expected = make_theta_sbc_fn(FakeModel(), theta_samples=3)(data, key)
    data["theta"] = data["theta"].at[1].set(jnp.inf)
    actual = make_theta_sbc_fn(InactiveNanModel(), theta_samples=3)(data, key)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert np.isnan(actual["theta_pit"][1])


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (make_model_sbc_fn, {"mask_samples": 0}),
        (make_theta_sbc_fn, {"theta_samples": 0}),
        (make_theta_sbc_fn, {"num_steps": 0}),
        (make_theta_sbc_fn, {"t_min": -0.1}),
    ],
)
def test_invalid_sample_settings_fail_before_evaluation(factory, kwargs):
    with pytest.raises(ValueError):
        factory(FakeModel(), **kwargs)


def test_source_mode_skips_only_the_combined_outer_key_split():
    data = observation()
    source_mask = make_model_sbc_fn(FakeModel(), source_keys=True)
    source_theta = make_theta_sbc_fn(FakeModel(), source_keys=True)
    combined = make_sbc_fn(FakeModel())
    for seed in (0, 1, 2):
        key = jax.random.key(seed)
        mask_key, theta_key, _ = jax.random.split(key, 3)
        expected = combined(data, key)
        actual = {**source_mask(data, mask_key), **source_theta(data, theta_key)}
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])


def test_separated_statistics_match_frozen_numeric_references():
    reference = json.loads(
        (
            Path(__file__).parent / "fixtures/dmri_calibration_statistics.json"
        ).read_text()
    )
    for case in reference["cases"]:
        dimension = case["dimension"]
        data = observation(dimension)
        data["mask_prior"] = jnp.array([0.37])
        mask_fn = make_model_sbc_fn(
            FakeModel(dimension),
            mask_samples=case["mask_samples"],
            source_keys=case["source_keys"],
        )
        theta_fn = make_theta_sbc_fn(
            FakeModel(dimension),
            theta_samples=case["theta_samples"],
            num_steps=64,
            t_min=1e-4,
            source_keys=case["source_keys"],
        )
        key = jax.random.key(case["seed"])
        actual = {**mask_fn(data, key), **theta_fn(data, key)}
        for field, expected in case["expected"].items():
            np.testing.assert_allclose(
                actual[field],
                np.asarray(expected, dtype=float),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
