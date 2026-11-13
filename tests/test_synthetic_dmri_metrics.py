"""Analytic checks and portable numeric references from frozen figure metrics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.dmri._shared.metrics import (
    _randomized_pit,
    make_ksd_fn,
    make_sbc_fn,
)
from prism.inference.ksd import multiscale_preconditioned_ksd_and_pvalue


class GaussianSimulator:
    """Unit-normal prior and a Gaussian observation in active coordinates."""

    theta_dim = 3

    def __init__(self, theta, mask):
        self.theta = theta
        self.mask = mask

    @staticmethod
    def theta_mask(mask):
        return mask

    @classmethod
    def from_theta(cls, theta, model_mask):
        return cls(theta, model_mask)

    def log_likelihood(self, acq, x):
        return -0.5 * jnp.sum(((self.theta - x) * self.mask) ** 2)


class ToyModel:
    def __init__(self):
        self.tokenizer = SimpleNamespace(simulator=GaussianSimulator)
        self.sampler_settings = []

    def sample_mask(self, key, *, acq, x, mask_prior):
        return jax.random.bernoulli(key, mask_prior[0], (3,))

    def log_prob_mask(self, mask, *, acq, x, mask_prior):
        # Per-bit log probabilities also exercise the frozen scalar reduction.
        p = mask_prior[0]
        return jnp.where(mask, jnp.log(p), jnp.log1p(-p))

    def sample_theta(self, key, acq=None, x=None, model_mask=None, **kwargs):
        self.sampler_settings.append(kwargs)
        return jax.random.normal(key, (3,)) + jnp.array([0.2, -0.5, 8.0])


@pytest.fixture(scope="module")
def frozen_reference():
    """Stored outputs from the exact frozen source; no sibling checkout needed."""
    path = Path(__file__).with_name("fixtures") / "synthetic_dmri_metrics.json"
    return json.loads(path.read_text())


@pytest.fixture
def data(frozen_reference):
    return {
        name: jnp.asarray(value) for name, value in frozen_reference["data"].items()
    }


def test_randomized_rank_endpoints_and_near_ties():
    tie_key, jitter_key = jax.random.split(jax.random.key(4))
    values = jnp.array([-2.0, -1.0, 1.0, 2.0])
    below = _randomized_pit(values, -3.0, tie_key, jitter_key, rtol=1e-6, atol=1e-6)
    above = _randomized_pit(values, 3.0, tie_key, jitter_key, rtol=1e-6, atol=1e-6)
    assert 0 <= float(below) < 0.2
    assert 0.8 <= float(above) < 1
    # Near-equality uses isclose; all four values must enter the tie group.
    near = _randomized_pit(
        jnp.array([0.999999, 1.0, 1.000001, 1.0]),
        1.0,
        tie_key,
        jitter_key,
        rtol=1e-3,
        atol=1e-3,
    )
    exact = _randomized_pit(jnp.ones(4), 1.0, tie_key, jitter_key, rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(near, exact)


def test_all_ties_produce_uniform_randomized_ranks():
    keys = jax.random.split(jax.random.key(75), 2000)

    def evaluate(key):
        tie_key, jitter_key = jax.random.split(key)
        return _randomized_pit(
            jnp.ones(8), 1.0, tie_key, jitter_key, rtol=1e-6, atol=1e-6
        )

    pits = np.asarray(jax.jit(jax.vmap(evaluate))(keys))
    grid = np.linspace(0.0, 1.0, 11)
    empirical = np.mean(pits[:, None] <= grid[None, :], axis=0)
    assert np.max(np.abs(empirical - grid)) < 0.04


def test_sbc_matches_frozen_scalar_statistics(data, frozen_reference):
    reference = frozen_reference["sbc"]
    model = ToyModel()
    key = jax.random.key(reference["seed"])
    actual = make_sbc_fn(
        model,
        mask_samples=reference["mask_samples"],
        theta_samples=reference["theta_samples"],
    )(data, key)
    for name, expected in reference["expected"].items():
        np.testing.assert_array_equal(actual[name], expected)
    assert model.sampler_settings == [reference["sampler"]]


def test_coordinate_ranks_reuse_theta_draws_and_mask_inactive(data):
    key = jax.random.key(19)
    model = ToyModel()
    actual = make_sbc_fn(
        model, mask_samples=5, theta_samples=7, num_steps=9, t_min=0.02
    )(data, key)
    _, theta_key, _ = jax.random.split(key, 3)
    sample_key, _, _ = jax.random.split(theta_key, 3)
    samples = np.asarray(
        jax.vmap(lambda k: ToyModel().sample_theta(k))(jax.random.split(sample_key, 7))
    )
    truth = np.asarray(data["theta"])
    expected = (0.5 + np.sum(samples < truth, axis=0)) / 8
    np.testing.assert_allclose(actual["theta_pit"][:2], expected[:2], rtol=1e-7)
    assert np.isnan(actual["theta_pit"][2])
    np.testing.assert_array_equal(actual["theta_mask"], data["model_mask"])
    assert len(model.sampler_settings) == 1  # One posterior draw operation per trace.
    assert model.sampler_settings[0]["num_steps"] == 9
    assert model.sampler_settings[0]["t_min"] == 0.02


def test_paper_coordinate_ranks_use_strict_comparison_and_midpoints(data):
    class FixedThetaModel(ToyModel):
        def sample_theta(self, key, **kwargs):
            return jnp.array([0.1, -0.2000001, 1e9])

    evaluate = make_sbc_fn(FixedThetaModel(), mask_samples=2, theta_samples=3)
    for seed in (2, 9):
        result = evaluate(data, jax.random.key(seed))
        # Exact ties do not count as lower; near ties strictly below do count.
        np.testing.assert_array_equal(result["theta_pit"][:2], [0.125, 0.875])
        assert np.isnan(result["theta_pit"][2])


def test_invalid_log_probabilities_produce_nan_model_pit(data):
    class InvalidProbabilityModel(ToyModel):
        def log_prob_mask(self, mask, **kwargs):
            return jnp.array([0.0, jnp.nan, 0.0])

    result = make_sbc_fn(InvalidProbabilityModel(), mask_samples=3, theta_samples=3)(
        data, jax.random.key(3)
    )
    assert np.isnan(result["model_pit"])
    assert np.isfinite(result["theta_sum_pit"])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, np.finfo(np.float32).max])
def test_invalid_active_draw_or_overflow_produces_nan_parameter_pits(data, invalid):
    class InvalidParameterModel(ToyModel):
        def sample_theta(self, key, **kwargs):
            return jnp.array([invalid, invalid, 0.0])

    result = make_sbc_fn(InvalidParameterModel(), mask_samples=3, theta_samples=3)(
        data, jax.random.key(3)
    )
    assert np.isfinite(result["model_pit"])
    assert np.isnan(result["theta_sum_pit"])
    assert np.all(np.isnan(result["theta_pit"]))


def test_invalid_active_truth_produces_nan_parameter_pits(data):
    invalid_data = dict(data, theta=data["theta"].at[0].set(jnp.nan))
    result = make_sbc_fn(ToyModel(), mask_samples=3, theta_samples=3)(
        invalid_data, jax.random.key(3)
    )
    assert np.isnan(result["theta_sum_pit"])
    assert np.all(np.isnan(result["theta_pit"]))


def test_inactive_nan_draws_and_truth_do_not_contaminate_active_statistics(data):
    class InactiveNanModel(ToyModel):
        def sample_theta(self, key, **kwargs):
            return super().sample_theta(key, **kwargs).at[2].set(jnp.nan)

    key = jax.random.key(3)
    expected = make_sbc_fn(ToyModel(), mask_samples=3, theta_samples=3)(data, key)
    invalid_data = dict(data, theta=data["theta"].at[2].set(jnp.nan))
    actual = make_sbc_fn(InactiveNanModel(), mask_samples=3, theta_samples=3)(
        invalid_data, key
    )
    np.testing.assert_array_equal(actual["theta_sum_pit"], expected["theta_sum_pit"])
    np.testing.assert_array_equal(actual["theta_pit"][:2], expected["theta_pit"][:2])
    assert np.isnan(actual["theta_pit"][2])


def _reference_ksd_draws(key, sample_count):
    posterior_key, prior_key = jax.random.split(key)
    posterior_sample_key, posterior_eval_key = jax.random.split(posterior_key)
    prior_sample_key, prior_eval_key = jax.random.split(prior_key)
    posterior = jax.vmap(lambda k: jax.random.normal(k, (3,)))(
        jax.random.split(posterior_sample_key, sample_count)
    ) + jnp.array([0.2, -0.5, 8.0])
    prior = jax.random.normal(prior_sample_key, (sample_count, 3))
    return ((posterior, posterior_eval_key), (prior, prior_eval_key))


def _gaussian_bootstrap_reference(thetas, data, key, bandwidths, n_bootstrap):
    """Direct float64 Gaussian Stein formula with explicit pair differences."""
    samples = np.asarray(thetas, dtype=np.float64)
    mask = np.asarray(data["model_mask"], dtype=bool)
    n, d = samples.shape
    centered = (samples - samples.mean(axis=0)) * mask
    lower = np.linalg.cholesky(centered.T @ centered / (n - 1) + 1e-3 * np.eye(d))
    # Preserve the frozen kernel's row-score transform, including its transpose.
    scores = ((np.asarray(data["x"]) - 2 * samples) * mask) @ lower.T
    scores *= mask
    z = samples @ np.linalg.inv(lower).T * mask
    difference = z[:, None, :] - z[None, :, :]
    squared_distance = np.sum(difference**2, axis=-1)
    cross_score = np.sum(
        (scores[:, None, :] - scores[None, :, :]) * difference, axis=-1
    )
    stein = np.zeros((n, n))
    for bandwidth in bandwidths:
        h2 = float(bandwidth) ** 2
        stein += (
            0.1
            / len(bandwidths)
            * np.exp(-squared_distance / (2 * h2))
            * (
                scores @ scores.T
                + cross_score / h2
                + mask.sum() / h2
                - squared_distance / h2**2
            )
        )
    np.fill_diagonal(stein, 0)
    observed = stein.sum() / (n * (n - 1))

    def draw(key, _):
        key, subkey = jax.random.split(key)
        signs = 2 * jax.random.bernoulli(subkey, 0.5, (n,)).astype(jnp.int32) - 1
        return key, signs

    signs = np.asarray(jax.lax.scan(draw, key, None, length=n_bootstrap)[1])
    statistics = np.einsum("bi,ij,bj->b", signs, stein, signs) / (n * (n - 1))
    # Constant signs give exactly the observed quadratic form mathematically.
    # Float32 sum and matrix-product reductions can put these on either side.
    ties = np.all(signs == signs[:, :1], axis=1)
    margin = np.min(np.abs(statistics[~ties] - observed))
    scale = max(1.0, abs(observed), np.max(np.abs(statistics)))
    assert margin > 64 * np.finfo(np.float32).eps * scale
    strict_tail = int(np.sum(statistics[~ties] > observed))
    return np.sqrt(max(observed, 0.0)), strict_tail, int(ties.sum())


def _assert_bootstrap_pvalue(value, strict_tail, ties, n_bootstrap):
    numerator = round(float(value) * (n_bootstrap + 1))
    expected = numerator / (n_bootstrap + 1)
    # Permit only scalar float32 rounding, not a different bootstrap count.
    np.testing.assert_allclose(
        value, expected, rtol=0, atol=np.spacing(np.float32(expected))
    )
    assert 1 + strict_tail <= numerator <= 1 + strict_tail + ties


@pytest.mark.parametrize("case", ["ksd", "ksd_shifted_observation"])
def test_ksd_matches_frozen_posterior_and_prior(data, frozen_reference, case):
    reference = frozen_reference[case]
    data = dict(data)
    data.update({
        name: jnp.asarray(value)
        for name, value in reference.get("data_override", {}).items()
    })
    model = ToyModel()
    key = jax.random.key(reference["seed"])
    actual = make_ksd_fn(
        model,
        GaussianSimulator,
        theta_samples=reference["theta_samples"],
        bandwidths=jnp.asarray(reference["bandwidths"]),
    )(data, key)
    for name in ("posterior_ksd", "prior_ksd"):
        # Allow float32 arithmetic differences between CPU architectures.
        np.testing.assert_allclose(
            actual[name], reference["expected"][name], rtol=1e-6, atol=1e-7
        )
    draws = _reference_ksd_draws(key, reference["theta_samples"])
    for name, (thetas, evaluation_key) in zip(
        ("posterior", "prior"), draws, strict=True
    ):
        _, strict_tail, ties = _gaussian_bootstrap_reference(
            thetas,
            data,
            evaluation_key,
            reference["bandwidths"],
            reference["n_bootstrap"],
        )
        for value in (
            actual[f"{name}_pvalue"],
            reference["expected"][f"{name}_pvalue"],
        ):
            _assert_bootstrap_pvalue(value, strict_tail, ties, reference["n_bootstrap"])
    for name, value in reference["sampler"].items():
        assert model.sampler_settings[0][name] == value


@pytest.mark.parametrize("mutation", [None, "normalization", "bootstrap_key", "mask"])
def test_bootstrap_without_ties_has_exact_tail_count(data, mutation):
    key = jax.random.key(101)
    bandwidths = [0.1, 0.5, 1.0, 10.0]

    def kernel(*args, **kwargs):
        if mutation == "bootstrap_key":
            kwargs["key"] = jax.random.fold_in(kwargs["key"], 1)
        elif mutation == "mask":
            kwargs["mask"] = jnp.ones_like(kwargs["mask"])
        ksd, pvalue = multiscale_preconditioned_ksd_and_pvalue(*args, **kwargs)
        if mutation == "normalization":
            pvalue *= 513 / 512
        return ksd, pvalue

    result = make_ksd_fn(
        ToyModel(),
        GaussianSimulator,
        theta_samples=32,
        bandwidths=bandwidths,
        kernel=kernel,
    )(data, key)
    thetas, evaluation_key = _reference_ksd_draws(key, 32)[0]
    _, strict_tail, ties = _gaussian_bootstrap_reference(
        thetas, data, evaluation_key, bandwidths, 512
    )
    assert ties == 0 and 0 < strict_tail < 512

    def check():
        _assert_bootstrap_pvalue(result["posterior_pvalue"], strict_tail, ties, 512)

    if mutation is None:
        check()
    else:
        with pytest.raises(AssertionError):
            check()


def test_kernel_matches_analytic_one_dimensional_gaussian_score():
    # On the active dimension, log likelihood + log prior has score -2 theta.
    # Deliberately biased samples make the U-statistic positive and distinguish
    # the returned KSD from its square. Inactive dimensions must have no effect.
    samples = np.array(
        [[1.5, 4.0, -3.0], [2.0, -2.0, 8.0], [2.5, 9.0, 5.0]], np.float32
    )
    theta = samples[:, 0]
    score = -2 * theta
    distance = theta[:, None] - theta[None, :]
    kernel = np.exp(-(distance**2) / 2)
    stein = (
        0.1
        * kernel
        * (
            score[:, None] * score[None, :]
            + (score[:, None] - score[None, :]) * distance
            + 1
            - distance**2
        )
    )
    np.fill_diagonal(stein, 0)
    expected = np.sqrt(stein.sum() / 6)
    actual, pvalue = multiscale_preconditioned_ksd_and_pvalue(
        GaussianSimulator,
        jnp.asarray(samples),
        mask=jnp.array([True, False, False]),
        acq=jnp.zeros(1),
        x=jnp.zeros(3),
        key=jax.random.key(8),
        bandwidths=jnp.array([1.0]),
        precond_matrix=jnp.eye(3),
        n_bootstrap=16,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert 1 / 17 <= float(pvalue) <= 1


@pytest.mark.parametrize(
    "kwargs", [{"theta_samples": 1}, {"num_steps": 0}, {"bandwidths": ()}]
)
def test_invalid_ksd_settings_fail_before_evaluation(kwargs):
    with pytest.raises(ValueError):
        make_ksd_fn(ToyModel(), GaussianSimulator, **kwargs)
