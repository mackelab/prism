"""Independent Gaussian and Stein-kernel checks for the supplementary correction."""

from __future__ import annotations

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from experiments.dmri._shared.metrics import (
    corrected_multiscale_ksd_and_pvalue,
    make_ksd_fn,
)
from prism.inference.ksd import multiscale_preconditioned_ksd_and_pvalue

LOWER = np.array([[0.8, 0.0], [0.3, 0.5]])
COVARIANCE = LOWER @ LOWER.T
POINTS = (
    np.array([[2.0, -0.5], [2.5, 0.0], [3.0, 1.0], [1.5, 0.8], [2.6, -1.0]]) @ LOWER.T
)


class GaussianSimulator:
    """Unit Gaussian prior plus a Gaussian likelihood gives covariance COVARIANCE."""

    theta_dim = 2

    def __init__(self, theta):
        self.theta = theta

    @staticmethod
    def theta_mask(mask):
        return mask

    @classmethod
    def from_theta(cls, theta, model_mask):
        return cls(theta)

    def log_likelihood(self, acq, x):
        likelihood_precision = jnp.linalg.inv(jnp.asarray(COVARIANCE)) - jnp.eye(2)
        return -0.5 * self.theta @ likelihood_precision @ self.theta


def analytic_ksd(points, preconditioner, bandwidths):
    """Direct pairwise Stein formula with the exact transformed Gaussian score."""
    points = np.asarray(points, dtype=np.float64)
    p = np.asarray(preconditioner, dtype=np.float64)
    z = points @ p.T
    # Gaussian score in the transformed coordinates, derived from its covariance.
    transformed_covariance = p @ COVARIANCE @ p.T
    scores = -z @ np.linalg.inv(transformed_covariance)
    total = 0.0
    for i in range(len(z)):
        for j in range(len(z)):
            if i == j:
                continue
            difference = z[i] - z[j]
            squared_distance = difference @ difference
            for bandwidth in bandwidths:
                h2 = bandwidth**2
                rbf = np.exp(-squared_distance / (2 * h2))
                total += (
                    (0.1 / len(bandwidths))
                    * rbf
                    * (
                        scores[i] @ scores[j]
                        + (scores[i] - scores[j]) @ difference / h2
                        + len(z[i]) / h2
                        - squared_distance / h2**2
                    )
                )
    return np.sqrt(max(total / (len(z) * (len(z) - 1)), 0.0))


def evaluate(kernel, preconditioner):
    return kernel(
        GaussianSimulator,
        thetas=jnp.asarray(POINTS, dtype=jnp.float32),
        mask=jnp.ones(2, dtype=bool),
        acq=jnp.zeros(1),
        x=jnp.zeros(2),
        key=jax.random.key(21),
        bandwidths=jnp.array([0.5, 1.0, 3.0]),
        precond_matrix=preconditioner,
        n_bootstrap=16,
    )


def test_correlated_gaussian_score_matches_autodiff():
    p = jnp.linalg.inv(jnp.asarray(LOWER))
    theta = jnp.array([1.0, -0.7])
    z = theta @ p.T

    def log_density_theta(value):
        return -0.5 * value @ jnp.linalg.solve(jnp.asarray(COVARIANCE), value)

    def transformed_log_density(value):
        return log_density_theta(jnp.linalg.solve(p, value)) - jnp.linalg.slogdet(p)[1]

    original_score = jax.grad(log_density_theta)(theta)
    autodiff_score = jax.grad(transformed_log_density)(z)
    correct_score = original_score @ jnp.linalg.inv(p)
    frozen_score = original_score @ jnp.linalg.inv(p).T
    np.testing.assert_allclose(autodiff_score, -z, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(correct_score, autodiff_score, rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(np.asarray(frozen_score - autodiff_score))) > 0.1


def test_nondiagonal_explicit_preconditioner_matches_analytic_stein_kernel():
    p = np.linalg.inv(LOWER)
    corrected, _ = evaluate(corrected_multiscale_ksd_and_pvalue, jnp.asarray(p))
    frozen, _ = evaluate(multiscale_preconditioned_ksd_and_pvalue, jnp.asarray(p))
    expected = analytic_ksd(POINTS, p, [0.5, 1.0, 3.0])
    np.testing.assert_allclose(corrected, expected, rtol=1e-5, atol=1e-6)
    assert abs(float(frozen) - expected) > 0.01


def test_empirical_preconditioner_matches_analytic_stein_kernel():
    covariance = np.cov(POINTS, rowvar=False) + 1e-3 * np.eye(2)
    p = np.linalg.inv(np.linalg.cholesky(covariance))
    corrected, _ = evaluate(corrected_multiscale_ksd_and_pvalue, None)
    expected = analytic_ksd(POINTS, p, [0.5, 1.0, 3.0])
    np.testing.assert_allclose(corrected, expected, rtol=1e-5, atol=1e-6)


def test_diagonal_preconditioner_matches_frozen_exactly():
    p = jnp.diag(jnp.array([2.0, 0.7]))
    corrected = evaluate(corrected_multiscale_ksd_and_pvalue, p)
    frozen = evaluate(multiscale_preconditioned_ksd_and_pvalue, p)
    np.testing.assert_array_equal(corrected[0], frozen[0])
    np.testing.assert_array_equal(corrected[1], frozen[1])


def test_only_kernel_code_change_is_row_score_transpose():
    frozen = ast.parse(inspect.getsource(multiscale_preconditioned_ksd_and_pvalue))
    corrected = ast.parse(inspect.getsource(corrected_multiscale_ksd_and_pvalue))
    corrected.body[0].name = frozen.body[0].name
    replacements = 0
    for node in ast.walk(corrected):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "scores_z"
                for target in node.targets
            )
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.MatMult)
        ):
            assert ast.dump(node.value.right) == ast.dump(
                ast.parse("P_inv_T.T", mode="eval").body
            )
            node.value.right = ast.Name(id="P_inv_T", ctx=ast.Load())
            replacements += 1
    assert replacements == 1
    assert ast.dump(corrected) == ast.dump(frozen)


def test_injected_kernel_receives_unchanged_draws_and_bootstrap_keys():
    class Model:
        @staticmethod
        def sample_theta(key, **kwargs):
            return jax.random.normal(key, (2,))

    def recording_kernel(sim_type, thetas, key, **kwargs):
        return thetas[0, 0], jax.random.uniform(key)

    key = jax.random.key(123)
    data = {"acq": jnp.zeros(1), "x": jnp.zeros(2), "model_mask": jnp.ones(2, bool)}
    result = make_ksd_fn(
        Model(), GaussianSimulator, theta_samples=3, kernel=recording_kernel
    )(data, key)
    posterior_key, prior_key = jax.random.split(key)
    posterior_sample_key, posterior_eval_key = jax.random.split(posterior_key)
    prior_sample_key, prior_eval_key = jax.random.split(prior_key)
    first_posterior = jax.random.normal(
        jax.random.split(posterior_sample_key, 3)[0], (2,)
    )[0]
    first_prior = jax.random.normal(prior_sample_key, (3, 2))[0, 0]
    np.testing.assert_array_equal(result["posterior_ksd"], first_posterior)
    np.testing.assert_array_equal(result["prior_ksd"], first_prior)
    np.testing.assert_array_equal(
        result["posterior_pvalue"], jax.random.uniform(posterior_eval_key)
    )
    np.testing.assert_array_equal(
        result["prior_pvalue"], jax.random.uniform(prior_eval_key)
    )
