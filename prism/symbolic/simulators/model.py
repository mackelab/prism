# Adapted from mackelab/sbmi_transformer_symbolic (MIT), c8e118b.
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import sympy as sp

from prism.symbolic.simulators.base import (
    NoiseCompartment,
    SymbolicCompartment,
)


def _bounded_to_theta(params, low, high, expected_dim, cls_name):
    """Map bounded params to an unconstrained Gaussian space."""
    params_arr = jnp.asarray(params)
    if params_arr.shape[0] != expected_dim:
        raise ValueError(
            f"{cls_name} expected {expected_dim} params, got {params_arr.shape[0]}"
        )
    if low is None or high is None:
        return params_arr
    width = high - low
    u = (params_arr - low) / width
    return jstats.norm.ppf(u)


def _theta_to_bounded(theta, low, high, expected_dim, cls_name):
    """Map unconstrained theta back to bounded params."""
    theta_arr = jnp.asarray(theta)
    if theta_arr.shape[0] != expected_dim:
        raise ValueError(
            f"{cls_name} expected theta_dim {expected_dim}, got {theta_arr.shape[0]}"
        )
    if low is None or high is None:
        return theta_arr
    u = jstats.norm.cdf(theta_arr)
    return low + (high - low) * u


class SymbolicExpressionCompartment(SymbolicCompartment):
    """Compartment wrapping a single symbolic expression."""

    theta_dim: int = 0
    expression: sp.Expr
    expression_str: str
    param_symbols: Sequence[sp.Symbol]
    param_low: jnp.ndarray | None = None
    param_high: jnp.ndarray | None = None
    variable: sp.Symbol = sp.Symbol("x")

    def __init__(self, params):
        self.params_array = jnp.asarray(params)
        self._lambda = sp.lambdify(
            (self.variable, *self.param_symbols), self.expression, modules="jax"
        )

    @property
    def params(self):
        return {"params": self.params_array}

    def __str__(self) -> str:
        return self.expression_str

    @classmethod
    def to_theta(cls, params):
        return _bounded_to_theta(
            params, cls.param_low, cls.param_high, cls.theta_dim, cls.__name__
        )

    @classmethod
    def to_params(cls, theta):
        params = _theta_to_bounded(
            theta, cls.param_low, cls.param_high, cls.theta_dim, cls.__name__
        )
        return (params,)

    def evaluate(self, x):
        res = self._lambda(x, *list(self.params_array))
        if jnp.ndim(res) == 0:
            res = jnp.broadcast_to(res, jnp.shape(x))
        return res


class SymbolicNoiseCompartment(NoiseCompartment):
    """Noise compartment wrapping a symbolic scale expression."""

    theta_dim: int = 0
    scale_expression: sp.Expr
    expression_str: str
    param_symbols: Sequence[sp.Symbol]
    param_low: jnp.ndarray | None = None
    param_high: jnp.ndarray | None = None
    variable: sp.Symbol = sp.Symbol("x")

    def __init__(self, params):
        self.params_array = jnp.asarray(params)
        self._scale_fn = sp.lambdify(
            (self.variable, *self.param_symbols), self.scale_expression, modules="jax"
        )

    @property
    def params(self):
        return {"params": self.params_array}

    def __str__(self) -> str:
        return self.expression_str

    @classmethod
    def to_theta(cls, params):
        return _bounded_to_theta(
            params, cls.param_low, cls.param_high, cls.theta_dim, cls.__name__
        )

    @classmethod
    def to_params(cls, theta):
        params = _theta_to_bounded(
            theta, cls.param_low, cls.param_high, cls.theta_dim, cls.__name__
        )
        return (params,)

    def _scale(self, x):
        scale = self._scale_fn(x, *list(self.params_array))
        if jnp.ndim(scale) == 0:
            scale = jnp.broadcast_to(scale, jnp.shape(x))
        return scale

    def _safe_scale(self, x):
        # logpdf requires strictly positive scale; noise expressions can cross 0.
        scale = self._scale(x)
        return jnp.maximum(jnp.abs(scale), 1e-6)

    def noise(self, x, rng: jax.random.PRNGKey, signal=None):
        scale = self._safe_scale(x)
        return jax.random.normal(rng, shape=jnp.shape(x)) * scale

    def log_likelihood(self, signal: jax.Array, signal_observed: jax.Array, x=None):
        if x is None:
            x = signal
        scale = self._safe_scale(x)
        return jstats.norm.logpdf(signal_observed, loc=signal, scale=scale).sum(axis=-1)
