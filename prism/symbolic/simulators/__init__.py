"""Additive symbolic simulators and a small polynomial example."""

import jax.numpy as jnp
import sympy as sp

from .base import AdditiveModel
from .model import SymbolicExpressionCompartment, SymbolicNoiseCompartment


class Linear(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("a*x")
    expression_str = "a*x"
    param_symbols = (sp.Symbol("a"),)
    param_low = jnp.array([-2.0])
    param_high = jnp.array([2.0])


class Quadratic(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("b*x**2")
    expression_str = "b*x**2"
    param_symbols = (sp.Symbol("b"),)
    param_low = jnp.array([-1.0])
    param_high = jnp.array([1.0])


class GaussianNoise(SymbolicNoiseCompartment):
    theta_dim = 1
    scale_expression = sp.Symbol("sigma")
    expression_str = "sigma"
    param_symbols = (sp.Symbol("sigma"),)
    param_low = jnp.array([0.05])
    param_high = jnp.array([0.25])


class SimplePolynomial(AdditiveModel):
    """Select linear/quadratic terms and infer their coefficients and noise scale."""

    model_types = [Linear, Quadratic]
    noise_types = [GaussianNoise]


__all__ = [
    "AdditiveModel",
    "SymbolicExpressionCompartment",
    "SymbolicNoiseCompartment",
    "Linear",
    "Quadratic",
    "GaussianNoise",
    "SimplePolynomial",
]
