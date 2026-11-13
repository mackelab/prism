"""Frozen Figure 4b and K=15/30/50 symbolic model specifications.

Adapted from mackelab/sbmi_transformer_symbolic (MIT), commit
c8e118b8bd557f2b001812f00af982b136f0d793. Component order, bounds and noise
expressions match that source; the small polynomial example is independent.
"""

import jax
import jax.numpy as jnp
import sympy as sp

from prism.symbolic.paper.mask_prior import SymbolicMaskPrior
from prism.symbolic.simulators.base import AdditiveModel
from prism.symbolic.simulators.model import (
    SymbolicExpressionCompartment,
    SymbolicNoiseCompartment,
)


class Linear(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("c_1*x")
    expression_str = "c_1*x"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([-2.0])
    param_high = jnp.array([2.0])


class Quadratic(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("c_1*x*x")
    expression_str = "c_1*x*x"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([-0.5])
    param_high = jnp.array([0.5])


class ShiftedSquare(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("(c_1+x)*(c_1+x)")
    expression_str = "(c_1+x)*(c_1+x)"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([-5.0])
    param_high = jnp.array([0.0])


class Cubic(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("c_1*x*x*x")
    expression_str = "c_1*x*x*x"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([-0.1])
    param_high = jnp.array([0.1])


class Sinusoidal(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*sin(c_2*x)")
    expression_str = "c_1*sin(c_2*x)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([0.0, 0.5])
    param_high = jnp.array([5.0, 5.0])


class Cosinusoidal(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*cos(c_2*x)")
    expression_str = "c_1*cos(c_2*x)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([0.0, 0.5])
    param_high = jnp.array([5.0, 5.0])


class ConstantWide(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("c_1")
    expression_str = "c_1"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([-5.0])
    param_high = jnp.array([5.0])


class ConstantPositive(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("c_1")
    expression_str = "c_1"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([0.0])
    param_high = jnp.array([10.0])


class TanhRight(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*tanh(x-c_2)")
    expression_str = "c_1*tanh(x-c_2)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([1.0, 2.0])
    param_high = jnp.array([10.0, 8.0])


class TanhLeft(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*tanh(-x+c_2)")
    expression_str = "c_1*tanh(-x+c_2)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([1.0, 2.0])
    param_high = jnp.array([10.0, 8.0])


class GaussianBump(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*exp(-(x-c_2)*(x-c_2))")
    expression_str = "c_1*exp(-(x-c_2)*(x-c_2))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([1.0, 2.0])
    param_high = jnp.array([10.0, 8.0])


class GaussianWide(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*exp(-(x-c_2)*(x-c_2)/8)")
    expression_str = "c_1*exp(-(x-c_2)*(x-c_2)/8)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([1.0, 2.0])
    param_high = jnp.array([10.0, 8.0])


class RampUp(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*Piecewise((0.0,x<c_2),(x, x>=c_2))")
    expression_str = "c_1*Piecewise((0.0,x<c_2),(x, x>=c_2))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([1.0, 2.0])
    param_high = jnp.array([5.0, 8.0])


class RampDown(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*Piecewise((0.0,x>c_2),(-x+c_2, x<=c_2))")
    expression_str = "c_1*Piecewise((0.0,x>c_2),(-x+c_2, x<=c_2))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([1.0, 2.0])
    param_high = jnp.array([5.0, 8.0])


expressions_v0 = [
    Linear,
    Linear,
    Quadratic,
    ShiftedSquare,
    Cubic,
    Sinusoidal,
    Cosinusoidal,
    ConstantWide,
    ConstantPositive,
    TanhRight,
    TanhLeft,
    GaussianBump,
    GaussianWide,
    RampUp,
    RampDown,
]


# Expressions for noise scale; include parameters as c_1, c_2, ...
# Noise expressions define scale for additive Gaussian noise via normal(c_i)
noise_expressions_v0 = [
    "normal(c_1)",  # 0: observer noise
    "normal(c_1) * (x+1)",  # 1: increasing noise
    "normal(c_1) * (11 - x)",  # 2: decreasing noise
    "normal(c_1) *(x**2 + 1)",  # 3: quadratic noise
    "normal(c_1) * (11 - x**2)",  # 4: quadratic decreasing noise
]


prior_bounds_v0 = jnp.array([
    [
        -2,
        -2,
        -0.5,
        -5,
        -0.1,
        0,
        0.5,
        0,
        0.5,
        -5,
        0,
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        0.1,
        0.5,
        0.5,
        0.2,
        0.2,
    ],
    [
        2,
        2,
        0.5,
        0,
        0.1,
        5,
        5,
        5,
        5,
        5,
        10,
        10,
        8,
        10,
        8,
        10,
        8,
        10,
        8,
        5,
        8,
        5,
        8,
        2,
        2,
        2,
        1,
        1,
    ],
])

_num_model_params = sum(cls.theta_dim for cls in expressions_v0)
_num_noise_params = 5  # one parameter per noise expression
assert prior_bounds_v0.shape[1] == _num_model_params + _num_noise_params, (
    "prior_bounds_v0 length must match total params"
)
_noise_low = prior_bounds_v0[0, -_num_noise_params:]
_noise_high = prior_bounds_v0[1, -_num_noise_params:]


class NoiseObserver(SymbolicNoiseCompartment):
    theta_dim = 1
    scale_expression = sp.sympify("c_1")
    expression_str = "normal(c_1)"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([_noise_low[0]])
    param_high = jnp.array([_noise_high[0]])


class NoiseIncreasing(SymbolicNoiseCompartment):
    theta_dim = 1
    scale_expression = sp.sympify("c_1 * (x+1)")
    expression_str = "normal(c_1) * (x+1)"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([_noise_low[1]])
    param_high = jnp.array([_noise_high[1]])


class NoiseDecreasing(SymbolicNoiseCompartment):
    theta_dim = 1
    scale_expression = sp.sympify("c_1 * (11 - x)")
    expression_str = "normal(c_1) * (11 - x)"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([_noise_low[2]])
    param_high = jnp.array([_noise_high[2]])


class NoiseQuadratic(SymbolicNoiseCompartment):
    theta_dim = 1
    scale_expression = sp.sympify("c_1 * (x**2 + 1)")
    expression_str = "normal(c_1) *(x**2 + 1)"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([_noise_low[3]])
    param_high = jnp.array([_noise_high[3]])


class NoiseQuadraticDecreasing(SymbolicNoiseCompartment):
    theta_dim = 1
    scale_expression = sp.sympify("c_1 * (11 - x**2)")
    expression_str = "normal(c_1) * (11 - x**2)"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([_noise_low[4]])
    param_high = jnp.array([_noise_high[4]])


symbolic_component_types_v0 = tuple(expressions_v0)
symbolic_noise_types_v0 = (
    NoiseObserver,
    NoiseIncreasing,
    NoiseDecreasing,
    NoiseQuadratic,
    NoiseQuadraticDecreasing,
)


class SymbolicModelV0(AdditiveModel):
    """The fifteen-function, five-noise model used for Figure 4b."""

    model_types = symbolic_component_types_v0
    noise_types = symbolic_noise_types_v0
    mask_prior_cls = SymbolicMaskPrior

    def log_likelihood(self, x: jax.Array, signal_observed: jax.Array):
        """Preserve the paper's sum over selected noise-component likelihoods."""
        mask = self._active_mask()
        values = []
        for active, comp in zip(mask, self.model_compartments, strict=True):
            val = comp.evaluate(x)
            values.append(jnp.where(active, val, jnp.zeros_like(val)))
        signal = (
            jnp.sum(jnp.stack(values, axis=0), axis=0) if values else jnp.array(0.0)
        )
        if not self.noise_compartments:
            matches = jnp.all(signal_observed == signal)
            return jnp.where(matches, 0.0, -jnp.inf)

        noise_mask = self._noise_active_mask()
        log_likes = []
        for active, comp in zip(noise_mask, self.noise_compartments, strict=True):
            ll_val = comp.log_likelihood(signal, signal_observed, x=x)
            log_likes.append(jnp.where(active, ll_val, 0.0))
        log_like = (
            jnp.sum(jnp.stack(log_likes, axis=0), axis=0)
            if log_likes
            else jnp.array(0.0)
        )
        return jnp.sum(log_like)


class SymbolicModelV15(AdditiveModel):
    """The source K=15 scaling model with a Beta-Bernoulli component prior."""

    model_types = symbolic_component_types_v0
    noise_types = symbolic_noise_types_v0

    log_likelihood = SymbolicModelV0.log_likelihood


class QuarticScaled(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("c_1*(x/10)**4")
    expression_str = "c_1*(x/10)**4"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([-5.0])
    param_high = jnp.array([5.0])


class QuinticScaled(SymbolicExpressionCompartment):
    theta_dim = 1
    expression = sp.sympify("c_1*(x/10)**5")
    expression_str = "c_1*(x/10)**5"
    param_symbols = (sp.Symbol("c_1"),)
    param_low = jnp.array([-5.0])
    param_high = jnp.array([5.0])


class SinusoidalPhase(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*sin(c_2*x + c_3)")
    expression_str = "c_1*sin(c_2*x + c_3)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.5, -jnp.pi])
    param_high = jnp.array([5.0, 5.0, jnp.pi])


class CosinusoidalPhase(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*cos(c_2*x + c_3)")
    expression_str = "c_1*cos(c_2*x + c_3)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.5, -jnp.pi])
    param_high = jnp.array([5.0, 5.0, jnp.pi])


class ExponentialDecay(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*exp(-c_2*x)")
    expression_str = "c_1*exp(-c_2*x)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([0.0, 0.1])
    param_high = jnp.array([10.0, 2.0])


class SaturatingExponential(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*(1-exp(-c_2*x))")
    expression_str = "c_1*(1-exp(-c_2*x))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([0.0, 0.1])
    param_high = jnp.array([10.0, 2.0])


class Logarithmic(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*log(x+c_2)")
    expression_str = "c_1*log(x+c_2)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([-5.0, 0.1])
    param_high = jnp.array([5.0, 2.0])


class SquareRoot(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*sqrt(x+c_2)")
    expression_str = "c_1*sqrt(x+c_2)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([-5.0, 0.0])
    param_high = jnp.array([5.0, 2.0])


class Reciprocal(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1/(x+c_2)")
    expression_str = "c_1/(x+c_2)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([-10.0, 0.5])
    param_high = jnp.array([10.0, 3.0])


class AbsoluteValue(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*Abs(x-c_2)")
    expression_str = "c_1*Abs(x-c_2)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([-5.0, 0.0])
    param_high = jnp.array([5.0, 10.0])


class InverseQuadratic(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1/(1 + c_2*(x-c_3)*(x-c_3))")
    expression_str = "c_1/(1 + c_2*(x-c_3)*(x-c_3))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.1, 0.0])
    param_high = jnp.array([10.0, 2.0, 10.0])


class Lorentzian(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1/(1 + ((x-c_2)/c_3)**2)")
    expression_str = "c_1/(1 + ((x-c_2)/c_3)**2)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.0, 0.5])
    param_high = jnp.array([10.0, 10.0, 5.0])


class Sigmoid(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1/(1+exp(-c_2*(x-c_3)))")
    expression_str = "c_1/(1+exp(-c_2*(x-c_3)))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.1, 0.0])
    param_high = jnp.array([10.0, 5.0, 10.0])


class DampedSinusoidal(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*exp(-c_2*x)*sin(c_3*x)")
    expression_str = "c_1*exp(-c_2*x)*sin(c_3*x)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.05, 0.5])
    param_high = jnp.array([5.0, 1.0, 8.0])


class DampedCosinusoidal(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*exp(-c_2*x)*cos(c_3*x)")
    expression_str = "c_1*exp(-c_2*x)*cos(c_3*x)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.05, 0.5])
    param_high = jnp.array([5.0, 1.0, 8.0])


class TanhCentered(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*tanh(c_2*(x-c_3))")
    expression_str = "c_1*tanh(c_2*(x-c_3))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([-10.0, 0.1, 2.0])
    param_high = jnp.array([10.0, 2.0, 8.0])


class ExponentialGrowth(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*exp(c_2*x)")
    expression_str = "c_1*exp(c_2*x)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([0.0, 0.05])
    param_high = jnp.array([10.0, 0.8])


class PowerLawDecay(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1/(x + c_2)**c_3")
    expression_str = "c_1/(x + c_2)**c_3"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.5, 0.5])
    param_high = jnp.array([10.0, 5.0, 3.0])


class ArctangentStep(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*atan(c_2*(x-c_3))")
    expression_str = "c_1*atan(c_2*(x-c_3))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.1, 0.0])
    param_high = jnp.array([10.0, 2.0, 10.0])


class HyperbolicSecant(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*sech(c_2*(x-c_3))")
    expression_str = "c_1*sech(c_2*(x-c_3))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.1, 0.0])
    param_high = jnp.array([8.0, 2.0, 10.0])


class SincDecay(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*sin(c_2*x)/(x+c_3)")
    expression_str = "c_1*sin(c_2*x)/(x+c_3)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.5, 0.5])
    param_high = jnp.array([5.0, 5.0, 5.0])


class AbsoluteSinusoidal(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*Abs(sin(c_2*x + c_3))")
    expression_str = "c_1*Abs(sin(c_2*x + c_3))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.5, -jnp.pi])
    param_high = jnp.array([5.0, 5.0, jnp.pi])


class RectifiedLinear(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*Piecewise((0, x < c_2), (x - c_2, True))")
    expression_str = "c_1*Piecewise((0, x < c_2), (x - c_2, True))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([0.0, 0.0])
    param_high = jnp.array([5.0, 8.0])


class Softplus(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*log(1+exp(c_2*(x-c_3)))/c_2")
    expression_str = "c_1*log(1+exp(c_2*(x-c_3)))/c_2"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.1, 0.0])
    param_high = jnp.array([5.0, 5.0, 10.0])


class Gompertz(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify("c_1*exp(-c_2*exp(-c_3*x))")
    expression_str = "c_1*exp(-c_2*exp(-c_3*x))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.1, 0.05])
    param_high = jnp.array([10.0, 3.0, 1.5])


class LinearFractional(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*x/(1 + c_2*x)")
    expression_str = "c_1*x/(1 + c_2*x)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([-5.0, 0.1])
    param_high = jnp.array([5.0, 2.0])


class SineSquared(SymbolicExpressionCompartment):
    theta_dim = 2
    expression = sp.sympify("c_1*sin(c_2*x)**2")
    expression_str = "c_1*sin(c_2*x)**2"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([0.0, 0.5])
    param_high = jnp.array([5.0, 5.0])


class TriangularBump(SymbolicExpressionCompartment):
    theta_dim = 3
    expression = sp.sympify(
        "c_1*Piecewise((0, Abs((x-c_2)/c_3) >= 1), (1 - Abs((x-c_2)/c_3), True))"
    )
    expression_str = (
        "c_1*Piecewise((0, Abs((x-c_2)/c_3) >= 1), (1 - Abs((x-c_2)/c_3), True))"
    )
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.5, 0.5])
    param_high = jnp.array([5.0, 5.0, 5.0])


class NoiseExponential(SymbolicNoiseCompartment):
    theta_dim = 2
    scale_expression = sp.sympify("c_1 * exp(c_2 * x)")
    expression_str = "normal(c_1) * exp(c_2 * x)"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"))
    param_low = jnp.array([0.0, 0.05])
    param_high = jnp.array([5.0, 0.5])


class NoiseSigmoid(SymbolicNoiseCompartment):
    theta_dim = 3
    scale_expression = sp.sympify("c_1/(1+exp(-c_2*(x-c_3)))")
    expression_str = "normal(c_1)/(1+exp(-c_2*(x-c_3)))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.1, 0.0])
    param_high = jnp.array([5.0, 1.0, 10.0])


class NoisePeaked(SymbolicNoiseCompartment):
    theta_dim = 3
    scale_expression = sp.sympify("c_1*exp(-((x-c_2)**2)/(c_3 + 1e-3))")
    expression_str = "normal(c_1) * exp(-((x-c_2)**2)/(c_3 + 1e-3))"
    param_symbols = (sp.Symbol("c_1"), sp.Symbol("c_2"), sp.Symbol("c_3"))
    param_low = jnp.array([0.0, 0.0, 0.5])
    param_high = jnp.array([5.0, 10.0, 5.0])


# Expanded set of expression types; used to build larger symbolic models.
symbolic_component_palette_v1 = (
    Linear,
    Quadratic,
    ShiftedSquare,
    Cubic,
    Sinusoidal,
    Cosinusoidal,
    ConstantWide,
    ConstantPositive,
    TanhRight,
    TanhLeft,
    TanhCentered,
    GaussianBump,
    GaussianWide,
    RampUp,
    RampDown,
    QuarticScaled,
    QuinticScaled,
    SinusoidalPhase,
    CosinusoidalPhase,
    ExponentialDecay,
    SaturatingExponential,
    Logarithmic,
    SquareRoot,
    Reciprocal,
    AbsoluteValue,
    InverseQuadratic,
    Lorentzian,
    Sigmoid,
    DampedSinusoidal,
    DampedCosinusoidal,
    ExponentialGrowth,
    PowerLawDecay,
    ArctangentStep,
    HyperbolicSecant,
    SincDecay,
    AbsoluteSinusoidal,
    RectifiedLinear,
    Softplus,
    Gompertz,
    LinearFractional,
    SineSquared,
    TriangularBump,
)


# The source takes the first 30 expressions and cycles the palette for K=50.
symbolic_component_types_v30 = symbolic_component_palette_v1[:30]
symbolic_component_types_v50 = (
    symbolic_component_palette_v1 + symbolic_component_palette_v1[:8]
)
symbolic_component_types_v80 = tuple(
    symbolic_component_palette_v1[i % len(symbolic_component_palette_v1)]
    for i in range(80)
)
symbolic_component_types_v100 = tuple(
    symbolic_component_palette_v1[i % len(symbolic_component_palette_v1)]
    for i in range(100)
)
symbolic_noise_types_v1 = symbolic_noise_types_v0 + (
    NoiseExponential,
    NoiseSigmoid,
    NoisePeaked,
)


class SymbolicModelV30(AdditiveModel):
    """The source K=30 model: thirty expressions and eight noise alternatives."""

    model_types = symbolic_component_types_v30
    noise_types = symbolic_noise_types_v1

    log_likelihood = SymbolicModelV0.log_likelihood


class SymbolicModelV50(AdditiveModel):
    """The source K=50 model: fifty expressions and eight noise alternatives."""

    model_types = symbolic_component_types_v50
    noise_types = symbolic_noise_types_v1

    # Source models sum the selected noise-component likelihoods.
    log_likelihood = SymbolicModelV0.log_likelihood


class SymbolicModelV80(AdditiveModel):
    """The source K=80 model: the expression palette cycled to eighty components."""

    model_types = symbolic_component_types_v80
    noise_types = symbolic_noise_types_v1

    # Source models sum the selected noise-component likelihoods.
    log_likelihood = SymbolicModelV0.log_likelihood


class SymbolicModelV100(AdditiveModel):
    """The source K=100 model: the expression palette cycled to one hundred components."""

    model_types = symbolic_component_types_v100
    noise_types = symbolic_noise_types_v1

    # Source models sum the selected noise-component likelihoods.
    log_likelihood = SymbolicModelV0.log_likelihood
