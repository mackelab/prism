import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from prism.dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    ssfp_acquisition_scheme,
)
from prism.dmri.simulators.base import SignalCompartment
from prism.dmri.simulators.spherical_distributions import Uniform
from prism.dmri.utils import ssfp_signal_fn


class Ball(SignalCompartment):
    r"""Isotropic Gaussian diffusion (free water).

    Signal follows the Stejskal–Tanner form

    .. math::
        S(b) = \exp(-b\,\lambda),

    with scalar diffusivity :math:`\lambda \in [0, 10^{-2}]\,\mathrm{mm}^2/\mathrm{s}`
    (Stejskal & Tanner, 1965).
    """

    theta_dim: int = 1
    lam_min: float = 0.0
    lam_max: float = 0.01  # 1e-3 mm^2/s is the diffusivity of free water

    def __init__(self, lam: float) -> None:
        """Initialize the Ball model with a lambda value."""
        self.lam = lam

    @classmethod
    def log_signal_fn(cls, acq: acquisition_scheme, lam: float, rng=None) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        logS = -acq.bvals * lam
        return logS

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam = theta[..., 0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam,)

    @classmethod
    def to_theta(cls, lam: ArrayLike) -> Array:
        """Convert the lambda value to the parameter space theta."""
        lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jnp.array([lam])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: Array) -> tuple:
        """Fit the Ball model to the log signal and b-values."""
        lam = -logS / bvals
        return (jnp.mean(lam),)

    def to_fod(self):
        return Uniform()


class StaticBall(Ball):
    """
    The StaticBall model is a Ball with a fixed lambda value.
    The shared diffusivity values are bound to each instance.
    """

    theta_dim: int = 0  # No learnable parameters, lam is fixed
    lam: float = None  # Class attribute for the fixed lambda value

    shared_parameter_names = ("lam",)

    def __init__(self, lam: ArrayLike | None = None) -> None:
        """Bind diffusivity to this instance; class values are legacy defaults."""
        self.lam = type(self).lam if lam is None else lam
        if self.lam is None:
            raise ValueError("lam must be provided for StaticBall")

    @classmethod
    def to_theta(cls, lam: ArrayLike | None = None) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        return jnp.array([])

    @classmethod
    def to_params(cls, theta: ArrayLike, lam: ArrayLike | None = None) -> tuple:
        """Convert the parameter space theta to parameters.
        The shared diffusivity is supplied separately from theta.
        """
        return (lam,)

    @classmethod
    def log_signal_fn(cls, acq: acquisition_scheme, rng=None, *, lam=None) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        acq: acquisition_scheme
        lam = cls.lam if lam is None else lam
        logS = -acq.bvals * lam
        return logS


class MultiShellBall(Ball):
    """Ball compartment parameterized by diffusivity mean and standard deviation."""

    theta_dim: int = 2  # No learnable parameters, lam is fixed
    lam: float
    lam_std: float
    lam_std_min: float = 0.0000001
    lam_std_max: float = 0.01

    def __init__(self, lam: float, lam_std: float) -> None:
        """Store the diffusivity parameters on this instance."""
        self.lam = lam
        self.lam_std = lam_std

    @classmethod
    def log_signal_fn(
        cls, acq: acquisition_scheme, lam: float, lam_std: float, rng=None
    ) -> ArrayLike:
        """Computes the log-signal for the ball compartment with uncertainty."""
        acq: acquisition_scheme
        return multi_shell_ball_log_signal_fn(acq.bvals, lam, lam_std)

    @classmethod
    def to_theta(cls, lam: float, lam_std: float) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        u_lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        u_lam_std = (lam_std - cls.lam_std_min) / (cls.lam_std_max - cls.lam_std_min)
        theta_u = jnp.array([u_lam, u_lam_std])
        theta = jax.scipy.stats.norm.ppf(theta_u)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to parameters.
        Since there are no learnable parameters, return an empty tuple.
        """
        u = jax.scipy.stats.norm.cdf(theta)
        lam = u[0] * (cls.lam_max - cls.lam_min) + cls.lam_min
        lam_std = u[1] * (cls.lam_std_max - cls.lam_std_min) + cls.lam_std_min
        return (lam, lam_std)


class MultiShellStaticBall(MultiShellBall):
    """
    The MultiShellBall model is a Ball with multiple lambda values.
    The shared diffusivity values are bound to each instance.
    """

    theta_dim: int = 0  # No learnable parameters, lam is fixed
    lam: float = None
    lam_std: float = None

    shared_parameter_names = ("lam", "lam_std")

    def __init__(
        self, lam: ArrayLike | None = None, lam_std: ArrayLike | None = None
    ) -> None:
        """Bind diffusivity to this instance; class values are legacy defaults."""
        self.lam = type(self).lam if lam is None else lam
        if self.lam is None:
            raise ValueError("lam must be provided for MultiShellStaticBall")
        self.lam_std = type(self).lam_std if lam_std is None else lam_std
        if self.lam_std is None:
            raise ValueError("lam_std must be provided for MultiShellStaticBall")

    @classmethod
    def to_theta(
        cls, lam: ArrayLike | None = None, lam_std: ArrayLike | None = None
    ) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        return jnp.array([])

    @classmethod
    def to_params(
        cls,
        theta: ArrayLike,
        lam: ArrayLike | None = None,
        lam_std: ArrayLike | None = None,
    ) -> tuple:
        """Convert the parameter space theta to parameters.
        The shared diffusivity is supplied separately from theta.
        """
        return (
            lam,
            lam_std,
        )

    @classmethod
    def log_signal_fn(
        cls, acq: acquisition_scheme, rng=None, *, lam=None, lam_std=None
    ) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        acq: acquisition_scheme
        lam = cls.lam if lam is None else lam
        lam_std = cls.lam_std if lam_std is None else lam_std
        return multi_shell_ball_log_signal_fn(acq.bvals, lam, lam_std)


class SSFPBall(SignalCompartment):
    """
    The SSFPBall model is a Ball with a fixed lambda value.
    The lam parameter is shared from a global parameter state as a class attribute.
    """

    theta_dim: int = 1
    lam_min: float = 0.0
    lam_max: float = 0.01

    def __init__(self, lam: float) -> None:
        self.lam = lam

    @classmethod
    def signal_fn(cls, acq: ssfp_acquisition_scheme, lam: float, rng=None) -> ArrayLike:
        """Computes the signal for the ball compartment."""
        # Relaxation terms
        # E1 = jnp.exp(-acq.TRs / (1e-3 * acq.T1))  # T1
        # E2 = jnp.exp(-acq.TRs / (1e-3 * acq.T2))  # T2

        # # Flipping terms
        # sa = jnp.sin(acq.flipAngles * acq.B1 * jnp.pi / 180.0)  # sin(flip * B1)
        # ca = jnp.cos(acq.flipAngles * acq.B1 * jnp.pi / 180.0)  # cos(flip * B1)
        qvals = acq.qvals
        E1 = acq.E1
        E2 = acq.E2
        sa = acq.sa
        ca = acq.ca
        TR = acq.TRs
        grad_diff_dur = acq.diffGradDur
        lam = lam

        signal = ssfp_signal_fn(lam, qvals, E1, E2, sa, ca, TR, grad_diff_dur)
        return signal

    @classmethod
    def log_signal_fn(
        cls, acq: ssfp_acquisition_scheme, lam: float, rng=None
    ) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        signal = cls.signal_fn(acq, lam, rng)
        return jnp.log(signal)

    @classmethod
    def to_theta(cls, lam: float) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        u_lam = (lam - cls.lam_min) / (cls.lam_max - cls.lam_min)
        theta = jax.scipy.stats.norm.ppf(u_lam)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to parameters.
        Since there are no learnable parameters, return an empty tuple.
        """
        u_lam = jax.scipy.stats.norm.cdf(theta)
        lam = u_lam * (cls.lam_max - cls.lam_min) + cls.lam_min
        return (lam,)


class SSFPStaticBall(SSFPBall):
    """
    The SSFPStaticBall model is a SSFPBall with a fixed lambda value.
    The shared diffusivity values are bound to each instance.
    """

    theta_dim: int = 0  # No learnable parameters, lam is fixed
    lam: float = None

    shared_parameter_names = ("lam",)

    def __init__(self, lam: ArrayLike | None = None) -> None:
        """Bind diffusivity to this instance; class values are legacy defaults."""
        self.lam = type(self).lam if lam is None else lam
        if self.lam is None:
            raise ValueError("lam must be provided for SSFPStaticBall")

    @classmethod
    def to_theta(cls, lam: ArrayLike | None = None) -> ArrayLike:
        """Convert to the parameter space theta.
        Since there are no learnable parameters, return an empty array.
        """
        return jnp.array([])

    @classmethod
    def to_params(cls, theta: ArrayLike, lam: ArrayLike | None = None) -> tuple:
        """Convert the parameter space theta to parameters.
        The shared diffusivity is supplied separately from theta.
        """
        return (lam,)

    @classmethod
    def signal_fn(
        cls, acq: ssfp_acquisition_scheme, rng=None, *, lam=None
    ) -> ArrayLike:
        """Computes the signal for the ball compartment."""
        lam = cls.lam if lam is None else lam
        return SSFPBall.signal_fn(acq, lam, rng)

    @classmethod
    def log_signal_fn(
        cls, acq: ssfp_acquisition_scheme, rng=None, *, lam=None
    ) -> ArrayLike:
        """Computes the log-signal for the ball compartment."""
        lam = cls.lam if lam is None else lam
        return SSFPBall.log_signal_fn(acq, lam, rng)


def multi_shell_ball_log_signal_fn(
    bvals: ArrayLike,
    lam: float,
    lam_std: float,
) -> ArrayLike:
    """Implementation with float32 numerical stability for MultiShellBall.

    Parameters
    ----------
    bvals : ArrayLike
        The b-values of the acquisition scheme
    lam : float
        The mean diffusivity
    lam_std : float
        The standard deviation of the diffusivity

    Returns
    -------
    ArrayLike
        The log signal
    """
    nugget = jnp.finfo(bvals.dtype).eps

    # Add small constant to prevent division by zero
    scaling = (lam / (lam_std + nugget)) ** 2

    # Use stable log computation
    logS = jnp.log(lam + nugget) - jnp.log(lam + bvals * lam_std**2 + nugget)

    return scaling * logS
