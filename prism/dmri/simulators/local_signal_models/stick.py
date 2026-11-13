import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from prism.dmri.simulators.base import SignalCompartment, acquisition_scheme
from prism.dmri.simulators.spherical_distributions import SymmetricDirac
from prism.dmri.utils import (
    cart2sph,
    fit_diffusion_tensor_linearized,
    sph2cart,
    ssfp_signal_fn,
)


class Stick(SignalCompartment):
    r"""Zero-radius cylinder aligned with unit vector :math:`\boldsymbol{\mu}`.

    Signal:

    .. math::
        S(b, \mathbf{g}) = \exp\big(-b\,\lambda_{\parallel} (\mathbf{g}\cdot\boldsymbol{\mu})^2\big),

    capturing fully anisotropic diffusion along the fiber (Behrens et al., 2003).
    """

    theta_dim = 3
    min_lam = 0.0
    max_lam = 0.01

    def __init__(self, mu: ArrayLike, lam_par: float) -> None:
        """Initialize the Stick model with a lambda value and eigenvector."""
        self.mu = mu
        self.lam_par = lam_par

    @classmethod
    def log_signal_fn(
        cls,
        acq: acquisition_scheme,
        mu: ArrayLike,
        lam_par: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        bvals = acq.bvals
        bvecs = acq.bvecs
        mu_cart = sph2cart(mu)
        logS = -bvals * lam_par * (jnp.sum(bvecs * mu_cart, axis=-1)) ** 2
        return logS

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: float) -> ArrayLike:
        """Convert the parameters to the parameter space theta."""
        lam_par = (lam_par - cls.min_lam) / (cls.max_lam - cls.min_lam)
        mu0_normalized = 1 - jnp.cos(
            mu[..., 0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[..., 1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([lam_par, mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value and eigenvector."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam_par = theta[..., 0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        mu1 = jnp.arccos(1 - theta[..., 1])  # Ensures output is in upper hemisphere
        mu2 = theta[..., 2] * 2 * jnp.pi - jnp.pi

        mu = jnp.array([mu1, mu2])

        return mu, lam_par

    def to_fod(self):
        mu_cart = sph2cart(self.mu)
        return SymmetricDirac(mu_cart)

    def fit(self, logS: ArrayLike, acq: acquisition_scheme) -> tuple:
        """Fit the Stick model to the log signal, b-values, and b-vectors."""
        bvals = acq.bvals
        bvecs = acq.bvecs
        D = fit_diffusion_tensor_linearized(logS, bvals, bvecs)
        eigvals, eigvecs = jnp.linalg.eigh(D)
        idx = jnp.argmax(eigvals)
        lam_par = eigvals[idx]
        eigvec = eigvecs[:, idx]
        mu = cart2sph(eigvec)
        return mu, lam_par


class StaticStick(Stick):
    """
    The StaticStick model is a Stick with a fixed lambda value.
    The shared diffusivity values are bound to each instance,
    while the mu parameter remains learnable.
    """

    theta_dim: int = 2  # Only mu parameters are learnable
    lam_par: float = None  # Class attribute for the fixed lambda value

    shared_parameter_names = ("lam_par",)

    def __init__(self, mu: ArrayLike, lam_par: ArrayLike | None = None) -> None:
        """Bind diffusivity to this instance; class values are legacy defaults."""
        self.mu = mu
        self.lam_par = type(self).lam_par if lam_par is None else lam_par
        if self.lam_par is None:
            raise ValueError("lam_par must be provided for StaticStick")

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: ArrayLike | None = None) -> ArrayLike:
        """Convert only the mu parameter to the parameter space theta.
        The shared diffusivity is not included in the local theta.
        """
        # Only convert mu parameters to theta space
        mu0_normalized = 1 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike, lam_par: ArrayLike | None = None) -> tuple:
        """Convert the parameter space theta to the mu value.
        The shared diffusivity is not included in the local theta.
        """
        theta = jax.scipy.stats.norm.cdf(theta)
        mu1 = jnp.arccos(1 - theta[0])  # Ensures output is in upper hemisphere
        mu2 = theta[1] * 2 * jnp.pi - jnp.pi
        mu = jnp.array([mu1, mu2])
        return (mu, lam_par)

    @classmethod
    def log_signal_fn(
        cls,
        acq: acquisition_scheme,
        mu: ArrayLike,
        rng=None,
        *,
        lam_par=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors.

        Parameters
        ----------
        acq : acquisition_scheme
            The acquisition scheme containing b-values and b-vectors
        mu : ArrayLike
            The orientation of the stick in spherical coordinates
        rng : jax.random.key, optional
            Random number generator key, by default None

        Returns
        -------
        ArrayLike
            The log signal
        """
        bvals = acq.bvals
        bvecs = acq.bvecs
        mu_cart = sph2cart(mu)
        lam_par = cls.lam_par if lam_par is None else lam_par
        logS = -bvals * lam_par * (jnp.sum(bvecs * mu_cart, axis=-1)) ** 2
        return logS


class MultiShellStick(Stick):
    """
    The MultiShellStick model is a Stick with a fixed lambda value.
    The lam_par parameter is shared from a global parameter state as a class attribute,
    while the mu parameter remains learnable.
    """

    theta_dim: int = 4
    lam_par: float
    lam_par_std: float
    lam_par_std_min: float = 0.0
    lam_par_std_max: float = 0.005

    def __init__(self, mu: ArrayLike, lam_par: float, lam_par_std: float) -> None:
        """Initialize the MultiShellStick model with a mu value, lam_par, and lam_par_std."""
        self.mu = mu
        self.lam_par = lam_par
        self.lam_par_std = lam_par_std

    @classmethod
    def log_signal_fn(
        cls,
        acq: acquisition_scheme,
        mu: ArrayLike,
        lam_par: float,
        lam_par_std: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors.

        Parameters
        ----------
        acq : acquisition_scheme
            The acquisition scheme containing b-values and b-vectors
        """
        bvals = acq.bvals
        bvecs = acq.bvecs

        return multi_shell_stick_log_signal_fn(bvals, bvecs, mu, lam_par, lam_par_std)

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: float, lam_par_std: float) -> ArrayLike:
        """Convert the parameters to the parameter space theta."""
        lam_par = (lam_par - cls.min_lam) / (cls.max_lam - cls.min_lam)
        lam_par_std = (lam_par_std - cls.lam_par_std_min) / (
            cls.lam_par_std_max - cls.lam_par_std_min
        )
        mu0_normalized = 1.0 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([lam_par, lam_par_std, mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the lambda value and eigenvector."""
        theta = jax.scipy.stats.norm.cdf(theta)
        lam_par = theta[0] * (cls.max_lam - cls.min_lam) + cls.min_lam
        lam_par_std = (
            theta[1] * (cls.lam_par_std_max - cls.lam_par_std_min) + cls.lam_par_std_min
        )

        mu1 = arccos_stable(1.0 - theta[2])  # Ensures output is in upper hemisphere
        mu2 = theta[3] * 2 * jnp.pi - jnp.pi

        mu = jnp.array([mu1, mu2])

        return mu, lam_par, lam_par_std


class MultiShellStaticStick(MultiShellStick):
    """
    The MultiShellStaticStick model is a Stick with a fixed lambda value.
    The shared diffusivity values are bound to each instance,
    while the mu parameter remains learnable.
    """

    theta_dim: int = 2
    lam_par: float = None
    lam_par_std: float = None

    shared_parameter_names = ("lam_par", "lam_par_std")

    def __init__(
        self,
        mu: ArrayLike,
        lam_par: ArrayLike | None = None,
        lam_par_std: ArrayLike | None = None,
    ) -> None:
        """Bind diffusivity to this instance; class values are legacy defaults."""
        self.mu = mu
        self.lam_par = type(self).lam_par if lam_par is None else lam_par
        if self.lam_par is None:
            raise ValueError("lam_par must be provided for MultiShellStaticStick")
        self.lam_par_std = (
            type(self).lam_par_std if lam_par_std is None else lam_par_std
        )
        if self.lam_par_std is None:
            raise ValueError("lam_par_std must be provided for MultiShellStaticStick")

    @classmethod
    def to_theta(
        cls,
        mu: ArrayLike,
        lam_par: ArrayLike | None = None,
        lam_par_std: ArrayLike | None = None,
    ) -> ArrayLike:
        """Convert only the mu parameter to the parameter space theta.
        The shared diffusivity is not included in the local theta.
        """
        # Only convert mu parameters to theta space
        mu0_normalized = 1 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(
        cls,
        theta: ArrayLike,
        lam_par: ArrayLike | None = None,
        lam_par_std: ArrayLike | None = None,
    ) -> tuple:
        """Convert the parameter space theta to the mu value.
        The shared diffusivity is not included in the local theta.
        """
        theta = jax.scipy.stats.norm.cdf(theta)
        mu1 = arccos_stable(1 - theta[0])  # Ensures output is in upper hemisphere
        mu2 = theta[1] * 2 * jnp.pi - jnp.pi
        mu = jnp.array([mu1, mu2])
        return (mu, lam_par, lam_par_std)

    @classmethod
    def log_signal_fn(
        cls,
        acq: acquisition_scheme,
        mu: ArrayLike,
        rng=None,
        *,
        lam_par=None,
        lam_par_std=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors.

        Parameters
        ----------
        acq : acquisition_scheme
            The acquisition scheme containing b-values and b-vectors
        mu : ArrayLike
            The orientation of the stick in spherical coordinates
        rng : jax.random.key, optional
            Random number generator key, by default None

        Returns
        -------
        ArrayLike
            The log signal
        """
        lam_par = cls.lam_par if lam_par is None else lam_par
        lam_par_std = cls.lam_par_std if lam_par_std is None else lam_par_std
        return MultiShellStick.log_signal_fn(acq, mu, lam_par, lam_par_std)


class SSFPStick(Stick):
    """
    The SSFPStick model is a Stick with a fixed lambda value.
    The lam_par parameter is shared from a global parameter state as a class attribute,
    while the mu parameter remains learnable.
    """

    theta_dim: int = 3
    lam_par: float
    lam_min: float = 0.0
    lam_max: float = 0.01

    def __init__(self, mu: ArrayLike, lam_par: float) -> None:
        self.mu = mu
        self.lam_par = lam_par

    @classmethod
    def signal_fn(
        cls,
        acq: acquisition_scheme,
        mu: ArrayLike,
        lam_par: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the signal for given b-values and b-vectors."""

        mu_cart = sph2cart(mu)
        adc_aniso = lam_par * (jnp.sum(acq.bvecs * mu_cart, axis=-1)) ** 2

        qvals = acq.qvals
        E1 = acq.E1
        E2 = acq.E2
        sa = acq.sa
        ca = acq.ca
        TR = acq.TRs
        grad_diff_dur = acq.diffGradDur

        signal = ssfp_signal_fn(adc_aniso, qvals, E1, E2, sa, ca, TR, grad_diff_dur)
        return signal

    @classmethod
    def log_signal_fn(
        cls,
        acq: acquisition_scheme,
        mu: ArrayLike,
        lam_par: float,
        rng=None,
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        signal = cls.signal_fn(acq, mu, lam_par, rng)
        return jnp.log(signal)


class SSFPStaticStick(SSFPStick):
    """
    The SSFPStaticStick model is a SSFPStick with a fixed lambda value.
    The shared diffusivity values are bound to each instance.
    """

    theta_dim: int = 2  # No learnable parameters, lam_par is fixed
    lam_par: float = None

    shared_parameter_names = ("lam_par",)

    def __init__(self, mu: ArrayLike, lam_par: ArrayLike | None = None) -> None:
        """Bind diffusivity to this instance; class values are legacy defaults."""
        self.mu = mu
        self.lam_par = type(self).lam_par if lam_par is None else lam_par
        if self.lam_par is None:
            raise ValueError("lam_par must be provided for SSFPStaticStick")

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: ArrayLike | None = None) -> ArrayLike:
        """Convert only the mu parameter to the parameter space theta.
        The shared diffusivity is not included in the local theta.
        """
        # Only convert mu parameters to theta space
        mu0_normalized = 1 - jnp.cos(
            mu[0]
        )  # Ensures uniform distribution on upper hemisphere
        mu1_normalized = (mu[1] + jnp.pi) / (2 * jnp.pi)
        theta = jnp.array([mu0_normalized, mu1_normalized])
        theta = jax.scipy.stats.norm.ppf(theta)
        return theta

    @classmethod
    def to_params(cls, theta: ArrayLike, lam_par: ArrayLike | None = None) -> tuple:
        """Convert the parameter space theta to the mu value.
        The shared diffusivity is not included in the local theta.
        """
        theta = jax.scipy.stats.norm.cdf(theta)
        mu1 = arccos_stable(1 - theta[0])  # Ensures output is in upper hemisphere
        mu2 = theta[1] * 2 * jnp.pi - jnp.pi
        mu = jnp.array([mu1, mu2])
        return (mu, lam_par)

    @classmethod
    def signal_fn(
        cls, acq: acquisition_scheme, mu: ArrayLike, rng=None, *, lam_par=None
    ) -> ArrayLike:
        """Compute the signal for given b-values and b-vectors."""
        lam_par = cls.lam_par if lam_par is None else lam_par
        return SSFPStick.signal_fn(acq, mu, lam_par, rng)

    @classmethod
    def log_signal_fn(
        cls, acq: acquisition_scheme, mu: ArrayLike, rng=None, *, lam_par=None
    ) -> ArrayLike:
        """Compute the log signal for given b-values and b-vectors."""
        lam_par = cls.lam_par if lam_par is None else lam_par
        return SSFPStick.log_signal_fn(acq, mu, lam_par, rng)


def multi_shell_stick_log_signal_fn(
    bvals: ArrayLike,
    bvecs: ArrayLike,
    mu: ArrayLike,
    lam_par: float,
    lam_par_std: float,
) -> ArrayLike:
    """Implementation with float32 numerical stability."""
    mu_cart = sph2cart(mu)
    nugget = jnp.finfo(bvals.dtype).eps

    # Add small constant to prevent division by zero when mu -> 0
    scaling = (lam_par / (lam_par_std + nugget)) ** 2

    # Add small constant to dot product to prevent gradient explosion
    dot_product = (lam_par_std * jnp.sum(bvecs * mu_cart, axis=-1)) ** 2

    # Stable log computation using log1p
    logS = -jnp.log1p(bvals * dot_product / (lam_par + nugget))

    return scaling * logS


@jax.custom_vjp
def arccos_stable(x: ArrayLike) -> ArrayLike:
    """Stable arccos implementation with custom gradient.

    Forward pass uses standard arccos, but the gradient is stabilized
    to prevent NaN or inf when x approaches ±1.
    """
    return jnp.arccos(x)


def arccos_stable_fwd(x):
    return arccos_stable(x), x


def arccos_stable_bwd(x, g):
    # Standard gradient for arccos is -1/sqrt(1-x²)
    # We stabilize by adding a small epsilon to avoid division by zero
    x = jnp.asarray(x)
    eps = 1.0 - jnp.finfo(x.dtype).eps
    a = jnp.minimum(x, eps)
    # a = x
    b = a**2
    c = 1.0 - b
    d = jax.lax.rsqrt(c)
    e = -d
    return (g * e,)


arccos_stable.defvjp(arccos_stable_fwd, arccos_stable_bwd)
