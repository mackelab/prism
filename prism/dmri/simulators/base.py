"""Signal, noise and shared-parameter interfaces for diffusion MRI."""

from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from prism.dmri.simulators.acquisition_scheme import acquisition_scheme
from prism.simulators.base import Compartment


class SharedParameterState(Compartment):
    share_with_compartments: dict[type, list[int]]

    def __init__(self, shared_parameters: ArrayLike):
        self.shared_parameters = jnp.atleast_1d(shared_parameters)

    def params_for_compartment(self, compartment_type: type) -> dict[str, ArrayLike]:
        """Return explicit shared values without modifying the compartment class."""
        indices = self.share_with_compartments.get(compartment_type)
        if indices is None:
            return {}
        return {
            name: self.shared_parameters[index]
            for name, index in zip(
                compartment_type.shared_parameter_names, indices, strict=True
            )
        }


class SignalCompartment(Compartment):
    # Shared values remain outside the compartment's optimization coordinates,
    # but must travel with an instance when JAX flattens its pytree.
    shared_parameter_names: tuple[str, ...] = ()

    def tree_flatten(self) -> tuple:
        shared_values = tuple(
            getattr(self, name) for name in self.shared_parameter_names
        )
        return (self.theta, *shared_values), (type(self),)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: list) -> "SignalCompartment":
        shared_values = dict(zip(cls.shared_parameter_names, children[1:], strict=True))
        return cls.from_theta(children[0], **shared_values)

    @classmethod
    def signal_fn(cls, acq: acquisition_scheme, *args, **kwargs) -> ArrayLike:
        """Computes the signal for the compartment."""
        return jnp.exp(cls.log_signal_fn(acq, *args, **kwargs))

    @classmethod
    @abstractmethod
    def log_signal_fn(cls, acq: acquisition_scheme, *args, **kwargs) -> ArrayLike:
        """Computes the log-signal for the compartment."""
        pass

    def signal(self, acq: acquisition_scheme, rng=None) -> ArrayLike:
        """Simulates the signal for the compartment."""
        return type(self).signal_fn(acq, rng=rng, **self.params)

    def log_signal(self, acq: acquisition_scheme, rng=None) -> ArrayLike:
        """Simulates the log-signal for the compartment."""
        return type(self).log_signal_fn(acq, rng=rng, **self.params)

    def fit(self, logS: ArrayLike, bvals: ArrayLike, bvecs: ArrayLike) -> Any:
        """Fits the compartment to the signal deterministically."""
        raise NotImplementedError("Fitting not implemented")

    def to_fod(self):
        """Converts the signal compartment to a fiber orientation distribution compartment."""
        raise NotImplementedError("Conversion to FOD not implemented")


class NoiseCompartment(Compartment):
    @abstractmethod
    def noise(self, signal, rng: jax.random.key) -> ArrayLike:
        """Simulates the noise for the compartment."""
        pass

    @abstractmethod
    def log_likelihood(self, signal_pred, signal_obs) -> Array:
        pass
