"""Shared parameter conversion and pytree interface for simulator components."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import jax.tree_util as jtu
from jax import Array
from jax.typing import ArrayLike


class Compartment(ABC):
    theta_dim: int

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jtu.register_pytree_node_class(cls)

    @property
    def params(self):
        return self.__dict__

    @property
    def theta(self):
        return self.to_theta(**self.params)

    @classmethod
    @abstractmethod
    def to_theta(cls, *kwargs) -> Array:
        """Transforms the natural parameters to the optimization parameters which are
        assumed to be normally distributed.
        """
        pass

    @classmethod
    @abstractmethod
    def to_params(cls, theta: ArrayLike) -> Any:
        """Transforms the optimization parameters to the natural parameters."""
        pass

    @classmethod
    def from_theta(cls, theta: ArrayLike, **kwargs) -> "Compartment":
        """Creates a compartment from the optimization parameters."""
        args = cls.to_params(theta, **kwargs)
        return cls(*args)

    @classmethod
    def create_from_theta(cls, theta: ArrayLike, **kwargs) -> "Compartment":
        """Create a simulator component from optimization-space parameters.

        Calls ``from_theta`` so existing subclasses keep their construction logic.
        """
        return cls.from_theta(theta, **kwargs)

    def tree_flatten(self) -> tuple:
        """Flattens the compartment into a list of children and auxiliary data."""
        theta = self.theta
        return (theta,), (type(self),)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: list) -> "Compartment":
        """Reconstructs the compartment from the list of children and auxiliary data."""
        return cls.from_theta(children[0])


class Simulator(Protocol):
    """Interface the method uses on a simulator class (a model family).

    A simulator describes a family of models: signal components
    (``model_types``) and mutually exclusive noise components (``noise_types``)
    that a binary model mask switches on or off. PRISM relies only on these
    members; everything else belongs to the application.

    Training samples are dicts with ``model_mask``, ``theta``, ``x`` and
    ``mask_prior``, plus optional ``acq`` (a per-sample design passed to the
    encoder and likelihood) and ``target_score``. Batches of them are passed
    as keyword arguments to the model's ``loss_fn``.
    """

    theta_dim: int
    model_types: list
    noise_types: list
    # Samples contain only arrays and can be saved for file-based training.
    supports_offline_data: bool

    @classmethod
    def split_idx(cls) -> Any:
        """Parameter split points, one block per component."""

    @classmethod
    def theta_mask(cls, model_mask: ArrayLike) -> Array:
        """Boolean mask of the parameters that the model mask activates."""

    @classmethod
    def from_theta(cls, theta: ArrayLike, **kwargs: Any) -> Any:
        """Build a parameterized model from unconstrained parameters."""

    @classmethod
    def create_mask_prior(cls, **overrides: Any) -> Any:
        """The prior over model masks, with optional hyperparameter overrides."""

    def log_likelihood(self, design: Any, observed: ArrayLike) -> Array:
        """Log-likelihood of observations under this parameterized model."""

    @classmethod
    def training_generators(
        cls, simulator_cfg: Any, mask_prior: Any, posterior_score: bool
    ) -> list[Any]:
        """Functions ``(rng, mask_prior_hyperparameter=None, model_mask=None)``
        that each return one training sample."""

    @classmethod
    def evaluation_design(cls, simulator_cfg: Any) -> Any:
        """Design shared by all observations (e.g. a fixed grid), or ``None``.

        Simulators that return a design also implement ``evaluate(design, rng)``
        so training can report posterior predictive errors on it.
        """


SIMULATOR_MEMBERS = (
    "theta_dim",
    "model_types",
    "noise_types",
    "split_idx",
    "theta_mask",
    "from_theta",
    "create_mask_prior",
    "log_likelihood",
    "training_generators",
    "evaluation_design",
)


def missing_simulator_members(cls: type) -> list[str]:
    """Return the members of :class:`Simulator` that ``cls`` lacks."""
    return [name for name in SIMULATOR_MEMBERS if not hasattr(cls, name)]
