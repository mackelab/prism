# Adapted from mackelab/sbmi_transformer_symbolic (MIT), c8e118b.
from abc import abstractmethod
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import sympy as sp
from jax.typing import ArrayLike

from prism.simulators.base import Compartment
from prism.simulators.mask_prior import BetaBernoulliMaskPrior, TotalParamPenalizedPrior


class SymbolicCompartment(Compartment):
    """Base class for symbolic compartments."""

    @abstractmethod
    def evaluate(self, x) -> jax.Array: ...

    def __call__(self, x):
        return self.evaluate(x)


class NoiseCompartment(Compartment):
    """Base class for compartments that add noise to signals."""

    @abstractmethod
    def noise(self, x, rng: jax.random.PRNGKey, signal=None) -> jax.Array: ...

    @abstractmethod
    def log_likelihood(
        self, signal: jax.Array, signal_observed: jax.Array, x=None
    ) -> jax.Array: ...


class AdditiveModel(SymbolicCompartment):
    """Add selected functions and one selected noise alternative.

    Parameters use independent standard-normal coordinates in ``theta``.
    A mask contains one bit per function, followed by one per noise alternative.
    """

    mask_prior_cls = BetaBernoulliMaskPrior
    mask_prior_kwargs = {}
    model_types: list
    noise_types: list
    _split_dims: tuple[int, ...] = ()
    _split_indices: tuple[int, ...] = ()
    _num_models: int = 0
    _num_noises: int = 0

    supports_offline_data = True

    @classmethod
    def training_generators(cls, simulator_cfg, mask_prior, posterior_score):
        """Return the sample generator for the configured observation format."""
        from prism.symbolic.simulators.training import training_generators

        return training_generators(cls, simulator_cfg, mask_prior, posterior_score)

    @classmethod
    def evaluation_design(cls, simulator_cfg):
        """Return the grid shared by signal-block observations, else ``None``."""
        from prism.symbolic.simulators.training import evaluation_design

        return evaluation_design(simulator_cfg)

    def __init_subclass__(cls):
        assert hasattr(cls, "model_types"), "model_types not defined"
        assert hasattr(cls, "noise_types"), "noise_types not defined"

        cls._num_models = len(cls.model_types)
        cls._num_noises = len(cls.noise_types)

        cls.theta_dim = sum(m.theta_dim for m in cls.model_types) + sum(
            n.theta_dim for n in cls.noise_types
        )
        split_dims: list[int] = []
        split_dims.extend(m.theta_dim for m in cls.model_types)
        split_dims.extend(n.theta_dim for n in cls.noise_types)
        cls._split_dims = tuple(split_dims)
        if len(split_dims) > 1:
            cls._split_indices = tuple(np.cumsum(split_dims, dtype=int)[:-1].tolist())
        else:
            cls._split_indices = ()

        jtu.register_pytree_node_class(cls)

    def __init__(
        self,
        model_compartments: list,
        noise_compartments: list,
        model_mask: ArrayLike | None = None,
    ):
        self.model_compartments = tuple(model_compartments)
        self.noise_compartments = tuple(noise_compartments)
        self.model_mask = None if model_mask is None else jnp.asarray(model_mask)
        assert tuple(type(m) for m in noise_compartments) == tuple(self.noise_types), (
            "Wrong noise"
        )

    @classmethod
    def num_compartments(cls):
        return len(cls.model_types) + len(cls.noise_types)

    @classmethod
    def split_idx(cls):
        return np.asarray(cls._split_dims)

    @classmethod
    def split_theta(cls, theta):
        if cls._split_indices:
            thetas_split = jnp.split(theta, cls._split_indices)
        else:
            thetas_split = (theta,)
        return thetas_split

    @classmethod
    def theta_mask(
        cls,
        model_mask,
        model_idx: Sequence[int] | None = None,
        noise_idx: Sequence[int] | None = None,
    ):
        if model_idx is not None or noise_idx is not None:
            raise ValueError("Use a full component mask for additive models")
        if model_mask is None:
            return jnp.ones((cls.theta_dim,), dtype=jnp.bool_)
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        if mask.shape[-1] != cls.num_compartments():
            raise ValueError("model_mask length must match the number of components")
        return jnp.repeat(
            mask,
            jnp.asarray(cls._split_dims),
            axis=-1,
            total_repeat_length=cls.theta_dim,
        )

    @classmethod
    def create_mask_prior(cls, **overrides):
        """Create the configured component prior; one noise alternative is active."""
        kwargs = dict(cls.mask_prior_kwargs)
        kwargs.update(overrides)
        if issubclass(cls.mask_prior_cls, TotalParamPenalizedPrior):
            kwargs.setdefault(
                "num_model_parameters", [m.theta_dim for m in cls.model_types]
            )
        return cls.mask_prior_cls(len(cls.model_types), len(cls.noise_types), **kwargs)

    def tree_flatten(self):
        return (self.theta, self.model_mask), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls.from_theta(children[0], model_mask=children[1])

    @classmethod
    def to_theta(
        cls,
        model_compartments: list,
        noise_compartments: list,
        model_mask: ArrayLike | None = None,
    ):
        theta_model = (
            jnp.concatenate([m.theta for m in model_compartments])
            if model_compartments
            else jnp.array([])
        )
        theta_noise = (
            jnp.concatenate([m.theta for m in noise_compartments])
            if noise_compartments
            else jnp.array([])
        )
        return jnp.concatenate([theta_model, theta_noise])

    @classmethod
    def to_params(
        cls,
        theta: ArrayLike,
        model_mask: ArrayLike | None = None,
    ):
        thetas_split = cls.split_theta(theta)
        model_thetas = thetas_split[: cls._num_models]
        noise_thetas = thetas_split[cls._num_models :]

        model_compartments = [
            m.from_theta(t) for m, t in zip(cls.model_types, model_thetas, strict=True)
        ]
        noise_compartments = [
            n.from_theta(t) for n, t in zip(cls.noise_types, noise_thetas, strict=True)
        ]
        return (
            model_compartments,
            noise_compartments,
            model_mask,
        )

    def _active_mask(self):
        if self.model_mask is None:
            return jnp.ones((self._num_models,), dtype=jnp.bool_)
        mask = jnp.asarray(self.model_mask, dtype=jnp.bool_)
        if mask.shape[0] < self._num_models:
            raise ValueError(
                f"model_mask has length {mask.shape[0]}, expected at least {self._num_models}"
            )
        return mask[: self._num_models]

    def _noise_active_mask(self):
        if self._num_noises == 0:
            return jnp.ones((0,), dtype=jnp.bool_)
        if self.model_mask is None:
            return jnp.ones((self._num_noises,), dtype=jnp.bool_)
        mask = jnp.asarray(self.model_mask, dtype=jnp.bool_)
        expected = self._num_models + self._num_noises
        if mask.shape[0] < expected:
            raise ValueError(
                f"model_mask has length {mask.shape[0]}, expected at least {expected}"
            )
        return mask[self._num_models : expected]

    def evaluate(self, x, rng=None):
        mask = self._active_mask()
        values = []
        for active, comp in zip(mask, self.model_compartments, strict=True):
            val = comp.evaluate(x)
            values.append(jnp.where(active, val, jnp.zeros_like(val)))
        total = jnp.sum(jnp.stack(values, axis=0), axis=0) if values else jnp.array(0.0)

        if self.noise_compartments and rng is not None:
            rngs = jax.random.split(rng, len(self.noise_compartments))
            noise_mask = self._noise_active_mask()
            noise_vals = []
            for active, comp, r in zip(
                noise_mask, self.noise_compartments, rngs, strict=True
            ):
                noise = comp.noise(x, r, signal=total)
                noise_vals.append(jnp.where(active, noise, jnp.zeros_like(noise)))
            total = total + jnp.sum(jnp.stack(noise_vals, axis=0), axis=0)
        return total

    def to_simpy(
        self,
        *,
        substitute_params: bool = True,
        return_params: bool = False,
        round_digits: int | None = None,
    ):
        exprs = []
        params_info = []
        for active, comp in zip(
            self._active_mask(), self.model_compartments, strict=True
        ):
            if not active:
                continue
            expr = getattr(comp, "expression", None)
            if expr is None:
                expr = sp.sympify(str(comp))
            param_values = {}
            if hasattr(comp, "param_symbols") and hasattr(comp, "params_array"):
                values = np.asarray(comp.params_array, dtype=float).ravel()
                param_values = {
                    sym: round(float(val), round_digits)
                    if round_digits is not None
                    else float(val)
                    for sym, val in zip(comp.param_symbols, values, strict=True)
                }
            if substitute_params and param_values:
                expr = expr.subs(param_values)
            exprs.append(expr)
            if return_params:
                params_info.append({
                    "component": type(comp).__name__,
                    "params": param_values,
                })
        total = sum(exprs, sp.Integer(0))
        if return_params:
            return total, params_info
        return total

    def log_likelihood(self, x: jax.Array, signal_observed: jax.Array):
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
        active_count = jnp.sum(noise_mask)
        deterministic_log_like = jnp.where(
            jnp.all(signal_observed == signal), 0.0, -jnp.inf
        )
        return jnp.where(
            active_count == 0,
            deterministic_log_like,
            jnp.where(active_count == 1, jnp.sum(log_like), -jnp.inf),
        )
