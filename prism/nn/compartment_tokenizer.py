"""Tokenizer for model families of switchable components without fractions.

It uses only the simulator interface (component lists, parameter splits and
masks). Figure 4b variants in :mod:`prism.symbolic.paper.tokenizer` reuse these
operations while preserving their original RNG stream and configuration tokens.
Adapted from mackelab/sbmi_transformer_symbolic (MIT).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike

from prism.nn.tokenizer import Tokenizer


class CompartmentTokenizerBase(Tokenizer, ABC):
    """Parameter initialization and encode/decode utilities for additive models."""

    num_components: int

    @staticmethod
    def _compute_linear_kwargs(
        dtype: DTypeLike | None,
        param_dtype: DTypeLike | None,
        precision: PrecisionLike | None,
        preferred_element_type: DTypeLike | None,
    ) -> dict[str, Any]:
        linear_kwargs: dict[str, Any] = {}
        for name, value in (
            ("dtype", dtype),
            ("param_dtype", param_dtype),
            ("precision", precision),
            ("preferred_element_type", preferred_element_type),
        ):
            if value is not None:
                linear_kwargs[name] = value
        return linear_kwargs

    def __init__(
        self,
        simulator: Any,
        *,
        token_dim: int = 64,
        params_dims: Sequence[int],
        embed_idx_count: int,
        theta_encode_nets: nnx.List[nnx.Module | None] | None = None,
        theta_decode_nets: nnx.List[nnx.Module | None] | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        skip_param_indices: Sequence[int] = (),
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.simulator = nnx.static(simulator)
        self.token_dim = token_dim
        self.params_dims = tuple(int(d) for d in params_dims)
        self.param_indices_with_params = tuple(
            idx
            for idx, dim in enumerate(self.params_dims)
            if dim > 0 and idx not in skip_param_indices
        )

        self._linear_kwargs = self._compute_linear_kwargs(
            dtype, param_dtype, precision, preferred_element_type
        )

        if rngs is None:
            raise ValueError("rngs must be provided for Tokenizer initialization.")

        self.embed_idx = nnx.Embed(
            rngs=rngs,
            num_embeddings=embed_idx_count,
            features=token_dim,
            embedding_init=nnx.initializers.orthogonal(),
        )

        if theta_encode_nets is None:
            theta_encode_nets = nnx.List([
                nnx.Linear(dim, token_dim, rngs=rngs, **self._linear_kwargs)
                if dim > 0
                else None
                for dim in self.params_dims
            ])
        if theta_decode_nets is None:
            theta_decode_nets = nnx.List([
                nnx.Linear(
                    token_dim,
                    dim,
                    rngs=rngs,
                    kernel_init=nnx.initializers.zeros,
                    **self._linear_kwargs,
                )
                if dim > 0
                else None
                for dim in self.params_dims
            ])
        self.theta_encode_nets = theta_encode_nets
        self.theta_decode_nets = theta_decode_nets

    def __call__(self, *args: Any, **kwds: Any) -> Array:
        return self.encode(*args, **kwds)

    @abstractmethod
    def embed_cfgs(self, model_mask: ArrayLike, **kwargs: Any) -> Array:
        """Embed the component configuration."""

    @abstractmethod
    def embed_theta(
        self,
        theta: Array,
        tokens_cfg: Array,
        model_mask: ArrayLike | None = None,
        **kwargs: Any,
    ) -> Array:
        """Embed parameters given the component configuration."""

    def embed_model_mask(
        self, tokens_cfg: Array, model_mask: ArrayLike, **kwargs: Any
    ) -> Array:
        """Use configuration tokens directly unless a variant adds mask features."""
        return tokens_cfg

    def encode(
        self,
        theta: ArrayLike | None = None,
        model_mask: ArrayLike | None = None,
        tokens_cfg: Array | None = None,
        **kwargs: Any,
    ) -> Array:
        if model_mask is None:
            model_mask = jnp.ones((self.num_components,), dtype=jnp.bool_)
        if tokens_cfg is None:
            tokens_cfg = self.embed_cfgs(model_mask, **kwargs)
        if theta is None:
            return tokens_cfg
        return self.embed_theta(theta, tokens_cfg, model_mask=model_mask, **kwargs)

    def decode(
        self, tokens: ArrayLike, model_mask: ArrayLike | None = None, **kwargs: Any
    ) -> Array:
        del kwargs
        return self.decode_theta(tokens, model_mask=model_mask)

    def _split_theta(self, theta: ArrayLike) -> tuple[Array, ...]:
        dims = np.asarray(self.params_dims, dtype=np.int32)
        split_dims = np.cumsum(dims)[:-1]
        return tuple(jnp.split(jnp.asarray(theta), split_dims, axis=-1))

    def _apply_theta_mask(self, out: Array, model_mask: ArrayLike | None) -> Array:
        if model_mask is None:
            return out
        mask_fn = self.simulator.theta_mask
        for _ in range(jnp.asarray(model_mask).ndim - 1):
            mask_fn = jax.vmap(mask_fn)
        theta_mask = mask_fn(model_mask)
        return jnp.where(theta_mask, out, 0.0)


class CompartmentTokenizer(CompartmentTokenizerBase):
    """Component tokens without mixture-fraction parameters.

    Configuration token j+1 holds component j's mask. The causal decoder
    predicts bit j from token j, so it sees only earlier decisions. Figure 4b
    uses the separate ``SymbolicPaperTokenizerPP`` variant with learned mask features.
    """

    def __init__(
        self,
        simulator: Any,
        *,
        token_dim: int = 64,
        theta_encode_nets: nnx.List[nnx.Module | None] | None = None,
        theta_decode_nets: nnx.List[nnx.Module | None] | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_models = len(simulator.model_types)
        self.num_noises = len(simulator.noise_types)
        self.num_components = self.num_models + self.num_noises
        self.params_dims = tuple(simulator.split_idx())
        self._linear_kwargs = self._compute_linear_kwargs(
            dtype, param_dtype, precision, preferred_element_type
        )
        super().__init__(
            simulator,
            token_dim=token_dim,
            params_dims=self.params_dims,
            embed_idx_count=self.num_components,
            theta_encode_nets=theta_encode_nets,
            theta_decode_nets=theta_decode_nets,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )
        self.components_with_params = self.param_indices_with_params
        # Learnable padding token
        self.pad_token = nnx.Param(
            nnx.initializers.zeros(self._pad_token_key(rngs), (1, self.token_dim))
        )

    @staticmethod
    def _pad_token_key(rngs: nnx.Rngs) -> Array:
        return rngs.params()

    @staticmethod
    def _validate_component_indices(
        model_idx: Sequence[int] | None, noise_idx: Sequence[int] | None
    ) -> None:
        if model_idx is not None or noise_idx is not None:
            raise ValueError("Use a full component mask for additive models")

    def theta_token_mask(
        self,
        model_mask: ArrayLike,
        model_idx: Sequence[int] | None = None,
        noise_idx: Sequence[int] | None = None,
    ) -> Array:
        self._validate_component_indices(model_idx, noise_idx)
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        if self.components_with_params:
            return mask[..., self.components_with_params]
        return jnp.zeros(mask.shape[:-1] + (0,), dtype=jnp.bool_)

    def embed_cfgs(
        self,
        model_mask: ArrayLike,
        alpha_prior=None,
        model_idx=None,
        noise_idx=None,
    ) -> Array:
        del alpha_prior
        self._validate_component_indices(model_idx, noise_idx)
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        *batch_dims, mask_dim = mask.shape
        if mask_dim != self.num_components:
            raise ValueError(
                f"model_mask length {mask_dim} does not match number of components "
                f"{self.num_components}"
            )
        pad_value = self.pad_token.value
        pad_token = jnp.broadcast_to(pad_value, tuple(batch_dims) + (1, self.token_dim))
        idx = jnp.arange(self.num_components, dtype=jnp.int32)
        idx_tokens = self.embed_idx(idx)
        for _ in batch_dims:
            idx_tokens = idx_tokens[None, ...]
        idx_tokens = idx_tokens * mask[..., None]
        return jnp.concatenate([pad_token, idx_tokens], axis=-2)

    def embed_theta(
        self,
        theta: Array,
        tokens_cfg: Array,
        model_mask: ArrayLike | None = None,
        **kwargs: Any,
    ) -> Array:
        del kwargs
        theta = jnp.asarray(theta)
        if model_mask is None:
            model_mask = jnp.ones((self.num_components,), dtype=jnp.bool_)
        model_mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        theta_split = self._split_theta(theta)

        component_cfg_tokens = tokens_cfg[..., 1:, :]
        tokens: list[Array] = []
        for comp_idx in self.components_with_params:
            comp_theta = theta_split[comp_idx]
            net = self.theta_encode_nets[comp_idx]
            if net is None:
                continue
            value_token = net(comp_theta)[..., None, :]
            cfg_token = component_cfg_tokens[..., comp_idx, :]
            cfg_token = cfg_token[..., None, :]
            mask = model_mask[..., comp_idx]
            mask_expanded = mask[..., None, None]
            tokens.append((value_token + cfg_token) * mask_expanded)

        return (
            jnp.concatenate(tokens, axis=-2)
            if tokens
            else jnp.zeros(theta.shape[:-1] + (0, self.token_dim))
        )

    def decode_theta(
        self,
        tokens: Array,
        model_mask: ArrayLike | None = None,
        **kwargs: Any,
    ) -> Array:
        del kwargs
        tokens_split = jnp.split(tokens, tokens.shape[-2], axis=-2)
        decode_nets: list[nnx.Module] = [
            self.theta_decode_nets[i]
            for i in self.components_with_params  # type: ignore
        ]
        assert len(decode_nets) == len(tokens_split), (
            f"Number of networks ({len(decode_nets)}) does not match number of tokens "
            f"({len(tokens_split)})"
        )
        decoded_parts = jax.tree_util.tree_map(
            lambda token, net: net(token), tokens_split, decode_nets
        )
        out = jnp.concatenate(decoded_parts, axis=-1)
        out = jnp.squeeze(out, axis=-2)

        if model_mask is not None:
            out = self._apply_theta_mask(out, model_mask)
        return out
