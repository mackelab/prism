"""Figure 4b tokenization variants over the shared additive-model tokenizer.

The paper variant intentionally consumes ``rngs.next()`` for its padding token
and ignores component-subset arguments. ``SymbolicPaperTokenizerPP`` additionally
uses neighboring component embeddings, learned start/end tokens, and shifted
mask embeddings. These deltas preserve the frozen Figure 4b parameter tree and
initialization order; the generic tokenizer lives in ``prism.nn.compartment_tokenizer``.
"""

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from flax import nnx
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike

from prism.nn.compartment_tokenizer import (
    CompartmentTokenizer as GenericSymbolicTokenizer,
)
from prism.nn.compartment_tokenizer import CompartmentTokenizerBase as Tokenizer

__all__ = ["Tokenizer", "SymbolicPaperTokenizer", "SymbolicPaperTokenizerPP"]


class SymbolicPaperTokenizer(GenericSymbolicTokenizer):
    """Source-compatible additive tokenizer used by the Figure 4b variants."""

    @staticmethod
    def _pad_token_key(rngs: nnx.Rngs) -> Array:
        # Keep the original RNG stream, even though the zero initializer ignores its key.
        return rngs.next()

    @staticmethod
    def _validate_component_indices(
        model_idx: Sequence[int] | None, noise_idx: Sequence[int] | None
    ) -> None:
        # The original implementation accepts and ignores subset-selection kwargs.
        pass

    def embed_cfgs(self, model_mask: ArrayLike, **_: Any) -> Array:
        return super().embed_cfgs(model_mask)


class SymbolicPaperTokenizerPP(SymbolicPaperTokenizer):
    """Tokenizer for symbolic AdditiveModel simulators with mask-aware encoding."""

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
        super().__init__(
            simulator,
            token_dim=token_dim,
            theta_encode_nets=theta_encode_nets,
            theta_decode_nets=theta_decode_nets,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            preferred_element_type=preferred_element_type,
            rngs=rngs,
        )
        self.start_end_token = nnx.Embed(2, token_dim // 2, rngs=rngs)
        self.model_mask_emb = nnx.Embed(2, token_dim, rngs=rngs)
        self.cfg_emb = nnx.Linear(
            token_dim, token_dim - token_dim // 2, rngs=rngs, **self._linear_kwargs
        )

    def embed_cfgs(self, model_mask: ArrayLike, **_: Any) -> Array:
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        (*batch_dims, mask_dim) = mask.shape
        if mask_dim != self.num_components:
            raise ValueError(
                f"model_mask length {mask_dim} does not match number of components {self.num_components}"
            )
        pad_value = self.pad_token.value
        pad_token = jnp.broadcast_to(pad_value, tuple(batch_dims) + (1, self.token_dim))
        idx = jnp.arange(self.num_components, dtype=jnp.int32)
        idx_tokens = self.embed_idx(idx)
        idx_tokens = jnp.broadcast_to(
            idx_tokens, tuple(batch_dims) + (self.num_components, self.token_dim)
        )
        return jnp.concatenate([pad_token, idx_tokens], axis=-2)

    def embed_model_mask(self, tokens_cfg: Array, model_mask: Array) -> Array:
        tokens_cfg = tokens_cfg[..., 1:, :]
        tokens_cfg = self.cfg_emb(tokens_cfg)
        ndims = tokens_cfg.ndim - 1
        start_token = self.start_end_token(jnp.array(0, dtype=jnp.int32))
        start_token = start_token.reshape((ndims - 1) * (1,) + (-1,))
        start_token = jnp.broadcast_to(
            start_token, tokens_cfg.shape[:-2] + (start_token.shape[-1],)
        )
        end_token = self.start_end_token(jnp.array(1, dtype=jnp.int32))
        end_token = end_token.reshape((ndims - 1) * (1,) + (-1,))
        end_token = jnp.broadcast_to(
            end_token, tokens_cfg.shape[:-2] + (end_token.shape[-1],)
        )
        tokens_inc = jnp.concatenate(
            [tokens_cfg[..., :-1, :], tokens_cfg[..., 1:, :]], axis=-1
        )
        tokens_start = jnp.concatenate([tokens_cfg[..., 0, :], start_token], axis=-1)
        tokens_end = jnp.concatenate([tokens_cfg[..., -1, :], end_token], axis=-1)
        tokens_cfg = jnp.concatenate(
            [tokens_start[..., None, :], tokens_inc, tokens_end[..., None, :]], axis=-2
        )
        model_mask_int = model_mask.astype(jnp.int32)
        embedded_mask = self.model_mask_emb(model_mask_int)
        zero_pad = jnp.zeros(
            embedded_mask.shape[:-2] + (2, embedded_mask.shape[-1]),
            dtype=embedded_mask.dtype,
        )
        embedded_mask = jnp.concatenate([zero_pad, embedded_mask[..., :-1, :]], axis=-2)
        return tokens_cfg + embedded_mask
