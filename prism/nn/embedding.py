"""Transformer encoder for observations given as token sequences."""

from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from probjax.nn import PosEncode, Transformer
from probjax.nn.layers.attention import flex_attention
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike


# Adapted from mackelab/sbmi_transformer_symbolic (MIT), c8e118b.
@dataclass
class TransformerEmbeddingConfig:
    input_dim: int = 1
    num_layers: int = 3
    num_heads: int = 4
    widening_factor: int = 4
    use_flash_attention: bool = False
    attn_size: int = 16
    positional_encoding: bool = True
    dropout_rate: float = 0.0
    y_seq_dim: int | None = None
    y_glob_dim: int | None = None
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None
    use_global_summary_token: bool = False
    out_norm: bool = True
    reduce_factor: int = 1


class TransformerEmbedding(nnx.Module):
    """Transformer encoder mapping observation tokens to sequence and summary tokens."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        input_dim: int,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        widening_factor: int = 2,
        attn_size: int = 16,
        positional_encoding: bool = True,
        dropout_rate: float = 0.0,
        use_flash_attention: bool = False,
        y_seq_dim: int | None = None,
        y_glob_dim: int | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        use_global_summary_token: bool = False,
        out_norm: bool = False,
        reduce_factor: int = 1,
    ) -> None:
        if input_dim <= 0 or reduce_factor <= 0 or model_dim % reduce_factor:
            raise ValueError(
                "input_dim and reduce_factor must be positive, and model_dim divisible by reduce_factor"
            )
        self.model_dim = model_dim // reduce_factor
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size
        self.use_global_summary_token = use_global_summary_token
        self.out_norm = out_norm
        self.reduce_factor = max(1, reduce_factor)
        self.input_dim = input_dim

        precision_kwargs = {
            "dtype": dtype,
            "param_dtype": param_dtype,
            "precision": precision,
            "preferred_element_type": preferred_element_type,
        }
        self.input_projection: nnx.Linear | None = (
            nnx.Linear(
                self.input_dim,
                self.model_dim,
                rngs=rngs,
                **precision_kwargs,
            )
            if self.input_dim != self.model_dim
            else None
        )

        attention_fn = flex_attention if use_flash_attention else None
        self.transformer = Transformer(
            self.model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
            attention_fn=attention_fn,
            **precision_kwargs,
        )

        if positional_encoding:
            self.pos_encoder = PosEncode(rngs=rngs)
        else:
            self.pos_encoder = None

        if self.use_global_summary_token:
            # Learned CLS-style token
            self.summary_token = nnx.Param(
                0.01 * nnx.initializers.normal()(rngs.params(), (1, self.model_dim))
            )
        else:
            self.summary_token = None

        self.y_seq_dim = y_seq_dim or self.model_dim
        self.y_glob_dim = y_glob_dim or self.model_dim
        self.output_seq_layer = (
            nnx.Linear(self.model_dim, self.y_seq_dim, rngs=rngs, **precision_kwargs)
            if self.y_seq_dim != self.model_dim
            else lambda x: x
        )
        self.output_glob_layer = (
            nnx.Linear(self.model_dim, self.y_glob_dim, rngs=rngs, **precision_kwargs)
            if self.y_glob_dim != self.model_dim
            else lambda x: x
        )
        self.out_norm_seq = (
            nnx.LayerNorm(self.y_seq_dim, rngs=rngs) if out_norm else lambda x: x
        )
        self.out_norm_glob = (
            nnx.LayerNorm(self.y_glob_dim, rngs=rngs) if out_norm else lambda x: x
        )

    def _prepend_summary_token(self, tokens: Array) -> Array:
        if self.summary_token is None:
            return tokens
        summary_val = self.summary_token.value
        batch_shape = tokens.shape[:-2]
        summary = jnp.broadcast_to(
            summary_val,
            batch_shape + summary_val.shape[-2:],
        )
        return jnp.concatenate([summary, tokens], axis=-2)

    def encode_observations(
        self,
        x: ArrayLike,
        acq: None = None,
        deterministic: bool | None = None,
        decode: bool = False,
    ) -> tuple[Array | None, Array]:
        """Encode observations; coordinates, if any, are columns of ``x``."""
        if acq is not None:
            raise ValueError("Put observation coordinates in x for the generic encoder")
        return self(x, deterministic=deterministic, decode=decode)

    def __call__(
        self,
        conditions: ArrayLike,
        deterministic: bool | None = None,
        decode: bool = False,
    ) -> tuple[Array | None, Array]:
        tokens = jnp.asarray(conditions)
        if tokens.ndim < 2 or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected observations shaped (..., points, {self.input_dim}), got {tokens.shape}"
            )
        if self.input_projection is not None:
            tokens = self.input_projection(tokens)

        if self.pos_encoder is not None:
            tokens = self.pos_encoder(tokens)

        tokens = self._prepend_summary_token(tokens)
        out_tokens = self.transformer(
            tokens, deterministic=deterministic, decode=decode
        )

        if self.summary_token is not None:
            global_summary = out_tokens[..., 0, :]
            sequence_tokens = out_tokens[..., 1:, :]
            global_summary = self.out_norm_glob(self.output_glob_layer(global_summary))
        else:
            global_summary = None
            sequence_tokens = out_tokens
        sequence_tokens = self.out_norm_seq(self.output_seq_layer(sequence_tokens))
        return global_summary, sequence_tokens
