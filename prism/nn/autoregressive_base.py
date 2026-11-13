"""Shared causal attention and sampling for binary model selection.

Decoder variants own token construction, output heads, and context preparation.
Keeping those methods separate preserves the default decoder's prior conditioning
and the Figure 4b shifted-mask architecture without duplicating their common
attention path.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from probjax.nn import CausalMask
from probjax.utils.typing import Array, ArrayLike, RngKey

from prism.nn.tokenizer import Tokenizer


class AutoregressiveDecoderBase(nnx.Module):
    """Common operations; subclasses construct modules in checkpoint order."""

    def _forward_tokens(
        self,
        input_tokens: Array,
        y: Array | None,
        context: Array | None = None,
        attention_mask: Array | None = None,
        decode: bool = False,
        deterministic: bool = False,
    ) -> Array:
        *_, seq_len, _ = input_tokens.shape

        # Autoregressive mask constrained
        if not self.use_flash_attention:
            base_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            if attention_mask is not None:
                base_mask = base_mask & attention_mask
        else:
            base_mask = CausalMask()
            if attention_mask is not None:
                raise NotImplementedError("Not supported with flash attention")

        batch_shape = input_tokens.shape[:-2]
        if self.total_context_dim > 0:
            if context is None:
                raise ValueError("Context expected but not provided.")
            context = jnp.asarray(context, dtype=input_tokens.dtype)
            if context.shape[:-1] != batch_shape:
                raise ValueError("Context batch shape does not match inputs.")
            if context.shape[-1] != self.total_context_dim:
                raise ValueError(
                    "Context dimension mismatch for autoregressive decoder."
                )
        elif context is not None:
            raise ValueError("Context provided but no context dimension configured.")

        if context is not None:
            context = context[..., None, :]

        x = self.transformer(
            input_tokens,
            y,
            y,
            context=context,
            mask=base_mask,
            deterministic=deterministic,
            decode=decode,
        )
        return x

    def log_prob(
        self,
        model_mask: Array,
        tokenizer: Tokenizer,
        y: Array,
        mask_prior: Array | None = None,
        additional_context: Array | None = None,
        **kwargs: Any,
    ) -> Array:
        model_mask_logits = self(
            model_mask,
            tokenizer,
            y=y,
            mask_prior=mask_prior,
            additional_context=additional_context,
            **kwargs,
        )
        # Correct Bernoulli log probability is negative binary cross entropy
        bernoulli_log_prob = -optax.sigmoid_binary_cross_entropy(
            model_mask_logits, model_mask
        )
        return jnp.sum(bernoulli_log_prob, axis=-1)


def sample_binary_mask(
    model: AutoregressiveDecoderBase,
    key: RngKey,
    tokenizer: Tokenizer,
    y: Array,
    dim: int,
    mask_prior: Array | None = None,
    additional_context: Array | None = None,
    temperature: ArrayLike | None = None,
) -> Array:
    """Sample each bit from the preceding bits, preserving the RNG split order.

    ``temperature=None`` keeps the default decoder free of temperature arithmetic.
    The Figure 4b decoder supports scalar and per-component temperatures.
    """
    x = jnp.zeros((dim,), dtype=jnp.bool_)
    if temperature is not None:
        temperature_arr = jnp.asarray(
            temperature, dtype=jnp.result_type(y, jnp.float32)
        )
        temperature_arr = jnp.broadcast_to(temperature_arr, (dim,))
        min_temperature = jnp.finfo(temperature_arr.dtype).tiny

    def scan_fn(carry: tuple[Array, int], k: RngKey) -> tuple[tuple[Array, int], None]:
        x, i = carry
        logits = model(
            x.astype(jnp.int32),
            tokenizer,
            y=y,
            mask_prior=mask_prior,
            additional_context=additional_context,
        )
        logit_i = logits[i]
        if temperature is not None:
            temp_i = jnp.maximum(temperature_arr[i], min_temperature)
            logit_i = logit_i / temp_i
        p_i = jax.nn.sigmoid(logit_i)
        x_i = jax.random.bernoulli(k, p_i)
        x = x.at[i].set(x_i)
        return (x, i + 1), None

    keys = jax.random.split(key, (dim,))
    x, _ = jax.lax.scan(scan_fn, (x, 0), keys)
    return x[0]
