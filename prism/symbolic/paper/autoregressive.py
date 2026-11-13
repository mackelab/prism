"""Figure 4b model selection with shifted masks and an MLP output head.

Causal attention, log probabilities, and sampling use the shared decoder base.
Token preparation, context broadcasting, and training loss retain the symbolic
source behavior and differ from the dMRI decoder.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from probjax.nn import MLP, GaussianFourierEmbedding, Transformer
from probjax.nn.layers.attention import flex_attention
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, RngKey

from prism.nn.autoregressive_base import AutoregressiveDecoderBase, sample_binary_mask

from .tokenizer import Tokenizer


@dataclass
class SymbolicPaperModelSelectionConfig:
    num_layers: int = 4
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16
    dropout_rate: float = 0.0
    prior_params_embed_dim: int = 0
    mask_prior_dim: int | None = None
    kv_in_features: int | None = None
    use_flash_attention: bool = False
    use_flash_cross_attention: bool = False
    normalize_qk_attn: bool = False
    normalize_qk_cross_attn: bool = False
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None
    outlayer_ctx: bool = False
    precondition_with_mask_prior: bool = False
    prior_mask: Any = None


@dataclass
class SymbolicPaperModelSelectionAmortizedPriorConfig(
    SymbolicPaperModelSelectionConfig
):
    prior_params_embed_dim: int = 64
    mask_prior_dim: int | None = 1


class BinaryAutoregressiveDecoder(AutoregressiveDecoderBase):
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 4
    widening_factor: int = 2
    attn_size: int = 16

    def __init__(
        self,
        rngs: nnx.Rngs,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        widening_factor: int = 4,
        attn_size: int = 16,
        dropout_rate: float = 0.0,
        prior_params_embed_dim: int = 0,
        mask_prior_dim: int | None = None,
        additional_context_dim: int = 0,
        kv_in_features: int | None = None,
        enable_cross_attention: bool = True,
        use_flash_attention: bool = False,
        use_flash_cross_attention: bool = False,
        normalize_qk_attn: bool = False,
        normalize_qk_cross_attn: bool = False,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        outlayer_ctx: bool = False,
        **kwargs: Any,
    ):
        self._validate_prior_configuration(
            prior_params_embed_dim, mask_prior_dim, kwargs
        )
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_size = attn_size
        self.prior_params_embed_dim = prior_params_embed_dim
        self.additional_context_dim = additional_context_dim
        self.total_context_dim = (
            self.prior_params_embed_dim + self.additional_context_dim
        )
        context_dim = self.total_context_dim if self.total_context_dim > 0 else None
        self.context_dim = context_dim
        self.use_flash_attention = use_flash_attention
        precision_kwargs: dict[str, Any] = {
            name: value
            for (name, value) in (
                ("dtype", dtype),
                ("param_dtype", param_dtype),
                ("precision", precision),
                ("preferred_element_type", preferred_element_type),
            )
            if value is not None
        }
        attn_fn = flex_attention if use_flash_attention else None
        cross_attn_fn = flex_attention if use_flash_cross_attention else None
        self._initialize_prior_embedding(rngs, mask_prior_dim, precision_kwargs)
        self.kv_in_features = (
            kv_in_features if kv_in_features is not None else model_dim
        )
        self.transformer = Transformer(
            model_dim,
            self.num_heads,
            self.num_layers,
            self.attn_size,
            kv_in_features=self.kv_in_features,
            context_dim=context_dim,
            widening_factor=self.widening_factor,
            rngs=rngs,
            dropout_rate=dropout_rate,
            attention_fn=attn_fn,
            cross_attention_fn=cross_attn_fn,
            enable_cross_attention=enable_cross_attention,
            normalize_qk_attn=normalize_qk_attn,
            normalize_qk_cross_attn=normalize_qk_cross_attn,
            **precision_kwargs,
        )
        self.out_norm = lambda x: x
        self.outlayer_ctx = outlayer_ctx
        context_dim_out = self.context_dim if outlayer_ctx else None
        self.output = MLP(
            [model_dim, widening_factor * model_dim, 1],
            rngs=rngs,
            context_dim=context_dim_out,
            **precision_kwargs,
        )

    def _validate_prior_configuration(
        self, prior_params_embed_dim: int, mask_prior_dim: int | None, kwargs: dict
    ) -> None:
        if (
            prior_params_embed_dim != 0
            or mask_prior_dim not in (None, 0)
            or kwargs.get("precondition_with_mask_prior", False)
            or kwargs.get("prior_mask") is not None
        ):
            raise ValueError(
                "This symbolic decoder requires mask_prior_dim=0 and no prior preconditioning."
            )

    def _initialize_prior_embedding(
        self, rngs: nnx.Rngs, mask_prior_dim: int | None, precision_kwargs: dict
    ) -> None:
        self.mask_prior_dim = None
        self.mask_prior_embed = None

    def _prepare_prior_context(
        self, batch_shape: tuple, dtype: jnp.dtype, mask_prior: Array | None
    ) -> list[Array]:
        if mask_prior is not None:
            raise ValueError(
                "Mask prior provided but prior_params_embed_dim is set to 0."
            )
        return []

    def _precondition_logits(
        self, logits: Array, model_mask: Array, mask_prior: Array | None
    ) -> Array:
        return logits

    def __call__(
        self,
        model_mask: Array,
        tokenizer: Tokenizer,
        mask_prior: Array | None = None,
        additional_context: Array | None = None,
        y: Array | None = None,
        mask: Array | None = None,
        decode: bool = False,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Array:
        input_tokens = self._encode_model_mask(model_mask, tokenizer, **kwargs)
        batch_shape = input_tokens.shape[:-2]
        context_vec = self._prepare_context(
            tuple(batch_shape),
            input_tokens.dtype,
            mask_prior=mask_prior,
            additional_context=additional_context,
        )
        output_tokens = self._forward_tokens(
            input_tokens,
            y,
            context=context_vec,
            attention_mask=mask,
            decode=decode,
            deterministic=deterministic,
        )
        output_tokens = self.out_norm(output_tokens)
        if self.outlayer_ctx:
            logits = self.output(output_tokens, context=context_vec)
        else:
            logits = self.output(output_tokens)
        return logits[..., :-1, 0]

    def _encode_model_mask(
        self, model_mask: ArrayLike, tokenizer: Tokenizer, **kwargs: Any
    ) -> Array:
        kwargs.pop("tokens_cfg", None)
        mask_for_cfg = jnp.ones_like(model_mask, dtype=jnp.bool_)
        cfg_tokens = tokenizer.encode(model_mask=mask_for_cfg, **kwargs)
        input_tokens = tokenizer.embed_model_mask(cfg_tokens, model_mask)
        (*_, _, model_dim) = input_tokens.shape
        assert model_dim == self.model_dim, (
            f"Token dim mismatch, is {model_dim}, expected {self.model_dim}"
        )
        return input_tokens

    def _prepare_context(
        self,
        batch_shape: tuple[Any, ...],
        dtype: jnp.dtype,
        mask_prior: Array | None,
        additional_context: Array | None,
    ) -> Array | None:
        batch_shape = tuple(batch_shape)
        context_parts = self._prepare_prior_context(batch_shape, dtype, mask_prior)
        if self.additional_context_dim > 0:
            if additional_context is None:
                additional_context_arr = jnp.zeros(
                    batch_shape + (self.additional_context_dim,), dtype=dtype
                )
            else:
                additional_context_arr = jnp.asarray(additional_context, dtype=dtype)
                target_shape = batch_shape + (self.additional_context_dim,)
                if additional_context_arr.shape[-1] != self.additional_context_dim:
                    raise ValueError(
                        "Additional context dimensionality does not match configuration."
                    )
                while (
                    additional_context_arr.ndim > len(target_shape)
                    and additional_context_arr.shape[0] == 1
                ):
                    additional_context_arr = jnp.squeeze(additional_context_arr, axis=0)
                if additional_context_arr.shape != target_shape:
                    try:
                        additional_context_arr = jnp.broadcast_to(
                            additional_context_arr, target_shape
                        )
                    except ValueError as err:
                        raise ValueError(
                            "Additional context batch shape does not match tokens."
                        ) from err
            context_parts.append(additional_context_arr)
        elif additional_context is not None:
            raise ValueError(
                "Additional context provided but additional_context_dim is 0."
            )
        if not context_parts:
            return None
        return jnp.concatenate(context_parts, axis=-1)

    def loss_fn(
        self,
        model_mask: ArrayLike,
        tokenizer: Tokenizer,
        y: Array,
        rng: RngKey | None = None,
        permute_order: bool = False,
        label_smoothing: float = 0.0,
        mask_prior: Array | None = None,
        additional_context: Array | None = None,
        tokens_cfg: Array | None = None,
        **kwargs: Any,
    ) -> Array:
        if permute_order:
            assert rng is not None, "rng must be provided if permute_order is True"
        del tokens_cfg
        cfg_mask = jnp.ones_like(model_mask, dtype=jnp.bool_)
        input_tokens = tokenizer.encode(model_mask=cfg_mask, **kwargs)
        (*batch_shape, seq_len, _) = input_tokens.shape
        if permute_order:
            assert rng is not None, "rng must be provided if permute_order is True"
            elements = jnp.arange(seq_len - 1)
            batch_orders = jax.vmap(lambda k: jax.random.permutation(k, elements))(
                jax.random.split(rng, int(jnp.prod(jnp.array(batch_shape))))
            ).reshape(batch_shape + [-1])
            tokens_except_first = input_tokens[..., 1:, :]

            def permute_batch_element(tokens, order):
                return tokens[order]

            tokens_except_first = jax.vmap(permute_batch_element)(
                tokens_except_first, batch_orders
            )
            input_tokens = input_tokens.at[..., 1:, :].set(tokens_except_first)
            model_mask = jax.vmap(permute_batch_element)(model_mask, batch_orders)
        input_tokens = tokenizer.embed_model_mask(input_tokens, model_mask)
        (*_, _, model_dim) = input_tokens.shape
        assert model_dim == self.model_dim, (
            f"Token dim mismatch, is {model_dim}, expected {self.model_dim}"
        )
        if label_smoothing > 0.0:
            model_mask = model_mask * (1.0 - label_smoothing) + 0.5 * label_smoothing
        context_vec = self._prepare_context(
            tuple(batch_shape),
            input_tokens.dtype,
            mask_prior=mask_prior,
            additional_context=additional_context,
        )
        output_tokens = self._forward_tokens(
            input_tokens, y, context=context_vec, **kwargs
        )
        if self.outlayer_ctx:
            model_mask_logits = self.output(output_tokens, context=context_vec)
        else:
            model_mask_logits = self.output(output_tokens)
        model_mask_logits = model_mask_logits[..., :-1, 0]
        model_mask_logits = self._precondition_logits(
            model_mask_logits, model_mask, mask_prior
        )
        nll = optax.sigmoid_binary_cross_entropy(model_mask_logits, model_mask).sum(-1)
        return jnp.mean(nll)

    def sample(
        self,
        key,
        tokenizer,
        y,
        dim,
        mask_prior: Array | None = None,
        additional_context: Array | None = None,
        temperature: ArrayLike = 1.0,
    ):
        return naive_autoregressive_decoding(
            self,
            key,
            tokenizer,
            y,
            dim,
            mask_prior=mask_prior,
            additional_context=additional_context,
            temperature=temperature,
        )


class PriorConditionedBinaryAutoregressiveDecoder(BinaryAutoregressiveDecoder):
    """Source symbolic head with a Fourier lambda context and prior log odds.

    The Fourier embedding is initialized before the transformer, matching the
    original symbolic source's parameter paths and RNG consumption. The fixed
    Figure 4b decoder above keeps its original initialization and rejection rules.
    Forward calls return learned logits; loss, log probability, and sampling add
    the conditional prior log odds, as in the source implementation.
    """

    def __init__(self, rngs: nnx.Rngs, *, prior_mask: Any = None, **kwargs: Any):
        self.prior_mask = prior_mask
        super().__init__(rngs, prior_mask=prior_mask, **kwargs)

    def _validate_prior_configuration(
        self, prior_params_embed_dim: int, mask_prior_dim: int | None, kwargs: dict
    ) -> None:
        if prior_params_embed_dim <= 0 or mask_prior_dim is None or mask_prior_dim <= 0:
            raise ValueError(
                "Prior-conditioned symbolic decoding requires positive "
                "prior_params_embed_dim and mask_prior_dim."
            )
        if (
            kwargs.get("precondition_with_mask_prior", False)
            and self.prior_mask is None
        ):
            raise ValueError("Prior preconditioning requires a prior_mask object.")
        if self.prior_mask is not None and not callable(
            getattr(self.prior_mask, "log_prob_autoregressive", None)
        ):
            raise ValueError("prior_mask must implement log_prob_autoregressive.")

    def _initialize_prior_embedding(
        self, rngs: nnx.Rngs, mask_prior_dim: int | None, precision_kwargs: dict
    ) -> None:
        self.mask_prior_dim = mask_prior_dim
        self.mask_prior_embed = GaussianFourierEmbedding(
            mask_prior_dim, self.prior_params_embed_dim, rngs=rngs, **precision_kwargs
        )

    def _broadcast_mask_prior(
        self, batch_shape: tuple, dtype: jnp.dtype, mask_prior: Array | None
    ) -> Array:
        if mask_prior is None:
            raise ValueError("mask_prior must be provided for prior-parameter context.")
        prior = jnp.asarray(mask_prior, dtype=dtype)
        expected_shape = batch_shape + (self.mask_prior_dim,)
        # A scalar is a shared hyperparameter; vectors may omit the final axis.
        if prior.ndim == 0 or prior.ndim == len(batch_shape):
            prior = prior[..., None]
        if prior.shape[-1] != self.mask_prior_dim:
            if prior.shape[-1] != 1:
                raise ValueError(
                    "Mask prior dimensionality does not match configured mask_prior_dim."
                )
            prior = jnp.broadcast_to(prior, prior.shape[:-1] + (self.mask_prior_dim,))
        try:
            return jnp.broadcast_to(prior, expected_shape)
        except ValueError as err:
            raise ValueError("Mask prior batch shape does not match tokens.") from err

    def _prepare_prior_context(
        self, batch_shape: tuple, dtype: jnp.dtype, mask_prior: Array | None
    ) -> list[Array]:
        prior = self._broadcast_mask_prior(batch_shape, dtype, mask_prior)
        return [self.mask_prior_embed(prior)]

    def _prior_logit_at(
        self, model_mask: Array, mask_prior: Array, position: Array
    ) -> Array:
        mask0 = model_mask.at[position].set(False)
        mask1 = model_mask.at[position].set(True)
        logp0 = self.prior_mask.log_prob_autoregressive(mask0, mask_prior)[position]
        logp1 = self.prior_mask.log_prob_autoregressive(mask1, mask_prior)[position]
        safe_diff = jnp.where(
            jnp.isfinite(logp1) & jnp.isfinite(logp0),
            logp1 - logp0,
            jnp.where(
                jnp.isfinite(logp1),
                50.0,
                jnp.where(jnp.isfinite(logp0), -50.0, 0.0),
            ),
        )
        return jnp.clip(safe_diff, -50.0, 50.0)

    def _prior_logits_single(self, model_mask: Array, mask_prior: Array) -> Array:
        return jax.vmap(lambda i: self._prior_logit_at(model_mask, mask_prior, i))(
            jnp.arange(model_mask.shape[-1])
        )

    def _precondition_logits(
        self, logits: Array, model_mask: Array, mask_prior: Array | None
    ) -> Array:
        if self.prior_mask is None:
            return logits
        model_mask = jnp.asarray(model_mask)
        prior = self._broadcast_mask_prior(
            model_mask.shape[:-1], logits.dtype, mask_prior
        )
        if model_mask.ndim == 1:
            return logits + self._prior_logits_single(model_mask, prior)
        flat_mask = model_mask.reshape((-1, model_mask.shape[-1]))
        flat_prior = prior.reshape((-1, self.mask_prior_dim))
        prior_logits = jax.vmap(self._prior_logits_single)(flat_mask, flat_prior)
        return logits + prior_logits.reshape(model_mask.shape)

    def log_prob(
        self,
        model_mask: Array,
        tokenizer: Tokenizer,
        y: Array,
        mask_prior: Array | None = None,
        additional_context: Array | None = None,
        **kwargs: Any,
    ) -> Array:
        logits = self(
            model_mask,
            tokenizer,
            y=y,
            mask_prior=mask_prior,
            additional_context=additional_context,
            **kwargs,
        )
        logits = self._precondition_logits(logits, model_mask, mask_prior)
        return -optax.sigmoid_binary_cross_entropy(logits, model_mask).sum(-1)

    def sample(
        self,
        key,
        tokenizer,
        y,
        dim,
        mask_prior=None,
        additional_context=None,
        temperature=1.0,
    ):
        if self.prior_mask is None:
            return super().sample(
                key,
                tokenizer,
                y,
                dim,
                mask_prior=mask_prior,
                additional_context=additional_context,
                temperature=temperature,
            )
        return naive_prior_autoregressive_decoding(
            self,
            key,
            tokenizer,
            y,
            dim,
            mask_prior=mask_prior,
            additional_context=additional_context,
            temperature=temperature,
        )


@partial(jax.jit, static_argnums=(2, 4))
def naive_prior_autoregressive_decoding(
    model: PriorConditionedBinaryAutoregressiveDecoder,
    key: RngKey,
    tokenizer: Tokenizer,
    y: Array,
    dim: int,
    mask_prior: Array | None = None,
    additional_context: Array | None = None,
    temperature: ArrayLike = 1.0,
) -> Array:
    model_mask = jnp.zeros((dim,), dtype=jnp.bool_)
    temperature = jnp.broadcast_to(
        jnp.asarray(temperature, dtype=jnp.result_type(y, jnp.float32)), (dim,)
    )
    min_temperature = jnp.finfo(temperature.dtype).tiny
    prior = model._broadcast_mask_prior((), temperature.dtype, mask_prior)

    def scan_fn(carry, step_key):
        model_mask, position = carry
        logits = model(
            model_mask.astype(jnp.int32),
            tokenizer,
            y=y,
            mask_prior=mask_prior,
            additional_context=additional_context,
        )
        # Source sampling tempers learned logits before adding the prior odds.
        logit = logits[position] / jnp.maximum(temperature[position], min_temperature)
        logit = logit + model._prior_logit_at(model_mask, prior, position)
        selected = jax.random.bernoulli(step_key, jax.nn.sigmoid(logit))
        return (model_mask.at[position].set(selected), position + 1), None

    (sampled, _), _ = jax.lax.scan(
        scan_fn, (model_mask, 0), jax.random.split(key, (dim,))
    )
    return sampled


@partial(jax.jit, static_argnums=(2, 4))
def naive_autoregressive_decoding(
    model: BinaryAutoregressiveDecoder,
    key: RngKey,
    tokenizer: Tokenizer,
    y: Array,
    dim: int,
    mask_prior: Array | None = None,
    additional_context: Array | None = None,
    temperature: ArrayLike = 1.0,
) -> Array:
    return sample_binary_mask(
        model,
        key,
        tokenizer,
        y,
        dim,
        mask_prior=mask_prior,
        additional_context=additional_context,
        temperature=temperature,
    )
