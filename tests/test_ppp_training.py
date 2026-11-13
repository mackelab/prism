"""Training and inference must use the same causal PPP mask encoding."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from prism.dmri.nn.embedding import DMRIEmbeddingConfig
from prism.dmri.nn.model import DMRIModelConfigMaskPriorAmortizedPPP
from prism.dmri.nn.tokenizer import DMRITokenizerPP, DMRITokenizerPPP
from prism.dmri.simulators import Ball2Stick
from prism.dmri.simulators.acquisition_scheme import acquisition_scheme
from prism.nn.autoregressive import (
    BinaryAutoregressiveDecoder,
    ModelSelectionAmortizedPriorConfig,
)
from prism.nn.model import PrismModel
from prism.nn.simformer import ThetaInferenceConfig


def make_decoder(tokenizer_cls):
    tokenizer = tokenizer_cls(Ball2Stick, token_dim=16, rngs=nnx.Rngs(12))
    decoder = BinaryAutoregressiveDecoder(
        nnx.Rngs(11),
        model_dim=16,
        num_heads=2,
        num_layers=1,
        attn_size=4,
        prior_params_embed_dim=4,
        mask_prior_dim=1,
    )
    decoder.eval()
    dim = tokenizer.num_models + tokenizer.num_noises
    masks = jnp.array([[i % 2 == row for i in range(dim)] for row in (0, 1)])
    y = jax.random.normal(jax.random.key(13), (2, 3, 16))
    prior = jnp.array([[0.3], [0.7]])
    return decoder, tokenizer, masks, y, prior


@pytest.mark.parametrize("tokenizer_cls", [DMRITokenizerPP, DMRITokenizerPPP])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_cached_and_uncached_training_match_inference(tokenizer_cls, smoothing):
    decoder, tokenizer, masks, y, prior = make_decoder(tokenizer_cls)
    logits = decoder(masks, tokenizer, y=y, mask_prior=prior)
    targets = masks * (1 - smoothing) + 0.5 * smoothing
    expected = optax.sigmoid_binary_cross_entropy(logits, targets).sum(-1).mean()
    for tokens in (None, tokenizer.embed_cfgs(masks)):
        actual = decoder.loss_fn(
            masks,
            tokenizer,
            y=y,
            mask_prior=prior,
            tokens_cfg=tokens,
            label_smoothing=smoothing,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_permuted_ppp_training_matches_inference_in_that_order():
    decoder, tokenizer, masks, y, prior = make_decoder(DMRITokenizerPPP)
    tokens = tokenizer.embed_cfgs(masks)
    key = jax.random.key(14)
    orders = jax.vmap(lambda k: jax.random.permutation(k, masks.shape[-1]))(
        jax.random.split(key, masks.shape[0])
    )
    permuted_masks = jax.vmap(lambda mask, order: mask[order])(masks, orders)
    permuted_tokens = tokens.at[:, 1:, :].set(
        jax.vmap(lambda token, order: token[order])(tokens[:, 1:, :], orders)
    )
    expected = -decoder.log_prob(
        permuted_masks,
        tokenizer,
        y,
        mask_prior=prior,
        tokens_cfg=permuted_tokens,
    ).mean()
    for cached in (None, tokens):
        actual = decoder.loss_fn(
            masks,
            tokenizer,
            y=y,
            mask_prior=prior,
            tokens_cfg=cached,
            rng=key,
            permute_order=True,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_cached_ppp_training_updates_mask_encoding_parameters():
    decoder, tokenizer, masks, y, prior = make_decoder(DMRITokenizerPPP)

    def loss(tok):
        return decoder.loss_fn(
            masks,
            tok,
            y=y,
            mask_prior=prior,
            tokens_cfg=tok.embed_cfgs(masks),
        )

    value, gradients = nnx.value_and_grad(loss)(tokenizer)
    assert np.isfinite(value)
    for name in ("cfg_emb", "model_mask_emb", "start_end_token"):
        leaves = jax.tree.leaves(gradients[name])
        assert all(np.isfinite(leaf).all() for leaf in leaves)
        assert sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves) > 0


def test_joint_ppp_loss_reports_inference_mask_nll():
    cfg = DMRIModelConfigMaskPriorAmortizedPPP(
        Ball2Stick,
        model_dim=16,
        embedding_cfg=DMRIEmbeddingConfig(num_layers=1, num_heads=2, attn_size=4),
        model_selection_cfg=ModelSelectionAmortizedPriorConfig(
            num_layers=1,
            num_heads=2,
            attn_size=4,
            prior_params_embed_dim=4,
            outlayer="mlp",
            outnorm=False,
        ),
        theta_inference_cfg=ThetaInferenceConfig(
            num_layers=1,
            num_heads=2,
            attn_size=4,
            time_embed_dim=16,
        ),
    )
    model = PrismModel(cfg, nnx.Rngs(15))
    model.eval()
    dim = model.tokenizer.num_models + model.tokenizer.num_noises
    masks = jnp.ones((2, dim), dtype=bool).at[0, 1].set(False)
    theta = jnp.zeros((2, Ball2Stick.theta_dim))
    x = jnp.linspace(0.1, 1.0, 16).reshape(2, 8)
    acq = acquisition_scheme(
        bvals=jnp.broadcast_to(jnp.linspace(0, 3000, 8), (2, 8)),
        bvecs=jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (2, 8, 3)),
    )
    prior = jnp.array([[0.3], [0.7]])
    losses = model.loss_fn(jax.random.key(16), masks, theta, x, acq, mask_prior=prior)
    context, observations = model._encode_observations(acq, x)
    expected = -model.model_decoder.log_prob(
        masks,
        model.tokenizer,
        observations,
        mask_prior=prior,
        additional_context=context,
    ).mean()
    assert np.isfinite(losses).all()
    np.testing.assert_allclose(losses[0], expected, rtol=1e-6, atol=1e-6)
