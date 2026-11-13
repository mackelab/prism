"""The shared causal decoder must define a normalized joint model distribution."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from prism.dmri.nn.tokenizer import DMRITokenizerPPP
from prism.dmri.simulators import Ball2Stick
from prism.nn.autoregressive import BinaryAutoregressiveDecoder as DMRIDecoder
from prism.symbolic.paper.autoregressive import (
    BinaryAutoregressiveDecoder as PaperDecoder,
)
from prism.symbolic.paper.tokenizer import SymbolicPaperTokenizerPP
from prism.symbolic.simulators import SimplePolynomial


@pytest.mark.parametrize("variant", ["dmri", "symbolic"])
def test_log_prob_is_normalized_over_all_masks_and_preserves_causality(variant):
    if variant == "dmri":
        decoder = DMRIDecoder(
            nnx.Rngs(11),
            model_dim=8,
            num_heads=2,
            num_layers=1,
            attn_size=4,
            prior_params_embed_dim=4,
            mask_prior_dim=1,
        )
        tokenizer = DMRITokenizerPPP(Ball2Stick, token_dim=8, rngs=nnx.Rngs(12))
        context = {"mask_prior": jnp.array([0.4])}
    else:
        decoder = PaperDecoder(
            nnx.Rngs(11), model_dim=8, num_heads=2, num_layers=1, attn_size=4
        )
        tokenizer = SymbolicPaperTokenizerPP(
            SimplePolynomial, token_dim=8, rngs=nnx.Rngs(12)
        )
        context = {}
    decoder.eval()
    dim = tokenizer.num_models + tokenizer.num_noises
    masks = jnp.array(list(product([False, True], repeat=dim)))
    observations = jnp.linspace(-1, 1, 24).reshape(3, 8)

    @jax.jit
    def evaluate():
        logits = jax.vmap(
            lambda mask: decoder(mask, tokenizer, y=observations, **context)
        )(masks)
        log_probs = jax.vmap(
            lambda mask: decoder.log_prob(mask, tokenizer, observations, **context)
        )(masks)
        return logits, log_probs

    logits, log_probs = evaluate()
    assert np.isfinite(log_probs).all()
    np.testing.assert_allclose(jnp.exp(log_probs).sum(), 1.0, rtol=1e-6)
    assert float(jnp.ptp(log_probs)) > 1e-3
    # Changing the current or any future bit cannot change its prediction.
    for position in range(dim):
        for prefix in product([False, True], repeat=position):
            same_prefix = np.all(np.asarray(masks)[:, :position] == prefix, axis=1)
            selected = np.asarray(logits)[same_prefix, position]
            np.testing.assert_allclose(selected, selected[0], rtol=1e-6, atol=1e-6)
