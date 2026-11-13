"""Compatibility of shared tokenization with the pre-cleanup model variants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from prism.nn.compartment_tokenizer import CompartmentTokenizer as GenericTokenizer
from prism.symbolic.paper.tokenizer import (
    SymbolicPaperTokenizer as PaperTokenizer,
)
from prism.symbolic.paper.tokenizer import (
    SymbolicPaperTokenizerPP,
)
from prism.symbolic.paper.tokenizer import (
    Tokenizer as PaperTokenizerBase,
)
from prism.symbolic.simulators import SimplePolynomial

TOKENIZERS = [GenericTokenizer, PaperTokenizer, SymbolicPaperTokenizerPP]

# Recorded before the cleanup with SimplePolynomial, width 4 and params seed 17.
# The Figure 4b whole-network fixture separately pins its default-stream weights.
INITIAL_EMBEDDING = np.array([
    [0.3729777336, -0.6898949742, 0.0633536801, 0.6171862483],
    [0.5930637717, 0.6212378740, 0.4201818407, 0.2928925455],
    [0.2378843427, -0.3700408936, 0.6461116076, -0.6237152219],
])
INITIAL_ENCODERS = np.array([
    [-0.0013154763, 0.0749774501, -2.1443896294, -0.6048249602],
    [-0.0816718638, 0.1102917567, 0.0194863416, 1.1692290306],
    [-0.2337273806, 1.7608468533, -0.0313179307, -1.2671915293],
])


def _make(cls):
    return cls(SimplePolynomial, token_dim=4, rngs=nnx.Rngs(params=17, next=23))


@pytest.mark.parametrize("cls", TOKENIZERS)
def test_initialization_and_checkpoint_parameter_paths(cls):
    tokenizer = _make(cls)
    np.testing.assert_allclose(
        tokenizer.embed_idx.embedding.value, INITIAL_EMBEDDING, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        np.stack([net.kernel.value[0] for net in tokenizer.theta_encode_nets]),
        INITIAL_ENCODERS,
        rtol=1e-6,
        atol=1e-7,
    )
    expected = {
        "['embed_idx']['embedding'].value": (3, 4),
        "['pad_token'].value": (1, 4),
    }
    for index in range(3):
        expected[f"['theta_encode_nets'][{index}]['kernel'].value"] = (1, 4)
        expected[f"['theta_encode_nets'][{index}]['bias'].value"] = (4,)
        expected[f"['theta_decode_nets'][{index}]['kernel'].value"] = (4, 1)
        expected[f"['theta_decode_nets'][{index}]['bias'].value"] = (1,)
    if cls is SymbolicPaperTokenizerPP:
        expected.update({
            "['cfg_emb']['bias'].value": (2,),
            "['cfg_emb']['kernel'].value": (4, 2),
            "['model_mask_emb']['embedding'].value": (2, 4),
            "['start_end_token']['embedding'].value": (2, 2),
        })
    paths = {
        jax.tree_util.keystr(path): value.shape
        for path, value in jax.tree_util.tree_flatten_with_path(
            nnx.state(tokenizer, nnx.Param)
        )[0]
    }
    assert paths == expected
    assert isinstance(tokenizer, PaperTokenizerBase)


@pytest.mark.parametrize(
    "cls,params_count,next_count",
    [
        (GenericTokenizer, 14, 0),
        (PaperTokenizer, 13, 1),
        (SymbolicPaperTokenizerPP, 17, 1),
    ],
)
def test_distinct_rng_stream_consumption_is_preserved(cls, params_count, next_count):
    rngs = nnx.Rngs(params=17, next=23)
    tokenizer = cls(SimplePolynomial, token_dim=4, rngs=rngs)
    assert int(rngs.params.count.value) == params_count
    assert int(rngs.next.count.value) == next_count
    if cls is SymbolicPaperTokenizerPP:
        # Moving the zero pad initializer from next to params changes these weights.
        np.testing.assert_allclose(
            tokenizer.start_end_token.embedding.value,
            [[0.0521771386, 0.3077472746], [0.1376315653, 1.2087504864]],
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.parametrize("cls", TOKENIZERS)
def test_batched_parameter_encoding_masking_and_restore(cls):
    tokenizer = _make(cls)
    # Nonzero decoder weights exercise decoding rather than just the zero output.
    for index, decoder in enumerate(tokenizer.theta_decode_nets):
        decoder.kernel.value = jnp.arange(4.0)[:, None] * (index + 1)
    theta = jnp.array([[0.2, -0.5, 0.7], [0.3, 0.1, -0.4]])
    mask = jnp.array([[True, False, True], [False, True, True]])
    tokens = tokenizer.encode(theta=theta, model_mask=mask)
    assert tokens.shape == (2, 3, 4)
    np.testing.assert_array_equal(tokens[~mask], 0.0)
    decoded = tokenizer.decode(tokens, model_mask=mask)
    assert decoded.shape == theta.shape
    np.testing.assert_array_equal(decoded[~mask], 0.0)
    assert np.any(np.asarray(decoded[mask]) != 0)

    graph, parameters, state = nnx.split(tokenizer, nnx.Param, ...)
    restored = nnx.merge(graph, parameters, state, copy=True)
    np.testing.assert_array_equal(restored.encode(theta=theta, model_mask=mask), tokens)
    np.testing.assert_array_equal(restored.decode(tokens, model_mask=mask), decoded)


@pytest.mark.parametrize("argument", ["model_idx", "noise_idx"])
def test_generic_rejects_subsets_while_paper_variants_ignore_them(argument):
    mask = jnp.array([True, False, True])
    kwargs = {argument: [0]}
    generic = _make(GenericTokenizer)
    for method in (generic.embed_cfgs, generic.theta_token_mask):
        with pytest.raises(ValueError, match="full component mask"):
            method(mask, **kwargs)
    for cls in (PaperTokenizer, SymbolicPaperTokenizerPP):
        tokenizer = _make(cls)
        for method in (tokenizer.embed_cfgs, tokenizer.theta_token_mask):
            np.testing.assert_array_equal(method(mask, **kwargs), method(mask))


@pytest.mark.parametrize("cls", [GenericTokenizer, PaperTokenizer])
def test_additive_configuration_only_exposes_previous_mask_bits(cls):
    tokenizer = _make(cls)
    mask = jnp.ones(3, dtype=bool)
    tokens = tokenizer.embed_cfgs(mask)
    for index in range(3):
        changed = tokenizer.embed_cfgs(mask.at[index].set(False))
        np.testing.assert_array_equal(changed[: index + 1], tokens[: index + 1])
        assert not np.array_equal(changed[index + 1], tokens[index + 1])
        np.testing.assert_array_equal(changed[index + 2 :], tokens[index + 2 :])


def test_paper_pp_keeps_mask_independent_cfg_and_shifted_mask_features():
    tokenizer = _make(SymbolicPaperTokenizerPP)
    mask = jnp.ones(3, dtype=bool)
    cfg = tokenizer.embed_cfgs(mask)
    tokens = tokenizer.embed_model_mask(cfg, mask)
    assert tokens.shape == (4, 4)
    for index in range(3):
        changed_mask = mask.at[index].set(False)
        changed_cfg = tokenizer.embed_cfgs(changed_mask)
        np.testing.assert_array_equal(changed_cfg, cfg)
        changed = tokenizer.embed_model_mask(changed_cfg, changed_mask)
        # The source PP variant has two prefix positions before the first mask bit.
        np.testing.assert_array_equal(changed[: index + 2], tokens[: index + 2])
        if index + 2 < len(tokens):
            assert not np.array_equal(changed[index + 2], tokens[index + 2])
            np.testing.assert_array_equal(changed[index + 3 :], tokens[index + 3 :])
