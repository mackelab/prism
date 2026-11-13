"""Lambda-conditioned head regression against the original symbolic code.

The source decoder was evaluated with the already source-checked symbolic
tokenizer and an analytic causal prior. Parameter perturbations expose gradients
through context projections whose initialization would otherwise hide changes.
"""

import json
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from omegaconf import OmegaConf

from prism.symbolic.paper.autoregressive import (
    PriorConditionedBinaryAutoregressiveDecoder,
)
from prism.symbolic.paper.model_specs import SymbolicModelV50
from prism.symbolic.paper.tokenizer import SymbolicPaperTokenizerPP
from prism.symbolic.simulators import SimplePolynomial
from prism.train.build_model import build_model

FIXTURES = Path(__file__).parent / "fixtures"
HEAD_KWARGS = dict(
    model_dim=8,
    num_heads=2,
    num_layers=1,
    widening_factor=2,
    attn_size=4,
    prior_params_embed_dim=4,
    mask_prior_dim=1,
    additional_context_dim=3,
    kv_in_features=6,
)


class CausalTestPrior:
    """An analytically normalized prior with lambda and prefix dependence."""

    def log_prob_autoregressive(self, model_mask, hyperparameters):
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        prefix_count = jnp.cumsum(mask) - mask
        logits = 0.6 * prefix_count - hyperparameters[0] * (
            jnp.arange(mask.shape[-1]) + 1
        )
        return -optax.sigmoid_binary_cross_entropy(logits, mask)


def _flatten(tree):
    return np.concatenate([
        np.asarray(leaf).reshape(-1) for leaf in jax.tree.leaves(tree)
    ])


def _perturb(model):
    parameters = nnx.state(model, nnx.Param)
    leaves, structure = jax.tree.flatten(parameters)
    perturbed = jax.tree.unflatten(
        structure,
        [
            value + jnp.sin(jnp.arange(value.size).reshape(value.shape) + index) * 0.015
            for index, value in enumerate(leaves)
        ],
    )
    nnx.update(model, perturbed)
    return parameters


def _make_batch():
    return dict(
        model_mask=jnp.array([[True, False, True], [False, True, True]]),
        y=jnp.sin(jnp.arange(48, dtype=jnp.float32) * 0.13).reshape(2, 4, 6),
        mask_prior=jnp.array([[0.2], [1.4]]),
        additional_context=jnp.linspace(-0.4, 0.7, 6).reshape(2, 3),
    )


@pytest.fixture(scope="module")
def reference():
    metadata = json.loads((FIXTURES / "symbolic_prior_head.json").read_text())
    with np.load(FIXTURES / "symbolic_prior_head.npz", allow_pickle=False) as arrays:
        return metadata, dict(arrays)


@pytest.fixture(scope="module")
def head():
    rngs = nnx.Rngs(11)
    decoder = PriorConditionedBinaryAutoregressiveDecoder(
        rngs, prior_mask=CausalTestPrior(), **HEAD_KWARGS
    )
    initial_parameters = _perturb(decoder)
    decoder.eval()
    tokenizer = SymbolicPaperTokenizerPP(
        SimplePolynomial, token_dim=8, rngs=nnx.Rngs(12)
    )
    return decoder, tokenizer, initial_parameters, int(rngs.default.count.value)


def test_conditioned_initialization_and_rng_match_frozen_source(head, reference):
    _, _, initial_parameters, rng_count = head
    metadata, arrays = reference
    paths = [
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(initial_parameters)[0]
    ]
    assert paths == metadata["parameter_paths"]
    assert any("mask_prior_embed" in path for path in paths)
    np.testing.assert_allclose(
        _flatten(initial_parameters), arrays["initial_params"], rtol=1e-6, atol=1e-7
    )
    assert rng_count == metadata["rng_count"]


def test_conditioned_forward_loss_and_all_gradients_match_frozen_source(
    head, reference
):
    decoder, tokenizer, _, _ = head
    _, arrays = reference
    batch = _make_batch()
    graph, parameters, state = nnx.split(decoder, nnx.Param, ...)

    def loss(params):
        restored = nnx.merge(graph, params, state, copy=True)
        return restored.loss_fn(tokenizer=tokenizer, label_smoothing=0.01, **batch)

    loss_value, gradients = jax.jit(jax.value_and_grad(loss))(parameters)
    logits = jax.jit(lambda: decoder(tokenizer=tokenizer, **batch))()
    log_prob = jax.jit(lambda: decoder.log_prob(tokenizer=tokenizer, **batch))()
    np.testing.assert_allclose(logits, arrays["logits"], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(log_prob, arrays["log_prob"], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(loss_value, arrays["loss"], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        _flatten(gradients), arrays["gradients"], rtol=2e-4, atol=2e-5
    )
    assert float(jnp.linalg.norm(gradients["mask_prior_embed"]["P"].value)) > 1e-5


def test_sampling_matches_source_with_temperature_and_prior(head, reference):
    decoder, tokenizer, _, _ = head
    _, arrays = reference
    batch = {name: value[0] for name, value in _make_batch().items()}
    batch.pop("model_mask")

    @jax.jit
    def sample():
        return jax.vmap(
            lambda key: decoder.sample(
                key,
                tokenizer,
                dim=3,
                temperature=jnp.array([0.3, 1.2, 2.5]),
                **batch,
            )
        )(jax.random.split(jax.random.key(5), 12))

    np.testing.assert_array_equal(sample(), arrays["samples"])


def test_lambda_changes_learned_context_and_preconditioned_distribution(head):
    decoder, tokenizer, _, _ = head
    masks = jnp.array(list(product([False, True], repeat=3)))
    batch = _make_batch()

    @jax.jit
    def evaluate(prior):
        kwargs = dict(
            y=batch["y"][0],
            additional_context=batch["additional_context"][0],
            mask_prior=jnp.array([prior]),
        )
        logits = jax.vmap(lambda mask: decoder(mask, tokenizer, **kwargs))(masks)
        log_prob = jax.vmap(lambda mask: decoder.log_prob(mask, tokenizer, **kwargs))(
            masks
        )
        return logits, jnp.exp(log_prob)

    logits_low, probability_low = evaluate(0.2)
    logits_high, probability_high = evaluate(1.4)
    assert float(jnp.max(jnp.abs(logits_low - logits_high))) > 1e-3
    np.testing.assert_allclose(probability_low.sum(), 1, atol=2e-6)
    np.testing.assert_allclose(probability_high.sum(), 1, atol=2e-6)
    complexity = masks.sum(-1)
    assert float(probability_low @ complexity - probability_high @ complexity) > 0.1
    # Future bits cannot alter either learned or preconditioned conditionals.
    for position in range(3):
        for prefix in product([False, True], repeat=position):
            same_prefix = np.all(np.asarray(masks)[:, :position] == prefix, axis=1)
            selected = np.asarray(logits_low)[same_prefix, position]
            np.testing.assert_allclose(selected, selected[0], atol=1e-6)


def test_prior_context_broadcasting_and_required_lambda(head):
    decoder, _, _, _ = head
    context = jnp.array([0.1, 0.2, 0.3])
    expected = decoder._prepare_context(
        (2,), jnp.float32, jnp.array([[0.4], [0.4]]), context
    )
    for prior in (0.4, jnp.array([0.4]), jnp.array([0.4, 0.4])):
        actual = decoder._prepare_context((2,), jnp.float32, prior, context)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
    with pytest.raises(ValueError, match="mask_prior must be provided"):
        decoder._prepare_context((2,), jnp.float32, None, context)
    with pytest.raises(ValueError, match="Mask prior dimensionality"):
        decoder._prepare_context((2,), jnp.float32, jnp.ones((2, 2)), context)


def test_conditioning_without_preconditioning_uses_learned_logits(head):
    decoder, tokenizer, _, _ = head
    conditioned = PriorConditionedBinaryAutoregressiveDecoder(
        nnx.Rngs(11), **HEAD_KWARGS
    )
    _perturb(conditioned)
    conditioned.eval()
    batch = _make_batch()

    @jax.jit
    def evaluate():
        logits = conditioned(tokenizer=tokenizer, **batch)
        return (
            logits,
            decoder(tokenizer=tokenizer, **batch),
            conditioned.loss_fn(tokenizer=tokenizer, **batch),
            conditioned.log_prob(tokenizer=tokenizer, **batch),
        )

    logits, preconditioned_forward, loss, log_prob = evaluate()
    np.testing.assert_array_equal(logits, preconditioned_forward)
    expected_nll = optax.sigmoid_binary_cross_entropy(logits, batch["model_mask"]).sum(
        -1
    )
    np.testing.assert_allclose(loss, expected_nll.mean(), atol=1e-6)
    np.testing.assert_allclose(log_prob, -expected_nll, atol=1e-6)


def test_preconditioning_clips_impossible_and_extreme_prior_odds(head):
    class LimitingPrior:
        def log_prob_autoregressive(self, model_mask, hyperparameters):
            logp0 = jnp.array([-jnp.inf, 0.0, -jnp.inf, -100.0, 0.0])
            logp1 = jnp.array([0.0, -jnp.inf, -jnp.inf, 0.0, -100.0])
            return jnp.where(model_mask, logp1, logp0)

    decoder = nnx.clone(head[0])
    decoder.prior_mask = LimitingPrior()
    logits = jnp.zeros((2, 5))
    actual = jax.jit(
        lambda: decoder._precondition_logits(logits, logits.astype(bool), 0.4)
    )()
    expected = jnp.broadcast_to(jnp.array([50, -50, 0, 50, -50]), (2, 5))
    np.testing.assert_array_equal(actual, expected)


@pytest.fixture(scope="module")
def full_k50_reference():
    metadata = json.loads((FIXTURES / "symbolic_k50_model.json").read_text())
    with np.load(FIXTURES / "symbolic_k50_model.npz", allow_pickle=False) as arrays:
        return metadata, dict(arrays)


@pytest.fixture(scope="module")
def full_k50_model(full_k50_reference):
    metadata, _ = full_k50_reference
    model = build_model(OmegaConf.create(metadata["config"]), SymbolicModelV50)
    initial_parameters = _perturb(model)
    return model, initial_parameters


def _full_k50_batch(arrays):
    return {
        name.removeprefix("batch_"): jnp.asarray(value)
        for name, value in arrays.items()
        if name.startswith("batch_")
    }


def test_full_k50_initialization_and_wiring_match_frozen_source(
    full_k50_model, full_k50_reference
):
    model, initial_parameters = full_k50_model
    metadata, arrays = full_k50_reference
    paths = [
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(initial_parameters)[0]
    ]
    assert paths == metadata["parameter_paths"]
    np.testing.assert_allclose(
        _flatten(initial_parameters), arrays["initial_params"], rtol=1e-6, atol=1e-7
    )
    assert int(model.encoder._rngs.default.count.value) == metadata["rng_count"]
    assert model.requires_mask_prior and model.mask_prior_dim == 1
    assert isinstance(model.model_decoder, PriorConditionedBinaryAutoregressiveDecoder)
    assert model.model_decoder.prior_mask.num_model_components == 50
    assert model.model_decoder.prior_mask.num_noise_components == 8
    assert model.cfg.simulator.theta_dim == metadata["theta_dim"] == 115
    assert model.y_ctx_dim == 0


def test_full_k50_training_losses_and_all_gradients_match_frozen_source(
    full_k50_model, full_k50_reference
):
    model, _ = full_k50_model
    _, arrays = full_k50_reference
    batch = _full_k50_batch(arrays)
    graph, parameters, state = nnx.split(model, nnx.Param, ...)

    def loss(params):
        restored = nnx.merge(graph, params, state, copy=True)
        losses = restored.loss_fn(
            jax.random.key(19), **batch, use_loss_mask=True, label_smoothing=0.0
        )
        return losses.sum(), losses

    (_, losses), gradients = jax.jit(jax.value_and_grad(loss, has_aux=True))(parameters)
    np.testing.assert_allclose(losses, arrays["losses"], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        _flatten(gradients), arrays["gradients"], rtol=2e-4, atol=2e-5
    )
    for module in ("encoder", "tokenizer", "model_decoder", "inference_decoder"):
        assert np.linalg.norm(_flatten(gradients[module])) > 1e-4


def test_full_k50_forward_and_theta_conditioning_match_frozen_source(
    full_k50_model, full_k50_reference
):
    model, _ = full_k50_model
    _, arrays = full_k50_reference
    batch = _full_k50_batch(arrays)
    model.eval()

    @jax.jit
    def predict():
        logits, theta_pred = model(**batch)
        swapped_logits, swapped_theta = model(
            **dict(batch, mask_prior=batch["mask_prior"][::-1])
        )
        log_prob = model.log_prob_mask(
            batch["model_mask"], x=batch["x"], mask_prior=batch["mask_prior"]
        )
        return dict(
            logits=logits,
            theta_pred=theta_pred,
            swapped_logits=swapped_logits,
            swapped_theta=swapped_theta,
            log_prob=log_prob,
        )

    actual = predict()
    for name, value in actual.items():
        np.testing.assert_allclose(value, arrays[name], rtol=2e-4, atol=2e-5)
    assert float(jnp.max(jnp.abs(actual["logits"] - actual["swapped_logits"]))) > 1e-3
    # Source theta inference conditions on observations and mask, not lambda.
    np.testing.assert_array_equal(actual["theta_pred"], actual["swapped_theta"])
