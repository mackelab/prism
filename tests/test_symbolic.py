import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from omegaconf import OmegaConf

from prism.nn.embedding import TransformerEmbedding
from prism.symbolic.simulators import SimplePolynomial
from prism.train.build_model import build_model
from prism.train.build_simulator import build_simulator
from prism.train.evaluator import build_pure_eval_fns


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "simulator": {
            "model_class": "prism.symbolic.simulators.SimplePolynomial",
            "num_points": 8,
            "x_min": -1.0,
            "x_max": 1.0,
        },
        "model": {
            "name": "PrismModelConfigMaskPriorAmortized",
            "model_dim": 8,
            "init_seed": 0,
            "use_attention_mask": True,
            "inference_loss_type": "v",
            "tokenizer_cls": "prism.nn.compartment_tokenizer.CompartmentTokenizer",
            "embedding_net": {
                "name": "prism.nn.embedding.TransformerEmbeddingConfig",
                "cls": "prism.nn.embedding.TransformerEmbedding",
                "params": {
                    "input_dim": 2,
                    "num_layers": 1,
                    "num_heads": 2,
                    "attn_size": 4,
                    "widening_factor": 2,
                    "positional_encoding": False,
                },
            },
            "model_selection_net": {
                "name": "ModelSelectionAmortizedPriorConfig",
                "params": {
                    "num_layers": 1,
                    "num_heads": 2,
                    "attn_size": 4,
                    "prior_params_embed_dim": 4,
                },
            },
            "inference_net": {
                "name": "ThetaInferenceConfig",
                "params": {
                    "num_layers": 1,
                    "num_heads": 2,
                    "attn_size": 4,
                    "time_embed_dim": 4,
                },
            },
        },
    })


def test_additive_simulator_mask_and_pytree():
    theta = jnp.array([0.5, -0.4, 0.1])
    mask = jnp.array([True, False, True])
    grid = jnp.linspace(-1.0, 1.0, 8)
    model = SimplePolynomial.from_theta(theta, model_mask=mask)
    np.testing.assert_allclose(
        model.evaluate(grid), model.model_compartments[0].evaluate(grid)
    )
    leaves, structure = jax.tree.flatten(model)
    restored = jax.tree.unflatten(structure, leaves)
    np.testing.assert_array_equal(restored.model_mask, mask)
    np.testing.assert_allclose(restored.evaluate(grid), model.evaluate(grid), atol=1e-6)
    derivative = jax.grad(
        lambda value: (
            SimplePolynomial.from_theta(value, model_mask=mask).evaluate(grid).sum()
        )
    )(theta)
    assert derivative[1] == 0
    assert jnp.all(jnp.isfinite(derivative))


def test_additive_simulation_and_true_likelihood(cfg):
    sim_type, generators = build_simulator(cfg)
    data = jax.jit(jax.vmap(generators[0]))(jax.random.split(jax.random.key(3), 2))
    assert set(data) == {"model_mask", "mask_prior", "theta", "x"}
    assert data["x"].shape == (2, 8, 2)
    assert data["theta"].shape == (2, 3)
    assert jnp.all(data["model_mask"][:, -1])
    np.testing.assert_allclose(data["x"][0, :, 0], jnp.linspace(-1, 1, 8))
    model = build_model(cfg, sim_type)
    graph, params, state = nnx.split(model, nnx.Param, ...)
    evaluator = build_pure_eval_fns(graph, sim_type)
    actual = evaluator.true_loglikelihood(data)
    expected = jax.vmap(
        lambda theta, mask, x: sim_type.from_theta(
            theta, model_mask=mask
        ).log_likelihood(x[:, 0], x[:, 1])
    )(data["theta"], data["model_mask"], data["x"])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    assert evaluator.log_prob_masks(params, state, data).shape == (2,)
    assert evaluator.sample_masks(params, state, jax.random.key(1), data).shape == (
        2,
        3,
    )


@pytest.mark.parametrize("summary", [False, True])
def test_generic_embedding_preserves_batch_shape_and_projects(summary):
    encoder = TransformerEmbedding(
        nnx.Rngs(1),
        input_dim=2,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        y_seq_dim=4,
        y_glob_dim=6,
        use_global_summary_token=summary,
        positional_encoding=False,
        out_norm=True,
    )
    x = jnp.ones((2, 3, 5, 2))
    global_features, sequence = encoder(x)
    assert sequence.shape == (2, 3, 5, 4)
    assert global_features.shape == (2, 3, 6) if summary else global_features is None
    single_global, single_sequence = encoder(x[0, 0])
    assert single_sequence.shape == (5, 4)
    assert single_global.shape == (6,) if summary else single_global is None
    with pytest.raises(ValueError, match="Expected observations"):
        encoder(jnp.ones((5, 3)))


def test_generic_encoder_without_positions_is_permutation_equivariant():
    encoder = TransformerEmbedding(
        nnx.Rngs(1),
        input_dim=2,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        positional_encoding=False,
        use_global_summary_token=True,
    )
    x = jax.random.normal(jax.random.key(3), (5, 2))
    order = jnp.array([2, 0, 4, 1, 3])
    global_features, sequence = encoder(x)
    permuted_global, permuted_sequence = encoder(x[order])
    np.testing.assert_allclose(permuted_global, global_features, atol=1e-5)
    np.testing.assert_allclose(permuted_sequence, sequence[order], atol=1e-5)


def test_additive_model_loss_and_mask_causality(cfg):
    sim_type, generators = build_simulator(cfg)
    batch = jax.vmap(generators[0])(jax.random.split(jax.random.key(2), 2))
    model = build_model(cfg, sim_type)
    logits, theta = model(**batch)
    assert logits.shape == (2, 3)
    assert theta.shape == (2, 3)
    loss, gradients = nnx.value_and_grad(
        lambda net: net.loss_fn(jax.random.key(4), **batch, use_loss_mask=True).sum()
    )(model)
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
    assert any(jnp.any(leaf != 0) for leaf in jax.tree.leaves(gradients))

    model.eval()
    mask = jnp.ones((3,), dtype=bool)
    _, y = model.encoder(batch["x"][0])
    base = model.model_decoder(mask, model.tokenizer, y=y, mask_prior=jnp.array([0.5]))
    for index in range(3):
        changed = mask.at[index].set(False)
        changed_logits = model.model_decoder(
            changed, model.tokenizer, y=y, mask_prior=jnp.array([0.5])
        )
        np.testing.assert_allclose(
            changed_logits[: index + 1], base[: index + 1], atol=1e-6
        )
    # Training uses the same logits as the likelihood, including the prior context.
    mask_loss = model.loss_fn(jax.random.key(4), **batch)[0]
    expected = (
        optax
        .sigmoid_binary_cross_entropy(model(**batch)[0], batch["model_mask"])
        .sum(-1)
        .mean()
    )
    np.testing.assert_allclose(mask_loss, expected, rtol=1e-5)


def test_additive_posterior_score_and_config_validation(cfg):
    cfg.simulator.posterior_score = True
    _, generators = build_simulator(cfg)
    data = generators[0](jax.random.key(0))
    assert data["target_score"].shape == (3,)
    assert jnp.all(jnp.isfinite(data["target_score"]))
    cfg.simulator.num_points = 0
    with pytest.raises(ValueError, match="num_points"):
        build_simulator(cfg)
