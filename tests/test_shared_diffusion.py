"""Regression coverage for the diffusion implementation shared by both variants.

The numeric references were evaluated from the pre-cleanup c02ebcee sources with
probjax 0.1.1. A deterministic parameter perturbation exercises gradients through
initially zero gates. The separate Figure 4b fixture covers its actual tokenizer,
full network parameter paths, gradients, and samples.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from prism.nn import simformer as dmri
from prism.symbolic.paper import simformer as symbolic


class LinearTokenizer:
    """Expose the diffusion network without introducing tokenizer parameters."""

    def encode(self, theta, **kwargs):
        return theta[..., None] * jnp.array([1.0, 0.4, -0.7, 1.1])

    def decode(self, values, **kwargs):
        return values[..., 0] + values[..., 1] * 0.2


def make_model(cls, loss_type):
    rngs = nnx.Rngs(7)
    model = cls(
        rngs,
        model_dim=4,
        time_embed_dim=4,
        additional_context_dim=2,
        num_heads=1,
        attn_size=4,
        num_layers=1,
        widening_factor=1,
        loss_type=loss_type,
    )
    initial = nnx.state(model, nnx.Param)
    leaves, structure = jax.tree.flatten(initial)
    perturbed = jax.tree.unflatten(
        structure,
        [
            value + jnp.sin(jnp.arange(value.size).reshape(value.shape) + i) * 0.015
            for i, value in enumerate(leaves)
        ],
    )
    nnx.update(model, perturbed)
    return model, initial, nnx.state(rngs)


def loss_and_grad(model, options):
    graph, params, state = nnx.split(model, nnx.Param, ...)

    def loss(parameters):
        restored = nnx.merge(graph, parameters, state, copy=True)
        return restored.loss(
            jax.random.key(19),
            jnp.array([[-0.5, 0.3, 1.0], [0.8, -0.2, 0.4]]),
            LinearTokenizer(),
            jnp.arange(16, dtype=jnp.float32).reshape(2, 2, 4) * 0.1,
            model_mask=jnp.array([[True, False, True], [True, True, True]]),
            tokens_cfg=jnp.ones((2, 3)),
            context=jnp.array([[0.3, -0.2], [0.2, 0.4]]),
            attention_mask=jnp.array([
                [True, True, False],
                [True, True, True],
                [False, True, True],
            ]),
            **options,
        )

    return jax.jit(jax.value_and_grad(loss))(params)


def flatten(tree):
    return jnp.concatenate([leaf.reshape(-1) for leaf in jax.tree.leaves(tree)])


def assert_equal_state(actual, expected):
    actual_leaves = jax.tree_util.tree_flatten_with_path(actual)[0]
    expected_leaves = jax.tree_util.tree_flatten_with_path(expected)[0]
    assert [jax.tree_util.keystr(path) for path, _ in actual_leaves] == [
        jax.tree_util.keystr(path) for path, _ in expected_leaves
    ]
    for (_, actual), (_, expected) in zip(actual_leaves, expected_leaves, strict=True):
        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
            actual, expected = (
                jax.random.key_data(actual),
                jax.random.key_data(expected),
            )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "loss_type,with_auxiliary_loss,expected",
    [
        ("x0", False, (5.1595158577, 16.7693767548, -2.5547237396)),
        ("x0", True, (9.6135416031, 42.3464126587, 2.1108524799)),
        ("v", False, (5.1595182419, 16.7693634033, -2.5547239780)),
        ("v", True, (9.6135425568, 42.3464126587, 2.1108522415)),
    ],
)
def test_diffusion_variants_preserve_loss_gradients_and_initialization(
    loss_type, with_auxiliary_loss, expected
):
    dmri_model, dmri_initial, dmri_rng = make_model(dmri.EDMSimformer, loss_type)
    symbolic_model, symbolic_initial, symbolic_rng = make_model(
        symbolic.EDMSimformer, loss_type
    )
    assert_equal_state(dmri_initial, symbolic_initial)
    assert_equal_state(dmri_rng, symbolic_rng)
    assert int(dmri_rng["default"]["count"].value) == 49
    initial_values = flatten(dmri_initial)
    np.testing.assert_allclose(
        (
            jnp.linalg.norm(initial_values),
            jnp.vdot(initial_values, jnp.sin(jnp.arange(initial_values.size))),
        ),
        (10.8508453369, 1.1887015104),
        rtol=2e-5,
        atol=2e-5,
    )
    assert type(dmri_model.net) is dmri.DiffusionTransformer
    assert type(symbolic_model.net) is symbolic.DiffusionTransformer
    assert dmri_model.solver_cfg.ode_method == "exp_ab2_scalarL"
    assert symbolic_model.solver_cfg.ode_method == "heun"
    assert dmri_model.train_cfg.t_min == symbolic_model.train_cfg.t_min == 2e-4

    options = {}
    if with_auxiliary_loss:
        options = {
            "target_score": jnp.array([[1.0, -0.5, 0.8], [0.6, 0.7, -0.4]]),
            "cut_off_tsm": 100.0,
            "weight_by_complexity": True,
            "loss_mask": jnp.array([[True, False, True], [False, True, True]]),
        }
    dmri_loss, dmri_gradients = loss_and_grad(dmri_model, options)
    symbolic_loss, symbolic_gradients = loss_and_grad(symbolic_model, options)
    np.testing.assert_array_equal(dmri_loss, symbolic_loss)
    assert_equal_state(dmri_gradients, symbolic_gradients)
    gradients = flatten(dmri_gradients)
    actual = (
        dmri_loss,
        jnp.linalg.norm(gradients),
        jnp.vdot(gradients, jnp.sin(jnp.arange(gradients.size))),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
