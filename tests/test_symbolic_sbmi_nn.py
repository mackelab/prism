"""Neural regression against the frozen Figure 4b source, without importing it.

The fixture uses the original network at width 8 and one layer per transformer.
Its deterministic parameter perturbation makes the initially zero output layers
nonzero, so gradients and sampling exercise the complete network.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from prism.symbolic.paper.autoregressive import BinaryAutoregressiveDecoder
from prism.symbolic.paper.model_specs import SymbolicModelV0
from prism.train.build_model import build_model

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def reference():
    metadata = json.loads((FIXTURES / "symbolic_sbmi_nn.json").read_text())
    with np.load(FIXTURES / "symbolic_sbmi_nn.npz", allow_pickle=False) as arrays:
        return metadata, dict(arrays)


def _flatten(tree):
    return np.concatenate([
        np.asarray(leaf).reshape(-1) for leaf in jax.tree.leaves(tree)
    ])


@pytest.fixture(scope="module")
def network(reference):
    metadata, _ = reference
    model = build_model(OmegaConf.create(metadata["config"]), SymbolicModelV0)
    initial_params = nnx.state(model, nnx.Param)
    paths = [
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(initial_params)[0]
    ]
    leaves, structure = jax.tree.flatten(initial_params)
    perturbed = jax.tree.unflatten(
        structure,
        [
            value
            + jnp.sin(
                jnp.arange(value.size, dtype=value.dtype).reshape(value.shape) + i
            )
            * 0.015
            for i, value in enumerate(leaves)
        ],
    )
    nnx.update(model, perturbed)
    return model, initial_params, paths


@pytest.fixture(scope="module")
def batch():
    return {
        "x": jnp.sin(jnp.arange(1000, dtype=jnp.float32) * 0.031).reshape(2, 50, 10),
        "theta": jnp.linspace(-0.5, 0.5, 56).reshape(2, 28),
        "model_mask": jnp.array([[i % 3 != 0 for i in range(20)], [True] * 20]),
        "mask_prior": None,
    }


def test_symbolic_initialization_matches_frozen_source(reference, network):
    metadata, arrays = reference
    _, initial_params, paths = network
    assert paths == metadata["parameter_paths"]
    np.testing.assert_allclose(
        _flatten(initial_params), arrays["initial_params"], rtol=1e-6, atol=1e-7
    )


def test_symbolic_loss_and_all_gradients_match_frozen_source(reference, network, batch):
    metadata, arrays = reference
    model, _, _ = network
    graph, params, state = nnx.split(model, nnx.Param, ...)

    def loss(parameters):
        restored = nnx.merge(graph, parameters, state, copy=True)
        losses = restored.loss_fn(
            jax.random.key(19), **batch, use_loss_mask=True, label_smoothing=0.01
        )
        return losses.sum(), losses

    (_, losses), gradients = jax.jit(jax.value_and_grad(loss, has_aux=True))(params)
    np.testing.assert_allclose(losses, metadata["losses"], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        _flatten(gradients), arrays["gradients"], rtol=2e-4, atol=2e-5
    )


def test_symbolic_prediction_and_sampling_match_frozen_source(
    reference, network, batch
):
    metadata, _ = reference
    model, _, _ = network
    model.eval()
    x, mask, theta = batch["x"][0], batch["model_mask"][0], batch["theta"][0]

    @jax.jit
    def predict():
        # Shared evaluation passes an unused acquisition argument positionally.
        logits, denoised = model(mask, theta, None, x)
        return {
            "logits": logits,
            "denoised": denoised,
            "sampled_mask": model.sample_mask(jax.random.key(5), None, x, None),
            "sampled_theta": model.sample_theta(
                jax.random.key(6), None, x, mask, num_steps=4
            ),
            "mask_log_prob": model.log_prob_mask(mask, None, x, None),
        }

    for name, actual in predict().items():
        expected = metadata["outputs"][name]
        if name == "sampled_mask":
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prior_params_embed_dim": 4},
        {"mask_prior_dim": 1},
        {"precondition_with_mask_prior": True},
    ],
)
def test_symbolic_paper_decoder_rejects_prior_conditioning(kwargs):
    with pytest.raises(
        ValueError, match="mask_prior_dim=0 and no prior preconditioning"
    ):
        BinaryAutoregressiveDecoder(nnx.Rngs(0), **kwargs)
