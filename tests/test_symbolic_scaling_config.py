"""Symbolic scaling prescriptions and simulator/head prior agreement."""

import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_module

from prism.simulators.config import resolve_simulator_model
from prism.train.build_model import build_model


def config(components, *overrides):
    with initialize_config_module(version_base=None, config_module="conf"):
        return compose(
            "train", overrides=[f"+experiment/train=symbolic{components}", *overrides]
        )


@pytest.mark.parametrize("components", [15, 30, 50])
def test_scaling_presets_match_source_training_prescription(components):
    cfg = config(components)
    simulator = resolve_simulator_model(cfg)
    assert len(simulator.model_types) == components
    assert len(simulator.noise_types) == (5 if components == 15 else 8)
    assert cfg.run.seed == cfg.model.init_seed == 0
    assert cfg.model.model_dim == 128
    assert cfg.model.embedding_net.params.num_layers == 2
    assert cfg.model.model_selection_net.params.num_layers == 6
    assert cfg.model.inference_net.params.num_layers == 6
    assert cfg.model.model_selection_net.params.prior_params_embed_dim == 128
    assert cfg.training.dataloader.name == "SimulationDataset"
    assert cfg.training.dataloader.train_loader.batch_size == 4096
    assert cfg.training.dataloader.dataset.buffer_size == 2_097_152
    assert cfg.training.optimizer.optimizer == "radam"
    assert cfg.training.optimizer.learning_rate == 5e-4
    assert cfg.training.optimizer.gradient_clip_value == 2.0
    assert cfg.training.optimizer.scheduler is None
    assert cfg.training.ema_decay == 0.99
    assert cfg.training.eval_freq == 4000
    assert cfg.training.model_selection_warmup_steps == 0
    assert cfg.training.model_selection_ramp_steps == 0
    assert cfg.training.label_smoothing == 0
    assert not cfg.training.continue_training
    assert not cfg.tracking.enabled


@pytest.mark.parametrize("components", [15, 30, 50])
def test_model_build_uses_simulator_prior_overrides(components):
    cfg = config(
        components,
        "model.model_dim=8",
        "model.embedding_net.params.num_layers=1",
        "model.model_selection_net.params.num_layers=1",
        "model.inference_net.params.num_layers=1",
        "simulator.mask_prior.min_active_models=0",
        "simulator.mask_prior.alpha=2.0",
        "simulator.mask_prior.beta=3.0",
    )
    simulator = resolve_simulator_model(cfg)
    model = build_model(cfg, simulator)
    prior = model.model_decoder.prior_mask
    assert (prior.alpha, prior.beta, prior.min_active_models) == (2.0, 3.0, 0)
    assert model.requires_mask_prior and model.mask_prior_dim == 1
    mask = jnp.zeros(simulator.num_compartments(), dtype=bool).at[components].set(True)
    logits = model.model_decoder._precondition_logits(
        jnp.zeros(simulator.num_compartments()), mask, jnp.array([0.2])
    )
    # With no forced active component, expression prior odds are independent.
    np.testing.assert_allclose(
        logits[:components], jnp.log(0.2 / 0.8), rtol=1e-5, atol=1e-5
    )
