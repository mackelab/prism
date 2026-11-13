"""Figure 4b configuration and evaluation protocol regression checks."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.symbolic.simulation_efficiency import evaluate as fig4b

ROOT = Path(__file__).resolve().parents[1]


def config(point="10k"):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        return compose(
            config_name="train",
            overrides=[f"+experiment/train=symbolic_sbmi_{point}"],
        )


@pytest.mark.parametrize(
    ("point", "size", "steps"),
    [("10k", 10000, 50000), ("25k", 25000, 100000), ("50k", 50000, 200000)],
)
def test_paper_presets_preserve_frozen_scientific_settings(point, size, steps):
    cfg = config(point)
    assert cfg.schema_version == 2
    assert cfg.run.seed == cfg.model.init_seed == 0
    assert cfg.training.max_train_steps == steps
    assert cfg.training.dataloader.dataset.dataset_size == size
    assert cfg.training.dataloader.dataset.simulation_batch_size == 10000
    assert cfg.training.dataloader.val_dataset.dataset_size == 1024
    assert cfg.training.dataloader.val_dataset.simulation_batch_size is None
    assert cfg.training.dataloader.train_loader.batch_size == 128
    assert cfg.training.dataloader.val_loader.batch_size == 128
    assert cfg.training.dataloader.train_loader.num_prefetch_host == 8
    assert cfg.training.dataloader.val_loader.drop_last is False
    assert cfg.model.model_dim == 64
    assert [
        cfg.model.embedding_net.params.num_layers,
        cfg.model.model_selection_net.params.num_layers,
        cfg.model.inference_net.params.num_layers,
    ] == [2, 4, 4]
    assert cfg.model.embedding_net.cls_params.input_dim == 10
    assert cfg.model.model_selection_net.params.mask_prior_dim == 0
    assert cfg.model.tokenizer_cls.endswith(".SymbolicPaperTokenizerPP")
    assert cfg.model.architecture == "prism.symbolic.paper.model.SymbolicPaperModel"
    assert cfg.training.optimizer.optimizer == "radam"
    assert cfg.training.optimizer.learning_rate == 1e-4
    assert cfg.training.optimizer.gradient_clip_value == 2.0
    assert cfg.training.label_smoothing == 0.0
    assert cfg.training.ema_decay == 0.999
    assert cfg.training.checkpoint_freq == cfg.training.eval_freq == 2000
    assert cfg.training.eval_at_end is False
    assert cfg.training.model_selection_warmup_steps == 15000
    assert cfg.training.model_selection_ramp_steps == 15000
    assert cfg.training.eval.ksd.iters == 0
    assert cfg.simulator.observation_format == "signal_blocks"
    assert cfg.simulator.x_step == 0.02
    assert cfg.tracking.enabled is False
    assert cfg.tracking.wandb.entity is None
    fig4b.validate_fig4b(cfg, point, steps)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("run.seed", 1),
        ("model.init_seed", 1),
        ("training.label_smoothing", 0.01),
        ("training.eval_at_end", True),
        ("training.dataloader.dataset.simulation_batch_size", 1024),
        ("model.tokenizer_cls", "prism.nn.compartment_tokenizer.CompartmentTokenizer"),
        ("training.optimizer.optimizer", "adamw"),
        ("training.model_selection_warmup_steps", 0),
    ],
)
def test_protocol_rejects_scientific_config_drift(path, value):
    cfg = config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(RuntimeError, match="config mismatch"):
        fig4b.validate_fig4b(cfg, "10k", 50000)


def test_protocol_checks_stored_completed_step_and_preserves_unresolved_metadata():
    cfg = config()
    with pytest.raises(RuntimeError, match="incomplete or over budget"):
        fig4b.validate_fig4b(cfg, "10k", 49999)
    cfg.training.max_train_hours = 96
    cfg.training.continue_training = True
    OmegaConf.update(cfg, "analysis_note", "${not_defined}", force_add=True)
    snapshot = OmegaConf.to_container(cfg, resolve=False)
    assert json.loads(json.dumps(snapshot))["analysis_note"] == "${not_defined}"
    fig4b.validate_fig4b(cfg, "10k", 50000)


def protocol_arguments(**overrides):
    return argparse.Namespace(**{
        "fig4b_point": "10k",
        "which": "best",
        "num_samples": 1000,
        "groups": 10,
        "seed": fig4b.NOTEBOOK_SEED,
        "lam": None,
        **overrides,
    })


def test_best_and_latest_keep_the_same_protocol_but_distinct_roles():
    assert fig4b.check_evaluation_protocol(protocol_arguments()) == "primary"
    assert (
        fig4b.check_evaluation_protocol(protocol_arguments(which="latest"))
        == "supplementary"
    )
    for change in (
        {"seed": 0},
        {"num_samples": 100},
        {"groups": 5},
        {"which": "50000"},
    ):
        with pytest.raises(ValueError, match="Figure 4b requires"):
            fig4b.check_evaluation_protocol(protocol_arguments(**change))


def test_missing_ema_fails_even_when_raw_weights_exist():
    with pytest.raises(RuntimeError, match="EMA weights are missing"):
        fig4b.require_ema({"params": {}, "step": 100})
    weights = {"value": jnp.ones((2,))}
    assert fig4b.require_ema({"params_ema": weights, "step": 100}) is weights


def test_checkpoint_metadata_distinguishes_completed_and_selected_steps(tmp_path):
    from prism.train.checkpointing import CheckpointManager

    checkpoint_root = tmp_path / "checkpoints"
    manager = CheckpointManager(str(checkpoint_root))
    weights = {"weight": jnp.array([1.0])}
    optimizer = {"momentum": jnp.array([0.0])}
    try:
        manager.save(5, weights, optimizer, loss=3.0, val_loss=2.0, params_ema=weights)
        manager.wait_until_finished()
        manager.save(7, weights, optimizer, loss=4.0, val_loss=4.0, params_ema=weights)
        manager.wait_until_finished()
    finally:
        manager.manager.close()
        manager.best_manager.close()
    assert fig4b.latest_saved_step(checkpoint_root) == 7
    assert fig4b.latest_saved_step(checkpoint_root / "best") == 5
    assert (
        fig4b.checkpoint_val_loss(checkpoint_root / "best" / "5", required=True) == 2.0
    )
    assert fig4b.checkpoint_val_loss(checkpoint_root / "7") == 4.0
    assert len(fig4b.tree_sha256(checkpoint_root / "best" / "5")) == 64
    with pytest.raises(RuntimeError, match="selection metric"):
        fig4b.checkpoint_val_loss(tmp_path / "missing", required=True)


def test_prediction_draw_matches_notebook_noise_and_rng_split():
    class Model:
        def sample_mask(self, key, *, x):
            del key, x
            return jnp.array([True])

        def sample_theta(self, key, *, x, model_mask):
            del key, x, model_mask
            return jnp.array([3.0])

    class Simulator:
        @classmethod
        def from_theta(cls, theta, *, model_mask):
            del model_mask
            result = cls()
            result.theta = theta
            return result

        def evaluate(self, grid, *, rng):
            return self.theta[0] + jax.random.normal(rng, grid.shape)

    def generate(key):
        return {"x": jax.random.normal(key, (50, 10))}

    key = jax.random.PRNGKey(fig4b.NOTEBOOK_SEED)
    key1, _, _, key4 = jax.random.split(key, 4)
    expected = jnp.sqrt(
        jnp.mean(
            (3.0 + jax.random.normal(key4, (500,)) - generate(key1)["x"].flatten()) ** 2
        )
    )
    actual = fig4b.prediction_rmse(Model(), generate, Simulator, key)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_summary_preserves_group_std_and_manual_comparison():
    rmses = jnp.arange(1000, dtype=jnp.float32) / 100 + fig4b.RMSE_GT_MEAN
    summary = fig4b.summarize(rmses, 10, "50k", "primary")
    group_means = rmses.reshape(10, 100).mean(axis=1)
    assert summary["rmse_std_over_groups"] == float(group_means.std())
    assert summary["comparison"]["assessment"] == "requires_review"
    assert summary["comparison"]["prism_criterion_met"] is False
    assert summary["comparison"]["sbmi_criterion_met"] is False
    assert (
        fig4b.assess_result(0.97, 0.97, 3.10, "supplementary")["assessment"]
        == "supplementary"
    )


def test_training_logs_cover_named_runs_and_multirun_jobs(tmp_path):
    named_log = tmp_path / "2026-09-11_12-00-00" / "train_script.log"
    named_log.parent.mkdir()
    named_log.write_text("Mask NLL: 1.0, Theta NLL: 2.0\n")
    job_log = tmp_path / "train_script.log"
    job_log.write_text("Mask NLL: 1.0, Theta NLL: 2.0\n")
    assert set(fig4b.training_log_paths(tmp_path)) == {named_log, job_log}


def test_100k_preset_matches_its_checkpoint_layout():
    # The published 100k checkpoint has six selection and four inference layers.
    cfg = config("100k")
    assert [
        cfg.model.embedding_net.params.num_layers,
        cfg.model.model_selection_net.params.num_layers,
        cfg.model.inference_net.params.num_layers,
    ] == [2, 6, 4]
    assert cfg.training.dataloader.dataset.dataset_size == 100_000
