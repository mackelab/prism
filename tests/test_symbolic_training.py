"""Training and validation contracts required by the symbolic paper presets."""

import json
import logging
import os
import subprocess
import sys
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from omegaconf import OmegaConf

from prism.train.evaluator import build_pure_eval_fns
from prism.train.train_script import TrainState, create_dataloaders, train_loop


def test_online_training_restores_ema_and_infers_at_two_priors(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "symbolic_prior"
    process = subprocess.run(
        [
            sys.executable,
            str(repo / "tests" / "check_symbolic_prior.py"),
            "--components",
            "15",
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "WANDB_MODE": "disabled"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    (tmp_path / "symbolic_prior.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((output / "smoke.json").read_text())
    assert report["training_step"] == 4
    assert report["best_step"] in (2, 4)
    assert report["weights"] == "ema" and report["ema_restored_exactly"]
    assert report["components"] == 15
    assert report["observation_shape"] == [50, 10]
    assert [item["lambda"] for item in report["inference"]] == [0.1, 0.9]
    assert all(item["outputs_finite"] for item in report["inference"])
    assert (output / "checkpoints" / "4").is_dir()
    assert (output / "artifact.yaml").is_file()
    assert not (tmp_path / "results").exists()
    with np.load(output / "samples.npz") as samples:
        assert all(np.isfinite(samples[name]).all() for name in samples.files)


@pytest.mark.parametrize("simulation_batch_size", [3, 10000, None])
def test_fixed_datasets_preserve_seeded_train_validation_samples(simulation_batch_size):
    def simulator(key):
        return {"key": key, "x": jax.random.normal(key, (3,))}

    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "dataloader": {
                "name": "StaticDataset",
                "dataset": {
                    "dataset_size": 8,
                    "simulation_batch_size": simulation_batch_size,
                },
                "val_dataset": {"dataset_size": 4},
                "train_loader": {"batch_size": 2, "shuffle": True},
                "val_loader": {"batch_size": 2, "shuffle": False},
            }
        },
    })
    loaders, validation = create_dataloaders(cfg, [simulator])
    try:
        train_key, validation_key = jax.random.split(jax.random.PRNGKey(0))
        for loader, key, count in (
            (loaders[0], train_key, 8),
            (validation, validation_key, 4),
        ):
            expected = jax.vmap(simulator)(jax.random.split(key, count))
            actual = loader._ds[:]
            np.testing.assert_array_equal(actual["key"], expected["key"])
            np.testing.assert_array_equal(actual["x"], expected["x"])
    finally:
        for loader in [*loaders, validation]:
            loader.close()


def test_signal_block_likelihood_uses_full_grid_and_flattened_signal():
    class Simulator:
        @staticmethod
        def from_theta(theta, model_mask):
            return SimpleNamespace(
                log_likelihood=lambda grid, signal: (
                    -jnp.sum((signal - theta[0] * grid) ** 2)
                )
            )

    grid = jnp.arange(6.0)
    evaluator = build_pure_eval_fns(None, Simulator, fixed_design=grid)
    data = {
        "theta": jnp.array([[2.0], [3.0]]),
        "model_mask": jnp.ones((2, 1), dtype=bool),
        "x": jnp.stack([2 * grid + 1, 3 * grid + 2]).reshape(2, 3, 2),
    }
    np.testing.assert_array_equal(evaluator.true_loglikelihood(data), [-6.0, -24.0])


@pytest.mark.parametrize("eval_at_end", [False, True])
def test_predictive_batch_sequence_and_terminal_checkpoint_selection(eval_at_end):
    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "inner_steps": 1,
            "eval_at_end": eval_at_end,
            "eval": {"nnl_mask": {"iters": 1}, "nnl_theta": {"iters": 1}},
        },
    })
    calls = []
    saves = []

    def evaluate(name, value):
        def metric(params, state, loader, iters):
            calls.append((name, next(loader)))
            return value

        return metric

    evaluator = SimpleNamespace(
        eval_nnl_mask=evaluate("mask", 2.0),
        eval_nnl_theta=evaluate("theta", 3.0),
        predictive_rmse=True,
        eval_predictive_rmse=evaluate("predictive", 99.0),
    )
    manager = SimpleNamespace(
        keep_best=True,
        save=lambda **kwargs: saves.append(kwargs),
        should_recover=lambda metric: False,
        wait_until_finished=lambda: None,
    )

    def update(params, state, optimizer, ema, batch, rng, step):
        return params + 1, state, optimizer, ema, (1.0, 1.0), 2.0

    train_loop(
        cfg,
        logging.getLogger(__name__),
        TrainState(jnp.array(0), {}, {}, None, jax.random.key(0)),
        optax.sgd(0.1),
        update,
        manager,
        evaluator,
        [cycle([{}])],
        iter(range(9)),
        track_ema=False,
        ema_transform=None,
        checkpoint_freq=1,
        eval_freq=1,
        restart_every=None,
        max_train_hours=float("inf"),
        wandb_active=False,
        max_train_steps=2,
    )
    expected = [
        ("mask", 0),
        ("theta", 1),
        ("predictive", 2),
        ("mask", 3),
        ("theta", 4),
        ("predictive", 5),
    ]
    if eval_at_end:
        expected.extend([("mask", 6), ("theta", 7), ("predictive", 8)])
    assert calls == expected
    selected = [saved for saved in saves if not saved["write_standard"]]
    assert len(selected) == (2 if eval_at_end else 1)
    assert selected[-1]["step"] == (2 if eval_at_end else 1)
    assert all(saved["val_loss"] == 5.0 for saved in selected)
    assert saves[-1]["write_standard"] and saves[-1]["step"] == 2
