"""Saved symbolic observations retain the existing training metrics."""

import logging
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from prism.train import evaluator as evaluator_module
from prism.train import simulate, train_script


@pytest.fixture
def model(monkeypatch):
    model = SimpleNamespace(
        eval=lambda: None,
        sample_theta=lambda rng, *, x, model_mask: jnp.array([2.0]),
        log_prob_mask=lambda mask, acq, x, prior: -jnp.sum(mask * prior),
        log_prob_theta=lambda theta, acq, x, mask, **kwargs: -jnp.sum(theta**2),
    )
    monkeypatch.setattr(evaluator_module.nnx, "merge", lambda *args, **kwargs: model)
    return model


@pytest.fixture
def data():
    grid = jnp.arange(6.0)
    return {
        "theta": jnp.array([[2.0], [3.0]]),
        "model_mask": jnp.ones((2, 1), dtype=bool),
        "mask_prior": jnp.array([[0.2], [0.8]]),
        "x": jnp.stack([2 * grid + 1, 2 * grid + 2]).reshape(2, 3, 2),
    }


def test_symbolic_predictive_rmse_remains_enabled_by_default(model, data):
    class Simulator:
        @staticmethod
        def from_theta(theta, model_mask):
            return SimpleNamespace(evaluate=lambda grid, rng: theta[0] * grid)

    evaluator = evaluator_module.build_pure_eval_fns(
        None, Simulator, fixed_design=jnp.arange(6.0)
    )

    assert evaluator.predictive_rmse is not None
    assert evaluator.eval_predictive_rmse({}, {}, [data]) == pytest.approx(1.5)


def test_nll_does_not_evaluate_the_simulator(model, data):
    class Simulator:
        @staticmethod
        def from_theta(*args, **kwargs):
            raise AssertionError("Offline NLL evaluation must not invoke the simulator")

    evaluator = evaluator_module.build_pure_eval_fns(
        None, Simulator, fixed_design=jnp.arange(6.0)
    )

    for metric, expected in (("eval_nnl_mask", 0.5), ("eval_nnl_theta", 6.5)):
        actual = getattr(evaluator, metric)({}, {}, [data])
        assert actual == pytest.approx(expected)


class PredictiveTestModel(nnx.Module):
    """Keep the trainer integration test small while using the real simulator."""

    def __init__(self, simulator):
        self.cfg = "predictive evaluation test"
        self.theta = nnx.Param(jnp.zeros(simulator.theta_dim))
        self.examples_seen = nnx.Variable(jnp.asarray(0, dtype=jnp.int32))

    def loss_fn(self, rng, theta, **kwargs):
        self.examples_seen[...] += theta.shape[0]
        return jnp.array(0.0), jnp.mean((theta - self.theta[...]) ** 2)

    def log_prob_mask(self, model_mask, acq, x, mask_prior):
        return jnp.array(0.0)

    def log_prob_theta(self, theta, acq, x, model_mask, **kwargs):
        return -jnp.sum((theta - self.theta[...]) ** 2)

    def sample_theta(self, rng, *, x, model_mask):
        return self.theta[...] + jax.random.normal(rng, self.theta.shape)


def test_file_training_preserves_signal_block_predictive_rmse(
    tmp_path, monkeypatch, caplog
):
    with initialize_config_module(version_base=None, config_module="conf"):
        source = compose("train", overrides=["+experiment/train=symbolic"])
    OmegaConf.set_struct(source, False)
    source.simulator.update({
        "observation_format": "signal_blocks",
        "input_dim": 2,
        "x_step": 0.5,
    })
    source.training.update({
        "inner_steps": 1,
        "max_train_steps": 1,
        "eval_freq": 1,
        "checkpoint_freq": 1,
        "eval_at_end": True,
        "track_ema": False,
        "optimizer": {"optimizer": "adam", "scheduler": None, "learning_rate": 0.01},
    })
    source.training.dataloader.dataset.update({
        "dataset_size": 4,
        "simulation_batch_size": 2,
    })
    source.training.dataloader.val_dataset.dataset_size = 2
    for split in ("train_loader", "val_loader"):
        source.training.dataloader[split].update({
            "batch_size": 2,
            "num_prefetch_host": 1,
            "num_prefetch_device": 1,
            "loop": True,
        })
    offline = simulate.generate_datasets(
        source,
        tmp_path / "data",
        train_size=4,
        validation_size=2,
        simulation_batch_size=2,
    )
    monkeypatch.setattr(
        train_script, "build_model", lambda cfg, sim: PredictiveTestModel(sim)
    )
    caplog.set_level(logging.INFO)
    results = []
    for name, cfg in (("static", source), ("file", offline)):
        output = tmp_path / name
        output.mkdir()
        monkeypatch.setattr(
            train_script,
            "configure_environment",
            lambda config, output=output: (
                logging.getLogger(__name__),
                str(output),
                str(output),
            ),
        )
        if name == "file":

            def fail_generation(*args, **kwargs):
                raise AssertionError("File-only training built a data generator")

            monkeypatch.setattr(train_script, "build_simulator", fail_generation)
        caplog.clear()
        train_script._main.__wrapped__(cfg)
        # Read the reported evaluation result, independent of compilation warm-up.
        reported_rmse = [
            float(record.getMessage().split("Predictive RMSE: ", 1)[1])
            for record in caplog.records
            if "Predictive RMSE: " in record.getMessage()
        ]
        assert reported_rmse, "Training did not report predictive RMSE."
        assert np.isfinite(reported_rmse[-1])
        results.append(reported_rmse[-1])

    assert results[1] == results[0]
