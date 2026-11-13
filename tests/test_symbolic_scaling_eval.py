"""Figure 4c source RNG, conditional-mask metric and protocol validation."""

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from experiments.symbolic.model_space_scaling import evaluate as scaling
from prism.simulators.config import resolve_simulator_model


class FakeModel:
    def sample_mask(self, *args, **kwargs):
        raise AssertionError("Figure 4c must condition on the true mask")

    def sample_theta(
        self, key, *, x, model_mask, sample_method, num_steps, t_min, t_max
    ):
        assert (sample_method, num_steps, t_min, t_max) == ("ode", 64, None, None)
        return jnp.array([
            jax.random.normal(key) + x[0, 0] + 2 * model_mask[0] + model_mask[1]
        ])


class FakeSimulator:
    @classmethod
    def from_theta(cls, theta, *, model_mask):
        value = cls()
        value.theta, value.mask = theta, model_mask
        return value

    def evaluate(self, grid, *, rng):
        return self.theta[0] + self.mask[0] * grid + jax.random.normal(rng, grid.shape)


def generate(key):
    key1, key2, key3 = jax.random.split(key, 3)
    return {
        "x": jax.random.normal(key1, (50, 10)),
        "model_mask": jax.random.bernoulli(key2, shape=(2,)),
        "theta": jax.random.normal(key3, (1,)),
        "mask_prior": jnp.array([0.23]),
    }


def source_error(key, mode):
    """Independent expression of the source's two distinct RNG splits."""
    if mode == "predictive":
        key1, key2, key3 = jax.random.split(key, 3)
        data = generate(key1)
        theta = (
            jax.random.normal(key2)
            + data["x"][0, 0]
            + 2 * data["model_mask"][0]
            + data["model_mask"][1]
        )
    else:
        key1, key3 = jax.random.split(key, 2)
        data = generate(key1)
        theta = data["theta"][0]
    grid = jnp.arange(0.0, 10.0, 0.02)
    y = theta + data["model_mask"][0] * grid + jax.random.normal(key3, (500,))
    return jnp.sqrt(jnp.mean((data["x"].flatten() - y) ** 2))


@pytest.mark.parametrize("components,index", [(15, 0), (30, 1), (50, 2)])
def test_independent_jobs_preserve_source_five_checkpoint_keys(components, index):
    root_key = jax.random.split(jax.random.PRNGKey(0), 5)[index]
    np.testing.assert_array_equal(scaling.evaluation_key(components), root_key)
    keys = jax.random.split(root_key, 7)
    expected = {
        mode: np.asarray(jax.vmap(lambda key, mode=mode: source_error(key, mode))(keys))
        for mode in ("predictive", "gt")
    }
    # Both chunk sizes leave a remainder. Neither may change samples or keys.
    for batch_size in (3, 4):
        actual = scaling.evaluate_errors(
            FakeModel(), generate, FakeSimulator, components, 7, batch_size
        )
        for mode in expected:
            np.testing.assert_allclose(actual[mode], expected[mode], rtol=1e-6)
    # Replacing the split(3) GT data key with split(2) must be observable.
    assert not np.array_equal(expected["predictive"], expected["gt"])


def test_mean_rmse_subtraction_and_fixed_gt_bootstrap_match_notebook_definition():
    predictive = np.array([1.0, 3.0, 9.0, 12.0, 30.0])
    gt = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
    metrics, means = scaling.summarize(predictive, gt, bootstrap_chunk_size=2)
    assert metrics["conditional_rmse_mean"] == predictive.mean()
    assert metrics["gt_rmse_mean"] == gt.mean()
    assert metrics["excess_rmse_mean"] == (predictive - gt.mean()).mean()
    assert metrics["excess_rmse_mean"] != (
        np.sqrt(np.mean(predictive**2)) - np.sqrt(np.mean(gt**2))
    )
    rng = np.random.default_rng(0)
    expected = np.array([
        (predictive - gt.mean())[rng.integers(0, 5, size=5)].mean() for _ in range(500)
    ])
    np.testing.assert_allclose(means, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        [metrics["excess_rmse_ci95_low"], metrics["excess_rmse_ci95_high"]],
        np.percentile(expected, [2.5, 97.5]),
        rtol=0,
        atol=1e-14,
    )
    _, unchunked = scaling.summarize(predictive, gt, bootstrap_chunk_size=100)
    np.testing.assert_allclose(means, unchunked, rtol=0, atol=1e-14)


def config(components=15):
    with initialize_config_module(version_base=None, config_module="conf"):
        return compose("train", overrides=[f"+experiment/train=symbolic{components}"])


@pytest.mark.parametrize("components", [15, 30, 50])
def test_task_validation_and_reduced_network_diagnostic_metadata(components):
    cfg = config(components)
    simulator = resolve_simulator_model(cfg)
    assert scaling.validate_config(cfg, simulator) == (components, [])
    cfg.model.model_dim = 8
    cfg.training.dataloader.train_loader.batch_size = 2
    OmegaConf.update(cfg, "analysis_note", "${not_defined}", force_add=True)
    actual_k, changes = scaling.validate_config(cfg, simulator)
    assert actual_k == components
    assert {change["path"] for change in changes} == {
        "model.model_dim",
        "training.dataloader.train_loader.batch_size",
    }
    snapshot = OmegaConf.to_container(cfg, resolve=False)
    assert json.loads(json.dumps(snapshot))["analysis_note"] == "${not_defined}"


@pytest.mark.parametrize(
    "path,value",
    [
        ("simulator.x_step", 0.01),
        ("simulator.mask_prior.alpha", 2.0),
        ("simulator.mask_prior.min_active_models", 0),
        ("simulator.observation_format", "coordinate_signal"),
        ("training.ema_decay", 0.999),
        ("model.model_selection_net.params.mask_prior_dim", 0),
    ],
)
def test_task_validation_rejects_incompatible_grid_prior_or_ema(path, value):
    cfg = config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError, match="Scaling config mismatch"):
        scaling.validate_config(cfg, resolve_simulator_model(cfg))


def test_missing_ema_fails_before_inference(monkeypatch, tmp_path):
    from prism.train import utils

    cfg = config()
    simulator = resolve_simulator_model(cfg)
    monkeypatch.setattr(utils, "load_cfg", lambda path: cfg)
    monkeypatch.setattr(
        utils,
        "load_checkpoint",
        lambda *args, **kwargs: ({"params": {}, "step": 4}, FakeModel(), simulator),
    )
    monkeypatch.setattr(scaling, "run_record", lambda: {})
    with pytest.raises(ValueError, match="no params_ema"):
        scaling.main([str(tmp_path), "--output", str(tmp_path / "eval.json")])
    assert not (tmp_path / "eval.json").exists()


def test_source_sampling_defaults_are_checked():
    class EDMSolverConfig:
        ode_method = "heun"

    model = SimpleNamespace(
        inference_decoder=SimpleNamespace(
            train_cfg=SimpleNamespace(t_min=2e-4, t_max=80.0),
            solver_cfg=EDMSolverConfig(),
        )
    )
    assert scaling.sampling_settings(model)["effective_t_min"] == 2e-4
    model.inference_decoder.train_cfg.t_min = 1e-4
    with pytest.raises(ValueError, match="sampling defaults changed"):
        scaling.sampling_settings(model)


def test_cli_defaults_and_result_overwrite_guard(tmp_path):
    output = tmp_path / "result.json"
    args = scaling.parse_args([str(tmp_path), "--output", str(output)])
    assert (args.which, args.num_samples, args.batch_size) == ("latest", 2**20, 256)
    output.with_suffix(".npz").write_bytes(b"existing evidence")
    with pytest.raises(SystemExit):
        scaling.parse_args([str(tmp_path), "--output", str(output)])


def test_snapshot_manifest_is_hashed_without_resolving_metadata(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"source_commit": "known", "snapshot_id": "payload1"}))
    monkeypatch.setenv("SYMBOLIC_SCALING_MANIFEST", str(path))
    record = scaling.run_record()
    assert record["snapshot_manifest"]["sha256"] == scaling.file_sha256(path)
    assert record["snapshot_manifest"]["identifiers"] == {
        "source_commit": "known",
        "snapshot_id": "payload1",
    }


def test_nonfinite_errors_cannot_be_silently_filtered():
    with pytest.raises(ValueError, match="finite"):
        scaling.summarize(np.array([1.0, np.nan]), np.ones(2))
