import logging

import jax.numpy as jnp
import nibabel as nb
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from prism.config import runtime_config
from prism.dmri.eval import eval_script
from prism.dmri.eval.eval_script import (
    _find_theta_samples_file,
    _has_model_selection_to_export,
    _load_theta_samples,
    _should_compute_feasible_model_probabilities,
)
from prism.dmri.eval.export_metrics import (
    MetricSpec,
    _scatter_metric_values,
    _select_voxel_subset,
)


def test_voxel_subset_is_deterministic_and_scattered():
    spec = MetricSpec(
        key="test",
        type="ksd",
        output_filename="test.nii.gz",
        options={"voxel_subset_size": 3, "voxel_subset_seed": 4},
    )

    first = _select_voxel_subset(spec, 10)
    second = _select_voxel_subset(spec, 10)
    assert np.array_equal(first, second)
    assert first is not None

    scattered = _scatter_metric_values(
        np.arange(3, dtype=np.float32), first, num_voxels=10
    )
    assert np.array_equal(scattered[first], np.arange(3, dtype=np.float32))
    assert np.isnan(np.delete(scattered, first)).all()


def test_load_theta_samples_extracts_brain_voxels(tmp_path):
    brain_mask = np.array([True, False, True, False])
    samples = np.arange(4 * 2 * 3, dtype=np.float32).reshape(2, 2, 2, 3)
    path = tmp_path / "thetas.nii.gz"
    nb.save(nb.Nifti1Image(samples, np.eye(4)), path)

    loaded = _load_theta_samples(
        str(path),
        brain_mask,
        brain_shape=(2, 2),
        expected_num_samples=2,
        expected_theta_dim=3,
    )

    assert loaded.shape == (2, 2, 3)
    assert jnp.array_equal(loaded, samples.reshape(4, 2, 3)[brain_mask])


def test_find_theta_samples_uses_exported_raw_filename(tmp_path):
    path = tmp_path / "nested" / "raw_thetas.nii.gz"
    path.parent.mkdir()
    path.touch()

    assert _find_theta_samples_file(tmp_path) == str(path)


def test_model_selection_export_requires_generated_masks():
    assert not _has_model_selection_to_export(None, None)
    assert _has_model_selection_to_export(np.ones((1, 2)), None)
    assert _has_model_selection_to_export(None, np.ones((1, 2, 3)))


def test_feasible_probability_export_is_enabled_by_config():
    cfg = OmegaConf.create({"sample_mask": True})
    export_cfg = OmegaConf.create({
        "export_feasible_model_probabilities": True,
        "feasible_models": [[True, False]],
    })

    assert _should_compute_feasible_model_probabilities(cfg, export_cfg, "ball3stick")
    cfg.sample_mask = False
    assert not _should_compute_feasible_model_probabilities(
        cfg, export_cfg, "ball3stick"
    )


class _WeightModel(nnx.Module):
    def __init__(self, value):
        self.weight = nnx.Param(jnp.array([value], dtype=jnp.float32))
        self.training = True

    def eval(self):
        self.training = False


class _ModelLoadedError(Exception):
    pass


@pytest.fixture
def evaluate_checkpoint(tmp_path, monkeypatch):
    def evaluate(source, params_name, checkpoint):
        model = _WeightModel(-1)
        restored_from = []
        loaded = []

        def restore_local(*args, **kwargs):
            restored_from.append("local")
            return checkpoint, model, None

        def restore_hub(**kwargs):
            restored_from.append("hub")
            return checkpoint, model, None

        def capture_model(model, *args):
            loaded.append(model)
            raise _ModelLoadedError

        monkeypatch.setattr(eval_script, "load_checkpoint", restore_local)
        monkeypatch.setattr("prism.hub.load_checkpoint", restore_hub)
        monkeypatch.setattr(eval_script, "_replicate_model", capture_model)
        cfg = OmegaConf.create({
            "schema_version": 2,
            "run": {"seed": 1},
            "checkpoint": {
                "model_name": "example",
                "results_root": str(tmp_path),
                "which": "best",
                "params_name": params_name,
                "pretrained": {
                    "repo_id": "owner/repo" if source == "hub" else None,
                    "local_files_only": True,
                },
            },
            "evaluation": {"precision": "fp32"},
        })
        try:
            eval_script._run_eval_pipeline(
                runtime_config(cfg, "eval"), logging.getLogger(__name__), None, ()
            )
        except _ModelLoadedError:
            pass
        assert restored_from == [source]
        assert len(loaded) == 1
        return loaded[0]

    return evaluate


@pytest.mark.parametrize("source", ["local", "hub"])
@pytest.mark.parametrize("params_name,value", [("params", 1), ("params_ema", 2)])
def test_evaluation_applies_requested_weights(
    evaluate_checkpoint, caplog, source, params_name, value
):
    checkpoint = {
        "params": nnx.state(_WeightModel(1), nnx.Param),
        "params_ema": nnx.state(_WeightModel(2), nnx.Param),
        "step": 42,
    }
    with caplog.at_level(logging.INFO):
        model = evaluate_checkpoint(source, params_name, checkpoint)
    np.testing.assert_array_equal(model.weight.value, [value])
    assert model.training is False
    assert "42" in caplog.text and params_name in caplog.text


@pytest.mark.parametrize("source", ["local", "hub"])
@pytest.mark.parametrize("ema_is_none", [False, True])
def test_evaluation_rejects_missing_ema(evaluate_checkpoint, source, ema_is_none):
    checkpoint = {"params": nnx.state(_WeightModel(1), nnx.Param), "step": 42}
    if ema_is_none:
        checkpoint["params_ema"] = None
    with pytest.raises(ValueError, match="params_ema") as error:
        evaluate_checkpoint(source, "params_ema", checkpoint)
    assert "42" in str(error.value)


@pytest.mark.parametrize("source", ["local", "hub"])
def test_evaluation_rejects_unknown_weights(evaluate_checkpoint, source):
    checkpoint = {
        "params": nnx.state(_WeightModel(1), nnx.Param),
        "params_ema": nnx.state(_WeightModel(2), nnx.Param),
        "step": 42,
    }
    with pytest.raises(ValueError, match="params_em"):
        evaluate_checkpoint(source, "params_em", checkpoint)
