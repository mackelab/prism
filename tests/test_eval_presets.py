import logging
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from prism.config import normalize_config, runtime_config
from prism.dmri.eval.eval_script import _restrict_brain_mask_to_slice

PRESET_DIR = Path(__file__).resolve().parents[1] / "conf" / "experiment" / "eval"
PRESETS = sorted(path.stem for path in PRESET_DIR.glob("*.yaml"))


def compose_preset(preset):
    with initialize_config_module(version_base=None, config_module="conf"):
        return compose(config_name="eval", overrides=[f"+experiment/eval={preset}"])


@pytest.mark.parametrize("preset", PRESETS)
def test_eval_presets_are_canonical_and_reach_runtime(preset):
    composed = compose_preset(preset)
    canonical = normalize_config(composed, "eval")
    runtime = runtime_config(canonical, "eval")

    # Current presets should need no migration. Resolve neither Hydra's runtime
    # directory nor file inputs: this exercises configuration without loading data.
    assert OmegaConf.to_container(canonical, resolve=False) == OmegaConf.to_container(
        composed, resolve=False
    )
    assert canonical.schema_version == 2
    for runtime_key, config in (
        ("data", canonical.evaluation.input),
        ("mask_sample", canonical.evaluation.sampling.mask),
        ("theta_sample", canonical.evaluation.sampling.theta),
        ("model_selection", canonical.evaluation.selection),
        ("export", canonical.evaluation.export.theta),
        ("export_model_selection", canonical.evaluation.export.model_selection),
    ):
        assert OmegaConf.to_container(runtime[runtime_key], resolve=False) == (
            OmegaConf.to_container(config, resolve=False)
        )
    for key, value in canonical.evaluation.pipeline.items():
        assert runtime[key] == value
    assert runtime.seed == canonical.run.seed


@pytest.mark.parametrize(
    "preset,default_mask,source,num_steps,batch_size,export_name",
    [
        (
            "eval_ground_truth_b1s",
            [True, True, False, False, True],
            "file",
            64,
            1024,
            "ground_truth_inference_results_b1s",
        ),
        (
            "eval_ground_truth_b2s",
            [True, True, True, False, True],
            "file",
            64,
            1024,
            "ground_truth_inference_results_b2s",
        ),
        (
            "eval_ground_truth_b3s",
            [True, True, True, True, True],
            "file",
            64,
            1024,
            "ground_truth_inference_results_b3s",
        ),
        (
            "eval_ground_truth_b3s_only_smc",
            [True, True, True, True, True],
            "file",
            0,
            3841,
            "ground_truth_inference_results_b3s_only_smc",
        ),
        (
            "eval_test",
            [True, True, True, True, True],
            "file",
            64,
            1024,
            "test_real",
        ),
        (
            "eval_test_syn",
            [True, True, True, True, True],
            "synthetic",
            64,
            1024,
            "test",
        ),
    ],
)
def test_fixed_mask_eval_presets_preserve_scientific_settings(
    preset, default_mask, source, num_steps, batch_size, export_name
):
    runtime = runtime_config(compose_preset(preset), "eval")

    assert runtime.seed == 42424242
    assert runtime.sample_mask is False
    assert runtime.select_models is False
    assert runtime.sample_theta is True
    assert runtime.default_mask == default_mask
    assert runtime.data.source == source
    assert runtime.theta_sample.num_samples == 50
    assert runtime.theta_sample.eval_batch_size == batch_size
    assert runtime.theta_sample.params.num_steps == num_steps
    assert runtime.theta_sample.params.t_max == 80
    assert runtime.theta_sample.corrector.name == "smc_corrected"
    assert runtime.theta_sample.corrector.lam_start == (0.0 if num_steps == 0 else 0.9)
    assert runtime.export.type == "raw"
    assert runtime.export.name == export_name
    if source == "synthetic":
        assert runtime.data.get("slice") is None
        assert runtime.data.use_true_model_mask_for_synthetic is True
        assert runtime.data.num_voxels == 512
    else:
        assert runtime.data.slice == {"x": None, "y": None, "z": 52}


def test_synthetic_eval_preset_selects_the_whole_generated_volume():
    runtime = runtime_config(compose_preset("eval_test_syn"), "eval")
    # Synthetic data uses (num_voxels, 1, 1), so a real-volume z=52 slice fails
    # before inference. Exercise the runtime slice resolver without simulation.
    brain_mask = np.ones((runtime.data.num_voxels, 1, 1), dtype=bool)

    selected_mask, volume_slice = _restrict_brain_mask_to_slice(
        runtime.data.get("slice"), brain_mask, logging.getLogger(__name__)
    )

    assert selected_mask.shape == (512, 1, 1)
    assert np.count_nonzero(selected_mask) == 512
    assert volume_slice is None


@pytest.mark.parametrize(
    "preset,selection,sample_mask,p_mask,export_name",
    [
        (
            "eval_b3s_average_model_selection",
            "average",
            True,
            0.35,
            "ball3stick_inference_results_average_model_selection",
        ),
        (
            "eval_b3s_best_model_selection",
            "best",
            True,
            0.35,
            "ball3stick_inference_results_best_model_selection",
        ),
        (
            "eval_b3s_no_selection",
            "none",
            False,
            0.3,
            "ball3stick_inference_results_no_selection",
        ),
    ],
)
def test_model_selection_eval_presets_preserve_pipeline_settings(
    preset, selection, sample_mask, p_mask, export_name
):
    runtime = runtime_config(compose_preset(preset), "eval")

    assert runtime.seed == 1
    assert runtime.model_selection.name == selection
    assert runtime.sample_mask is sample_mask
    assert runtime.select_models is True
    assert runtime.sample_theta is True
    assert runtime.default_mask is None
    assert runtime.data.source == "file"
    assert runtime.data.slice == {"x": None, "y": None, "z": None}
    assert runtime.mask_sample.p_mask == p_mask
    assert runtime.theta_sample.num_samples == 50
    assert runtime.theta_sample.params.num_steps == 40
    assert runtime.theta_sample.corrector.name == "auto"
    assert runtime.export.type == "ball3stick"
    assert runtime.export.name == export_name
    if selection == "average":
        assert runtime.export.metrics.ksd.enabled is False
