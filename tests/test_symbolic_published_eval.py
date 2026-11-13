"""Symbolic evaluators read published safetensors weights (--preset/--weights-dir)."""

import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.symbolic.classification import evaluate as classification
from experiments.symbolic.model_space_scaling import evaluate as scaling
from experiments.symbolic.simulation_efficiency import evaluate as fig4b
from prism.symbolic.pretrained import build_from_config, save_params_safetensors

CONF = Path(__file__).resolve().parents[1] / "conf"


def _bundle(root, preset, step=123):
    """Lay out a preset like the published repository (config, best/, latest/)."""
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose("train.yaml", overrides=[f"+experiment/train={preset}"])
    root.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({
            k: OmegaConf.to_container(cfg[k], resolve=True)
            for k in ("model", "simulator")
        }),
        root / "config.yaml",
    )
    model, _, _ = build_from_config(OmegaConf.load(root / "config.yaml"))
    for which in ("best", "latest"):
        (root / which).mkdir()
        save_params_safetensors(
            model,
            root / which / "params_ema.safetensors",
            {"step": str(step), "prism_preset": preset},
        )
    return root


@pytest.fixture(scope="module")
def sbmi_10k(tmp_path_factory):
    return _bundle(
        tmp_path_factory.mktemp("hub") / "symbolic_sbmi_10k", "symbolic_sbmi_10k"
    )


@pytest.fixture(scope="module")
def symbolic15(tmp_path_factory):
    return _bundle(tmp_path_factory.mktemp("hub") / "symbolic15", "symbolic15")


def test_fig4b_evaluates_published_weights(sbmi_10k, tmp_path):
    output = tmp_path / "fig4b.json"
    fig4b.main([
        "--preset", "symbolic_sbmi_10k", "--weights-dir", str(sbmi_10k),
        "--which", "best", "--num-samples", "2", "--groups", "1",
        "--output", str(output),
    ])  # fmt: skip
    result = json.loads(output.read_text())
    assert result["checkpoint"] is None
    assert result["published_weights"]["preset"] == "symbolic_sbmi_10k"
    assert result["published_weights"]["training_evidence"].startswith("not available")
    assert result["restored_step"] == 123
    assert result["fig4b_protocol_checks_passed"] is False
    assert np.isfinite(result["rmse_mean"])


def test_fig4c_evaluates_published_weights_without_a_training_section(
    symbolic15, tmp_path
):
    output = tmp_path / "fig4c.json"
    scaling.main([
        "--preset", "symbolic15", "--weights-dir", str(symbolic15),
        "--num-samples", "2", "--batch-size", "1", "--output", str(output),
    ])  # fmt: skip
    report = json.loads(output.read_text())
    assert report["published_weights"]["preset"] == "symbolic15"
    assert report["checkpoint"] is None and report["components"] == 15
    assert report["checkpoint_sha256"] == report["published_weights"]["weights_sha256"]
    assert output.with_suffix(".npz").is_file()


def test_classification_evaluates_published_weights(symbolic15, tmp_path):
    output = tmp_path / "classification.json"
    record = classification.main([
        "--preset", "symbolic15", "--weights-dir", str(symbolic15),
        "--num-classes", "5", "--num-per-class", "1", "--pred-batch-size", "5",
        "--output", str(output),
    ])  # fmt: skip
    assert record["published_weights"]["which"] == "latest"
    assert record["step"] == 123 and record["checkpoint"] is None


@pytest.mark.parametrize(
    "argv",
    [
        [],  # neither a run directory nor --preset
        ["results/run", "--preset", "symbolic15"],  # both
        ["--weights-dir", "x"],  # --weights-dir without --preset
    ],
)
def test_weights_source_must_be_exactly_one(argv, tmp_path):
    with pytest.raises(SystemExit):
        classification.main([*argv, "--output", str(tmp_path / "x.json")])


def test_fig4b_point_requires_the_matching_published_preset(tmp_path):
    with pytest.raises(SystemExit):
        fig4b.main([
            "--preset", "symbolic_sbmi_25k", "--fig4b-point", "10k",
            "--output", str(tmp_path / "x.json"),
        ])  # fmt: skip
    with pytest.raises(SystemExit):
        fig4b.main([
            "--preset", "symbolic_sbmi_100k", "--fig4b-point", "100k",
            "--completion", "c.json", "--output", str(tmp_path / "x.json"),
        ])  # fmt: skip
