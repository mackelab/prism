"""Existing checkpoint configs keep resolving after experiment support moves."""

import importlib
import pickle

import pytest
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

from experiments.dmri._shared.checkpoint import portable_config


def test_calibration_storage_import_keeps_existing_resume_scripts_working():
    old = importlib.import_module("experiments.dmri.calibration.storage")
    current = importlib.import_module("experiments.dmri.calibration.chunks")
    assert old is current
    assert get_method("experiments.dmri.calibration.storage.evaluate_sequence") is (
        current.evaluate_sequence
    )


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("model", "PaperDMRIModel"),
        ("acquisition", "random_ukb_acquisition"),
        ("acquisition", "random_hcp_acquisition"),
    ],
)
def test_saved_calibration_targets_resolve_to_the_same_implementation(module, name):
    old_module = f"experiments.dmri.calibration.{module}"
    new_module = f"experiments.dmri._shared.{module}"
    old = importlib.import_module(old_module)
    new = importlib.import_module(new_module)
    assert old is new
    resolve = get_class if module == "model" else get_method
    assert resolve(f"{old_module}.{name}") is resolve(f"{new_module}.{name}")
    saved_global = f"c{old_module}\n{name}\n.".encode()
    assert pickle.loads(saved_global) is getattr(new, name)


@pytest.mark.parametrize(
    "prefix", ["experiments.dmri_paper", "experiments.dmri.calibration"]
)
def test_historical_and_current_saved_configs_migrate_without_setting_changes(prefix):
    model_module = "runtime" if prefix.endswith("dmri_paper") else "model"
    cfg = OmegaConf.create({
        "model": {"architecture": f"{prefix}.{model_module}.PaperDMRIModel"},
        "acquisition": {"_target_": f"{prefix}.acquisition.random_hcp_acquisition"},
        "training": {"seed": 0, "learning_rate": 1e-4},
        "note": "${missing.value}",
    })
    original = OmegaConf.to_yaml(cfg, resolve=False)
    actual, replacements = portable_config(cfg)
    assert len(replacements) == 2
    assert actual.model.architecture == "experiments.dmri._shared.model.PaperDMRIModel"
    assert (
        actual.acquisition._target_
        == "experiments.dmri._shared.acquisition.random_hcp_acquisition"
    )
    assert actual.training == cfg.training
    assert OmegaConf.to_container(actual, resolve=False)["note"] == "${missing.value}"
    assert OmegaConf.to_yaml(cfg, resolve=False) == original
