"""Check saved-config compatibility and strict checkpoint selection on CPU."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from experiments.dmri._shared import checkpoint as loader
from prism.config import build_artifact_config, load_artifact_config
from prism.dmri.simulators.models import Ball3StickSharedDiffusivity


def test_portable_config_only_replaces_known_targets_in_memory():
    old, new = next(iter(loader.LEGACY_TARGETS.items()))
    original = OmegaConf.create({
        "model": {"_target_": old, "dim": 128},
        "generators": [{"_target_": key} for key in loader.LEGACY_TARGETS],
        "note": "${unavailable.value}",
        "other": "prism.nn.model.PrismModel",
    })
    before = OmegaConf.to_yaml(original, resolve=False)
    portable, replacements = loader.portable_config(original)
    assert portable.model._target_ == new
    assert portable.model.dim == 128
    assert portable.other == original.other
    assert (
        OmegaConf.to_container(portable, resolve=False)["note"]
        == "${unavailable.value}"
    )
    assert replacements == loader.LEGACY_TARGETS
    assert OmegaConf.to_yaml(original, resolve=False) == before


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    sim_type = Ball3StickSharedDiffusivity
    for folder in ("checkpoints/100", "checkpoints/best/80"):
        target = tmp_path / folder
        target.mkdir(parents=True)
        (target / "metadata").write_text("test checkpoint")
    cfg = OmegaConf.create({
        "simulator": {
            "model_class": "prism.dmri.simulators.models.Ball3StickSharedDiffusivity"
        },
        "note": "${unavailable.value}",
    })
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
    model.inference_decoder = SimpleNamespace(
        schedule=SimpleNamespace(t_min=1e-4, t_max=80.0)
    )
    raw = nnx.state(model, nnx.Param)
    ema = nnx.state(nnx.Linear(2, 1, rngs=nnx.Rngs(9)), nnx.Param)
    checkpoint = {"params": raw, "params_ema": ema, "val_loss": 16.42}
    calls = []

    class Manager:
        best_manager = SimpleNamespace(latest_step=lambda: 80)

        def __init__(self, path, continue_training):
            assert continue_training

        def get_latest_step(self):
            return 100

        def restore_parameters(self, **kwargs):
            calls.append(kwargs)
            return {"step": kwargs["step"], **checkpoint}

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(loader, "CheckpointManager", Manager)
    monkeypatch.setattr(loader, "load_cfg", lambda path: cfg)
    monkeypatch.setattr(loader, "load_artifact_config", lambda path, **kwargs: None)
    monkeypatch.setattr(loader, "resolve_simulator_model", lambda cfg: sim_type)
    monkeypatch.setattr(loader, "build_simulator", lambda cfg: (sim_type, [None, None]))
    monkeypatch.setattr(loader, "build_model", lambda cfg, sim: model)
    return tmp_path, model, checkpoint, calls


@pytest.mark.parametrize("which, step", [("latest", 100), ("best", 80)])
def test_loads_selected_ema_and_records_best_latest_and_loss(bundle, which, step):
    path, model, checkpoint, calls = bundle
    restored, _, generators, metadata = loader.load_run_model(path, which, step)
    assert restored is model and len(generators) == 2
    for name, parameter in nnx.state(model, nnx.Param).items():
        np.testing.assert_array_equal(
            parameter.value, checkpoint["params_ema"][name].value
        )
    assert calls[0]["from_best"] is (which == "best")
    assert calls[0]["partial_restore"] is False
    assert calls[-1] == "closed"
    assert metadata["selected_step"] == step
    assert metadata["selected_best_checkpoint_step"] == 80
    assert metadata["latest_saved_step"] == 100
    assert metadata["val_loss"] == 16.42
    assert metadata["saved_config"]["note"] == "${unavailable.value}"
    assert len(metadata["selected_checkpoint_sha256"]) == 64
    json.dumps(metadata, allow_nan=False)


def test_wrong_selected_step_stops_before_restore(bundle):
    path, _, _, calls = bundle
    with pytest.raises(ValueError, match="Expected selected step"):
        loader.load_run_model(path, "best", 100)
    assert calls == ["closed"]


def test_missing_ema_never_falls_back_to_raw(bundle):
    path, model, checkpoint, _ = bundle
    before = np.asarray(model.kernel.value).copy()
    del checkpoint["params_ema"]
    with pytest.raises(ValueError, match="no params_ema"):
        loader.load_run_model(path)
    np.testing.assert_array_equal(model.kernel.value, before)


def test_stored_step_must_match_selected_step(bundle):
    path, _, checkpoint, _ = bundle
    checkpoint["step"] = 99
    with pytest.raises(ValueError, match="Stored step differs"):
        loader.load_run_model(path)


def test_artifact_mismatch_is_rejected_before_restore(bundle, monkeypatch):
    path, _, _, calls = bundle
    monkeypatch.setattr(
        loader,
        "load_artifact_config",
        lambda path, **kwargs: OmegaConf.create({"dim": 64}),
    )
    monkeypatch.setattr(
        loader, "build_artifact_config", lambda cfg: OmegaConf.create({"dim": 128})
    )
    with pytest.raises(ValueError, match="Artifact differs"):
        loader.load_run_model(path)
    assert calls == []


@pytest.mark.parametrize("conflicting_prior", [False, True])
def test_legacy_artifact_recovers_prior_and_preserves_model_target_mapping(
    bundle, monkeypatch, conflicting_prior
):
    path, model, _, calls = bundle
    old_target = "experiments.dmri_paper.runtime.PaperDMRIModel"
    prior = {"alpha": 2.0, "beta": 3.0}
    cfg = OmegaConf.create({
        "model": {"_target_": old_target},
        "simulator": {
            "model_class": "prism.dmri.simulators.models.Ball3StickSharedDiffusivity",
            "mask_prior": prior,
        },
        "note": "${unavailable.value}",
    })
    artifact = build_artifact_config(cfg)
    del artifact.simulator.mask_prior
    if conflicting_prior:
        artifact.simulator.mask_prior = {"alpha": 7.0, "beta": 3.0}
    artifact_path = path / "artifact.yaml"
    OmegaConf.save(artifact, artifact_path)
    original_bytes = artifact_path.read_bytes()
    monkeypatch.setattr(loader, "load_cfg", lambda path: cfg)
    monkeypatch.setattr(loader, "load_artifact_config", load_artifact_config)
    monkeypatch.setattr(loader, "build_artifact_config", build_artifact_config)

    if conflicting_prior:
        with pytest.raises(ValueError, match="Artifact differs"):
            loader.load_run_model(path, "best", 80)
        assert calls == []
    else:
        restored, _, _, metadata = loader.load_run_model(path, "best", 80)
        assert restored is model
        assert metadata["selected_step"] == 80
        assert metadata["legacy_import_targets"] == {
            old_target: "experiments.dmri._shared.model.PaperDMRIModel"
        }
        assert metadata["saved_config"]["model"]["_target_"] == old_target
        assert metadata["saved_config"]["simulator"]["mask_prior"] == prior
        assert calls[-1] == "closed"
    assert artifact_path.read_bytes() == original_bytes


def test_artifact_under_former_package_root_matches_its_config(bundle, monkeypatch):
    # The public msb3s_2_4_6_128 artifact names its simulator as dmri.simulators...
    path, model, _, calls = bundle
    current = "prism.dmri.simulators.models.Ball3StickSharedDiffusivity"
    legacy = current.removeprefix("prism.")
    cfg = OmegaConf.create({
        "model": {"_target_": "experiments.dmri._shared.model.PaperDMRIModel"},
        "simulator": {"model_class": current, "mask_prior": {}},
    })
    artifact = build_artifact_config(cfg)
    artifact.simulator.model_class = legacy
    artifact_path = path / "artifact.yaml"
    OmegaConf.save(artifact, artifact_path)
    original_bytes = artifact_path.read_bytes()
    monkeypatch.setattr(loader, "load_cfg", lambda path: cfg)
    monkeypatch.setattr(loader, "load_artifact_config", load_artifact_config)
    monkeypatch.setattr(loader, "build_artifact_config", build_artifact_config)

    restored, _, _, metadata = loader.load_run_model(path, "best", 80)

    assert restored is model
    assert metadata["legacy_import_targets"] == {legacy: current}
    assert calls[-1] == "closed"
    assert artifact_path.read_bytes() == original_bytes
