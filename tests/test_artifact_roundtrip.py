import json
from contextlib import closing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from prism.config import build_artifact_config, load_artifact_config
from prism.simulators.config import resolve_simulator_model, simulator_model_config
from prism.train.build_model import build_model
from prism.train.checkpointing import CheckpointManager
from prism.train.utils import load_checkpoint


def _config():
    fixture = Path(__file__).parent / "fixtures" / "symbolic_sbmi_nn.json"
    cfg = OmegaConf.create(json.loads(fixture.read_text())["config"])
    cfg.simulator.model_class = "prism.symbolic.paper.model_specs.SymbolicModelV15"
    cfg.simulator.input_dim = 5
    cfg.simulator.mask_prior = {
        "alpha": 2.0,
        "beta": 3.0,
        "min_active_models": 0,
    }
    cfg.model.name = (
        "prism.symbolic.paper.model.SymbolicPaperModelConfigMaskPriorAmortized"
    )
    cfg.model.model_selection_net.name = "prism.symbolic.paper.autoregressive.SymbolicPaperModelSelectionAmortizedPriorConfig"
    cfg.model.model_selection_net.params.prior_params_embed_dim = 8
    cfg.model.model_selection_net.params.mask_prior_dim = 1
    cfg.model.model_selection_net.params.precondition_with_mask_prior = True
    return cfg


def _log_probs(model):
    mask = jnp.zeros(20, dtype=jnp.bool_).at[0].set(True).at[15].set(True)
    prior = jnp.array([0.2])
    observation = jnp.sin(jnp.arange(100, dtype=jnp.float32) * 0.031).reshape(20, 5)
    return np.asarray([
        model.model_decoder.prior_mask.log_prob(mask, prior),
        model.log_prob_mask(mask, x=observation, mask_prior=prior),
    ])


def test_artifact_preserves_construction_options_without_runtime_dependencies():
    cfg = _config()
    cfg.simulator.acquisitions = "${unavailable_acquisition}"
    cfg.simulator.input_dim = "${model.embedding_net.cls_params.input_dim}"
    cfg.model.embedding_net.cls_params.input_dim = 5
    cfg.training = {"output_dir": "${unavailable_output_dir}"}

    artifact = build_artifact_config(cfg)

    assert artifact.simulator == {
        "model_class": cfg.simulator.model_class,
        "input_dim": 5,
        "mask_prior": {"alpha": 2.0, "beta": 3.0, "min_active_models": 0},
    }
    assert "training" not in artifact


@pytest.mark.parametrize("artifact_kind", ["current", "early_v1", "absent"])
def test_checkpoint_restores_nondefault_encoder_and_prior(
    tmp_path, monkeypatch, artifact_kind
):
    cfg = _config()
    simulator = resolve_simulator_model(cfg)
    original = build_model(cfg, simulator)
    original.eval()
    params = nnx.state(original, nnx.Param)
    ema = jax.tree.map(lambda value: value + 0.01, params)
    nnx.update(original, ema)
    expected = _log_probs(original)
    assert np.isfinite(expected).all()
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    artifact = build_artifact_config(cfg)
    if artifact_kind == "early_v1":
        artifact.simulator = {"model_class": cfg.simulator.model_class}
    if artifact_kind != "absent":
        OmegaConf.save(artifact, tmp_path / "artifact.yaml")
    with closing(CheckpointManager(str(tmp_path / "checkpoints"))) as manager:
        manager.save(
            12,
            params,
            {"momentum": jnp.zeros(1)},
            loss=1.0,
            val_loss=0.5,
            params_ema=ema,
        )

    readers = []

    def reader(**kwargs):
        manager = CheckpointManager(**kwargs)
        readers.append(manager)
        return manager

    monkeypatch.setattr("prism.train.utils.CheckpointManager", reader)
    try:
        checkpoint, restored, restored_simulator = load_checkpoint(
            tmp_path, which="best"
        )
        nnx.update(restored, checkpoint["params_ema"])
        assert checkpoint["step"] == 12
        assert restored_simulator is simulator
        assert restored.encoder.input_dim == 5
        assert restored.encoder.input_projection.kernel.shape == (
            original.encoder.input_projection.kernel.shape
        )
        for actual, saved in zip(
            jax.tree.leaves(nnx.state(restored, nnx.Param)),
            jax.tree.leaves(ema),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, saved)
        np.testing.assert_array_equal(_log_probs(restored), expected)
    finally:
        for manager in readers:
            manager.close()


def test_early_artifact_recovers_legacy_prior_aliases_without_modifying_source(
    tmp_path,
):
    cfg = _config()
    del cfg.simulator.mask_prior
    cfg.simulator.prior_mask_alpha = 2.0
    cfg.simulator.prior_mask_beta = 3.0
    artifact = build_artifact_config(cfg)
    assert artifact.simulator.mask_prior == {"alpha": 2.0, "beta": 3.0}
    artifact.simulator = {"sim_type": {"name": cfg.simulator.model_class}}
    path = tmp_path / "artifact.yaml"
    OmegaConf.save(artifact, path)
    original_bytes = path.read_bytes()

    recovered = load_artifact_config(tmp_path, cfg=cfg)

    assert recovered.simulator.input_dim == 5
    assert recovered.simulator.mask_prior == {"alpha": 2.0, "beta": 3.0}
    assert resolve_simulator_model(recovered) is resolve_simulator_model(cfg)
    assert path.read_bytes() == original_bytes
    assert load_artifact_config(tmp_path) == artifact


@pytest.mark.parametrize("prior", [{}, {"alpha": 7.0}])
def test_explicit_artifact_options_take_precedence(tmp_path, prior):
    cfg = _config()
    artifact = build_artifact_config(cfg)
    artifact.simulator.input_dim = 10
    artifact.simulator.mask_prior = prior
    OmegaConf.save(artifact, tmp_path / "artifact.yaml")

    restored = load_artifact_config(tmp_path, cfg=cfg)

    assert restored == artifact


def test_complete_artifact_does_not_resolve_overridden_saved_options(tmp_path):
    cfg = _config()
    artifact = build_artifact_config(cfg)
    artifact.simulator.mask_prior = {}
    OmegaConf.save(artifact, tmp_path / "artifact.yaml")
    cfg.simulator.input_dim = "${unavailable_dimension}"
    cfg.simulator.mask_prior = "${unavailable_prior}"

    assert load_artifact_config(tmp_path, cfg=cfg) == artifact


@pytest.mark.parametrize("missing_field", ["input_dim", "mask_prior"])
def test_partial_recovery_only_resolves_the_missing_saved_option(
    tmp_path, missing_field
):
    cfg = _config()
    expected = build_artifact_config(cfg)
    artifact = OmegaConf.create(OmegaConf.to_container(expected, resolve=False))
    del artifact.simulator[missing_field]
    OmegaConf.save(artifact, tmp_path / "artifact.yaml")
    overridden_field = "mask_prior" if missing_field == "input_dim" else "input_dim"
    cfg.simulator[overridden_field] = "${unavailable_override}"

    assert load_artifact_config(tmp_path, cfg=cfg) == expected


def test_explicit_legacy_prior_alias_ignores_unresolved_saved_prior(tmp_path):
    cfg = _config()
    artifact = build_artifact_config(cfg)
    del artifact.simulator.mask_prior
    artifact.simulator.prior_mask_alpha = 4.0
    OmegaConf.save(artifact, tmp_path / "artifact.yaml")
    cfg.simulator.mask_prior = "${unavailable_prior}"

    assert load_artifact_config(tmp_path, cfg=cfg) == artifact


def test_omitted_options_cannot_be_recovered_from_a_different_simulator(tmp_path):
    cfg = _config()
    artifact = build_artifact_config(cfg)
    artifact.simulator = {"model_class": "prism.dmri.simulators.models.Ball3StickNoise"}
    OmegaConf.save(artifact, tmp_path / "artifact.yaml")

    with pytest.raises(ValueError, match="different simulator"):
        load_artifact_config(tmp_path, cfg=cfg)


def test_default_simulator_and_frozen_artifact_remain_unchanged(tmp_path):
    cfg = OmegaConf.create({
        "model": {"name": "Model"},
        "simulator": {"sim_type": {"name": "Ball3StickNoise"}},
    })
    artifact = build_artifact_config(cfg)
    assert simulator_model_config(cfg) == {
        "model_class": "prism.dmri.simulators.models.Ball3StickNoise"
    }
    OmegaConf.save(artifact, tmp_path / "artifact.yaml")

    assert load_artifact_config(tmp_path, cfg=cfg) == artifact
