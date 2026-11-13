"""Round trip and validation for the safetensors symbolic checkpoint loader."""

from pathlib import Path

import jax
import numpy as np
import pytest
from flax import nnx
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from prism.symbolic.pretrained import (
    build_from_config,
    load_params_safetensors,
    load_pretrained_symbolic,
    save_params_safetensors,
)

CONF = Path(__file__).resolve().parents[1] / "conf"
PRESET = "symbolic_sbmi_10k"


def _params(model):
    leaves = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]
    return [np.asarray(v) for _, v in leaves]


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose("train.yaml", overrides=[f"+experiment/train={PRESET}"])
    root = tmp_path_factory.mktemp("hub")
    OmegaConf.save(
        OmegaConf.create({
            k: OmegaConf.to_container(cfg[k], resolve=True)
            for k in ("model", "simulator")
        }),
        root / "config.yaml",
    )
    model, _, _ = build_from_config(OmegaConf.load(root / "config.yaml"))
    (root / "best").mkdir()
    save_params_safetensors(
        model,
        root / "best" / "params_ema.safetensors",
        {"step": "123", "prism_preset": PRESET},
    )
    return root, model


def test_round_trip_restores_parameters_exactly(saved):
    root, original = saved
    loaded = load_pretrained_symbolic(path=root, which="best")
    assert loaded.step == 123 and loaded.preset == PRESET and loaded.which == "best"
    for a, b in zip(_params(original), _params(loaded.model), strict=True):
        assert a.dtype == b.dtype and np.array_equal(a, b)


def test_weights_are_applied_not_just_initialised(saved):
    root, original = saved
    cfg = OmegaConf.load(root / "config.yaml")
    cfg.model.init_seed = 99
    other, _, _ = build_from_config(cfg)
    assert not all(
        np.array_equal(a, b)
        for a, b in zip(_params(original), _params(other), strict=True)
    )
    load_params_safetensors(other, root / "best" / "params_ema.safetensors")
    assert all(
        np.array_equal(a, b)
        for a, b in zip(_params(original), _params(other), strict=True)
    )


def test_architecture_mismatch_is_rejected_before_changing_the_model(saved):
    root, _ = saved
    cfg = OmegaConf.load(root / "config.yaml")
    cfg.model.model_selection_net.params.num_layers += 2
    bigger, _, _ = build_from_config(cfg)
    before = _params(bigger)
    with pytest.raises(ValueError, match="does not match the model"):
        load_params_safetensors(bigger, root / "best" / "params_ema.safetensors")
    assert all(
        np.array_equal(a, b) for a, b in zip(before, _params(bigger), strict=True)
    )


def test_selector_arguments_are_validated(saved):
    root, _ = saved
    with pytest.raises(ValueError, match="best"):
        load_pretrained_symbolic(path=root, which="final")
    with pytest.raises(ValueError, match="exactly one"):
        load_pretrained_symbolic(PRESET, path=root)


def test_figure_checkpoint_directory_and_unknown_selection(saved, tmp_path):
    import shutil

    root, _ = saved
    shutil.copytree(root, tmp_path / "hub")
    shutil.copytree(tmp_path / "hub" / "best", tmp_path / "hub" / "figure")
    loaded = load_pretrained_symbolic(path=tmp_path / "hub", which="figure")
    assert loaded.which == "figure" and loaded.step == 123
    with pytest.raises(ValueError, match="which must be one of"):
        load_pretrained_symbolic(path=root, which="final")
