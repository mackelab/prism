"""Small loader checks; the real public bundle is validated separately."""

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from experiments.dmri._shared import checkpoint as loader
from prism.dmri.simulators.models import MultiShellBall3StickSharedDiffusivity


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    sim_type = MultiShellBall3StickSharedDiffusivity
    cfg = OmegaConf.create({
        "schema_version": 2,
        "run": {"seed": 0},
        "model": {"init_seed": 0},
        "simulator": {
            "sim_type": {"name": sim_type.__name__},
            "prior_mask_alpha": 1.0,
            "prior_mask_beta": 1.0,
            "with_posterior_score": False,
            "acquisition_scheme": {
                "name": "multi",
                "params": {
                    "schemes": [
                        {
                            "name": "hcp",
                            "params": {
                                "num_acquisitions": 105,
                                "typical_prob": 0.5,
                                "random_prob": 0.5,
                            },
                        },
                        {
                            "name": "hcp_large",
                            "params": {
                                "num_acquisitions": 297,
                                "typical_prob": 0.8,
                                "random_prob": 0.2,
                            },
                        },
                    ]
                },
            },
        },
        "unresolved_note": "${missing.value}",
    })
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
    model.cfg = SimpleNamespace(simulator=sim_type)
    raw = nnx.state(model, nnx.Param)
    other = nnx.Linear(2, 1, rngs=nnx.Rngs(9))
    ema = nnx.state(other, nnx.Param)
    checkpoint = {"step": 1520000, "params": raw, "params_ema": ema}
    monkeypatch.setattr(
        loader, "load_checkpoint", lambda *a, **kw: (checkpoint, model, sim_type)
    )
    return tmp_path, model, checkpoint


@pytest.mark.parametrize("which, step", [("best", 1520000), ("latest", 2280000)])
def test_load_model_selects_exact_ema_and_preserves_saved_config(bundle, which, step):
    path, model, checkpoint = bundle
    checkpoint["step"] = step
    result, sim_type, generators, metadata = loader.load_public_model(path, which)
    assert result is model
    assert sim_type is MultiShellBall3StickSharedDiffusivity
    assert len(generators) == 2
    for name, value in nnx.state(model, nnx.Param).items():
        np.testing.assert_array_equal(value.value, checkpoint["params_ema"][name].value)
    assert metadata["selected_step"] == step
    assert metadata["params_name"] == "params_ema"
    assert metadata["saved_config"]["unresolved_note"] == "${missing.value}"
    assert metadata["simulator_config"]["prior_mask_alpha"] == 1.0
    assert metadata["theta_sampler_defaults"]["num_steps"] == 64
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("missing", [True, False])
def test_missing_ema_fails_without_updating_model(bundle, missing):
    path, model, checkpoint = bundle
    before = np.asarray(model.kernel.value).copy()
    if missing:
        del checkpoint["params_ema"]
    else:
        checkpoint["params_ema"] = None
    with pytest.raises(ValueError, match="no params_ema"):
        loader.load_public_model(path)
    np.testing.assert_array_equal(model.kernel.value, before)


def test_wrong_step_is_rejected(bundle):
    path, _, checkpoint = bundle
    checkpoint["step"] = 100
    with pytest.raises(ValueError, match="Expected public best step 1520000"):
        loader.load_public_model(path)


def test_nonfinite_ema_is_rejected(bundle):
    path, _, checkpoint = bundle
    checkpoint["params_ema"]["kernel"].value = jnp.full((2, 1), jnp.nan)
    with pytest.raises(ValueError, match="not finite, exact EMA"):
        loader.load_public_model(path)
