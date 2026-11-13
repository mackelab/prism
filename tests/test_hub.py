import logging
from dataclasses import FrozenInstanceError

import pytest

from prism import (
    DEFAULT_MODEL_NAME,
    DEFAULT_REPO_ID,
    PretrainedModel,
    list_pretrained_models,
    load_pretrained,
)


class FakeModel:
    cfg = {"name": "model-config"}

    def __init__(self):
        self.evaluated = False

    def eval(self):
        self.evaluated = True


def test_load_pretrained_uses_defaults_and_ema(monkeypatch, caplog):
    calls = {}
    model = FakeModel()
    simulator = object()

    def fake_load_checkpoint(**kwargs):
        calls["load"] = kwargs
        return (
            {"params": "params", "params_ema": "ema", "step": 12},
            model,
            simulator,
        )

    monkeypatch.setattr("prism.hub.load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        "prism.hub.nnx.update",
        lambda updated_model, params: calls.update(update=(updated_model, params)),
    )

    with caplog.at_level(logging.INFO, logger="prism.train.utils"):
        result = load_pretrained()

    assert calls["load"] == {
        "repo_id": DEFAULT_REPO_ID,
        "model_name": DEFAULT_MODEL_NAME,
        "which": "best",
        "revision": None,
        "cache_dir": None,
        "token": None,
        "local_files_only": False,
        "precision": None,
    }
    assert calls["update"] == (model, "ema")
    assert result == PretrainedModel(model, simulator, model.cfg, 12)
    assert result.params_name == "params_ema"
    assert model.evaluated is True
    selection_logs = [
        record for record in caplog.records if record.name == "prism.train.utils"
    ]
    assert len(selection_logs) == 1
    assert "12" in selection_logs[0].getMessage()
    assert "params_ema" in selection_logs[0].getMessage()


@pytest.mark.parametrize("ema", ["absent", None, "ema"])
def test_load_pretrained_forwards_options_and_uses_explicit_raw_params(
    monkeypatch, caplog, ema
):
    calls = {}
    model = FakeModel()

    def fake_load_checkpoint(**kwargs):
        calls["load"] = kwargs
        checkpoint = {"params": "params", "step": 19}
        if ema != "absent":
            checkpoint["params_ema"] = ema
        return checkpoint, model, "simulator"

    monkeypatch.setattr("prism.hub.load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        "prism.hub.nnx.update",
        lambda updated_model, params: calls.update(update=(updated_model, params)),
    )

    with caplog.at_level(logging.INFO, logger="prism.train.utils"):
        result = load_pretrained(
            "other-model",
            "owner/repo",
            "latest",
            revision="v1",
            cache_dir="cache",
            token="secret",
            local_files_only=True,
            precision="bf16",
            params_name="params",
        )

    assert calls["load"] == {
        "repo_id": "owner/repo",
        "model_name": "other-model",
        "which": "latest",
        "revision": "v1",
        "cache_dir": "cache",
        "token": "secret",
        "local_files_only": True,
        "precision": "bf16",
    }
    assert calls["update"] == (model, "params")
    assert result.step == 19
    assert result.params_name == "params"
    assert model.evaluated is True
    selection_logs = [
        record for record in caplog.records if record.name == "prism.train.utils"
    ]
    assert len(selection_logs) == 1
    assert "19" in selection_logs[0].getMessage()
    assert "params" in selection_logs[0].getMessage()


@pytest.mark.parametrize(
    ("checkpoint", "params_name"),
    [
        ({"params": "raw", "step": 31}, "params_ema"),
        ({"params": "raw", "params_ema": None, "step": 32}, "params_ema"),
        ({"params_ema": "ema", "step": 33}, "params"),
        ({"params_ema": "ema", "params": None, "step": 34}, "params"),
    ],
)
def test_load_pretrained_rejects_missing_weights_before_model_update(
    monkeypatch, checkpoint, params_name
):
    model = FakeModel()
    updates = []
    monkeypatch.setattr(
        "prism.hub.load_checkpoint", lambda **kwargs: (checkpoint, model, None)
    )
    monkeypatch.setattr("prism.hub.nnx.update", lambda *args: updates.append(args))
    options = {} if params_name == "params_ema" else {"params_name": params_name}

    with pytest.raises(ValueError, match=f"step {checkpoint['step']}") as exc:
        load_pretrained(**options)

    assert params_name in str(exc.value)
    if params_name == "params_ema":
        assert "params_name='params'" in str(exc.value)
    assert updates == []
    assert model.evaluated is False


@pytest.mark.parametrize("params_name", ["ema", "raw", "best", "", None])
def test_load_pretrained_rejects_invalid_parameter_selection(monkeypatch, params_name):
    model = FakeModel()
    updates = []
    monkeypatch.setattr(
        "prism.hub.load_checkpoint",
        lambda **kwargs: (
            {"params": "raw", "params_ema": "ema", "step": 7},
            model,
            None,
        ),
    )
    monkeypatch.setattr("prism.hub.nnx.update", lambda *args: updates.append(args))
    with pytest.raises(ValueError, match="params_ema.*params"):
        load_pretrained(params_name=params_name)
    assert updates == []
    assert model.evaluated is False


def test_pretrained_model_is_frozen():
    result = PretrainedModel(None, None, None, None)
    assert result.params_name == "params_ema"

    with pytest.raises(FrozenInstanceError):
        result.step = 1


def test_list_pretrained_models(monkeypatch):
    calls = {}

    class FakeApi:
        def __init__(self, token):
            calls["token"] = token

        def list_repo_files(self, **kwargs):
            calls["list"] = kwargs
            return [
                "z-model/config.yaml",
                "README.md",
                "a-model/checkpoints/best/1/data",
                "a-model/config.yaml",
                "nested/model/config.yaml",
            ]

    monkeypatch.setattr("prism.hub.HfApi", FakeApi)

    result = list_pretrained_models("owner/repo", revision="v2", token="secret")

    assert result == ["a-model", "z-model"]
    assert calls == {
        "token": "secret",
        "list": {
            "repo_id": "owner/repo",
            "repo_type": "model",
            "revision": "v2",
        },
    }
