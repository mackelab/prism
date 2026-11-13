"""Regression checks for the larger fixed-data symbolic runs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments.symbolic.simulation_efficiency import completion as evaluate
from experiments.symbolic.simulation_efficiency import evaluate as fig4b
from experiments.symbolic.simulation_efficiency import train
from prism.config.aliases import resolve_class_path

ROOT = Path(__file__).resolve().parents[1]


def test_training_command_uses_shared_trainer_and_explicit_output(tmp_path):
    output = tmp_path / "space in path" / "custom_name"
    command = train.training_command("200k", output, ["tracking.enabled=true"])
    assert command[1:3] == ["-m", "prism.train.train_script"]
    assert 'run.name="custom_name"' in command
    assert f"hydra.run.dir={json.dumps(str(output))}" in command
    assert "tracking.enabled=true" in command
    assert not output.exists()
    with pytest.raises(ValueError, match="fresh run"):
        train.training_command("200k", output, ["training.continue_training=true"])


def test_failed_training_writes_process_record_but_no_completion(tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert command[0] == "git"
        return SimpleNamespace(returncode=0, stdout="test revision")

    def fail():
        raise RuntimeError("Training failed")

    monkeypatch.setattr(train.subprocess, "run", run)
    monkeypatch.setattr(train.shared_trainer, "main", fail)
    output = tmp_path / "failed"
    original_argv = train.sys.argv
    with pytest.raises(RuntimeError, match="Training failed"):
        train.main(["--point", "200k", "--output", str(output)])
    record = json.loads((output / "training-process.json").read_text())
    assert record["process_exit_code"] == 1
    assert train.sys.argv is original_argv
    assert not (output / "training-complete.json").exists()
    with pytest.raises(FileExistsError):
        train.main(["--point", "200k", "--output", str(output)])


def test_completion_requires_terminal_log_and_actual_saved_steps(tmp_path, monkeypatch):
    cfg = config("100k")
    log = tmp_path / "train_script.log"
    log.write_text("Training complete\n")
    monkeypatch.setattr(train, "load_cfg", lambda _: cfg)
    monkeypatch.setattr(
        train,
        "latest_saved_step",
        lambda path: 480000 if path.name == "best" else 500000,
    )
    monkeypatch.setattr(train, "checkpoint_val_loss", lambda *args, **kwargs: 2.5)
    process = {
        "point": "100k",
        "process_exit_code": 0,
        "elapsed_training_seconds": 30000,
    }
    with pytest.raises(RuntimeError, match="normal training limit"):
        train.completion_record("100k", tmp_path, process)
    log.write_text("Training limit reached at step 500000.\nTraining complete\n")
    record = train.completion_record("100k", tmp_path, process)
    assert record["completed_training_step"] == 500000
    assert record["selected_best_checkpoint_step"] == 480000
    assert record["best_val_loss"] == 2.5
    evaluate.validate_completion(record, cfg, "100k", 500000, tmp_path)


def config(point, overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        return compose(
            config_name="train",
            overrides=[f"+experiment/train=symbolic_sbmi_{point}", *overrides],
        )


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


@pytest.mark.parametrize("point", fig4b.LARGER_POINTS)
def test_scientific_settings_match_existing_larger_run_builder(point):
    fixture = json.loads(
        (ROOT / "tests/fixtures/fig4b_larger_configs.json").read_text()
    )
    cfg = config(point)
    settings = {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        for key in ("simulator", "model", "training")
    }
    settings["run.seed"] = cfg.run.seed
    expected = fixture["common"] | fixture["points"][point]
    # The fixture records class paths as the existing runs wrote them.
    expected = {
        key: resolve_class_path(value)
        if isinstance(value, str) and value.startswith("prism.")
        else value
        for key, value in expected.items()
    }
    assert flatten(settings) == expected
    assert cfg.tracking.enabled is False
    assert cfg.tracking.wandb.entity is None
    fig4b.validate_fig4b(cfg, point, fig4b.FIG4B[point][1] or 4000)


@pytest.mark.parametrize(
    "override",
    [
        "training.dataloader.train_loader.batch_size=1024",
        "training.optimizer.learning_rate=5e-4",
        "simulator.x_step=0.01",
        "model.inference_net.params.num_layers=4",
        "training.model_selection_warmup_steps=15000",
        "training.max_train_steps=1000000",
        "training.max_train_hours=7",
    ],
)
def test_larger_protocol_rejects_recipe_drift(override):
    with pytest.raises(RuntimeError):
        fig4b.validate_fig4b(config("200k", [override]), "200k", 4000)


@pytest.mark.parametrize("point", fig4b.LARGER_POINTS)
def test_best_latest_keep_full_notebook_protocol(point):
    args = SimpleNamespace(
        fig4b_point=point,
        which="best",
        num_samples=1000,
        groups=10,
        seed=1110000000000,
        lam=None,
    )
    assert fig4b.check_evaluation_protocol(args) == "primary"
    args.which = "latest"
    assert fig4b.check_evaluation_protocol(args) == "supplementary"
    args.num_samples = 100
    with pytest.raises(ValueError):
        fig4b.check_evaluation_protocol(args)


@pytest.mark.parametrize(
    "point,steps,hours,reason",
    [
        ("100k", 500_000, 24, "step_limit"),
        ("200k", None, 8, "time_limit"),
        ("500k", None, 8, "time_limit"),
        ("1000k", None, 8, "time_limit"),
    ],
)
def test_explicit_step_or_time_completion_accepted(
    tmp_path, point, steps, hours, reason
):
    cfg = OmegaConf.create({
        "training": {"max_train_steps": steps, "max_train_hours": hours}
    })
    completed = steps if steps is not None else 123_456
    record = completion_record(tmp_path, point, completed, steps, hours, reason)
    evaluate.validate_completion(record, cfg, point, completed, tmp_path)


def completion_record(
    run_dir, point="200k", completed=123_456, cap=None, hours=8, reason="time_limit"
):
    return {
        "point": point,
        "run_dir": str(run_dir),
        "process_exit_code": 0,
        "stop_reason": reason,
        "completed_training_step": completed,
        "configured_max_train_steps": cap,
        "configured_max_train_hours": hours,
        "elapsed_training_seconds": hours * 3600 + 123,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("point", "500k"),
        ("run_dir", "/different/run"),
        ("process_exit_code", 1),
        ("process_exit_code", False),
        ("completed_training_step", 123_455),
        ("configured_max_train_steps", 500_000),
        ("configured_max_train_hours", 7),
        ("stop_reason", "step_limit"),
        ("elapsed_training_seconds", 28_799),
        ("elapsed_training_seconds", -1),
        ("elapsed_training_seconds", float("nan")),
        ("elapsed_training_seconds", True),
    ],
)
def test_incomplete_or_mismatched_time_run_is_rejected(tmp_path, field, value):
    cfg = OmegaConf.create({
        "training": {"max_train_steps": None, "max_train_hours": 8}
    })
    record = completion_record(tmp_path)
    record[field] = value
    with pytest.raises(RuntimeError):
        evaluate.validate_completion(record, cfg, "200k", 123_456, tmp_path)


def test_early_time_stop_cannot_pass_with_a_checkpoint(tmp_path):
    cfg = OmegaConf.create({
        "training": {"max_train_steps": None, "max_train_hours": 8}
    })
    record = completion_record(tmp_path, completed=1000)
    with pytest.raises(RuntimeError, match="first validation"):
        evaluate.validate_completion(record, cfg, "200k", 1000, tmp_path)


def test_step_capped_run_must_reach_exact_cap(tmp_path):
    cfg = OmegaConf.create({
        "training": {"max_train_steps": 500_000, "max_train_hours": 24}
    })
    record = completion_record(tmp_path, "100k", 499_999, 500_000, 24, "step_limit")
    with pytest.raises(RuntimeError, match="exactly"):
        evaluate.validate_completion(record, cfg, "100k", 499_999, tmp_path)


def test_missing_exit_evidence_and_nonfinite_json_rejected_before_inference(tmp_path):
    cfg = OmegaConf.create({
        "training": {"max_train_steps": None, "max_train_hours": 8}
    })
    record = completion_record(tmp_path)
    del record["process_exit_code"]
    with pytest.raises(RuntimeError, match="missing fields"):
        evaluate.validate_completion(record, cfg, "200k", 123_456, tmp_path)
    path = tmp_path / "completion.json"
    path.write_text('{"elapsed_training_seconds": NaN}')
    with pytest.raises(ValueError):
        evaluate.read_completion(path)


def test_restored_actual_diffusion_bounds_must_be_original_defaults():
    from probjax.nn.nets.denoising_diffusion_model import EDM

    decoder = EDM(None, loss_type="v")
    model = SimpleNamespace(inference_decoder=decoder)
    evaluate.validate_model_bounds(model)
    decoder.train_cfg.t_min = 1e-4
    with pytest.raises(RuntimeError, match="diffusion bounds"):
        evaluate.validate_model_bounds(model)
