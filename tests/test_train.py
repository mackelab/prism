import logging
import time
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from hydra.conf import HydraConf
from hydra.types import RunMode
from omegaconf import OmegaConf

from prism.train.build_simulator import build_simulator
from prism.train.checkpointing import CheckpointManager
from prism.train.dataset import SimulationDataset
from prism.train.train_script import (
    TrainState,
    _main,
    apply_checkpoint_to_state,
    configure_environment,
    ensure_fresh_checkpoint_dir,
    model_selection_weight,
    resolve_run_directory,
    train_loop,
)
from prism.train.utils import (
    bundle_checkpoint,
    download_checkpoint_from_hub,
    load_cfg,
    load_checkpoint,
    upload_checkpoint_to_hub,
)


def make_checkpoint_manager(threshold):
    manager = CheckpointManager.__new__(CheckpointManager)
    manager.recovery_threshold = threshold
    manager.prev_metric = None
    return manager


def test_recovery_threshold_handles_negative_metrics():
    manager = make_checkpoint_manager(1.5)

    assert not manager.should_recover(-10.0)
    assert not manager.should_recover(-6.0)
    assert manager.should_recover(-2.0)


def test_recovery_triggers_for_non_finite_metrics():
    manager = make_checkpoint_manager(2.0)

    assert manager.should_recover(float("nan"))
    assert manager.should_recover(float("inf"))

    disabled = make_checkpoint_manager(float("inf"))
    assert not disabled.should_recover(float("nan"))


def test_checkpoint_save_builds_args_without_persisting():
    manager = CheckpointManager.__new__(CheckpointManager)
    manager.keep_best = False

    class UnexpectedPersistence:
        def save(self, *args, **kwargs):
            raise AssertionError("checkpoint persistence should be disabled")

    manager.manager = UnexpectedPersistence()

    manager.save(
        step=1,
        params={"weight": jnp.array([1.0])},
        optimizer_state=None,
        loss=0.5,
        write_standard=False,
    )


def test_checkpoint_restore_targets_rng_to_current_device():
    manager = CheckpointManager.__new__(CheckpointManager)
    rng = jax.random.key(0)

    args = manager._build_restore_args(
        params={"weight": jnp.array([1.0])},
        optimizer_state=None,
        rng=rng,
    )

    assert args.rng.restore_args.sharding == rng.sharding


def test_missing_optimizer_state_is_reinitialized():
    params = {"weight": jnp.array([1.0])}
    optimizer = optax.sgd(0.1)
    state = TrainState(
        params={"weight": jnp.array([0.0])},
        model_state={},
        opt_state=None,
        ema_state=None,
        rng=jax.random.key(0),
    )
    checkpoint = {"params": params, "step": 4}
    cfg = OmegaConf.create({"train": {"track_ema": False}})

    restored = apply_checkpoint_to_state(
        state,
        checkpoint,
        cfg,
        optimizer,
        ema_transform=None,
        log=logging.getLogger(__name__),
        rebuild_optimizer=False,
    )

    assert restored.step == 4
    assert restored.opt_state is not None


def test_simulator_accepts_explicit_mask_and_prior():
    cfg = OmegaConf.create({
        "simulator": {
            "sim_type": {"name": "Ball3StickNoise"},
            "acquisition_scheme": {
                "name": "hcp",
                "params": {
                    "schemes": [
                        {
                            "name": "hcp",
                            "params": {
                                "num_acquisitions": 105,
                                "typical_prob": 1.0,
                                "random_prob": 0.0,
                            },
                        }
                    ]
                },
            },
            "prior_mask_alpha": 1.0,
            "prior_mask_beta": 1.0,
            "with_posterior_score": False,
        }
    })
    sim_type, simulators = build_simulator(cfg)
    num_components = len(sim_type.model_types) + len(sim_type.noise_types)
    model_mask = jnp.ones(num_components, dtype=jnp.bool_)
    prior = jnp.array([0.25])

    result = simulators[0](
        jax.random.key(1), mask_prior_hyperparameter=prior, model_mask=model_mask
    )

    assert jnp.array_equal(result["model_mask"], model_mask)
    assert jnp.array_equal(result["mask_prior"], prior)


def test_simulation_dataset_surfaces_background_failure():
    def simulator(key):
        return jnp.asarray([1.0], dtype=jnp.float32)

    dataset = SimulationDataset(
        simulator,
        simulation_batch_size=1,
        rng=jax.random.key(0),
        simulation_device=jax.devices("cpu")[0],
        jit_simulator=False,
        buffer_size=1,
    )
    try:
        dataset._stop_producer()

        def fail_batch():
            raise ValueError("simulator failed")

        dataset._produce_batch = fail_batch
        with dataset._condition:
            dataset._pending_refresh = dataset._batch_size
        dataset._start_producer()

        deadline = time.monotonic() + 5
        while dataset._producer_exception is None and time.monotonic() < deadline:
            time.sleep(0.01)

        with pytest.raises(
            RuntimeError, match="Background simulation failed"
        ) as exc_info:
            dataset[0]
        assert isinstance(exc_info.value.__cause__, ValueError)
    finally:
        dataset.close()


def test_new_simulator_config_builds_direct_class_and_acquisition():
    cfg = OmegaConf.create({
        "simulator": {
            "model_class": "prism.dmri.simulators.models.Ball3StickNoise",
            "posterior_score": False,
            "mask_prior": {},
            "acquisitions": [
                {
                    "_target_": ("prism.dmri.simulators.random_hcp_acquisition"),
                    "_partial_": True,
                    "num_acquisitions": 105,
                    "typical_prob": 1.0,
                    "random_prob": 0.0,
                }
            ],
        }
    })

    model_class, simulators = build_simulator(cfg)
    result = simulators[0](jax.random.key(2))

    assert model_class.__name__ == "Ball3StickNoise"
    assert result["x"].shape == (105,)
    assert "target_score" not in result


def make_checkpoint_tree(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "best" / "12"
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    (checkpoint / ".zarray").write_text("{}")
    latest = run_dir / "checkpoints" / "15"
    latest.mkdir()
    (latest / "_CHECKPOINT_METADATA").write_text("{}")
    OmegaConf.save(
        {
            "name": "example-model",
            "model": {"name": "ExampleModel"},
            "simulator": {"sim_type": {"name": "Ball3StickNoise"}},
        },
        run_dir / "config.yaml",
    )
    return run_dir


def test_bundle_checkpoint_creates_portable_layout(tmp_path):
    run_dir = make_checkpoint_tree(tmp_path)

    model_dir = bundle_checkpoint(run_dir, tmp_path / "bundle", which="best")

    assert model_dir.name == "example-model"
    assert (model_dir / "config.yaml").exists()
    assert (model_dir / "artifact.yaml").exists()
    assert (model_dir / "checkpoints" / "best" / "12" / "_CHECKPOINT_METADATA").exists()
    assert (model_dir / "checkpoints" / "best" / "12" / ".zarray").exists()

    config = OmegaConf.load(model_dir / "config.yaml")
    artifact = OmegaConf.load(model_dir / "artifact.yaml")
    assert config.schema_version == 2
    assert config.run.name == "example-model"
    assert artifact.artifact_version == 1
    assert "training" not in artifact
    assert artifact.simulator == {
        "model_class": "prism.dmri.simulators.models.Ball3StickNoise"
    }
    assert "acquisitions" not in artifact.simulator


def test_bundle_checkpoint_defaults_to_best_and_latest(tmp_path):
    run_dir = make_checkpoint_tree(tmp_path)
    temporary = run_dir / "checkpoints" / "best" / "broken.orbax-checkpoint-tmp"
    temporary.mkdir()

    model_dir = bundle_checkpoint(run_dir, tmp_path / "bundle")

    assert (model_dir / "checkpoints" / "best" / "12").is_dir()
    assert (model_dir / "checkpoints" / "15").is_dir()
    assert not (model_dir / "checkpoints" / "best" / temporary.name).exists()


def test_load_cfg_migrates_legacy_timestamp_layout(tmp_path):
    hydra_dir = tmp_path / "2026-01-02_03-04-05" / ".hydra"
    hydra_dir.mkdir(parents=True)
    OmegaConf.save(
        {
            "name": "legacy",
            "seed": 9,
            "model": {"name": "ExampleModel"},
            "simulator": {"sim_type": {"name": "Ball3StickNoise"}},
            "train": {"track_ema": False},
        },
        hydra_dir / "config.yaml",
    )

    cfg = load_cfg(tmp_path)

    assert cfg.schema_version == 2
    assert cfg.run.name == "legacy"
    assert cfg.run.seed == 9
    assert cfg.training.track_ema is False


def test_inference_restore_requests_ema_only_when_present(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    (checkpoint_dir / "12" / "params_ema").mkdir(parents=True)
    calls = {}

    class FakeOrbaxManager:
        def latest_step(self):
            return 12

        def restore(self, step, args):
            calls["step"] = step
            calls["args"] = args
            return {"params": {"weight": 1}, "params_ema": {"weight": 2}}

    manager = CheckpointManager.__new__(CheckpointManager)
    manager.ckpt_dir = str(checkpoint_dir)
    manager.best_ckpt_dir = str(checkpoint_dir / "best")
    manager.manager = FakeOrbaxManager()
    manager.best_manager = FakeOrbaxManager()

    restored = manager.restore_parameters(None, {"weight": 0})

    assert calls["step"] == 12
    assert restored["params_ema"]["weight"] == 2
    assert restored["step"] == 12


def test_upload_checkpoint_uses_model_subfolder(tmp_path, monkeypatch):
    run_dir = make_checkpoint_tree(tmp_path)
    calls = {}

    class FakeApi:
        def __init__(self, token):
            calls["token"] = token

        def create_repo(self, *args, **kwargs):
            calls["create"] = (args, kwargs)

        def upload_folder(self, **kwargs):
            calls["upload"] = kwargs
            assert (Path(kwargs["folder_path"]) / "config.yaml").exists()
            assert (Path(kwargs["folder_path"]) / "checkpoints" / "best").exists()
            assert (Path(kwargs["folder_path"]) / "checkpoints" / "15").exists()
            return "commit"

    monkeypatch.setattr("prism.train.utils.HfApi", FakeApi)

    result = upload_checkpoint_to_hub(
        run_dir,
        "owner/dmri-pretrained",
        token="secret",
        private=True,
    )

    assert result == "commit"
    assert calls["create"][0] == ("owner/dmri-pretrained",)
    assert calls["create"][1]["private"] is True
    assert calls["upload"]["path_in_repo"] == "example-model"


def test_download_checkpoint_selects_model_subfolder(tmp_path, monkeypatch):
    model_dir = tmp_path / "snapshot" / "example-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text("name: example-model\n")
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return model_dir.parent

    monkeypatch.setattr("prism.train.utils.snapshot_download", fake_snapshot_download)

    result = download_checkpoint_from_hub(
        "owner/dmri-pretrained", "example-model", revision="v1"
    )

    assert result == model_dir.resolve()
    assert calls["allow_patterns"] == ["example-model/**"]
    assert calls["revision"] == "v1"


@pytest.mark.parametrize(
    ("relative", "name", "expected"),
    [
        ("results/example/2026-01-01_12-00-00", "example", "results/example"),
        ("results/example/2026-01-01_12-00-00/0", "example", "results/example"),
        ("custom/output", "example", "custom/output"),
        ("results/example", "example", "results/example"),
    ],
)
def test_run_directory_honors_custom_hydra_output(tmp_path, relative, name, expected):
    assert resolve_run_directory(str(tmp_path / relative), name) == str(
        tmp_path / expected
    )


@pytest.mark.parametrize("mode", [RunMode.RUN, RunMode.MULTIRUN])
def test_hydra_multirun_jobs_have_separate_checkpoint_roots(
    tmp_path, monkeypatch, mode
):
    cfg = OmegaConf.create({"name": "example"})
    roots = []
    for job in (0, 1):
        hydra_cfg = OmegaConf.structured(HydraConf)
        hydra_cfg.mode = mode
        output_dir = tmp_path / "results" / "example" / "2026-01-01_12-00-00" / str(job)
        hydra_cfg.runtime.output_dir = str(output_dir)
        monkeypatch.setattr(
            "prism.train.train_script.HydraConfig.get", lambda config=hydra_cfg: config
        )
        _, actual_output, root = configure_environment(cfg)
        assert actual_output == str(output_dir)
        roots.append(root)
    if mode == RunMode.MULTIRUN:
        assert roots[0] != roots[1]
        assert roots[0].endswith("/0") and roots[1].endswith("/1")
    else:
        assert roots == [str(tmp_path / "results" / "example")] * 2


def test_best_checkpoint_survives_resume_and_exposes_validation_loss(tmp_path):
    checkpoint_dir = str(tmp_path / "checkpoints")
    params = {"weight": jnp.array([1.0])}
    opt_state = {"momentum": jnp.array([0.0])}
    manager = CheckpointManager(checkpoint_dir)
    try:
        manager.save(1, params, opt_state, loss=3.0, val_loss=2.0, params_ema=params)
        manager.wait_until_finished()
    finally:
        manager.manager.close()
        manager.best_manager.close()

    resumed = CheckpointManager(checkpoint_dir, continue_training=True)
    try:
        assert resumed.best_val_loss == 2.0
        resumed.save(2, params, opt_state, loss=4.0, val_loss=5.0, params_ema=params)
        resumed.wait_until_finished()
        assert resumed.best_manager.latest_step() == 1
        checkpoint = resumed.restore_parameters(None, params, from_best=True)
        assert checkpoint["step"] == 1
        assert checkpoint["val_loss"] == 2.0
        assert checkpoint["loss"] == 3.0
        assert "optimizer_state" not in checkpoint
        assert "params_ema" in checkpoint
        training_checkpoint = resumed.restore(None, params, opt_state, from_best=True)
        assert training_checkpoint["val_loss"] == 2.0
    finally:
        resumed.manager.close()
        resumed.best_manager.close()


def test_model_selection_warmup_and_ramp():
    cfg = OmegaConf.create({"train": {"model_selection_weight": 2.0}})
    assert float(model_selection_weight(cfg, 0)) == 2.0
    cfg.train.model_selection_warmup_steps = 2
    cfg.train.model_selection_ramp_steps = 4
    values = jax.jit(jax.vmap(lambda step: model_selection_weight(cfg, step)))(
        jnp.array([0, 2, 3, 4, 6, 20])
    )
    assert jnp.allclose(values, jnp.array([0.0, 0.0, 0.5, 1.0, 2.0, 2.0]))


@pytest.mark.parametrize(
    ("step_cap", "hours", "final_step"), [(5, float("inf"), 5), (None, 0, 3)]
)
def test_training_limits_log_validate_and_save_final_step(
    monkeypatch, step_cap, hours, final_step
):
    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "inner_steps": 3,
            "eval": {"nnl_mask": {"iters": 1}, "nnl_theta": {"iters": 1}},
        },
    })
    state = TrainState(jnp.array(0), {}, {}, None, jax.random.key(0))
    logged = []
    saves = []
    evaluated = []
    monkeypatch.setattr("prism.train.train_script.wandb.log", logged.append)

    def update(params, model_state, opt_state, ema, batch, rng, step):
        return params + 1, model_state, opt_state, ema, (0.5, 0.5), 1.0

    def evaluate(params, *args, **kwargs):
        evaluated.append(int(params))
        return 0.5

    manager = SimpleNamespace(
        keep_best=True,
        save=lambda **kwargs: saves.append(kwargs),
        should_recover=lambda metric: False,
        wait_until_finished=lambda: None,
    )
    evaluator = SimpleNamespace(eval_nnl_mask=evaluate, eval_nnl_theta=evaluate)
    result = train_loop(
        cfg,
        logging.getLogger(__name__),
        state,
        optax.sgd(0.1),
        update,
        manager,
        evaluator,
        [cycle([{}])],
        cycle([{}]),
        track_ema=False,
        ema_transform=None,
        checkpoint_freq=3,
        eval_freq=3,
        restart_every=None,
        max_train_hours=hours,
        wandb_active=True,
        max_train_steps=step_cap,
    )

    assert result.step == final_step
    assert int(result.params) == final_step
    assert evaluated[-2:] == [final_step, final_step]
    final_saves = [saved for saved in saves if saved["step"] == final_step]
    assert len(final_saves) == 2  # validation-selected and regular checkpoint
    assert any(saved["val_loss"] == 1.0 for saved in final_saves)
    assert any(saved["write_standard"] for saved in final_saves)
    assert any(row.get("step") == final_step and "loss mask" in row for row in logged)
    assert any(
        row.get("step") == final_step and "mask_negative_log_likelihood" in row
        for row in logged
    )


def test_finished_training_does_not_take_an_extra_step():
    state = TrainState(None, None, None, None, jax.random.key(0), step=5)
    cfg = OmegaConf.create({"train": {"inner_steps": 3}})
    result = train_loop(
        cfg,
        logging.getLogger(__name__),
        state,
        None,
        None,
        None,
        None,
        [iter(())],
        None,
        False,
        None,
        None,
        None,
        None,
        float("inf"),
        False,
        max_train_steps=5,
    )
    assert result is state


class _CheckpointModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.ones(1))
        self.cfg = {}


@pytest.mark.parametrize("config_format", ["schema_v2", "legacy", "defaults"])
def test_load_checkpoint_uses_saved_checkpoint_options(
    tmp_path, monkeypatch, config_format
):
    config = {
        "model": {"name": "ExampleModel"},
        "simulator": {"model_class": "prism.dmri.simulators.models.BallStick"},
    }
    if config_format == "schema_v2":
        config.update({
            "schema_version": 2,
            "training": {
                "checkpoint": {"max_checkpoints": 2, "keep_best_checkpoint": False},
                "recovery_threshold": 2.5,
            },
        })
    elif config_format == "legacy":
        config.update({
            "max_checkpoints": 2,
            "keep_best_checkpoint": False,
            "recovery_threshold": 2.5,
        })
    OmegaConf.save(OmegaConf.create(config), tmp_path / "config.yaml")
    _, params, _ = nnx.split(_CheckpointModel(), nnx.Param, ...)
    writer = CheckpointManager(str(tmp_path / "checkpoints"), keep_best=False)
    try:
        writer.save(1, params, {"momentum": jnp.zeros(1)}, loss=1.0)
    finally:
        writer.close()

    captured = {}
    readers = []

    def make_reader(**kwargs):
        captured.update(kwargs)
        manager = CheckpointManager(**kwargs)
        readers.append(manager)
        return manager

    monkeypatch.setattr(
        "prism.train.utils.build_model", lambda *args: _CheckpointModel()
    )
    monkeypatch.setattr("prism.train.utils.CheckpointManager", make_reader)
    try:
        checkpoint, _, _ = load_checkpoint(tmp_path)
        assert checkpoint["step"] == 1
        assert captured["max_to_keep"] == (5 if config_format == "defaults" else 2)
        assert captured["keep_best"] == (config_format == "defaults")
        assert captured["recovery_threshold"] == (
            float("inf") if config_format == "defaults" else 2.5
        )
    finally:
        for reader in readers:
            reader.close()


@pytest.mark.parametrize("config_format", ["schema_v2", "legacy"])
def test_training_checkpoint_retention_uses_configured_limit(
    tmp_path, monkeypatch, config_format
):
    training = {
        "optimizer": {"optimizer": "sgd", "scheduler": None, "learning_rate": 0.1},
        "track_ema": False,
        "inner_steps": 1,
        "checkpoint_freq": 1,
        "eval_freq": 0,
    }
    config = {
        "model": {"name": "ExampleModel"},
        "simulator": {"model_class": "prism.dmri.simulators.models.BallStick"},
    }
    if config_format == "schema_v2":
        training["checkpoint"] = {"max_checkpoints": 2, "keep_best_checkpoint": False}
        config.update({"schema_version": 2, "training": training})
    else:
        config.update({
            "train": training,
            "max_checkpoints": 2,
            "keep_best_checkpoint": False,
        })

    monkeypatch.setattr(
        "prism.train.train_script.configure_environment",
        lambda cfg: (logging.getLogger(__name__), str(tmp_path), str(tmp_path)),
    )
    monkeypatch.setattr(
        "prism.train.train_script.build_model", lambda *args: _CheckpointModel()
    )
    monkeypatch.setattr(
        "prism.train.train_script.build_simulator",
        lambda *args: (SimpleNamespace(evaluation_design=lambda cfg: None), []),
    )
    monkeypatch.setattr(
        "prism.train.train_script.create_dataloaders",
        lambda *args: (
            [SimpleNamespace(close=lambda: None)],
            SimpleNamespace(close=lambda: None),
        ),
    )

    def save_three_checkpoints(*, checkpoint_manager, train_state, **kwargs):
        assert checkpoint_manager.keep_best is False
        for step in range(1, 4):
            checkpoint_manager.save(
                step,
                train_state.params,
                train_state.opt_state,
                loss=1.0,
                val_loss=0.5,
            )
        checkpoint_manager.wait_until_finished()

    monkeypatch.setattr("prism.train.train_script.train_loop", save_three_checkpoints)
    _main.__wrapped__(OmegaConf.create(config))

    retained = sorted(
        path.name
        for path in (tmp_path / "checkpoints").iterdir()
        if path.name.isdigit()
    )
    assert retained == ["2", "3"]
    assert not any((tmp_path / "checkpoints" / "best").iterdir())


def _write_checkpoint(checkpoint_dir, step):
    params = {"weight": jnp.array([1.0])}
    manager = CheckpointManager(checkpoint_dir)
    try:
        manager.save(step, params, {"m": jnp.array([0.0])}, val_loss=1.0)
        manager.wait_until_finished()
    finally:
        manager.close()


def test_fresh_run_refuses_a_checkpoint_directory_that_is_in_use(tmp_path):
    # A second fresh run with the same run.name used to train in full while
    # Orbax silently kept the first run's checkpoints.
    checkpoint_dir = str(tmp_path / "checkpoints")
    empty = CheckpointManager(checkpoint_dir)
    try:
        ensure_fresh_checkpoint_dir(empty, continue_training=False)
    finally:
        empty.close()
    _write_checkpoint(checkpoint_dir, 25)

    reused = CheckpointManager(checkpoint_dir)
    try:
        assert reused.saved_steps() == [25]
        with pytest.raises(FileExistsError, match="continue_training=true"):
            ensure_fresh_checkpoint_dir(reused, continue_training=False)
    finally:
        reused.close()
    resumed = CheckpointManager(checkpoint_dir, continue_training=True)
    try:
        ensure_fresh_checkpoint_dir(resumed, continue_training=True)
    finally:
        resumed.close()


def test_a_save_orbax_declines_is_reported(tmp_path, caplog):
    checkpoint_dir = str(tmp_path / "checkpoints")
    _write_checkpoint(checkpoint_dir, 25)
    manager = CheckpointManager(checkpoint_dir, continue_training=True)
    try:
        with caplog.at_level(logging.WARNING):
            manager.save(25, {"weight": jnp.array([2.0])}, None)
            manager.wait_until_finished()
    finally:
        manager.close()
    assert "Checkpoint for step 25 was not written" in caplog.text


def test_partial_restore_does_not_block_the_next_save(tmp_path):
    # Used to fail at the first save after resume: Item "params" and args
    # "StandardSaveArgs" does not match with any registered handler.
    checkpoint_dir = str(tmp_path / "checkpoints")
    params = {"weight": jnp.array([1.0]), "bias": jnp.array([0.5])}
    manager = CheckpointManager(checkpoint_dir)
    try:
        manager.save(10, params, {"m": jnp.array([0.0])}, val_loss=1.0)
    finally:
        manager.close()

    resumed = CheckpointManager(checkpoint_dir, continue_training=True)
    try:
        restored = resumed.restore(
            None, {"weight": jnp.zeros(1)}, None, partial_restore=True
        )
        assert float(restored["params"]["weight"][0]) == 1.0
        resumed.save(20, params, {"m": jnp.array([0.0])}, val_loss=0.5)
        resumed.wait_until_finished()
        assert resumed.saved_steps() == [10, 20]
    finally:
        resumed.close()
