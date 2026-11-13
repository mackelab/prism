"""Saved offline configs retain preset launch settings without submitting jobs."""

import pytest
from hydra import compose, initialize_config_dir, initialize_config_module
from hydra.conf import RunMode
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import SlurmLauncher
from omegaconf import OmegaConf

from prism.train import simulate


@pytest.mark.parametrize(
    ("preset", "overrides", "launcher", "mode", "timeout"),
    [
        ("symbolic", [], "submitit_local", None, 120),
        ("symbolic_sbmi_10k", [], "submitit_slurm", RunMode.MULTIRUN, 700),
        ("symbolic_sbmi_100k", [], "submitit_slurm", RunMode.MULTIRUN, 1560),
        (
            "symbolic_sbmi_10k",
            [
                "hydra.launcher.partition=custom-gpu",
                "hydra.launcher.account=test-account",
                "hydra.launcher.mem_gb=48",
                "hydra.launcher.timeout_min=900",
                "hydra.sweep.dir=results/${run.name}/custom",
            ],
            "submitit_slurm",
            RunMode.MULTIRUN,
            900,
        ),
    ],
)
def test_offline_cli_preserves_launch_settings(
    tmp_path, monkeypatch, preset, overrides, launcher, mode, timeout
):
    def fail_submission(*args, **kwargs):
        raise AssertionError("Dataset generation must not submit Slurm jobs")

    monkeypatch.setattr(SlurmLauncher, "launch", fail_submission)
    with initialize_config_module(version_base=None, config_module="conf"):
        original = compose(
            "train",
            overrides=[f"+experiment/train={preset}", *overrides],
            return_hydra_config=True,
        )

    output = tmp_path / "data"
    simulate.main([
        f"+experiment/train={preset}",
        *overrides,
        "--output",
        str(output),
        "--train-size",
        "2",
        "--validation-size",
        "2",
        "--simulation-batch-size",
        "2",
    ])
    saved = OmegaConf.load(output / "training.yaml")
    assert set(saved.hydra) == {"mode", "launcher", "run", "sweep"}
    assert "hydra" not in OmegaConf.load(output / "simulation.yaml")

    with initialize_config_dir(version_base=None, config_dir=str(output)):
        restored = compose("training", return_hydra_config=True)
        local = compose(
            "training",
            overrides=["hydra.mode=RUN", "run.name=inside_allocation"],
            return_hydra_config=True,
        )
    assert restored.hydra.runtime.choices["hydra/launcher"] == launcher
    assert restored.hydra.mode == mode
    assert restored.hydra.launcher.timeout_min == timeout
    for name in ("launcher", "run", "sweep"):
        assert OmegaConf.to_container(
            restored.hydra[name], resolve=False
        ) == OmegaConf.to_container(original.hydra[name], resolve=False)
    assert restored.training.dataloader.name == "FileDataset"
    assert restored.training.max_train_steps == original.training.max_train_steps
    for name in ("train_loader", "val_loader"):
        expected = OmegaConf.to_container(original.training.dataloader[name])
        expected["drop_last"] = False
        assert restored.training.dataloader[name] == expected
    assert f"{original.run.name}_offline" in restored.hydra.sweep.dir
    assert local.hydra.mode == RunMode.RUN
    assert "inside_allocation" in local.hydra.run.dir
