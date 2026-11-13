"""`--slurm` submission for experiment scripts, without a cluster."""

import pytest

from experiments.symbolic.model_space_scaling import evaluate as scaling_evaluate
from prism.utils import slurm


def test_partition_presets_are_the_slurm_choices():
    assert slurm.partitions() == ["2h100_long", "a100", "h100_long"]


def test_a100_settings_map_to_submitit_parameters():
    init, update = slurm.submitit_parameters(
        slurm.launcher_settings("a100", name="job")
    )

    assert init == {"slurm_max_num_timeout": 0}
    assert update["timeout_min"] == 700
    assert update["cpus_per_task"] == 4
    assert update["mem_gb"] == 32
    assert update["name"] == "job"
    assert update["slurm_partition"] == "a100-galvani"
    assert update["slurm_gres"] == "gpu"
    assert not {"_target_", "submitit_folder", "slurm_submitit_folder"} & set(update)


def test_two_gpu_preset_and_timeout_override():
    _, update = slurm.submitit_parameters(
        slurm.launcher_settings("2h100_long", timeout_min=15)
    )

    assert update["slurm_gres"] == "gpu:2"
    assert update["slurm_partition"] == "h100-ferranti"
    assert update["timeout_min"] == 15


def test_strip_slurm_arguments():
    argv = ["ckpt", "--slurm", "a100", "--output", "x.json", "--slurm-timeout-min=5"]

    assert slurm.strip_slurm_arguments(argv) == ["ckpt", "--output", "x.json"]


def test_script_submits_itself_before_evaluating(tmp_path, monkeypatch):
    submitted = []

    class FakeExecutor:
        def submit(self, function, *args):
            submitted.append((function, args))
            return type(
                "Job", (), {"job_id": "1", "paths": type("P", (), {"folder": "f"})}
            )

    monkeypatch.setattr(slurm, "_executor", lambda *args: FakeExecutor())
    output = tmp_path / "out.json"
    argv = [str(tmp_path / "ckpt"), "--output", str(output), "--slurm", "a100"]

    scaling_evaluate.main(argv)

    assert submitted == [
        (
            slurm.run_in_job,
            (
                "experiments.symbolic.model_space_scaling.evaluate",
                [str(tmp_path / "ckpt"), "--output", str(output)],
            ),
        )
    ]
    assert not output.exists()


def test_job_requires_a_slurm_gpu_backend(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="GPU backend"):
        slurm.run_in_job("experiments.symbolic.model_space_scaling.evaluate", [])


def test_job_runs_the_module_main(monkeypatch):
    import jax

    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(scaling_evaluate, "main", calls.append)

    slurm.run_in_job("experiments.symbolic.model_space_scaling.evaluate", ["a"])

    assert calls == [["a"]]
