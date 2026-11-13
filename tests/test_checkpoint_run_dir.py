import pytest

from prism.train.utils import find_checkpoint_run_dir


def saved_step(run_dir, step=10):
    (run_dir / "checkpoints" / str(step)).mkdir(parents=True)
    return run_dir


def slurm_job(root, timestamp, job="0", step=None):
    run = root / timestamp
    (run / ".submitit" / "123").mkdir(parents=True, exist_ok=True)
    (run / "multirun.yaml").write_text("{}")
    job_dir = run / job
    (job_dir / ".hydra").mkdir(parents=True)
    if step is None:
        # A crashed or pending job: the manager already created empty folders.
        (job_dir / "checkpoints" / "best").mkdir(parents=True)
    else:
        saved_step(job_dir, step)
    return job_dir


def test_single_run_is_returned_unchanged(tmp_path):
    root = saved_step(tmp_path / "run")
    (root / "2026-01-01_00-00-00" / ".hydra").mkdir(parents=True)
    assert find_checkpoint_run_dir(root) == root


def test_missing_or_unrelated_paths_are_returned_unchanged(tmp_path):
    assert find_checkpoint_run_dir(tmp_path / "missing") == tmp_path / "missing"
    (tmp_path / "empty").mkdir()
    assert find_checkpoint_run_dir(tmp_path / "empty") == tmp_path / "empty"


def test_single_job_multirun_resolves_to_newest_job(tmp_path):
    root = tmp_path / "run"
    slurm_job(root, "2026-01-01_00-00-00", step=5)
    newest = slurm_job(root, "2026-02-01_00-00-00", step=7)
    assert find_checkpoint_run_dir(root) == newest
    assert find_checkpoint_run_dir(newest) == newest


def test_orbax_temporary_directories_are_not_saved_steps(tmp_path):
    root = tmp_path / "run"
    job = slurm_job(root, "2026-01-01_00-00-00")
    (job / "checkpoints" / "10.orbax-checkpoint-tmp-1").mkdir()
    with pytest.raises(FileNotFoundError, match="no saved checkpoint"):
        find_checkpoint_run_dir(root)


def test_newest_job_without_checkpoint_is_not_replaced_by_older_run(tmp_path):
    root = tmp_path / "run"
    older = slurm_job(root, "2026-01-01_00-00-00", step=5)
    slurm_job(root, "2026-02-01_00-00-00")
    with pytest.raises(FileNotFoundError, match=str(older)):
        find_checkpoint_run_dir(root)


def test_sweep_requires_an_explicit_job(tmp_path):
    root = tmp_path / "run"
    slurm_job(root, "2026-01-01_00-00-00", job="0", step=5)
    slurm_job(root, "2026-01-01_00-00-00", job="1", step=5)
    with pytest.raises(ValueError, match="contains 2 jobs"):
        find_checkpoint_run_dir(root)


def test_empty_single_run_checkpoints_do_not_shadow_a_job(tmp_path):
    root = tmp_path / "run"
    (root / "checkpoints" / "best").mkdir(parents=True)
    job = slurm_job(root, "2026-01-01_00-00-00", step=5)
    assert find_checkpoint_run_dir(root) == job


def test_single_run_checkpoints_win_over_jobs(tmp_path, caplog):
    root = saved_step(tmp_path / "run")
    slurm_job(root, "2026-01-01_00-00-00", step=5)
    assert find_checkpoint_run_dir(root) == root
    assert "timestamped job directories" in caplog.text
