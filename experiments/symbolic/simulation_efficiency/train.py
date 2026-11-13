"""Run a larger Figure 4b preset and record its successful step/time stop.

This invokes the shared trainer in the same process. Add ``--slurm a100`` to
submit the whole recording as a Slurm job.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
from hydra import compose, initialize_config_dir

from experiments.symbolic.simulation_efficiency.completion import validate_completion
from experiments.symbolic.simulation_efficiency.evaluate import (
    FIG4B,
    LARGER_POINTS,
    checkpoint_val_loss,
    latest_saved_step,
    training_log_paths,
    validate_fig4b,
)
from prism.train import train_script as shared_trainer
from prism.train.utils import load_cfg
from prism.utils.slurm import add_slurm_arguments, submit

ROOT = Path(__file__).resolve().parents[3]


def training_command(point, run_dir, overrides):
    """Compose and validate exactly the settings passed to the shared trainer."""
    run_dir = Path(run_dir).resolve()
    arguments = [
        f"+experiment/train=symbolic_sbmi_{point}",
        "infrastructure/launcher=local",
        "infrastructure/partition=none",
        "hydra.mode=RUN",
        *overrides,
        f"run.name={json.dumps(run_dir.name)}",
        f"hydra.run.dir={json.dumps(str(run_dir))}",
        "hydra.job.chdir=false",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = compose(config_name="train", overrides=arguments)
    # This validates the recipe, not completion; the actual saved step is
    # checked again after the process has returned.
    validate_fig4b(cfg, point, FIG4B[point][1] or 2000)
    if cfg.training.continue_training:
        raise ValueError(
            "This completion recorder starts a fresh run; resume separately."
        )
    return [sys.executable, "-m", "prism.train.train_script", *arguments]


def completion_record(point, run_dir, process):
    """Combine process evidence, saved configuration, steps, and terminal logs."""
    run_dir = Path(run_dir).resolve()
    cfg = load_cfg(str(run_dir))
    completed = latest_saved_step(run_dir / "checkpoints")
    validate_fig4b(cfg, point, completed)
    logs = training_log_paths(run_dir)
    if not any(
        f"Training limit reached at step {completed}." in path.read_text()
        for path in logs
    ):
        raise RuntimeError("Missing evidence of a normal training limit exit.")
    best = latest_saved_step(run_dir / "checkpoints" / "best")
    if best > completed:
        raise RuntimeError("Best step exceeds completed step.")
    record = {
        **process,
        "protocol_id": "fig4b-128-v1",
        "run_dir": str(run_dir),
        "configured_max_train_steps": cfg.training.max_train_steps,
        "configured_max_train_hours": cfg.training.max_train_hours,
        "stop_reason": "step_limit" if FIG4B[point][1] is not None else "time_limit",
        "completed_training_step": completed,
        "selected_best_checkpoint_step": best,
        "best_val_loss": checkpoint_val_loss(
            run_dir / "checkpoints" / "best" / str(best), required=True
        ),
        "training_logs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in logs
        ],
    }
    validate_completion(record, cfg, point, completed, run_dir)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", choices=LARGER_POINTS, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    add_slurm_arguments(parser)
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides, e.g. tracking.enabled=true"
    )
    args = parser.parse_args(argv)
    if args.slurm:
        return submit(args, argv, main)
    output = args.output.expanduser().resolve()
    command = training_command(args.point, output, args.overrides)
    output.mkdir(parents=True, exist_ok=False)
    source = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    start = time.monotonic()
    exit_code = 1
    original_argv = sys.argv
    try:
        # Keep JAX and W&B in one process, as in the existing cluster wrapper.
        # A second GPU process could compete with this process's allocation.
        sys.argv = command[2:]
        shared_trainer.main()
        exit_code = 0
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        raise
    finally:
        sys.argv = original_argv
        process = {
            "point": args.point,
            "process_exit_code": exit_code,
            "elapsed_training_seconds": time.monotonic() - start,
            "timer_scope": "Full training invocation, including data generation and compilation",
            "command": command,
            "source_commit": source.stdout.strip() if source.returncode == 0 else None,
            "working_tree_status": status.stdout if status.returncode == 0 else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "devices": [str(device) for device in jax.devices()],
        }
        (output / "training-process.json").write_text(
            json.dumps(process, indent=2, allow_nan=False) + "\n"
        )
    record = completion_record(args.point, output, process)
    with (output / "training-complete.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
