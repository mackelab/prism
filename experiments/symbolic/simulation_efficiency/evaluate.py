#!/usr/bin/env python3
"""Evaluate one PRISM checkpoint the way Figure 4b (SBMI comparison) does.

Replicates the original Figure 4b evaluation for a single run:
load the checkpoint (EMA params), draw `--num-samples` synthetic datasets from the
run's own simulator, sample a model mask and parameters from the network, simulate
the prediction, and report the RMSE against the observed data. The number reported
in the paper is this RMSE minus the noise floor `rmse_gt_mean` = 15.8048 stored with the original
Figure 4b results.

Usage (from the repo root):
    uv run python -m experiments.symbolic.simulation_efficiency.evaluate results/symbolic_sbmi_10k --fig4b-point 10k \
        --slurm a100 --output repro/fig4b_10k.json

EMA is mandatory. With --fig4b-point, the completed training step and evaluation
protocol are checked before evaluation. The JSON and adjacent NPZ retain both
steps, individual RMSEs, group means and separate comparisons with PRISM/SBMI.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from prism.utils.slurm import add_slurm_arguments, submit

from .._published import add_published_arguments, check_weights_source, load_published

RMSE_GT_MEAN = (
    15.80477523803711  # data_dict["rmse_gt_mean"] in rmse_n_training_big_v2.pkl
)
NOTEBOOK_SEED = 1110000000000  # jax.random.PRNGKey used in the notebook
GRID = jnp.arange(0.0, 10.0, 0.02)
FIG4B = {
    "10k": (10_000, 50_000, 1.584268682756616, 9.610879898071289),
    "25k": (25_000, 100_000, 1.6509112189614341, 6.373382568359375),
    "50k": (50_000, 200_000, 0.9695588748797037, 3.0993194580078125),
    "100k": (100_000, 500_000, 0.6671239693352534, 2.6385421752929688),
    "200k": (200_000, None, 0.5131398273070032, 1.8614387512207031),
    "500k": (500_000, None, 0.2612112356297004, 0.9344310760498047),
    "1000k": (1_000_000, None, 0.2119389819702875, 0.9478530883789062),
}
LARGER_POINTS = ("100k", "200k", "500k", "1000k")
ACCEPTANCE_VERSION = "fig4b-cshort-v1"
PRISM_MARGIN = 1.0
SBMI_FRACTION = 0.5


def assess_result(excess, prism, sbmi, role):
    prism_slack = PRISM_MARGIN - abs(excess - prism)
    sbmi_slack = SBMI_FRACTION * sbmi - excess
    return {
        "acceptance_version": ACCEPTANCE_VERSION,
        "prism_absolute_error_limit": PRISM_MARGIN,
        "sbmi_excess_fraction_limit": SBMI_FRACTION,
        "prism_criterion_met": prism_slack >= 0,
        "sbmi_criterion_met": sbmi_slack >= 0,
        "minimum_acceptance_slack": min(prism_slack, sbmi_slack),
        "assessment": "supplementary" if role == "supplementary" else "requires_review",
        "note": "Predeclared comparison criteria for human review; no automatic borderline trigger, followup, or reproduction verdict.",
    }


def checkpoint_val_loss(checkpoint_dir, required=False):
    path = Path(checkpoint_dir) / "val_loss"
    if not path.is_dir():
        if required:
            raise RuntimeError(
                f"Best checkpoint is missing its selection metric: {path}"
            )
        return None
    with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as reader:
        value = float(reader.restore(path))
    if not np.isfinite(value):
        raise RuntimeError(f"Non-finite checkpoint val_loss: {path}")
    return value


def check_evaluation_protocol(args):
    if not args.fig4b_point:
        return "diagnostic"
    if (
        args.which not in ("best", "latest")
        or args.num_samples != 1000
        or args.groups != 10
        or args.seed != NOTEBOOK_SEED
        or args.lam is not None
    ):
        raise ValueError(
            f"Figure 4b requires best/latest, 1000 datasets, 10 groups, "
            f"seed {NOTEBOOK_SEED} and no --lam"
        )
    return "primary" if args.which == "best" else "supplementary"


def require_ema(checkpoint):
    if checkpoint is None or checkpoint.get("params_ema") is None:
        raise RuntimeError(
            "EMA weights are missing; raw-parameter fallback is disabled."
        )
    if checkpoint.get("step") is None:
        raise RuntimeError("The restored checkpoint does not record its step.")
    return checkpoint["params_ema"]


def latest_saved_step(checkpoint_dir):
    """Read a finalized checkpoint's stored step, not a configured training cap."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"Checkpoint directory is missing: {checkpoint_dir}")
    options = ocp.CheckpointManagerOptions(read_only=True, create=False)
    with ocp.CheckpointManager(checkpoint_dir, options=options) as manager:
        latest = manager.latest_step()
    if latest is None:
        raise RuntimeError(f"No finalized checkpoint in {checkpoint_dir}")
    with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as reader:
        stored = int(reader.restore(checkpoint_dir / str(latest) / "step"))
    if stored != latest:
        raise RuntimeError(
            f"Checkpoint directory step {latest} disagrees with stored step {stored}."
        )
    return stored


def validate_fig4b(cfg, point, completed_step):
    _, cap, _, _ = FIG4B[point]
    if cap is not None and completed_step != cap:
        raise RuntimeError(
            f"Training incomplete or over budget: saved step {completed_step}, required {cap}."
        )
    if cap is None:
        if completed_step < 2000:
            raise RuntimeError("Training did not reach its first validation interval.")
        if cfg.training.max_train_hours != 8:
            raise RuntimeError("The larger time-limited runs use eight training hours.")
    config_dir = Path(__file__).resolve().parents[3] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        reference = compose(
            config_name="train", overrides=[f"+experiment/train=symbolic_sbmi_{point}"]
        )

    # Compare all scientific settings, including optional overrides. Hardware
    # scheduling, W&B, run names, retention, and ordinary resume are operational.
    def scientific_settings(config):
        selected = {
            key: OmegaConf.to_container(config[key], resolve=True)
            for key in ("simulator", "model", "training")
        }
        selected["run.seed"] = config.run.seed
        for key in ("max_train_hours", "continue_training", "checkpoint"):
            selected["training"].pop(key, None)
        return selected

    def compare(actual, expected, path=""):
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in sorted(set(expected) | set(actual)):
                item_path = f"{path}.{key}" if path else key
                if key not in actual or key not in expected:
                    raise RuntimeError(
                        f"Figure 4b config mismatch at {item_path}: unexpected or missing setting."
                    )
                compare(actual[key], expected[key], item_path)
        elif actual != expected:
            raise RuntimeError(
                f"Figure 4b config mismatch: {path}={actual!r}, expected {expected!r}."
            )

    compare(scientific_settings(cfg), scientific_settings(reference))


def training_log_paths(run_dir):
    """Locate current single-run/multirun logs and logs kept from the original runs."""
    root = Path(run_dir)
    files = set()
    for name in ("train_script.log", "hydra_script.log"):
        for pattern in (name, f"*/{name}", f"*/0/{name}"):
            files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def prediction_rmse(model, simulator, sim_type, key, lam=None):
    """One notebook draw, including fresh observation and prediction noise."""
    key1, key2, key3, key4 = jax.random.split(key, 4)
    batch = simulator(key1)
    if lam is None:
        model_mask_pred = model.sample_mask(key2, x=batch["x"])
    else:
        model_mask_pred = model.sample_mask(
            key2, x=batch["x"], mask_prior=jnp.array([lam])
        )
    theta_pred = model.sample_theta(key3, x=batch["x"], model_mask=model_mask_pred)
    sim = sim_type.from_theta(theta_pred, model_mask=model_mask_pred)
    prediction = sim.evaluate(GRID, rng=key4).flatten()
    observed = batch["x"].flatten()
    if prediction.shape != observed.shape:
        raise ValueError(
            "Figure 4b evaluation requires 500 signal values on the notebook grid."
        )
    return jnp.mean((prediction - observed) ** 2) ** 0.5


def summarize(rmses, groups, point=None, role="primary"):
    if rmses.ndim != 1 or groups <= 0 or rmses.size == 0 or rmses.size % groups:
        raise ValueError(
            "RMSE samples must split evenly into positive, nonempty groups."
        )
    # Keep JAX reductions, grouping and ddof=0 exactly as in notebook cell 7.
    bs = jnp.asarray(rmses).reshape(groups, -1).mean(axis=1)
    if (
        not np.isfinite(np.asarray(rmses)).all()
        or not np.isfinite(np.asarray(bs)).all()
    ):
        raise RuntimeError("Evaluation produced non-finite RMSEs or group means.")
    mean = float(bs.mean())
    std = float(bs.std())
    if not np.isfinite([mean, std]).all():
        raise RuntimeError("Evaluation produced non-finite summary metrics.")
    excess = mean - RMSE_GT_MEAN
    result = {
        "rmse_mean": mean,
        "rmse_std_over_groups": std,
        "excess_rmse_over_gt": excess,
        "rmse_gt_mean": RMSE_GT_MEAN,
        "groups": groups,
        "samples_per_group": int(rmses.size // groups),
        "group_means": np.asarray(bs).tolist(),
    }
    if point:
        _, _, prism, sbmi = FIG4B[point]
        assessment = (
            {
                "assessment": "supplementary"
                if role == "supplementary"
                else "requires_review",
                "note": "No acceptance margin is declared for these larger runs; compare the recorded PRISM and SBMI references directly.",
            }
            if point in LARGER_POINTS
            else assess_result(excess, prism, sbmi, role)
        )
        result["comparison"] = {
            "paper_prism_excess_rmse_from_svg": prism,
            "sbmi_excess_rmse_from_pickle": sbmi,
            "difference_from_paper_prism": excess - prism,
            "absolute_difference_from_paper_prism": abs(excess - prism),
            "improvement_over_sbmi": sbmi - excess,
            "fractional_reduction_in_excess_rmse_vs_sbmi": (sbmi - excess) / sbmi,
            **assessment,
        }
    return result


def tree_sha256(directory):
    """Hash relative names and contents of the selected, finalized checkpoint."""
    digest = hashlib.sha256()
    directory = Path(directory)
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        with path.open("rb") as f:
            content_digest = hashlib.file_digest(f, "sha256").hexdigest()
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        digest.update(content_digest.encode() + b"\n")
    return digest.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "checkpoint",
        nargs="?",
        help="run directory, e.g. results/symbolic_sbmi_10k; a single-job Slurm "
        "run resolves to its newest <timestamp>/<job> directory (or --preset)",
    )
    p.add_argument(
        "--which", default="best", help="best | latest | <int step> (figure used best)"
    )
    p.add_argument("--num-samples", type=int, default=1000, help="notebook: 1000")
    p.add_argument(
        "--groups", type=int, default=10, help="notebook: 10 disjoint groups of 100"
    )
    p.add_argument("--seed", type=int, default=NOTEBOOK_SEED)
    p.add_argument("--output", default=None, help="optional .json to write")
    p.add_argument(
        "--fig4b-point",
        choices=FIG4B,
        help="enforce the selected training recipe and notebook protocol",
    )
    p.add_argument(
        "--completion",
        type=Path,
        help="training completion JSON; required for the four larger points",
    )
    add_published_arguments(p)
    add_slurm_arguments(p)
    p.add_argument(
        "--lam",
        type=float,
        default=None,
        help="optional prior input for diagnostic evaluation; omit for Figure 4b",
    )
    args = p.parse_args(argv)
    check_weights_source(p, args.checkpoint, args.preset, args.weights_dir)
    if args.preset:
        if args.completion:
            p.error("--completion applies to training runs, not --preset")
        if args.which not in ("best", "latest"):
            p.error("--preset supports --which best or latest")
        if args.fig4b_point and args.preset != f"symbolic_sbmi_{args.fig4b_point}":
            p.error(
                f"--fig4b-point {args.fig4b_point} needs "
                f"--preset symbolic_sbmi_{args.fig4b_point}"
            )
    if args.num_samples <= 0 or args.groups <= 0 or args.num_samples % args.groups:
        p.error("--num-samples must be positive and divisible by positive --groups")
    try:
        role = check_evaluation_protocol(args)
    except ValueError as exc:
        p.error(str(exc))
    output = Path(args.output) if args.output else None
    if output and (
        output.suffix != ".json"
        or output.exists()
        or output.with_suffix(".npz").exists()
    ):
        p.error(
            "Use a new .json output path; existing result files are not overwritten"
        )
    if args.slurm:
        return submit(args, argv, main)

    # Deferred import allows metadata/metric checks without importing the model.
    from prism.train.build_simulator import build_simulator
    from prism.train.utils import find_checkpoint_run_dir, load_cfg, load_checkpoint

    which = int(args.which) if args.which.isdigit() else args.which
    if args.preset:
        bundle, published = load_published(args.preset, which, args.weights_dir)
        model, sim_type, simulators = bundle.model, bundle.simulator, [bundle.generator]
        model.eval()
        config_snapshot = OmegaConf.to_container(
            OmegaConf.create(bundle.config), resolve=False
        )
        json.dumps(config_snapshot, allow_nan=False)
        provenance = {
            "checkpoint": None,
            "published_weights": published,
            "which": args.which,
            "restored_step": bundle.step,
            "evaluation_role": role,
            "params_used": "ema",
            "fig4b_point": args.fig4b_point,
            # The evaluation protocol is enforced; training evidence is not available.
            "fig4b_protocol_checks_passed": False,
            "fig4b_evaluation_protocol_checked": bool(args.fig4b_point),
            "saved_config": config_snapshot,
            "config_interpolations_resolved": False,
        }
        completion_evidence = None
    else:
        run_dir = find_checkpoint_run_dir(Path(args.checkpoint).resolve())
        completed_step = latest_saved_step(run_dir / "checkpoints")
        cfg = load_cfg(str(run_dir))
        # Capture unresolved config before inference: unrelated Hydra interpolations
        # must never discard an otherwise completed, expensive evaluation.
        config_snapshot = OmegaConf.to_container(cfg, resolve=False)
        json.dumps(config_snapshot, allow_nan=False)  # Fail serialization early.
        if args.fig4b_point:
            validate_fig4b(cfg, args.fig4b_point, completed_step)
        completion_evidence = None
        if args.fig4b_point in LARGER_POINTS:
            from experiments.symbolic.simulation_efficiency.completion import (
                read_completion,
                validate_completion,
            )

            completion_path = args.completion or run_dir / "training-complete.json"
            completion, completion_evidence = read_completion(completion_path)
            validate_completion(
                completion, cfg, args.fig4b_point, completed_step, run_dir
            )
        elif args.completion:
            p.error("--completion is used only with a larger --fig4b-point")
        best_step = (
            latest_saved_step(run_dir / "checkpoints" / "best")
            if args.fig4b_point or which == "best"
            else None
        )
        best_dir = run_dir / "checkpoints" / "best" / str(best_step)
        best_val_loss = (
            checkpoint_val_loss(best_dir, required=bool(args.fig4b_point))
            if best_step is not None
            else None
        )
        selected_step = (
            best_step
            if which == "best"
            else completed_step
            if which == "latest"
            else which
        )
        selected_dir = run_dir / "checkpoints"
        if which == "best":
            selected_dir /= "best"
        selected_dir /= str(selected_step)
        val_loss = (
            best_val_loss if which == "best" else checkpoint_val_loss(selected_dir)
        )
        selected_hash = tree_sha256(selected_dir)
        training_logs = training_log_paths(run_dir)
        if args.fig4b_point and not training_logs:
            raise RuntimeError(
                "Required training log with validation-curve evidence is missing."
            )
        if args.fig4b_point in LARGER_POINTS and not any(
            f"Training limit reached at step {completed_step}." in path.read_text()
            for path in training_logs
        ):
            raise RuntimeError("Missing evidence of a normal training limit exit.")
        if not (selected_dir / "params_ema").is_dir():
            raise RuntimeError(
                "EMA weights are missing; raw-parameter fallback is disabled."
            )
        checkpoint, model, sim_type = load_checkpoint(str(run_dir), which=which)
        params = require_ema(checkpoint)
        restored_step = int(checkpoint["step"])
        if restored_step != selected_step or restored_step > completed_step:
            raise RuntimeError(
                "Restored step disagrees with the selected/completed checkpoint steps."
            )
        nnx.update(model, params)
        model.eval()
        if args.fig4b_point in LARGER_POINTS:
            from experiments.symbolic.simulation_efficiency.completion import (
                validate_model_bounds,
            )

            validate_model_bounds(model)
        generated_sim_type, simulators = build_simulator(cfg)
        if generated_sim_type is not sim_type or len(simulators) != 1:
            raise RuntimeError(
                "Evaluation simulator differs from the restored model's simulator."
            )
        provenance = {
            "checkpoint": str(run_dir),
            "which": args.which,
            "completed_training_step": completed_step,
            "completed_training_step_evidence": "stored step in latest finalized standard checkpoint",
            "selected_best_checkpoint_step": best_step,
            "restored_step": restored_step,
            "val_loss": val_loss,
            "best_checkpoint_val_loss": best_val_loss,
            "checkpoint_selection_metric": "mask NLL + theta NLL; lower is better",
            "evaluation_role": role,
            "params_used": "ema",
            "fig4b_point": args.fig4b_point,
            "configured_max_train_steps": OmegaConf.select(
                cfg, "training.max_train_steps"
            ),
            "fig4b_protocol_checks_passed": bool(args.fig4b_point),
            "selected_checkpoint_sha256": selected_hash,
            "completed_checkpoint_sha256": tree_sha256(
                run_dir / "checkpoints" / str(completed_step)
            ),
            "saved_config": config_snapshot,
            "config_interpolations_resolved": False,
            "training_logs": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in training_logs
            ],
        }

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_samples)
    rmses = jax.vmap(
        lambda key: prediction_rmse(model, simulators[0], sim_type, key, args.lam)
    )(keys)
    metrics = summarize(rmses, args.groups, args.fig4b_point, role)
    source = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    result = {
        **provenance,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "lambda": args.lam,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": source.stdout.strip() if source.returncode == 0 else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "devices": [str(d) for d in jax.devices()],
        "python_version": platform.python_version(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy")
        },
        **metrics,
    }
    if completion_evidence is not None:
        result["training_completion"] = completion_evidence
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        samples_path = output.with_suffix(".npz")
        np.savez_compressed(
            samples_path,
            rmses=np.asarray(rmses),
            group_means=np.asarray(metrics["group_means"]),
            keys=np.asarray(keys),
        )
        result["samples_file"] = str(samples_path.resolve())
        result["samples_sha256"] = hashlib.sha256(samples_path.read_bytes()).hexdigest()
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
