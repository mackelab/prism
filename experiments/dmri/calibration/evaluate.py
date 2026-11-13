"""Evaluate condition-wise model/parameter SBC with the frozen Figure 6b recipe."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import uuid
from pathlib import Path

import jax

from experiments.dmri._shared.checkpoint import load_run_model as load_model
from experiments.dmri._shared.metrics import (
    make_model_sbc_fn,
    make_theta_sbc_fn,
)
from experiments.dmri._shared.storage import source_provenance
from prism.train.utils import find_checkpoint_run_dir
from prism.utils.slurm import add_slurm_arguments, submit

from .chunks import (
    canonical_hash,
    evaluate_sequence,
    file_sha256,
    finalize_sequence,
    prepare_output,
    write_new_json,
)
from .protocol import aggregate, aggregate_plot, conditions, make_batch_function

PROTOCOL_ID = "dmri-condition-calibration-v1"
ROOT = Path(__file__).resolve().parents[3]


def run_label(checkpoint):
    """Name a run by ``run.name``, also for a ``<run.name>/<timestamp>/<job>`` path."""
    if checkpoint.name.isdigit() and len(checkpoint.parents) > 1:
        return checkpoint.parents[1].name
    return checkpoint.name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("model", "theta"), required=True)
    parser.add_argument("--which", choices=("latest", "best"), default="latest")
    parser.add_argument("--expected-step", type=int)
    parser.add_argument(
        "--trials", type=int, help="Datasets per condition: model 100000, theta 10000"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--mask-samples", type=int, default=200)
    parser.add_argument("--theta-samples", type=int, default=50)
    parser.add_argument("--ode-steps", type=int, default=64)
    parser.add_argument("--t-min", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    add_slurm_arguments(parser)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.trials is None:
        args.trials = 100000 if args.mode == "model" else 10000
    if (
        min(
            args.trials,
            args.batch_size,
            args.chunk_size,
            args.mask_samples,
            args.theta_samples,
            args.ode_steps,
        )
        < 1
        or args.trials >= 10_000_000
    ):
        parser.error("Counts must be positive and trials less than 10 million")
    if args.chunk_size % args.batch_size:
        parser.error("chunk-size must be divisible by batch-size")
    if not 0 <= args.t_min < 80:
        parser.error("t-min must be finite and in [0, 80)")
    if args.expected_step is not None and args.expected_step < 1:
        parser.error("expected-step must be positive")
    if args.wandb and not all((args.wandb_entity, args.wandb_project, args.run_id)):
        parser.error("W&B requires an explicit entity, project, and run-id")
    args.checkpoint = find_checkpoint_run_dir(args.checkpoint.expanduser().resolve())
    args.output = args.output.expanduser().resolve()
    if args.output == args.checkpoint:
        parser.error("Use a separate evaluation output directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.slurm:
        return submit(args, argv, main)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume: {args.output}")
    model, sim_type, simulators, checkpoint = load_model(
        args.checkpoint, args.which, args.expected_step
    )
    cases = conditions(args.mode)
    source_dirs = [ROOT / "prism", ROOT / "experiments/dmri"]
    sources = {
        "checkout": source_provenance(ROOT),
        "files": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for directory in source_dirs
            for path in sorted(directory.rglob("*.py"))
        },
    }
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "sources": sources,
        "checkpoint": checkpoint,
        "mode": args.mode,
        "evaluation_seed": 0,
        "trials_per_condition": args.trials,
        "conditions": cases,
        "total_synthetic_datasets": args.trials * len(cases),
        "chunk_size": args.chunk_size,
        "tracking": {
            "enabled": args.wandb,
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "run_id": args.run_id,
        },
        "sampling": {
            "mask_samples": args.mask_samples if args.mode == "model" else 0,
            "theta_samples": args.theta_samples if args.mode == "theta" else 0,
            "ode_steps": args.ode_steps if args.mode == "theta" else None,
            "t_min": args.t_min if args.mode == "theta" else None,
            "last_euler_step": False,
        },
        "paper_statistic": "randomized log-model-probability rank"
        if args.mode == "model"
        else "randomized rank of sum of active standard-normal latent parameters",
        "supplementary_statistic": "coordinate midpoint rank from the same parameter draws"
        if args.mode == "theta"
        else None,
        "rng_policy": "Frozen ICML seed 0: two sequential outer splits per condition; random.split(condition_key, N) per trial. Parameter SHORT fixed masks start after 22 outer splits and LONG after 74. No combined-evaluator outer split is applied to the PIT key.",
        "scope": "Figure 6b condition allocation and aggregation on the explicitly selected checkpoint and its saved simulator; checkpoint identity is not inferred.",
        "sampling_matches_source_recipe": args.trials
        == (100000 if args.mode == "model" else 10000)
        and (
            args.mask_samples == 200
            if args.mode == "model"
            else args.theta_samples == 50
            and args.ode_steps == 64
            and args.t_min == 1e-4
        ),
        "wandb_uploads": "Protocol, summaries and plots only; no checkpoints or trial arrays",
    }
    output = args.output
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another full SBC evaluator already owns this output"
            ) from error
        protocol_hash = prepare_output(output, protocol, resume=True)
        attempts = output / "attempts"
        attempts.mkdir(exist_ok=True)
        attempt = uuid.uuid4().hex
        write_new_json(
            attempts / f"{attempt}.json",
            {
                "hostname": platform.node(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "python": platform.python_version(),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "jax",
                        "jaxlib",
                        "flax",
                        "orbax-checkpoint",
                        "numpy",
                        "probjax",
                    )
                },
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "devices": [str(device) for device in jax.devices()],
            },
        )
        sbc = (
            make_model_sbc_fn(model, mask_samples=args.mask_samples, source_keys=True)
            if args.mode == "model"
            else make_theta_sbc_fn(
                model,
                theta_samples=args.theta_samples,
                num_steps=args.ode_steps,
                t_min=args.t_min,
                source_keys=True,
            )
        )
        measure = jax.jit(jax.vmap(sbc))
        run = None
        try:
            if args.wandb:
                import wandb

                os.environ["WANDB_DISABLE_CODE"] = "true"
                os.environ["WANDB_SAVE_CODE"] = "false"
                run = wandb.init(
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    id=args.run_id,
                    resume="allow" if args.resume else "never",
                    mode="online",
                    name=f"{run_label(args.checkpoint)}-{args.which}-Figure6b-{args.mode}",
                    group=PROTOCOL_ID,
                    job_type="calibration",
                    config=protocol,
                    settings=wandb.Settings(disable_code=True),
                )
            summaries = {}
            for case in cases:
                name, sequence = case["name"], case["sequence"]
                destination = output / name
                case_hash = canonical_hash({
                    "protocol": protocol_hash,
                    "condition": case,
                })

                def progress(values, dataset=name):
                    if run:
                        run.log({
                            f"progress/{dataset}/datasets": values["completed_trials"],
                            f"progress/{dataset}/last_chunk_seconds": values[
                                "last_chunk_seconds"
                            ],
                        })

                compute = make_batch_function(
                    simulators[sequence],
                    sbc,
                    sequence=sequence,
                    condition_index=case["condition_index"],
                    hyper=case["hyper"],
                    mask=case["mask"],
                    trials=args.trials,
                    measure=measure,
                )
                arrays, chunks = evaluate_sequence(
                    destination,
                    compute,
                    trials=args.trials,
                    batch_size=args.batch_size,
                    chunk_size=args.chunk_size,
                    sequence=sequence,
                    protocol_hash=case_hash,
                    on_chunk=progress,
                )
                summary = finalize_sequence(
                    destination,
                    arrays,
                    chunks,
                    sequence=sequence,
                    protocol_hash=case_hash,
                    label=f"{run_label(args.checkpoint)}, {args.which} EMA, {name}, N={args.trials}",
                )
                if summary["completed_trials"] != args.trials:
                    raise ValueError(
                        "Condition completion does not match the full trial count"
                    )
                summaries[name] = summary
                if run:
                    for field in ("model_pit", "theta_sum_pit", "theta_pit"):
                        if field in summary:
                            run.summary[f"{name}/{field}/calibration_error"] = summary[
                                field
                            ]["calibration_error"]
                    run.save(
                        str(destination / "summary" / "summary.json"),
                        base_path=str(output),
                        policy="now",
                    )
                del arrays
            combined = aggregate(args.mode, summaries)
            result = {
                "status": "completed",
                "protocol": protocol,
                "results": summaries,
                "aggregation": combined,
                "exact_total_datasets": args.trials * len(cases),
            }
            result_path = output / "results.json"
            if result_path.exists():
                if json.loads(result_path.read_text()) != result:
                    raise ValueError(
                        "Existing final results differ from verified condition summaries"
                    )
            else:
                write_new_json(result_path, result)
            aggregate_plot(combined, output, args.mode)
            if run:
                import wandb

                run.summary["evaluation_status"] = "completed"
                run.summary["exact_total_datasets"] = result["exact_total_datasets"]
                run.summary["figure6b/calibration_error"] = combined[
                    "calibration_error"
                ]
                run.log({
                    f"plots/figure6b_{args.mode}": wandb.Image(
                        str(output / f"figure6b-{args.mode}.png")
                    )
                })
                for filename in ("protocol.json", "results.json"):
                    run.save(
                        str(output / filename), base_path=str(output), policy="now"
                    )
                run.finish()
        except Exception as error:
            write_new_json(
                attempts / f"{attempt}-failure.json",
                {"error_type": type(error).__name__, "message": str(error)},
            )
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
