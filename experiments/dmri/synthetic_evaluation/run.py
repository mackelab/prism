"""Evaluate a public dMRI checkpoint with synthetic SBC and relative KSD.

The scientific statistics follow the Figure 6 analysis in ``mackelab/dmri``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.dmri._shared.checkpoint import load_public_model as load_model
from experiments.dmri._shared.metrics import (
    make_ksd_fn,
    make_sbc_fn,
    summarize,
    validate_batch,
)
from experiments.dmri._shared.plotting import plots
from experiments.dmri._shared.storage import digest, source_provenance, write_json
from prism.utils.slurm import add_slurm_arguments, submit

REVISION = "017561ffbf77b109c9fe6c3d9c7d48831780ddc3"


def evaluate(
    model, simulator, functions, args, seq_index, condition_index, hyper, output, run
):
    root = jax.random.fold_in(jax.random.PRNGKey(args.seed), seq_index)
    root = jax.random.fold_in(root, condition_index)
    roots = [jax.random.fold_in(root, i) for i in range(3)]
    generator = jax.jit(
        jax.vmap(
            lambda key: simulator(
                key,
                mask_prior_hyperparameter=(
                    None if hyper is None else jnp.array([hyper])
                ),
            )
        )
    )
    batch_functions = {name: jax.jit(jax.vmap(fn)) for name, fn in functions.items()}
    chunks = []
    timings = []
    started = time.monotonic()
    for start in range(0, args.num_samples, args.batch_size):
        end = min(start + args.batch_size, args.num_samples)
        # Pad only the final compute batch; indexed keys preserve every real trial.
        indexes = jnp.arange(start, start + args.batch_size, dtype=jnp.uint32)
        keys = [
            jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, indexes)
            for key in roots
        ]
        batch_started = time.monotonic()
        data = generator(keys[0])
        values = {
            "model_mask": data["model_mask"],
            "theta_true": data["theta"],
            "mask_prior": data["mask_prior"],
        }
        for name, fn in batch_functions.items():
            values.update(fn(data, keys[1 if name == "sbc" else 2]))
        values = {k: np.asarray(v)[: end - start] for k, v in values.items()}
        duration = time.monotonic() - batch_started
        chunk_path = output / f"batch_{start:07d}.npz"
        np.savez_compressed(chunk_path, **values)
        validate_batch(values)
        chunks.append(values)
        timings.append(duration)
        if start == 0 or end == args.num_samples or end % 100 < args.batch_size:
            progress = {
                "completed": end,
                "target": args.num_samples,
                "elapsed_seconds": time.monotonic() - started,
                "last_batch_seconds": duration,
            }
            write_json(output / "progress.json", progress)
            print(json.dumps({"dataset": output.name, **progress}), flush=True)
            if run:
                run.log({
                    f"progress/{output.name}/datasets": end,
                    f"progress/{output.name}/elapsed_seconds": progress[
                        "elapsed_seconds"
                    ],
                })
    arrays = {
        key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]
    }
    raw = output / "samples.npz"
    np.savez_compressed(raw, **arrays)
    summary = summarize(arrays)
    summary.update({
        "hyperparameter": hyper,
        "sequence_index": seq_index,
        "elapsed_seconds": time.monotonic() - started,
        "first_batch_seconds_including_compilation": timings[0],
        "median_later_batch_seconds": float(np.median(timings[1:]))
        if len(timings) > 1
        else None,
        "samples_sha256": digest(raw),
        "batch_size": args.batch_size,
    })
    write_json(output / "summary.json", summary)
    label = (
        f"{args.which} EMA, {output.name}, n={args.num_samples}, {args.ksd_variant} KSD"
    )
    paths = plots(arrays, summary, output, label)
    if run:
        import wandb

        for name in ("model_pit", "theta_sum_pit", "theta_pit"):
            if summary.get(name):
                run.summary[f"{output.name}/{name}/calibration_error"] = summary[name][
                    "calibration_error"
                ]
        for path in paths:
            if path.suffix == ".png":
                run.log({f"plots/{output.name}/{path.stem}": wandb.Image(str(path))})
        columns = [
            "mask",
            "count",
            "theta_sum_ce",
            "theta_coordinate_ce",
            "rksd_mean",
            "rksd_median",
        ]
        rows = [
            [
                r["label"],
                r["count"],
                (r.get("theta_sum_pit") or {}).get("calibration_error"),
                (r.get("theta_pit") or {}).get("calibration_error"),
                r.get("rksd_mean"),
                r.get("rksd_median"),
            ]
            for r in summary["per_mask"]
        ]
        run.log({f"tables/{output.name}": wandb.Table(columns=columns, data=rows)})
        run.save(str(output / "summary.json"), base_path=str(args.output), policy="now")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--which", choices=("best", "latest"), default="best")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence", choices=("short", "long", "both"), default="both")
    parser.add_argument("--mode", choices=("sbc", "ksd", "both"), default="both")
    parser.add_argument("--mask-samples", type=int, default=200)
    parser.add_argument("--theta-samples", type=int, default=50)
    parser.add_argument("--sbc-steps", type=int, default=64)
    parser.add_argument("--sbc-t-min", type=float, default=None)
    parser.add_argument("--ksd-steps", type=int, default=128)
    parser.add_argument("--ksd-t-min", type=float, default=0.005)
    parser.add_argument(
        "--ksd-variant", choices=("frozen", "corrected"), default="frozen"
    )
    parser.add_argument(
        "--mask-hypers",
        default="",
        help="Additional fixed prior probabilities; overall is always included",
    )
    add_slurm_arguments(parser)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.wandb and not all((args.wandb_entity, args.wandb_project, args.run_id)):
        parser.error(
            "--wandb requires explicit --wandb-entity, --wandb-project and --run-id"
        )
    if (
        min(
            args.num_samples,
            args.batch_size,
            args.mask_samples,
            args.sbc_steps,
            args.ksd_steps,
        )
        < 1
        or args.theta_samples < 2
    ):
        parser.error("Counts must be positive; at least two theta samples are required")
    if args.slurm:
        return submit(args, argv, main)
    import prism

    source = Path(prism.__file__).resolve().parent.parent
    source_git = source_provenance(source)
    commit = source_git["commit"]
    args.output.mkdir(parents=True, exist_ok=False)
    model, sim_type, simulators, checkpoint_info = load_model(
        args.checkpoint, args.which
    )
    protocol = {
        "source_commit": commit,
        "source_git": source_git,
        "checkpoint_revision": REVISION,
        "checkpoint": checkpoint_info,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "python": platform.python_version(),
        "devices": [str(d) for d in jax.devices()],
        "device_details": [
            {"name": str(d), "kind": d.device_kind, "platform": d.platform}
            for d in jax.devices()
        ],
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy", "probjax")
        },
        "evaluation_source_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): digest(p)
            for directory in (
                Path(__file__).parent,
                Path(__file__).parents[1] / "_shared",
            )
            for p in sorted(directory.glob("*.py"))
        },
        "rng_policy": "seed -> fold_in(sequence, condition, stream, trial); stream 0=data, 1=SBC, 2=KSD; stable across batch sizes",
        "comparison_scope": "Public checkpoint scientific behavior; exact Figure 6 checkpoint identity unconfirmed",
        "sbc_definitions": {
            "model": "Frozen randomized rank of log model probability, approximate ties",
            "theta_sum": "Frozen randomized rank of sum over active latent parameters",
            "theta_coordinate": "Paper per-active-coordinate ranks, pooled; shared posterior draws with theta_sum",
        },
        "sampling": {
            "last_euler_step": False,
            "sbc_steps": args.sbc_steps,
            "sbc_t_min": args.sbc_t_min,
            "ksd_steps": args.ksd_steps,
            "ksd_t_min": args.ksd_t_min,
        },
        "rksd_definition": "Each rooted posterior KSD / mean rooted prior KSD for the same exact mask and acquisition group",
        "ksd_kernel": {
            "variant": args.ksd_variant,
            "score_transform": (
                "row scores @ L.T (frozen figure implementation)"
                if args.ksd_variant == "frozen"
                else "row scores @ L (corrected chain rule for z_rows=theta_rows@inv(L).T)"
            ),
            "bandwidths": [0.1, 0.5, 1.0, 10.0],
            "scale": 0.1,
            "covariance_jitter": 0.001,
            "wild_bootstrap_replicates": 512,
            "prior_samples": args.theta_samples,
            "preconditioning": "Separate empirical covariance for posterior and prior samples, active dimensions masked",
        },
        "rksd_plot_trimming": "Drop values at or above each mask's 99th percentile in violin plots only; raw values and summaries are untrimmed",
        "paper_definition_differences": [
            "Parameter-sum SBC follows frozen code; coordinate SBC follows paper",
            "New indexed RNG schedule; seed 0 is the script default, not proof of the original invocation",
            "Per-mask calibration in this pilot conditions the overall sample by mask; original right panel simulated a fixed count separately for each mask",
        ],
    }
    write_json(args.output / "protocol.json", protocol)
    print(
        json.dumps({
            "source_commit": commit,
            "checkpoint": {
                k: checkpoint_info[k]
                for k in (
                    "which",
                    "selected_step",
                    "params_name",
                    "theta_dim",
                    "simulator_class",
                )
            },
            "output": str(args.output),
        }),
        flush=True,
    )
    functions = {}
    if args.mode in ("sbc", "both"):
        functions["sbc"] = make_sbc_fn(
            model,
            mask_samples=args.mask_samples,
            theta_samples=args.theta_samples,
            num_steps=args.sbc_steps,
            t_min=args.sbc_t_min,
        )
    if args.mode in ("ksd", "both"):
        kernel = None
        if args.ksd_variant == "corrected":
            from experiments.dmri._shared.metrics import (
                corrected_multiscale_ksd_and_pvalue,
            )

            kernel = corrected_multiscale_ksd_and_pvalue
        functions["ksd"] = make_ksd_fn(
            model,
            sim_type,
            theta_samples=args.theta_samples,
            num_steps=args.ksd_steps,
            t_min=args.ksd_t_min,
            kernel=kernel,
        )
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            id=args.run_id,
            resume="never",
            mode="online",
            name=f"msb3s-{args.which}-{args.num_samples}"
            + ("-ksd-corrected" if args.ksd_variant == "corrected" else ""),
            group="msb3s-fig6",
            config=protocol,
            settings=wandb.Settings(disable_code=True),
        )
        (args.output / "wandb-url.txt").write_text(run.url + "\n")
    try:
        conditions = [None] + [
            float(v) for v in args.mask_hypers.split(",") if v.strip()
        ]
        if any(v is not None and not 0 <= v <= 1 for v in conditions):
            raise ValueError("Mask prior probabilities must lie in [0,1]")
        summaries = {}
        for seq_index in {"short": [0], "long": [1], "both": [0, 1]}[args.sequence]:
            for condition_index, hyper in enumerate(conditions):
                name = ("short" if seq_index == 0 else "long") + (
                    "_overall" if hyper is None else f"_prior_{hyper:g}"
                )
                output = args.output / name
                output.mkdir()
                summaries[name] = evaluate(
                    model,
                    simulators[seq_index],
                    functions,
                    args,
                    seq_index,
                    condition_index,
                    hyper,
                    output,
                    run,
                )
        write_json(
            args.output / "results.json",
            {"protocol": protocol, "results": summaries, "status": "completed"},
        )
        if run:
            run.save(
                str(args.output / "protocol.json"),
                base_path=str(args.output),
                policy="now",
            )
            run.save(
                str(args.output / "results.json"),
                base_path=str(args.output),
                policy="now",
            )
            run.summary["evaluation_status"] = "completed"
            run.finish()
    except Exception as error:
        write_json(
            args.output / "failure.json",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        if run:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
