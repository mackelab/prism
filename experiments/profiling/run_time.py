"""Run time and memory of the model operations of a trained PRISM checkpoint.

Works for any saved run (symbolic or dMRI). It profiles, for one simulated
observation, drawing a model mask, its log probability, drawing parameters
given the mask, and the parameter log probability; and, for a simulated batch,
the training loss forward and forward-plus-backward pass. Inputs come from the
run's own simulator, and weights are the EMA parameters.

Example (GPU, from the repository root):

    python -m experiments.profiling.run_time results/symbolic50 \
        --output results/symbolic50/run_time_gpu.json

Set ``JAX_PLATFORMS=cpu`` to time on the CPU (use ``--skip-training``: the
training step is not meaningful there). Compile times include XLA compilation
of each operation, and the first call is reported separately from steady state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from prism.utils.slurm import add_slurm_arguments, submit

from .timing import profile_function

OPERATIONS = (
    "sample_mask",
    "log_prob_mask",
    "sample_theta",
    "log_prob_theta",
    "loss_forward",
    "loss_backward",
)
COMPONENTS = ("encoder", "inference_decoder", "model_decoder", "tokenizer")


def count_parameters(tree: Any) -> int:
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(tree)))


def parameter_counts(params: Any) -> dict[str, int]:
    counts = {"total": count_parameters(params)}
    counts.update({
        name: count_parameters(params[name]) for name in COMPONENTS if name in params
    })
    return counts


def load_run(path: Path, which: str):
    """Return ``(model, generators, run metadata)`` with EMA weights applied."""
    from prism.train.build_simulator import build_simulator
    from prism.train.utils import load_cfg, load_checkpoint, select_checkpoint_params

    checkpoint, model, _ = load_checkpoint(path, which=which)
    nnx.update(model, select_checkpoint_params(checkpoint, "params_ema"))
    model.eval()
    _, generators = build_simulator(load_cfg(path))
    return (
        model,
        generators,
        {
            "checkpoint": str(path),
            "which": which,
            "step": int(checkpoint["step"]),
        },
    )


def build_cases(
    model,
    generator,
    *,
    seed: int = 0,
    theta_num_steps: int = 64,
    loss_batch_size: int = 2048,
    include_training: bool = True,
) -> dict[str, tuple]:
    """Map operation name to ``(function, arguments)`` for :func:`profile_function`.

    ``generator`` is one of the run's simulator generators, ``key -> dict`` with
    ``x``, ``model_mask``, ``theta``, ``mask_prior`` and, for dMRI, ``acq``.
    """
    key_example, key_batch, *keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    example = jax.block_until_ready(generator(key_example))
    # Only the dMRI model takes an acquisition design.
    design = {"acq": example["acq"]} if "acq" in example else {}
    x, mask, theta, prior = (
        example[k] for k in ("x", "model_mask", "theta", "mask_prior")
    )

    cases = {
        "sample_mask": (
            lambda key: model.sample_mask(key, x=x, mask_prior=prior, **design),
            (keys[0],),
        ),
        "log_prob_mask": (
            lambda: model.log_prob_mask(mask, x=x, mask_prior=prior, **design),
            (),
        ),
        "sample_theta": (
            lambda key: model.sample_theta(
                key, x=x, model_mask=mask, num_steps=theta_num_steps, **design
            ),
            (keys[1],),
        ),
        "log_prob_theta": (
            lambda: model.log_prob_theta(
                theta, x=x, model_mask=mask, num_steps=theta_num_steps, **design
            ),
            (),
        ),
    }
    if include_training:
        if loss_batch_size < 1:
            raise ValueError("loss_batch_size must be >= 1")
        batch = jax.block_until_ready(
            jax.vmap(generator)(jax.random.split(key_batch, loss_batch_size))
        )
        graphdef, params, state = nnx.split(model, nnx.Param, ...)

        def loss(params, key, batch):
            training_model = nnx.merge(graphdef, params, state, copy=True)
            training_model.train()
            return jnp.sum(training_model.loss_fn(key, **batch))

        cases["loss_forward"] = (loss, (params, keys[2], batch))
        cases["loss_backward"] = (
            lambda params, key, batch: jax.value_and_grad(loss)(params, key, batch),
            (params, keys[3], batch),
        )
    return cases


def run(path, *, which="latest", simulator_index=0, warmup=1, repeat=10, **options):
    model, generators, metadata = load_run(Path(path), which)
    if not 0 <= simulator_index < len(generators):
        raise ValueError(f"simulator_index must be in [0, {len(generators) - 1}]")
    cases = build_cases(model, generators[simulator_index], **options)
    _, params, _ = nnx.split(model, nnx.Param, ...)
    return {
        **metadata,
        "simulator_index": simulator_index,
        "backend": jax.default_backend(),
        "devices": [device.device_kind for device in jax.devices()],
        "jax_version": jax.__version__,
        "theta_num_steps": options.get("theta_num_steps", 64),
        "loss_batch_size": options.get("loss_batch_size", 2048),
        "parameter_counts": parameter_counts(params),
        "operations": {
            name: profile_function(fn, args, name=name, warmup=warmup, repeat=repeat)
            for name, (fn, args) in cases.items()
        },
    }


def format_table(result: dict) -> str:
    rows = [
        f"{'operation':<16}{'compile s':>11}{'first s':>10}{'mean s':>10}{'std s':>10}"
    ]
    for name, op in result["operations"].items():
        steady = op["steady_state"]
        rows.append(
            f"{name:<16}{op['compile_s']:>11.2f}{op['first_exec_s']:>10.4f}"
            f"{steady['mean_s']:>10.4f}{steady['std_s']:>10.4f}"
        )
    return "\n".join(rows)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="saved run directory (config.yaml and checkpoints/)",
    )
    parser.add_argument("--which", default="latest", choices=("latest", "best"))
    parser.add_argument("--simulator-index", type=int, default=0)
    parser.add_argument("--theta-num-steps", type=int, default=64)
    parser.add_argument("--loss-batch-size", type=int, default=2048)
    parser.add_argument(
        "--skip-training", action="store_true", help="skip loss_forward/loss_backward"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, help="write the full result as JSON (must not exist)"
    )
    add_slurm_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    if args.slurm:
        return submit(args, argv, main)
    result = run(
        args.checkpoint,
        which=args.which,
        simulator_index=args.simulator_index,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        theta_num_steps=args.theta_num_steps,
        loss_batch_size=args.loss_batch_size,
        include_training=not args.skip_training,
    )
    print(format_table(result))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
