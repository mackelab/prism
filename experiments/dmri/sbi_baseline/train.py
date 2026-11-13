"""Train the joint neural-spline-flow baseline on simulated dMRI data.

Every step simulates fresh data from the Ball-and-Stick prior for the two
random acquisition families (105 and 297 measurements) and averages their
negative log-likelihoods. Needs ``torch`` and ``sbi`` (see the README).

    uv run python -m experiments.dmri.sbi_baseline.train --output results/sbi_b3s \
        --num-steps 100000 --slurm h100_long
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# torch and sbi are imported inside the functions that need them, so the
# script can print --help (and be imported) without the optional packages.
from prism.dmri.simulators.acquisition_scheme import (
    random_hcp_acquisition,
    random_hcp_large_acquisition,
)
from prism.dmri.simulators.models import (
    Ball3StickSharedDiffusivity,
    MultiShellBall3StickSharedDiffusivity,
)
from prism.utils.slurm import add_slurm_arguments, submit

SIMULATORS = {
    cls.__name__: cls
    for cls in (Ball3StickSharedDiffusivity, MultiShellBall3StickSharedDiffusivity)
}


def batch_function(sim_type, *, large_acquisition: bool, batch_size: int):
    """A jitted ``key -> (theta, signal, bvals, bvecs)`` over ``batch_size`` voxels.

    The parameters come from the standard-normal prior, all model components
    are active, and each voxel has its own random acquisition.
    """
    acquisition = (
        random_hcp_large_acquisition if large_acquisition else random_hcp_acquisition
    )
    model_mask = jnp.ones((sim_type.num_compartments(),), dtype=bool)

    def simulate(key):
        acq_key, theta_key, noise_key = jax.random.split(key, 3)
        acq = acquisition(acq_key)
        theta = jax.random.normal(theta_key, (sim_type.theta_dim,))
        signal = sim_type.from_theta(theta, model_mask=model_mask).signal(
            acq, rng=noise_key
        )
        return theta, signal, acq.bvals, acq.bvecs

    return jax.jit(lambda key: jax.vmap(simulate)(jax.random.split(key, batch_size)))


def to_torch(batch, device):
    """Convert a simulated batch to ``(theta, packed condition)`` float tensors."""
    import torch

    from experiments.dmri.sbi_baseline.model import pack_condition

    theta, signal, bvals, bvecs = (
        torch.from_numpy(np.array(a, dtype=np.float32)).to(device) for a in batch
    )
    return theta, pack_condition(signal, bvals, bvecs)


def mean_loss(flow, batches):
    import torch

    from experiments.dmri.sbi_baseline.model import negative_log_likelihood

    return torch.stack([
        negative_log_likelihood(flow, *b).mean() for b in batches
    ]).mean()


def save_checkpoint(path, flow, optimizer, step, best_loss, config):
    import torch

    torch.save(
        {
            "model_state_dict": flow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_eval_loss": best_loss,
            "config": config,
        },
        path,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="run directory")
    parser.add_argument(
        "--sim-type", choices=sorted(SIMULATORS), default="Ball3StickSharedDiffusivity"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from output/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--device", default=None, help="torch device (default: CUDA if available)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=20_000)
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="voxels per acquisition family"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    for name, default in [
        ("model-dim", 128),
        ("output-dim", 512),
        ("num-cls-tokens", 4),
        ("num-layers", 4),
        ("hidden-features", 256),
        ("num-transforms", 8),
        ("num-bins", 10),
    ]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--tail-bound", type=float, default=3.0)
    add_slurm_arguments(parser)
    return parser


FLOW_OPTIONS = (
    "model_dim",
    "output_dim",
    "num_cls_tokens",
    "num_layers",
    "hidden_features",
    "num_transforms",
    "num_bins",
    "tail_bound",
)
# Settings that define the saved network; a resumed run must reuse them.
ARCHITECTURE = ("sim_type", *FLOW_OPTIONS)


def apply_saved_architecture(args, saved, parser):
    """On --resume, rebuild the saved run's network instead of the CLI defaults.

    A flag left at its default takes the saved value; a flag given with a
    different value than the saved run is an error, because the checkpoint
    could not be loaded into that network.
    """
    for key in ARCHITECTURE:
        if key not in saved:
            continue
        given = getattr(args, key)
        if given != parser.get_default(key) and given != saved[key]:
            raise ValueError(
                f"--{key.replace('_', '-')} {given} differs from the resumed "
                f"run's {saved[key]}; omit it to reuse the saved value."
            )
        setattr(args, key, saved[key])


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.slurm:
        return submit(args, argv, main)
    config_path = args.output / "config.json"
    if args.resume:
        if not config_path.is_file():
            parser.error(f"--resume needs {config_path} from the original run")
        try:
            apply_saved_architecture(args, json.loads(config_path.read_text()), parser)
        except ValueError as error:
            parser.error(str(error))
    import torch

    from experiments.dmri.sbi_baseline.model import build_flow

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    sim_type = SIMULATORS[args.sim_type]
    simulate = [
        batch_function(sim_type, large_acquisition=large, batch_size=args.batch_size)
        for large in (False, True)
    ]
    evaluate = [
        batch_function(
            sim_type, large_acquisition=large, batch_size=args.eval_batch_size
        )
        for large in (False, True)
    ]
    base_key = jax.random.PRNGKey(args.seed)
    eval_batches = [
        to_torch(fn(jax.random.fold_in(jax.random.PRNGKey(args.seed + 1), i)), device)
        for i, fn in enumerate(evaluate)
    ]

    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=args.resume)
    config = {
        k: v for k, v in vars(args).items() if k not in ("slurm", "slurm_timeout_min")
    }
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}

    theta, condition = eval_batches[0]
    flow = build_flow(theta, condition, **{k: getattr(args, k) for k in FLOW_OPTIONS})
    optimizer = torch.optim.AdamW(
        flow.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start, best_loss = 0, float("inf")
    if args.resume:
        state = torch.load(checkpoints / "latest.pt", map_location=device)
        flow.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start, best_loss = state["step"], state["best_eval_loss"]
    # Written only once the network (and any resumed state) loaded.
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"device={device} parameters={sum(p.numel() for p in flow.parameters()):,} start_step={start}"
    )

    flow.train()
    began = time.perf_counter()
    for step in range(start + 1, args.num_steps + 1):
        step_key = jax.random.fold_in(base_key, step)
        keys = jax.random.split(step_key, len(simulate))
        batches = [
            to_torch(fn(key), device) for fn, key in zip(simulate, keys, strict=True)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss = mean_loss(flow, batches)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            flow.parameters(), args.grad_clip_norm
        )
        optimizer.step()
        if step % args.log_every == 0 or step == start + 1:
            rate = (step - start) / (time.perf_counter() - began)
            print(
                f"step {step}/{args.num_steps} loss={loss.item():.4f} grad_norm={grad_norm:.2f} steps/s={rate:.2f}"
            )
        improved = False
        if step % args.eval_every == 0 or step == args.num_steps:
            flow.eval()
            with torch.no_grad():
                eval_loss = mean_loss(flow, eval_batches).item()
            flow.train()
            print(
                f"step {step} eval_loss={eval_loss:.4f} best={min(best_loss, eval_loss):.4f}"
            )
            if eval_loss < best_loss:
                best_loss, improved = eval_loss, True
                save_checkpoint(
                    checkpoints / "best.pt", flow, optimizer, step, best_loss, config
                )
        if improved or step % args.checkpoint_every == 0 or step == args.num_steps:
            save_checkpoint(
                checkpoints / "latest.pt", flow, optimizer, step, best_loss, config
            )
    print(f"finished; best eval loss {best_loss:.4f}")


if __name__ == "__main__":
    main()
