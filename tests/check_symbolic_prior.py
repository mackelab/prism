"""Exercise symbolic online training and prior-conditioned EMA inference on CPU.

Uses the standard 500-point simulator and reduced networks, batches and buffers.
This checks the pipeline; it does not measure Figure 4 accuracy.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from prism.train.build_simulator import build_simulator
from prism.train.utils import load_cfg, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=int, choices=(15, 30, 50), default=15)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    output_dir = (
        args.output_dir or Path(f"results/symbolic{args.components}_smoke")
    ).resolve()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if output_dir.exists():
        parser.error(f"Choose a new output directory: {output_dir} already exists")

    command = [
        sys.executable,
        "-m",
        "prism",
        "train",
        f"+experiment/train=symbolic{args.components}",
        "infrastructure/launcher=local",
        "infrastructure/partition=none",
        "hydra.mode=RUN",
        f"run.name={output_dir.name}",
        f"hydra.run.dir={output_dir}",
        f"training.max_train_steps={args.steps}",
        "training.inner_steps=1",
        "training.checkpoint_freq=2",
        "training.eval_freq=2",
        "training.eval_at_end=true",
        "training.eval.nnl_mask.iters=1",
        "training.eval.nnl_theta.iters=1",
        "training.dataloader.dataset.buffer_size=32",
        "training.dataloader.dataset.simulation_batch_size=8",
        "training.dataloader.val_dataset.buffer_size=8",
        "training.dataloader.val_dataset.simulation_batch_size=4",
        "training.dataloader.train_loader.batch_size=2",
        "training.dataloader.val_loader.batch_size=2",
        "training.dataloader.train_loader.num_prefetch_host=1",
        "training.dataloader.train_loader.num_prefetch_device=1",
        "training.dataloader.val_loader.num_prefetch_host=1",
        "training.dataloader.val_loader.num_prefetch_device=1",
        "model.model_dim=16",
        "model.embedding_net.params.num_layers=1",
        "model.model_selection_net.params.num_layers=1",
        "model.inference_net.params.num_layers=1",
    ]
    subprocess.run(command, check=True)

    latest, _, _ = load_checkpoint(output_dir, which="latest")
    if int(latest["step"]) != args.steps:
        raise RuntimeError(f"Expected step {args.steps}; restored {latest['step']}")
    checkpoint, model, simulator_class = load_checkpoint(output_dir, which="best")
    nnx.update(model, checkpoint["params_ema"])
    restored_params = nnx.state(model, nnx.Param)
    if not all(
        np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(restored_params),
            jax.tree.leaves(checkpoint["params_ema"]),
            strict=True,
        )
    ):
        raise RuntimeError("Restored parameters differ from the saved EMA")
    model.eval()
    cfg = load_cfg(output_dir)
    _, simulators = build_simulator(cfg)
    observation = simulators[0](jax.random.key(1))
    num_expressions = len(simulator_class.model_types)
    grid = jnp.arange(cfg.simulator.x_min, cfg.simulator.x_max, cfg.simulator.x_step)

    @nnx.jit
    def infer(model, x, prior):
        mask = model.sample_mask(jax.random.key(2), x=x, mask_prior=prior)
        theta = model.sample_theta(jax.random.key(3), x=x, model_mask=mask, num_steps=8)
        log_prob = model.log_prob_mask(mask, x=x, mask_prior=prior)
        return mask, theta, log_prob

    outputs = {"observation": np.asarray(observation["x"]), "grid": np.asarray(grid)}
    summaries = []
    for value in (0.1, 0.9):
        mask, theta, log_prob = infer(model, observation["x"], jnp.array([value]))
        prediction = simulator_class.from_theta(theta, model_mask=mask).evaluate(
            grid, rng=jax.random.key(4)
        )
        if not all(
            np.isfinite(np.asarray(item)).all()
            for item in (theta, log_prob, prediction)
        ):
            raise RuntimeError(f"Non-finite inference at lambda={value}")
        if int(mask[-len(simulator_class.noise_types) :].sum()) != 1 or not bool(
            mask[:num_expressions].any()
        ):
            raise RuntimeError("Sampled model violates the configured prior support")
        prefix = f"lambda_{value}"
        outputs.update({
            f"{prefix}_mask": np.asarray(mask),
            f"{prefix}_theta": np.asarray(theta),
            f"{prefix}_prediction": np.asarray(prediction),
        })
        summaries.append({
            "lambda": value,
            "active_expressions": int(mask[:num_expressions].sum()),
            "mask_log_prob": float(log_prob),
            "outputs_finite": True,
        })
    report = {
        "components": num_expressions,
        "training_step": int(latest["step"]),
        "best_step": int(checkpoint["step"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "weights": "ema",
        "ema_restored_exactly": True,
        "observation_shape": list(observation["x"].shape),
        "mask_dim": simulator_class.num_compartments(),
        "theta_dim": simulator_class.theta_dim,
        "inference": summaries,
        "command": command,
    }
    np.savez(output_dir / "samples.npz", **outputs)
    (output_dir / "smoke.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
