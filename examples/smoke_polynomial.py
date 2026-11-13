"""Train a small polynomial model, restore its EMA checkpoint, and sample it."""

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
    parser.add_argument("--output-dir", type=Path, default=Path("results/smoke"))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if output_dir.exists():
        parser.error(f"Choose a new output directory: {output_dir} already exists")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "prism",
            "train",
            "+experiment/train=symbolic",
            f"run.name={output_dir.name}",
            f"hydra.run.dir={output_dir}",
            f"training.max_train_steps={args.steps}",
        ],
        check=True,
    )

    latest, _, _ = load_checkpoint(output_dir, which="latest")
    if int(latest["step"]) != args.steps:
        raise RuntimeError(
            f"Training stopped at {latest['step']}, expected {args.steps}"
        )
    checkpoint, model, simulator_class = load_checkpoint(output_dir, which="best")
    nnx.update(model, checkpoint["params_ema"])
    model.eval()
    cfg = load_cfg(output_dir)
    _, simulators = build_simulator(cfg)
    observation = simulators[0](jax.random.key(1))

    @nnx.jit
    def sample(model, key, x, prior):
        mask_key, theta_key = jax.random.split(key)
        mask = model.sample_mask(mask_key, x=x, mask_prior=prior)
        theta = model.sample_theta(theta_key, x=x, model_mask=mask, num_steps=8)
        return mask, theta

    mask, theta = sample(
        model, jax.random.key(2), observation["x"], observation["mask_prior"]
    )
    prediction = simulator_class.create_from_theta(theta, model_mask=mask).evaluate(
        observation["x"][:, 0], rng=jax.random.key(3)
    )
    if not bool(jnp.all(jnp.isfinite(theta)) & jnp.all(jnp.isfinite(prediction))):
        raise RuntimeError("Inference produced non-finite values")
    result = {
        "training_step": int(latest["step"]),
        "best_step": int(checkpoint["step"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "weights": "ema",
        "observation_shape": list(observation["x"].shape),
        "mask": np.asarray(mask).tolist(),
        "theta_shape": list(theta.shape),
        "prediction_shape": list(prediction.shape),
    }
    np.savez(
        output_dir / "samples.npz",
        observation=np.asarray(observation["x"]),
        mask=np.asarray(mask),
        theta=np.asarray(theta),
        prediction=np.asarray(prediction),
    )
    (output_dir / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
