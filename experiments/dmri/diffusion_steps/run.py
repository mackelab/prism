"""ESS and run time of the B3S posterior against the number of diffusion steps.

For one real voxel, 1000 posterior samples of the full three-stick model (all
five components active) are drawn with 2, 4, ..., 512 solver steps. Each setting
is timed over several runs after a compile run. The effective sample size (ESS)
is computed from importance weights ``posterior / q`` of the samples, with ``q``
evaluated by the network at 1024 steps, and divided by 1000.

NOT TESTED: this script was written from an analysis notebook without being run. The timings
depend on the hardware (the paper used an Nvidia H100) and the voxel on the data.

Example: python -m experiments.dmri.diffusion_steps.run results/b3s \
    --data-dir data/hcp --voxel 50000 --output results/b3s/steps_ess.npz
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

NUM_STEPS = (2, 4, 8, 16, 32, 64, 128, 256, 512)
NUM_SAMPLES = 1000
LOG_PROB_STEPS = 1024
T_MIN = 5e-4


def load_voxel(data_dir, index):
    """Normalised signal of one in-brain voxel and the sorted acquisition.

    ``data_dir`` holds ``data.nii.gz``, ``nodif_brain_mask.nii.gz``, ``bvals`` and
    ``bvecs``. b-values are rounded to 0/1000/2000, signals are divided by the mean
    b=0 signal, outliers above the 99.9% quantile are clipped, and volumes are sorted
    by b-value, as in the original analysis. Requires ``dipy``.
    """
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti

    from prism.dmri.simulators.acquisition_scheme import acquisition_scheme

    data_dir = Path(data_dir)
    data, _ = load_nifti(str(data_dir / "data.nii.gz"))
    brain, _ = load_nifti(str(data_dir / "nodif_brain_mask.nii.gz"))
    bvals, bvecs = read_bvals_bvecs(str(data_dir / "bvals"), str(data_dir / "bvecs"))
    bvals = np.clip(np.round(bvals / 1000) * 1000, 0, None)
    s0 = data[..., bvals == 0].mean(-1)
    signal = np.where(brain[..., None], data / s0[..., None], 0.0)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(bvals)
    signal = signal.reshape(-1, signal.shape[-1])[brain.reshape(-1).astype(bool)][
        :, order
    ]
    signal = np.clip(signal, 0, np.quantile(signal, 0.999))
    return jnp.asarray(signal[index]), acquisition_scheme(bvals[order], bvecs[order])


def make_functions(model, sim_type, acq, mask):
    @partial(jax.jit, static_argnames=("num_steps",))
    def sample(rng, x, num_steps):
        one = jax.vmap(
            partial(
                model.sample_theta,
                last_euler_step=False,
                t_min=T_MIN,
                num_steps=num_steps,
                model_mask=mask,
            ),
            in_axes=(0, None, None),
        )
        return one(jax.random.split(rng, NUM_SAMPLES), acq, x)

    @jax.jit
    def ess(samples, x):
        log_q = jax.vmap(
            partial(
                model.log_prob_theta,
                acq=acq,
                x=x,
                model_mask=mask,
                num_steps=LOG_PROB_STEPS,
            )
        )(samples)
        theta_mask = sim_type.theta_mask(mask)

        def log_posterior(theta):
            simulator = sim_type.from_theta(theta, model_mask=mask)
            log_prior = jnp.sum(jax.scipy.stats.norm.logpdf(theta) * theta_mask)
            return simulator.log_likelihood(acq, x) + log_prior

        weights = jax.nn.softmax(jax.vmap(log_posterior)(samples) - log_q)
        return 1.0 / jnp.sum(weights**2) / NUM_SAMPLES

    return sample, ess


def sweep(sample, ess, x, *, num_steps=NUM_STEPS, runs=5, seed=0):
    """ESS and wall-clock seconds, each of shape ``(len(num_steps), runs)``."""
    ess_values, run_times = [], []
    for steps in num_steps:
        jax.block_until_ready(
            sample(jax.random.key(seed), x, num_steps=steps)
        )  # compile
        ess_runs, time_runs = [], []
        for run in range(runs):
            start = time.perf_counter()
            out = jax.block_until_ready(sample(jax.random.key(run), x, num_steps=steps))
            time_runs.append(time.perf_counter() - start)
            ess_runs.append(float(ess(out, x)))
        ess_values.append(ess_runs)
        run_times.append(time_runs)
    return np.asarray(ess_values), np.asarray(run_times)


def load_run(path, which):
    from flax import nnx

    from prism.train.build_simulator import build_simulator
    from prism.train.utils import load_cfg, load_checkpoint, select_checkpoint_params

    checkpoint, model, _ = load_checkpoint(path, which=which)
    nnx.update(model, select_checkpoint_params(checkpoint, "params_ema"))
    model.eval()
    sim_type, _ = build_simulator(load_cfg(path))
    return model, sim_type, int(checkpoint["step"])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path, help="saved B3S run directory")
    parser.add_argument("--which", default="latest", choices=("latest", "best"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--voxel", type=int, default=50_000, help="index among in-brain voxels"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, required=True, help="npz (must not exist)"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    model, sim_type, step = load_run(args.checkpoint, args.which)
    x, acq = load_voxel(args.data_dir, args.voxel)
    mask = jnp.ones(sim_type.num_compartments(), dtype=bool)  # all components active
    sample, ess = make_functions(model, sim_type, acq, mask)
    ess_values, run_times = sweep(sample, ess, x, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        num_steps=np.asarray(NUM_STEPS),
        ess=ess_values,
        run_time_s=run_times,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "which": args.which,
                "step": step,
                "voxel": args.voxel,
                "runs": args.runs,
                "device": str(jax.devices()[0]),
                "jax_version": jax.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
