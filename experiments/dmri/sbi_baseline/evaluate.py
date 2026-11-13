"""Evaluate a baseline checkpoint on simulated dMRI problems.

For each acquisition family it draws ``--num-datasets`` problems (parameters
from the prior, a random acquisition, one noisy signal), samples
``--num-samples`` parameters from the flow and reports, per problem:

* ``ess``: the effective sample size of the importance weights
  ``p(theta | x) / q(theta | x)`` as a fraction of the samples, with the
  unnormalised posterior ``likelihood * N(0, I)``;
* ``rmse``: the root-mean-square error between the observed signal and a
  signal simulated from each posterior sample;
* ``ksd``: the multi-scale preconditioned kernel Stein discrepancy and its
  bootstrap p-value, from ``prism.inference.ksd``.

    uv run python -m experiments.dmri.sbi_baseline.evaluate \
        results/sbi_b3s --which best --output results/sbi_b3s_eval.json

Needs ``torch`` and ``sbi`` (see the README). Real-data evaluation is not included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# torch and sbi are imported inside functions, so --help works without them.
from experiments.dmri.sbi_baseline.train import FLOW_OPTIONS, SIMULATORS, batch_function
from prism.dmri.simulators.acquisition_scheme import (
    random_hcp_acquisition,
    random_hcp_large_acquisition,
)
from prism.inference.ksd import multiscale_preconditioned_ksd_and_pvalue
from prism.utils.slurm import add_slurm_arguments, submit

REGIMES = {"hcp": random_hcp_acquisition, "hcp_large": random_hcp_large_acquisition}
KSD_BANDWIDTHS = (0.05, 0.1, 0.5)


def load_flow(checkpoint: Path, device):
    """Rebuild the flow from a checkpoint file (a run directory is rejected)."""
    import torch

    from experiments.dmri.sbi_baseline.model import build_flow, pack_condition

    path = Path(checkpoint)
    if path.is_dir():
        raise ValueError("Pass the checkpoint file, e.g. RUN/checkpoints/best.pt.")
    state = torch.load(path, map_location=device)
    config = state["config"]
    sim_type = SIMULATORS[config["sim_type"]]
    example = batch_function(sim_type, large_acquisition=False, batch_size=8)(
        jax.random.PRNGKey(0)
    )
    theta, signal, bvals, bvecs = (
        torch.from_numpy(np.array(a, np.float32)).to(device) for a in example
    )
    flow = build_flow(
        theta,
        pack_condition(signal, bvals, bvecs),
        **{k: config[k] for k in FLOW_OPTIONS},
    )
    flow.load_state_dict(state["model_state_dict"])
    return flow.eval(), sim_type, state


def effective_sample_fraction(log_p, log_q):
    """``(sum w)^2 / sum w^2 / n`` for ``w = exp(log_p - log_q)``, ignoring non-finite weights."""
    log_w = log_p - log_q
    finite = jnp.isfinite(log_w)
    log_w = jnp.where(finite, log_w, -jnp.inf)
    w = jnp.where(finite, jnp.exp(log_w - jnp.max(log_w)), 0.0)
    return jnp.sum(w) ** 2 / jnp.maximum(jnp.sum(w * w), 1e-30) / log_w.shape[0]


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q10": float(np.quantile(values, 0.1)),
        "q90": float(np.quantile(values, 0.9)),
    }


def evaluate_regime(flow, sim_type, acquisition, *, num_datasets, num_samples, device):
    import torch

    from experiments.dmri.sbi_baseline.model import pack_condition

    mask = jnp.ones((sim_type.num_compartments(),), dtype=bool)

    @jax.jit
    def problem(key):
        acq_key, theta_key, noise_key = jax.random.split(key, 3)
        acq = acquisition(acq_key)
        theta = jax.random.normal(theta_key, (sim_type.theta_dim,))
        return acq, sim_type.from_theta(theta, model_mask=mask).signal(
            acq, rng=noise_key
        )

    @jax.jit
    def log_posterior(thetas, acq, x):
        def one(theta):
            log_likelihood = sim_type.from_theta(theta, model_mask=mask).log_likelihood(
                acq, x
            )
            return log_likelihood + jnp.sum(jax.scipy.stats.norm.logpdf(theta))

        return jax.vmap(one)(thetas)

    @jax.jit
    def prediction_rmse(thetas, acq, x, key):
        keys = jax.random.split(key, thetas.shape[0])
        predicted = jax.vmap(
            lambda theta, k: sim_type.from_theta(theta, model_mask=mask).signal(
                acq, rng=k
            )
        )(thetas, keys)
        return jnp.sqrt(jnp.mean((predicted - x) ** 2, axis=-1))

    ess, rmse, ksd, pvalues = [], [], [], []
    for i, key in enumerate(jax.random.split(jax.random.PRNGKey(0), num_datasets)):
        problem_key, rmse_key, ksd_key = jax.random.split(key, 3)
        acq, x = problem(problem_key)
        condition = pack_condition(
            *(
                torch.from_numpy(np.array(a, np.float32))[None].to(device)
                for a in (x, acq.bvals, acq.bvecs)
            )
        )
        with torch.no_grad():
            samples = flow.sample(num_samples, context=condition)[0]
            log_q = flow.log_prob(samples, context=condition.expand(num_samples, -1))
        thetas = jnp.asarray(samples.cpu().numpy())
        ess.append(
            float(
                effective_sample_fraction(
                    log_posterior(thetas, acq, x), jnp.asarray(log_q.cpu().numpy())
                )
            )
        )
        rmse.append(np.asarray(prediction_rmse(thetas, acq, x, rmse_key)))
        value, pvalue = multiscale_preconditioned_ksd_and_pvalue(
            sim_type=sim_type,
            thetas=thetas,
            mask=mask,
            acq=acq,
            x=x,
            key=ksd_key,
            bandwidths=jnp.asarray(KSD_BANDWIDTHS),
            n_bootstrap=256,
        )
        ksd.append(float(value))
        pvalues.append(float(pvalue))
        if (i + 1) % max(1, num_datasets // 10) == 0:
            print(f"  {i + 1}/{num_datasets}")
    return {
        "ess": summarize(ess),
        "rmse": summarize(np.concatenate(rmse)),
        "ksd": summarize(ksd),
        "ksd_pvalue": summarize(pvalues),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="checkpoint file, e.g. RUN/checkpoints/best.pt"
    )
    parser.add_argument(
        "--regimes", nargs="+", choices=sorted(REGIMES), default=sorted(REGIMES)
    )
    parser.add_argument("--num-datasets", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", type=Path, help="write the metrics to this new .json file"
    )
    add_slurm_arguments(parser)
    args = parser.parse_args(argv)
    if args.slurm:
        return submit(args, argv, main)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    import torch

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    flow, sim_type, state = load_flow(args.checkpoint, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "step": state["step"],
        "best_eval_loss": state["best_eval_loss"],
        "simulator": sim_type.__name__,
        "num_datasets": args.num_datasets,
        "num_samples": args.num_samples,
        "device": str(device),
    }
    for name in args.regimes:
        print(f"{name}: {args.num_datasets} datasets")
        result[name] = evaluate_regime(
            flow,
            sim_type,
            REGIMES[name],
            num_datasets=args.num_datasets,
            num_samples=args.num_samples,
            device=device,
        )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
