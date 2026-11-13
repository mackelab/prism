"""Posterior illustrations on one symbolic observation (Figures 2, 4a, A.1).

One observation is simulated from the run's generator (seed 69, mask-prior
hyperparameter 0.02). The model-selection network is queried across the mask
prior hyperparameter lambda, and complete posterior draws (mask, then
parameters) are taken at lambda = 0.9, 0.5 and 0.1. The draw shown for each of
lambda = 0.5 and 0.1 is the one that best fits the observations at the lowest
complexity (smallest ``error**20 + 0.1 * signal components``). The parameter posterior
is sampled for the lambda = 0.1 draw's mask.

* Figures 2 and 4a: run ``symbolic50_illustrative`` (ten observations).
* Figure A.1: run ``symbolic50`` (the standard observation grid), with
  ``--no-edit-observation --num-samples 1000 --display-index 0.5=627
  --display-index 0.1=12 --posterior-index 12``.

Example: python -m experiments.symbolic.illustrations.illustrate results/symbolic50_illustrative \
    --which latest --slurm a100 --output-dir results/symbolic50_illustrative/illustrations

Use small ``--num-samples`` / ``--num-posterior`` values for pipeline diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from prism.utils.slurm import add_slurm_arguments, submit

OBSERVATION_SEED = 69
OBSERVATION_PRIOR = 0.02
LAMBDAS_SHOWN = (0.9, 0.5, 0.1)
# The Figure 2 notebook replaces the first three simulated values (x = 0, 1, 2).
OBSERVATION_EDIT = {0: 7.1, 1: 13.567, 2: 9.0002313}
CHUNK = 5_000


def simulate_observation(generator, *, edit=False, seed=OBSERVATION_SEED):
    data = generator(
        jax.random.PRNGKey(seed),
        mask_prior_hyperparameter=jnp.array([OBSERVATION_PRIOR]),
    )
    if edit:
        x = data["x"]
        if x.reshape(x.shape[0], -1).shape[1] != 1:
            raise ValueError(
                "the Figure 2 edit applies to single-valued observations (input_dim 1); "
                "pass --no-edit-observation for other simulators"
            )
        for index, value in OBSERVATION_EDIT.items():
            x = x.at[index].set(value)
        data = {**data, "x": x}
    return data


def signal_components(sim_type, masks):
    """Number of active signal components (the mask's last entries flag noise)."""
    return jnp.asarray(masks)[..., : len(sim_type.model_types)].sum(-1)


def _mask_sampler(model, x):
    return lambda key, prior: model.sample_mask(key, x=x, mask_prior=prior)


def model_space(model, x, *, num_lambda=100, num_samples=100, seed=1234):
    """Marginal inclusion probability of every component for each lambda."""
    sample = _mask_sampler(model, x)

    def one(key, prior):
        draws = jax.vmap(lambda k: sample(k, prior))(jax.random.split(key, num_samples))
        return draws.mean(0)

    lambdas = jnp.linspace(0.01, 0.99, num_lambda)
    key = jax.random.PRNGKey(seed)
    return np.asarray(jax.lax.map(lambda prior: one(key, prior), lambdas))


def component_count_quantiles(
    model, x, sim_type, *, num_lambda=100, num_samples=500, seed=5678
):
    """Median and 95% interval of the number of active components over lambda."""
    sample = _mask_sampler(model, x)

    def one(key, prior):
        draws = jax.vmap(lambda k: sample(k, prior))(jax.random.split(key, num_samples))
        counts = signal_components(sim_type, draws)
        return (
            jnp.median(counts),
            jnp.quantile(counts, 0.025),
            jnp.quantile(counts, 0.975),
        )

    lambdas = jnp.linspace(0.0, 1.0, num_lambda)
    key = jax.random.PRNGKey(seed)
    out = jax.lax.map(lambda prior: one(key, prior), lambdas)
    return np.stack([np.asarray(v) for v in out])


def posterior_draws(model, x, prior, *, num_samples, mask_seed, theta_seed):
    """Masks from q(M | x, lambda), then one parameter draw per mask."""
    masks = jax.lax.map(
        lambda k: model.sample_mask(k, x=x, mask_prior=prior),
        jax.random.split(jax.random.PRNGKey(mask_seed), num_samples),
        batch_size=CHUNK,
    )
    thetas = jax.lax.map(
        lambda args: model.sample_theta(args[0], x=x, model_mask=args[1]),
        (jax.random.split(jax.random.PRNGKey(theta_seed), num_samples), masks),
        batch_size=CHUNK,
    )
    return masks, thetas


def predictions(sim_type, thetas, masks, grid, key=None):
    """Evaluate each draw on ``grid`` (noisy if ``key`` is given)."""
    keys = (
        jax.random.split(key, len(thetas)) if key is not None else [None] * len(thetas)
    )

    def one(theta, mask, k):
        return sim_type.from_theta(theta, model_mask=mask).evaluate(grid, k)

    if key is None:
        return jax.lax.map(
            lambda a: one(a[0], a[1], None), (thetas, masks), batch_size=CHUNK
        )
    return jax.lax.map(lambda a: one(*a), (thetas, masks, keys), batch_size=CHUNK)


def simplest_good_fit(sim_type, thetas, masks, grid_obs, x_flat):
    """Index of the draw minimising ``error**20 + 0.1 * components``."""
    y = predictions(sim_type, thetas, masks, grid_obs)
    error = jnp.abs(y - x_flat[None, :]).mean(-1)
    return int(jnp.argmin(error**20 + signal_components(sim_type, masks) * 0.1))


def natural_parameters(sim_type, thetas, mask):
    """Component parameters of the active components, one row per draw."""

    flags = np.asarray(mask)  # concrete, so the selection below is not traced

    def one(theta):
        sim = sim_type.from_theta(theta, model_mask=mask)
        # The mask also holds the noise-component flags; zip stops at the signal components.
        active = [c for c, on in zip(sim.model_compartments, flags, strict=False) if on]
        return ravel_pytree([c.params for c in active])[0]

    return np.asarray(jax.lax.map(one, thetas, batch_size=CHUNK))


def equation(sim_type, theta, mask, digits=1):
    import sympy as sp

    sim = sim_type.from_theta(theta, model_mask=mask)
    expr = sim.to_simpy(round_digits=digits)
    return {
        "sympy": str(expr),
        "latex": sp.latex(expr),
        "components": int(signal_components(sim_type, mask)),
    }


def illustrate(
    model,
    generator,
    sim_type,
    grid_obs,
    *,
    edit_observation=False,
    num_samples=50_000,
    num_posterior=100_000,
    grid_eval=None,
    seed=OBSERVATION_SEED,
    display_index=None,
    posterior_index=None,
):
    grid_obs = jnp.asarray(grid_obs)
    grid_eval = (
        jnp.linspace(0.0, 15.0, 1000) if grid_eval is None else jnp.asarray(grid_eval)
    )
    data = simulate_observation(generator, edit=edit_observation, seed=seed)
    display_index = {float(k): int(v) for k, v in (display_index or {}).items()}
    unknown = set(display_index) - {0.5, 0.1}
    if unknown:
        raise ValueError(
            f"display indices exist for lambda 0.5 and 0.1, got {sorted(unknown)}"
        )
    for name, index in (*display_index.items(), ("posterior", posterior_index)):
        if index is not None and not 0 <= index < num_samples:
            raise ValueError(
                f"index {index} for {name} is outside 0..{num_samples - 1}"
            )
    x, true_mask, true_theta = data["x"], data["model_mask"], data["theta"]
    x_flat = jnp.ravel(x)

    arrays = {
        "x_observed": np.asarray(x_flat),
        "grid_obs": np.asarray(grid_obs),
        "grid_eval": np.asarray(grid_eval),
        "true_mask": np.asarray(true_mask),
        "model_space": model_space(model, x),
        "component_quantiles": component_count_quantiles(model, x, sim_type),
        "true_components": np.asarray(signal_components(sim_type, true_mask)),
    }
    true_sim = sim_type.from_theta(true_theta, model_mask=true_mask)
    arrays["true_function"] = np.asarray(true_sim.evaluate(grid_eval, None))
    summary = {
        "observation_seed": seed,
        "observation_edited": bool(edit_observation),
        "true_components": int(signal_components(sim_type, true_mask)),
        "true_equation": equation(sim_type, true_theta, true_mask),
        "selected": {},
    }

    draws = {}
    for i, lam in enumerate(LAMBDAS_SHOWN):
        prior = jnp.asarray(lam)
        draws[lam] = posterior_draws(
            model,
            x,
            prior,
            num_samples=num_samples,
            mask_seed=42 + i,
            theta_seed=45 + i,
        )
        arrays[f"mean_components_{lam}"] = float(
            signal_components(sim_type, draws[lam][0]).mean()
        )

    for lam in (0.5, 0.1):
        masks, thetas = draws[lam]
        index = display_index.get(lam)
        if index is None:
            index = simplest_good_fit(sim_type, thetas, masks, grid_obs, x_flat)
        noisy = predictions(sim_type, thetas, masks, grid_eval, jax.random.PRNGKey(1))
        free = sim_type.from_theta(thetas[index], model_mask=masks[index]).evaluate(
            grid_eval, None
        )
        arrays[f"predictive_lo_{lam}"] = np.percentile(np.asarray(noisy), 2.5, axis=0)
        arrays[f"predictive_hi_{lam}"] = np.percentile(np.asarray(noisy), 97.5, axis=0)
        arrays[f"selected_function_{lam}"] = np.asarray(free)
        arrays[f"selected_mask_{lam}"] = np.asarray(masks[index])
        summary["selected"][str(lam)] = {
            "index": index,
            "index_given": lam in display_index,
            **equation(sim_type, thetas[index], masks[index]),
        }

    # Parameter posterior of one lambda = 0.1 draw's model (the displayed draw by default).
    if posterior_index is None:
        mask = jnp.asarray(arrays["selected_mask_0.1"])
    else:
        mask = jnp.asarray(draws[0.1][0][posterior_index])
    summary["posterior_index"] = posterior_index
    thetas = jax.lax.map(
        lambda k: model.sample_theta(k, x=x, model_mask=mask),
        jax.random.split(jax.random.PRNGKey(45), num_posterior),
        batch_size=CHUNK,
    )
    arrays["parameter_posterior"] = natural_parameters(sim_type, thetas, mask)
    return arrays, summary


def load_run(path, which):
    """(model, generator, sim_type, grid_obs, step) from a saved run."""
    from flax import nnx

    from prism.train.build_simulator import build_simulator
    from prism.train.utils import load_cfg, load_checkpoint, select_checkpoint_params

    checkpoint, model, _ = load_checkpoint(path, which=which)
    nnx.update(model, select_checkpoint_params(checkpoint, "params_ema"))
    model.eval()
    cfg = load_cfg(path)
    sim_type, generators = build_simulator(cfg)
    simulator = cfg["simulator"]
    grid = np.arange(
        float(simulator["x_min"]), float(simulator["x_max"]), float(simulator["x_step"])
    )
    return model, generators[0], sim_type, grid, int(checkpoint["step"])


def load_pretrained(preset, which, weights_dir=None):
    """Same tuple as :func:`load_run`, from safetensors weights.

    ``preset`` is a Hub preset (for example ``symbolic50``); ``weights_dir`` a local
    directory laid out as ``<which>/params_ema.safetensors`` next to ``config.yaml``.
    ``which`` may be ``best``, ``latest`` or, for ``symbolic50_illustrative``, ``figure``.
    """
    from prism.symbolic.pretrained import load_pretrained_symbolic

    bundle = load_pretrained_symbolic(
        preset=preset if weights_dir is None else None,
        path=weights_dir,
        which=which,
    )
    sim = bundle.config["simulator"]
    grid = np.arange(float(sim["x_min"]), float(sim["x_max"]), float(sim["x_step"]))
    step = -1 if bundle.step is None else bundle.step
    return bundle.model, bundle.generator, bundle.simulator, grid, step


def _lam_index(text):
    lam, _, index = text.partition("=")
    return float(lam), int(index)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path, nargs="?", help="saved run directory")
    parser.add_argument(
        "--preset", help="published safetensors preset instead of a run directory"
    )
    parser.add_argument(
        "--weights-dir", type=Path, help="local copy of the preset's directory"
    )
    parser.add_argument(
        "--which", default="latest", choices=("latest", "best", "figure")
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=OBSERVATION_SEED,
        help="observation seed (Figures 2, 4a and A.1 use the default 69)",
    )
    parser.add_argument(
        "--display-index",
        action="append",
        default=[],
        type=_lam_index,
        metavar="LAMBDA=INDEX",
        help="show this posterior draw at lambda (repeatable); default: best-fit simplest",
    )
    parser.add_argument(
        "--posterior-index",
        type=int,
        default=None,
        help="lambda=0.1 draw whose model gets the parameter posterior",
    )
    parser.add_argument(
        "--edit-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the Figure 2 notebook's edit to the first three observations",
    )
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--num-posterior", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, required=True, help="must not exist")
    add_slurm_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.slurm:
        return submit(args, argv, main)
    if (args.checkpoint is None) == (args.preset is None):
        raise SystemExit("give exactly one of a run directory or --preset")
    if args.checkpoint is not None and (args.weights_dir or args.which == "figure"):
        raise SystemExit("--weights-dir and --which figure apply to --preset only")
    if args.preset:
        model, generator, sim_type, grid, step = load_pretrained(
            args.preset, args.which, args.weights_dir
        )
    else:
        model, generator, sim_type, grid, step = load_run(args.checkpoint, args.which)
    arrays, summary = illustrate(
        model,
        generator,
        sim_type,
        grid,
        edit_observation=args.edit_observation,
        num_samples=args.num_samples,
        num_posterior=args.num_posterior,
        seed=args.seed,
        display_index=dict(args.display_index),
        posterior_index=args.posterior_index,
    )
    summary.update(
        checkpoint=str(args.checkpoint or args.preset),
        which=args.which,
        step=step,
        params_used="ema",
        num_samples=args.num_samples,
        num_posterior=args.num_posterior,
        jax_version=jax.__version__,
    )
    args.output_dir.mkdir(parents=True)
    np.savez_compressed(args.output_dir / "arrays.npz", **arrays)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
