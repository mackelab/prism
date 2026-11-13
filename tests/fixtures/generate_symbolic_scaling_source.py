"""Record reference arrays from the frozen ``sbmi_pp`` source for K=80/100.

Run this inside a checkout of ``mackelab/sbmi_transformer_symbolic`` at commit
c8e118b8bd557f2b001812f00af982b136f0d793 (no prism code is imported)::

    PYTHONPATH=<source checkout> python generate_symbolic_scaling_source.py \
        --source <source checkout> --out <directory> [--k 80 100]

It writes ``symbolic_scaling_large_source.{npz,json}`` with the same array
names as ``symbolic_scaling_source.npz`` (prefixed ``k<K>_``). ``--k 30``
regenerates the K=30 block of the existing fixture, which is how the
procedure itself was checked.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from omegaconf import OmegaConf
from sbmi_pp.train.build_simulator import build_simulator

COMMIT = "c8e118b8bd557f2b001812f00af982b136f0d793"
SEEDS = [0, 1, 7, 42]
FIXED_HYPER_SEED = 13
COMPONENT_NOISE_SEED = 23
SOURCE_FILES = [
    "sbmi_pp/simulators/base.py",
    "sbmi_pp/simulators/mask_prior.py",
    "sbmi_pp/simulators/symbolic_model/base.py",
    "sbmi_pp/simulators/symbolic_model/model.py",
    "sbmi_pp/simulators/symbolic_model/model_specs.py",
    "sbmi_pp/train/build_simulator.py",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(k, source):
    conf_path = f"conf/simulator/symbolic_{k}.yaml"
    sim_cfg = yaml.safe_load((Path(source) / conf_path).read_text())
    cfg = OmegaConf.create({"simulator": sim_cfg})
    sim_type, generators = build_simulator(cfg)
    (generator,) = generators
    theta_dim = int(sim_type.theta_dim)
    num_models, num_noises = len(sim_type.model_types), len(sim_type.noise_types)
    n = num_models + num_noises
    components = sim_type.model_types + sim_type.noise_types

    grid = jnp.arange(0.0, 10.0, 0.02)
    observed = jnp.sin(grid)
    theta = jnp.linspace(-0.8, 0.8, theta_dim)
    masks = jnp.concatenate([
        jnp.eye(n, dtype=bool),
        jnp.zeros((1, n), bool),
        jnp.ones((1, n), bool),
    ])
    arrays = {
        "grid": grid,
        "observed": observed,
        "transform_theta": theta,
        "component_masks": masks,
        "split_dims": np.asarray(sim_type.split_idx()),
        "param_low": np.concatenate([np.asarray(c.param_low) for c in components]),
        "param_high": np.concatenate([np.asarray(c.param_high) for c in components]),
    }
    model = sim_type.from_theta(theta)
    parts = model.model_compartments + model.noise_compartments
    arrays["bounded_params"] = jnp.concatenate([c.params_array for c in parts])
    arrays["roundtrip_theta"] = model.theta
    arrays["theta_mask"] = jax.vmap(sim_type.theta_mask)(masks)
    key = jax.random.PRNGKey(COMPONENT_NOISE_SEED)
    arrays["component_signals"] = jax.vmap(
        lambda m: sim_type.from_theta(theta, model_mask=m).evaluate(grid, key)
    )(masks)
    arrays["log_likelihood"] = jax.vmap(
        lambda m: sim_type.from_theta(theta, model_mask=m).log_likelihood(
            grid, observed
        )
    )(masks)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(SEEDS, dtype=jnp.int32))
    out = jax.vmap(generator)(keys)
    for name in ("theta", "model_mask", "mask_prior", "x"):
        arrays[name] = out[name]
    hyper = jnp.asarray([[0.1], [0.9]], dtype=jnp.float32)
    fkey = jax.random.PRNGKey(FIXED_HYPER_SEED)
    fixed = jax.vmap(lambda h: generator(fkey, mask_prior_hyper=h))(hyper)
    for name in ("theta", "model_mask", "mask_prior", "x"):
        arrays[f"fixed_{name}"] = fixed[name]
    meta = {
        "source_class": sim_type.__name__,
        "components": [c.__name__ for c in components],
        "expressions": [c.expression_str for c in components],
        "theta_dim": theta_dim,
        "num_models": num_models,
        "num_noises": num_noises,
        "generator_config": {"simulator": sim_cfg},
        "source_config": conf_path,
        "source_config_sha256": sha256(Path(source) / conf_path),
    }
    return {f"k{k}_{name}": np.asarray(value) for name, value in arrays.items()}, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[80, 100])
    parser.add_argument("--name", default="symbolic_scaling_large_source")
    args = parser.parse_args()
    head = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"Source is at {head}, expected {COMMIT}.")
    arrays, variants = {}, {}
    for k in args.k:
        block, meta = record(k, args.source)
        arrays.update(block)
        variants[str(k)] = meta
    arrays["seeds"] = np.asarray(SEEDS, dtype=np.int32)
    arrays["fixed_hyper"] = np.asarray([[0.1], [0.9]], dtype=np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    npz = args.out / f"{args.name}.npz"
    np.savez(npz, **arrays)
    metadata = {
        "source_commit": COMMIT,
        "source_generator": "sbmi_pp/train/build_simulator.py:simulate_symbolic",
        "versions": {
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "source_files_sha256": {f: sha256(args.source / f) for f in SOURCE_FILES},
        "seeds": SEEDS,
        "fixed_hyper_seed": FIXED_HYPER_SEED,
        "component_noise_seed": COMPONENT_NOISE_SEED,
        "transform_theta": "jax.numpy.linspace(-0.8, 0.8, theta_dim)",
        "component_masks": "Identity masks, then all inactive, then all active.",
        "observed": "jax.numpy.sin(grid), grid = jax.numpy.arange(0.0, 10.0, 0.02)",
        "variants": variants,
        "fixture_sha256": sha256(npz),
    }
    (args.out / f"{args.name}.json").write_text(json.dumps(metadata, indent=1) + "\n")
    print("wrote", npz, sorted(variants))


if __name__ == "__main__":
    main()
