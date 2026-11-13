"""Evaluate Figure 4c conditional-theta error for a symbolic scaling checkpoint.

The true mask is supplied to theta inference. Each prediction includes fresh
noise, and the reported excess is mean RMSE minus the independently simulated
GT mean. This is distinct from the joint mask/theta Figure 4b evaluation.

Example: python -m experiments.symbolic.model_space_scaling.evaluate results/symbolic15 \
    --slurm a100 --output results/symbolic15/fig4c-latest.json

Use small --num-samples values for pipeline diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

from prism.utils.slurm import add_slurm_arguments, submit

from .._published import add_published_arguments, check_weights_source, load_published

PROTOCOL = "symbolic-scaling-conditional-rmse-v1"
SEED = 0
NUM_SAMPLES = 2**20
SOURCE_BATCH_SIZE = 2**15
SOURCE_K = (15, 30, 50, 80, 100)
DIMENSIONS = {15: (5, 20, 28), 30: (8, 38, 73), 50: (8, 58, 115)}
GRID = jnp.arange(0.0, 10.0, 0.02)
NUM_STEPS = 64
BOOTSTRAP_SEED = 0
BOOTSTRAP_REPLICATES = 500


def evaluation_key(components):
    """Keep the original five-checkpoint split when evaluating a single run."""
    if components not in DIMENSIONS:
        raise ValueError("Supported expression counts are 15, 30 and 50")
    return jax.random.split(jax.random.PRNGKey(SEED), len(SOURCE_K))[
        SOURCE_K.index(components)
    ]


def trial_rmse(model, data_gen, simulator, key, *, mode):
    if mode == "predictive":
        data_key, theta_key, noise_key = jax.random.split(key, 3)
        data = data_gen(data_key)
        theta = model.sample_theta(
            theta_key,
            x=data["x"],
            model_mask=data["model_mask"],
            sample_method="ode",
            num_steps=NUM_STEPS,
            t_min=None,
            t_max=None,
        )
    elif mode == "gt":
        data_key, noise_key = jax.random.split(key, 2)
        data = data_gen(data_key)
        theta = data["theta"]
    else:
        raise ValueError("mode must be predictive or gt")
    if data["x"].shape != (50, 10):
        raise ValueError("Figure 4c requires 500 signal values in (50, 10) blocks")
    prediction = simulator.from_theta(theta, model_mask=data["model_mask"]).evaluate(
        GRID, rng=noise_key
    )
    if prediction.shape != GRID.shape:
        raise ValueError("Simulation must produce one signal per grid point")
    return jnp.sqrt(jnp.mean((data["x"].flatten() - prediction) ** 2))


def evaluate_errors(model, data_gen, simulator, components, num_samples, batch_size):
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive")
    # Split once at the full requested sample count; chunking never reseeds.
    keys = jax.random.split(evaluation_key(components), num_samples)
    errors = {}
    for mode in ("predictive", "gt"):
        values = jax.lax.map(
            lambda key, mode=mode: trial_rmse(
                model, data_gen, simulator, key, mode=mode
            ),
            keys,
            batch_size=min(batch_size, num_samples),
        )
        errors[mode] = np.asarray(jax.device_get(values))
        if not np.isfinite(errors[mode]).all():
            raise RuntimeError(f"Non-finite {mode} RMSE; samples cannot be discarded")
    return errors


def summarize(predictive, gt, *, bootstrap_chunk_size=65536):
    """Bootstrap predictive datasets with the measured GT mean held fixed.

    Seed zero fixes the bootstrap draws. Float64 block sums bound memory without
    changing draws with chunk size. The confidence interval holds the separately
    estimated GT mean fixed and excludes uncertainty in that estimate.
    """
    predictive, gt = np.asarray(predictive), np.asarray(gt)
    if (
        predictive.ndim != 1
        or gt.shape != predictive.shape
        or not predictive.size
        or bootstrap_chunk_size <= 0
        or not np.isfinite(predictive).all()
        or not np.isfinite(gt).all()
    ):
        raise ValueError("Expected equal, finite, nonempty 1D RMSE arrays")
    gt_mean = gt.mean()
    excess = predictive - gt_mean
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        for start in range(0, predictive.size, bootstrap_chunk_size):
            size = min(bootstrap_chunk_size, predictive.size - start)
            indices = rng.integers(0, predictive.size, size=size)
            total += excess[indices].sum(dtype=np.float64)
        means[replicate] = total / predictive.size
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "conditional_rmse_mean": float(predictive.mean()),
        "gt_rmse_mean": float(gt_mean),
        "excess_rmse_mean": float(excess.mean()),
        "excess_rmse_ci95_low": float(lower),
        "excess_rmse_ci95_high": float(upper),
    }, means


def validate_config(cfg, simulator, *, training=True):
    """Require the scaling task; report reduced training/network settings.

    Published weights ship no training section, so ``training=False`` skips the
    required training keys (they are reported as unavailable instead).
    """
    components = len(simulator.model_types)
    if components not in DIMENSIONS:
        raise ValueError("Supported expression counts are 15, 30 and 50")
    expected_class = f"prism.symbolic.paper.model_specs.SymbolicModelV{components}"
    required = {
        "simulator.model_class": expected_class,
        "simulator.observation_format": "signal_blocks",
        "simulator.x_min": 0.0,
        "simulator.x_max": 10.0,
        "simulator.x_step": 0.02,
        "simulator.input_dim": 10,
        "simulator.posterior_score": False,
        "simulator.mask_prior.alpha": 1.0,
        "simulator.mask_prior.beta": 1.0,
        "simulator.mask_prior.min_active_models": 1,
        "model.name": "prism.symbolic.paper.model.SymbolicPaperModelConfigMaskPriorAmortized",
        "model.architecture": "prism.symbolic.paper.model.SymbolicPaperModel",
        "model.tokenizer_cls": "prism.symbolic.paper.tokenizer.SymbolicPaperTokenizerPP",
        "model.model_selection_net.params.mask_prior_dim": 1,
        "model.model_selection_net.params.precondition_with_mask_prior": True,
        "model.inference_loss_type": "v",
        "training.track_ema": True,
        "training.ema_decay": 0.99,
    }
    for path, expected in required.items():
        if not training and path.startswith("training."):
            continue
        if OmegaConf.select(cfg, path) != expected:
            raise ValueError(f"Scaling config mismatch: {path} must be {expected!r}")
    dimensions = (
        len(simulator.noise_types),
        simulator.num_compartments(),
        simulator.theta_dim,
    )
    if dimensions != DIMENSIONS[components]:
        raise ValueError(f"Unexpected noise/mask/theta dimensions: {dimensions}")
    standard = {
        "run.seed": 0,
        "model.init_seed": 0,
        "model.model_dim": 128,
        "model.embedding_net.params.num_layers": 2,
        "model.model_selection_net.params.num_layers": 6,
        "model.inference_net.params.num_layers": 6,
        "model.model_selection_net.params.prior_params_embed_dim": 128,
        "training.dataloader.name": "SimulationDataset",
        "training.dataloader.dataset.buffer_size": 2097152,
        "training.dataloader.dataset.simulation_batch_size": 8192,
        "training.dataloader.train_loader.batch_size": 4096,
        "training.optimizer.optimizer": "radam",
        "training.optimizer.learning_rate": 5e-4,
        "training.optimizer.gradient_clip_value": 2.0,
        "training.optimizer.scheduler": None,
        "training.model_selection_warmup_steps": 0,
        "training.model_selection_ramp_steps": 0,
        "training.label_smoothing": 0.0,
        "training.checkpoint_freq": 2000,
        "training.eval_freq": 4000,
    }
    differences = [
        {"path": path, "standard": value, "actual": OmegaConf.select(cfg, path)}
        for path, value in standard.items()
        if OmegaConf.select(cfg, path) != value
    ]
    return components, differences


def sampling_settings(model):
    decoder = model.inference_decoder
    settings = {
        "method": "ode",
        "num_steps": NUM_STEPS,
        "t_min_argument": None,
        "t_max_argument": None,
        "effective_t_min": float(decoder.train_cfg.t_min),
        "effective_t_max": float(decoder.train_cfg.t_max),
        "solver": type(decoder.solver_cfg).__name__,
        "ode_method": decoder.solver_cfg.ode_method,
    }
    if (
        settings["effective_t_min"] != 2e-4
        or settings["effective_t_max"] != 80.0
        or settings["solver"] != "EDMSolverConfig"
        or settings["ode_method"] != "heun"
    ):
        raise ValueError(f"Source diffusion sampling defaults changed: {settings}")
    return settings


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tree_sha256(directory):
    digest = hashlib.sha256()
    for path in sorted(path for path in directory.rglob("*") if path.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        digest.update(file_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def run_record():
    root = Path(__file__).resolve().parents[3]
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    result = {
        "checkout_commit": git.stdout.strip() if git.returncode == 0 else None,
        "evaluator_sha256": file_sha256(__file__),
    }
    manifest = os.environ.get("SYMBOLIC_SCALING_MANIFEST")
    if manifest:
        path = Path(manifest).resolve()
        record = json.loads(path.read_text())
        result["snapshot_manifest"] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "identifiers": {
                key: record[key]
                for key in (
                    "protocol_id",
                    "snapshot_id",
                    "source_commit",
                    "source_commits",
                    "source_sha256",
                    "snapshot_sha256",
                    "payload_sha256",
                )
                if key in record
            },
        }
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, nargs="?", help="run directory (or --preset)"
    )
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--which", default="latest", help="latest | best | integer step"
    )
    parser.add_argument("--output", required=True, type=Path, help="new .json path")
    add_published_arguments(parser)
    add_slurm_arguments(parser)
    args = parser.parse_args(argv)
    check_weights_source(parser, args.checkpoint, args.preset, args.weights_dir)
    if args.preset and args.which not in ("latest", "best"):
        parser.error("--preset supports --which latest or best")
    if args.num_samples <= 0 or args.batch_size <= 0:
        parser.error("--num-samples and --batch-size must be positive")
    if args.which not in ("latest", "best"):
        try:
            args.which = int(args.which)
            if args.which < 0:
                raise ValueError
        except ValueError:
            parser.error("--which must be latest, best, or a nonnegative integer")
    if args.output.suffix != ".json":
        parser.error("--output must be a .json path")
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        parser.error("Choose a new output path; evaluation files already exist")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.slurm:
        return submit(args, argv, main)
    from prism.simulators.config import resolve_simulator_model
    from prism.train.build_simulator import build_simulator
    from prism.train.utils import (
        find_checkpoint_run_dir,
        load_cfg,
        load_checkpoint,
        select_checkpoint_params,
    )

    published = None
    if args.preset:
        bundle, published = load_published(args.preset, args.which, args.weights_dir)
        run_dir = None
        cfg = OmegaConf.create(bundle.config)
        config_snapshot = OmegaConf.to_container(cfg, resolve=False)
        json.dumps(config_snapshot, allow_nan=False)
        simulator = bundle.simulator
        components, differences = validate_config(cfg, simulator, training=False)
        run = run_record()
        model = bundle.model
        model.eval()
        step = int(bundle.step)
        generators = [bundle.generator]
        sampling = sampling_settings(model)
        checkpoint_hash = published["weights_sha256"]
    else:
        run_dir = find_checkpoint_run_dir(args.checkpoint.resolve())
        cfg = load_cfg(run_dir)
        config_snapshot = OmegaConf.to_container(cfg, resolve=False)
        json.dumps(config_snapshot, allow_nan=False)
        components, differences = validate_config(cfg, resolve_simulator_model(cfg))
        run = run_record()
        checkpoint, model, simulator = load_checkpoint(run_dir, which=args.which)
        # Reuse the repository's strict selector: never substitute raw parameters.
        nnx.update(model, select_checkpoint_params(checkpoint, "params_ema"))
        model.eval()
        step = int(checkpoint["step"])
        if isinstance(args.which, int) and step != args.which:
            raise RuntimeError("Restored step disagrees with the requested checkpoint")
        checkpoint_dir = run_dir / "checkpoints"
        if args.which == "best":
            checkpoint_dir /= "best"
        checkpoint_dir /= str(step)
        if not checkpoint_dir.is_dir():
            raise RuntimeError("Restored checkpoint directory is missing")
        generated_simulator, generators = build_simulator(cfg)
        if generated_simulator is not simulator or len(generators) != 1:
            raise RuntimeError("Checkpoint and evaluation simulator disagree")
        sampling = sampling_settings(model)
        checkpoint_hash = tree_sha256(checkpoint_dir)
    errors = evaluate_errors(
        model, generators[0], simulator, components, args.num_samples, args.batch_size
    )
    metrics, bootstrap_means = summarize(errors["predictive"], errors["gt"])
    key = np.asarray(evaluation_key(components)).tolist()
    report = {
        "protocol_id": PROTOCOL,
        "checkpoint": str(run_dir) if run_dir else None,
        "published_weights": published,
        "which": str(args.which),
        "restored_step": step,
        "params_used": "ema",
        "checkpoint_sha256": checkpoint_hash,
        "components": components,
        "mask_dim": simulator.num_compartments(),
        "theta_dim": simulator.theta_dim,
        "observation_shape": [50, 10],
        "grid": {"start": 0.0, "stop_exclusive": 10.0, "step": 0.02, "size": 500},
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "source_batch_size": SOURCE_BATCH_SIZE,
        "batch_size_is_computational_chunk_only": True,
        "seed": SEED,
        "source_checkpoint_count": len(SOURCE_K),
        "source_checkpoint_index": SOURCE_K.index(components),
        "evaluation_key": key,
        "prior": "Beta(1,1), min one expression, exactly one noise; no fixed lambda",
        "sampling": sampling,
        "metrics": metrics,
        "aggregation": "mean of predictive per-dataset RMSE after subtracting GT mean",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "source_seed": "unspecified",
            "confidence_level": 0.95,
            "generator": "numpy.default_rng (PCG64)",
            "method": "percentile, predictive datasets resampled with GT mean fixed",
            "gt_mean_uncertainty_included": False,
            "accumulation_dtype": "float64",
            "max_draws_per_chunk": 65536,
        },
        "standard_config_differences": differences,
        "source_sample_count_used": args.num_samples == NUM_SAMPLES,
        "source_checkpoint_selection_used": args.which == "latest",
        "reproduction_verdict": "not_assessed",
        "configured_max_train_steps": OmegaConf.select(cfg, "training.max_train_steps"),
        "saved_config": config_snapshot,
        "config_interpolations_resolved": False,
        "config_snapshot_sha256": hashlib.sha256(
            json.dumps(config_snapshot, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "artifact_sha256": (
            file_sha256(run_dir / "artifact.yaml")
            if run_dir is not None and (run_dir / "artifact.yaml").is_file()
            else None
        ),
        "run": run,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "devices": [str(device) for device in jax.devices()],
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "jax",
                "jaxlib",
                "flax",
                "probjax",
                "numpy",
                "orbax-checkpoint",
            )
        },
    }
    # Validate JSON before creating either result, retaining unresolved Hydra text.
    json.dumps(report, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.output.with_suffix(".npz")
    with npz_path.open("xb") as handle:
        np.savez(
            handle,
            predictive_errors=errors["predictive"],
            gt_errors=errors["gt"],
            bootstrap_means=bootstrap_means,
            seed=SEED,
            evaluation_key=key,
            source_checkpoint_index=SOURCE_K.index(components),
            components=components,
            checkpoint=str(run_dir) if run_dir else f"published:{args.preset}",
            which=str(args.which),
            step=step,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )
    report["samples_file"] = str(npz_path.resolve())
    report["samples_sha256"] = file_sha256(npz_path)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
