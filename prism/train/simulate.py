"""Save fixed symbolic datasets from a configured simulator class."""

import argparse
import hashlib
from pathlib import Path

import jax
import numpy as np
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from prism.config import normalize_config
from prism.simulators.config import resolve_simulator_model
from prism.train.build_simulator import build_simulator
from prism.train.offline_data import save_dataset


def generate_datasets(
    cfg,
    output,
    *,
    train_size,
    validation_size,
    simulation_batch_size=256,
    device="cpu",
):
    """Save train/validation data and a file-only training config from a preset."""
    if any(
        not isinstance(n, int) or n < 1
        for n in (train_size, validation_size, simulation_batch_size)
    ):
        raise ValueError(
            "Dataset sizes and simulation batch size must be positive integers."
        )
    cfg = normalize_config(cfg, "train")
    launch_config = _training_launch_config(cfg.pop("hydra", None))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if not getattr(
        resolve_simulator_model(cfg.simulator), "supports_offline_data", False
    ):
        raise ValueError(
            "Offline data generation requires a simulator whose samples contain "
            "only arrays (supports_offline_data); dMRI acquisitions are objects."
        )
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new dataset directory: {output}")
    loader = OmegaConf.to_container(cfg.training.dataloader, resolve=True)
    for name in ("train_loader", "val_loader"):
        params = loader.get(name) or {}
        for settings in params if isinstance(params, list) else [params]:
            settings["drop_last"] = False
        loader[name] = params
    checksums = {}

    with jax.default_device(jax.devices(device)[0]):
        _, (generate_sample,) = build_simulator(cfg)
        train_key, validation_key = jax.random.split(jax.random.PRNGKey(cfg.run.seed))
        train_key = _dataset_key(loader, "train", train_key)
        validation_key = _dataset_key(loader, "val", validation_key)
        if np.array_equal(train_key, validation_key):
            raise ValueError(
                "Training and validation must use different dataset seeds."
            )

        output.mkdir(parents=True)
        OmegaConf.save(cfg, output / "simulation.yaml")
        for split, size, key in (
            ("train", train_size, train_key),
            ("validation", validation_size, validation_key),
        ):
            sample_keys = jax.random.split(key, size)
            data = jax.lax.map(
                generate_sample, sample_keys, batch_size=simulation_batch_size
            )
            dataset_path = output / f"{split}.npz"
            save_dataset(
                dataset_path,
                data,
                simulator_config=cfg.simulator,
                generation={
                    "split": split,
                    "seed": int(cfg.run.seed),
                    "dataset_key": np.asarray(key).tolist(),
                    "simulation_batch_size": simulation_batch_size,
                },
            )
            with dataset_path.open("rb") as stream:
                checksums[split] = hashlib.file_digest(stream, "sha256").hexdigest()
            print(f"Saved {size} samples to {dataset_path}", flush=True)
            del data

    cfg.run.name += "_offline"
    cfg.training.dataloader = {
        "name": "FileDataset",
        "dataset": {"directory": str(output), "device": "cpu"},
        "train_dataset": {
            "path": "${training.dataloader.dataset.directory}/train.npz",
            "sha256": checksums["train"],
        },
        "val_dataset": {
            "path": "${training.dataloader.dataset.directory}/validation.npz",
            "sha256": checksums["validation"],
        },
        "train_loader": loader["train_loader"],
        "val_loader": loader.get("val_loader", {}),
    }
    # KSD requires a likelihood; run task-specific diagnostics separately.
    OmegaConf.update(cfg, "training.eval.ksd.iters", 0, force_add=True)
    cfg = OmegaConf.merge(cfg, launch_config)
    OmegaConf.save(cfg, output / "training.yaml")
    return cfg


def _training_launch_config(hydra_config):
    """Keep reusable launch settings without freezing Hydra's runtime state."""
    if hydra_config is None:
        return {}
    config = {
        "hydra": OmegaConf.to_container(
            OmegaConf.masked_copy(hydra_config, ["mode", "launcher", "run", "sweep"]),
            resolve=False,
        )
    }
    launcher = OmegaConf.select(hydra_config, "runtime.choices.hydra/launcher")
    if launcher is not None:
        config["defaults"] = [{"override hydra/launcher": launcher}, "_self_"]
    return config


def _dataset_key(loader, split, default_key):
    """Use the same optional dataset seed overrides as the existing trainer."""
    overrides = loader.get(f"{split}_dataset") or {}
    if isinstance(overrides, list):
        (overrides,) = overrides
    params = {**loader.get("dataset", {}), **overrides}
    seed = params.get("rng")
    if seed is None:
        seed = params.get("seed")
    return default_key if seed is None else jax.random.PRNGKey(int(seed))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="prism simulate", description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-size", type=int, required=True)
    parser.add_argument("--simulation-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args, overrides = parser.parse_known_args(argv)
    if any("=" not in value for value in overrides):
        parser.error("Configuration overrides must use key=value syntax.")
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="train", overrides=overrides, return_hydra_config=True
        )
    generate_datasets(cfg, **vars(args))
    print(f"Training configuration: {args.output.resolve() / 'training.yaml'}")


if __name__ == "__main__":
    main()
