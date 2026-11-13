import importlib
import json
import logging
import os
import random
import socket
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

# memory_fraction = 0.9  # Use 98% of available memory
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, ListConfig, OmegaConf

from prism import console
from prism.config import build_artifact_config, runtime_config
from prism.simulators.config import resolve_simulator_model
from prism.train.build_model import build_model
from prism.train.build_simulator import build_simulator
from prism.train.checkpointing import CheckpointManager
from prism.train.dataset import instantiate_dataloader
from prism.train.evaluator import build_pure_eval_fns


@dataclass
class TrainState:
    params: Any
    model_state: Any
    opt_state: Any
    ema_state: Any
    rng: Any
    step: int = 0


def tree_copy(pytree: Any) -> Any:
    """Create a shallow copy of a pytree."""
    return jax.tree_util.tree_map(lambda x: x, pytree)


def resolve_run_directory(output_dir: str, name: str) -> str:
    """Keep named run directories stable across launches; honor custom paths."""
    output_dir = os.path.abspath(output_dir)
    candidate = output_dir
    while os.path.dirname(candidate) != candidate:
        if os.path.basename(candidate) == name:
            return candidate
        candidate = os.path.dirname(candidate)
    return output_dir


def configure_environment(cfg: DictConfig) -> tuple[logging.Logger, str, str]:
    """Configure logging and resolve Hydra output directories."""
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    output_super_dir = (
        output_dir
        if hydra_cfg.mode == RunMode.MULTIRUN
        else resolve_run_directory(output_dir, cfg.name)
    )
    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    log.info(f"Output super directory: {output_super_dir}")
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Jax devices: {jax.devices()}")
    return log, output_dir, output_super_dir


def init_wandb_if_needed(cfg: DictConfig) -> bool:
    """Initialise wandb if requested."""
    if not cfg.use_wandb:
        return False
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    return True


def seed_everything(seed: int) -> jax.Array:
    """Seed Python, numpy and JAX RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.key(seed)


def _normalise_loader_params(params_cfg: Any, count: int) -> list[dict[Any, Any]]:
    """Convert Hydra loader params into a list of dictionaries."""
    if params_cfg is None:
        return [dict() for _ in range(count)]
    if isinstance(params_cfg, (DictConfig, ListConfig)):
        container = OmegaConf.to_container(params_cfg, resolve=True)
    else:
        container = params_cfg
    if container is None:
        return [dict() for _ in range(count)]
    if isinstance(container, list):
        entries = [dict(entry) for entry in container]
        if len(entries) < count:
            last_entry = entries[-1] if entries else {}
            for _ in range(count - len(entries)):
                entries.append(dict(last_entry))
        return entries[:count]
    if isinstance(container, dict):
        return [dict(container) for _ in range(count)]
    raise TypeError(f"Unsupported loader parameter type: {type(container)}")


def _resolve_device_spec(
    device_spec: Any, *, default: jax.Device | None = None
) -> jax.Device | None:
    """Resolve a device specification into a concrete ``jax.Device``."""
    if device_spec is None:
        return default
    if isinstance(device_spec, str):
        kind_part, _, index_part = device_spec.strip().partition(":")
        kind = kind_part or None
        index = int(index_part) if index_part else 0
    elif isinstance(device_spec, dict):
        kind = device_spec.get("kind")
        index = device_spec.get("index", 0)
    else:
        raise TypeError(f"Unsupported device specification type: {type(device_spec)}")
    devices = jax.devices(kind) if kind else jax.devices()
    if not devices:
        raise ValueError(f"No devices available for specification {device_spec!r}")
    index = int(index)
    if index < 0 or index >= len(devices):
        raise ValueError(
            f"Device index {index} out of range for specification {device_spec!r}"
        )
    return devices[index]


def _ensure_prng_key(value: Any) -> jax.Array:
    """Normalise user-provided RNG representations to a JAX PRNGKey."""
    if isinstance(value, jax.Array):
        return value
    if isinstance(value, (np.ndarray, list, tuple)):
        raise TypeError(
            "Provide integer seeds for RNG configuration, not array-like values."
        )
    return jax.random.PRNGKey(int(value))


def create_dataloaders(
    cfg: DictConfig, simulators: Sequence[Any]
) -> tuple[list[Any], Any]:
    """Instantiate training and evaluation dataloaders."""
    dataset_module = importlib.import_module("prism.train.dataset")
    loader_name = cfg.train.dataloader.name
    dataset_type = getattr(dataset_module, loader_name)
    file_dataset = issubclass(dataset_type, dataset_module.FileDataset)
    static_dataset = issubclass(dataset_type, dataset_module.StaticDataset)
    simulators = list(simulators)
    if file_dataset:
        datasets = cfg.train.dataloader.get("train_dataset")
        count = len(datasets) if isinstance(datasets, (list, ListConfig)) else 1
        simulators = [None] * count
    if not simulators:
        raise ValueError("Expected at least one simulator callable.")

    loader_cfg_dict = OmegaConf.to_container(cfg.train.dataloader, resolve=True)
    if not isinstance(loader_cfg_dict, dict):
        raise TypeError("cfg.train.dataloader must resolve to a mapping.")

    dataset_base_cfg = loader_cfg_dict.get("dataset")
    if not isinstance(dataset_base_cfg, dict):
        raise ValueError(
            "cfg.train.dataloader.dataset must be defined and resolve to a mapping."
        )

    train_dataset_overrides = _normalise_loader_params(
        loader_cfg_dict.get("train_dataset"), len(simulators)
    )
    val_dataset_override = _normalise_loader_params(
        loader_cfg_dict.get("val_dataset"), 1
    )[0]

    train_loader_section = loader_cfg_dict.get("train_loader")
    if train_loader_section is None:
        train_loader_section = loader_cfg_dict.get("train_params")
    val_loader_section = loader_cfg_dict.get("val_loader")
    if val_loader_section is None:
        val_loader_section = loader_cfg_dict.get("val_params")

    train_loader_overrides = _normalise_loader_params(
        train_loader_section, len(simulators)
    )

    if not any(train_loader_overrides):
        default_train_loader = {
            key: loader_cfg_dict[key]
            for key in ("batch_size", "shuffle", "drop_last")
            if key in loader_cfg_dict
        }
        if not default_train_loader:
            raise ValueError(
                "cfg.train.dataloader.train_loader.batch_size (or legacy keys) must be provided."
            )
        train_loader_overrides = [
            dict(default_train_loader) for _ in range(len(simulators))
        ]

    for override in train_loader_overrides:
        if override.get("batch_size") is None:
            raise ValueError(
                "Each train dataloader configuration must specify batch_size."
            )

    val_loader_override = (
        _normalise_loader_params(val_loader_section, 1)[0]
        if val_loader_section is not None
        else {}
    )

    def build_dataset_params(
        base: dict[str, Any], override: dict[str, Any], rng
    ) -> dict[str, Any]:
        params = dict(base)
        params.update(override)

        if file_dataset:
            path = params.get("path")
            if not path:
                raise ValueError("FileDataset requires a dataset path.")
            return {
                "path": path,
                "sha256": params.get("sha256"),
                "device": _resolve_device_spec(
                    params.get("device"), default=jax.devices()[0]
                ),
                "return_numpy": bool(params.get("return_numpy", False)),
            }

        if static_dataset:
            dataset_size = params.get("dataset_size")
            if dataset_size is None or int(dataset_size) <= 0:
                raise ValueError("StaticDataset requires a positive dataset_size.")
            params["dataset_size"] = int(dataset_size)
            params["return_numpy"] = bool(params.get("return_numpy", False))
            device_spec = params.pop("device", params.pop("simulation_device", None))
            params["device"] = _resolve_device_spec(
                device_spec, default=jax.devices()[0]
            )
            seed = params.pop("seed", None)
            configured_rng = params.pop("rng", None)
            params["rng"] = (
                _ensure_prng_key(configured_rng)
                if configured_rng is not None
                else jax.random.PRNGKey(int(seed))
                if seed is not None
                else rng
            )
            simulation_batch_size = params.get("simulation_batch_size", 1024)
            if simulation_batch_size is not None:
                simulation_batch_size = int(simulation_batch_size)
                if simulation_batch_size <= 0:
                    raise ValueError("simulation_batch_size must be positive or null.")
            params["simulation_batch_size"] = simulation_batch_size
            return params

        batch_size = params.get("simulation_batch_size")
        if batch_size is None:
            raise ValueError(
                "cfg.train.dataloader.dataset.simulation_batch_size must be specified."
            )
        buffer_size = params.get("buffer_size", 4096)
        jit_simulator = params.get("jit_simulator", True)

        params["simulation_batch_size"] = int(batch_size)
        params["buffer_size"] = max(1, int(buffer_size))
        params["jit_simulator"] = bool(jit_simulator)

        params["return_numpy"] = bool(params.get("return_numpy", False))

        simulation_device_spec = params.pop("simulation_device", None)
        default_device = jax.devices()[0]
        params["simulation_device"] = _resolve_device_spec(
            simulation_device_spec, default=default_device
        )

        provided_rng = params.pop("rng", None)
        provided_seed = params.pop("seed", None)
        if provided_rng is not None:
            params["rng"] = _ensure_prng_key(provided_rng)
        elif provided_seed is not None:
            params["rng"] = jax.random.PRNGKey(int(provided_seed))
        else:
            params["rng"] = rng
        return params

    def build_loader_params(base_cfg: dict[str, Any], *, seed: int) -> dict[str, Any]:
        params = dict(base_cfg)
        batch_size = params.get("batch_size")
        if batch_size is None:
            raise ValueError("Each dataloader configuration must specify batch_size.")
        params["batch_size"] = int(batch_size)
        if params.get("seed") is None:
            params["seed"] = seed
        if "devices" in params:
            devices = [
                _resolve_device_spec(dev_spec) for dev_spec in params.get("devices", [])
            ]
        else:
            devices = [jax.devices()[0]]
        params["devices"] = devices
        return params

    base_seed = int(cfg.seed)
    rngs = jax.random.split(jax.random.PRNGKey(base_seed), len(simulators) + 1)
    train_rngs = rngs[: len(simulators)]
    val_rng = rngs[-1]

    def create_dataset(simulator, params, split="train"):
        if file_dataset:
            return dataset_type(simulator_config=cfg.simulator, split=split, **params)
        if not static_dataset:
            return dataset_type(simulator, **params)
        params = dict(params)
        size = params.pop("dataset_size")
        data_rng = params.pop("rng")
        simulation_batch_size = params.pop("simulation_batch_size")
        keys = jax.random.split(data_rng, size)
        data = (
            jax.vmap(simulator)(keys)
            if simulation_batch_size is None
            else jax.lax.map(simulator, keys, batch_size=simulation_batch_size)
        )
        return dataset_type(data, **params)

    train_datasets = []
    for simulator, override, rng in zip(
        simulators, train_dataset_overrides, train_rngs, strict=True
    ):
        dataset_params = build_dataset_params(dataset_base_cfg, override, rng)
        logging.info(f"Creating training dataset with params: {dataset_params}")
        dataset = create_dataset(simulator, dataset_params)
        train_datasets.append(dataset)

    val_dataset_params = build_dataset_params(
        dataset_base_cfg, val_dataset_override, val_rng
    )
    val_dataset = create_dataset(simulators[0], val_dataset_params, split="validation")
    if file_dataset and any(ds.path == val_dataset.path for ds in train_datasets):
        raise ValueError("Training and validation must use different dataset files.")

    train_loaders = []
    for index, (dataset, loader_cfg) in enumerate(
        zip(train_datasets, train_loader_overrides, strict=True)
    ):
        loader_params = build_loader_params(loader_cfg, seed=base_seed + index)
        train_loaders.append(
            instantiate_dataloader(
                dataset,
                loader_params,
                seed=base_seed + index,
            )
        )

    if val_loader_override.get("batch_size") is None:
        val_loader_override["batch_size"] = train_loader_overrides[0]["batch_size"]
    val_loader_params = build_loader_params(
        val_loader_override, seed=base_seed + len(simulators)
    )
    val_loader = instantiate_dataloader(
        val_dataset,
        val_loader_params,
        seed=base_seed + len(simulators),
        default_shuffle=False,
        default_drop_last=False,
    )
    return train_loaders, val_loader


def align_to_inner_steps(
    value: int | None, inner_steps: int, name: str, log: logging.Logger
) -> int | None:
    """Align cadence values to multiples of inner_steps; warn if adjustment is needed."""
    if value is None:
        return None
    if inner_steps <= 0:
        return value
    aligned = (value // inner_steps) * inner_steps
    if aligned == 0 and value > 0:
        log.warning(
            f"{name} ({value}) is smaller than inner_steps ({inner_steps}); using inner_steps instead."
        )
        aligned = inner_steps
    return aligned


def initialize_ema_state(
    track_ema: bool, ema_transform: optax.GradientTransformation | None, params: Any
) -> Any | None:
    """Initialise EMA state if requested."""
    if not track_ema or ema_transform is None:
        return None
    ema_state = ema_transform.init(params)
    return ema_state


def get_ema_params(ema_state: Any | None) -> Any | None:
    """Extract EMA parameters from the EMA state."""
    if ema_state is None:
        return None
    ema_value = getattr(ema_state, "ema", None)
    if ema_value is None:
        return None
    return ema_value


def apply_checkpoint_to_state(
    train_state: TrainState,
    checkpoint: Any | dict[str, Any],
    cfg: DictConfig,
    optimizer: optax.GradientTransformation,
    ema_transform: optax.GradientTransformation | None,
    log: logging.Logger,
    rebuild_optimizer: bool,
) -> TrainState:
    """Populate the in-memory train_state with data from a checkpoint payload."""
    train_state.params = checkpoint["params"]
    train_state.step = checkpoint.get("step", train_state.step)

    if "model_state" in checkpoint:
        train_state.model_state = checkpoint["model_state"]
    else:
        log.warning(
            "Checkpoint does not include model_state; continuing with existing state."
        )

    if rebuild_optimizer or "optimizer_state" not in checkpoint:
        reference = (
            checkpoint.get("params_ema")
            if cfg.train.track_ema and "params_ema" in checkpoint
            else train_state.params
        )
        train_state.opt_state = optimizer.init(reference)  # type: ignore
        if not rebuild_optimizer:
            log.warning(
                "Checkpoint missing optimizer_state; reinitialising optimizer state."
            )
    else:
        train_state.opt_state = checkpoint["optimizer_state"]

    if cfg.train.track_ema:
        stored_ema_state = checkpoint.get("ema_state")
        stored_ema_params = checkpoint.get("params_ema")

        if stored_ema_state is not None:
            train_state.ema_state = stored_ema_state
        elif stored_ema_params is not None and ema_transform is not None:
            train_state.ema_state = initialize_ema_state(
                True,
                ema_transform,
                stored_ema_params,
            )
            log.warning(
                "EMA state missing from checkpoint; reinitialising EMA statistics from stored EMA params."
            )
        elif train_state.ema_state is None and ema_transform is not None:
            train_state.ema_state = initialize_ema_state(
                True,
                ema_transform,
                train_state.params,
            )
            log.warning(
                "EMA information missing from checkpoint; reinitialising EMA statistics from current params."
            )

    if "rng" in checkpoint:
        train_state.rng = checkpoint["rng"]
    else:
        log.warning("Checkpoint missing RNG key; stochastic components will reset.")

    return train_state


def save_training_checkpoint(
    checkpoint_manager: CheckpointManager,
    train_state: TrainState,
    loss_value: float,
    track_ema: bool,
    *,
    val_loss_value: float | None = None,
    write_standard: bool = True,
) -> None:
    """Persist the full training state via the checkpoint manager."""
    checkpoint_manager.save(
        step=train_state.step,
        params=train_state.params,
        optimizer_state=train_state.opt_state,
        loss=loss_value,
        val_loss=val_loss_value,
        write_standard=write_standard,
        params_ema=get_ema_params(train_state.ema_state) if track_ema else None,
        model_state=train_state.model_state,
        ema_state=train_state.ema_state if track_ema else None,
        rng=train_state.rng,
    )


def ensure_fresh_checkpoint_dir(
    checkpoint_manager: CheckpointManager, continue_training: bool
) -> None:
    """Refuse to start a new run in a checkpoint directory that is already used.

    Orbax will not overwrite existing steps, so a second fresh run with the same
    ``run.name`` would train in full while keeping the first run's checkpoints
    as ``latest`` and ``best``.
    """
    if continue_training:
        return
    steps = checkpoint_manager.saved_steps()
    if steps:
        raise FileExistsError(
            f"{checkpoint_manager.ckpt_dir} already holds checkpoints for steps "
            f"{steps}. Set training.continue_training=true to resume that run, "
            "or choose a new run.name."
        )


def resume_from_checkpoint(
    cfg: DictConfig,
    checkpoint_manager: CheckpointManager,
    optimizer: optax.GradientTransformation,
    train_state: TrainState,
    ema_transform: optax.GradientTransformation | None,
    log: logging.Logger,
) -> TrainState:
    """Load the latest checkpoint if continue_training is enabled."""
    if not cfg.train.get("continue_training", False):
        return train_state

    latest_step = checkpoint_manager.get_latest_step()
    if latest_step is None or latest_step <= 0:
        log.warning("No valid checkpoint found. Starting from scratch.")
        return train_state

    log.info(f"Restoring checkpoint at step {latest_step}")
    partial_restore = cfg.train.get("partial_restore", False)
    reference_ema = (
        get_ema_params(train_state.ema_state) if cfg.train.track_ema else None
    )
    checkpoint = checkpoint_manager.restore(
        step=latest_step,
        params=train_state.params,
        optimizer_state=None if partial_restore else train_state.opt_state,
        params_ema=reference_ema,
        model_state=train_state.model_state,
        ema_state=None
        if partial_restore
        else (train_state.ema_state if cfg.train.track_ema else None),
        rng=train_state.rng,
        partial_restore=partial_restore,
    )
    if checkpoint is None:
        log.warning("Failed to restore checkpoint. Starting from scratch.")
        return train_state

    train_state = apply_checkpoint_to_state(
        train_state=train_state,
        checkpoint=checkpoint,
        cfg=cfg,
        optimizer=optimizer,
        ema_transform=ema_transform,
        log=log,
        rebuild_optimizer=cfg.train.restart_optimizer or partial_restore,
    )
    log.info(f"Resumed training from step {train_state.step}")
    return train_state


def model_selection_weight(cfg: DictConfig, step):
    """Optionally delay and ramp the model-selection contribution to the loss."""
    weight = jnp.asarray(cfg.train.model_selection_weight, dtype=jnp.float32)
    warmup = max(0, int(cfg.train.get("model_selection_warmup_steps", 0) or 0))
    ramp = max(0, int(cfg.train.get("model_selection_ramp_steps", 0) or 0))
    if warmup == 0 and ramp == 0:
        return weight
    if ramp == 0:
        return jnp.where(step < warmup, 0.0, weight)
    return weight * jnp.clip((jnp.asarray(step) - warmup) / ramp, 0.0, 1.0)


def build_loss_fn(cfg: DictConfig, graphdef: Any):
    """Create the loss function closure."""

    def loss_fn(params, state, data, rng, step=0):
        if isinstance(data, (tuple, list)):
            batches = tuple(data)
        else:
            batches = (data,)
        # ENSURE NO NaNs IN THE BATCHES
        batches = jax.tree_util.tree_map(jax.numpy.nan_to_num, batches)
        num_batches = len(batches)
        if num_batches == 0:
            raise ValueError("Expected at least one batch for loss computation.")

        rngs = jax.random.split(rng, num_batches)
        model = nnx.merge(graphdef, params, state, copy=True)
        model.train()

        loss1 = []
        loss2 = []

        for batch, subkey in zip(batches, rngs, strict=True):
            losses = model.loss_fn(
                subkey,
                **batch,
                weight_by_complexity=cfg.train.weight_by_complexity,
                label_smoothing=cfg.train.label_smoothing,
                use_loss_mask=cfg.train.use_loss_mask,
                cut_off_tsm=cfg.train.cut_off_tsm,
                permute_order=cfg.train.get("permute_order", False),
            )
            loss1.append(losses[0])
            loss2.append(losses[1])

        normaliser = 1.0 / num_batches
        loss1 = model_selection_weight(cfg, step) * normaliser * sum(loss1)
        loss2 = cfg.train.model_inference_loss_weight * normaliser * sum(loss2)
        total_loss = loss1 + loss2
        _, _, new_state = nnx.split(model, nnx.Param, ...)
        return total_loss, (loss1, loss2, new_state)

    return loss_fn


def build_update_fn(
    optimizer: optax.GradientTransformation,
    loss_fn,
    use_ema: bool,
    ema_transform: optax.GradientTransformation | None,
):
    """Create the JIT-compiled update step including optional EMA maintenance."""
    if use_ema and ema_transform is None:
        raise ValueError("EMA requested but no ema_transform provided.")

    @jax.jit
    def update(params, state, opt_state, ema_state, data, rng, step=0):
        (total_loss, (loss1, loss2, new_state)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, state, data, rng, step)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        if use_ema:
            _, ema_state = ema_transform.update(new_params, ema_state)  # type: ignore
        return new_params, new_state, opt_state, ema_state, (loss1, loss2), total_loss

    return update


def train_loop(
    cfg: DictConfig,
    log: logging.Logger,
    train_state: TrainState,
    optimizer: optax.GradientTransformation,
    update_step,
    checkpoint_manager: CheckpointManager,
    evaluator,
    loaders: Sequence[Any],
    eval_loader: Any,
    track_ema: bool,
    ema_transform: optax.GradientTransformation | None,
    checkpoint_freq: int | None,
    eval_freq: int | None,
    restart_every: int | None,
    max_train_hours: float,
    wandb_active: bool,
    max_train_steps: int | None = None,
):
    """Main training loop."""
    loaders = list(loaders)
    if not loaders:
        raise ValueError("At least one training dataloader is required.")
    inner_steps = cfg.train.inner_steps
    if inner_steps <= 0:
        raise ValueError("inner_steps must be positive.")
    if max_train_steps is not None:
        if max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive or null.")
        if train_state.step >= max_train_steps:
            log.info(f"Training already reached step {train_state.step}.")
            return train_state
    datastreams = [iter(loader) for loader in loaders]
    start_time = time.time()

    def recover_from_latest_checkpoint(reason: str) -> bool:
        nonlocal train_state, datastreams
        log.warning(reason)
        try:
            latest_step = checkpoint_manager.get_latest_step()
            if latest_step is None:
                log.warning("No checkpoints available for recovery. Cannot reset.")
                return False
            partial_restore = cfg.train.get("partial_restore", False)
            checkpoint = checkpoint_manager.restore(
                step=latest_step,
                params=train_state.params,
                optimizer_state=None if partial_restore else train_state.opt_state,
                params_ema=get_ema_params(train_state.ema_state) if track_ema else None,
                model_state=train_state.model_state,
                ema_state=None
                if partial_restore
                else (train_state.ema_state if track_ema else None),
                rng=train_state.rng,
                partial_restore=partial_restore,
            )
            if checkpoint is None:
                log.warning("Failed to restore recovery checkpoint. Cannot reset.")
                return False
            train_state = apply_checkpoint_to_state(
                train_state=train_state,
                checkpoint=checkpoint,
                cfg=cfg,
                optimizer=optimizer,
                ema_transform=ema_transform,
                log=log,
                rebuild_optimizer=partial_restore,
            )
            datastreams = [iter(loader) for loader in loaders]
            log.info(f"Recovered to step {train_state.step}")
            return True
        except Exception as err:  # pragma: no cover - defensive logging
            log.error(f"Error during recovery: {err}")
            log.warning("Continuing without recovery.")
            return False

    log.info("Compiling training and evaluation step...")
    _ = update_step(
        train_state.params,
        train_state.model_state,
        train_state.opt_state,
        train_state.ema_state,
        next(datastreams[0]),
        train_state.rng,
        jnp.asarray(train_state.step, dtype=jnp.int32),
    )
    _ = evaluator.eval_nnl_mask(
        train_state.params,
        train_state.model_state,
        eval_loader,
        iters=cfg.train.eval.nnl_mask.iters,
    )
    _ = evaluator.eval_nnl_theta(
        train_state.params,
        train_state.model_state,
        eval_loader,
        iters=cfg.train.eval.nnl_theta.iters,
    )
    if getattr(evaluator, "predictive_rmse", None) is not None:
        _ = evaluator.eval_predictive_rmse(
            train_state.params,
            train_state.model_state,
            eval_loader,
            iters=cfg.train.eval.nnl_theta.iters,
        )

    log.info(f"Maximum training time: {max_train_hours} hours")
    if max_train_steps is not None:
        log.info(f"Maximum training steps: {max_train_steps}")
    while True:
        loss_mask = []
        loss_theta = []
        for _ in range(inner_steps):
            train_state.rng, subkey = jax.random.split(train_state.rng)
            batches = tuple(next(stream) for stream in datastreams)
            data = batches if len(batches) > 1 else batches[0]
            (
                train_state.params,
                train_state.model_state,
                train_state.opt_state,
                train_state.ema_state,
                losses,
                _,
            ) = update_step(
                train_state.params,
                train_state.model_state,
                train_state.opt_state,
                train_state.ema_state,
                data,
                subkey,
                jnp.asarray(train_state.step, dtype=jnp.int32),
            )
            train_state.step += 1
            loss_mask.append(losses[0])
            loss_theta.append(losses[1])
            if max_train_steps is not None and train_state.step >= max_train_steps:
                break
        loss_mask = float(sum(loss_mask) / len(loss_mask))
        loss_theta = float(sum(loss_theta) / len(loss_theta))
        total_loss_value = float(loss_mask + loss_theta)

        if not (
            np.isfinite(loss_mask)
            and np.isfinite(loss_theta)
            and np.isfinite(total_loss_value)
        ):
            if recover_from_latest_checkpoint(
                f"Non-finite loss detected at step {train_state.step}. "
                f"loss_mask={loss_mask}, loss_theta={loss_theta}. Resetting to latest checkpoint."
            ):
                continue
            log.error(
                "Non-finite loss encountered but no checkpoint could be restored. Stopping training."
            )
            break

        # Collect and average dataset stats from training loaders
        dataset_stats = {}
        train_datasets = [loader._ds for loader in loaders if hasattr(loader, "_ds")]
        if train_datasets:
            # Collect stats from all training datasets
            all_stats = []
            for dataset in train_datasets:
                if hasattr(dataset, "get_stats"):
                    all_stats.append(dataset.get_stats())

            # Compute averaged production time across datasets
            avg_times = [
                stats.get("avg_production_time_per_batch")
                for stats in all_stats
                if stats and "avg_production_time_per_batch" in stats
            ]
            if avg_times:
                dataset_stats["avg_production_time"] = sum(avg_times) / len(avg_times)
            samples_written = sum(
                stats.get("samples_written", 0) for stats in all_stats if stats
            )
            samples_read = sum(
                stats.get("samples_requested", 0) for stats in all_stats if stats
            )
            if samples_written:
                dataset_stats["samples_read/samples_written"] = (
                    samples_read / samples_written
                )
        log.info(
            f"Step {train_state.step}, Loss mask: {loss_mask}, Loss theta: {loss_theta}, sim. time: {dataset_stats.get('avg_production_time', 0.0):.4f}s, read/wrt: {dataset_stats.get('samples_read/samples_written', 0.0):.4f}"
        )

        if restart_every and train_state.step % restart_every == 0:
            log.info(f"Restarting optimizer at step {train_state.step}")
            reference = get_ema_params(train_state.ema_state) if track_ema else None
            if reference is None:
                reference = train_state.params
            train_state.opt_state = optimizer.init(reference)

        if wandb_active:
            wandb_dict = {
                "loss mask": loss_mask,
                "loss theta": loss_theta,
                "step": train_state.step,
                "sim_time": dataset_stats.get("avg_production_time", 0.0),
                "sim_read_write_ratio": dataset_stats.get(
                    "samples_read/samples_written", 0.0
                ),
            }
            # Add dataset stats to wandb
            wandb_dict.update(dataset_stats)
            wandb.log(wandb_dict)

        elapsed_hours = (time.time() - start_time) / 3600
        stopping = elapsed_hours >= max_train_hours or (
            max_train_steps is not None and train_state.step >= max_train_steps
        )

        if (
            eval_freq
            and train_state.step > 0
            and (not stopping or cfg.train.get("eval_at_end", True))
            and (train_state.step % eval_freq == 0 or stopping)
        ):
            log.info(f"Evaluating model at step {train_state.step}")
            params_eval = get_ema_params(train_state.ema_state) if track_ema else None
            if params_eval is None:
                params_eval = train_state.params
            mask_nnl = evaluator.eval_nnl_mask(
                params_eval,
                train_state.model_state,
                eval_loader,
                iters=cfg.train.eval.nnl_mask.iters,
            )
            theta_nnl = evaluator.eval_nnl_theta(
                params_eval,
                train_state.model_state,
                eval_loader,
                iters=cfg.train.eval.nnl_theta.iters,
            )
            predictive_rmse = None
            if getattr(evaluator, "predictive_rmse", None) is not None:
                predictive_rmse = evaluator.eval_predictive_rmse(
                    params_eval,
                    train_state.model_state,
                    eval_loader,
                    iters=cfg.train.eval.nnl_theta.iters,
                )
            log.info(
                f"Mask NLL: {mask_nnl}, Theta NLL: {theta_nnl}"
                + (
                    f", Predictive RMSE: {predictive_rmse}"
                    if predictive_rmse is not None
                    else ""
                )
            )
            if wandb_active:
                metrics = {
                    "mask_negative_log_likelihood": float(mask_nnl),
                    "theta_negative_log_likelihood": float(theta_nnl),
                    "step": train_state.step,
                }
                if predictive_rmse is not None:
                    metrics["predictive_rmse"] = float(predictive_rmse)
                wandb.log(metrics)
            validation_metric = float(mask_nnl + theta_nnl)
            ksd_metrics = None
            ksd_cfg = cfg.train.eval.get("ksd", None)
            if ksd_cfg:
                ksd_metrics = evaluator.eval_ksd(
                    params_eval,
                    train_state.model_state,
                    eval_loader,
                    iters=ksd_cfg.get("iters", 1),
                    num_samples=ksd_cfg.get("num_samples", 32),
                    bandwidths=ksd_cfg.get("bandwidths", (0.05, 0.1, 0.5)),
                    n_bootstrap=ksd_cfg.get("n_bootstrap", 128),
                    ksd_seed=ksd_cfg.get("seed", cfg.seed),
                )
                log.info(
                    f"KSD: {ksd_metrics.get('ksd')}, p-value: {ksd_metrics.get('ksd_pvalue')}"
                )
                if wandb_active:
                    wandb.log({
                        "ksd": float(ksd_metrics.get("ksd", float("nan"))),
                        "ksd_pvalue": float(
                            ksd_metrics.get("ksd_pvalue", float("nan"))
                        ),
                        "step": train_state.step,
                    })
            if checkpoint_manager.keep_best:
                save_training_checkpoint(
                    checkpoint_manager,
                    train_state,
                    total_loss_value,
                    track_ema,
                    val_loss_value=validation_metric,
                    write_standard=False,
                )
            if checkpoint_manager.should_recover(validation_metric):
                if recover_from_latest_checkpoint(
                    f"Recovery triggered at step {train_state.step} based on validation metrics. Restoring from checkpoint."
                ):
                    continue

        if stopping or (
            checkpoint_freq
            and train_state.step > 0
            and train_state.step % checkpoint_freq == 0
        ):
            log.info(f"Saving checkpoint at step {train_state.step}")
            save_training_checkpoint(
                checkpoint_manager,
                train_state,
                total_loss_value,
                track_ema,
            )
        if stopping:
            log.info(f"Training limit reached at step {train_state.step}.")
            break

    log.info("Training complete")
    checkpoint_manager.wait_until_finished()
    return train_state


def main():
    """Main script function."""
    console.print_logo()
    _main()


def build_optimizer(optimizer_cfg):
    optimizer_type = getattr(optax, optimizer_cfg.optimizer)
    if optimizer_cfg.scheduler:
        scheduler_type = getattr(optax, optimizer_cfg.scheduler)
    else:
        scheduler_type = None
    use_ema = optimizer_cfg.get("ema", False)
    use_adaptive_clip = optimizer_cfg.get("adaptive_gradient_clipping", False)
    grad_transforms = []
    if use_adaptive_clip:
        grad_clip = optax.adaptive_grad_clip(
            optimizer_cfg.get("gradient_clip_value", 10.0)
        )
        grad_transforms.append(grad_clip)
    if scheduler_type:
        scheduler = scheduler_type(**optimizer_cfg.scheduler_params)
        optimizer = optimizer_type(scheduler)
    else:
        lr = optimizer_cfg.get("learning_rate", 1e-4)
        optimizer = optimizer_type(learning_rate=lr)
    grad_transforms.append(optimizer)
    if use_ema:
        grad_transforms.append(optax.ema(optimizer_cfg.get("ema_decay", 0.8)))

    return optax.chain(*grad_transforms)


def record_offline_datasets(train_loaders, eval_loader, run_dir, *, resume):
    """Preserve dataset identity across resumed runs, allowing files to move."""
    from pathlib import Path

    def dataset_record(loader):
        dataset = loader._ds
        return {"path": str(dataset.path), "sha256": dataset.sha256}

    def identity(record):
        return (
            [item["sha256"] for item in record["train"]],
            record["validation"]["sha256"],
        )

    record = {
        "train": [dataset_record(loader) for loader in train_loaders],
        "validation": dataset_record(eval_loader),
    }
    path = Path(run_dir) / "datasets.json"
    if resume:
        if not path.is_file():
            raise ValueError(
                "Cannot verify the checkpoint's datasets: datasets.json is missing. "
                "Restore the original record or start a new run directory."
            )
        previous = json.loads(path.read_text())
        if identity(previous) != identity(record):
            raise ValueError(
                "Dataset checksum or stream order changed since this run started. "
                "Resume with the same datasets or start a new run directory."
            )
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


@hydra.main(config_path="../../conf", config_name="train.yaml", version_base=None)
def _main(cfg: DictConfig):
    if OmegaConf.select(cfg, "model.name") is None:
        raise ValueError(
            "No model selected. Choose a preset, e.g. "
            "`prism train +experiment/train=symbolic`, or set `simulator=NAME model=NAME`."
        )
    cfg = runtime_config(cfg, "train")
    offline = OmegaConf.select(cfg, "train.dataloader.name") == "FileDataset"
    if offline and OmegaConf.select(cfg, "train.eval.ksd.iters", default=0) > 0:
        raise ValueError(
            "File-only training requires training.eval.ksd.iters=0; "
            "KSD uses the simulator likelihood. Evaluate it separately."
        )
    log, output_dir, output_super_dir = configure_environment(cfg)
    OmegaConf.save(
        build_artifact_config(cfg), os.path.join(output_dir, "artifact.yaml")
    )
    wandb_active = init_wandb_if_needed(cfg)

    with ExitStack() as cleanup:
        if wandb_active:
            cleanup.callback(wandb.finish)
        rng_key = seed_everything(cfg.seed)
        log.info(f"Seed: {cfg.seed}")

        log.info("Loading simulator definition" if offline else "Building simulator")
        log.info(f"Simulator cfg: {cfg.simulator}")
        if offline:
            sim_type, simulators = resolve_simulator_model(cfg.simulator), []
        else:
            sim_type, simulators = build_simulator(cfg)
        log.info(f"Simulator type: {sim_type}")

        model = build_model(cfg, sim_type)
        model.train()
        log.info(f"Model cfg: {model.cfg}")
        graphdef, params, state = nnx.split(model, nnx.Param, ...)

        evaluator = build_pure_eval_fns(
            graphdef,
            sim_type,
            fixed_design=sim_type.evaluation_design(cfg.simulator),
        )

        # Single runs share a named root for resume; multirun jobs stay isolated.
        checkpoint_dir = os.path.join(output_super_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log.info(f"Checkpoint directory: {checkpoint_dir}")
        checkpoint_manager = CheckpointManager(
            ckpt_dir=checkpoint_dir,
            max_to_keep=OmegaConf.select(
                cfg, "training.checkpoint.max_checkpoints", default=3
            ),
            keep_best=OmegaConf.select(
                cfg, "training.checkpoint.keep_best_checkpoint", default=True
            ),
            recovery_threshold=OmegaConf.select(
                cfg, "training.recovery_threshold", default=float("inf")
            ),
            continue_training=cfg.train.get("continue_training", False),
        )
        cleanup.callback(checkpoint_manager.close)
        ensure_fresh_checkpoint_dir(
            checkpoint_manager, cfg.train.get("continue_training", False)
        )

        optimizer = build_optimizer(cfg.train.optimizer)
        opt_state = optimizer.init(params)

        ema_transform = (
            optax.ema(cfg.train.ema_decay, debias=False)
            if cfg.train.track_ema
            else None
        )
        ema_state = initialize_ema_state(cfg.train.track_ema, ema_transform, params)

        train_state = TrainState(
            params=params,
            model_state=state,
            opt_state=opt_state,
            ema_state=ema_state,
            rng=rng_key,
            step=0,
        )
        train_state = resume_from_checkpoint(
            cfg, checkpoint_manager, optimizer, train_state, ema_transform, log
        )

        loss_fn = build_loss_fn(cfg, graphdef)
        update_step = build_update_fn(
            optimizer, loss_fn, cfg.train.track_ema, ema_transform
        )

        train_loaders, eval_loader = create_dataloaders(cfg, simulators)
        for loader in [*train_loaders, eval_loader]:
            dataset_close = getattr(getattr(loader, "_ds", None), "close", None)
            if dataset_close is not None:
                cleanup.callback(dataset_close)
            cleanup.callback(loader.close)

        if offline:
            record_offline_datasets(
                train_loaders,
                eval_loader,
                output_super_dir,
                resume=train_state.step > 0,
            )

        inner_steps = cfg.train.inner_steps
        checkpoint_freq = align_to_inner_steps(
            cfg.train.checkpoint_freq, inner_steps, "checkpoint frequency", log
        )
        eval_freq = align_to_inner_steps(
            cfg.train.eval_freq, inner_steps, "evaluation frequency", log
        )
        restart_every = align_to_inner_steps(
            cfg.train.get("restart_every"), inner_steps, "restart frequency", log
        )

        max_train_hours = cfg.train.get("max_train_hours", float("inf"))
        max_train_steps = cfg.train.get("max_train_steps")
        if max_train_steps is not None:
            max_train_steps = int(max_train_steps)

        train_loop(
            cfg=cfg,
            log=log,
            train_state=train_state,
            optimizer=optimizer,
            update_step=update_step,
            checkpoint_manager=checkpoint_manager,
            evaluator=evaluator,
            loaders=train_loaders,
            eval_loader=eval_loader,
            track_ema=cfg.train.track_ema,
            ema_transform=ema_transform,
            checkpoint_freq=checkpoint_freq,
            eval_freq=eval_freq,
            restart_every=restart_every,
            max_train_hours=max_train_hours,
            wandb_active=wandb_active,
            max_train_steps=max_train_steps,
        )


if __name__ == "__main__":
    main()
