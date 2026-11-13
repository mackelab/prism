"""Load symbolic PRISM checkpoints stored as safetensors.

The files hold bare EMA parameters. Each tensor key is the Flax NNX parameter
path joined with ``/`` (the trailing ``value`` node is dropped). A
``config.yaml`` next to the weights holds the resolved ``model`` and
``simulator`` sections, so a checkpoint is rebuilt without knowing its preset.

The published layout is ``<preset>/<best|latest>/params_ema.safetensors`` and
``<preset>/config.yaml``; see :data:`DEFAULT_REPO_ID`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_REPO_ID = "manugloeck/prism-symbolic-pretrained"
WEIGHTS_FILE = "params_ema.safetensors"
CONFIG_FILE = "config.yaml"
WHICH_CHOICES = (
    "best",
    "latest",
    "figure",
)  # "figure": the checkpoint a paper figure used


@dataclass(frozen=True)
class PretrainedSymbolic:
    model: Any
    simulator: Any  # the simulator class (``sim_type``)
    generator: Any  # one training-data generator, usable as ``generator(key)``
    config: Any
    step: int | None
    preset: str | None
    which: str | None


def _key_of(path) -> str:
    parts = [
        str(getattr(k, "key", getattr(k, "idx", getattr(k, "name", k)))) for k in path
    ]
    if parts and parts[-1] == "value":
        parts = parts[:-1]
    return "/".join(parts)


def save_params_safetensors(model, path, metadata: dict[str, str] | None = None):
    """Write ``model``'s parameters in the published safetensors layout."""
    import jax
    import numpy as np
    from flax import nnx
    from safetensors.numpy import save_file

    leaves = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]
    tensors = {_key_of(p): np.ascontiguousarray(np.asarray(v)) for p, v in leaves}
    if len(tensors) != len(leaves):
        raise ValueError("Model parameter paths are not unique.")
    save_file(
        tensors, os.fspath(path), metadata={"key_separator": "/", **(metadata or {})}
    )


def load_params_safetensors(model, path) -> dict[str, str]:
    """Copy safetensors parameters into ``model`` and return the file metadata.

    The tensor names and shapes must equal the model's parameters exactly;
    anything else raises ``ValueError`` before the model is changed.
    """
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from safetensors import safe_open

    state = nnx.state(model, nnx.Param)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
    expected = {_key_of(p): leaf for p, leaf in leaves}
    if len(expected) != len(leaves):
        raise ValueError("Model parameter paths are not unique.")
    with safe_open(os.fspath(path), "np") as reader:
        stored = set(reader.keys())
        missing, extra = sorted(set(expected) - stored), sorted(stored - set(expected))
        if missing or extra:
            raise ValueError(
                f"Checkpoint does not match the model: {len(missing)} missing "
                f"(e.g. {missing[:2]}), {len(extra)} unexpected (e.g. {extra[:2]})."
            )
        values = []
        for p, leaf in leaves:
            key = _key_of(p)
            array = reader.get_tensor(key)
            if tuple(array.shape) != tuple(leaf.shape):
                raise ValueError(
                    f"Shape mismatch for {key}: file {tuple(array.shape)}, "
                    f"model {tuple(leaf.shape)}."
                )
            values.append(jnp.asarray(array, dtype=leaf.dtype))
        metadata = dict(reader.metadata() or {})
    nnx.update(model, treedef.unflatten(values))
    return metadata


def build_from_config(cfg):
    """Build ``(model, sim_type, generator)`` from a saved model/simulator config."""
    from prism.train.build_model import build_model
    from prism.train.build_simulator import build_simulator

    sim_type, generators = build_simulator(cfg)
    if len(generators) != 1:
        raise ValueError("Expected a single simulator generator in the config.")
    return build_model(cfg, sim_type), sim_type, generators[0]


def load_pretrained_symbolic(
    preset: str | None = None,
    which: str = "best",
    *,
    repo_id: str = DEFAULT_REPO_ID,
    path=None,
    revision: str | None = None,
    cache_dir=None,
    token=None,
    local_files_only: bool = False,
) -> PretrainedSymbolic:
    """Rebuild a symbolic model and load its EMA weights.

    Pass ``preset`` (for example ``"symbolic_sbmi_10k"`` or ``"symbolic15"``)
    to download from the Hub, or ``path`` to a local directory laid out as
    ``<which>/params_ema.safetensors`` next to ``config.yaml``.
    """
    from omegaconf import OmegaConf

    if which not in WHICH_CHOICES:
        raise ValueError(f"which must be one of {WHICH_CHOICES}, got {which!r}.")
    if (path is None) == (preset is None):
        raise ValueError("Provide exactly one of preset (Hub) or path (local).")
    if path is None:
        from huggingface_hub import hf_hub_download

        def fetch(filename):
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )

        config_path = fetch(f"{preset}/{CONFIG_FILE}")
        weights_path = fetch(f"{preset}/{which}/{WEIGHTS_FILE}")
    else:
        root = os.fspath(path)
        config_path = os.path.join(root, CONFIG_FILE)
        weights_path = os.path.join(root, which, WEIGHTS_FILE)

    cfg = OmegaConf.load(config_path)
    model, sim_type, generator = build_from_config(cfg)
    metadata = load_params_safetensors(model, weights_path)
    model.eval()
    step = metadata.get("step")
    return PretrainedSymbolic(
        model=model,
        simulator=sim_type,
        generator=generator,
        config=cfg,
        step=int(step) if step is not None else None,
        preset=metadata.get("prism_preset", preset),
        which=which,
    )
