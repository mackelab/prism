"""Save and load fixed training data as numeric NPZ arrays and JSON."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

_REQUIRED = {"x", "theta", "model_mask", "mask_prior"}


def _json_mapping(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise ValueError(
            "Dataset configuration and generation metadata must be mappings"
        )
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def _validate(data):
    if not _REQUIRED.issubset(data) or set(data) - _REQUIRED - {"target_score"}:
        raise ValueError(
            "Dataset fields must be x, theta, model_mask, mask_prior and optional target_score"
        )
    arrays = [
        value
        for name, value in data.items()
        if name != "mask_prior" or value is not None
    ]
    for array in arrays:
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("Dataset arrays must contain finite real numeric values")
        if array.ndim == 0:
            raise ValueError("Dataset arrays need a sample dimension")
    count = data["theta"].shape[0]
    if count == 0 or any(array.shape[0] != count for array in arrays):
        raise ValueError("Dataset arrays need the same nonempty sample dimension")
    for name in ("theta", "model_mask"):
        if data[name].ndim != 2 or data[name].shape[1] == 0:
            raise ValueError(f"Dataset {name} must have shape (samples, components)")
    if not np.isin(data["model_mask"], (0, 1)).all():
        raise ValueError("Dataset model_mask must contain only binary values")
    if data["mask_prior"] is not None and data["mask_prior"].ndim != 2:
        raise ValueError("Dataset mask_prior must have shape (samples, prior_features)")
    if data["x"].ndim < 2 or 0 in data["x"].shape[1:]:
        raise ValueError("Dataset x needs nonempty observation dimensions")
    if "target_score" in data and data["target_score"].shape != data["theta"].shape:
        raise ValueError("Dataset target_score shape must match theta")
    return count


def save_dataset(path, data, *, simulator_config, generation):
    """Save one dataset; never overwrite an existing path."""
    if not isinstance(data, Mapping):
        raise ValueError("Dataset must be a mapping")
    data = {
        name: None if name == "mask_prior" and value is None else np.asarray(value)
        for name, value in data.items()
    }
    count = _validate(data)
    metadata = {
        "version": 1,
        "simulator_config": _json_mapping(simulator_config),
        "generation": _json_mapping(generation),
        "count": count,
        "mask_prior_is_none": data["mask_prior"] is None,
    }
    arrays = {name: value for name, value in data.items() if value is not None}
    encoded = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    path = Path(path)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            np.savez(stream, **arrays, __metadata__=encoded)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    return metadata


def load_dataset(path, *, simulator_config=None):
    """Return (data, metadata) without constructing or calling a simulator.

    NumPy arrays and an optional None prior can be passed to StaticDataset.
    Earlier symbolic archives with extra descriptive metadata remain readable.
    """
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Dataset must be an NPZ archive")
    with archive:
        if "__metadata__" not in archive.files:
            raise ValueError("Dataset archive is missing metadata")
        encoded = archive["__metadata__"]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("Dataset metadata must be a scalar Unicode JSON string")
        metadata = json.loads(encoded.item())
        required = {
            "version",
            "simulator_config",
            "generation",
            "count",
            "mask_prior_is_none",
        }
        if not isinstance(metadata, dict) or not required.issubset(metadata):
            raise ValueError("Dataset metadata is incomplete")
        if type(metadata["version"]) is not int or metadata["version"] != 1:
            raise ValueError("Unsupported dataset format version")
        saved_config = _json_mapping(metadata["simulator_config"])
        _json_mapping(metadata["generation"])
        if simulator_config is not None and saved_config != _json_mapping(
            simulator_config
        ):
            raise ValueError(
                "Dataset simulator_config does not match the requested configuration"
            )
        data = {name: archive[name] for name in archive.files if name != "__metadata__"}
    if type(metadata["mask_prior_is_none"]) is not bool:
        raise ValueError("Dataset mask_prior_is_none must be Boolean")
    if metadata["mask_prior_is_none"]:
        if "mask_prior" in data:
            raise ValueError("Dataset contains an array for a None mask_prior")
        data["mask_prior"] = None
    count = _validate(data)
    if type(metadata["count"]) is not int or metadata["count"] != count:
        raise ValueError("Dataset sample count does not match metadata")
    return data, metadata
