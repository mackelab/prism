"""Dataset roundtrips and failures relevant to file-only training."""

import json

import jax
import numpy as np
import pytest
from omegaconf import OmegaConf

from prism.train.build_simulator import build_simulator
from prism.train.dataset import StaticDataset
from prism.train.offline_data import load_dataset, save_dataset


def _data():
    return {
        "x": np.ones((3, 4, 2), dtype=np.float32),
        "theta": np.ones((3, 3), dtype=np.float32),
        "model_mask": np.tile([True, False, True], (3, 1)),
        "mask_prior": np.full((3, 1), 0.5, dtype=np.float32),
    }


def _save(path, data=None):
    return save_dataset(
        path,
        _data() if data is None else data,
        simulator_config={"model_class": "example.Polynomial"},
        generation={"split": "train", "seed": 12},
    )


def _rewrite(path, mutate):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(arrays.pop("__metadata__").item())
    mutate(arrays, metadata)
    np.savez(path, **arrays, __metadata__=np.asarray(json.dumps(metadata)))


@pytest.mark.parametrize("fixed_prior", [False, True])
def test_real_symbolic_roundtrip_and_static_dataset(tmp_path, monkeypatch, fixed_prior):
    config = OmegaConf.create({
        "model_class": "prism.symbolic.simulators.SimplePolynomial",
        "num_points": "${.points}",
        "points": 8,
        "posterior_score": True,
    })
    if fixed_prior:
        config = OmegaConf.create({
            "model_class": "prism.symbolic.paper.model_specs.SymbolicModelV0",
            "observation_format": "signal_blocks",
            "x_min": 0.0,
            "x_max": 0.2,
            "x_step": 0.05,
            "input_dim": 2,
        })
    _, generators = build_simulator(config)
    data = jax.vmap(generators[0])(jax.random.split(jax.random.key(17), 2))
    path = tmp_path / "training.npz"
    metadata = save_dataset(
        path, data, simulator_config=config, generation={"seed": 17, "split": "train"}
    )

    def fail_if_called(*args, **kwargs):
        pytest.fail("Reading saved data must not construct or call a simulator")

    monkeypatch.setattr("prism.train.build_simulator.build_simulator", fail_if_called)
    restored, loaded_metadata = load_dataset(
        path, simulator_config=OmegaConf.to_container(config, resolve=True)
    )
    assert loaded_metadata == metadata
    assert metadata["count"] == 2
    assert metadata["mask_prior_is_none"] == fixed_prior
    assert jax.tree.structure(data) == jax.tree.structure(restored)
    for actual, expected in zip(
        jax.tree.leaves(restored), jax.tree.leaves(data), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    dataset = StaticDataset(restored, return_numpy=True)
    np.testing.assert_array_equal(dataset[1]["x"], data["x"][1])
    with np.load(path, allow_pickle=False) as archive:
        assert archive["__metadata__"].dtype.kind == "U"
        assert all(archive[name].dtype.kind != "O" for name in archive.files)


def test_prior_symbolic_metadata_remains_readable(tmp_path):
    path = tmp_path / "old-symbolic.npz"
    _save(path)
    _rewrite(path, lambda arrays, meta: meta.update(acquisition=None, arrays={}))
    restored, _ = load_dataset(path)
    np.testing.assert_array_equal(restored["theta"], _data()["theta"])


def test_config_mismatch_and_pickled_payload_are_rejected(tmp_path):
    path = tmp_path / "invalid.npz"
    _save(path)
    _rewrite(
        path, lambda arrays, _: arrays.update(x=np.array([object()], dtype=object))
    )
    with pytest.raises(ValueError, match="simulator_config does not match"):
        load_dataset(path, simulator_config={"model_class": "different.Model"})
    with pytest.raises(ValueError, match="Object arrays"):
        load_dataset(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"acq": np.ones(3)}, "Dataset fields"),
        ({"x": np.ones((2, 4, 2))}, "same nonempty sample"),
        ({"x": np.full((3, 4, 2), np.nan)}, "finite real numeric"),
        ({"theta": np.ones(3)}, "theta must have shape"),
        ({"model_mask": np.full((3, 3), 2)}, "binary values"),
        ({"mask_prior": np.ones(3)}, "mask_prior must have shape"),
        ({"target_score": np.ones((3, 1))}, "target_score shape"),
    ],
)
def test_bad_symbolic_data_is_rejected_before_writing(tmp_path, changes, message):
    path = tmp_path / "invalid.npz"
    with pytest.raises(ValueError, match=message):
        _save(path, {**_data(), **changes})
    assert not path.exists()


@pytest.mark.parametrize("change", ["missing_metadata", "wrong_count", "missing_theta"])
def test_incomplete_or_inconsistent_files_are_rejected(tmp_path, change):
    path = tmp_path / "invalid.npz"
    _save(path)
    if change == "missing_metadata":
        np.savez(path, **_data())
    elif change == "wrong_count":
        _rewrite(path, lambda arrays, meta: meta.update(count=4))
    else:
        _rewrite(path, lambda arrays, meta: arrays.pop("theta"))
    with pytest.raises(ValueError):
        load_dataset(path)


def test_existing_file_and_partial_write_cleanup(tmp_path, monkeypatch):
    path = tmp_path / "existing.npz"
    _save(path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        _save(path)
    assert path.read_bytes() == before

    def fail_save(stream, **arrays):
        stream.write(b"partial archive")
        raise OSError("disk write failed")

    monkeypatch.setattr("prism.train.offline_data.np.savez", fail_save)
    partial = tmp_path / "partial.npz"
    with pytest.raises(OSError, match="disk write failed"):
        _save(partial)
    assert not partial.exists()
    assert path.read_bytes() == before
