import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from prism.train.dataset import DataLoader, StaticDataset, instantiate_dataloader
from prism.train.train_script import create_dataloaders


def _sample_data(n: int = 4):
    return {
        "a": jnp.arange(n, dtype=jnp.int32),
        "b": jnp.ones((n, 2), dtype=jnp.float32),
    }


def test_static_dataset_len_and_indexing_behaviour():
    data = _sample_data(5)
    ds = StaticDataset(data, return_numpy=True)

    assert len(ds) == 5

    full = ds[:]
    assert isinstance(full["a"], np.ndarray)
    assert full["a"].shape == (5,)
    assert full["b"].shape == (5, 2)

    single = ds[2]
    assert np.asarray(single["a"]).shape == ()
    assert single["a"] == data["a"][2]

    wrapped = ds[[4, 5, -1]]
    np.testing.assert_array_equal(
        wrapped["a"], np.array([data["a"][4], data["a"][0], data["a"][4]])
    )


def test_static_dataset_respects_device_and_array_type():
    device = jax.devices()[0]
    data = _sample_data(3)
    ds = StaticDataset(data, device=device, return_numpy=False)

    batch = ds[:]
    assert isinstance(batch["a"], jax.Array)
    assert batch["a"].device == device
    assert batch["b"].device == device


def test_static_dataset_validates_input_shapes():
    bad_data = {"a": jnp.zeros((3,)), "b": jnp.zeros((2, 1))}
    with pytest.raises(ValueError):
        StaticDataset(bad_data)

    empty_data = {"a": jnp.zeros((0,)), "b": jnp.zeros((0, 1))}
    with pytest.raises(ValueError):
        StaticDataset(empty_data)

    with pytest.raises(ValueError):
        StaticDataset({})
    with pytest.raises(ValueError):
        StaticDataset({"a": jnp.zeros((3,)), "b": jnp.array(1.0)})


def test_create_dataloaders_supports_static_dataset():
    def simulator(rng):
        return {"x": jax.random.normal(rng, (2,)), "y": jax.random.uniform(rng, ())}

    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "dataloader": {
                "name": "StaticDataset",
                "dataset": {
                    "dataset_size": 4,
                    "device": {"kind": "cpu", "index": 0},
                    "return_numpy": False,
                },
                "train_loader": {
                    "batch_size": 2,
                    "shuffle": False,
                    "drop_last": False,
                },
                "val_loader": {
                    "batch_size": 2,
                    "shuffle": False,
                    "drop_last": False,
                },
            }
        },
    })

    loaders, val_loader = create_dataloaders(cfg, [simulator])
    train_loader = loaders[0]

    try:
        batch = next(iter(train_loader))
        assert batch["x"].shape == (2, 2)
        assert batch["y"].shape == (2,)
        val_batch = next(iter(val_loader))
        assert val_batch["x"].shape == (2, 2)
        # Training and validation simulations use independent seeds.
        assert not jnp.array_equal(batch["x"], val_batch["x"])
    finally:
        train_loader.close()
        val_loader.close()


def test_streaming_dataloaders_respect_configured_buffer_sizes():
    def simulator(rng):
        return {"x": jax.random.normal(rng, (2,))}

    cfg = OmegaConf.create({
        "seed": 0,
        "train": {
            "dataloader": {
                "name": "SimulationDataset",
                "dataset": {"simulation_batch_size": 2, "buffer_size": 6},
                "val_dataset": {"buffer_size": 4},
                "train_loader": {"batch_size": 2, "shuffle": False},
                "val_loader": {"batch_size": 2, "shuffle": False},
            }
        },
    })
    loaders, val_loader = create_dataloaders(cfg, [simulator])
    try:
        assert len(loaders[0]._ds) == 6
        assert len(val_loader._ds) == 4
        assert next(iter(loaders[0]))["x"].shape == (2, 2)
    finally:
        for loader in [*loaders, val_loader]:
            loader.close()
            loader._ds.close()


def test_loader_shutdown_does_not_block_on_full_prefetch_queue():
    loader = DataLoader(
        StaticDataset({"x": np.ones((4, 2))}),
        batch_size=2,
        num_prefetch_host=2,
        num_prefetch_device=1,
    )
    deadline = time.monotonic() + 5
    while not loader._q.full() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert loader._q.full()
    closing = threading.Thread(target=loader.close, daemon=True)
    closing.start()
    closing.join(timeout=3)
    assert not closing.is_alive()
    assert not loader._producer_th.is_alive()


def test_batch_larger_than_a_dropping_dataset_is_rejected():
    # Used to hang: with drop_last a 129-sample batch from 128 samples never fills.
    dataset = StaticDataset(_sample_data(4), return_numpy=True)
    with pytest.raises(ValueError, match="exceeds the 4 samples"):
        instantiate_dataloader(dataset, {"batch_size": 5, "drop_last": True})
    loader = instantiate_dataloader(dataset, {"batch_size": 5, "drop_last": False})
    loader.close()
    loader = instantiate_dataloader(dataset, {"batch_size": 4, "drop_last": True})
    loader.close()
