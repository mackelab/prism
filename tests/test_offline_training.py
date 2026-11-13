"""Saved simulations feed the shared optimizer, EMA and checkpoint loop on CPU."""

import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from prism.config import runtime_config
from prism.nn.model import PrismModel
from prism.symbolic.simulators import SimplePolynomial
from prism.train import simulate, train_script, utils
from prism.train.build_simulator import build_simulator
from prism.train.dataset import FileDataset


@pytest.fixture
def source_config():
    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose("train", overrides=["+experiment/train=symbolic"])
    OmegaConf.set_struct(cfg, False)
    cfg.run.name = "offline_contract"
    cfg.simulator.update({
        "observation_format": "signal_blocks",
        "input_dim": 2,
        "x_step": 0.5,
    })
    cfg.training.dataloader.dataset.update({
        "dataset_size": 6,
        "simulation_batch_size": 3,
    })
    cfg.training.dataloader.val_dataset.dataset_size = 4
    for split in ("train_loader", "val_loader"):
        cfg.training.dataloader[split].update({
            "batch_size": 2,
            "num_prefetch_host": 1,
            "num_prefetch_device": 1,
            "loop": True,
        })
    return cfg


@pytest.fixture
def generated(tmp_path, source_config):
    directory = tmp_path / "dataset"
    training = simulate.generate_datasets(
        source_config,
        directory,
        train_size=6,
        validation_size=4,
        simulation_batch_size=3,
    )
    return directory, training


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if hasattr(left, "dtype") and hasattr(right, "dtype"):
            assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)


def fail_simulation(*args, **kwargs):
    raise AssertionError("File-only training invoked the simulator")


@pytest.mark.parametrize("dataset_seed", [None, 23])
def test_generation_matches_existing_static_train_validation(
    tmp_path, source_config, dataset_seed
):
    if dataset_seed is not None:
        source_config.training.dataloader.dataset.seed = dataset_seed
        source_config.training.dataloader.dataset.rng = None
        source_config.training.dataloader.val_dataset.seed = dataset_seed + 1
    _, generators = build_simulator(source_config)
    directory = tmp_path / "parity"
    training = simulate.generate_datasets(
        source_config,
        directory,
        train_size=6,
        validation_size=4,
        simulation_batch_size=3,
    )
    online, validation = train_script.create_dataloaders(
        runtime_config(source_config, "train"), generators
    )
    file_train, file_validation = train_script.create_dataloaders(
        runtime_config(OmegaConf.load(directory / "training.yaml"), "train"), []
    )
    try:
        assert len(online) == len(file_train) == 1
        for expected, actual in zip(
            [*online, validation],
            [*file_train, file_validation],
            strict=True,
        ):
            assert_tree_equal(actual._ds[:], expected._ds[:])
        assert training == OmegaConf.load(directory / "training.yaml")
        assert {path.name for path in directory.iterdir()} == {
            "train.npz",
            "validation.npz",
            "simulation.yaml",
            "training.yaml",
        }
    finally:
        for loader in [*online, validation, *file_train, file_validation]:
            loader.close()


def test_generated_config_reads_separate_files_without_simulator(
    generated, monkeypatch
):
    directory, _ = generated
    monkeypatch.setattr(train_script, "build_simulator", fail_simulation)
    monkeypatch.setattr(SimplePolynomial, "from_theta", fail_simulation)
    cfg = OmegaConf.load(directory / "training.yaml")
    assert cfg.training.eval.ksd.iters == 0
    training, validation = train_script.create_dataloaders(
        runtime_config(cfg, "train"), []
    )
    try:
        assert len(training) == 1
        assert training[0]._ds.path != validation._ds.path
        assert len(training[0]._ds) == 6
        assert len(validation._ds) == 4
        assert training[0]._ds.path == directory / "train.npz"
        assert validation._ds.path == directory / "validation.npz"
        assert cfg.training.dataloader.train_dataset.sha256 == training[0]._ds.sha256
        assert cfg.training.dataloader.val_dataset.sha256 == validation._ds.sha256
        assert not np.array_equal(
            training[0]._ds[:4]["theta"], validation._ds[:]["theta"]
        )
        assert next(iter(training[0]))["theta"].shape == (2, 3)
        assert next(iter(validation))["theta"].shape == (2, 3)
    finally:
        for loader in [*training, validation]:
            loader.close()


def test_generation_rejects_non_symbolic_models(tmp_path, source_config, monkeypatch):
    source_config.simulator = {
        "model_class": "prism.dmri.simulators.models.Ball3StickNoise"
    }
    monkeypatch.setattr(simulate, "build_simulator", fail_simulation)
    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="supports_offline_data"):
        simulate.generate_datasets(
            source_config, output, train_size=6, validation_size=4
        )
    assert not output.exists()


@pytest.mark.parametrize(("as_list", "batch_size"), [(False, 2), (True, 2), (False, 8)])
def test_file_loaders_keep_every_sample(tmp_path, source_config, as_list, batch_size):
    source_config.training.dataloader.val_loader.batch_size = None
    for name in ("train_loader", "val_loader"):
        source_config.training.dataloader[name].shuffle = False
        source_config.training.dataloader[name].drop_last = True
        if as_list:
            source_config.training.dataloader[name] = [
                source_config.training.dataloader[name]
            ]
    directory = tmp_path / "data"
    simulate.generate_datasets(
        source_config,
        directory,
        train_size=5,
        validation_size=3,
        simulation_batch_size=2,
    )
    training = OmegaConf.load(directory / "training.yaml")
    for name in ("train_loader", "val_loader"):
        original = source_config.training.dataloader[name]
        settings = training.training.dataloader[name]
        if as_list:
            original, settings = original[0], settings[0]
        assert original.drop_last is True
        assert settings.drop_last is False
        if name == "train_loader":
            # A training-time override must also work when it exceeds the dataset.
            settings.batch_size = batch_size

    train_loaders, validation = train_script.create_dataloaders(
        runtime_config(training, "train"), []
    )
    try:
        for loader in [*train_loaders, validation]:
            size = len(loader._ds)
            expected_sizes = [
                min(batch_size, size - start) for start in range(0, size, batch_size)
            ]
            batches = iter(loader)
            for _ in range(2):
                epoch = [next(batches)["theta"] for _ in expected_sizes]
                assert [len(batch) for batch in epoch] == expected_sizes
                np.testing.assert_array_equal(
                    np.concatenate(epoch), loader._ds[:]["theta"]
                )
    finally:
        for loader in [*train_loaders, validation]:
            loader.close()


@pytest.mark.parametrize("invalid", ["checksum", "simulator", "validation_split"])
def test_file_training_rejects_invalid_dataset(generated, invalid):
    directory, _ = generated
    cfg = OmegaConf.load(directory / "training.yaml")
    if invalid == "checksum":
        with (directory / "train.npz").open("ab") as stream:
            stream.write(b"changed after generation")
        message = "checksum mismatch"
    elif invalid == "simulator":
        cfg.simulator.mask_prior.alpha = 2.0
        message = "[Ss]imulator.*mismatch|[Ss]imulator.*match"
    else:
        cfg.training.dataloader.val_dataset = cfg.training.dataloader.train_dataset
        message = "validation|different dataset files"
    with pytest.raises(ValueError, match=message):
        train_script.create_dataloaders(runtime_config(cfg, "train"), [])


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("change", ["size", "seed"])
def test_first_run_rejects_wrong_dataset_directory(
    generated, tmp_path, source_config, split, change
):
    directory, _ = generated
    if change == "seed":
        source_config.run.seed += 1
    other = tmp_path / "other_dataset"
    simulate.generate_datasets(
        source_config,
        other,
        train_size=3 if change == "size" and split == "train" else 6,
        validation_size=2 if change == "size" and split == "validation" else 4,
        simulation_batch_size=3,
    )
    # Replace only one split so both generated hashes are independently checked.
    unchanged = "validation" if split == "train" else "train"
    shutil.copyfile(directory / f"{unchanged}.npz", other / f"{unchanged}.npz")
    cfg = OmegaConf.load(directory / "training.yaml")
    cfg.training.dataloader.dataset.directory = str(other)
    with pytest.raises(ValueError, match=rf"checksum mismatch: .*{split}\.npz"):
        train_script.create_dataloaders(runtime_config(cfg, "train"), [])


def test_generated_dataset_directory_can_be_relocated(generated, tmp_path):
    directory, _ = generated
    moved = tmp_path / "on_cluster"
    shutil.move(directory, moved)
    cfg = OmegaConf.load(moved / "training.yaml")
    cfg.training.dataloader.dataset.directory = str(moved)
    training, validation = train_script.create_dataloaders(
        runtime_config(cfg, "train"), []
    )
    try:
        assert training[0]._ds.path.parent == moved
        assert validation._ds.path.parent == moved
    finally:
        for loader in [*training, validation]:
            loader.close()


def test_file_training_rejects_simulator_ksd_before_startup(generated, monkeypatch):
    directory, _ = generated
    cfg = OmegaConf.load(directory / "training.yaml")
    cfg.training.eval.ksd.iters = 1
    monkeypatch.setattr(train_script, "configure_environment", fail_simulation)
    with pytest.raises(ValueError, match="KSD uses the simulator likelihood"):
        train_script._main.__wrapped__(cfg)


@pytest.fixture
def recorded_loaders(tmp_path):
    def loader(name, checksum):
        return SimpleNamespace(
            _ds=SimpleNamespace(
                path=tmp_path / name,
                sha256=checksum * 64,
            )
        )

    return [loader("train.npz", "a")], loader("validation.npz", "b")


def test_resume_requires_original_dataset_provenance(tmp_path, recorded_loaders):
    training, validation = recorded_loaders
    with pytest.raises(ValueError, match="datasets.json is missing"):
        train_script.record_offline_datasets(
            training, validation, tmp_path, resume=True
        )
    assert not (tmp_path / "datasets.json").exists()


@pytest.mark.parametrize("change", ["train_hash", "validation_hash"])
def test_resume_rejects_changed_corpus_without_overwriting_provenance(
    tmp_path, recorded_loaders, change
):
    training, validation = recorded_loaders
    train_script.record_offline_datasets(training, validation, tmp_path, resume=False)
    path = tmp_path / "datasets.json"
    original = path.read_bytes()
    if change == "train_hash":
        training[0]._ds.sha256 = "d" * 64
    else:
        validation._ds.sha256 = "d" * 64
    with pytest.raises(ValueError, match="Dataset checksum or stream order changed"):
        train_script.record_offline_datasets(
            training, validation, tmp_path, resume=True
        )
    assert path.read_bytes() == original


def test_resume_accepts_relocated_files_with_identical_hashes(
    tmp_path, recorded_loaders
):
    training, validation = recorded_loaders
    train_script.record_offline_datasets(training, validation, tmp_path, resume=False)
    path = tmp_path / "datasets.json"
    previous = json.loads(path.read_text())
    for loader in [*training, validation]:
        loader._ds.path = tmp_path / "relocated" / loader._ds.path.name
    train_script.record_offline_datasets(training, validation, tmp_path, resume=True)
    current = json.loads(path.read_text())
    for old, new in zip(
        [*previous["train"], previous["validation"]],
        [*current["train"], current["validation"]],
        strict=True,
    ):
        assert old["sha256"] == new["sha256"]
        assert Path(new["path"]).parent == tmp_path / "relocated"


class TinyInferenceModel(nnx.Module):
    """Small differentiable model to exercise real training/checkpoint machinery."""

    def __init__(self, simulator):
        self.cfg = "offline checkpoint test"
        self.theta = nnx.Param(jnp.zeros(simulator.theta_dim))
        self.mask_logits = nnx.Param(jnp.zeros(simulator.num_compartments()))
        self.examples_seen = nnx.Variable(jnp.asarray(0, dtype=jnp.int32))

    def loss_fn(self, rng, theta, model_mask, **kwargs):
        self.examples_seen[...] += theta.shape[0]
        return (
            optax.sigmoid_binary_cross_entropy(
                self.mask_logits[...], model_mask
            ).mean(),
            jnp.mean((theta - self.theta[...]) ** 2),
        )

    def log_prob_mask(self, model_mask, acq, x, mask_prior):
        return -optax.sigmoid_binary_cross_entropy(
            self.mask_logits[...], model_mask
        ).sum()

    def log_prob_theta(self, theta, acq, x, model_mask, **kwargs):
        return -jnp.sum((theta - self.theta[...]) ** 2)

    def sample_mask(self, rng, acq, x, mask_prior):
        return jax.random.bernoulli(rng, jax.nn.sigmoid(self.mask_logits[...]))

    def sample_theta(self, rng, acq=None, x=None, model_mask=None, **kwargs):
        return self.theta[...] + jax.random.normal(rng, self.theta.shape)


def test_offline_train_ema_checkpoint_resume_and_inference(
    generated, tmp_path, monkeypatch
):
    directory, _ = generated
    cfg = OmegaConf.load(directory / "training.yaml")
    # Six saved samples exercise both full and partial optimizer batches.
    cfg.training.dataloader.train_loader.batch_size = 4
    cfg.training.update({
        "inner_steps": 1,
        "max_train_steps": 2,
        "checkpoint_freq": 1,
        "eval_freq": 1,
        "track_ema": True,
        "ema_decay": 0.8,
        "recovery_threshold": None,
        "optimizer": {"optimizer": "adam", "scheduler": None, "learning_rate": 0.01},
    })
    output = tmp_path / "run"
    output.mkdir()
    attempt = output / "first_attempt"
    attempt.mkdir()
    OmegaConf.save(cfg, output / "config.yaml")
    monkeypatch.setattr(
        train_script,
        "configure_environment",
        lambda config: (logging.getLogger(__name__), str(attempt), str(output)),
    )
    monkeypatch.setattr(train_script, "build_simulator", fail_simulation)
    monkeypatch.setattr(simulate, "build_simulator", fail_simulation)
    monkeypatch.setattr(
        train_script, "build_model", lambda cfg, sim: TinyInferenceModel(sim)
    )
    monkeypatch.setattr(utils, "build_model", lambda cfg, sim: TinyInferenceModel(sim))

    states = []
    real_loop = train_script.train_loop

    def run_and_record(**kwargs):
        initial = kwargs["train_state"]
        initial_values = (
            initial.step,
            initial.params,
            initial.opt_state,
            initial.ema_state,
            jax.random.key_data(initial.rng),
            initial.model_state,
        )
        result = real_loop(**kwargs)
        states.append((initial_values, result))
        return result

    monkeypatch.setattr(train_script, "train_loop", run_and_record)
    train_script._main.__wrapped__(cfg)
    assert states[0][1].step == 2
    assert (output / "checkpoints" / "2").is_dir()
    assert (attempt / "artifact.yaml").is_file()
    assert not (attempt / "datasets.json").exists()
    provenance = json.loads((output / "datasets.json").read_text())
    assert provenance["train"] == [
        {
            "path": str(directory / "train.npz"),
            "sha256": FileDataset(directory / "train.npz").sha256,
        }
    ]
    assert provenance["validation"] == {
        "path": str(directory / "validation.npz"),
        "sha256": FileDataset(directory / "validation.npz").sha256,
    }

    cfg.training.continue_training = True
    cfg.training.max_train_steps = 3
    attempt = output / "second_attempt"
    attempt.mkdir()
    train_script._main.__wrapped__(cfg)
    assert not (attempt / "datasets.json").exists()
    restored, final = states[1]
    previous = states[0][1]
    assert restored[0] == 2
    assert_tree_equal(restored[1], previous.params)
    assert_tree_equal(restored[2], previous.opt_state)
    assert_tree_equal(restored[3], previous.ema_state)
    np.testing.assert_array_equal(restored[4], jax.random.key_data(previous.rng))
    assert_tree_equal(restored[5], previous.model_state)
    assert final.step == 3
    assert (output / "checkpoints" / "3").is_dir()
    assert any(np.any(value != 0) for value in jax.tree.leaves(final.params))

    checkpoint, model, simulator = utils.load_checkpoint(output, which="latest")
    assert checkpoint["step"] == 3
    assert simulator is SimplePolynomial
    expected_ema = train_script.get_ema_params(final.ema_state)
    assert_tree_equal(checkpoint["params_ema"], expected_ema)
    nnx.update(model, checkpoint["params_ema"])
    assert_tree_equal(nnx.state(model, nnx.Param), expected_ema)
    observation = FileDataset(directory / "validation.npz")[0]
    theta = model.sample_theta(
        jax.random.key(5), x=observation["x"], model_mask=observation["model_mask"]
    )
    assert theta.shape == (3,)
    assert np.isfinite(theta).all()


def test_saved_symbolic_data_trains_real_network_and_restores_ema(
    generated, tmp_path, monkeypatch
):
    directory, cfg = generated
    cfg.model.model_dim = 8
    for name in ("embedding_net", "model_selection_net", "inference_net"):
        params = cfg.model[name].params
        params.update({
            "num_layers": 1,
            "num_heads": 1,
            "attn_size": 4,
            "widening_factor": 1,
            "use_flash_attention": False,
        })
        if "use_flash_cross_attention" in params:
            params.use_flash_cross_attention = False
    cfg.model.model_selection_net.params.prior_params_embed_dim = 8
    cfg.model.inference_net.params.time_embed_dim = 8
    cfg.training.update({
        "inner_steps": 1,
        "max_train_steps": 1,
        "checkpoint_freq": 1,
        "eval_freq": 1,
        "track_ema": True,
        "ema_decay": 0.8,
        "recovery_threshold": None,
        "optimizer": {"optimizer": "adam", "scheduler": None, "learning_rate": 0.001},
    })
    cfg.training.eval.nnl_mask.iters = 1
    cfg.training.eval.nnl_theta.iters = 0
    # Training reads saved observations; forward evaluations are only used
    # for posterior predictive metrics, without building data generators.
    monkeypatch.setattr(simulate, "build_simulator", fail_simulation)
    monkeypatch.setattr(train_script, "build_simulator", fail_simulation)

    output = tmp_path / "trained"
    output.mkdir()
    OmegaConf.save(cfg, output / "config.yaml")
    monkeypatch.setattr(
        train_script,
        "configure_environment",
        lambda config: (logging.getLogger(__name__), str(output), str(output)),
    )
    states = []
    real_loop = train_script.train_loop

    def run_and_record(**kwargs):
        state = kwargs["train_state"]
        before = (
            state.step,
            state.params,
            state.opt_state,
            state.ema_state,
            state.model_state,
            jax.random.key_data(state.rng),
        )
        result = real_loop(**kwargs)
        states.append((before, result))
        return result

    monkeypatch.setattr(train_script, "train_loop", run_and_record)
    train_script._main.__wrapped__(cfg)
    assert states[0][1].step == 1
    artifact = OmegaConf.load(output / "artifact.yaml")
    assert artifact.simulator.model_class == cfg.simulator.model_class

    cfg.training.continue_training = True
    cfg.training.max_train_steps = 2
    train_script._main.__wrapped__(cfg)
    restored, final = states[1]
    previous = states[0][1]
    assert restored[0] == 1
    for actual, expected in zip(
        restored[1:5],
        (previous.params, previous.opt_state, previous.ema_state, previous.model_state),
        strict=True,
    ):
        assert_tree_equal(actual, expected)
    np.testing.assert_array_equal(restored[5], jax.random.key_data(previous.rng))
    assert final.step == 2
    assert all(np.isfinite(value).all() for value in jax.tree.leaves(final.params))

    checkpoint, model, simulator = utils.load_checkpoint(output, which="latest")
    assert isinstance(model, PrismModel)
    assert checkpoint["step"] == 2
    assert simulator is SimplePolynomial
    expected_ema = train_script.get_ema_params(final.ema_state)
    assert_tree_equal(checkpoint["params_ema"], expected_ema)
    nnx.update(model, checkpoint["params_ema"])
    model.eval()
    assert_tree_equal(nnx.state(model, nnx.Param), expected_ema)
    observation = FileDataset(directory / "validation.npz")[0]

    @nnx.jit
    def infer(model, data):
        mask = model.sample_mask(
            jax.random.key(2), x=data["x"], mask_prior=data["mask_prior"]
        )
        theta = model.sample_theta(
            jax.random.key(3), x=data["x"], model_mask=mask, num_steps=2
        )
        log_prob = model.log_prob_mask(mask, x=data["x"], mask_prior=data["mask_prior"])
        return mask, theta, log_prob

    mask, theta, log_prob = infer(model, observation)
    assert mask.shape == theta.shape == (3,)
    assert np.isin(mask, [False, True]).all()
    assert np.isfinite(theta).all() and np.isfinite(log_prob)
    assert all(device.platform == "cpu" for device in theta.devices())
