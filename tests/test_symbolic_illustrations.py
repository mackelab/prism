import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import prism.config  # noqa: F401  (prism.simulators.config needs it imported first)
from experiments.symbolic.illustrations import illustrate, plot
from prism.simulators.config import resolve_simulator_model
from prism.train.build_model import build_model
from prism.train.build_simulator import build_simulator

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _config():
    fixture = FIXTURES / "symbolic_sbmi_nn.json"
    cfg = OmegaConf.create(json.loads(fixture.read_text())["config"])
    cfg.simulator.model_class = "prism.symbolic.paper.model_specs.SymbolicModelV15"
    cfg.simulator.input_dim = 1
    cfg.simulator.observation_format = "signal_blocks"
    cfg.simulator.x_min, cfg.simulator.x_max, cfg.simulator.x_step = 0.0, 10.0, 1.0
    cfg.simulator.mask_prior = {"alpha": 1.0, "beta": 1.0, "min_active_models": 1}
    cfg.model.name = (
        "prism.symbolic.paper.model.SymbolicPaperModelConfigMaskPriorAmortized"
    )
    cfg.model.model_selection_net.name = "prism.symbolic.paper.autoregressive.SymbolicPaperModelSelectionAmortizedPriorConfig"
    cfg.model.model_selection_net.params.prior_params_embed_dim = 8
    cfg.model.model_selection_net.params.mask_prior_dim = 1
    cfg.model.model_selection_net.params.precondition_with_mask_prior = True
    return cfg


@pytest.fixture(scope="module")
def symbolic():
    cfg = _config()
    sim_type = resolve_simulator_model(cfg)
    model = build_model(cfg, sim_type)
    model.eval()
    return model, build_simulator(cfg)[1][0], sim_type


@pytest.fixture(scope="module")
def result(symbolic):
    model, generator, sim_type = symbolic
    return illustrate.illustrate(
        model,
        generator,
        sim_type,
        np.arange(0.0, 10.0, 1.0),
        edit_observation=True,
        num_samples=6,
        num_posterior=8,
    )


def test_observation_is_seeded_and_edit_is_applied(symbolic):
    _, generator, _ = symbolic
    plain = illustrate.simulate_observation(generator)
    again = illustrate.simulate_observation(generator)
    edited = illustrate.simulate_observation(generator, edit=True)
    np.testing.assert_array_equal(plain["x"], again["x"])
    np.testing.assert_allclose(
        np.ravel(edited["x"])[:3], [7.1, 13.567, 9.0002313], rtol=1e-6
    )
    np.testing.assert_array_equal(np.ravel(edited["x"])[3:], np.ravel(plain["x"])[3:])


def test_outputs_have_the_documented_shapes(result, symbolic):
    arrays, summary = result
    _, _, sim_type = symbolic
    signal = len(sim_type.model_types)
    assert arrays["model_space"].shape == (100, sim_type.num_compartments())
    assert arrays["component_quantiles"].shape == (3, 100)
    assert (arrays["component_quantiles"][1] <= arrays["component_quantiles"][2]).all()
    assert arrays["grid_eval"].shape == arrays["true_function"].shape == (1000,)
    for lam in (0.5, 0.1):
        assert arrays[f"predictive_lo_{lam}"].shape == (1000,)
        assert (
            arrays[f"predictive_lo_{lam}"] <= arrays[f"predictive_hi_{lam}"] + 1e-6
        ).all()
        assert 1 <= summary["selected"][str(lam)]["components"] <= signal
    assert arrays["parameter_posterior"].shape[0] == 8
    assert summary["true_components"] == int(arrays["true_components"]) <= signal


def test_illustration_is_reproducible(result, symbolic):
    model, generator, sim_type = symbolic
    again, _ = illustrate.illustrate(
        model,
        generator,
        sim_type,
        np.arange(0.0, 10.0, 1.0),
        edit_observation=True,
        num_samples=6,
        num_posterior=8,
    )
    first, _ = result
    for key in ("model_space", "selected_function_0.1", "parameter_posterior"):
        np.testing.assert_allclose(first[key], again[key], rtol=1e-5, atol=1e-5)


def test_panels_and_plot_cli(tmp_path, result):
    arrays, summary = result
    for figure in (
        plot.plot_model_fits(arrays),
        plot.plot_num_components(arrays),
        plot.plot_model_space(arrays),
        plot.plot_parameter_posterior(arrays["parameter_posterior"]),
    ):
        assert figure.axes
    source = tmp_path / "illustrations"
    source.mkdir()
    np.savez(source / "arrays.npz", **arrays)
    (source / "summary.json").write_text(json.dumps(summary))
    out = tmp_path / "panels"
    plot.main([str(source), "--output-dir", str(out)])
    assert {p.name for p in out.iterdir()} == {
        "model_fits.svg",
        "num_components.svg",
        "model_space.svg",
        "parameter_posterior.svg",
        "equations.txt",
    }
    with pytest.raises(FileExistsError):
        plot.main([str(source), "--output-dir", str(out)])


def test_main_refuses_to_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        illustrate.main([str(tmp_path), "--output-dir", str(tmp_path)])


def test_illustrative_preset_matches_the_source_run():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(
            "train.yaml", overrides=["+experiment/train=symbolic50_illustrative"]
        )
    assert (cfg.simulator.x_step, cfg.simulator.input_dim) == (1.0, 1)
    assert cfg.run.seed == 1 and cfg.training.dataloader.train_loader.batch_size == 8192
    standard = compose_standard()
    assert (
        standard.simulator.x_step == 0.02
        and standard.training.dataloader.train_loader.batch_size == 4096
    )


def compose_standard():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return compose("train.yaml", overrides=["+experiment/train=symbolic50"])


def test_given_indices_seed_and_posterior_index(symbolic):
    model, generator, sim_type = symbolic
    grid = np.arange(0.0, 10.0, 1.0)
    kwargs = dict(num_samples=6, num_posterior=8, seed=147, edit_observation=False)
    arrays, summary = illustrate.illustrate(
        model,
        generator,
        sim_type,
        grid,
        display_index={0.5: 2, 0.1: 1},
        posterior_index=3,
        **kwargs,
    )
    assert summary["observation_seed"] == 147 and summary["posterior_index"] == 3
    assert (
        summary["selected"]["0.5"]["index"] == 2
        and summary["selected"]["0.5"]["index_given"]
    )
    assert summary["selected"]["0.1"]["index"] == 1
    other, _ = illustrate.illustrate(model, generator, sim_type, grid, **kwargs)
    assert not np.array_equal(
        arrays["x_observed"],
        np.ravel(illustrate.simulate_observation(generator)["x"]),
    )
    assert other["parameter_posterior"].shape[0] == 8


def test_load_pretrained_reads_published_layout(tmp_path, symbolic):
    from prism.symbolic.pretrained import save_params_safetensors

    model, _, _ = symbolic
    cfg = _config()
    (tmp_path / "latest").mkdir()
    save_params_safetensors(
        model, tmp_path / "latest" / "params_ema.safetensors", {"step": "7"}
    )
    OmegaConf.save(
        OmegaConf.create({
            k: OmegaConf.to_container(cfg[k], resolve=True)
            for k in ("model", "simulator")
        }),
        tmp_path / "config.yaml",
    )
    loaded, generator, sim_type, grid, step = illustrate.load_pretrained(
        "tiny", "latest", tmp_path
    )
    assert step == 7 and grid.shape == (10,)
    x = illustrate.simulate_observation(generator)["x"]
    key = jax.random.PRNGKey(0)
    np.testing.assert_allclose(
        model.sample_mask(key, x=x, mask_prior=0.5),
        loaded.sample_mask(key, x=x, mask_prior=0.5),
    )


def test_observation_edit_is_refused_for_multivalued_observations():
    def generator(key, mask_prior_hyperparameter):
        return {"x": jnp.zeros((50, 10))}

    with pytest.raises(ValueError, match="input_dim 1"):
        illustrate.simulate_observation(generator, edit=True)
    data = illustrate.simulate_observation(
        lambda key, **_: {"x": jnp.zeros((50, 1))}, edit=True
    )
    assert float(data["x"][1, 0]) == pytest.approx(13.567)


def test_display_indices_are_validated(symbolic):
    model, generator, sim_type = symbolic
    grid = np.arange(0.0, 10.0, 1.0)
    common = dict(num_samples=6, num_posterior=4, grid_obs=grid, grid_eval=grid)
    with pytest.raises(ValueError, match="lambda"):
        illustrate.illustrate(
            model, generator, sim_type, display_index={0.9: 1}, **common
        )
    with pytest.raises(ValueError, match="outside"):
        illustrate.illustrate(
            model, generator, sim_type, display_index={0.5: 99}, **common
        )
    with pytest.raises(ValueError, match="outside"):
        illustrate.illustrate(model, generator, sim_type, posterior_index=-1, **common)
