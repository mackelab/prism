import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

import prism.config  # noqa: F401  (prism.simulators.config needs it imported first)
from experiments.profiling import run_time
from experiments.profiling.timing import profile_function
from prism.simulators.config import resolve_simulator_model
from prism.train.build_model import build_model
from prism.train.build_simulator import build_simulator


def test_profile_function_reports_compile_first_call_and_steady_state():
    result = profile_function(
        lambda a, b: jnp.tanh(a @ b),
        (jnp.ones((64, 64)), jnp.ones((64, 64))),
        name="matmul",
        warmup=1,
        repeat=3,
    )
    steady = result["steady_state"]
    assert result["name"] == "matmul"
    assert result["compile_s"] > 0 and result["first_exec_s"] > 0
    assert steady["runs"] == 3
    assert steady["min_s"] <= steady["median_s"] <= steady["max_s"]
    assert result["input_nbytes"] == 2 * 64 * 64 * 4
    assert result["output_nbytes"] == 64 * 64 * 4


def test_profile_function_rejects_empty_repeat():
    with pytest.raises(ValueError):
        profile_function(lambda: jnp.zeros(()), name="x", repeat=0)


def _symbolic_config():
    fixture = Path(__file__).parent / "fixtures" / "symbolic_sbmi_nn.json"
    cfg = OmegaConf.create(json.loads(fixture.read_text())["config"])
    cfg.simulator.model_class = "prism.symbolic.paper.model_specs.SymbolicModelV15"
    cfg.simulator.input_dim = 10
    cfg.simulator.observation_format = "signal_blocks"
    cfg.simulator.x_min, cfg.simulator.x_max, cfg.simulator.x_step = 0.0, 10.0, 0.02
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
def symbolic_run():
    cfg = _symbolic_config()
    model = build_model(cfg, resolve_simulator_model(cfg))
    model.eval()
    _, generators = build_simulator(cfg)
    return model, generators


def test_every_operation_is_profiled_for_a_symbolic_model(symbolic_run):
    model, generators = symbolic_run
    cases = run_time.build_cases(
        model, generators[0], theta_num_steps=2, loss_batch_size=4
    )
    assert tuple(cases) == run_time.OPERATIONS
    for name, (fn, args) in cases.items():
        result = profile_function(fn, args, name=name, warmup=0, repeat=1)
        assert np.isfinite(result["steady_state"]["mean_s"])


def test_training_operations_can_be_skipped(symbolic_run):
    model, generators = symbolic_run
    cases = run_time.build_cases(model, generators[0], include_training=False)
    assert tuple(cases) == run_time.OPERATIONS[:4]


def test_parameter_counts_split_components(symbolic_run):
    from flax import nnx

    model, _ = symbolic_run
    _, params, _ = nnx.split(model, nnx.Param, ...)
    counts = run_time.parameter_counts(params)
    components = {k: v for k, v in counts.items() if k != "total"}
    assert set(components) <= set(run_time.COMPONENTS) and components
    assert 0 < sum(components.values()) <= counts["total"]


def test_main_refuses_to_overwrite(tmp_path):
    target = tmp_path / "result.json"
    target.write_text("{}")
    with pytest.raises(FileExistsError):
        run_time.main([str(tmp_path), "--output", str(target)])


def test_saved_run_is_profiled_with_its_ema_weights(tmp_path):
    from contextlib import closing

    from flax import nnx

    from prism.train.checkpointing import CheckpointManager

    cfg = _symbolic_config()
    model = build_model(cfg, resolve_simulator_model(cfg))
    params = nnx.state(model, nnx.Param)
    ema = jax.tree.map(lambda value: value + 0.5, params)
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    with closing(CheckpointManager(str(tmp_path / "checkpoints"))) as manager:
        manager.save(
            3,
            params,
            {"momentum": jnp.zeros(1)},
            loss=1.0,
            val_loss=0.5,
            params_ema=ema,
        )

    output = tmp_path / "result.json"
    result = run_time.main([
        str(tmp_path),
        "--skip-training",
        "--theta-num-steps",
        "2",
        "--repeat",
        "1",
        "--warmup",
        "0",
        "--output",
        str(output),
    ])

    assert result["step"] == 3 and result["which"] == "latest"
    assert tuple(result["operations"]) == run_time.OPERATIONS[:4]
    assert (
        json.loads(output.read_text())["parameter_counts"] == result["parameter_counts"]
    )
    loaded, _, _ = run_time.load_run(tmp_path, "latest")
    for actual, saved in zip(
        jax.tree.leaves(nnx.state(loaded, nnx.Param)), jax.tree.leaves(ema), strict=True
    ):
        np.testing.assert_array_equal(actual, saved)
