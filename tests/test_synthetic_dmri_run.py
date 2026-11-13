"""Regression checks for dMRI evaluation summaries, validation, and plotting."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.dmri._shared import storage as file_storage


@pytest.fixture(scope="module")
def runner():
    from experiments.dmri.synthetic_evaluation import run

    return run


@pytest.fixture
def capture_plot_values(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    recorded = {"violin": [], "scatter": []}
    monkeypatch.setattr(
        Axes,
        "violinplot",
        lambda self, data, **kwargs: recorded["violin"].extend(
            np.asarray(v) for v in data
        ),
    )
    monkeypatch.setattr(
        Axes,
        "scatter",
        lambda self, x, y, **kwargs: recorded["scatter"].append(np.asarray(y)),
    )
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    yield recorded
    plt.close("all")


def test_rksd_uses_mean_rooted_prior_for_each_exact_mask(runner):
    # A and B share the same compartment count; A and C differ only in noise.
    a, b, c = [1, 1, 0, 0, 1], [1, 0, 1, 0, 1], [1, 1, 0, 0, 0]
    arrays = {
        "model_mask": np.array([a, a, b, b, c, c], dtype=bool),
        "posterior_ksd": np.array([2.0, 10.0, 4.0, 8.0, 1.0, 3.0]),
        "prior_ksd": np.array([1.0, 3.0, 10.0, 30.0, 2.0, 6.0]),
    }
    summary = runner.summarize(arrays)
    rows = {tuple(row["mask"]): row for row in summary["per_mask"]}
    assert len(rows) == 3
    for mask, denominator, mean in ((a, 2.0, 3.0), (b, 20.0, 0.3), (c, 4.0, 0.5)):
        row = rows[tuple(mask)]
        assert row["count"] == 2
        assert row["prior_ksd_mean"] == denominator
        assert row["rksd_mean"] == pytest.approx(mean)
    assert rows[tuple(a)]["posterior_ksd_mean"] == 6.0
    assert rows[tuple(a)]["rksd_fraction_below_one"] == 0.0


def test_summary_retains_outlier_but_violin_uses_trimmed_values(
    runner, tmp_path, capture_plot_values
):
    rooted_ksd = np.r_[np.arange(100, dtype=float), 10000.0]
    arrays = {
        "model_mask": np.tile([1, 1, 0, 0, 1], (len(rooted_ksd), 1)),
        "posterior_ksd": rooted_ksd,
        "prior_ksd": np.ones(len(rooted_ksd)),
    }
    before = rooted_ksd.copy()
    summary = runner.summarize(arrays)
    row = summary["per_mask"][0]
    assert row["rksd_mean"] == rooted_ksd.mean()
    assert row["rksd_mean"] > 100
    assert row["count"] == 101
    assert row["plot_trim_cutoff"] == 99.0
    assert row["posterior_ksd_zero_count"] == 1
    runner.plots(arrays, summary, tmp_path, "fixture")
    assert len(capture_plot_values["violin"]) == 1
    np.testing.assert_array_equal(capture_plot_values["violin"][0], np.arange(99))
    np.testing.assert_array_equal(arrays["posterior_ksd"], before)
    assert row["rksd_mean"] == rooted_ksd.mean()


def test_constant_trimmed_values_do_not_restore_outlier_in_scatter(
    runner, tmp_path, capture_plot_values
):
    arrays = {
        "model_mask": np.tile([1, 1, 0, 0, 1], (3, 1)),
        "posterior_ksd": np.array([1.0, 1.0, 100.0]),
        "prior_ksd": np.ones(3),
    }
    summary = runner.summarize(arrays)
    runner.plots(arrays, summary, tmp_path, "fixture")
    assert not capture_plot_values["violin"]
    assert len(capture_plot_values["scatter"]) == 1
    np.testing.assert_array_equal(capture_plot_values["scatter"][0], [1.0, 1.0])
    assert summary["per_mask"][0]["rksd_mean"] == 34.0


@pytest.fixture
def sbc_arrays():
    return {
        "model_mask": np.array([[1, 1, 0, 0, 1], [1, 0, 0, 0, 1]], dtype=bool),
        "theta_true": np.zeros((2, 3)),
        "theta_mask": np.array([[1, 0, 1], [0, 0, 1]], dtype=bool),
        "theta_pit": np.array([[0.1, np.nan, 0.9], [np.nan, np.nan, 0.5]]),
        "model_pit": np.array([0.25, 0.75]),
        "theta_sum_pit": np.array([0.2, 0.8]),
    }


def test_inactive_theta_nans_are_excluded_and_summary_is_json_safe(
    runner, sbc_arrays, tmp_path
):
    runner.validate_batch(sbc_arrays)
    summary = runner.summarize(sbc_arrays)
    assert summary["theta_pit"]["count"] == 3
    assert summary["theta_pit"]["cdf"][50] == pytest.approx(2 / 3)
    assert sorted(row["theta_pit"]["count"] for row in summary["per_mask"]) == [1, 2]
    target = tmp_path / "summary.json"
    runner.write_json(target, summary)
    assert json.loads(target.read_text()) == summary
    assert "NaN" not in target.read_text()


def test_all_inactive_coordinates_have_no_coverage_plot(runner, tmp_path):
    arrays = {
        "model_mask": np.array([[1, 0, 0, 0, 1]], dtype=bool),
        "theta_mask": np.zeros((1, 3), dtype=bool),
        "theta_pit": np.full((1, 3), np.nan),
    }
    runner.validate_batch(arrays)
    summary = runner.summarize(arrays)
    assert summary["theta_pit"] is None
    assert summary["per_mask"][0]["theta_pit"] is None
    assert runner.plots(arrays, summary, tmp_path, "inactive") == []


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_active_theta_is_rejected(runner, sbc_arrays, value):
    sbc_arrays["theta_pit"][0, 0] = value
    with pytest.raises(FloatingPointError, match="active parameter PIT"):
        runner.validate_batch(sbc_arrays)


def test_other_nonfinite_metrics_are_rejected(runner, sbc_arrays):
    sbc_arrays["model_pit"][0] = np.inf
    with pytest.raises(FloatingPointError, match="Non-finite model_pit"):
        runner.validate_batch(sbc_arrays)


@pytest.mark.parametrize("prior", [[0.0, 0.0], [np.inf, 1.0], [np.nan, 1.0]])
def test_invalid_mask_group_prior_denominator_fails_clearly(runner, prior):
    arrays = {
        "model_mask": np.tile([1, 0, 0, 0, 1], (2, 1)),
        "posterior_ksd": np.ones(2),
        "prior_ksd": np.asarray(prior),
    }
    with pytest.raises(ValueError, match="Invalid prior KSD denominator"):
        runner.summarize(arrays)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_json_does_not_replace_valid_result(runner, tmp_path, value):
    target = tmp_path / "summary.json"
    runner.write_json(target, {"status": "complete", "optional": None})
    before = target.read_bytes()
    with pytest.raises(ValueError, match="Out of range float"):
        runner.write_json(target, {"invalid": value})
    assert target.read_bytes() == before
    assert not target.with_suffix(".json.tmp").exists()


def test_indexed_trials_are_unchanged_by_compute_batch_size(
    runner, tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "plots", lambda *args: [])

    def simulator(key, *, mask_prior_hyperparameter):
        return {
            "model_mask": jnp.array([True, True, False, False, True]),
            "theta": jax.random.normal(key, (3,)),
            "mask_prior": jnp.array([0.5]),
        }

    functions = {
        "sbc": lambda data, key: {"model_pit": jax.random.uniform(key)},
        "ksd": lambda data, key: {
            "posterior_ksd": 0.1 + jax.random.uniform(key),
            "prior_ksd": 1.0 + jax.random.uniform(key),
        },
    }
    observed = []
    for batch_size in (3, 5):
        output = tmp_path / f"batch_{batch_size}"
        output.mkdir()
        args = SimpleNamespace(
            seed=7,
            num_samples=7,
            batch_size=batch_size,
            which="best",
            ksd_variant="frozen",
        )
        runner.evaluate(None, simulator, functions, args, 0, 0, None, output, None)
        with np.load(output / "samples.npz") as samples:
            observed.append({name: samples[name] for name in samples.files})

    for name in observed[0]:
        assert len(observed[0][name]) == 7  # Padded compute trials are never saved.
        np.testing.assert_array_equal(observed[0][name], observed[1][name])
    # Data, SBC and KSD use separate indexed streams for each actual trial.
    root = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(7), 0), 0)
    for trial in (0, 6):
        data_key, sbc_key, ksd_key = [
            jax.random.fold_in(jax.random.fold_in(root, stream), trial)
            for stream in range(3)
        ]
        np.testing.assert_array_equal(
            observed[0]["theta_true"][trial], jax.random.normal(data_key, (3,))
        )
        np.testing.assert_array_equal(
            observed[0]["model_pit"][trial], jax.random.uniform(sbc_key)
        )
        np.testing.assert_allclose(
            observed[0]["posterior_ksd"][trial], 0.1 + jax.random.uniform(ksd_key)
        )


def test_source_provenance_records_the_current_checkout(runner):
    source = Path(__file__).resolve().parents[1]
    expected = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    result = runner.source_provenance(source)
    assert result["commit"] == expected
    assert result["dirty"] == bool(result["status"])


def test_source_provenance_accepts_dirty_code_without_claiming_a_frozen_revision(
    runner, monkeypatch, tmp_path
):
    revision = "1" * 40
    status = " M experiments/dmri/synthetic_evaluation/run.py\n?? new_file.py\n"
    monkeypatch.setattr(
        file_storage.subprocess,
        "check_output",
        lambda command, **kwargs: revision if command[-1] == "HEAD" else status,
    )
    assert runner.source_provenance(tmp_path) == {
        "commit": revision,
        "dirty": True,
        "status": status.splitlines(),
    }


def test_source_provenance_outside_git_is_explicitly_unknown(runner, tmp_path):
    assert runner.source_provenance(tmp_path) == {
        "commit": None,
        "dirty": None,
        "status": None,
    }


@pytest.mark.parametrize("missing", ["--wandb-entity", "--wandb-project", "--run-id"])
def test_wandb_requires_an_explicit_destination_before_loading_or_writing(
    runner, tmp_path, monkeypatch, capsys, missing
):
    output = tmp_path / "results"
    arguments = {
        "--wandb-entity": "research-team",
        "--wandb-project": "synthetic-check",
        "--run-id": "test-run",
    }
    arguments.pop(missing)
    argv = ["run", "--checkpoint", "unused", "--output", str(output), "--wandb"]
    for name, value in arguments.items():
        argv.extend([name, value])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        runner,
        "load_model",
        lambda *args: pytest.fail("Loaded checkpoint before validation"),
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "--wandb requires explicit" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("enable_wandb", [False, True])
def test_run_records_current_provenance_and_uses_only_explicit_wandb_destination(
    runner, tmp_path, monkeypatch, enable_wandb
):
    source_git = {"commit": "2" * 40, "dirty": True, "status": [" M example.py"]}
    checkpoint_info = {
        "which": "best",
        "selected_step": 1520000,
        "params_name": "params_ema",
        "theta_dim": 3,
        "simulator_class": "ToySimulator",
    }
    monkeypatch.setattr(runner, "source_provenance", lambda source: source_git)
    monkeypatch.setattr(
        runner, "load_model", lambda *args: (None, None, [None, None], checkpoint_info)
    )
    monkeypatch.setattr(runner, "make_sbc_fn", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "evaluate", lambda *args: {"datasets": 2})
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "test")

    calls = []
    run = SimpleNamespace(
        url="https://wandb.ai/research-team/synthetic-check/runs/test-run",
        summary={},
        save=lambda *args, **kwargs: None,
        finish=lambda **kwargs: None,
    )

    def initialize(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=initialize, Settings=lambda **kwargs: kwargs),
    )
    output = tmp_path / "results"
    argv = [
        "run",
        "--checkpoint",
        "unused",
        "--output",
        str(output),
        "--num-samples",
        "2",
        "--mode",
        "sbc",
        "--sequence",
        "short",
    ]
    if enable_wandb:
        argv.extend([
            "--wandb",
            "--wandb-entity",
            "research-team",
            "--wandb-project",
            "synthetic-check",
            "--run-id",
            "test-run",
        ])
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    protocol = json.loads((output / "protocol.json").read_text())
    sources = protocol["evaluation_source_sha256"]
    for name in ("checkpoint", "metrics", "plotting", "storage"):
        assert len(sources[f"_shared/{name}.py"]) == 64
    assert protocol["source_commit"] == source_git["commit"]
    assert protocol["source_git"] == source_git
    assert protocol["source_commit"] != protocol["original_core_commit"]
    assert protocol["figure_source_commit"] == runner.FIGURE_SOURCE_COMMIT
    if enable_wandb:
        assert len(calls) == 1
        assert calls[0]["entity"] == "research-team"
        assert calls[0]["project"] == "synthetic-check"
        assert calls[0]["id"] == "test-run"
        assert calls[0]["config"] == protocol
        assert run.summary["evaluation_status"] == "completed"
    else:
        assert calls == []
        assert not (output / "wandb-url.txt").exists()
