"""Protect exact condition counts, resume integrity, RNG and paper aggregation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.dmri.calibration import chunks as runner
from experiments.dmri.calibration import evaluate as entrypoint
from experiments.dmri.calibration import protocol as recipe
from prism.dmri.simulators.acquisition_scheme import acquisition_scheme


def values(indices, mode="model"):
    indices = np.asarray(indices)
    size = len(indices)
    result = {
        "trial_index": indices,
        "model_mask": np.ones((size, 5), dtype=bool),
        "theta_true": np.repeat(indices[:, None], 2, axis=1).astype(float),
        "mask_prior": np.full((size, 1), 0.5),
        "num_acquisitions": np.full(size, 105),
    }
    if mode == "model":
        result["model_pit"] = (indices % 9 + 0.5) / 10
    else:
        result.update(
            theta_sum_pit=(indices % 9 + 0.5) / 10,
            theta_pit=np.full((size, 2), 0.5),
            theta_mask=np.ones((size, 2), dtype=bool),
        )
    return result


def execute(path, compute=values, **overrides):
    options = {
        "trials": 10,
        "batch_size": 2,
        "chunk_size": 4,
        "sequence": 0,
        "protocol_hash": "protocol",
    }
    options.update(overrides)
    return runner.evaluate_sequence(path, compute, **options)


def test_exact_trial_count_and_final_partial_chunk(tmp_path):
    arrays, records = execute(tmp_path)
    np.testing.assert_array_equal(arrays["trial_index"], np.arange(10))
    assert [(row["start"], row["stop"]) for row in records] == [(0, 4), (4, 8), (8, 10)]
    assert len(list((tmp_path / "chunks").iterdir())) == 3


def test_resume_reuses_verified_chunks_without_recomputing_or_overwriting(tmp_path):
    calls = []

    def interrupted(indices):
        calls.append(int(indices[0]))
        if int(indices[0]) == 4:
            raise RuntimeError("simulated interruption")
        return values(indices)

    with pytest.raises(RuntimeError, match="interruption"):
        execute(tmp_path, interrupted)
    first = tmp_path / "chunks" / "0000000-0000004" / "samples.npz"
    timestamp, content = first.stat().st_mtime_ns, first.read_bytes()
    calls.clear()

    def resumed(indices):
        calls.append(int(indices[0]))
        return values(indices)

    arrays, _ = execute(tmp_path, resumed)
    assert calls == [4, 6, 8]
    assert first.read_bytes() == content and first.stat().st_mtime_ns == timestamp
    np.testing.assert_array_equal(arrays["trial_index"], np.arange(10))

    def forbidden(indices):
        raise AssertionError("Completed work must not run again")

    complete, _ = execute(tmp_path, forbidden)
    np.testing.assert_array_equal(complete["model_pit"], arrays["model_pit"])


def test_partial_temporary_chunk_is_ignored_and_missing_trials_recomputed(tmp_path):
    temporary = tmp_path / "chunks" / ".writing-interrupted"
    temporary.mkdir(parents=True)
    (temporary / "samples.npz").write_bytes(b"partial")
    arrays, _ = execute(tmp_path)
    assert len(arrays["trial_index"]) == 10


@pytest.mark.parametrize("mutation", ["content", "protocol", "gap", "target_count"])
def test_resume_rejects_changed_content_protocol_or_ranges(tmp_path, mutation):
    execute(tmp_path)
    first = tmp_path / "chunks" / "0000000-0000004"
    options = {}
    if mutation == "content":
        with (first / "samples.npz").open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "protocol":
        options["protocol_hash"] = "another"
    elif mutation == "gap":
        first.rename(tmp_path / "chunks" / "0000001-0000004")
    else:
        options["trials"] = 9
    with pytest.raises(ValueError):
        execute(tmp_path, **options)


def test_duplicate_chunk_publication_is_rejected(tmp_path):
    sample = values(np.arange(4))
    options = {
        "start": 0,
        "stop": 4,
        "sequence": 0,
        "protocol_hash": "same",
        "seconds": 0.1,
    }
    runner.commit_chunk(tmp_path, sample, **options)
    with pytest.raises(FileExistsError):
        runner.commit_chunk(tmp_path, sample, **options)


def test_inactive_parameter_nan_is_allowed_but_active_nan_is_rejected():
    sample = values(np.arange(2), "theta")
    sample["theta_mask"][:, 1] = False
    sample["theta_pit"][:, 1] = np.nan
    runner.validate_values(sample, 0, 2)
    sample["theta_pit"][0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="active parameter PIT"):
        runner.validate_values(sample, 0, 2)


def test_changed_trial_ids_are_rejected():
    sample = values(np.arange(4))
    sample["trial_index"][2] = 1
    with pytest.raises(ValueError, match="indices"):
        runner.validate_values(sample, 0, 4)


def test_resume_protocol_is_immutable(tmp_path):
    output = tmp_path / "run"
    protocol = {"seed": 0, "checkpoint": "ema-sha", "conditions": [1, 2]}
    digest = runner.prepare_output(output, protocol, resume=False)
    assert runner.prepare_output(output, protocol, resume=True) == digest
    with pytest.raises(FileExistsError):
        runner.prepare_output(output, protocol, resume=False)
    with pytest.raises(ValueError, match="protocol differs"):
        runner.prepare_output(output, {**protocol, "seed": 1}, resume=True)


def test_completed_summary_is_verified_and_not_rewritten(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "plots", lambda *args: [])
    arrays, records = execute(tmp_path)
    options = {"sequence": 0, "protocol_hash": "protocol", "label": "test"}
    summary = runner.finalize_sequence(tmp_path, arrays, records, **options)
    marker = tmp_path / "summary" / "summary.json"
    timestamp = marker.stat().st_mtime_ns
    assert runner.finalize_sequence(tmp_path, arrays, records, **options) == summary
    assert marker.stat().st_mtime_ns == timestamp
    marker.write_text("{}")
    with pytest.raises(ValueError, match="summary files changed"):
        runner.finalize_sequence(tmp_path, arrays, records, **options)


def test_declared_conditions_match_frozen_notebook():
    model = recipe.conditions("model")
    assert len(model) == 11 and {case["sequence"] for case in model} == {0}
    assert [case["hyper"] for case in model] == [
        None,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.8,
        0.9,
        1,
    ]
    theta = recipe.conditions("theta")
    assert len(theta) == 30
    assert len({tuple(case["mask"]) for case in theta}) == 15
    assert all(case["mask"][-1] == 1 and any(case["mask"][:4]) for case in theta)
    assert [case["mask_index"] for case in theta[:15]] == list(range(15))
    assert recipe.RIGHT_PANEL_INDICES == (0, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14)


@pytest.mark.parametrize(
    "mode,field,total", [("model", "model_pit", 11), ("theta", "theta_sum_pit", 30)]
)
def test_curve_aggregation_is_balanced_not_weighted_by_trial_or_mask_count(
    mode, field, total
):
    summaries = {
        case["name"]: {
            field: {
                "grid": [0, 0.5, 1],
                "cdf": [0, index / total, 1],
                "count": index + 1,
            }
        }
        for index, case in enumerate(recipe.conditions(mode))
    }
    result = recipe.aggregate(mode, summaries)
    np.testing.assert_allclose(result["cdf"], [0, (total - 1) / (2 * total), 1])
    assert result["statistic"] == field
    summaries.pop(next(iter(summaries)))
    with pytest.raises(ValueError, match="every declared condition"):
        recipe.aggregate(mode, summaries)


def test_fixed_mask_and_lambda_generators_preserve_indexed_rng_across_batch_sizes(
    tmp_path,
):
    def simulator(key, mask_prior_hyperparameter=None, model_mask=None):
        mask = jnp.ones(5, bool) if model_mask is None else model_mask
        prior = (
            jnp.array([0.25])
            if mask_prior_hyperparameter is None
            else mask_prior_hyperparameter
        )
        return {
            "theta": jax.random.normal(key, (2,)),
            "model_mask": mask,
            "mask_prior": prior,
            "acq": acquisition_scheme(jnp.zeros(7), jnp.zeros((7, 3))),
        }

    def sbc(data, key):
        return {"model_pit": jax.random.uniform(key)}

    compute = recipe.make_batch_function(
        simulator,
        sbc,
        sequence=1,
        condition_index=3,
        mask=[1, 0, 0, 0, 1],
        hyper=0.8,
        trials=10,
    )
    first, _ = execute(tmp_path / "batch2", compute, batch_size=2)
    second, _ = execute(tmp_path / "batch4", compute, batch_size=4)
    for field in first:
        np.testing.assert_allclose(first[field], second[field], rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(first["mask_prior"], 0.8)
    np.testing.assert_array_equal(first["num_acquisitions"], np.full(10, 7))
    np.testing.assert_array_equal(
        first["model_mask"], np.tile([1, 0, 0, 0, 1], (10, 1))
    )


def test_cli_counts_modes_and_gpu_defaults(tmp_path):
    for mode, count in (("model", 100000), ("theta", 10000)):
        args = entrypoint.parse_args([
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--output",
            str(tmp_path / "eval"),
            "--mode",
            mode,
        ])
        assert args.trials == count
        assert args.which == "latest"
        assert args.slurm is None and not args.resume and not args.wandb
        assert args.mask_samples == 200 and args.theta_samples == 50
        assert args.ode_steps == 64 and args.t_min == 1e-4


def test_condition_roots_match_sequential_frozen_launcher():
    model_key = jax.random.PRNGKey(0)
    for index in range(11):
        model_key, expected_data = jax.random.split(model_key)
        model_key, expected_pit = jax.random.split(model_key)
        actual = recipe.condition_roots("model", 0, index)
        np.testing.assert_array_equal(actual[0], expected_data)
        np.testing.assert_array_equal(actual[1], expected_pit)
    theta_key = jax.random.PRNGKey(0)
    for sequence in range(2):
        # The frozen launcher generates eleven mixture/hyperprior conditions first.
        for _ in range(11):
            theta_key, _ = jax.random.split(theta_key)
            theta_key, _ = jax.random.split(theta_key)
        for mask_index in range(15):
            theta_key, expected_data = jax.random.split(theta_key)
            theta_key, expected_pit = jax.random.split(theta_key)
            actual = recipe.condition_roots("theta", sequence, mask_index + 1)
            np.testing.assert_array_equal(actual[0], expected_data)
            np.testing.assert_array_equal(actual[1], expected_pit)


def test_split_trial_keys_are_preserved_when_accessed_in_resumed_chunks():
    def simulator(key, **kwargs):
        return {
            "theta": key,
            "model_mask": jnp.ones(5, bool),
            "mask_prior": jnp.array([0.5]),
            "acq": acquisition_scheme(jnp.zeros(3), jnp.zeros((3, 3))),
        }

    def statistic(data, key):
        return {"model_pit": jax.random.uniform(key)}

    compute = recipe.make_batch_function(
        simulator, statistic, sequence=0, condition_index=7, trials=10
    )
    data_root, pit_root = recipe.condition_roots("model", 0, 7)
    expected_data = jax.random.split(data_root, 10)
    expected_pit = jax.vmap(jax.random.uniform)(jax.random.split(pit_root, 10))
    for indices in (jnp.arange(4), jnp.arange(4, 8), jnp.arange(8, 12)):
        value = compute(indices)
        count = min(4, 10 - int(indices[0]))
        np.testing.assert_array_equal(
            np.asarray(value["theta_true"])[:count],
            expected_data[int(indices[0]) : int(indices[0]) + count],
        )
        np.testing.assert_array_equal(
            np.asarray(value["model_pit"])[:count],
            expected_pit[int(indices[0]) : int(indices[0]) + count],
        )


def test_nonfinite_batch_is_preserved_without_committing_chunk(tmp_path):
    def broken(indices):
        sample = values(indices)
        sample["model_pit"][0] = np.nan
        return sample

    with pytest.raises(FloatingPointError):
        execute(tmp_path, broken)
    assert len(list((tmp_path / "failures").glob("*.npz"))) == 1
    assert not (tmp_path / "chunks").exists()


def test_conditions_can_share_one_precompiled_metric_function():
    def simulator(key, **kwargs):
        return {
            "theta": jnp.zeros(2),
            "model_mask": jnp.ones(5, bool),
            "mask_prior": jnp.array([0.5]),
            "acq": acquisition_scheme(jnp.zeros(3), jnp.zeros((3, 3))),
        }

    def unused(data, key):
        raise AssertionError(
            "An injected metric must replace per-condition compilation"
        )

    calls = []

    def shared_metric(data, keys):
        calls.append(keys.shape)
        return {"model_pit": jnp.full(keys.shape[0], 0.5)}

    for condition in (0, 3):
        compute = recipe.make_batch_function(
            simulator,
            unused,
            sequence=0,
            condition_index=condition,
            trials=4,
            measure=shared_metric,
        )
        compute(jnp.arange(4))
    assert calls == [(4, 2), (4, 2)]


def test_tracking_requires_an_explicit_destination_and_run_id(tmp_path):
    args = [
        "--checkpoint",
        str(tmp_path),
        "--output",
        str(tmp_path / "out"),
        "--mode",
        "model",
    ]
    assert entrypoint.parse_args(args).wandb is False
    with pytest.raises(SystemExit):
        entrypoint.parse_args([*args, "--wandb"])
    parsed = entrypoint.parse_args([
        *args,
        "--wandb",
        "--wandb-entity",
        "example-team",
        "--wandb-project",
        "calibration",
        "--run-id",
        "example-run",
    ])
    assert parsed.wandb and parsed.wandb_entity == "example-team"


def test_checkpoint_manifest_hashes_hub_file_symlinks_only_when_requested(tmp_path):
    target = tmp_path / "blob"
    target.write_bytes(b"checkpoint content")
    folder = tmp_path / "snapshot"
    folder.mkdir()
    (folder / "weights").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        runner.file_manifest(folder)
    assert runner.file_manifest(folder, allow_file_symlinks=True) == {
        "weights": runner.file_sha256(target)
    }
