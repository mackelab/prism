"""Check numerical grouping and calibration/posterior-quality panel construction."""

import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.dmri.calibration import plot as calibration
from experiments.dmri.synthetic_evaluation import plot as ksd


def author_masks():
    return ksd.expand_label_masks(ksd.build_label_masks(ksd.LABELS_MASK, 5))


def ksd_fixture():
    masks, _ = author_masks()
    scale = np.arange(1, 16, dtype=float)[:, None]
    return {
        "mask_support": masks,
        "ksd": scale * np.array([0.0, 0.1, 1.0, 10.0]),
        "ksd_prior": np.broadcast_to(np.array([2.0, 2.0, 2.0, 2.0]), (15, 4)).copy(),
    }


def theta_fixture():
    masks, labels = author_masks()
    alpha = np.linspace(0, 1, 101)
    coverage = np.array([alpha ** (1 + i * 0.03) for i in range(15)])
    return {
        "alpha": alpha,
        "theta_mask_coverage": coverage,
        "theta_mask_errors": np.abs(coverage - alpha).mean(1),
        "theta_mask_labels": np.array(labels),
        "theta_mask_support": masks,
        "theta_mask_counts": np.full(15, 10),
    }


def source_fingerprint(node):
    """Use the Python 3.12 AST schema recorded in the frozen fingerprints."""
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if "type_params" not in child._fields:
                # Python 3.11 has no type-parameter field on these nodes.
                child._fields = (*child._fields, "type_params")
                child.type_params = []
    # Python 3.13 omits empty lists and None fields from ast.dump unless asked.
    show_empty = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    dump = ast.dump(node, include_attributes=False, **show_empty)
    return hashlib.sha256(dump.encode()).hexdigest()


def test_imported_helpers_and_style_match_frozen_author_fingerprints():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/dmri_author_plots.json").read_text()
    )
    for name, expected in fixture["functions"].items():
        source = ast.parse(inspect.getsource(getattr(ksd, name)))
        actual = source_fingerprint(source.body[0])
        assert actual == expected, name
    assert (
        hashlib.sha256(calibration.STYLE.read_bytes()).hexdigest()
        == fixture["style_sha256"]
    )


def test_fingerprints_match_without_python312_type_parameter_fields():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/dmri_author_plots.json").read_text()
    )
    for name, expected in fixture["functions"].items():
        source = ast.parse(inspect.getsource(getattr(ksd, name))).body[0]
        for child in ast.walk(source):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child._fields = tuple(f for f in child._fields if f != "type_params")
                if hasattr(child, "type_params"):
                    del child.type_params
        assert source_fingerprint(source) == expected, name


def test_fingerprint_still_detects_changed_calculations():
    original = ast.parse("def calculate(x):\n    return x + 1").body[0]
    changed = ast.parse("def calculate(x):\n    return x + 2").body[0]
    assert source_fingerprint(original) != source_fingerprint(changed)


def test_source_normalization_trim_and_notebook_mask_selection():
    data = ksd_fixture()
    values, labels = ksd.preprocess(data)
    for i, group in enumerate(values):
        np.testing.assert_allclose(group, (i + 1) * np.array([0.0, 0.05, 0.5]))
    figure, groups, selected_labels, group_labels = ksd.plot_inputs(
        {"synthetic": data}, 105
    )
    try:
        expected_indices = [0, 1, *range(6, 15)]
        assert selected_labels == [labels[i] for i in expected_indices]
        assert group_labels == ["syn."]
        for value, i in zip(groups[0], expected_indices, strict=True):
            np.testing.assert_array_equal(value, values[i])
        assert figure.get_size_inches().tolist() == [3.3, 1.8]
        assert figure.axes[0].get_yscale() == "log"
        assert figure.axes[0].get_ylabel() == "rKSD (n=105)"
    finally:
        plt.close(figure)


def test_original_violin_receives_linear_values_including_zeros(monkeypatch):
    calls = []
    original = plt.Axes.violinplot

    def capture(self, dataset, *args, **kwargs):
        calls.append([np.array(values) for values in dataset])
        return original(self, dataset, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "violinplot", capture)
    data = ksd_fixture()
    fig, groups, _, names = ksd.plot_inputs(
        {name: data for name, _, _ in ksd.GROUPS}, 105
    )
    try:
        assert names == ["syn.", "real (selected)", "real (fixed)"]
        assert len(calls) == 3
        for given, expected in zip(calls, groups, strict=True):
            for a, b in zip(given, expected, strict=True):
                np.testing.assert_array_equal(a, b)
                assert a[0] == 0
    finally:
        plt.close(fig)


def test_padded_author_rows_preserve_finite_filtering_and_zero_prior_behavior():
    data = ksd_fixture()
    data["ksd"][0, -1] = np.nan
    data["ksd_prior"][1] = 0
    values, _ = ksd.preprocess(data)
    np.testing.assert_array_equal(values[0], [0.0, 0.05])
    assert len(values[1]) == 0


def test_model_and_parameter_panel_means_match_notebook():
    short = theta_fixture()
    long = {**short, "theta_mask_coverage": short["theta_mask_coverage"] ** 1.1}
    coverage = short["theta_mask_coverage"][:11]
    model_fig = calibration.plot_model(short["alpha"], coverage)
    theta_fig = calibration.plot_theta(short, long)
    try:
        np.testing.assert_array_equal(
            model_fig.axes[0].lines[0].get_ydata(), coverage.mean(0)
        )
        expected = (
            (short["theta_mask_coverage"] + long["theta_mask_coverage"]) / 2
        ).mean(0)
        np.testing.assert_array_equal(theta_fig.axes[0].lines[0].get_ydata(), expected)
        assert model_fig.get_size_inches().tolist() == [0.65, 0.65]
        assert theta_fig.axes[0].lines[0].get_color() == "black"
    finally:
        plt.close(model_fig)
        plt.close(theta_fig)


def test_parameter_error_subset_colors_and_supplied_reference_band():
    data = theta_fixture()
    figure = calibration.plot_theta_errors(data, data, (0.001, 0.02))
    try:
        subset = [0, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14]
        axis = figure.axes[0]
        np.testing.assert_array_equal(
            axis.lines[0].get_ydata(), data["theta_mask_errors"][subset]
        )
        assert [line.get_color() for line in axis.lines] == ["#1b7a52", "#2ecf8b"]
        assert [line.get_drawstyle() for line in axis.lines] == [
            "steps-mid",
            "steps-mid",
        ]
        assert len(axis.collections) == 1
        assert figure.get_size_inches().tolist() == [1.5, 0.65]
    finally:
        plt.close(figure)


def test_reference_band_matches_reduced_original_notebook_calculation():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/dmri_author_plots.json").read_text()
    )["reference_band"]
    result = calibration.random_errors(
        jax.random.PRNGKey(0), 16, repetitions=32, batch_size=8
    )
    np.testing.assert_array_equal(result, fixture["values"])


def test_condition_adapter_reorders_by_mask_bits_not_record_position(tmp_path):
    masks, labels = author_masks()
    alpha = np.linspace(0, 1, 101)
    conditions, rows = [], {}
    for sequence, scheme in enumerate(("short", "long")):
        for index in reversed(range(15)):
            mask = masks[index]
            name = scheme + "_" + "".join(str(int(bit)) for bit in mask)
            conditions.append({
                "name": name,
                "sequence": sequence,
                "mask": mask.astype(int).tolist(),
            })
            cdf = alpha ** (1 + index * 0.03)
            rows[name] = {
                "completed_trials": 10,
                "theta_sum_pit": {
                    "grid": alpha.tolist(),
                    "cdf": cdf.tolist(),
                    "calibration_error": float(np.abs(cdf - alpha).mean()),
                },
            }
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({
            "status": "completed",
            "protocol": {
                "mode": "theta",
                "conditions": conditions,
                "trials_per_condition": 10,
            },
            "results": rows,
        })
    )
    groups = calibration.load_condition_result(path, "theta")
    calibration.validate_theta(groups["short"], groups["long"])
    assert groups["short"]["theta_mask_labels"].tolist() == labels
    np.testing.assert_array_equal(groups["short"]["theta_mask_support"], masks)
    np.testing.assert_array_equal(
        groups["short"]["theta_mask_coverage"][12], alpha ** (1 + 12 * 0.03)
    )
    assert not path.with_name("plot.json").exists()


def test_original_npz_and_trial_array_adapter_preserve_values(tmp_path):
    data = ksd_fixture()
    original = tmp_path / "author.npz"
    np.savez(original, **data)
    loaded, source = ksd.read_input(original, "frozen")
    for key in data:
        np.testing.assert_array_equal(loaded[key], data[key])
    assert source["schema"] == "author per-mask NPZ"
    directory = tmp_path / "run" / "short_overall"
    directory.mkdir(parents=True)
    sample = directory / "samples.npz"
    np.savez(
        sample,
        model_mask=data["mask_support"],
        posterior_ksd=data["ksd"][:, 2],
        prior_ksd=data["ksd_prior"][:, 2],
    )
    (directory / "summary.json").write_text(
        json.dumps({"samples_sha256": hashlib.sha256(sample.read_bytes()).hexdigest()})
    )
    protocol = {
        "ksd_kernel": {"variant": "corrected"},
        "checkpoint": {},
        "arguments": {},
    }
    (directory.parent / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="kernel differs"):
        ksd.read_input(sample, "frozen")
    mapped, _ = ksd.read_input(sample, "corrected")
    np.testing.assert_array_equal(mapped["ksd"], data["ksd"][:, 2])
    with sample.open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        ksd.read_input(sample, "corrected")


def test_original_npz_plot_entrypoint_writes_only_requested_groups(tmp_path):
    data = ksd_fixture()
    source = tmp_path / "synthetic.npz"
    np.savez(source, **data)
    before = source.read_bytes()
    output = tmp_path / "plots"
    ksd.main([
        "--synthetic",
        str(source),
        "--measurements",
        "105",
        "--output",
        str(output),
    ])
    manifest = json.loads((output / "plot.json").read_text())
    assert manifest["group_labels"] == ["syn."]
    assert manifest["all_paper_groups_present"] is False
    assert len(manifest["mask_support"]) == 11
    assert all(
        (output / f"ksd_violin.{ext}").is_file() for ext in ("png", "svg", "pdf")
    )
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        ksd.main([
            "--synthetic",
            str(source),
            "--measurements",
            "105",
            "--output",
            str(output),
        ])


def test_calibration_npz_entrypoint_and_band_count_validation(tmp_path):
    data = theta_fixture()
    path = tmp_path / "theta.npz"
    np.savez(path, **data)
    output = tmp_path / "plots"
    calibration.main([
        "--theta-short",
        str(path),
        "--theta-long",
        str(path),
        "--output",
        str(output),
    ])
    assert (output / "dmri_theta_coverage.svg").is_file()
    assert (output / "coverage_calibration_error_subset.png").is_file()
    band = tmp_path / "wrong_band.json"
    band.write_text(
        json.dumps({
            "lower": 0.0,
            "upper": 0.1,
            "datasets_per_condition": 11,
            "alpha_count": 101,
        })
    )
    with pytest.raises(ValueError, match="count differs"):
        calibration.main([
            "--theta-short",
            str(path),
            "--theta-long",
            str(path),
            "--theta-band",
            str(band),
            "--output",
            str(tmp_path / "invalid"),
        ])
    plt.close("all")
