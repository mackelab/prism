import numpy as np
import pytest

from experiments.symbolic.scaling_appendix import plot

TASKS = (20, 30, 50, 80, 100)


def _write_archives(root, n=200):
    rng = np.random.default_rng(0)
    gt = rng.normal(1.0, 0.01, (5, n)).astype("float32")
    np.savez(root / "eval_pred_error_gt.npz", errors=gt)
    for size in ("", "_small", "_tiny"):
        np.savez(
            root / f"eval_pred_error_predictive{size}.npz",
            errors=(gt + 0.5 + 0.1 * rng.random((5, n))).astype("float32"),
        )
        for task in TASKS:
            np.savez(
                root / f"eval_ksd_updated2_symbolic{task}{size}.npz",
                ksd=np.r_[np.full(n, 0.5), 1e9, np.nan],
                ksd_prior=np.r_[np.full(n, 2.0), 1e9, np.inf],
            )
            for mode in ("mask", "theta"):
                alpha = np.linspace(0, 1, 101)
                np.savez(
                    root / f"eval_coverage_{mode}_updated2_symbolic{task}{size}.npz",
                    alpha=alpha,
                    coverage=alpha + 0.01 * (-1.0) ** np.arange(101),
                )


@pytest.fixture
def archives(tmp_path):
    _write_archives(tmp_path)
    return tmp_path


def test_aggregation(archives):
    rrmse = plot.rrmse_summary(archives, n_boot=20)
    mean, low, high = rrmse[""]
    assert mean.shape == (5,)
    assert (low <= mean).all() and (mean <= high).all()
    assert (mean > 0.5).all() and (mean < 0.7).all()

    entries = plot.rksd_entries(archives)
    assert len(entries) == 15
    # outliers (1e9) are dropped from ksd and prior; 0.5 / 2.0 = 0.25
    for e in entries:
        np.testing.assert_allclose(e["rksd"], 0.25)
    assert {e["size"] for e in entries} == {"", "small", "tiny"}

    cal = plot.calibration_entries(archives)
    assert set(cal) == {"mask", "theta"}
    assert len(cal["mask"]) == 15
    # coverage - alpha alternates +-0.01, so the mean absolute error is 0.01
    assert all(e["error"] == pytest.approx(0.01) for e in cal["mask"])

    low_band, high_band = plot.random_calibration_band(n=1000, reps=50)
    assert 0 < low_band < high_band < 0.1


def test_bootstrap_helpers():
    mean, low, high = plot.bootstrap_mean_ci(np.arange(100.0))
    assert low < mean < high
    assert plot.bootstrap_mean_ci(np.array([3.0])) == (3.0, 3.0, 3.0)
    assert np.isnan(plot.bootstrap_mean_ci(np.array([]))[0])
    kept = plot.exclude_outliers(np.arange(101.0))
    assert kept.min() >= 2.5 and kept.max() <= 97.5


def test_main_writes_figure_once(archives, tmp_path):
    out = tmp_path / "fig.svg"
    plot.main([str(archives), "--output", str(out), "--n-boot", "20"])
    assert out.stat().st_size > 0
    with pytest.raises(FileExistsError):
        plot.main([str(archives), "--output", str(out)])


def test_first_checkpoint_is_drawn_at_two_to_the_fifteen(archives):
    fig = plot.plot_scaling(
        plot.rrmse_summary(archives, n_boot=5),
        plot.rksd_entries(archives),
        plot.calibration_entries(archives),
        (0.0, 0.01),
    )
    cal_x = [line.get_xdata() for line in fig.axes[2].lines]
    assert all(x[0] == 2.0**15 and x[-1] == 2.0**100 for x in cal_x)
    ksd_x = fig.axes[1].lines[0].get_xdata()
    assert ksd_x[0] == 2.0**20  # kept from the notebook


def test_missing_checkpoint_is_reported(archives):
    (archives / "eval_ksd_updated2_symbolic50_tiny.npz").unlink()
    with pytest.raises(ValueError, match="tiny"):
        plot.rksd_entries(archives)
    (archives / "eval_coverage_mask_updated2_symbolic20.npz").unlink()
    with pytest.raises(ValueError, match="mask calibration"):
        plot.calibration_entries(archives)


def test_unrecognised_file_names_are_ignored(archives):
    np.savez(archives / "eval_ksd_updated2_symbolic50_huge.npz", ksd=np.zeros(3))
    assert len(plot.rksd_entries(archives)) == 15
