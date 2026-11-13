"""Figure A.2: rRMSE, rKSD and calibration error against model-space size.

Aggregates and plots stored evaluation archives (they are not produced by this
module). The directory must contain the files used by the original notebook:

* ``eval_pred_error_{predictive,gt}.npz`` and the ``_small`` / ``_tiny``
  predictive variants (``errors`` with shape (checkpoints, samples)),
* ``eval_ksd_updated2_symbolic{K}[_small|_tiny].npz`` (``ksd``, ``ksd_prior``),
* ``eval_coverage_{mask,theta}_updated2_symbolic{K}[_small|_tiny].npz``
  (``alpha``, ``coverage``).

Example: python -m experiments.symbolic.scaling_appendix.plot RESULTS_DIR \
    --output scaling_metrics_appendix.svg
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Model-space sizes (powers of two) of the five rRMSE checkpoints, in the order
# of the stored archives. The notebook labels the first one 2**15 although the
# stored checkpoint is updated2_symbolic20 (see README).
RRMSE_POWERS = (15, 30, 50, 80, 100)
TASKS = (20, 30, 50, 80, 100)  # symbolic{K} checkpoints in the stored archives
FIRST_TASK = TASKS[0]
PIT_COUNT = 32768  # num_samples of the stored coverage archives
SIZES = ("", "small", "tiny")
SIZE_STYLES = {
    "": {"label": "standard", "color": "#0D0D86", "marker": "o"},
    "small": {"label": "small", "color": "#0000C0", "marker": "s"},
    "tiny": {"label": "tiny", "color": "#5858FF", "marker": "^"},
}
PARAM_LABELS = {"standard": "~5M params", "small": "~2M params", "tiny": "~1M params"}
MODE_STYLES = {"mask": "-", "theta": "--"}


def bootstrap_ci(data, n_boot=500, ci=95, seed=0):
    """Percentile interval of the column means of ``data`` (resampling rows)."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    means = np.array([data[rng.integers(0, n, n)].mean(axis=0) for _ in range(n_boot)])
    return (
        np.percentile(means, (100 - ci) / 2, axis=0),
        np.percentile(means, 100 - (100 - ci) / 2, axis=0),
    )


def bootstrap_mean_ci(data, n_boot=1000, ci=95, seed=0):
    """Mean and bootstrap interval of a 1-d sample."""
    if data.size == 0:
        return np.nan, np.nan, np.nan
    if data.shape[0] == 1:
        mean = float(np.mean(data))
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    means = np.array([
        rng.choice(data, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    return (
        float(np.mean(data)),
        np.percentile(means, (100 - ci) / 2),
        np.percentile(means, 100 - (100 - ci) / 2),
    )


def exclude_outliers(data, q=0.95):
    """Keep values inside the central ``q`` quantile range."""
    if data.size == 0:
        return data
    low = np.quantile(data, (1 - q) / 2)
    high = np.quantile(data, 1 - (1 - q) / 2)
    return data[(data >= low) & (data <= high)]


def rrmse_summary(results_dir, n_boot=500):
    """Per width: mean, lower and upper of predictive error minus ground truth."""
    results_dir = Path(results_dir)
    gt = np.load(results_dir / "eval_pred_error_gt.npz", allow_pickle=False)["errors"]
    gt_mean = gt.mean(axis=1)
    out = {}
    for size in SIZES:
        suffix = f"_{size}" if size else ""
        path = results_dir / f"eval_pred_error_predictive{suffix}.npz"
        errors = np.load(path, allow_pickle=False)["errors"] - gt_mean[:, None]
        lower, upper = bootstrap_ci(errors.T, n_boot=n_boot)
        out[size] = (errors.mean(axis=-1), lower, upper)
        if errors.shape[0] != len(RRMSE_POWERS):
            raise ValueError(
                f"{path.name}: expected {len(RRMSE_POWERS)} checkpoints, got {errors.shape[0]}"
            )
    return out


def _require_complete(entries, what):
    """Each width must have exactly the checkpoints K = 20, 30, 50, 80, 100."""
    for size in SIZES:
        found = sorted(e["task"] for e in entries if e["size"] == size)
        if found != sorted(TASKS):
            raise ValueError(
                f"{what} width {size or 'standard'!r}: found K={found}, need {TASKS}"
            )


def rksd_entries(results_dir):
    """Per file: ksd divided by the mean prior ksd, after dropping outliers."""
    entries = []
    for path in sorted(Path(results_dir).glob("eval_ksd_updated2_symbolic*.npz")):
        match = re.fullmatch(
            r"eval_ksd_updated2_symbolic(\d+)(?:_(small|tiny))?\.npz", path.name
        )
        if not match:
            continue
        data = np.load(path, allow_pickle=False)
        ksd = data["ksd"][np.isfinite(data["ksd"])]
        prior = data["ksd_prior"][np.isfinite(data["ksd_prior"])]
        prior_mean = np.mean(exclude_outliers(prior))
        if not np.isfinite(prior_mean) or prior_mean <= 0:
            raise ValueError(f"{path.name}: invalid mean prior KSD {prior_mean}")
        entries.append({
            "task": int(match.group(1)),
            "size": match.group(2) or "",
            "rksd": exclude_outliers(ksd) / prior_mean,
        })
    _require_complete(entries, "rKSD")
    return entries


def calibration_entries(results_dir):
    """Per mode and file: mean |coverage - alpha|."""
    out = {mode: [] for mode in MODE_STYLES}
    for mode in MODE_STYLES:
        pattern = rf"eval_coverage_{mode}_updated2_symbolic(\d+)(?:_(small|tiny))?\.npz"
        for path in sorted(
            Path(results_dir).glob(f"eval_coverage_{mode}_updated2_symbolic*.npz")
        ):
            match = re.fullmatch(pattern, path.name)
            if not match:
                continue
            data = np.load(path, allow_pickle=False)
            out[mode].append({
                "task": int(match.group(1)),
                "size": match.group(2) or "",
                "error": float(np.mean(np.abs(data["coverage"] - data["alpha"]))),
            })
        _require_complete(out[mode], f"{mode} calibration")
        out[mode].sort(key=lambda e: (e["task"], e["size"]))
    return out


def random_calibration_band(n=PIT_COUNT, reps=1000, seed=0, alpha=None):
    """2.5% / 97.5% quantiles of the calibration error of uniform PITs."""
    alpha = np.linspace(0, 1, 101) if alpha is None else alpha
    rng = np.random.default_rng(seed)
    errors = [
        np.abs(np.quantile(rng.uniform(0, 1, n), alpha) - alpha).mean()
        for _ in range(reps)
    ]
    return np.quantile(errors, 0.025), np.quantile(errors, 0.975)


def plot_scaling(rrmse, rksd, calibration, band):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0))
    ax_rmse, ax_ksd, ax_cal = axes
    x_rmse = 2.0 ** np.asarray(RRMSE_POWERS, dtype=float)

    for size in SIZES:
        mean, low, high = (np.asarray(v, dtype=float) for v in rrmse[size])
        style = SIZE_STYLES[size]
        ax_rmse.errorbar(
            x_rmse,
            mean,
            yerr=np.maximum(0.0, [mean - low, high - mean]),
            fmt=style["marker"] + "-",
            color=style["color"],
            label=PARAM_LABELS[style["label"]],
            capsize=3.0,
            capthick=1.0,
            elinewidth=1.0,
            alpha=0.95,
            zorder=3,
        )
    ax_rmse.set_ylabel("rRMSE")

    for size in SIZES:
        entries = sorted(
            (e for e in rksd if e["size"] == size), key=lambda e: e["task"]
        )
        if not entries:
            continue
        stats = np.array([bootstrap_mean_ci(e["rksd"]) for e in entries])
        style = SIZE_STYLES[size]
        ax_ksd.errorbar(
            2.0 ** np.array([e["task"] for e in entries], dtype=float),
            stats[:, 0],
            yerr=np.maximum(
                0.0, [stats[:, 0] - stats[:, 1], stats[:, 2] - stats[:, 0]]
            ),
            fmt=style["marker"] + "-",
            color=style["color"],
            linewidth=1.6,
            markersize=4.5,
            capsize=3.0,
            alpha=0.95,
        )
    ax_ksd.set_ylabel("rKSD")

    for mode, entries in calibration.items():
        for size in SIZES:
            sized = sorted(
                (e for e in entries if e["size"] == size), key=lambda e: e["task"]
            )
            if not sized:
                continue
            tasks = np.array([e["task"] for e in sized], dtype=float)
            # as in the notebook: the K=20 checkpoint is drawn at 2**15
            tasks[tasks == FIRST_TASK] = RRMSE_POWERS[0]
            style = SIZE_STYLES[size]
            ax_cal.plot(
                2.0**tasks,
                [e["error"] for e in sized],
                style["marker"] + MODE_STYLES[mode],
                color=style["color"],
                linewidth=1.6,
                markersize=4.5,
                alpha=0.95,
            )
    ax_cal.fill_between(x_rmse, band[0], band[1], color="gray", alpha=0.2)
    ax_cal.set_ylabel("Calibration error")
    ax_cal.legend(
        handles=[
            Line2D(
                [],
                [],
                color="black",
                linestyle=MODE_STYLES["mask"],
                label="model posterior",
            ),
            Line2D(
                [],
                [],
                color="black",
                linestyle=MODE_STYLES["theta"],
                label="parameter posterior",
            ),
        ],
        frameon=False,
        loc="upper right",
        fontsize=7,
        handlelength=1.5,
    )

    ticks = 2.0 ** np.asarray(RRMSE_POWERS)
    labels = [rf"$2^{{{k}}}$" for k in RRMSE_POWERS]
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Model space size")
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.minorticks_off()
        ax.spines[["top", "right"]].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    handles, names = ax_rmse.get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="upper center",
        ncol=len(names),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "results_dir", type=Path, help="directory with the stored archives"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="figure file (must not exist)"
    )
    parser.add_argument(
        "--n-boot", type=int, default=500, help="bootstrap repetitions for rRMSE"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="bootstrap and random-band seed"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    fig = plot_scaling(
        rrmse_summary(args.results_dir, n_boot=args.n_boot),
        rksd_entries(args.results_dir),
        calibration_entries(args.results_dir),
        random_calibration_band(seed=args.seed),
    )
    fig.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
