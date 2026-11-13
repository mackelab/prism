"""Plot the illustration panels from ``illustrate`` output.

Writes ``model_fits.svg`` (Figure 2 left), ``num_components.svg`` (Figure 2 right),
``model_space.svg`` and ``parameter_posterior.svg`` (Figure 4a, Figure A.1)
and ``equations.txt`` into ``--output-dir``.

Example: python -m experiments.symbolic.illustrations.plot \
    results/symbolic50_illustrative/illustrations --output-dir results/symbolic50_illustrative/panels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

COLORS = {0.5: "#2596be", 0.1: "#be4d25"}


def plot_model_fits(a):
    fig, ax = plt.subplots(figsize=(1.8, 2.2))
    grid = a["grid_eval"]
    for lam in (0.5, 0.1):
        lo = gaussian_filter1d(a[f"predictive_lo_{lam}"], 2)
        hi = gaussian_filter1d(a[f"predictive_hi_{lam}"], 2)
        ax.fill_between(grid, lo, hi, color=COLORS[lam], alpha=0.15)
        ax.plot(grid, a[f"selected_function_{lam}"], color=COLORS[lam], lw=1.5)
    ax.plot(grid, a["true_function"], color="black", lw=1.5)
    ax.scatter(
        a["grid_obs"],
        a["x_observed"],
        color="black",
        s=0.3,
        marker="x",
        alpha=0.4,
        zorder=5,
    )
    ax.set_ylim(0, 22)
    ax.set_yticks([0, 5, 10, 15, 20], ["0", "", "", "", "20"])
    ax.set_xticks([0, 5, 10, 15], ["0", "", "", "15"])
    ax.set_xlabel("Input $x$", labelpad=-10)
    ax.set_ylabel("Output $f(x)$", labelpad=-13)
    ax.set_title("Global posterior pred.\n($\\lambda=0.5$, $\\lambda=0.1$)", pad=15)
    fig.tight_layout()
    return fig


def plot_num_components(a, *, show_true=True):
    fig, ax = plt.subplots(figsize=(1.6, 1.5))
    lam = np.linspace(0.0, 1.0, a["component_quantiles"].shape[1])
    median, lo, hi = (gaussian_filter1d(q, 2) for q in a["component_quantiles"])
    ax.plot(lam, median, color="black")
    ax.fill_between(lam, lo, hi, color="gray", alpha=0.5)
    if show_true:
        ax.hlines(a["true_components"], 0, 1, color="red")
    ax.set_ylabel("# components")
    ax.set_ylim(0, 51)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0.0", "", "1.0"])
    ax.set_xlabel("Model complexity $\\lambda$", labelpad=2)
    return fig


def plot_model_space(a):
    space = a["model_space"]
    count = space.shape[1]
    fig, (top, main) = plt.subplots(
        2, 1, figsize=(2.2, 1.5), gridspec_kw={"height_ratios": [1, 5]}
    )
    top.imshow(a["true_mask"][None, :], aspect="auto", cmap="gray")
    top.set_ylabel("True")
    top.set_xlim(0, count)
    top.set_xticks([])
    top.set_yticks([])
    image = main.imshow(space, aspect="auto", origin="lower", cmap="Greys_r")
    main.set_ylabel("$\\lambda$", labelpad=-15)
    main.set_yticks(np.linspace(0, space.shape[0] - 1, 5), ["0.0", "", "", "", "1.0"])
    main.set_xlabel("Model component index")
    colorbar = fig.colorbar(image, ax=[top, main])
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0.", "", "", "", "1.0"])
    colorbar.set_label("Probability", labelpad=-10)
    return fig


def plot_parameter_posterior(samples, bins=100):
    dims = samples.shape[1]
    fig, axes = plt.subplots(
        dims,
        dims,
        figsize=(1.5 * max(dims / 3, 1), 1.5 * max(dims / 3, 1)),
        squeeze=False,
    )
    low, high = np.quantile(samples, 0.001, axis=0), np.quantile(samples, 0.999, axis=0)
    for row in range(dims):
        for col in range(dims):
            ax = axes[row, col]
            if col < row:
                ax.axis("off")
                continue
            if row == col:
                ax.hist(
                    samples[:, row],
                    bins=bins * 2,
                    range=(low[row], high[row]),
                    color="black",
                )
            else:
                ax.hist2d(
                    samples[:, col],
                    samples[:, row],
                    bins=bins,
                    cmap="afmhot",
                    range=[(low[col], high[col]), (low[row], high[row])],
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"$\\theta_{{{col}}}$", fontsize=6)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="output directory of illustrate")
    parser.add_argument("--output-dir", type=Path, required=True, help="must not exist")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    arrays = dict(np.load(args.directory / "arrays.npz"))
    summary = json.loads((args.directory / "summary.json").read_text())
    args.output_dir.mkdir(parents=True)
    panels = {
        "model_fits.svg": plot_model_fits(arrays),
        "num_components.svg": plot_num_components(arrays),
        "model_space.svg": plot_model_space(arrays),
        "parameter_posterior.svg": plot_parameter_posterior(
            arrays["parameter_posterior"]
        ),
    }
    for name, fig in panels.items():
        fig.savefig(args.output_dir / name, bbox_inches="tight")
        plt.close(fig)
    lines = [
        f"true ({summary['true_equation']['components']} components): "
        f"{summary['true_equation']['latex']}"
    ]
    for lam, info in summary["selected"].items():
        lines.append(f"lambda={lam} ({info['components']} components): {info['latex']}")
    (args.output_dir / "equations.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
