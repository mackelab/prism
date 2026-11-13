"""Figure A.3: confusion matrix with the support masks, ordered by similarity.

Example: python -m experiments.symbolic.classification.plot results/symbolic50/classification \
    --output results/symbolic50/confusion.svg

The argument is the stem written by ``evaluate --save-arrays``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import similarity_order  # noqa: E402


def plot_confusion(
    matrix, masks, class_counts, title="Confusion matrix for model selection"
):
    order = similarity_order(masks, class_counts)
    bits = np.asarray(masks).astype(int)[order]
    matrix = np.asarray(matrix)[np.ix_(order, order)]
    classes, components = bits.shape
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(3.0, 3.0),
        gridspec_kw={"width_ratios": [0.3, 3], "height_ratios": [3, 0.3]},
        sharex="col",
        sharey="row",
    )
    left, center, empty, bottom = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    center.imshow(
        matrix,
        cmap="bone_r",
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=(-0.5, classes - 0.5, classes - 0.5, -0.5),
    )
    left.imshow(
        bits,
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=(-0.5, components - 0.5, classes - 0.5, -0.5),
    )
    bottom.imshow(
        bits.T,
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=(-0.5, classes - 0.5, components - 0.5, -0.5),
    )
    for axis in (left, center, bottom):
        axis.set_xticks([])
        axis.set_yticks([])
    bottom.set_xlabel("Model subset (as mask)")
    empty.axis("off")
    fig.suptitle(title)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stem", type=Path, help="path stem of the evaluation arrays")
    parser.add_argument(
        "--output", type=Path, required=True, help="figure file (must not exist)"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    stem = str(args.stem)
    fig = plot_confusion(
        np.load(f"{stem}_matrix.npy"),
        np.load(f"{stem}_masks.npy"),
        np.load(f"{stem}_class_counts.npy"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
