"""Figure A.4: ESS and run time (ms) against the number of diffusion steps.

Example: python -m experiments.dmri.diffusion_steps.plot results/b3s/steps_ess.npz \
    --output results/b3s/steps_ess.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def summarize(values):
    """Mean and 2.5%/97.5% percentiles over runs (axis 1)."""
    values = np.asarray(values)
    return (
        values.mean(1),
        np.percentile(values, 2.5, axis=1),
        np.percentile(values, 97.5, axis=1),
    )


def plot_steps(num_steps, ess, run_time_s):
    steps = np.asarray(num_steps)
    fig, (top, bottom) = plt.subplots(2, figsize=(3, 3))
    for ax, values in ((top, ess), (bottom, 1000 * np.asarray(run_time_s))):
        mean, low, high = summarize(values)
        ax.plot(steps, mean, color="black")
        ax.fill_between(steps, low, high, color="gray", alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(steps)
    top.set_ylabel("ESS")
    top.set_xticklabels([""] * len(steps))
    bottom.set_yscale("log")
    bottom.set_xticklabels([str(int(s)) for s in steps])
    bottom.set_xlabel("Number of diffusion steps")
    bottom.set_ylabel("Run time (ms)")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, help="npz written by run.py")
    parser.add_argument(
        "--output", type=Path, required=True, help="figure file (must not exist)"
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    data = np.load(args.results)
    fig = plot_steps(data["num_steps"], data["ess"], data["run_time_s"])
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
