"""Plot Figure 6b calibration panels from NPZ files or condition-wise results."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.dmri._shared.plotting import (
    LABELS_MASK,
    build_label_masks,
    expand_label_masks,
)
from prism.utils.slurm import add_slurm_arguments, submit

STYLE = Path(__file__).parents[1] / "_shared/paper.mplstyle"


def plot_model(alpha_s, coverage_s):
    """cov_fig.ipynb cell 5, with explicit inputs and a returned figure."""
    fig = plt.figure(figsize=(0.65, 0.65))
    plt.plot(alpha_s, coverage_s.mean(0), color="black")
    plt.xlabel("Nominal", labelpad=-2.5)
    plt.ylabel("Empirical", labelpad=-1.5)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])
    return fig


def plot_theta(pits_short_theta, pits_long_theta):
    """cov_fig.ipynb cells 10-11; retain the two-stage equal averaging."""
    alpha_t = pits_long_theta["alpha"]
    coverage_t = pits_long_theta["theta_mask_coverage"]
    coverage_t_s = pits_short_theta["theta_mask_coverage"]
    coverage_t = (coverage_t + coverage_t_s) / 2
    fig = plt.figure(figsize=(0.65, 0.65))
    plt.plot(alpha_t, coverage_t.mean(0), color="black")
    plt.xlabel("Nominal", labelpad=-2.5)
    plt.ylabel("Empirical", labelpad=-1.5)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])
    return fig


def plot_theta_errors(short, long, band=None, curve_labels=("HCP-like", "UKB-like")):
    """cov_fig.ipynb cell 9; labels retain the notebook defaults explicitly."""
    errors_theta_short = np.asarray(short["theta_mask_errors"])
    errors_theta_long = np.asarray(long["theta_mask_errors"])
    labels_theta = [str(label) for label in np.atleast_1d(short["theta_mask_labels"])]
    color1, color2 = "#1b7a52", "#2ecf8b"
    fig, axes = plt.subplots(1, 1, figsize=(1.5, 0.65))
    axes = [axes]
    subset = [0, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14]
    subset_positions = np.arange(len(subset))
    subset_labels = [labels_theta[i] for i in subset]
    axes[0].step(
        subset_positions,
        errors_theta_short[subset],
        color=color1,
        where="mid",
        label=curve_labels[0],
    )
    axes[0].step(
        subset_positions,
        errors_theta_long[subset],
        color=color2,
        where="mid",
        label=curve_labels[1],
    )
    if band is not None:
        lower_t, upper_t = band
        axes[0].fill_between(
            subset_positions, lower_t, upper_t, color="grey", alpha=0.1
        )
    axes[0].set_xticks(subset_positions)
    axes[0].set_xticklabels(subset_labels, rotation=90)
    axes[0].set_title("Calibration error (parameters)")
    fig.legend(
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.55, 0.99),
        handles=axes[0].get_lines(),
        handlelength=0.3,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    return fig


def random_errors(
    key, counts, *, alpha_count=101, repetitions=100_000, batch_size=1000
):
    """Notebook cell 6; counts and Monte Carlo workload are explicit."""

    def best_calibrated_errors(key):
        u = jax.random.uniform(key, shape=(counts,))
        alpha = jnp.linspace(0, 1, alpha_count)
        cov = jnp.quantile(u, alpha)
        error = jnp.abs(cov - alpha).mean()
        return error

    keys = jax.random.split(key, num=repetitions)
    errors = jax.lax.map(best_calibrated_errors, keys, batch_size=batch_size)
    q005 = jnp.quantile(errors, 0.005)
    q995 = jnp.quantile(errors, 0.995)
    return q005, q995


def load_condition_result(path, mode):
    """Map completed condition summaries to the plotting fields without resampling."""
    result = json.loads(path.read_text())
    protocol = result["protocol"]
    if result.get("status") != "completed" or protocol["mode"] != mode:
        raise ValueError(f"Expected a completed {mode} condition-wise evaluation")
    masks, labels = expand_label_masks(build_label_masks(LABELS_MASK, 5))
    groups = {}
    schemes = ("short",) if mode == "model" else ("short", "long")
    for sequence, scheme in enumerate(schemes):
        if mode == "model":
            cases = protocol["conditions"]
            if len(cases) != 11 or any(case["sequence"] != 0 for case in cases):
                raise ValueError(
                    "Model plotting expects the eleven short-acquisition conditions"
                )
        else:
            by_mask = {
                tuple(case["mask"]): case
                for case in protocol["conditions"]
                if case["sequence"] == sequence
            }
            if set(by_mask) != {tuple(mask.astype(int)) for mask in masks}:
                raise ValueError(
                    "Parameter plotting requires all fifteen fixed masks per acquisition"
                )
            cases = [by_mask[tuple(mask.astype(int))] for mask in masks]
        rows = [result["results"][case["name"]] for case in cases]
        field = "model_pit" if mode == "model" else "theta_sum_pit"
        grid = np.asarray(rows[0][field]["grid"])
        if any(not np.array_equal(row[field]["grid"], grid) for row in rows):
            raise ValueError("Condition grids differ")
        if any(
            row["completed_trials"] != protocol["trials_per_condition"] for row in rows
        ):
            raise ValueError("Condition trial counts differ from the saved protocol")
        curves = np.asarray([row[field]["cdf"] for row in rows])
        groups[scheme] = {"alpha": grid}
        if mode == "model":
            groups[scheme]["coverage"] = curves
        else:
            groups[scheme].update({
                "theta_mask_coverage": curves,
                "theta_mask_errors": np.asarray([
                    row[field]["calibration_error"] for row in rows
                ]),
                "theta_mask_labels": np.asarray(labels),
                "theta_mask_support": masks,
                "theta_mask_counts": np.asarray([
                    row["completed_trials"] for row in rows
                ]),
            })
    return groups


def load_npz(path, theta=False):
    fields = (
        ("alpha", "theta_mask_coverage", "theta_mask_errors", "theta_mask_labels")
        if theta
        else ("alpha", "coverage")
    )
    with np.load(path, allow_pickle=False) as archive:
        data = {field: archive[field] for field in fields}
        for field in ("theta_mask_counts", "theta_mask_support"):
            if field in archive.files:
                data[field] = archive[field]
    return data


def validate_theta(short, long):
    masks, labels = expand_label_masks(build_label_masks(LABELS_MASK, 5))
    for data in (short, long):
        if list(data["theta_mask_labels"]) != labels:
            raise ValueError("Theta labels differ from the author's fifteen-mask order")
        if "theta_mask_support" in data and not np.array_equal(
            data["theta_mask_support"], masks
        ):
            raise ValueError("Theta mask order differs from the author's order")
        if data["theta_mask_coverage"].shape != (15, len(data["alpha"])) or data[
            "theta_mask_errors"
        ].shape != (15,):
            raise ValueError("Invalid parameter coverage/error dimensions")
        if (
            not np.isfinite(data["theta_mask_coverage"]).all()
            or not np.isfinite(data["theta_mask_errors"]).all()
        ):
            raise ValueError("Nonfinite parameter coverage or errors")
    if not np.array_equal(short["alpha"], long["alpha"]):
        raise ValueError("Short and long parameter grids differ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        help="Author mask-short NPZ or completed model results.json",
    )
    parser.add_argument(
        "--theta-result", type=Path, help="Completed condition-wise theta results.json"
    )
    parser.add_argument("--theta-short", type=Path)
    parser.add_argument("--theta-long", type=Path)
    parser.add_argument(
        "--theta-band", type=Path, help="JSON produced with --make-band"
    )
    parser.add_argument(
        "--curve-labels",
        nargs=2,
        default=["HCP-like", "UKB-like"],
        metavar=("SHORT_LABEL", "LONG_LABEL"),
    )
    parser.add_argument("--make-band", type=int, metavar="DATASETS_PER_CONDITION")
    parser.add_argument("--band-trials", type=int, default=100_000)
    parser.add_argument("--band-batch-size", type=int, default=1000)
    parser.add_argument("--band-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    add_slurm_arguments(parser)
    args = parser.parse_args(argv)
    inputs = [
        value
        for value in (
            args.model,
            args.theta_result,
            args.theta_short,
            args.theta_long,
            args.theta_band,
        )
        if value is not None
    ]
    if args.make_band is not None:
        if inputs or min(args.make_band, args.band_trials, args.band_batch_size) < 1:
            parser.error("--make-band requires positive counts and no plot inputs")
    elif (
        not inputs
        or bool(args.theta_short) != bool(args.theta_long)
        or (args.theta_result and args.theta_short)
        or (args.theta_band and not (args.theta_result or args.theta_short))
    ):
        parser.error(
            "Provide --model and/or either --theta-result or both --theta-short and --theta-long"
        )
    if args.slurm:
        if args.make_band is None:
            parser.error("--slurm is only needed for --make-band")
        return submit(args, argv, main)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "support_sha256": {
            str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (Path(__file__).parents[1] / "_shared/plotting.py", STYLE)
        },
        "inputs": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        "curve_labels": args.curve_labels,
        "figures": [],
    }
    if args.make_band is not None:
        lower, upper = random_errors(
            jax.random.PRNGKey(args.band_seed),
            args.make_band,
            repetitions=args.band_trials,
            batch_size=args.band_batch_size,
        )
        band = {
            "lower": float(lower),
            "upper": float(upper),
            "datasets_per_condition": args.make_band,
            "repetitions": args.band_trials,
            "seed": args.band_seed,
            "alpha_count": 101,
            "statistic": "cov_fig.ipynb cell 6: quantiles of Uniform draws, mean absolute deviation, 0.005/0.995 Monte Carlo bounds",
        }
        (output / "reference_band.json").write_text(json.dumps(band, indent=2) + "\n")
    else:
        figures = []
        with plt.style.context(STYLE):
            if args.model:
                model = (
                    load_condition_result(args.model, "model")["short"]
                    if args.model.suffix == ".json"
                    else load_npz(args.model)
                )
                if (
                    model["coverage"].ndim != 2
                    or model["coverage"].shape[1] != len(model["alpha"])
                    or not np.isfinite(model["coverage"]).all()
                ):
                    raise ValueError("Invalid model coverage arrays")
                figures.append((
                    "dmri_model_coverage",
                    plot_model(model["alpha"], model["coverage"]),
                ))
            if args.theta_result or args.theta_short:
                pair = (
                    load_condition_result(args.theta_result, "theta")
                    if args.theta_result
                    else {
                        "short": load_npz(args.theta_short, True),
                        "long": load_npz(args.theta_long, True),
                    }
                )
                short, long = pair["short"], pair["long"]
                validate_theta(short, long)
                bounds = None
                if args.theta_band:
                    band = json.loads(args.theta_band.read_text())
                    for data in (short, long):
                        if "theta_mask_counts" not in data or not np.all(
                            data["theta_mask_counts"] == band["datasets_per_condition"]
                        ):
                            raise ValueError(
                                "Reference-band count differs from the parameter evaluation"
                            )
                    bounds = (band["lower"], band["upper"])
                    if (
                        band["alpha_count"] != len(short["alpha"])
                        or not np.isfinite(bounds).all()
                        or not 0 <= bounds[0] <= bounds[1]
                    ):
                        raise ValueError("Invalid reference band/grid")
                    manifest["reference_band"] = band
                figures.extend([
                    ("dmri_theta_coverage", plot_theta(short, long)),
                    (
                        "coverage_calibration_error_subset",
                        plot_theta_errors(
                            short, long, bounds, tuple(args.curve_labels)
                        ),
                    ),
                ])
            for name, fig in figures:
                for suffix in ("svg", "png", "pdf"):
                    fig.savefig(
                        output / f"{name}.{suffix}",
                        dpi=300,
                        bbox_inches="tight",
                        transparent=True,
                    )
                    manifest["figures"].append(f"{name}.{suffix}")
                plt.close(fig)
    (output / "plot.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
