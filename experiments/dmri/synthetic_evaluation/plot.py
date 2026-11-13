"""Plot Figure 6c posterior-quality violins grouped by model configuration."""

# ruff: noqa: B905

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.dmri._shared.plotting import (
    LABELS_MASK,
    build_label_masks,
    expand_label_masks,
)

STYLE = Path(__file__).parents[1] / "_shared/paper.mplstyle"
GROUPS = (
    ("synthetic", "syn.", "#608fb9"),
    ("real_selected", "real (selected)", "#8fb960"),
    ("real_fixed", "real (fixed)", "#b9608f"),
)


def normalize_by_prior(
    ksd: list[np.ndarray], ksd_prior: list[np.ndarray]
) -> list[np.ndarray]:
    normalized = []
    for values, prior_values in zip(ksd, ksd_prior):
        values = np.asarray(values)
        prior_values = np.asarray(prior_values)
        if values.size > 0:
            values = values[np.isfinite(values)]
        if prior_values.size > 0:
            prior_values = prior_values[np.isfinite(prior_values)]
        if len(values) == 0 or len(prior_values) == 0:
            normalized.append(np.array([], dtype=np.float32))
            continue
        denom = np.mean(prior_values)
        if not np.isfinite(denom) or denom == 0.0:
            normalized.append(np.array([], dtype=np.float32))
            continue
        normalized.append(values / denom)
    return normalized


def trim_values(values: list[np.ndarray], quantile: float | None) -> list[np.ndarray]:
    trimmed = []
    for arr in values:
        arr = np.asarray(arr)
        if arr.size > 0:
            arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            trimmed.append(arr)
            continue
        if quantile is None:
            trimmed.append(arr)
            continue
        cutoff = np.quantile(arr, quantile)
        trimmed.append(arr[arr < cutoff])
    return trimmed


def group_by_mask_list(
    values: np.ndarray,
    masks: np.ndarray,
    mask_list: np.ndarray,
) -> list[np.ndarray]:
    grouped = []
    mask_arr = np.asarray(masks)
    for mask in mask_list:
        subset = np.all(mask_arr == mask, axis=1)
        grouped.append(values[subset])
    return grouped


def style_violin(violin, color: str) -> None:
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.4)
        body.set_alpha(0.7)
    if "cmins" in violin:
        violin["cmins"].set_color(color)
    if "cmaxes" in violin:
        violin["cmaxes"].set_color(color)
    for key in ("cbars", "cmeans"):
        if key in violin:
            violin[key].set_color("black")
            violin[key].set_linewidth(0.4)


def plot_violin(
    groups: list[list[np.ndarray]],
    labels: list[str],
    out_path: Path | None,
    title: str,
    ylabel: str,
    *,
    group_labels: list[str] | None = None,
    colors: list[str] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = (4.2, 1.4),
    violin_width: float | None = None,
    group_span: float = 0.4,
    rotation: float = 35.0,
    xlabel: str | None = "Model configuration",
    xtick_fontsize: int = 9,
    ylabel_fontsize: int = 9,
    legend: bool = True,
    legend_loc: str = "upper right",
    legend_kwargs: dict[str, object] | None = None,
    tight_layout: bool = True,
    show: bool = False,
    save_kwargs: dict[str, object] | None = None,
) -> tuple[plt.Figure, plt.Axes] | None:
    if not groups:
        print(f"No data to plot for {out_path}")
        return

    group_arrays = [[np.asarray(values) for values in group] for group in groups]

    if not labels:
        print(f"No data to plot for {out_path}")
        return

    for group in group_arrays:
        if len(group) != len(labels):
            raise ValueError(
                f"Group length {len(group)} does not match labels length {len(labels)}."
            )

    num_groups = len(group_arrays)
    if group_labels is None:
        group_labels = [f"Group {idx + 1}" for idx in range(num_groups)]
    if len(group_labels) != num_groups:
        raise ValueError("Number of group_labels must match groups.")

    if colors is None:
        default_colors = [
            "#4C72B0",
            "#DD8452",
            "#55A868",
            "#C44E52",
            "#8172B2",
            "#64B5CD",
        ]
        colors = [
            default_colors[idx % len(default_colors)] for idx in range(num_groups)
        ]
    if len(colors) != num_groups:
        raise ValueError("Number of colors must match groups.")

    positions = np.arange(len(labels))
    if ax is None:
        if figsize is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if num_groups == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-group_span / 2, group_span / 2, num_groups)

    if violin_width is None:
        if num_groups == 1:
            violin_width = 0.35
        else:
            spacing = group_span / (num_groups - 1)
            violin_width = min(0.35, spacing * 0.9)

    handles = []
    for group, label, color, offset in zip(group_arrays, group_labels, colors, offsets):
        keep_idx = [idx for idx, values in enumerate(group) if len(values) > 0]
        if not keep_idx:
            print(f"No data selected for {label} in {out_path}")
            continue
        group_positions = positions[keep_idx] + offset
        violin = ax.violinplot(
            [group[idx] for idx in keep_idx],
            positions=group_positions,
            widths=violin_width,
        )
        style_violin(violin, color)
        handles.append((label, color))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=xtick_fontsize, rotation=rotation, ha="right")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    if title:
        ax.set_title(title)

    from matplotlib.patches import Patch

    if handles and legend:
        ax.legend(
            handles=[
                Patch(facecolor=color, edgecolor="black", label=label)
                for label, color in handles
            ],
            frameon=False,
            loc=legend_loc,
            **(legend_kwargs or {}),
        )
    if tight_layout:
        fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, **(save_kwargs or {}))
    if show:
        plt.show()
    # plt.close(fig)
    return fig, ax


def preprocess(data):
    label_masks = build_label_masks(LABELS_MASK, data["mask_support"].shape[1])
    ksds_short = data["ksd"]
    ksds_short_prior = data["ksd_prior"]
    masks_short = data["mask_support"]
    mask_list, labels = expand_label_masks(label_masks)
    ksd_short_model = group_by_mask_list(ksds_short, masks_short, mask_list)
    ksd_short_prior_model = group_by_mask_list(ksds_short_prior, masks_short, mask_list)
    ksd_short_ratio = normalize_by_prior(ksd_short_model, ksd_short_prior_model)
    ksd_short_ratio = trim_values(ksd_short_ratio, 0.99)
    return ksd_short_ratio, labels


def read_input(path, kernel):
    """Read mask-grouped NPZ data or map saved per-trial diagnostic fields."""
    metadata = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "declared_kernel": kernel,
    }
    with np.load(path, allow_pickle=False) as archive:
        if {"mask_support", "ksd", "ksd_prior"} <= set(archive.files):
            data = {
                name: archive[name] for name in ("mask_support", "ksd", "ksd_prior")
            }
            metadata["schema"] = "author per-mask NPZ"
        else:
            data = {
                name: archive[field]
                for name, field in (
                    ("mask_support", "model_mask"),
                    ("ksd", "posterior_ksd"),
                    ("ksd_prior", "prior_ksd"),
                )
            }
            metadata["schema"] = "saved per-trial diagnostic; field names mapped only"
            protocol_path = path.parent.parent / "protocol.json"
            if not protocol_path.is_file():
                raise FileNotFoundError(
                    "Per-trial diagnostics require their enclosing protocol.json"
                )
            protocol = json.loads(protocol_path.read_text())
            recorded_kernel = protocol["ksd_kernel"].get("variant")
            if recorded_kernel is not None and recorded_kernel != kernel:
                raise ValueError("Declared KSD kernel differs from the saved protocol")
            metadata["recorded_kernel"] = recorded_kernel
            metadata["kernel_label_source"] = (
                "saved protocol"
                if recorded_kernel is not None
                else "caller declaration; legacy protocol has no variant field"
            )
            summary_path = path.with_name("summary.json")
            summary = json.loads(summary_path.read_text())
            if summary["samples_sha256"] != metadata["sha256"]:
                raise ValueError("Input hash differs from the saved summary")
            metadata.update({
                "protocol_path": str(protocol_path.resolve()),
                "protocol_sha256": hashlib.sha256(
                    protocol_path.read_bytes()
                ).hexdigest(),
                "checkpoint": protocol["checkpoint"],
                "arguments": protocol["arguments"],
            })
    masks = data["mask_support"]
    if masks.ndim != 2 or masks.shape[1] != 5 or not np.isin(masks, [0, 1]).all():
        raise ValueError("Expected five-bit B3S mask support")
    if any(
        values.ndim not in (1, 2) or len(values) != len(masks)
        for values in (data["ksd"], data["ksd_prior"])
    ):
        raise ValueError("KSD rows must correspond to mask-support rows")
    return data, metadata


def plot_inputs(inputs, measurements, kernel="frozen"):
    """ksd_fig.ipynb cells 7/9, allowing absent real-data groups explicitly."""
    groups, labels, group_labels, colors = [], None, [], []
    for key, label, color in GROUPS:
        if key not in inputs:
            continue
        values, current_labels = preprocess(inputs[key])
        # Preserve the notebook's selection exactly; it does not pool masks.
        current_labels = current_labels[:2] + current_labels[6:]
        values = values[:2] + values[6:]
        if labels is not None and labels != current_labels:
            raise ValueError("Group model labels differ")
        labels = current_labels
        groups.append(values)
        group_labels.append(label)
        colors.append(color)
    if not groups:
        raise ValueError("No KSD groups supplied")
    ylabel = (
        f"rKSD (n={measurements})"
        if kernel == "frozen"
        else f"Corrected rKSD (n={measurements})"
    )
    fig, ax = plot_violin(
        groups,
        labels,
        None,
        None,
        ylabel,
        group_labels=group_labels,
        colors=colors,
        figsize=(3.3, 1.8),
        violin_width=0.3,
        group_span=0.6,
        legend=True,
        legend_kwargs=dict(
            ncol=3, bbox_to_anchor=(1.0, 1.3), handlelength=0.3, columnspacing=0.5
        ),
    )
    ax.set_yscale("log")
    return fig, groups, labels, group_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--real-selected", type=Path)
    parser.add_argument("--real-fixed", type=Path)
    parser.add_argument(
        "--measurements",
        type=int,
        required=True,
        help="Acquisition count for the axis label; 105 for the paper main panel",
    )
    parser.add_argument("--kernel", choices=("frozen", "corrected"), default="frozen")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.measurements < 1:
        parser.error("--measurements must be positive")
    inputs, sources = {}, {}
    for key, _, _ in GROUPS:
        path = getattr(args, key)
        if path is not None:
            inputs[key], sources[key] = read_input(path, args.kernel)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    with plt.style.context(STYLE):
        fig, groups, labels, group_labels = plot_inputs(
            inputs, args.measurements, args.kernel
        )
        for suffix in ("svg", "png", "pdf"):
            fig.savefig(
                output / f"ksd_violin.{suffix}",
                dpi=300,
                bbox_inches="tight",
                transparent=True,
            )
        plt.close(fig)
    np.savez_compressed(
        output / "plot_data.npz",
        **{
            f"group_{i}_mask_{j}": values
            for i, group in enumerate(groups)
            for j, values in enumerate(group)
        },
    )
    masks, _ = expand_label_masks(build_label_masks(LABELS_MASK, 5))
    selected = np.concatenate((masks[:2], masks[6:]))
    manifest = {
        "script_sha256": {
            str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).parents[1] / "_shared/plotting.py",
                STYLE,
            )
        },
        "inputs": sources,
        "kernel": args.kernel,
        "measurements": args.measurements,
        "group_labels": group_labels,
        "mask_labels": labels,
        "mask_support": selected.astype(int).tolist(),
        "display_counts": [[len(values) for values in group] for group in groups],
        "all_paper_groups_present": len(groups) == 3,
        "plot_data_sha256": hashlib.sha256(
            (output / "plot_data.npz").read_bytes()
        ).hexdigest(),
        "definition": "Original preprocess: per-exact-mask prior-mean normalization, strict q99 trimming, [:2]+[6:] selection; ordinary matplotlib violinplot followed by log y-axis.",
        "scope": "Original plot code applied to the supplied results. Missing real-data groups are omitted; historical checkpoint identity is not inferred.",
    }
    (output / "plot.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
