"""Shared dMRI mask labels and diagnostic plots from existing evaluations."""

import jax.numpy as jnp
import numpy as np

LABELS_MASK = {
    "B": jnp.array([[True, False, False, False, True]]),
    "S": jnp.array([
        [False, True, False, False, True],
        [False, False, True, False, True],
        [False, False, False, True, True],
    ]),
    "S2": jnp.array([
        [False, True, True, False, True],
        [False, False, True, True, True],
        [False, True, False, True, True],
    ]),
    "S3": jnp.array([[False, True, True, True, True]]),
    "BS1": jnp.array([
        [True, True, False, False, True],
        [True, False, True, False, True],
        [True, False, False, True, True],
    ]),
    "BS2": jnp.array([
        [True, True, True, False, True],
        [True, True, False, True, True],
        [True, False, True, True, True],
    ]),
    "BS3": jnp.array([[True, True, True, True, True]]),
}


def build_label_masks(label_map: dict, expected_dim: int) -> dict[str, np.ndarray]:
    label_masks = {}
    for label, mask_arr in label_map.items():
        arr = np.asarray(mask_arr, dtype=bool)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != expected_dim:
            print(
                f"Skipping label {label}: expected mask length {expected_dim}, got {arr.shape[1]}"
            )
            continue
        label_masks[label] = arr
    return label_masks


def expand_label_masks(
    label_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    masks = []
    labels = []
    for label, mask_list in label_masks.items():
        for mask in np.asarray(mask_list):
            masks.append(mask)
            labels.append(label)
    if not masks:
        return np.zeros((0, 0), dtype=bool), []
    return np.stack(masks, axis=0), labels


def plots(arrays, summary, destination, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = []
    fields = [
        (k, t)
        for k, t in [
            ("model_pit", "Model SBC"),
            ("theta_sum_pit", "Parameter sum SBC (frozen script)"),
            ("theta_pit", "Parameter-wise SBC (paper definition)"),
        ]
        if summary.get(k)
    ]
    if fields:
        fig, axes = plt.subplots(
            1, len(fields), figsize=(5 * len(fields), 4), squeeze=False
        )
        for ax, (field, title) in zip(axes[0], fields, strict=True):
            stat = summary[field]
            ax.plot(
                stat["grid"],
                stat["cdf"],
                color="#1976a8",
                label=f"CE={stat['calibration_error']:.4f}",
            )
            ax.plot([0, 1], [0, 1], "--", color="0.45", label="Uniform reference")
            ax.set(
                xlabel="Nominal probability",
                ylabel="Empirical CDF",
                title=title,
                xlim=(0, 1),
                ylim=(0, 1),
            )
            ax.legend(fontsize=8)
        fig.suptitle(label)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = destination / f"sbc.{suffix}"
            fig.savefig(path, dpi=150)
            result.append(path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4))
        rows = [r for r in summary["per_mask"] if r.get("theta_sum_pit")]
        ax.bar(
            range(len(rows)),
            [r["theta_sum_pit"]["calibration_error"] for r in rows],
            color="#1976a8",
        )
        ax.set_xticks(
            range(len(rows)), [r["label"] for r in rows], rotation=65, ha="right"
        )
        ax.set(
            ylabel="Calibration error", title=f"Conditional parameter-sum SBC: {label}"
        )
        fig.tight_layout()
        path = destination / "sbc_per_mask.png"
        fig.savefig(path, dpi=150)
        result.append(path)
        plt.close(fig)

    if "posterior_ksd" in arrays:
        fig, ax = plt.subplots(figsize=(11, 4))
        masks = arrays["model_mask"].astype(bool)
        for index, row in enumerate(summary["per_mask"]):
            selected = np.all(masks == np.asarray(row["mask"], bool), axis=1)
            ratio = arrays["posterior_ksd"][selected] / row["prior_ksd_mean"]
            trimmed = ratio[ratio < row["plot_trim_cutoff"]]
            if len(trimmed) > 1 and np.ptp(trimmed) > 0:
                ax.violinplot([trimmed], positions=[index], showextrema=True)
            else:
                ax.scatter(np.full(len(trimmed), index), trimmed, s=8)
        ax.axhline(1, linestyle="--", color="0.45", label="Prior reference")
        ax.set_xticks(
            range(len(summary["per_mask"])),
            [r["label"] for r in summary["per_mask"]],
            rotation=65,
            ha="right",
        )
        ax.set(ylabel="rKSD", title=f"Synthetic posterior rKSD: {label}")
        ax.legend()
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = destination / f"rksd.{suffix}"
            fig.savefig(path, dpi=150)
            result.append(path)
        plt.close(fig)
    return result
