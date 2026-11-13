"""Frozen condition order, random keys, and calibration-curve aggregation."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.dmri._shared.metrics import mask_label

LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0)
MASK_CODES = (
    "10001",
    "01001",
    "00101",
    "00011",
    "01101",
    "00111",
    "01011",
    "01111",
    "11001",
    "10101",
    "10011",
    "11101",
    "10111",
    "11011",
    "11111",
)
RIGHT_PANEL_INDICES = (0, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14)


def condition_roots(mode, sequence, condition_index):
    """Recover the frozen coverage launcher's sequential outer split positions."""
    if mode == "model":
        if sequence != 0 or condition_index not in range(11):
            raise ValueError("Model coverage uses the eleven SHORT conditions")
        skipped = 2 * condition_index
    elif mode == "theta":
        if sequence not in (0, 1) or condition_index not in range(1, 16):
            raise ValueError("Parameter coverage uses fifteen fixed masks per scheme")
        skipped = 22 + 52 * sequence + 2 * (condition_index - 1)
    else:
        raise ValueError("Unknown calibration mode")
    key = jax.random.PRNGKey(0)
    for _ in range(skipped):
        key, _ = jax.random.split(key)
    key, data_key = jax.random.split(key)
    _, pit_key = jax.random.split(key)
    return data_key, pit_key


def make_batch_function(
    simulator,
    sbc,
    *,
    sequence,
    condition_index=0,
    hyper=None,
    mask=None,
    trials=100000,
    measure=None,
):
    mode = "model" if mask is None else "theta"
    data_root, sbc_root = condition_roots(mode, sequence, condition_index)
    key_tables = [jax.random.split(root, trials) for root in (data_root, sbc_root)]
    generate = jax.jit(
        jax.vmap(
            lambda key: simulator(
                key,
                mask_prior_hyperparameter=None if hyper is None else jnp.array([hyper]),
                model_mask=None if mask is None else jnp.asarray(mask, dtype=bool),
            )
        )
    )
    if measure is None:
        measure = jax.jit(jax.vmap(sbc))

    def compute(indices):
        # Only the final batch can extend past N; padded rows are discarded.
        keys = [table[jnp.minimum(indices, trials - 1)] for table in key_tables]
        data = generate(keys[0])
        return {
            **measure(data, keys[1]),
            "trial_index": indices,
            "model_mask": data["model_mask"],
            "theta_true": data["theta"],
            "mask_prior": data["mask_prior"],
            "num_acquisitions": jnp.full(
                indices.shape, data["acq"].bvals.shape[-1], dtype=jnp.int32
            ),
        }

    return compute


def conditions(mode):
    if mode == "model":
        return [
            {
                "name": "short_overall",
                "sequence": 0,
                "condition_index": 0,
                "hyper": None,
                "mask": None,
            }
        ] + [
            {
                "name": f"short_lambda_{value:.1f}",
                "sequence": 0,
                "condition_index": index + 1,
                "hyper": value,
                "mask": None,
            }
            for index, value in enumerate(LAMBDAS)
        ]
    if mode != "theta":
        raise ValueError("Unknown calibration mode")
    return [
        {
            "name": f"{scheme}_mask_{code}",
            "sequence": sequence,
            "condition_index": index + 1,
            "hyper": None,
            "mask": [int(bit) for bit in code],
            "mask_index": index,
        }
        for sequence, scheme in enumerate(("short", "long"))
        for index, code in enumerate(MASK_CODES)
    ]


def aggregate(mode, summaries):
    cases = conditions(mode)
    if set(summaries) != {case["name"] for case in cases}:
        raise ValueError("Calibration aggregation requires every declared condition")
    field = "model_pit" if mode == "model" else "theta_sum_pit"
    grid = np.asarray(summaries[cases[0]["name"]][field]["grid"])
    curves = []
    for case in cases:
        value = summaries[case["name"]][field]
        if not np.array_equal(value["grid"], grid):
            raise ValueError("Calibration grids differ between conditions")
        curves.append(np.asarray(value["cdf"]))
    curves = np.stack(curves)
    if mode == "model":
        curve = curves.mean(0)
        details = {
            "recipe": "Equal mean of SHORT overall plus ten fixed-lambda model coverage curves",
            "conditions": [case["name"] for case in cases],
        }
    else:
        per_scheme = curves.reshape(2, 15, -1)
        curve = per_scheme.mean(0).mean(0)
        details = {
            "recipe": "Average short and long masked-parameter-sum coverage per fixed model, then average the fifteen model curves equally",
            "mask_codes": list(MASK_CODES),
            "right_panel_mask_indices": list(RIGHT_PANEL_INDICES),
            "per_mask": [
                {
                    "mask_code": code,
                    "mask_index": index,
                    "short_calibration_error": float(
                        np.abs(per_scheme[0, index] - grid).mean()
                    ),
                    "long_calibration_error": float(
                        np.abs(per_scheme[1, index] - grid).mean()
                    ),
                    "average_curve_calibration_error": float(
                        np.abs(per_scheme[:, index].mean(0) - grid).mean()
                    ),
                }
                for index, code in enumerate(MASK_CODES)
            ],
        }
    return {
        "statistic": field,
        "grid": grid.tolist(),
        "cdf": curve.tolist(),
        "calibration_error": float(np.abs(curve - grid).mean()),
        "observed_acquisitions_by_condition": {
            case["name"]: summaries[case["name"]].get("observed_acquisition_counts", {})
            for case in cases
        },
        **details,
    }


def aggregate_plot(summary, output, mode):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        1 if mode == "model" else 2,
        figsize=(4, 4) if mode == "model" else (11, 4),
        squeeze=False,
        constrained_layout=True,
    )
    axis = axes[0, 0]
    axis.plot(
        summary["grid"],
        summary["cdf"],
        color="#287c8e",
        label=f"CE={summary['calibration_error']:.5f}",
    )
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Nominal",
        ylabel="Empirical",
        title="Model" if mode == "model" else "Parameter sum",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(frameon=False)
    if mode == "theta":
        axis = axes[0, 1]
        selected = [summary["per_mask"][index] for index in RIGHT_PANEL_INDICES]
        labels = [
            mask_label([int(bit) for bit in row["mask_code"]]) for row in selected
        ]
        for scheme, color in (("short", "#21b987"), ("long", "#22694b")):
            axis.step(
                range(len(selected)),
                [row[f"{scheme}_calibration_error"] for row in selected],
                where="mid",
                color=color,
                label=scheme,
            )
        axis.set_xticks(range(len(selected)), labels, rotation=65, ha="right")
        axis.set(ylabel="Calibration error", title="Parameters by fixed model")
        axis.legend(frameon=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        target = output / f"figure6b-{mode}.{suffix}"
        if not target.exists():
            fig.savefig(target, dpi=180)
    plt.close(fig)
