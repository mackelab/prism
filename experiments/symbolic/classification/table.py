"""Collect per-K metrics JSON files into the Table 1 layout.

Example: python -m experiments.symbolic.classification.table \
    15=results/symbolic15/classification.json 30=results/symbolic30/classification.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = (
    ("Accuracy (Top-1)", "top1_micro"),
    ("Accuracy (Top-5), per sample", "top5_micro"),
    ("Accuracy (Top-5), per class", "top5_macro"),
    ("Macro precision", "macro_precision"),
    ("Macro recall", "macro_recall"),
    ("Macro F1", "macro_f1"),
)


def markdown_table(results: dict[str, dict]) -> str:
    """``{"15": metrics, ...}`` to a markdown table with three decimals."""
    sizes = list(results)
    lines = [
        "| Metric | " + " | ".join(sizes) + " |",
        "| --- |" + " ---: |" * len(sizes),
    ]
    for label, key in ROWS:
        lines.append(
            f"| {label} | " + " | ".join(f"{results[k][key]:.3f}" for k in sizes) + " |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("entries", nargs="+", metavar="K=metrics.json")
    parser.add_argument(
        "--output", type=Path, help="write the table as markdown (must not exist)"
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    results = {}
    for entry in args.entries:
        size, _, path = entry.partition("=")
        if not path:
            raise ValueError(f"Expected K=path, got {entry!r}")
        results[size] = json.loads(Path(path).read_text())
    table = markdown_table(results)
    print(table)
    if args.output is not None:
        args.output.write_text(table + "\n")
    return table


if __name__ == "__main__":
    main()
