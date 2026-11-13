"""Published symbolic weights (safetensors) as an alternative to a training run.

The evaluators normally read an Orbax training run. ``--preset`` loads the
converted EMA weights of ``manugloeck/prism-symbolic-pretrained`` instead, and
``--weights-dir`` a local copy laid out as ``config.yaml`` plus
``<which>/params_ema.safetensors``. A published preset carries no training
history, so checks that need one (completed steps, logs, training config) do not
apply; the provenance record says which weights were evaluated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def add_published_arguments(parser):
    """Add ``--preset`` and ``--weights-dir`` to an evaluator's parser."""
    parser.add_argument(
        "--preset",
        help="published preset of manugloeck/prism-symbolic-pretrained "
        "(e.g. symbolic15, symbolic_sbmi_10k) instead of a run directory",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="local copy of the preset directory (config.yaml and "
        "<which>/params_ema.safetensors); use together with --preset",
    )


def check_weights_source(parser, checkpoint, preset, weights_dir) -> None:
    """Require exactly one of a run directory or ``--preset``."""
    if (checkpoint is None) == (preset is None):
        parser.error("give exactly one of a run directory or --preset")
    if weights_dir is not None and preset is None:
        parser.error("--weights-dir needs --preset")


def load_published(preset: str, which: str, weights_dir: Path | None = None):
    """Load published EMA weights; return the bundle and a provenance record."""
    from prism.symbolic.pretrained import (
        DEFAULT_REPO_ID,
        WEIGHTS_FILE,
        load_pretrained_symbolic,
    )

    if weights_dir is None:
        from huggingface_hub import hf_hub_download

        bundle = load_pretrained_symbolic(preset=preset, which=which)
        weights = Path(
            hf_hub_download(DEFAULT_REPO_ID, f"{preset}/{which}/{WEIGHTS_FILE}")
        )
        source = {
            "repo_id": DEFAULT_REPO_ID,
            "path": f"{preset}/{which}/{WEIGHTS_FILE}",
        }
    else:
        bundle = load_pretrained_symbolic(path=weights_dir, which=which)
        weights = Path(weights_dir) / which / WEIGHTS_FILE
        source = {"local_dir": str(Path(weights_dir).resolve())}
    record = {
        "weights": "published safetensors (EMA)",
        "preset": preset,
        "which": which,
        "step": bundle.step,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "training_evidence": "not available for published weights",
        **source,
    }
    return bundle, record
