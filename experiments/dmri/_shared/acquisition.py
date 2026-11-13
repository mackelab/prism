"""Acquisition draws following PRISM Appendix B.2.3.

The paper specifies 105 and 295 acquisitions but does not publish the exact
typical templates. The frozen code supplies 105 and 297 entries. This adapter
uses the 105-entry template and derives 295 entries by removing the final two
b=0 entries from the b-sorted 297-entry template. All diffusion-weighted
measurements are retained (90 directions in each of three shells). The 295-entry
template is a reconstruction; ``protocol_metadata`` records this assumption,
the removed indices and template hashes.

Use Hydra targets ``experiments.dmri._shared.acquisition.random_ukb_acquisition``
and ``experiments.dmri._shared.acquisition.random_hcp_acquisition`` with
``_partial_: true`` in ``simulator.acquisitions``. No core registration is needed.
"""

from __future__ import annotations

import hashlib
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from prism.dmri.simulators.acquisition_scheme import (
    acquisition_scheme,
    bvals_typ,
    bvals_typ_large,
    bvecs_typ,
    bvecs_typ_large,
)

MAX_B_VALUE = 6000.0
TYPICAL_PROBABILITY = 0.5
ZERO_DROPOUT_PROBABILITY = 0.01


def _array_sha256(array: np.ndarray) -> str:
    """Hash shape and canonical float64 values independently of host endianness."""
    canonical = np.asarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@cache
def _typical_arrays(kind: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if kind == "ukb":
        bvals, bvecs = np.asarray(bvals_typ), np.asarray(bvecs_typ)
        expected_size = 105
        removed = ()
    elif kind == "hcp":
        bvals, bvecs = np.asarray(bvals_typ_large), np.asarray(bvecs_typ_large)
        expected_size = 297
        zero_indices = np.flatnonzero(bvals == 0)
        if len(zero_indices) != 27:
            raise ValueError(
                "The frozen HCP template must contain exactly 27 b=0 entries"
            )
        removed = tuple(int(index) for index in zero_indices[-2:])
    else:
        raise ValueError(f"Unknown acquisition template {kind!r}")
    if bvals.shape != (expected_size,) or bvecs.shape != (expected_size, 3):
        raise ValueError(f"Unexpected frozen {kind} template dimensions")
    norms = np.linalg.norm(bvecs, axis=-1)
    if (
        not np.isfinite(bvals).all()
        or not np.isfinite(norms).all()
        or np.any(norms == 0)
    ):
        raise ValueError(
            "Typical acquisition templates must contain finite nonzero directions"
        )
    # Normalize decimal-rounded directions without altering their orientations.
    bvecs = bvecs / norms[:, None]
    bvals = np.delete(bvals, removed, axis=0).astype(np.float32)
    bvecs = np.delete(bvecs, removed, axis=0).astype(np.float32)
    bvals.setflags(write=False)
    bvecs.setflags(write=False)
    return bvals, bvecs, removed


def protocol_metadata() -> dict:
    """Return JSON-serializable provenance for the acquisition interpretation."""
    templates = {}
    for kind, source_bvals, source_bvecs in (
        ("ukb", bvals_typ, bvecs_typ),
        ("hcp", bvals_typ_large, bvecs_typ_large),
    ):
        bvals, bvecs, removed = _typical_arrays(kind)
        shells, counts = np.unique(bvals, return_counts=True)
        templates[kind] = {
            "num_acquisitions": len(bvals),
            "source_num_acquisitions": len(source_bvals),
            "removed_sorted_source_indices": list(removed),
            "source_bvals_sha256": _array_sha256(source_bvals),
            "source_bvecs_sha256": _array_sha256(source_bvecs),
            "used_bvals_sha256": _array_sha256(bvals),
            "used_bvecs_sha256": _array_sha256(bvecs),
            "shell_counts": {
                str(int(shell)): int(count)
                for shell, count in zip(shells, counts, strict=True)
            },
        }
    return {
        "source": "https://arxiv.org/html/2603.15292#A2.SS2.SSS3",
        "random_bvalue_interval": [0.0, MAX_B_VALUE],
        "interval_upper_endpoint_included": False,
        "typical_probability": TYPICAL_PROBABILITY,
        "zero_dropout_probability": ZERO_DROPOUT_PROBABILITY,
        "mixture_scope": "independent for every acquisition and independently for bvals and bvecs",
        "ordering": "template index order; no sorting",
        "typical_directions": "renormalized to remove decimal rounding error",
        "typical_templates_author_confirmed": False,
        "hcp_295_assumption": (
            "Remove the last two b=0 entries of the frozen b-sorted 297-entry template; "
            "retain all 270 diffusion-weighted measurements"
        ),
        "templates": templates,
    }


def _random_acquisition(rng: jax.Array, kind: str) -> acquisition_scheme:
    typical_bvals, typical_bvecs, _ = _typical_arrays(kind)
    num_acquisitions = len(typical_bvals)
    key_bvals, key_bval_choice, key_dropout, key_bvecs, key_bvec_choice = (
        jax.random.split(rng, 5)
    )
    random_bvals = jax.random.uniform(
        key_bvals, (num_acquisitions,), minval=0.0, maxval=MAX_B_VALUE
    )
    use_typical_bvals = jax.random.bernoulli(
        key_bval_choice, TYPICAL_PROBABILITY, (num_acquisitions,)
    )
    bvals = jnp.where(use_typical_bvals, jnp.asarray(typical_bvals), random_bvals)
    retained = jax.random.bernoulli(
        key_dropout, 1.0 - ZERO_DROPOUT_PROBABILITY, (num_acquisitions,)
    )
    bvals = jnp.where(retained, bvals, 0.0)
    random_bvecs = jax.random.normal(key_bvecs, (num_acquisitions, 3))
    random_bvecs = random_bvecs / jnp.linalg.norm(random_bvecs, axis=-1, keepdims=True)
    use_typical_bvecs = jax.random.bernoulli(
        key_bvec_choice, TYPICAL_PROBABILITY, (num_acquisitions,)
    )
    bvecs = jnp.where(
        use_typical_bvecs[:, None], jnp.asarray(typical_bvecs), random_bvecs
    )
    return acquisition_scheme(bvals=bvals, bvecs=bvecs)


def random_ukb_acquisition(rng: jax.Array) -> acquisition_scheme:
    """Draw 105 UKB-like measurements with independent 50:50 mixtures."""
    return _random_acquisition(rng, "ukb")


def random_hcp_acquisition(rng: jax.Array) -> acquisition_scheme:
    """Draw 295 HCP-like measurements using the documented reconstructed template."""
    return _random_acquisition(rng, "hcp")
