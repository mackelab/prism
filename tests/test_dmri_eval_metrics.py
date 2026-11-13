"""dMRI evaluation metrics and synthetic acquisitions used by `prism eval`."""

import numpy as np
import pytest

from prism.dmri.eval.export_metrics import (
    MetricContext,
    MetricSpec,
    _compute_ksd_metric,
)
from prism.dmri.simulators import acquisition_scheme
from prism.dmri.simulators.models import Ball3StickSharedDiffusivity


def _context(model_mask, num_samples=6):
    rng = np.random.default_rng(0)
    bvals = np.repeat([0.0, 1000.0, 2000.0], 6)
    bvecs = rng.normal(size=(bvals.size, 3))
    bvecs /= np.linalg.norm(bvecs, axis=-1, keepdims=True)
    sim_type = Ball3StickSharedDiffusivity
    num_voxels = model_mask.shape[0]
    return MetricContext(
        cfg=None,
        sim_type=sim_type,
        acq=acquisition_scheme(bvals, bvecs),
        full_data_flat_in_brain=rng.uniform(0.2, 1.0, (num_voxels, bvals.size)),
        model_parameters_brain=rng.normal(
            size=(num_voxels, num_samples, sim_type.theta_dim)
        ).astype(np.float32),
        model_mask=model_mask,
        model_mask_samples=None,
        brain_mask_flat=np.ones(num_voxels, dtype=bool),
        brain_shape=(num_voxels, 1, 1),
        orig_data=None,
    )


def _spec():
    return MetricSpec(
        key="ksd",
        type="ksd",
        output_filename="metric_ksd.nii.gz",
        batch_size=8,
        preferred_device_kinds=(),
        options={"n_bootstrap": 8, "export_pvalue": False, "random_seed": 0},
    )


B1S = [True, False, False, True, True]
B2S = [True, True, False, True, True]


def test_ksd_scores_voxels_whose_samples_share_a_model():
    # Used to fail with "Incompatible shapes for broadcasting" on (V, S, C) masks.
    masks = np.array([[B1S] * 6, [B1S] * 3 + [B2S] * 3, [B2S] * 6], dtype=bool)
    values = _compute_ksd_metric(_spec(), _context(masks), devices=None)
    assert values.shape == (3,)
    assert np.isfinite(values[[0, 2]]).all()
    assert np.isnan(values[1])


def test_ksd_is_skipped_when_every_voxel_mixes_models():
    masks = np.array([[B1S, B2S] * 3] * 2, dtype=bool)
    assert _compute_ksd_metric(_spec(), _context(masks), devices=None) is None


@pytest.mark.parametrize("shape", ["one", "per_voxel"])
def test_ksd_with_a_mask_per_voxel_is_unchanged(shape):
    masks = np.array(B2S, dtype=bool)
    if shape == "per_voxel":
        masks = np.array([B1S, B2S], dtype=bool)
    num_voxels = 1 if shape == "one" else 2
    context = _context(np.zeros((num_voxels, 6, 5), dtype=bool))
    context.model_mask = masks
    values = _compute_ksd_metric(_spec(), context, devices=None)
    assert values.shape == (num_voxels,) and np.isfinite(values).all()


def test_sliced_wasserstein_accepts_read_only_samples(tmp_path):
    # Theta samples arrive as read-only views of JAX arrays; the metric used to
    # clean them in place ("assignment destination is read-only").
    import jax.numpy as jnp

    from prism.dmri.eval.export_metrics import _compute_swd_metric

    rng = np.random.default_rng(1)
    reference = rng.normal(size=(2, 6, 4)).astype(np.float32)
    np.savez(tmp_path / "reference.npz", thetas=reference)
    theta = np.asarray(jnp.asarray(reference + 0.5))
    assert not theta.flags.writeable
    context = _context(np.zeros((2, 6, 5), dtype=bool))
    context.model_parameters_brain = theta
    spec = MetricSpec(
        key="swd",
        type="sliced_wasserstein",
        output_filename="metric_swd.nii.gz",
        options={"reference_samples_path": str(tmp_path / "reference.npz")},
    )
    distances = _compute_swd_metric(spec, context)
    assert distances.shape == (2,) and np.all(distances > 0)


def test_advanced_research_acquisition_default_is_valid():
    # The default of 500 drew 500 distinct b-values from a 400-point grid.
    import jax

    from prism.dmri.simulators import random_advanced_reasearch_acquisition_scheme

    acq = random_advanced_reasearch_acquisition_scheme(jax.random.PRNGKey(0))
    assert np.asarray(acq.bvals).shape == (355,)
