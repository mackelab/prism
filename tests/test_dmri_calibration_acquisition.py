"""Check the acquisition distribution and frozen-core integration."""

import jax
import numpy as np
import pytest
from experiments.dmri.calibration.acquisition import (
    _typical_arrays,
    protocol_metadata,
    random_hcp_acquisition,
    random_ukb_acquisition,
)
from omegaconf import OmegaConf

from prism.dmri.simulators.acquisition_scheme import (
    bvals_typ,
    bvals_typ_large,
    bvecs_typ,
    bvecs_typ_large,
)
from prism.dmri.simulators.training import resolve_acquisition_factories
from prism.train.build_simulator import build_simulator


@pytest.fixture(
    scope="module",
    params=[("ukb", random_ukb_acquisition, 105), ("hcp", random_hcp_acquisition, 295)],
)
def draws(request):
    kind, generator, count = request.param
    sampled = jax.jit(jax.vmap(generator))(
        jax.random.split(jax.random.PRNGKey(1827), 512)
    )
    return kind, generator, count, np.asarray(sampled.bvals), np.asarray(sampled.bvecs)


def test_shapes_finite_unit_directions_and_bvalue_support(draws):
    _, _, count, bvals, bvecs = draws
    assert bvals.shape == (512, count)
    assert bvecs.shape == (512, count, 3)
    assert np.isfinite(bvals).all() and np.isfinite(bvecs).all()
    assert bvals.min() == 0.0
    assert 5900.0 < bvals.max() < 6000.0
    np.testing.assert_allclose(np.linalg.norm(bvecs, axis=-1), 1.0, atol=2e-7)


def test_mixtures_are_per_acquisition_and_bvec_choices_are_independent(draws):
    kind, _, _, bvals, bvecs = draws
    typical_bvals, typical_bvecs, _ = _typical_arrays(kind)
    nonzero_indices = typical_bvals > 0
    retained = bvals[:, nonzero_indices] != 0
    b_is_typical = bvals[:, nonzero_indices] == typical_bvals[None, nonzero_indices]
    g_is_typical = np.all(
        bvecs[:, nonzero_indices] == typical_bvecs[None, nonzero_indices], axis=-1
    )
    assert 0.49 < np.mean(b_is_typical[retained]) < 0.51
    assert 0.49 < np.mean(g_is_typical) < 0.51
    # A whole-protocol mixture would make the within-row fractions near 0 or 1.
    fractions = b_is_typical.sum(axis=-1) / retained.sum(axis=-1)
    assert np.all((fractions > 0.25) & (fractions < 0.75))
    # Shared bval/bvec mixture keys would suppress the two mixed combinations.
    for b_choice in (False, True):
        for g_choice in (False, True):
            joint = (b_is_typical == b_choice) & (g_is_typical == g_choice)
            assert 0.24 < np.mean(joint[retained]) < 0.26


def test_uniform_random_bvalues_dropout_and_isotropic_random_directions(draws):
    kind, _, _, bvals, bvecs = draws
    typical_bvals, typical_bvecs, _ = _typical_arrays(kind)
    nonzero_indices = typical_bvals > 0
    assert 0.008 < np.mean(bvals[:, nonzero_indices] == 0) < 0.012
    random_b = bvals[(bvals != typical_bvals[None, :]) & (bvals > 0)]
    assert abs(random_b.mean() - 3000.0) < 40.0
    frequencies, _ = np.histogram(random_b, bins=np.linspace(0, 6000, 11))
    np.testing.assert_allclose(frequencies / len(random_b), 0.1, atol=0.006)
    random_g = bvecs[~np.all(bvecs == typical_bvecs[None, :, :], axis=-1)]
    np.testing.assert_allclose(random_g.mean(axis=0), 0.0, atol=0.012)
    np.testing.assert_allclose((random_g**2).mean(axis=0), 1 / 3, atol=0.012)


@pytest.mark.parametrize("generator", [random_ukb_acquisition, random_hcp_acquisition])
def test_seed_is_deterministic_and_generators_do_not_change_frozen_templates(generator):
    before = [
        np.asarray(array).copy()
        for array in (bvals_typ, bvecs_typ, bvals_typ_large, bvecs_typ_large)
    ]
    first, repeat, changed = [
        generator(jax.random.PRNGKey(seed)) for seed in (77, 77, 78)
    ]
    np.testing.assert_array_equal(first.bvals, repeat.bvals)
    np.testing.assert_array_equal(first.bvecs, repeat.bvecs)
    assert not np.array_equal(first.bvals, changed.bvals)
    assert not np.array_equal(first.bvecs, changed.bvecs)
    for old, current in zip(
        before, (bvals_typ, bvecs_typ, bvals_typ_large, bvecs_typ_large), strict=True
    ):
        np.testing.assert_array_equal(old, current)


def test_hcp_reconstruction_removes_only_two_b0_and_keeps_every_weighted_measurement():
    bvals, bvecs, removed = _typical_arrays("hcp")
    assert removed == (25, 26)
    assert np.all(np.asarray(bvals_typ_large)[list(removed)] == 0)
    np.testing.assert_array_equal(
        bvals[bvals > 0], bvals_typ_large[bvals_typ_large > 0]
    )
    original = np.asarray(bvecs_typ_large)[bvals_typ_large > 0]
    original = original / np.linalg.norm(original, axis=-1, keepdims=True)
    np.testing.assert_allclose(bvecs[bvals > 0], original, atol=3e-8)
    metadata = protocol_metadata()
    assert metadata["typical_templates_author_confirmed"] is False
    assert metadata["templates"]["hcp"]["shell_counts"] == {
        "0": 25,
        "1000": 90,
        "2000": 90,
        "3000": 90,
    }
    assert metadata["templates"]["hcp"]["removed_sorted_source_indices"] == [25, 26]
    assert len(metadata["templates"]["hcp"]["used_bvecs_sha256"]) == 64


def test_hydra_targets_and_full_simulator_need_no_core_registration():
    cfg = OmegaConf.create({
        "simulator": {
            "model_class": "prism.dmri.simulators.models.MultiShellBall3StickSharedDiffusivity",
            "acquisitions": [
                {
                    "_target_": "experiments.dmri.calibration.acquisition.random_ukb_acquisition",
                    "_partial_": True,
                },
                {
                    "_target_": "experiments.dmri.calibration.acquisition.random_hcp_acquisition",
                    "_partial_": True,
                },
            ],
        }
    })
    assert len(resolve_acquisition_factories(cfg)) == 2
    model, generators = build_simulator(cfg)
    assert model.theta_dim == 12
    for expected, generator in zip((105, 295), generators, strict=True):
        sample = generator(jax.random.PRNGKey(20))
        assert sample["x"].shape == (expected,)
        assert sample["acq"].bvecs.shape == (expected, 3)
        assert np.isfinite(np.asarray(sample["x"])).all()


def test_unknown_typical_template_fails():
    with pytest.raises(ValueError, match="Unknown acquisition template"):
        _typical_arrays("unknown")
