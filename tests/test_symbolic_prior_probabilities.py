"""Check prior conditionals against an exhaustive independent generative law."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prism.simulators.mask_prior import BetaBernoulliMaskPrior


def _enumerated_distribution(num_models, num_noises, num_forced, probability):
    """Enumerate raw Bernoulli outcomes before forcing the selected subset on."""
    masks = np.array(
        list(itertools.product([False, True], repeat=num_models + num_noises)),
        dtype=np.bool_,
    )
    weights = np.zeros(len(masks))
    indices = {tuple(mask): i for i, mask in enumerate(masks)}
    forced_subsets = list(itertools.combinations(range(num_models), num_forced))
    noise_choices = np.eye(num_noises, dtype=np.bool_) if num_noises else [()]
    for raw in itertools.product([False, True], repeat=num_models):
        active = sum(raw)
        raw_weight = probability**active * (1 - probability) ** (num_models - active)
        for forced in forced_subsets:
            model = np.array(raw, dtype=np.bool_)
            model[list(forced)] = True
            for noise in noise_choices:
                mask = tuple(model) + tuple(noise)
                weights[indices[mask]] += (
                    raw_weight / len(forced_subsets) / len(noise_choices)
                )
    np.testing.assert_allclose(weights.sum(), 1.0)
    return masks, weights


@pytest.mark.parametrize("num_forced", [0, 1, 2, 4])
@pytest.mark.parametrize("probability", [0.2, 0.7])
@pytest.mark.parametrize("num_noises", [0, 3])
def test_beta_conditionals_match_exhaustive_prefix_probabilities(
    num_forced, probability, num_noises
):
    prior = BetaBernoulliMaskPrior(4, num_noises, min_active_models=num_forced)
    masks, weights = _enumerated_distribution(4, num_noises, num_forced, probability)
    hyperparameters = jnp.array([probability])
    actual = np.asarray(
        jax.jit(
            jax.vmap(lambda mask: prior.log_prob_autoregressive(mask, hyperparameters))
        )(jnp.asarray(masks))
    )
    assert actual.shape == masks.shape
    assert not np.isnan(actual).any()

    for row, mask in enumerate(masks):
        for column in range(masks.shape[1]):
            previous = np.all(masks[:, :column] == mask[:column], axis=1)
            denominator = weights[previous].sum()
            if denominator == 0:
                # Conditionals after an impossible prefix are undefined.
                continue
            matching = previous & (masks[:, column] == mask[column])
            numerator = weights[matching].sum()
            if numerator == 0:
                assert np.isneginf(actual[row, column])
            else:
                np.testing.assert_allclose(
                    actual[row, column], np.log(numerator / denominator), atol=3e-6
                )

    np.testing.assert_allclose(
        np.exp(actual.sum(axis=1)), weights, atol=3e-7, rtol=1e-5
    )


def test_prefix_forces_the_last_required_expression_on():
    prior = BetaBernoulliMaskPrior(4, 0, min_active_models=2)
    accepted = prior.log_prob_autoregressive(
        jnp.array([False, True, False, True]), jnp.array([0.3])
    )
    rejected = prior.log_prob_autoregressive(
        jnp.array([False, True, False, False]), jnp.array([0.3])
    )
    assert accepted[-1] == 0.0
    assert jnp.isneginf(rejected[-1])
    np.testing.assert_array_equal(accepted[:-1], rejected[:-1])


@pytest.mark.parametrize(
    "noise_mask, expected",
    [
        ([True, False, False], [np.log(1 / 3), 0.0, 0.0]),
        ([False, True, False], [np.log(2 / 3), np.log(1 / 2), 0.0]),
        ([False, False, True], [np.log(2 / 3), np.log(1 / 2), 0.0]),
        ([False, False, False], [np.log(2 / 3), np.log(1 / 2), -np.inf]),
        ([True, True, False], [np.log(1 / 3), -np.inf, 0.0]),
    ],
)
def test_noise_conditionals_require_exactly_one_active_alternative(
    noise_mask, expected
):
    prior = BetaBernoulliMaskPrior(1, 3, min_active_models=0)
    actual = prior.log_prob_autoregressive(
        jnp.array([True, *noise_mask]), jnp.array([0.5])
    )
    np.testing.assert_allclose(actual[1:], expected, atol=2e-7)
