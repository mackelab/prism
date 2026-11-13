"""The generic factory preserves simulator behavior and subclass overrides."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prism.dmri.simulators import Ball, acquisition_scheme
from prism.dmri.simulators.models import Ball3StickNoise
from prism.symbolic.simulators import Linear, SimplePolynomial


@pytest.mark.parametrize("model_class", [Ball3StickNoise, SimplePolynomial])
def test_factory_matches_legacy_model_parameters_masks_and_simulations(model_class):
    theta = jnp.linspace(-0.5, 0.5, model_class.theta_dim)
    mask = jnp.arange(model_class.num_compartments()) % 2 == 0
    old = model_class.from_theta(theta, model_mask=mask)
    new = model_class.create_from_theta(theta, model_mask=mask)

    assert type(new) is type(old) is model_class
    np.testing.assert_array_equal(new.model_mask, mask)
    np.testing.assert_array_equal(new.theta, old.theta)
    key = jax.random.key(4)
    if model_class is SimplePolynomial:
        grid = jnp.linspace(-1.0, 1.0, 8)
        expected = old.evaluate(grid, key)
        actual = new.evaluate(grid, key)
    else:
        acq = acquisition_scheme(
            jnp.array([0.0, 1000.0]),
            jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        )
        expected = old.signal(acq, key)
        actual = new.signal(acq, key)
    np.testing.assert_array_equal(actual, expected)


def test_factory_dispatches_to_legacy_subclass_override_with_kwargs():
    class ScaledBall(Ball):
        @classmethod
        def from_theta(cls, theta, *, scale):
            original = super().from_theta(theta)
            return cls(original.lam * scale)

    theta = jnp.array([0.25])
    model = ScaledBall.create_from_theta(theta, scale=2.0)

    assert type(model) is ScaledBall
    np.testing.assert_array_equal(
        model.lam, ScaledBall.from_theta(theta, scale=2.0).lam
    )
    np.testing.assert_array_equal(model.lam, Ball.from_theta(theta).lam * 2.0)


@pytest.mark.parametrize("model_class", [Ball, Linear])
def test_factory_supports_pytree_roundtrip_and_jitted_vmap(model_class):
    theta = jnp.array([0.25])
    old = model_class.from_theta(theta)
    new = model_class.create_from_theta(theta)
    leaves, structure = jax.tree.flatten(new)
    restored = jax.tree.unflatten(structure, leaves)

    assert type(restored) is model_class
    np.testing.assert_array_equal(restored.theta, old.theta)

    batch = jnp.array([[-0.5], [0.25], [0.75]])
    expected = jax.jit(jax.vmap(lambda value: model_class.from_theta(value).theta))(
        batch
    )
    actual = jax.jit(
        jax.vmap(lambda value: model_class.create_from_theta(value).theta)
    )(batch)
    np.testing.assert_array_equal(actual, expected)
