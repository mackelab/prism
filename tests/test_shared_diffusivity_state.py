"""Shared diffusivity must belong to a model instance, including under JAX."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simulator_reference import reference_acquisition, reference_ssfp_acquisition

from prism.dmri.simulators.models import (
    Ball3StickSharedDiffusivity,
    MultiShellBall3StickSharedDiffusivity,
    SSFPBall3StickSharedDiffusivity,
)

MODELS = (
    Ball3StickSharedDiffusivity,
    MultiShellBall3StickSharedDiffusivity,
    SSFPBall3StickSharedDiffusivity,
)


def _inputs(model_type):
    acq = (
        reference_ssfp_acquisition()
        if model_type is SSFPBall3StickSharedDiffusivity
        else reference_acquisition()
    )
    theta_a = jnp.linspace(-0.8, 0.7, model_type.theta_dim)
    theta_b = theta_a.at[3].set(1.2)
    if model_type is MultiShellBall3StickSharedDiffusivity:
        theta_b = theta_b.at[4].set(-1.0)
    return acq, theta_a, theta_b


@pytest.mark.parametrize("model_type", MODELS)
def test_likelihood_and_components_are_independent_of_construction_order(model_type):
    acq, theta_a, theta_b = _inputs(model_type)
    model_a = model_type.from_theta(theta_a)
    signal_a = model_a.signal(acq)
    observed = signal_a + 0.01
    expected_ll = model_a.log_likelihood(acq, observed)
    expected_components = [
        component.signal(acq) for component in model_a.model_compartments
    ]

    model_b = model_type.from_theta(theta_b)
    assert not np.allclose(model_b.signal(acq), signal_a)
    # Evaluate likelihood first: signal() used to refresh the global class state
    # and accidentally hide the order dependence of log_likelihood().
    np.testing.assert_array_equal(model_a.log_likelihood(acq, observed), expected_ll)
    for component, expected in zip(
        model_a.model_compartments, expected_components, strict=True
    ):
        np.testing.assert_array_equal(component.signal(acq), expected)
    np.testing.assert_array_equal(model_a.signal(acq), signal_a)


@pytest.mark.parametrize("model_type", MODELS)
def test_construction_does_not_write_shared_values_to_classes(model_type):
    _, theta_a, theta_b = _inputs(model_type)
    before = {
        (component, name): getattr(component, name)
        for component in model_type.model_types
        for name in component.shared_parameter_names
    }
    model_type.from_theta(theta_a)
    model_type.from_theta(theta_b)
    for (component, name), original in before.items():
        assert getattr(component, name) is original


@pytest.mark.parametrize("model_type", MODELS[:2])
def test_likelihood_gradient_does_not_depend_on_another_instance(model_type):
    acq, theta_a, theta_b = _inputs(model_type)
    observed = model_type.from_theta(theta_a).signal(acq) + 0.01

    def independent(theta):
        return model_type.from_theta(theta).log_likelihood(acq, observed)

    def interleaved(theta, other_theta):
        model = model_type.from_theta(theta)
        model_type.from_theta(other_theta)
        return model.log_likelihood(acq, observed)

    expected_value, expected_grad = jax.value_and_grad(independent)(theta_a)
    value, (gradient_a, gradient_b) = jax.value_and_grad(interleaved, argnums=(0, 1))(
        theta_a, theta_b
    )
    np.testing.assert_array_equal(value, expected_value)
    np.testing.assert_array_equal(gradient_a, expected_grad)
    np.testing.assert_array_equal(gradient_b, jnp.zeros_like(theta_b))
    assert np.isfinite(gradient_a).all()


@pytest.mark.parametrize("model_type", MODELS)
def test_jit_vmap_likelihood_has_no_shared_tracer_state(model_type):
    acq, theta_a, theta_b = _inputs(model_type)
    observed = model_type.from_theta(theta_a).signal(acq) + 0.01
    thetas = jnp.stack((theta_a, theta_b))

    def likelihood(theta):
        return model_type.from_theta(theta).log_likelihood(acq, observed)

    expected = jnp.stack([likelihood(theta) for theta in thetas])
    with jax.checking_leaks():
        actual = jax.jit(jax.vmap(likelihood))(thetas)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_type", MODELS)
def test_shared_component_pytree_keeps_bound_diffusivity(model_type):
    acq, theta_a, theta_b = _inputs(model_type)
    model_a = model_type.from_theta(theta_a)
    expected = [component.signal(acq) for component in model_a.model_compartments]
    model_type.from_theta(theta_b)
    for component, signal in zip(model_a.model_compartments, expected, strict=True):
        # Static diffusivities are absent from the local theta, but they must
        # survive flatten/unflatten when a compartment is itself a JIT argument.
        actual = jax.jit(lambda model: model.signal(acq))(component)
        np.testing.assert_allclose(actual, signal, rtol=1e-5, atol=1e-6)
        assert component.theta.shape == (type(component).theta_dim,)


@pytest.mark.parametrize("model_type", MODELS)
def test_direct_constructor_binds_shared_values_without_modifying_inputs(model_type):
    acq, theta_a, theta_b = _inputs(model_type)
    model_a = model_type.from_theta(theta_a)
    model_b = model_type.from_theta(theta_b)
    signal_a = model_a.signal(acq)
    expected = model_b.signal(acq)

    rebound = model_type(
        model_a.model_fractions,
        model_a.model_compartments,
        model_a.noise_compartments,
        model_a.model_mask,
        model_b.shared_parameter,
    )

    np.testing.assert_array_equal(rebound.signal(acq), expected)
    np.testing.assert_array_equal(model_a.signal(acq), signal_a)
