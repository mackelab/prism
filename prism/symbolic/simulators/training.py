"""Training data for additive symbolic simulators.

Two observation formats exist. ``coordinate_signal`` stacks coordinates and
noisy values as the two columns of ``x``. ``signal_blocks`` evaluates a fixed
grid and reshapes the values into rows of ``input_dim``, as the paper does.
"""

import math

import jax
import jax.numpy as jnp


def _make_additive_generator(model_class, grid, mask_prior, posterior_score):
    """Return observations with columns (coordinate, noisy signal)."""

    def simulator(rng, mask_prior_hyperparameter=None, model_mask=None):
        rng_theta, rng_prior, rng_mask, rng_noise = jax.random.split(rng, 4)
        theta = jax.random.normal(rng_theta, (model_class.theta_dim,))
        prior = (
            mask_prior.sample_hyperparameters(rng_prior)
            if mask_prior_hyperparameter is None
            else mask_prior_hyperparameter
        )
        if model_mask is None:
            model_mask = mask_prior.sample_model_mask(rng_mask, prior)
        model = model_class.from_theta(theta, model_mask=model_mask)
        observed = model.evaluate(grid, rng=rng_noise)
        data = {
            "mask_prior": prior,
            "model_mask": model_mask,
            "theta": theta,
            "x": jnp.stack((grid, observed), axis=-1),
        }
        if posterior_score:

            def potential(value):
                candidate = model_class.from_theta(value, model_mask=model_mask)
                return (
                    candidate.log_likelihood(grid, observed)
                    + jax.scipy.stats.norm.logpdf(value).sum()
                )

            data["target_score"] = jax.grad(potential)(theta)
        return data

    return simulator


def _make_signal_block_generator(model_class, grid, mask_prior, input_dim):
    """Preserve the symbolic paper's RNG order and signal-only observation blocks."""

    def simulator(rng, mask_prior_hyperparameter=None, model_mask=None):
        rng_theta, rng_mask, rng_hyper, rng_noise = jax.random.split(rng, 4)
        theta = jax.random.normal(rng_theta, shape=(model_class.theta_dim,))
        prior = (
            mask_prior.sample_hyperparameters(rng_hyper)
            if mask_prior_hyperparameter is None
            else mask_prior_hyperparameter
        )
        if model_mask is None:
            model_mask = mask_prior.sample_model_mask(rng_mask, prior)
        model = model_class.from_theta(theta, model_mask=model_mask)
        observed = model.evaluate(grid, rng=rng_noise).reshape(-1)
        remainder = (-observed.size) % input_dim
        if remainder:
            observed = jnp.concatenate([
                observed,
                jnp.zeros((remainder,), dtype=observed.dtype),
            ])
        return {
            "mask_prior": prior,
            "model_mask": model_mask,
            "theta": theta,
            "x": observed.reshape(-1, input_dim),
        }

    return simulator


def _signal_block_grid(simulator_cfg):
    x_min = float(simulator_cfg["x_min"])
    x_max = float(simulator_cfg["x_max"])
    x_step = float(simulator_cfg["x_step"])
    input_dim = simulator_cfg["input_dim"]
    if (
        not all(math.isfinite(value) for value in (x_min, x_max, x_step))
        or not x_min < x_max
        or x_step <= 0
        or not isinstance(input_dim, int)
        or input_dim <= 0
    ):
        raise ValueError(
            "Signal blocks require x_min < x_max, positive x_step and input_dim"
        )
    return jnp.arange(x_min, x_max, x_step)


def _observation_format(simulator_cfg):
    observation_format = simulator_cfg.get("observation_format", "coordinate_signal")
    if observation_format not in ("coordinate_signal", "signal_blocks"):
        raise ValueError(f"Unknown additive observation_format {observation_format!r}")
    return observation_format


def training_generators(model_class, simulator_cfg, mask_prior, posterior_score):
    """Return the sample generator for the configured observation format."""
    if _observation_format(simulator_cfg) == "signal_blocks":
        grid = _signal_block_grid(simulator_cfg)
        if posterior_score:
            raise ValueError(
                "The Figure 4b signal-block simulator does not provide posterior scores"
            )
        return [
            _make_signal_block_generator(
                model_class, grid, mask_prior, simulator_cfg["input_dim"]
            )
        ]
    num_points = simulator_cfg.get("num_points", 32)
    x_min = float(simulator_cfg.get("x_min", -1.0))
    x_max = float(simulator_cfg.get("x_max", 1.0))
    if not isinstance(num_points, int) or num_points < 2 or not x_min < x_max:
        raise ValueError(
            "Additive simulation requires num_points >= 2 and x_min < x_max"
        )
    grid = jnp.linspace(x_min, x_max, num_points)
    return [_make_additive_generator(model_class, grid, mask_prior, posterior_score)]


def evaluation_design(simulator_cfg):
    """Return the grid shared by all signal-block observations, else ``None``."""
    if _observation_format(simulator_cfg) != "signal_blocks":
        return None
    return _signal_block_grid(simulator_cfg)
