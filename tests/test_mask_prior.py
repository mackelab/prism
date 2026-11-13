import jax
import jax.numpy as jnp

from prism.simulators.mask_prior import BetaBernoulliMaskPrior


def test_beta_bernoulli_mask_prior_jittable():
    prior = BetaBernoulliMaskPrior(
        num_model_components=4,
        num_noise_components=2,
        alpha=2.0,
        beta=3.0,
        min_active_models=1,
    )

    def sample_pair(rng):
        sample = prior.sample(rng)
        return sample.hyperparameters, sample.model_mask

    sample_fn = jax.jit(sample_pair)
    key = jax.random.PRNGKey(0)
    hyper, mask = sample_fn(key)

    assert hyper.shape == (1,)
    assert mask.shape == (6,)
    assert mask.dtype == jnp.bool_

    log_prob_fn = jax.jit(
        lambda m, h: prior.log_prob(m, h)  # noqa: E731
    )
    lp = log_prob_fn(mask, hyper)
    assert jnp.isfinite(lp)
