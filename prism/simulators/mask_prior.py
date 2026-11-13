"""Priors over which compartments are active in a multi-compartment model.

Implements Beta–Bernoulli style mask priors with utilities to sample masks and
convert them into boolean vectors that gate compartments during simulation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, logsumexp
from jax.typing import ArrayLike


@dataclass(frozen=True)
class MaskPriorSample:
    """Container holding sampled hyperparameters and the resulting mask."""

    hyperparameters: jax.Array
    model_mask: jax.Array


jax.tree_util.register_dataclass(
    MaskPriorSample,
    data_fields=("hyperparameters", "model_mask"),
    meta_fields=(),
)


class MaskPrior:
    """Base class for hierarchical priors over simulator model masks."""

    mask_prior_dim: int = 1

    def __init__(
        self,
        num_model_components: int,
        num_noise_components: int,
    ) -> None:
        if num_model_components <= 0:
            raise ValueError("num_model_components must be positive.")
        if num_noise_components < 0:
            raise ValueError("num_noise_components cannot be negative.")
        self.num_model_components = int(num_model_components)
        self.num_noise_components = int(num_noise_components)

    def sample(self, rng: jax.random.KeyArray) -> MaskPriorSample:
        """Sample hyperparameters and a mask conditioned on them."""
        rng_h, rng_mask = jax.random.split(rng)
        hyperparameters = self.sample_hyperparameters(rng_h)
        model_mask = self.sample_model_mask(rng_mask, hyperparameters)
        return MaskPriorSample(hyperparameters=hyperparameters, model_mask=model_mask)

    def log_prob(self, model_mask: ArrayLike, hyperparameters: ArrayLike) -> jax.Array:
        """Return log probability of a binary mask conditioned on hyperparameters."""
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        model_mask_slice = mask[: self.num_model_components]
        noise_mask_slice = mask[self.num_model_components :]
        logp_models = self.log_prob_model_components(model_mask_slice, hyperparameters)
        logp_noise = self.log_prob_noise_components(noise_mask_slice, hyperparameters)
        return logp_models + logp_noise

    def prob(self, model_mask: ArrayLike, hyperparameters: ArrayLike) -> jax.Array:
        """Return probability of a binary mask conditioned on hyperparameters."""
        return jnp.exp(self.log_prob(model_mask, hyperparameters))

    def sample_model_mask(
        self,
        rng: jax.random.KeyArray,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """Sample masks for model and noise components and concatenate them."""
        rng_models, rng_noise = jax.random.split(rng)
        model_mask = self.sample_model_components(rng_models, hyperparameters)
        noise_mask = self.sample_noise_components(rng_noise, hyperparameters)
        if self.num_noise_components == 0:
            return model_mask
        return jnp.concatenate([model_mask, noise_mask], axis=-1)

    def sample_noise_components(
        self,
        rng: jax.random.KeyArray,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """Default noise prior: exactly one active component (if any exist)."""
        del hyperparameters
        if self.num_noise_components == 0:
            return jnp.zeros((0,), dtype=jnp.bool_)

        idx = jax.random.randint(
            rng,
            shape=(1,),
            minval=0,
            maxval=self.num_noise_components,
        )
        noise_mask = jnp.zeros((self.num_noise_components,), dtype=jnp.bool_)
        noise_mask = noise_mask.at[idx].set(True)
        return noise_mask

    def log_prob_noise_components(
        self,
        noise_mask: ArrayLike,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        del hyperparameters
        if self.num_noise_components == 0:
            return jnp.array(0.0)
        mask = jnp.asarray(noise_mask, dtype=jnp.bool_)
        num_active = jnp.sum(mask)
        log_uniform = -jnp.log(self.num_noise_components)
        return jnp.where(num_active == 1, log_uniform, -jnp.inf)

    def _log_prob_autoregressive_noise_components(
        self,
        noise_mask: ArrayLike,
    ) -> jax.Array:
        """Factor the uniform, exactly-one-noise prior along its mask prefix."""
        if self.num_noise_components == 0:
            return jnp.zeros((0,), dtype=jnp.float32)

        mask = jnp.asarray(noise_mask, dtype=jnp.bool_)
        n = self.num_noise_components

        def step(carry, z_i):
            chosen, remaining = carry
            remaining_f = remaining.astype(jnp.float32)
            p_i = jnp.where(chosen, 0.0, 1.0 / remaining_f)
            logp_i = jnp.where(z_i, jnp.log(p_i), jnp.log1p(-p_i))
            chosen = jnp.logical_or(chosen, z_i)
            remaining = remaining - jnp.array(1, dtype=remaining.dtype)
            return (chosen, remaining), logp_i

        _, logps = jax.lax.scan(
            step,
            (jnp.array(False), jnp.array(n, dtype=jnp.int32)),
            mask,
        )
        return logps

    def sample_model_components(
        self,
        rng: jax.random.KeyArray,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """Override to draw model-component masks."""
        raise NotImplementedError

    def sample_hyperparameters(self, rng: jax.random.KeyArray) -> jax.Array:
        """Override to sample hyperparameters for the prior."""
        raise NotImplementedError

    def log_prob_model_components(
        self,
        model_mask: ArrayLike,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """Override to evaluate log probability under the prior."""
        raise NotImplementedError


class BetaBernoulliMaskPrior(MaskPrior):
    """Independent Bernoulli masks driven by a shared Beta hyper-prior."""

    mask_prior_dim: int = 1

    def __init__(
        self,
        num_model_components: int,
        num_noise_components: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        min_active_models: int = 1,
    ) -> None:
        super().__init__(num_model_components, num_noise_components)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive.")
        if min_active_models < 0:
            raise ValueError("min_active_models cannot be negative.")
        if min_active_models > self.num_model_components:
            raise ValueError(
                "min_active_models cannot exceed available model components."
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_active_models = int(min_active_models)
        if self.min_active_models > 0:
            combos = np.array(
                list(
                    itertools.combinations(
                        range(self.num_model_components),
                        self.min_active_models,
                    )
                ),
                dtype=np.int32,
            )
            if combos.size == 0:
                combos = np.zeros((1, 0), dtype=np.int32)
            forced_masks = np.zeros(
                (len(combos), self.num_model_components),
                dtype=np.bool_,
            )
            for idx, combo in enumerate(combos):
                forced_masks[idx, combo] = True
            self._forced_masks = jnp.asarray(forced_masks)
            self._num_forced_combos = forced_masks.shape[0]
        else:
            self._forced_masks = None
            self._num_forced_combos = 1

    def sample_hyperparameters(self, rng: jax.random.KeyArray) -> jax.Array:
        return jax.random.beta(
            rng,
            a=self.alpha,
            b=self.beta,
            shape=(self.mask_prior_dim,),
        )

    def sample_model_components(
        self,
        rng: jax.random.KeyArray,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        prob = jnp.clip(jnp.asarray(hyperparameters)[0], 0.0, 1.0)
        rng_mask, rng_force = jax.random.split(rng)
        base_mask = jax.random.bernoulli(
            rng_mask,
            p=prob,
            shape=(self.num_model_components,),
        )
        if self.min_active_models == 0:
            return base_mask.astype(jnp.bool_)

        # Force a subset of components to remain active.
        if self.min_active_models == 1:
            forced_idx = jax.random.randint(
                rng_force,
                shape=(1,),
                minval=0,
                maxval=self.num_model_components,
            )
        else:
            forced_idx = jax.random.choice(
                rng_force,
                self.num_model_components,
                shape=(self.min_active_models,),
                replace=False,
            )
        base_mask = base_mask.at[forced_idx].set(True)
        return base_mask.astype(jnp.bool_)

    def log_prob_model_components(
        self,
        model_mask: ArrayLike,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        prob = jnp.clip(jnp.asarray(hyperparameters)[0], 1e-6, 1 - 1e-6)
        log_on = jnp.log(prob)
        log_off = jnp.log1p(-prob)
        component_log = jnp.where(mask, log_on, log_off)

        if self.min_active_models == 0:
            return jnp.sum(component_log)

        forced_masks = self._forced_masks
        assert forced_masks is not None
        valid_combo = jnp.all(jnp.where(forced_masks, mask[None, :], True), axis=-1)
        masked_component_log = jnp.where(~forced_masks, component_log[None, :], 0.0)
        log_prob_cond = jnp.sum(masked_component_log, axis=-1)
        log_prob_cond = jnp.where(valid_combo, log_prob_cond, -jnp.inf)
        log_weight = -jnp.log(self._num_forced_combos)
        return logsumexp(log_prob_cond + log_weight)

    def log_prob_autoregressive(
        self,
        model_mask: ArrayLike,
        hyperparameters: ArrayLike,
    ) -> jax.Array:
        """Return source-compatible log probabilities given preceding mask bits.

        Adapted from mackelab/sbmi_transformer_symbolic (MIT).
        The forced-component mixture is marginalized over each unfinished suffix.
        """
        full_mask = jnp.asarray(model_mask, dtype=jnp.bool_)
        mask = full_mask[: self.num_model_components]
        noise_mask = full_mask[self.num_model_components :]
        prob = jnp.clip(jnp.asarray(hyperparameters).reshape(-1)[0], 1e-6, 1 - 1e-6)
        log_on = jnp.log(prob)
        log_off = jnp.log1p(-prob)

        if self.min_active_models == 0:
            model_logp = jnp.where(mask, log_on, log_off)
            noise_logp = self._log_prob_autoregressive_noise_components(noise_mask)
            return jnp.concatenate([model_logp, noise_logp], axis=0)

        k = self.num_model_components
        m = self.min_active_models
        a_vals = jnp.arange(k + 1, dtype=jnp.int32)

        def log_binom(n: jax.Array, r: jax.Array) -> jax.Array:
            r = jnp.asarray(r)
            invalid = (r < 0) | (r > n)
            return jnp.where(
                invalid,
                -jnp.inf,
                gammaln(n + 1.0) - gammaln(r + 1.0) - gammaln(n - r + 1.0),
            )

        log_forced = log_binom(a_vals, m) - log_binom(k, m)
        log_f = jnp.where(
            a_vals >= m,
            log_forced + (a_vals - m) * log_on + (k - a_vals) * log_off,
            -jnp.inf,
        )

        idxs = jnp.arange(k, dtype=jnp.int32)

        def step(active_count: jax.Array, inputs: tuple[jax.Array, jax.Array]):
            z_i, i = inputs
            n_rem = k - i - 1
            k0 = a_vals - active_count
            k1 = a_vals - active_count - 1
            log_c0 = log_binom(n_rem, k0)
            log_c1 = log_binom(n_rem, k1)
            log_num0 = logsumexp(log_f + log_c0)
            log_num1 = logsumexp(log_f + log_c1)
            log_den = logsumexp(jnp.stack([log_num0, log_num1]))
            log_prob0 = log_num0 - log_den
            log_prob1 = log_num1 - log_den
            log_prob = jnp.where(z_i, log_prob1, log_prob0)
            log_prob = jnp.where(jnp.isfinite(log_den), log_prob, -jnp.inf)
            active_count = active_count + z_i.astype(active_count.dtype)
            return active_count, log_prob

        _, log_probs = jax.lax.scan(step, jnp.array(0, dtype=jnp.int32), (mask, idxs))
        noise_logp = self._log_prob_autoregressive_noise_components(noise_mask)
        return jnp.concatenate([log_probs, noise_logp], axis=0)


class TotalParamPenalizedPrior(MaskPrior):
    """Global-complexity prior (scan) with fixed p0 and u∈[0,1] controlling penalty.

    Conditionals:
      - quadratic mode (default):
          logit P(z_i=1 | T) = logit(p0) - λ * (2*T*s_i + s_i**2)
        (global coupling via T; stricter as the model grows)

      - linear mode (AIC/BIC-like):
          logit P(z_i=1) = logit(p0) - λ * s_i
        (independent across components; fixed linear size penalty)

    Hyperparameter:
      - u ~ Beta(u_alpha, u_beta), or pass a fixed u in [0,1] when calling.
      - u=1 → no penalty (reverts to p0). u=0 → strongest penalty.
      - We map u to a desired worst-case logit drop delta* and set
            λ = delta* / scale_max,
        where scale_max depends on the mode:
          * quadratic: scale_max = max_i (2*S*s_i - s_i**2), S = sum_j s_j
          * linear:    scale_max = max_i s_i
    """

    mask_prior_dim: int = 1  # hyperparameters = [u]

    def __init__(
        self,
        num_model_components: int,
        num_noise_components: int,
        num_model_parameters: list[int],
        *,
        p0: float = 0.5,
        u_alpha: float = 1.0,
        u_beta: float = 1.0,
        penalty_min: float = 0.0,  # min worst-case logit penalty (often 0)
        penalty_max: float = 5.0,  # max worst-case logit penalty
        penalty_mode: str = "linear",  # "quadratic" or "linear"
        eps: float = 1e-3,
    ) -> None:
        super().__init__(num_model_components, num_noise_components)

        if len(num_model_parameters) != self.num_model_components:
            raise ValueError(
                "len(num_model_parameters) must equal num_model_components."
            )
        # Allow zero-sized components
        if any(n < 0 for n in num_model_parameters):
            raise ValueError(
                "All entries in num_model_parameters must be non-negative."
            )
        if not (0.0 < p0 < 1.0):
            raise ValueError("p0 must be in (0,1).")
        if u_alpha <= 0 or u_beta <= 0:
            raise ValueError("u_alpha and u_beta must be positive.")
        if penalty_min < 0 or penalty_max < penalty_min:
            raise ValueError("Require 0 <= penalty_min <= penalty_max.")
        if penalty_mode not in ("quadratic", "linear"):
            raise ValueError('penalty_mode must be "quadratic" or "linear".')
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self._sizes = jnp.asarray(
            np.asarray(num_model_parameters, dtype=np.float32)
        )  # [K]
        self._S_total = jnp.sum(self._sizes)
        self.p0 = float(p0)
        self.u_alpha = float(u_alpha)
        self.u_beta = float(u_beta)
        self.penalty_min = float(penalty_min)
        self.penalty_max = float(penalty_max)
        self.penalty_mode = penalty_mode
        self._eps = float(eps)

        # mode-dependent worst-case scale for one step
        if self.penalty_mode == "quadratic":
            # max over T∈[0,S−s_i] of (2*T*s_i + s_i^2) occurs at T=S−s_i → 2*S*s_i − s_i^2
            raw = 2.0 * self._S_total * self._sizes - self._sizes**2
        else:  # "linear"
            # worst case step is simply the largest s_i
            raw = self._sizes

        self._scale_max = jnp.maximum(jnp.max(raw), jnp.array(eps, dtype=jnp.float32))

        # Precompute baseline logit
        p0_clip = np.clip(self.p0, eps, 1.0 - eps)
        self._base_logit = float(np.log(p0_clip) - np.log1p(-p0_clip))

    # ----- Hyperparameters (u only) -----

    def sample_hyperparameters(self, rng: jax.random.KeyArray) -> jax.Array:
        """Sample u ~ Beta(u_alpha, u_beta). Caller may also pass a fixed u instead."""
        u = jax.random.beta(rng, a=self.u_alpha, b=self.u_beta, shape=(1,))
        return u  # shape [1]

    def _u_to_lambda(self, u: ArrayLike) -> jax.Array:
        """Map u∈[0,1] to λ using mode-dependent worst-case scaling."""
        u = jnp.clip(jnp.asarray(u).reshape(-1)[0], 0.0, 1.0)
        # desired worst-case logit drop (u=1 → penalty_min; u=0 → penalty_max)
        delta_star = self.penalty_min + (1.0 - u) * (
            self.penalty_max - self.penalty_min
        )
        lam = delta_star / self._scale_max
        return lam  # scalar

    # ----- Sampling with jax.scan -----

    def sample_model_components(
        self,
        rng: jax.random.PRNGKey,
        hyperparameters: ArrayLike,  # expects u
    ) -> jax.Array:
        lam = self._u_to_lambda(hyperparameters)
        base_logit = jnp.array(self._base_logit, dtype=jnp.float32)

        if self.penalty_mode == "quadratic":
            # (unchanged) autoregressive via scan
            keys = jax.random.split(rng, self.num_model_components)
            sizes = self._sizes

            def step(T, inputs):
                s_i, key_i = inputs
                penalty = lam * (2.0 * T * s_i + s_i * s_i)
                logit_i = base_logit - penalty
                p_i = jax.nn.sigmoid(logit_i)
                z_i = jax.random.bernoulli(key_i, p=p_i)
                T_new = T + s_i * z_i.astype(T.dtype)
                return T_new, z_i.astype(jnp.bool_)

            T0 = jnp.array(0.0, dtype=jnp.float32)
            _, zs = jax.lax.scan(step, T0, (sizes, keys))
            return zs
        else:
            # --- LINEAR MODE: no scan needed, vectorized & order-invariant ---
            logits = base_logit - lam * self._sizes  # [K]
            probs = jax.nn.sigmoid(logits)  # [K]
            zs = jax.random.bernoulli(rng, p=probs, shape=probs.shape)
            return zs.astype(jnp.bool_)

    def log_prob_model_components(
        self,
        model_mask: ArrayLike,
        hyperparameters: ArrayLike,  # expects u
    ) -> jax.Array:
        mask = jnp.asarray(model_mask, dtype=jnp.bool_)[: self.num_model_components]
        lam = self._u_to_lambda(hyperparameters)
        base_logit = jnp.array(self._base_logit, dtype=jnp.float32)

        if self.penalty_mode == "quadratic":
            # (unchanged) autoregressive via scan
            def step(carry, inputs):
                T, logp = carry
                s_i, z_i = inputs
                penalty = lam * (2.0 * T * s_i + s_i * s_i)
                logit_i = base_logit - penalty
                p_i = jax.nn.sigmoid(logit_i)
                p_i = jnp.clip(p_i, self._eps, 1.0 - self._eps)
                logp_i = jnp.where(z_i, jnp.log(p_i), jnp.log1p(-p_i))
                T_new = T + s_i * z_i.astype(T.dtype)
                return (T_new, logp + logp_i), None

            (T_final, logp_sum), _ = jax.lax.scan(
                step,
                (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)),
                (self._sizes, mask),
            )
            del T_final
            return logp_sum
        else:
            # --- LINEAR MODE: no scan, independent Bernoullis ---
            logits = base_logit - lam * self._sizes  # [K]
            probs = jax.nn.sigmoid(logits)  # [K]
            probs = jnp.clip(probs, self._eps, 1.0 - self._eps)
            logp = jnp.where(mask, jnp.log(probs), jnp.log1p(-probs))
            return jnp.sum(logp)
