"""Symbolic inference architecture used for the Figure 4b experiments."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from probjax.utils.odeint import odeint
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, RngKey

from prism.nn.diffusion import DiffusionTransformer as SharedDiffusionTransformer
from prism.nn.diffusion import EDMSimformer as SharedEDMSimformer

from .tokenizer import Tokenizer


@dataclass
class SymbolicPaperThetaInferenceConfig:
    num_layers: int = 6
    num_heads: int = 4
    widening_factor: int = 3
    attn_size: int = 16
    time_embed_dim: int = 64
    dropout_rate: float = 0.0
    use_flash_attention: bool = False
    use_flash_cross_attention: bool = False
    normalize_qk_attn: bool = False
    normalize_qk_cross_attn: bool = False
    gate_attention: bool = False
    gate_mlp: bool = False
    kv_in_features: int | None = None
    dtype: DTypeLike | None = None
    param_dtype: DTypeLike | None = None
    precision: PrecisionLike | None = None
    preferred_element_type: DTypeLike | None = None


class DiffusionTransformer(SharedDiffusionTransformer):
    """Preserve the public symbolic class path while sharing the transformer."""


class EDMSimformer(SharedEDMSimformer):
    """Figure 4b sampling with the original probjax solver defaults."""

    _transformer_cls = DiffusionTransformer

    def sample(
        self,
        rng: RngKey,
        tokenizer: Tokenizer,
        y: Array,
        dim: int,
        tokens_cfg: Array | None = None,
        model_mask: ArrayLike | None = None,
        context: ArrayLike | None = None,
        attention_mask: ArrayLike | None = None,
        sample_method: str = "ode",
        num_steps: int = 64,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> Array:
        (rng, rng_init) = jax.random.split(rng)
        eps = jax.random.normal(rng_init, (dim,)) * self.marginal_std(
            self.train_cfg.t_max
        )
        if sample_method == "ode":
            return super().sample_ode(
                eps,
                t_max=t_max,
                t_min=t_min,
                num_steps=num_steps,
                tokenizer=tokenizer,
                y=y,
                model_mask=model_mask,
                tokens_cfg=tokens_cfg,
                context=context,
                attention_mask=attention_mask,
            )
        elif sample_method == "sde":
            return super().sample_sde(
                rng,
                eps,
                t_max=t_max,
                t_min=t_min,
                num_steps=num_steps,
                tokenizer=tokenizer,
                y=y,
                model_mask=model_mask,
                tokens_cfg=tokens_cfg,
                context=context,
                attention_mask=attention_mask,
            )
        else:
            raise ValueError(f"Sample method {sample_method} not recognized.")

    def sample_and_log_prob(
        self,
        rng: RngKey,
        tokenizer: Tokenizer,
        y: Array,
        dim: int,
        tokens_cfg: Array | None = None,
        model_mask: ArrayLike | None = None,
        context: ArrayLike | None = None,
        attention_mask: ArrayLike | None = None,
        sample_method: str = "ode",
        num_steps: int = 64,
        last_euler_step: bool = False,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> tuple[Array, Array]:
        if sample_method != "ode":
            raise ValueError("sample_and_log_prob only supports ODE sampling.")
        (rng, rng_init) = jax.random.split(rng)
        t_min = t_min if t_min is not None else self.train_cfg.t_min
        t_max = t_max if t_max is not None else self.train_cfg.t_max
        eps = jax.random.normal(rng_init, (dim,)) * self.marginal_std(t_max)
        ts = self.solver_cfg.solve_schedule(
            t_max=t_max, t_min=t_min, num_steps=num_steps
        )
        theta_mask = tokenizer.simulator.theta_mask(model_mask)
        drift = self.solver_cfg.build_ode_drift(
            self,
            tokenizer,
            y=y,
            tokens_cfg=tokens_cfg,
            model_mask=model_mask,
            context=context,
            attention_mask=attention_mask,
        )

        def drift_with_logp(t, state):
            (data, logp) = state
            dx_dt = drift(t, data)
            div = jnp.sum(
                jnp.diagonal(jax.jacrev(lambda z: drift(t, z))(data)) * theta_mask
            )
            return (dx_dt, -div)

        (sample, logp) = odeint(
            drift_with_logp,
            (eps, jnp.array(0.0, dtype=eps.dtype)),
            ts,
            collect_trace=False,
            method=self.solver_cfg.ode_method,
        )
        if last_euler_step and t_min is not None and (t_min > 0.0):
            dt = -ts[-1]
            f_tmin = drift(ts[-1], sample)
            div_tmin = jnp.sum(
                jnp.diagonal(jax.jacrev(lambda z: drift(ts[-1], z))(sample))
                * theta_mask
            )
            sample = sample + f_tmin * dt
            logp = logp - div_tmin * dt
        sigma = self.marginal_std(t_max)
        theta_count = jnp.sum(theta_mask, axis=-1, keepdims=True)
        quad = jnp.sum(eps**2 * theta_mask, axis=-1, keepdims=True)
        base_logp = -0.5 * quad / sigma**2
        base_logp += -0.5 * theta_count * jnp.log(2 * np.pi * sigma**2)
        return (sample, jnp.squeeze(logp + base_logp))

    def log_prob(
        self,
        x: Array,
        tokenizer: Tokenizer,
        y: Array,
        tokens_cfg: Array | None = None,
        context: ArrayLike | None = None,
        attention_mask: ArrayLike | None = None,
        model_mask: ArrayLike | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
        num_steps: int = 64,
    ) -> Array:
        t_min = t_min if t_min is not None else self.train_cfg.t_min
        t_max = t_max if t_max is not None else self.train_cfg.t_max
        ts = self.solver_cfg.solve_schedule(t_min, t_max, num_steps)[::-1]
        theta_mask = tokenizer.simulator.theta_mask(model_mask)

        def dx_dt_fn(t, z):
            f_ = self.drift(t, z)
            g_ = self.diffusion(t, z)
            s_ = self.score(
                t,
                z,
                tokenizer=tokenizer,
                tokens_cfg=tokens_cfg,
                y=y,
                context=context,
                attention_mask=attention_mask,
                model_mask=model_mask,
            )
            return f_ - 0.5 * g_**2 * s_

        x = x
        logp0 = 0.0

        def drift(t, state):
            (data, _) = state
            dx_dt = dx_dt_fn(t, data)
            div = jnp.diagonal(jax.jacrev(lambda z: dx_dt_fn(t, z))(data))
            div = jnp.sum(div * theta_mask)
            return (dx_dt, div)

        x_final = odeint(drift, (x, logp0), ts, method="heun", collect_trace=False)
        (x_final, logp_final) = x_final
        sigma = self.marginal_std(t_max)
        theta_count = jnp.sum(theta_mask, axis=-1, keepdims=True)
        quad = jnp.sum(x_final**2 * theta_mask, axis=-1, keepdims=True)
        base_logp = -0.5 * quad / sigma**2
        base_logp += -0.5 * theta_count * jnp.log(2 * np.pi * sigma**2)
        final = logp_final + base_logp
        return jnp.squeeze(final)
