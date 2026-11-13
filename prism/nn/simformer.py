"""Default diffusion parameter inference with its solver and sampling behavior."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, RngKey

from prism.nn.diffusion import DiffusionTransformer as SharedDiffusionTransformer
from prism.nn.diffusion import EDMSimformer as SharedEDMSimformer
from prism.nn.tokenizer import Tokenizer


def masked_standard_normal_log_prob(x: Array, sigma: Array, mask: Array) -> Array:
    """Return a spherical Gaussian log density over active dimensions only."""
    active_dimensions = jnp.sum(mask)
    squared_norm = jnp.sum(x**2 * mask)
    return -0.5 * squared_norm / sigma**2 - 0.5 * active_dimensions * jnp.log(
        2 * np.pi * sigma**2
    )


@dataclass
class ThetaInferenceConfig:
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
    """Default denoising transformer; shares construction with the base class."""


class EDMSimformer(SharedEDMSimformer):
    """Default sampling, including its optional final Euler step."""

    _transformer_cls = DiffusionTransformer
    _ode_method = "exp_ab2_scalarL"

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
        last_euler_step: bool = False,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> Array:
        rng, rng_init = jax.random.split(rng)
        eps = jax.random.normal(rng_init, (dim,)) * self.marginal_std(
            self.train_cfg.t_max
        )
        t_min = t_min if t_min is not None else self.train_cfg.t_min
        t_max = t_max if t_max is not None else self.train_cfg.t_max
        ts = self.solver_cfg.solve_schedule(
            t_max=t_max, t_min=t_min, num_steps=num_steps
        )

        if sample_method == "ode":
            drift = self.solver_cfg.build_ode_drift(
                self,
                tokenizer,
                y=y,
                tokens_cfg=tokens_cfg,
                model_mask=model_mask,
                context=context,
                attention_mask=attention_mask,
            )
            out = odeint(
                drift, eps, ts, collect_trace=False, method=self.solver_cfg.ode_method
            )
            if last_euler_step and t_min is not None and t_min > 0.0:
                # One last Euler step at t_min
                dt = -ts[-1]  # ts are in decreasing order
                f_tmin = drift(ts[-1], out)
                out = out + f_tmin * dt
            return out
        elif sample_method == "sde":
            sde_drift, sde_diffusion = self.solver_cfg.build_sde(
                self,
                tokenizer,
                y=y,
                tokens_cfg=tokens_cfg,
                model_mask=model_mask,
                context=context,
                attention_mask=attention_mask,
            )
            out = sdeint(
                sde_drift,
                sde_diffusion,
                eps,
                ts,
                rng=rng,
                method=self.solver_cfg.sde_method,
            )
            if last_euler_step and t_min is not None and t_min > 0.0:
                # One last Euler step at t_min
                dt = -ts[-1]  # ts are in decreasing order
                f_tmin = sde_drift(ts[-1], out)
                g_tmin = sde_diffusion(ts[-1], out)
                z = jax.random.normal(rng, out.shape)
                out = out + f_tmin * dt + g_tmin * jnp.sqrt(-dt) * z
            return out

        else:
            raise ValueError(f"Sample method {sample_method} not recognized.")

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
        theta_mask = (
            jnp.ones_like(x, dtype=jnp.bool_)
            if model_mask is None
            else tokenizer.simulator.theta_mask(model_mask)
        )

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
            data, _ = state
            dx_dt = dx_dt_fn(t, data)
            jacobian_diagonal = jnp.diagonal(jax.jacrev(lambda z: dx_dt_fn(t, z))(data))
            div = jnp.sum(jacobian_diagonal * theta_mask)
            return (dx_dt, div)

        x_final = odeint(
            drift,
            (x, logp0),
            ts,
            method="heun",
            collect_trace=False,
        )
        x_final, logp_final = x_final

        sigma = self.marginal_std(t_max)
        base_logp = masked_standard_normal_log_prob(x_final, sigma, theta_mask)
        final = logp_final + base_logp

        return jnp.squeeze(final)
