"""Shared dMRI calibration statistics, KSD kernels and result summaries.

The mask and summed-parameter PIT statistics follow the frozen ICML evaluator.
Coordinate midpoint ranks use the same parameter draws and can be pooled over
active coordinates. KSD values retain the frozen kernel's square-root convention;
prior normalization is performed over the appropriate mask group.

For rows transformed by thetas @ P.T, the corrected kernel transforms scores
with scores_theta @ inverse(P). The frozen kernel uses the transposed score
transform. Kernel selection is explicit; the frozen kernel is the default.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from prism.inference.ksd import multiscale_preconditioned_ksd_and_pvalue


def _randomized_pit(samples, truth, tie_key, jitter_key, *, rtol, atol):
    """Randomize ties and rank jitter along the leading sample dimension."""
    equal = jnp.isclose(samples, truth, rtol=rtol, atol=atol)
    less = (samples < truth) & ~equal
    n_less = jnp.sum(less, axis=0)
    n_equal = jnp.sum(equal, axis=0)
    tie = jax.random.randint(tie_key, n_less.shape, 0, n_equal + 1)
    jitter = jax.random.uniform(jitter_key, n_less.shape, minval=0.0, maxval=1.0)
    return (n_less + tie + jitter) / (samples.shape[0] + 1)


def _validate_sampler(theta_samples, num_steps, t_min, *, minimum_samples=1):
    if theta_samples < minimum_samples:
        raise ValueError(f"theta_samples must be at least {minimum_samples}.")
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if t_min is not None and t_min < 0:
        raise ValueError("t_min must be nonnegative or None.")


def make_model_sbc_fn(model, mask_samples=200, *, source_keys=False):
    """Return the model-probability PIT without sampling any parameters.

    The first of three outer RNG streams matches ``make_sbc_fn`` exactly.
    ``source_keys=True`` skips that split and directly uses the supplied PIT key.
    Non-finite model log probabilities produce NaN for the runner to reject.
    The model must contain the selected checkpoint weights and be in eval mode.
    """
    if mask_samples < 1:
        raise ValueError("mask_samples must be positive.")

    def evaluate(data, key):
        mask_key = key if source_keys else jax.random.split(key, 3)[0]
        sample_key, tie_key, jitter_key = jax.random.split(mask_key, 3)
        masks = jax.vmap(
            lambda k: model.sample_mask(
                k, acq=data["acq"], x=data["x"], mask_prior=data["mask_prior"]
            )
        )(jax.random.split(sample_key, mask_samples))

        def log_prob_mask(mask):
            return model.log_prob_mask(
                mask,
                acq=data["acq"],
                x=data["x"],
                mask_prior=data["mask_prior"],
            )

        true_log_probs = jnp.reshape(log_prob_mask(data["model_mask"]), (-1,))
        sample_log_probs = jnp.reshape(
            jax.vmap(log_prob_mask)(masks), (mask_samples, -1)
        )
        true_log_prob = true_log_probs.sum()
        sample_log_prob = sample_log_probs.sum(1)
        valid_log_probs = (
            jnp.all(jnp.isfinite(true_log_probs))
            & jnp.all(jnp.isfinite(sample_log_probs))
            & jnp.isfinite(true_log_prob)
            & jnp.all(jnp.isfinite(sample_log_prob))
        )
        model_pit = _randomized_pit(
            sample_log_prob,
            true_log_prob,
            tie_key,
            jitter_key,
            rtol=1e-3,
            atol=1e-3,
        )
        return {"model_pit": jnp.where(valid_log_probs, model_pit, jnp.nan)}

    return jax.jit(evaluate)


def make_theta_sbc_fn(
    model, theta_samples=50, num_steps=64, t_min=None, *, source_keys=False
):
    """Return active-parameter sum and coordinate PITs without model sampling.

    Parameters are conditioned on the known generating model structure. The
    second outer RNG stream, ODE settings, tie handling, and coordinate midpoint
    ranks match ``make_sbc_fn``. ``source_keys=True`` directly uses the supplied
    PIT key. The paper recipe passes ``t_min=1e-4``; ``None``
    retains the loaded model's diffusion bound. No final Euler step is taken.
    Inactive coordinate PITs are NaN by design; non-finite active values invalidate
    both parameter statistics so the runner cannot silently discard bad trials.
    """
    _validate_sampler(theta_samples, num_steps, t_min)

    def evaluate(data, key):
        theta_key = key if source_keys else jax.random.split(key, 3)[1]
        sample_key, tie_key, jitter_key = jax.random.split(theta_key, 3)
        samples = jax.vmap(
            lambda k: model.sample_theta(
                k,
                acq=data["acq"],
                x=data["x"],
                model_mask=data["model_mask"],
                sample_method="ode",
                num_steps=num_steps,
                last_euler_step=False,
                t_min=t_min,
            )
        )(jax.random.split(sample_key, theta_samples))
        theta_mask = model.tokenizer.simulator.theta_mask(data["model_mask"])
        active_samples = jnp.where(theta_mask, samples, 0)
        active_truth = jnp.where(theta_mask, data["theta"], 0)
        sample_sums = jnp.sum(active_samples, axis=-1)
        truth_sum = jnp.sum(active_truth, axis=-1)
        valid_theta = (
            jnp.all(jnp.isfinite(active_samples))
            & jnp.all(jnp.isfinite(active_truth))
            & jnp.all(jnp.isfinite(sample_sums))
            & jnp.isfinite(truth_sum)
        )
        theta_sum_pit = _randomized_pit(
            sample_sums,
            truth_sum,
            tie_key,
            jitter_key,
            rtol=1e-6,
            atol=1e-6,
        )
        theta_pit = (0.5 + jnp.sum(samples < data["theta"], axis=0)) / (
            theta_samples + 1
        )
        return {
            "theta_sum_pit": jnp.where(valid_theta, theta_sum_pit, jnp.nan),
            "theta_pit": jnp.where(theta_mask & valid_theta, theta_pit, jnp.nan),
            "theta_mask": theta_mask,
        }

    return jax.jit(evaluate)


def make_sbc_fn(model, mask_samples=200, theta_samples=50, num_steps=64, t_min=None):
    """Combine the two SBC heads using their original independent RNG streams.

    The split-head factories default to the original combined evaluator's
    outer split. Model/parameter sampling and all output fields are unchanged.
    """
    model_pit = make_model_sbc_fn(model, mask_samples=mask_samples)
    theta_pit = make_theta_sbc_fn(
        model, theta_samples=theta_samples, num_steps=num_steps, t_min=t_min
    )

    def evaluate(data, key):
        return {**model_pit(data, key), **theta_pit(data, key)}

    return jax.jit(evaluate)


def make_ksd_fn(
    model,
    sim_type,
    theta_samples=50,
    num_steps=128,
    t_min=0.005,
    bandwidths=(0.1, 0.5, 1.0, 10.0),
    kernel=None,
):
    """Return a JIT-compiled ``evaluate(data, key)`` for raw posterior/prior KSD.

    Returns ``posterior_ksd``, ``prior_ksd``, ``posterior_pvalue``, and
    ``prior_pvalue``. Required data keys are ``acq``, ``x``, and ``model_mask``.
    All values are scalars. These are sqrt(max(KSD squared U-statistic, 0)),
    exactly as returned by the frozen kernel, despite its KSD-squared docstring.
    They are neither squared nor normalized by the individual voxel's prior.

    Defaults preserve frozen KSD sampling: ODE, 128 steps, t_min=0.005, no
    final Euler step, and 512 wild-bootstrap draws in the shared kernel.
    ``key`` splits into independent posterior/prior keys, each then split into
    a sampling key and a bootstrap key in the frozen evaluator's order.
    The model must already contain the selected weights and be in eval mode.
    ``kernel=None`` preserves the frozen kernel. A supplementary diagnostic may
    explicitly supply a compatible kernel; its runner must record that choice.
    """
    _validate_sampler(theta_samples, num_steps, t_min, minimum_samples=2)
    bandwidths = tuple(bandwidths)
    if not bandwidths or any(bandwidth <= 0 for bandwidth in bandwidths):
        raise ValueError("bandwidths must contain positive values.")
    bandwidth_array = jnp.asarray(bandwidths)
    ksd_kernel = multiscale_preconditioned_ksd_and_pvalue if kernel is None else kernel

    def evaluate(data, key):
        posterior_key, prior_key = jax.random.split(key)
        sample_key, evaluation_key = jax.random.split(posterior_key)
        thetas = jax.vmap(
            lambda k: model.sample_theta(
                k,
                acq=data["acq"],
                x=data["x"],
                model_mask=data["model_mask"],
                sample_method="ode",
                num_steps=num_steps,
                last_euler_step=False,
                t_min=t_min,
            )
        )(jax.random.split(sample_key, theta_samples))
        posterior_ksd, posterior_pvalue = ksd_kernel(
            sim_type,
            thetas=thetas,
            mask=data["model_mask"],
            acq=data["acq"],
            x=data["x"],
            key=evaluation_key,
            bandwidths=bandwidth_array,
        )

        sample_key, evaluation_key = jax.random.split(prior_key)
        prior_thetas = jax.random.normal(
            sample_key, (theta_samples, sim_type.theta_dim)
        )
        prior_ksd, prior_pvalue = ksd_kernel(
            sim_type,
            thetas=prior_thetas,
            mask=data["model_mask"],
            acq=data["acq"],
            x=data["x"],
            key=evaluation_key,
            bandwidths=bandwidth_array,
        )
        return {
            "posterior_ksd": posterior_ksd,
            "prior_ksd": prior_ksd,
            "posterior_pvalue": posterior_pvalue,
            "prior_pvalue": prior_pvalue,
        }

    return jax.jit(evaluate)


def mask_label(mask):
    ball, *sticks = np.asarray(mask, dtype=bool)[:4]
    n = sum(sticks)
    family = ("B" if ball else "") + (f"S{n}" if n else "")
    return f"{family or 'empty'} [{''.join(str(int(v)) for v in mask[:4])}]"


def coverage(pits):
    values = np.asarray(pits).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    grid = np.linspace(0, 1, 101)
    cdf = np.mean(values[:, None] <= grid, axis=0)
    return {
        "count": int(values.size),
        "grid": grid.tolist(),
        "cdf": cdf.tolist(),
        "calibration_error": float(np.abs(cdf - grid).mean()),
    }


def summarize(arrays):
    summary = {"datasets": len(arrays["model_mask"]), "per_mask": []}
    for field in ("model_pit", "theta_sum_pit", "theta_pit"):
        if field in arrays:
            summary[field] = coverage(arrays[field])
    masks = arrays["model_mask"].astype(bool)
    for mask in np.unique(masks, axis=0):
        selected = np.all(masks == mask, axis=1)
        row = {
            "mask": mask.astype(int).tolist(),
            "label": mask_label(mask),
            "count": int(selected.sum()),
        }
        for field in ("model_pit", "theta_sum_pit", "theta_pit"):
            if field in arrays:
                row[field] = coverage(arrays[field][selected])
        if "posterior_ksd" in arrays:
            posterior = arrays["posterior_ksd"][selected]
            prior = arrays["prior_ksd"][selected]
            denominator = float(prior.mean())
            if denominator <= 0 or not np.isfinite(denominator):
                raise ValueError(f"Invalid prior KSD denominator for {row['label']}")
            ratios = posterior / denominator
            cutoff = np.quantile(ratios, 0.99)
            row.update({
                "posterior_ksd_mean": float(posterior.mean()),
                "prior_ksd_mean": denominator,
                "rksd_mean": float(ratios.mean()),
                "rksd_median": float(np.median(ratios)),
                "rksd_quantiles": np.quantile(
                    ratios, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
                ).tolist(),
                "rksd_fraction_below_one": float(np.mean(ratios < 1)),
                "plot_trim_quantile": 0.99,
                "plot_trim_cutoff": float(cutoff),
                "posterior_ksd_zero_count": int(np.sum(posterior == 0)),
                "prior_ksd_zero_count": int(np.sum(prior == 0)),
            })
        summary["per_mask"].append(row)
    return summary


def validate_batch(values):
    for name, array in values.items():
        if name == "theta_pit":
            if not np.all(np.isfinite(array[values["theta_mask"].astype(bool)])):
                raise FloatingPointError("Non-finite active parameter PIT")
        elif not np.isfinite(array).all():
            raise FloatingPointError(f"Non-finite {name}; saved the offending batch")


# Supplementary correction; the default frozen kernel above is unchanged.
def corrected_multiscale_ksd_and_pvalue(
    sim_type,
    thetas,
    mask,
    acq,
    x,
    key,
    bandwidths,
    sigma=0.1,
    precond_matrix=None,
    n_bootstrap: int = 512,
    eps: float = 1e-8,
):
    """
    Multi-scale, preconditioned Kernel Stein Discrepancy (KSD^2) + wild-bootstrap p-value,
    restricted to the dimensions indicated by `theta_mask`.

    This version is written to reduce peak memory usage:
      - avoids (N, N, D) tensors (no explicit pairwise differences),
      - avoids (M, N, N) tensor across bandwidths,
      - avoids (N_bootstrap, N, N) intermediates in the bootstrap.
    """

    thetas = jnp.asarray(thetas)
    n, d = thetas.shape
    dtype = thetas.dtype

    # theta_mask indicates which dimensions are relevant (shape: (D,))
    theta_mask = sim_type.theta_mask(mask).astype(dtype)  # (D,)
    d_eff = jnp.sum(theta_mask)  # effective dimension (scalar)

    # ---------- log posterior ----------
    def log_posterior(theta, mask, acq, x):
        simulator = sim_type.from_theta(theta, model_mask=mask)
        prior = jax.scipy.stats.norm.logpdf(theta, loc=0.0, scale=1.0)
        # Only apply prior on masked dimensions
        prior = jnp.sum(prior * theta_mask, axis=-1)
        ll = simulator.log_likelihood(acq, x)
        return ll + prior

    # ---------- 0. Build / use preconditioner P ----------
    if precond_matrix is None:
        # Empirical covariance over *masked* dimensions
        theta_mean = jnp.mean(thetas, axis=0)
        theta_centered = thetas - theta_mean
        theta_centered_masked = theta_centered * theta_mask  # (N, D)

        cov = (theta_centered_masked.T @ theta_centered_masked) / jnp.maximum(
            n - 1, 1
        )  # (D, D)

        jitter = 1e-3 * jnp.eye(d, dtype=dtype)
        L = jnp.linalg.cholesky(cov + jitter)

        # Whitening: P = L^{-1}
        P = jnp.linalg.inv(L)
        # For score transform we need P^{-T} = (P^{-1})^T = L^T
        P_inv_T = L.T
    else:
        P = jnp.asarray(precond_matrix, dtype=dtype)
        P_inv_T = jnp.linalg.inv(P).T

    # Normalize bandwidths
    bandwidths = jnp.atleast_1d(bandwidths).astype(dtype)
    h2s = jnp.maximum(bandwidths**2, eps)  # (M,)
    m = h2s.shape[0]

    # ---------- 1. Score in theta, then transform to z coordinates ----------
    def score_theta(theta):
        g = jax.grad(log_posterior, argnums=0)(theta, mask, acq, x)  # (D,)
        return g * theta_mask

    scores_theta = jax.vmap(score_theta)(thetas)  # (N, D)
    scores_z = scores_theta @ P_inv_T.T  # (N, D)
    scores_z = scores_z * theta_mask  # (N, D)

    # ---------- 2. Transform samples to z-space for distances ----------
    z = thetas @ P.T  # (N, D)
    z_masked = z * theta_mask  # (N, D)

    # Pairwise squared distances using Gram trick (no (N, N, D))
    z_sq_norm = jnp.sum(z_masked**2, axis=1)  # (N,)
    # sq_dist_z[i,j] = ||z_i||^2 + ||z_j||^2 - 2 z_i·z_j
    gram_z = z_masked @ z_masked.T  # (N, N)
    sq_dist_z = z_sq_norm[:, None] + z_sq_norm[None, :] - 2.0 * gram_z  # (N, N)
    sq_dist_z = jnp.maximum(sq_dist_z, 0.0)  # numerical safety

    # Precompute Stein-related Gram matrices (all (N, N))
    S = scores_z  # (N, D)
    gram_s = S @ S.T  # (N, N)  term1: s_i^T s_j
    SZT = S @ z_masked.T  # (N, N)  A_ij = s_i^T z_j
    b = jnp.diag(SZT)  # (N,)   b_i = s_i^T z_i

    # ---------- 3. Multi-scale Stein kernel (memory-friendly) ----------
    def body_h2(i, H):
        h2 = h2s[i]

        # RBF kernel in z
        k = jnp.exp(-sq_dist_z / (2.0 * h2))  # (N, N)

        # Term 1: s_i^T s_j * k_ij
        term1 = gram_s * k  # (N, N)

        # Term 2 and 3 via analytic forms (no (N, N, D) tensors):
        # term2(i,j) = k/h2 * (s_i^T z_i - s_i^T z_j) = k/h2 * (b_i - SZT_ij)
        # term3(i,j) = -k/h2 * (s_j^T z_i - s_j^T z_j) = -k/h2 * (SZT_ji - b_j)
        k_over_h2 = k / h2
        term2 = k_over_h2 * (b[:, None] - SZT)  # (N, N)
        term3 = -k_over_h2 * (SZT.T - b[None, :])  # (N, N)

        # Term 4: trace Hessian
        # trace = k * (d_eff / h2 - ||z_i - z_j||^2 / h2^2)
        trace_hess = k * (d_eff / h2 - sq_dist_z / (h2**2))

        # Weight for multi-scale combination
        weight = sigma * 1 / len(h2s)
        H_update = weight * (term1 + term2 + term3 + trace_hess)

        return H + H_update

    H0 = jnp.zeros_like(sq_dist_z)
    H = jax.lax.fori_loop(0, m, body_h2, H0)  # (N, N)

    # Remove diagonal for U-statistic
    diag_H = jnp.diag(jnp.diag(H))
    H_no_diag = H - diag_H
    denom = n * (n - 1)
    ksd2_obs = jnp.sum(H_no_diag) / denom
    ksd = jnp.sqrt(jnp.maximum(ksd2_obs, 0.0))

    # ---------- 4. Wild bootstrap (memory-friendly) ----------
    def bootstrap_body(i, state):
        key_i, count = state
        key_i, subkey = jax.random.split(key_i)

        # Rademacher ±1
        xi = 2.0 * jax.random.bernoulli(subkey, 0.5, shape=(n,)) - 1.0  # (N,)

        # T_star = xi^T H_no_diag xi / (n(n-1))
        Hx = H_no_diag @ xi  # (N,)
        T_star = jnp.dot(xi, Hx) / denom

        count = count + (T_star >= ksd2_obs)
        return (key_i, count)

    init_state = (key, jnp.array(0.0, dtype=dtype))
    _, ge_count = jax.lax.fori_loop(0, n_bootstrap, bootstrap_body, init_state)
    p_value = (1.0 + ge_count) / (n_bootstrap + 1.0)

    return ksd, p_value
