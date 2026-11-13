"""Multi-scale preconditioned kernel Stein discrepancy between posterior samples
and the true posterior, with a wild-bootstrap p-value."""

import jax
import jax.numpy as jnp


def multiscale_preconditioned_ksd_and_pvalue(
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
    scores_z = scores_theta @ P_inv_T  # (N, D)
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
