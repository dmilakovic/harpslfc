#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lfc.fitting.gp_tinygp

tinygp/JAX-based replacement for gp.py's gaussian_process_smooth, migrating
one component at a time (dispersion first, as agreed) rather than
converting everything before any of it is validated.

Kept as a SEPARATE module from gp.py deliberately, while the migration is
still in progress: width and envelope/background still call gp.py's
gaussian_process_smooth, and this file coexists with it rather than
replacing it outright until each component has been converted and checked
in turn.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels
from scipy.optimize import minimize


def rational_quadratic_smooth(
    x: np.ndarray,
    z: np.ndarray,
    z_err: np.ndarray,
    x_grid: np.ndarray,
    n_restarts: int = 3,
    clip_sigma: float = 4.0,
    min_length_scale: float = None,
    max_length_scale: float = None,
    fixed_length_scale: float = None,
    fixed_alpha: float = None,
) -> dict:
    """
    tinygp/JAX equivalent of gp.py's gaussian_process_smooth, using a
    Rational Quadratic kernel instead of Matern-3/2.

    Chosen for dispersion specifically because RQ has a natural
    interpretation as a continuous MIXTURE of RBF kernels across a range
    of length scales (controlled by its alpha parameter): as alpha -> inf
    it reduces to a single-scale RBF, and smaller alpha blends in
    progressively shorter scales. This matches "smooth, almost
    polynomial, but might have sharper features" without needing to
    commit to a SPECIFIC second length scale the way the width model's
    explicit sum of two kernels does -- the data determines how much
    short-scale mixture is needed via alpha, rather than that being
    fixed in advance.

    Interface matches gp.py's gaussian_process_smooth exactly (same
    parameters, same returned dict keys) so it can be swapped in directly
    at call sites without touching anything downstream.

    Uses exact JAX gradients for the hyperparameter optimisation (via
    jax.grad through tinygp's log_probability), rather than gp.py's
    gradient-free L-BFGS-B-on-a-finite-difference-approximated-surface --
    this should be both faster and less prone to landing on a poor local
    optimum purely from noisy gradient estimates.
    """
    finite = np.isfinite(z) & np.isfinite(z_err) & (z_err > 0)
    x_f, z_f, e_f = x[finite], z[finite], z_err[finite]
    if len(x_f) < 4:
        return {
            'x_grid': x_grid, 'z_mean': np.full_like(x_grid, np.nan),
            'z_std': np.full_like(x_grid, np.nan),
            'length_scale': np.nan, 'signal_std': np.nan, 'n_used': 0,
            'alpha': np.nan,
        }

    # Sigma-clipping: identical scheme to gp.py, for consistency between
    # the two while both are in use during the migration.
    sort_idx = np.argsort(x_f)
    x_s, z_s, e_s = x_f[sort_idx], z_f[sort_idx], e_f[sort_idx]
    win = max(5, len(x_s) // 10)
    z_baseline = np.array([
        np.median(z_s[max(0, i - win):i + win + 1]) for i in range(len(z_s))
    ])
    dev = np.abs(z_s - z_baseline)
    mad = np.median(dev)
    sigma_scale = 1.4826 * mad if mad > 0 else np.std(z_s - z_baseline)
    good = dev <= clip_sigma * max(sigma_scale, np.median(e_s))
    x_c, z_c, e_c = x_s[good], z_s[good], e_s[good]
    if len(x_c) < 4:
        x_c, z_c, e_c = x_s, z_s, e_s
    n_used = len(x_c)

    # Normalise for numerical stability, same convention as gp.py.
    x_range = x_c.max() - x_c.min() if x_c.max() - x_c.min() > 0 else 1.0
    z_range = z_c.max() - z_c.min() if z_c.max() - z_c.min() > 0 else 1.0
    x_mu, x_n = x_c.mean(), (x_c - x_c.mean()) / x_range
    z_mu, z_n = z_c.mean(), (z_c - z_c.mean()) / z_range
    e_n = e_c / z_range
    x_grid_n = (x_grid - x_mu) / x_range

    x_n_j, z_n_j, e_n_j = jnp.asarray(x_n), jnp.asarray(z_n), jnp.asarray(e_n)

    default_log_l_bounds = (-3.0, 1.0)
    log_l_lower = (np.log(min_length_scale / x_range) if min_length_scale is not None
                   else default_log_l_bounds[0])
    log_l_upper = (np.log(max_length_scale / x_range) if max_length_scale is not None
                   else default_log_l_bounds[1])
    log_a_bounds = (-4.0, 2.0)
    log_alpha_bounds = (-2.0, 3.0)  # alpha from ~0.14 (strongly multi-scale)
                                      # to ~20 (close to a single-scale RBF)

    # If both are frozen, no optimisation is needed at all: skip straight
    # to the posterior mean/std at the fixed hyperparameters. This is the
    # RQ equivalent of gp.py's width freeze-after-warmup mechanism --
    # necessary here because RQ's extra alpha parameter, on top of the
    # length scale, made hyperparameter fitting considerably less stable
    # between repeated calls (confirmed directly on real dispersion data:
    # length scale ranging from 0.5 to 25 nm across 6 outer iterations,
    # where the single-parameter Matern-3/2 kernel settled on the same
    # value every time).
    both_frozen = fixed_length_scale is not None and fixed_alpha is not None
    if both_frozen:
        log_length_norm = np.log(fixed_length_scale / x_range)
    param_bounds = [log_a_bounds, (log_l_lower, log_l_upper), log_alpha_bounds]

    def build_gp(params):
        log_amp, log_length, log_alpha = params
        kernel = jnp.exp(2 * log_amp) * kernels.RationalQuadratic(
            alpha=jnp.exp(log_alpha), scale=jnp.exp(log_length))
        return GaussianProcess(kernel, x_n_j, diag=e_n_j**2)

    @jax.jit
    def neg_log_likelihood(params):
        return -build_gp(params).log_probability(z_n_j)

    grad_fn = jax.jit(jax.grad(neg_log_likelihood))

    rng = np.random.default_rng(0)

    if not both_frozen:
        # Neither frozen: full 3-parameter search, as before.
        log_l_mid = 0.5 * (log_l_lower + log_l_upper)
        init_guesses = [[0.0, log_l_mid, 0.0]]
        for _ in range(n_restarts - 1):
            init_guesses.append([
                rng.uniform(-2, 1),
                rng.uniform(log_l_lower, log_l_upper),
                rng.uniform(log_alpha_bounds[0], log_alpha_bounds[1]),
            ])
        best_nll, best_params = np.inf, np.array(init_guesses[0])
        for p0 in init_guesses:
            res = minimize(
                lambda p: float(neg_log_likelihood(jnp.asarray(p))), p0,
                jac=lambda p: np.asarray(grad_fn(jnp.asarray(p))),
                method='L-BFGS-B', bounds=param_bounds,
                options={'maxiter': 200, 'ftol': 1e-10},
            )
            if res.success and res.fun < best_nll:
                best_nll, best_params = res.fun, res.x
    else:
        # Both frozen: only the amplitude still needs fitting, a much
        # simpler and better-behaved 1D search.
        log_alpha_fixed = np.log(fixed_alpha)

        def neg_log_likelihood_amp_only(log_amp):
            return neg_log_likelihood(jnp.array([log_amp[0], log_length_norm, log_alpha_fixed]))

        grad_amp_only = jax.jit(jax.grad(lambda p: neg_log_likelihood_amp_only(p)))
        best_nll, best_log_amp = np.inf, 0.0
        for p0 in ([0.0] if n_restarts <= 1 else
                   [0.0] + list(rng.uniform(-2, 1, n_restarts - 1))):
            res = minimize(
                lambda p: float(neg_log_likelihood_amp_only(jnp.asarray(p))), [p0],
                jac=lambda p: np.asarray(grad_amp_only(jnp.asarray(p))),
                method='L-BFGS-B', bounds=[log_a_bounds],
                options={'maxiter': 200, 'ftol': 1e-10},
            )
            if res.success and res.fun < best_nll:
                best_nll, best_log_amp = res.fun, res.x[0]
        best_params = np.array([best_log_amp, log_length_norm, log_alpha_fixed])

    log_amp_opt, log_length_opt, log_alpha_opt = best_params
    gp_train = build_gp(best_params)
    cond_gp = gp_train.condition(z_n_j, jnp.asarray(x_grid_n)).gp
    z_mean_n = np.asarray(cond_gp.mean)
    z_var_n = np.asarray(cond_gp.variance)

    z_mean = z_mean_n * z_range + z_mu
    z_std = np.sqrt(np.maximum(z_var_n, 0.0)) * z_range

    return {
        'x_grid': x_grid,
        'z_mean': z_mean,
        'z_std': z_std,
        'length_scale': float(np.exp(log_length_opt) * x_range),
        'signal_std': float(np.exp(log_amp_opt) * z_range),
        'alpha': float(np.exp(log_alpha_opt)),
        'n_used': n_used,
    }
