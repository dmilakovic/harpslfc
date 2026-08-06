#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
from   harps.core import np, plt
import harps.lsf.aux as aux
import jax
import jax.numpy as jnp
import jaxopt
import tinygp
import scipy
# from   tinygp import kernels, GaussianProcess, noise
import functools 
import gc
import logging
import ray
from . import gp_aux


# ── Loss functions ────────────────────────────────────────────────────────────

def loss_LSF(params: dict,
             x    : jnp.ndarray,
             # mask : jnp.ndarray,   # NEW: 1=real, 0=padded
             y    : jnp.ndarray,
             yerr : jnp.ndarray,
             # use_yerr : bool = False,
             use_scatter: bool = False,
             scatter : list = [],
             ) -> jnp.ndarray:
    """
    Negative log marginal likelihood for the LSF GP model.
    Padded points are excluded via mask (their err is 1e9, so they
    contribute negligibly even without masking — mask is a safety net).
    """
    use_scatter = bool(use_scatter)
    gp = build_LSF_GP(params,x,yerr, use_scatter,  scatter)
    return -gp.log_probability(y)

def loss_scatter(params: dict,
                 x    : jnp.ndarray,
                 # mask : jnp.ndarray,   # NEW: 1=real, 0=padded
                 y    : jnp.ndarray,
                 yerr : jnp.ndarray,
                 ) -> jnp.ndarray:
    """Negative log marginal likelihood including a scatter (noise floor) term."""
    gp = build_scatter_GP(params,x, yerr)
    return -gp.log_probability(y)

def run_lsf_optimization_local(theta_start  : dict,
                                x            : jnp.ndarray,
                                y            : jnp.ndarray,
                                yerr        : jnp.ndarray,
                                use_scatter  : bool,          # static bool
                                bounds       : tuple,
                                scatter_params: dict  = None, # scatter theta only
                                maxiter      : int   = 300,
                                ) -> tuple[dict, jnp.ndarray]:
    """
    Single L-BFGS-B run from one starting point.
    use_scatter is a static bool — resolved at trace time, not a JAX value.
    Supersedes: one call inside the loop of train_LSF_multistart_ray.
    vmapped over theta_start axis (axis 0) by vectorized_run_lsf_optimization_local.
    """
    scatter_list = scatter_params if scatter_params is not None else []
    lbfgsb = jaxopt.LBFGSB(
        fun     = functools.partial(loss_LSF,
                                    x          = x,
                                    y          = y,
                                    yerr       = yerr,
                                    use_scatter = use_scatter,
                                    scatter     = scatter_list),
        maxiter = maxiter,
        tol     = 1e-5,
    )
    result     = lbfgsb.run(theta_start, bounds=bounds)
    final_loss = loss_LSF(result.params, x, y, yerr,
                          use_scatter=use_scatter,
                          scatter=scatter_list)
    return result.params, final_loss


# vmap over the starts axis — this is the direct replacement for
# the Python loop inside the old (removed) train_LSF_multistart_ray.
# Only theta_start varies across the batch; everything else (including
# scatter_params and maxiter) is broadcast.
vectorized_run_lsf_optimization_local = jax.vmap(
    run_lsf_optimization_local,
    in_axes=(0, None, None, None, None, None, None, None)
)


# ── Single-segment multi-start trainer (replaces removed train_LSF_tinygp /
#    train_LSF_multistart_ray) ────────────────────────────────────────────────

def train_LSF_multistart(X          : jnp.ndarray,
                         Y          : jnp.ndarray,
                         Y_err      : jnp.ndarray,
                         scatter    = None,
                         num_starts : int = 4,
                         maxiter    : int = 300,
                         seed       : int = 0,
                         ) -> tuple[dict, jnp.ndarray]:
    """
    Multi-start L-BFGS-B fit of the LSF GP hyperparameters for ONE segment.

    This is the direct, currently-used replacement for the old scipy-backed
    train_LSF_tinygp / the no-longer-existing train_LSF_multistart_ray.
    Internally it vmaps `num_starts` random initial guesses through
    `run_lsf_optimization_local` (pure JAX, jaxopt.LBFGSB) and returns the
    best (lowest final loss) solution.

    Parameters
    ----------
    X, Y, Y_err : 1D arrays for one segment (no batching over segments here —
        for batched/GPU multi-segment fitting see fit_segment_phase /
        make_phase_fitter instead).
    scatter : None, or the 4-tuple (theta_scatter, logvar_x, logvar_y,
        logvar_err) returned by train_scatter_tinygp.
    """
    use_scatter  = scatter is not None
    scatter_list = list(scatter) if use_scatter else []

    A0     = jnp.nanmax(Y)
    x_std  = jnp.clip(jnp.nanstd(X), 0.3, 2.0)
    y_std  = jnp.clip(jnp.nanstd(Y), 1e-6, None)
    key    = jax.random.PRNGKey(seed)
    k_amp, k_loc, k_sig, k_gpamp = jax.random.split(key, 4)

    starts = dict(
        mf_amp       = A0 * (1.0 + 0.1  * jax.random.normal(k_amp, (num_starts,))),
        mf_loc       = 0.02 * jax.random.normal(k_loc, (num_starts,)),
        mf_log_sig   = jnp.log(x_std) + 0.1 * jax.random.normal(k_sig, (num_starts,)),
        mf_const     = jnp.zeros(num_starts),
        gp_log_amp   = jnp.log(y_std) + 0.3 * jax.random.normal(k_gpamp, (num_starts,)),
        gp_log_scale = jnp.zeros(num_starts),
        log_var_add  = jnp.full(num_starts, -5.0),
    )
    lower = dict(
        mf_amp       = 0.3 * A0,
        mf_loc       = -1.0,
        mf_log_sig   = jnp.log(1e-3),
        mf_const     = -0.5 * A0,
        gp_log_amp   = -6.0,
        gp_log_scale = -3.0,
        log_var_add  = -15.0,
    )
    upper = dict(
        mf_amp       = 2.0 * A0,
        mf_loc       = 1.0,
        mf_log_sig   = jnp.log(5.0),
        mf_const     = 0.5 * A0,
        gp_log_amp   = 6.0,
        gp_log_scale = 3.0,
        log_var_add  = 1.5,
    )
    bounds = (lower, upper)

    params_batched, losses_batched = vectorized_run_lsf_optimization_local(
        starts, X, Y, Y_err, use_scatter, bounds, scatter_list, maxiter
    )
    # Guard against a start that diverged to NaN loss
    safe_losses = jnp.where(jnp.isfinite(losses_batched),
                            losses_batched, jnp.inf)
    best_idx   = jnp.argmin(safe_losses)
    best_params = jax.tree_util.tree_map(lambda a: a[best_idx], params_batched)
    best_loss   = losses_batched[best_idx]
    return best_params, best_loss

# def _run_one_start(params0 : dict,
#                    x       : jnp.ndarray,
#                    y       : jnp.ndarray,
#                    yerr    : jnp.ndarray,
#                    use_yerr: bool,
#                    mask    : jnp.ndarray,
#                    bounds  : tuple[dict, dict],
#                    scatter : list = [],
#                    use_scatter : bool = False,
#                    maxiter : int = 300
#                    ) -> tuple[dict, jnp.ndarray]:
#     """
#     Run L-BFGS-B from a single starting point.
#     Pure JAX — vmappable over the starting-point axis.
#     """
    
#     # bounds = generate_bounds_batch(x, y, yerr)
    
#     solver = jaxopt.LBFGSB(
#         fun = functools.partial(
#                     loss_LSF,
#                     x = x, 
#                     y = y,
#                     yerr = yerr,
#                     use_yerr = use_yerr,
#                     scatter = scatter,
#                     use_scatter = use_scatter),
#         maxiter = maxiter,
#         tol     = 1e-5,
#     )
#     result = solver.run(params0, bounds=bounds)
#     # Evaluate loss at solution using the unchanged loss_LSF
#     final_loss = loss_LSF(result.params, x, y, yerr, use_yerr, scatter, use_scatter)
#     return result.params, final_loss

# ═════════════════════════════════════════════════════════════════════════════
# GPU-BATCH FITTING INFRASTRUCTURE — CURRENTLY UNUSED
#
# Everything from here down to train_scatter_batch (generate_starts_batch,
# generate_bounds_batch, SegmentState, _sigma_clip_mask, _recentering_iter,
# fit_segment_phase, make_phase_fitter, train_scatter_batch) was built to
# support a batched, jax.vmap'd, multi-GPU fitting pipeline driven by a
# Ray actor called GPUFitter in construct.py.
#
# That GPUFitter pipeline turned out to be dead code (it was shadowed by a
# second, later definition of from_spectrum_2d in the same module, so it
# was never actually reachable) and has been removed from construct.py.
# The pipeline that IS currently used (construct.from_spectrum_2d ->
# model_1d -> model_1s -> construct_tinygp -> train_LSF_multistart above)
# fits one segment at a time, not batched.
#
# This block is kept — rather than deleted — because it's a legitimate,
# mostly-correct sketch of how to do batched multi-segment GPU fitting
# later. It is currently NOT called from anywhere. If you revive it, wire
# it up to a corrected GPUFitter and test it end-to-end before trusting it.
# ═════════════════════════════════════════════════════════════════════════════

def generate_starts_batch(x_batch    : jnp.ndarray,   # (N_seg, max_len)
                          flx_batch  : jnp.ndarray,
                          err_batch  : jnp.ndarray,
                          num_starts : int = 4,
                          ) -> dict:
    """
    Generate starting guesses for all segments in a batch.

    Returns a dict where each leaf has shape (N_seg, num_starts),
    ready for the inner vmap over starts inside fit_one_segment.
    """
    import jax
    N_seg = x_batch.shape[0]
    # Generate starts for one segment, then vmap over segments
    def starts_for_one(x, flx, err):
        # Reuse your existing single-segment start generation logic here.
        # Returns dict with each leaf of shape (num_starts,)
        
        A0 = jnp.nanmax(flx) * (1 + jax.random.normal(jax.random.PRNGKey(2), (num_starts,)) * 0.1)
        s0 = jnp.clip(jnp.nanstd(x), 0.5, 2.0) * (1 + jax.random.normal(jax.random.PRNGKey(3), (num_starts,)) * 0.1)
        log_amp   = jnp.log(jnp.nanstd(flx))
        log_scale = jnp.log(jnp.nanstd(x) * 0.5)
        log_amp_starts   = log_amp   + jax.random.normal(
            jax.random.PRNGKey(0), (num_starts,)) * 0.5
        log_scale_starts = log_scale + jax.random.normal(
            jax.random.PRNGKey(1), (num_starts,)) * 0.3
        return {
            'mf_amp'   : A0,
            'mf_loc'   : jnp.zeros(num_starts),
            'mf_log_sig' : jnp.log(s0),
            'mf_const'   : jnp.zeros(num_starts),
            'gp_log_amp'  : log_amp_starts,
            'gp_log_scale': log_scale_starts,
            'log_var_add' : jnp.full(num_starts, -5.),
        }

    return jax.vmap(starts_for_one)(x_batch, flx_batch, err_batch)

def generate_bounds_batch(x_batch    : jnp.ndarray,   # (N_seg, max_len)
                          flx_batch  : jnp.ndarray,
                          err_batch  : jnp.ndarray,
                          num_starts : int = 4,
                          ) -> dict:
    """
    Generate starting guesses for all segments in a batch.

    Returns a dict where each leaf has shape (N_seg, num_starts),
    ready for the inner vmap over starts inside fit_one_segment.
    """
    import jax
    N_seg = x_batch.shape[0]
    # Generate starts for one segment, then vmap over segments
    def bounds_for_one(x, flx, err):
        A0 = jnp.nanmax(flx)
        lower = dict(
            mf_amp       = 0.8  * A0,
            mf_loc       = -1.0,
            mf_log_sig   = jnp.log(1e-10),
            mf_const     = -0.1,
            gp_log_amp   = -4.0,
            gp_log_scale = -1.0,
            log_var_add  = -15.0,
        )
        upper = dict(
            mf_amp       = 1.2  * A0,
            mf_loc       = +1.0,
            mf_log_sig   = jnp.log(2.0),
            mf_const     = +0.1,
            gp_log_amp   = +4.0,
            gp_log_scale = +1.0,
            log_var_add  = +1.5,
        )
        return lower, upper

    return jax.vmap(bounds_for_one)(x_batch, flx_batch, err_batch)

# NOTE: an earlier draft of a batched, vmap'd per-segment fitter
# (fit_one_segment / make_batch_fitter) used to live here. It was never
# called from anywhere and was internally inconsistent with
# run_lsf_optimization_local's real signature. It has been removed —
# fit_segment_phase / make_phase_fitter below is the correct, exercised
# replacement (used by construct.GPUFitter).

# ── Outlier mask update using existing get_residuals ─────────────────────────

def _update_mask_from_residuals(x        : jnp.ndarray,
                                y        : jnp.ndarray,
                                y_err    : jnp.ndarray,
                                theta    : dict,
                                mask     : jnp.ndarray,
                                scatter  = None,
                                sigma_clip: float = 4.0,
                                ) -> jnp.ndarray:
    """
    Recompute outlier mask using get_residuals (which uses build_LSF_GP).
    Padded points (mask=0) are always excluded regardless of residual.
    """
    # get_residuals uses build_LSF_GP internally — do not duplicate that logic
    rsd = get_residuals(x, y, y_err, theta, scatter)

    # Sigma-clip using only currently-good real points
    good      = mask > 0.5
    rsd_good  = jnp.where(good, rsd, 0.0)
    n_good    = jnp.maximum(jnp.sum(good), 1)
    mean_rsd  = jnp.sum(rsd_good) / n_good
    var_rsd   = jnp.sum(jnp.where(good, (rsd - mean_rsd)**2, 0.0)) / n_good
    sigma_rsd = jnp.sqrt(var_rsd)

    is_outlier = jnp.abs(rsd - mean_rsd) > sigma_clip * sigma_rsd
    # Outliers become 0, padded points stay 0, good points stay 1
    return jnp.where(is_outlier, 0.0, mask)



# ── Centre estimation using existing estimate_centre_anderson ─────────────────

def _get_centre(x      : jnp.ndarray,
                y      : jnp.ndarray,
                y_err  : jnp.ndarray,
                theta  : dict,
                scatter = None,
                ) -> tuple[float, float]:
    """
    Thin wrapper around estimate_centre_anderson which uses build_LSF_GP.
    Returns (shift, shift_err).
    """
    return estimate_centre_anderson(x, y, y_err, theta, scatter)



# ── Loop state ────────────────────────────────────────────────────────────────

from typing import NamedTuple

class SegmentState(NamedTuple):
    params     : dict
    shift      : jnp.ndarray   # total cumulative shift (scalar)
    mask       : jnp.ndarray   # good-point mask (max_len,): 1=good, 0=bad/padded
    delta      : jnp.ndarray   # |shift_j - shift_{j-1}|
    delta_prev : jnp.ndarray   # delta from j-2 (oscillation detection)
    shift_prev : jnp.ndarray   # shift from previous iteration
    converged  : jnp.ndarray   # bool scalar
    
def _sigma_clip_mask(rsd       : jnp.ndarray,
                     mask      : jnp.ndarray,
                     thresh    : float = 3.5,
                     ) -> jnp.ndarray:
    """
    JAX-compatible sigma clipping. No if/else — uses jnp.where throughout.
    Replaces hf.is_outlier_original + keep_full logic in model_1s.
    """
    good       = mask > 0.5
    rsd_good   = jnp.where(good, rsd, 0.0)
    n_good     = jnp.maximum(jnp.sum(good), 1.0)
    median_rsd = jnp.median(rsd_good, axis = 0)
    diff       = jnp.sqrt(jnp.sum((rsd - median_rsd)**2, axis = -1))
    med_abs_deviation = np.median(diff)
    
    modified_z_score = 0.6745 * diff / med_abs_deviation
    is_outlier = modified_z_score > thresh
    # sq_dev     = jnp.where(good, jnp.square(rsd - mean_rsd), 0.0)
    # sigma_rsd  = jnp.sqrt(jnp.sum(sq_dev) / n_good)
    # is_outlier = jnp.abs(rsd - mean_rsd) > sigma_clip * sigma_rsd
    
    
    # median = np.median(points, axis=0)
    # diff = np.sum((points - median)**2, axis=-1)
    # diff = np.sqrt(diff)
    # med_abs_deviation = np.median(diff)

    

    # return modified_z_score > thresh


    # Outliers become 0; padded points (already 0) stay 0
    return jnp.where(is_outlier, 0.0, mask)

# ── One recentering iteration ─────────────────────────────────────────────────


def _recentering_iter(i           : int,
                      state       : SegmentState,
                      x           : jnp.ndarray,
                      y           : jnp.ndarray,
                      y_err       : jnp.ndarray,
                      starts      : dict,
                      bounds      : tuple,
                      use_scatter : bool,
                      scatter_y_err: jnp.ndarray,
                      maxiter     : int,
                      delta_lim   : float,
                      shift_lim   : float,
                      ) -> SegmentState:
    """
    One recentering iteration. Body of lax.fori_loop.

    use_scatter is a static bool (resolved at trace time).
    scatter_y_err is the pre-rescaled Y_err when use_scatter=True,
    or the original Y_err when use_scatter=False.
    This avoids calling rescale_errors inside the loop.

    No if/else anywhere — all branching via jnp.where or static bools.
    """
    # When already converged: body is a no-op via jnp.where on all outputs
    # This replaces the 'break' in model_1s — lax.fori_loop has no break

    # Step 1: apply cumulative shift; clamp runaway (replaces np.abs(shift)>1 check)
    safe_shift  = jnp.where(jnp.abs(state.shift) > 1.0,
                             jnp.sign(state.shift) * 0.25,
                             state.shift)
    x_shifted   = x + safe_shift

    # Step 2: inflate errors for masked points (replaces keep_jm1 indexing)
    # use scatter_y_err which is already rescaled if use_scatter=True
    y_err_active = jnp.where(state.mask > 0.5, scatter_y_err, 1e9)

    # Step 3: multi-start L-BFGS-B
    # vectorized_run_lsf_optimization_local vmaps over starts axis
    params_batched, losses_batched = vectorized_run_lsf_optimization_local(
        starts, x_shifted, y, y_err_active, use_scatter, bounds
    )
    best_idx   = jnp.argmin(losses_batched)
    new_params = jax.tree_util.tree_map(lambda a: a[best_idx], params_batched)

    # Step 4: estimate centre via anderson method (uses build_LSF_GP internally)
    shift_j, _ = estimate_centre_anderson(
        x_shifted, y, y_err_active, new_params,
        # use_scatter=use_scatter, 
    )
    # Replace: if not np.isfinite(shift_j) → jnp.where
    shift_j = jnp.where(jnp.isfinite(shift_j), shift_j, 0.0)
    # Clamp runaway shifts
    shift_j = jnp.clip(shift_j, -1.0, 1.0)

    new_shift = safe_shift + shift_j

    # Step 5: update outlier mask (replaces hf.is_outlier_original)
    rsd      = get_residuals(x_shifted, y, y_err_active, new_params)
    new_mask = _sigma_clip_mask(rsd, state.mask)

    # Step 6: convergence check — stored as bool, replaces 'break'
    new_delta      = jnp.abs(shift_j - state.shift_prev)
    new_delta_prev = state.delta
    oscillating    = (new_delta == state.delta_prev)
    converged_now  = (
        (new_delta          < delta_lim) |
        (jnp.abs(new_shift) < shift_lim) |
        oscillating
    )
    new_converged = state.converged | converged_now

    # Step 7: no-op when converged — jnp.where selects old vs new
    # This is the JAX equivalent of 'break': zero extra compute after convergence
    def sel(new_val, old_val):
        return jnp.where(state.converged, old_val, new_val)

    return SegmentState(
        params     = jax.tree_util.tree_map(sel, new_params,  state.params),
        shift      = sel(new_shift,      state.shift),
        mask       = sel(new_mask,       state.mask),
        delta      = sel(new_delta,      state.delta),
        delta_prev = sel(new_delta_prev, state.delta_prev),
        shift_prev = sel(shift_j,        state.shift_prev),
        converged  = new_converged,
    )


def fit_segment_phase(x           : jnp.ndarray,   # (max_len,)
                      y           : jnp.ndarray,
                      y_err       : jnp.ndarray,   # original errors
                      mask        : jnp.ndarray,   # 1=real, 0=padded
                      starts      : dict,           # each leaf (num_starts,)
                      bounds      : tuple,
                      use_scatter : bool,           # STATIC — resolved at trace time
                      scatter_y_err: jnp.ndarray,  # rescaled errors (=y_err if no scatter)
                      numiter     : int = 5,
                      maxiter     : int = 300,
                      delta_lim   : float = 1e-3,
                      shift_lim   : float = 1e-3,
                      ) -> SegmentState:
    """
    One phase of the iterative fit (either Phase 1 or Phase 2).
    use_scatter is static — JAX compiles a different graph for each phase.

    scatter_y_err: pre-computed rescaled errors passed in from outside the loop.
    This is the key design decision: rescale_errors (which calls build_scatter_GP
    and gp.condition) is NOT inside the lax.fori_loop. It runs once at Python
    level before the loop starts, then the rescaled errors are passed in as
    a fixed array. This keeps the loop body pure and vmappable.
    """
    init_state = SegmentState(
        params     = jax.tree_util.tree_map(lambda s: s[0], starts),
        shift      = jnp.array(0.0),
        mask       = mask,
        delta      = jnp.array(jnp.inf),
        delta_prev = jnp.array(jnp.inf),
        shift_prev = jnp.array(0.0),
        converged  = jnp.array(False),
    )

    body = functools.partial(
        _recentering_iter,
        x            = x,
        y            = y,
        y_err        = y_err,
        starts       = starts,
        bounds       = bounds,
        use_scatter  = use_scatter,
        scatter_y_err= scatter_y_err,
        maxiter      = maxiter,
        delta_lim    = delta_lim,
        shift_lim    = shift_lim,
    )

    return jax.lax.fori_loop(0, numiter, body, init_state)


def make_phase_fitter(use_scatter : bool,
                      numiter     : int = 5,
                      maxiter     : int = 300,
                      ) -> callable:
    """
    Returns a jit+vmap compiled fitter for one phase.
    use_scatter is baked into the compiled graph at construction time.

    Two separate compiled functions are created:
        phase1_fitter = make_phase_fitter(use_scatter=False)
        phase2_fitter = make_phase_fitter(use_scatter=True)

    Signature:
        fitter(x, y, y_err, mask, starts, bounds, scatter_y_err)
            -> SegmentState  (all fields batched over N_seg)
    """
    def _wrapper(x, y, y_err, mask, starts, bounds, scatter_y_err):
        return fit_segment_phase(
            x, y, y_err, mask, starts, bounds,
            use_scatter   = use_scatter,   # static — baked in
            scatter_y_err = scatter_y_err,
            numiter       = numiter,
            maxiter       = maxiter,
        )

    vmapped = jax.vmap(_wrapper, in_axes=(0, 0, 0, 0, 0, 0, 0))
    return jax.jit(vmapped)


# ─────────────────────────────────────────────────────────────────────────────
# Scatter training across a batch of segments (CPU, between phases)
# ─────────────────────────────────────────────────────────────────────────────

def train_scatter_batch(x_batch      : jnp.ndarray,   # (N_seg, max_len)
                        y_batch      : jnp.ndarray,
                        y_err_batch  : jnp.ndarray,
                        mask_batch   : jnp.ndarray,
                        params_batch : dict,           # each leaf (N_seg, ...)
                        minpts       : int = 15,
                        ) -> tuple[list, jnp.ndarray]:
    """
    Train scatter GP for each segment using Phase 1 results.
    Runs on CPU — train_scatter_tinygp uses scipy (not vmappable).
    Returns:
        scatter_list    : list of scatter tuples, one per segment
        scatter_y_err   : (N_seg, max_len) rescaled errors for Phase 2
    """
    N_seg          = x_batch.shape[0]
    scatter_list   = []
    scatter_y_err  = np.array(y_err_batch)   # will be filled in-place

    for i in range(N_seg):
        # Extract this segment's data (unpadded points only)
        good   = np.array(mask_batch[i]) > 0.5
        x_i    = np.array(x_batch[i]   [good])
        y_i    = np.array(y_batch[i]   [good])
        ye_i   = np.array(y_err_batch[i][good])
        theta_i = jax.tree_util.tree_map(lambda a: a[i], params_batch)

        try:
            scatter_i = train_scatter_tinygp(
                x_i, y_i, ye_i, theta_i, minpts=minpts
            )
            scatter_list.append(scatter_i)

            # Rescale errors for Phase 2 — only for good (unpadded) points
            S_i, _ = rescale_errors(True, scatter_i, jnp.array(x_i), jnp.array(ye_i))
            scatter_y_err[i][good] = np.array(S_i)

        except Exception as e:
            # Scatter training failed — fall back to original errors
            scatter_list.append(None)
            # scatter_y_err[i] already holds original y_err

    return scatter_list, jnp.array(scatter_y_err)




















def make_dummy_scatter(X):
    theta_scatter = {
        "sct_log_const": 0.0,
        "sct_log_amp": -20.0,   # kills GP amplitude
        "sct_log_scale": 1.0,
        "sct_log_epsilon0": -10.0,
    }

    zeros = jnp.zeros_like(X)

    return (theta_scatter, zeros, zeros, zeros)
    
def get_model(x_test,X,Y,Y_err,theta,scatter=None):
    # print('get_model',*[np.shape(_) for _ in [x_test,X,Y,Y_err]])
    # print('get_model',theta)
    use_scatter = scatter is not None
    gp = build_LSF_GP(theta, X, Y_err,
                      use_scatter=use_scatter,
                      scatter=list(scatter) if use_scatter else [])
    _, cond = gp.condition(Y,x_test)
    model = cond.mean
    var   = jnp.sqrt(cond.variance)
    return model, var
    
    
def get_residuals(X,Y,Y_err,theta,scatter=None):
    '''
    Returns the residuals to the LSF model

    Parameters
    ----------
    X : array-like
        x-coordinates values.
    Y : array-like
        y-coordinates values.
    Y_err : array-like
        Standard deviation (error) on the y-coordinate values.
    theta : dictionary
        Parameters of the LSF model.
    scatter : tuple, optional
        Output of train_scatter_gp. The default is None.

    Returns
    -------
    rsd : TYPE
        Normalised residuals of the data to the model. 
        No rescaling is done internally on the errors. One may modify the
        Y_err before passing it to this function.

    '''
    model, variance = get_model(X, X, Y, Y_err, theta, None)
    rsd = jnp.array((Y - model)/Y_err)
    return rsd
    
def estimate_variance(X,Y,Y_err,theta,minpts,plot=False,ax=None):
    """
    Estimates the variance based on the residuals to the provided GP parameters
    
    The returned variance is in units of data variance! 
    One should multiply this variance with the variance on the data to get
    accurate results. 

    Parameters
    ----------
    X : jax array
        Contains the x-coordinates
    Y : jax array
        Contains the y-coordinates
    Y_err : jax array
        Contains the error on the y-coordinates.
    theta : dictionary
        Contains the LSF hyper-parameters.
    scale : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.
    minpts : TYPE
        DESCRIPTION.

    Returns
    -------
    logvar_x : TYPE
        DESCRIPTION.
    logvar_y : TYPE
        DESCRIPTION.

    """
    
    
    # Optimally bin the counts    
    counts, bin_edges = aux.bin_optimally(X,minpts)
    # Define bin centres
    # bin_cens = jnp.array((bin_edges[1:]+bin_edges[:-1])/2.)
    
    rsd = get_residuals(X,Y,Y_err,theta)
    # Calculate the relevant statistics
    calculate=['mean','std','sam_variance','sam_variance_variance',
               'pop_variance','pop_variance_variance']
    arrays = aux.get_bin_stat(X, rsd, bin_edges,calculate=calculate,
                              remove_outliers=True)
    # means = arrays['mean']
    # stds  = arrays['std']
    bin_cens = arrays['bin_centres']
    sam_var_ = arrays['sam_variance']
    sam_var_var = arrays['sam_variance_variance']
    # pop_var_ = arrays['pop_variance']
    # pop_var_var = arrays['pop_variance_variance']
    
    # Remove empty bins
    # try:
    #     cut = np.where(sam_var_!=0)[0]
    # except:
    # cut = jnp.where(sam_var_!=0,size=len(sam_var_))
    # cut = jnp.isfinite(sam_var_)
    # print(sam_var_)
    x_array     = bin_cens
    # pop_var     = pop_var_[cut]
    # pop_err     = jnp.sqrt(pop_var)  # error = sqrt of population variance
    sam_var     = sam_var_
    sam_err     = jnp.sqrt(sam_var)
    # pop_var_err = jnp.sqrt(pop_var_var[cut])
    sam_var_err = jnp.sqrt(sam_var_var)
    log_sam_var, log_sam_var_err = aux.lin2log(sam_var,sam_var_err)
    
    
    # plot_flag = plot | (ax is not None)
    # if plot_flag:
    #     if ax is not None:
    #         pass
    #     else:
    #         fig, ax = plt.subplots(1)
    #     ax.scatter(X,rsd,marker='o',s=3,label='rsd')
    #     ax.errorbar(bin_cens[cut],
    #                  np.zeros_like(bin_cens[cut]),
    #                  # means[cut],
    #                  sam_err[cut],
    #                  marker='s',ls='',c='red',
    #                  label = 'means')
    #     ax.errorbar(bin_cens[cut],sam_var,sam_var_err,marker='x',ls='',
    #                 capsize=2,c='C1')
    #     # for i in [-1,1]:
    #     #     for j in [1]:
    #     #         ax.plot(x_array, j*sam_var,color='r',lw=2)
            
    #     #         ax.fill_between(x_array, 
    #     #                           j*sam_var + i*sam_var_err,
    #     #                           j*sam_var - i*sam_var_err, 
    #     #                           color='red',alpha=0.3,zorder=10)
    #     # plt.xlim(-8,7)
    #     ax.set_xlabel("Distance from centre (pix)")
    #     ax.set_ylabel(r"$S^2 (\sigma^2)$")
    #     ax.legend()
        
    return x_array, log_sam_var, log_sam_var_err

def train_scatter_tinygp(X,Y,Y_err,theta_lsf,minpts=15,
                         include_error=True):
    '''
    Based on Kersting et al. 2007 :
        Most Likely Heteroscedastic Gaussian Process Regression

    '''

    x_array, log_var, err_log_var_ = estimate_variance(X,Y,Y_err,
                                                          theta_lsf,
                                                          minpts,plot=False)
    # print(x_array,log_var,err_log_var_)
    # err_log_var = None
    # if include_error:
    err_log_var = err_log_var_
    
    # print(f"Optimizing scatter parameters, err_log_variance = {err_log_variance}")
    theta = dict(
        sct_log_const  = -5.0,
        sct_log_amp    = -0.2,
        sct_log_scale  = 0.0,
        sct_log_epsilon0 = -3.,
        )
    lower_bounds = dict(
        sct_log_const  =-10.0,
        sct_log_amp    =-3.0,
        sct_log_scale  =-1.0,
        sct_log_epsilon0 = -15.,
        )
    upper_bounds = dict(
        sct_log_const  = 5.0,
        sct_log_amp    = 3.0,
        sct_log_scale  = 3.0,
        sct_log_epsilon0 = 3.,
        )
    bounds = (lower_bounds, upper_bounds)
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=functools.partial(loss_scatter,
                                                      x=x_array,
                                                      y=log_var,
                                                      yerr=err_log_var),
                                          method="l-bfgs-b")
    solution = lbfgsb.run(jax.tree.map(jnp.asarray, theta), bounds=bounds)
    # solver = jaxopt.GradientDescent(fun=functools.partial(loss_scatter,
    #                                           X=x_array,
    #                                           Y=log_var,
    #                                           Y_err=err_log_var,
    #                                           ))
    # solution = solver.run(jax.tree_map(jnp.asarray, theta))
    # print("Scatter solution:",solution.params)
    # print(f"Scatter final negative log likelihood: {solution.state.fun_val}")
    return solution.params, x_array, log_var, err_log_var

def get_scatter_covar(X,Y,Y_err,theta_lsf):
    gp = build_LSF_GP(theta_lsf, X, Y_err, use_scatter=False, scatter=[])
    _, cond = gp.condition(Y,X,include_mean=False)
    # mean_lsf = cond.loc
    # plt.plot(X,mean_lsf)
    return cond.covariance


    


def rescale_errors(use_scatter : bool,
                   scatter : list | tuple,
                   X : jnp.ndarray,
                   Y_err : jnp.ndarray
                   ) -> tuple[jnp.ndarray, jnp.ndarray]:
    '''
    JAX-compatible error rescaling.
    
    use_scatter : static bool — resolved at trace time
    scatter     : [theta_scatter, logvar_x, logvar_y, logvar_err]
                  or [] when not used
    
    When use_scatter=False, returns Y_err unchanged with zero variance.
    When use_scatter=True, rescales Y_err using the scatter GP.
    
    Both branches are always computed (JAX requirement for jnp.where),
    but the static use_scatter flag means only one branch is ever
    included in the compiled graph — so there is zero runtime cost
    for the unused branch.
    '''
    use_scatter = bool(use_scatter)
    # Branch 1: no scatter — identity, zero variance
    S_no_scatter    = Y_err
    S_var_no_scatter = jnp.zeros_like(Y_err)

    # Branch 2: scatter rescaling — always computed when use_scatter=True
    if use_scatter:
        theta_scatter, logvar_x, logvar_y, logvar_err = scatter
        sct_gp        = build_scatter_GP(theta_scatter, logvar_x, logvar_err)
        _, sct_cond   = sct_gp.condition(logvar_y, X)
        F_mean        = sct_cond.mean
        F_sigma       = jnp.sqrt(sct_cond.variance)
        # S_scatter     = Y_err * jnp.exp(F_mean / 2.)
        # deriv         = jax.grad(
        #     lambda x: sct_gp.condition(logvar_y, jnp.atleast_1d(x))[1].mean[0]
        # )
        # dFdx          = jax.vmap(deriv)(X)
        # S_var_scatter = jnp.square(S_scatter / 2. * dFdx * F_sigma)
        S_scatter, S_var_scatter = transform(X,Y_err,F_mean,F_sigma,sct_gp,logvar_y)
    else:
        S_scatter     = S_no_scatter
        S_var_scatter = S_var_no_scatter

    return S_scatter, S_var_scatter
    

def plot_variance_GP(scatter,X,Y_err,plot=False,ax=None):
    theta_scatter, logvar_x, logvar_y, logvar_err = scatter
    
    
    sct_gp        = build_scatter_GP(theta_scatter,logvar_x,logvar_err)
    _, sct_cond   = sct_gp.condition(logvar_y,X)
    F_mean  = sct_cond.mean
    F_sigma = jnp.sqrt(sct_cond.variance)
    
    S, S_var = transform(X,Y_err,F_mean,F_sigma,sct_gp,logvar_y)
    
    plot_flag = plot | (ax is not None)
    if plot_flag:
        import matplotlib.ticker as ticker
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots(1)
        X_grid = jnp.linspace(X.min(),X.max(),200)
        
        
        _, sct_cond_grid = sct_gp.condition(logvar_y,X_grid)
        F_mean_grid  = sct_cond_grid.mean
        F_sigma_grid = jnp.sqrt(sct_cond_grid.variance)
        # print(np.shape(F_mean_grid));sys.exit()
        # f_grid, f_var_grid = transform(X_grid,np.full_like(X_grid,1.),
        #                                 F_mean_grid,F_sigma_grid,
        #                                 sct_gp,logvar_y)
        # logvar_grid_y, logvar_grid_err = aux.lin2log(f_grid, np.sqrt(f_var_grid))
        
        
        linvar_y, linvar_err = aux.log2lin(logvar_y, logvar_err)
        ax.errorbar(logvar_x,logvar_y,
                    logvar_err,ls='',capsize=2,marker='s',
                    label='binned')
        
        
        
        ax.plot(X_grid,F_mean_grid,'-C0',label=r'$g(\Delta x;\phi_g)$')
        ax.fill_between(X_grid,
                        F_mean_grid + F_sigma_grid, 
                        F_mean_grid - F_sigma_grid, 
                        color='C0',
                        alpha=0.3)
        # ax.scatter(X,(S/Y_err)**2.,c='r',s=2)
        ax.set_ylabel(r'$\log(\frac{S^2}{\sigma^2})$')
        # ax.set_yscale('log')
        ax.set_xlabel('Distance from centre (pix)')
        ax.set_ylim(-1.5, 3.5)
        ax.yaxis.tick_left()
        # ax.yaxis.set_ticks_position('left')
        axr = ax.secondary_yaxis('right', functions=(lambda x: np.exp(x), 
                                                      lambda x: np.log(x)))
        axr.yaxis.set_major_locator(ticker.FixedLocator([1,5,10,15,20]))
        axr.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axr.set_ylabel(r'$S^2 (\sigma^2)$',labelpad=-3)
        # axr.set_yticks([1, 5, 10,20])
        # axr.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.legend()
    return ax
    
    
    

def F(x,gp,logvar_y):
    '''
    For a scalar input x and a Gaussian Process GP(mu,sigma**2), 
    returns a scalar output GP(mu (x))
    
    Parameters
    ----------
        x : float32
    Output:
        value : float32
    '''
    
    value = gp.condition(logvar_y,jnp.atleast_1d(x))[1].mean
    return value[0]

def transform(x, sigma, GP_mean, GP_sigma, GP, logvar_y):
    '''
    Rescales the old error value at x-coordinate x using the GP mean 
    and sigma evaluated at x.
    
    F ~ GP(mean, sigma^2)
    F(x=x_i) = log( S_i^2 / sigma_i^2 )
    ==> S_i = sqrt( exp( F(x=x_i) ) ) * sigma_i
            = sqrt( exp( GP_mean) ) * sigma_i
        because of the property of logarithms:
            = exp (GP_mean/2.) * sigma_i
    
    
    Propagation of error gives:
    sigma(S_i) = | S_i / 2 * d(F)/dx|_{x_i} * GP_sigma |
    
    where
    GP_mean = F(x=x_i)
    GP_sigma = sigma(F(x=x_i)) 

    Parameters
    ----------
    x : float32, array_like
        x-coordinate.
    sigma : float32, array_like
        error on the y-coordinate value at x.
    GP_mean : float32, array_like
        mean of the GP evaluated at x.
    GP_sigma : float32, array_like
        sigma of the GP evaluated at x.

    Returns
    -------
    S : float32, array_like
        rescaled error on the y-coordinate at x.
    S_var : float32, array_like
        variance on the rescaled error due to uncertainty on the GP mean.

    '''
    deriv = jax.grad(functools.partial(F,gp=GP,logvar_y=logvar_y))
    dFdx  = jax.vmap(deriv)(x)
    S = sigma * jnp.exp(GP_mean/2.)
    S_var = jnp.power(S / 2. * dFdx * GP_sigma,2.)
    return S, S_var
def gaussian_mean_function(theta, X):
    '''
    Returns the Gaussian profile with parameters encapsulated in dictionary
    theta, evaluated a points in X

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    mean  = theta["mf_loc"]
    sigma = jnp.exp(theta["mf_log_sig"])
    gauss = jnp.exp(-0.5 * jnp.square((X - mean)/sigma)) \
            / jnp.sqrt(2*jnp.pi) / sigma
    beta = jnp.array([gauss,1])
    
    return jnp.array([theta['mf_amp'],theta['mf_const']]) @ beta

def build_scatter_GP(theta : dict,
                     X : jnp.ndarray,
                     Y_err : jnp.ndarray,
                     use_yerr : bool = False):
    '''
    Returns Gaussian Process for the intrinsic scatter of points (beyond noise)

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    sct_const  = jnp.exp(theta['sct_log_const'])
    sct_amp    = jnp.exp(theta['sct_log_amp'])
    sct_scale  = jnp.exp(theta['sct_log_scale'])
    pred = Y_err!=None
    
    def true_func():
        
        return tinygp.noise.Diagonal(jnp.power(Y_err,2.))
    def false_func():
        val = 1e-8
        # val = jnp.exp(theta['sct_log_epsilon0'])
        return tinygp.noise.Diagonal(jnp.full_like(X,val))
    noise1d = jax.lax.cond(pred,true_func,false_func)# + (1 - mask) * 1e18
    sct_kernel = sct_amp * tinygp.kernels.ExpSquared(sct_scale) #+ kernels.Constant(sct_const)
    # sct_kernel = sct_amp * kernels.Matern52(sct_scale) #+ kernels.Constant(sct_const)
    return tinygp.GaussianProcess(
        sct_kernel,
        X,
        noise= noise1d,
        mean = sct_const
    )

def build_LSF_GP(theta_lsf : dict,
                 X : jnp.ndarray,
                 # Y : jnp.ndarray = None,
                 Y_err : jnp.ndarray,
                 # use_yerr : bool = False,
                 use_scatter = False,
                 scatter : list = [], 
                 ) -> tinygp.GaussianProcess:
    '''
    Returns a Gaussian Process for the LSF. If scatter is not None, tries to 
    include a second GP for the intrinsic scatter of datapoints beyond the
    error on each individual point.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    Y_err : TYPE
        DESCRIPTION.
    scatter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    use_scatter = bool(use_scatter)
    gp_amp   = jnp.exp(theta_lsf['gp_log_amp'])
    gp_scale = jnp.exp(theta_lsf["gp_log_scale"])
    kernel = gp_amp * tinygp.kernels.ExpSquared(gp_scale) # LSF kernel
    # Various variances (obs=observed, add=constant random noise, tot=total)
    var_add = jnp.exp(theta_lsf['log_var_add']) 
    
    S, S_var = rescale_errors(use_scatter, scatter, X, Y_err)
    var_data  = jnp.square(S)
    var_tot = var_data + var_add
    # noise2d = jnp.diag(var_tot) #+ (1 - mask) * 1e18
    
    return tinygp.GaussianProcess(
        kernel,
        X,
        noise = jnp.diag(var_tot),
        mean=functools.partial(gaussian_mean_function, theta_lsf),
    )

def estimate_centre_numerically(X,Y,Y_err,LSF_solution,scatter=None,N=10):
    use_scatter = scatter is not None
    gp = build_LSF_GP(LSF_solution, X, Y_err,
                      use_scatter=use_scatter,
                      scatter=list(scatter) if use_scatter else [])
    rng_key = jax.random.PRNGKey(1234)
    X_grid  = jnp.linspace(-1,1,100)
    _, cond = gp.condition(Y,X_grid)
    samples = cond.sample(rng_key,shape=(N,))
    der=hf.derivative(samples,X_grid,order=1,accuracy=8)

def estimate_centre(X,Y,Y_err,LSF_solution,scatter=None,N=10):
    
    def value_(x):
        _, cond = gp.condition(Y,jnp.array([x]))
        sample = cond.sample(rng_key,shape=())
        return sample[0]
    # @partial(gp=cond,Y=Y)
    def derivative_(x):#,gp,Y,rng_key):
        # return jax.grad(partial(value_,gp=gp,Y=Y,rng_key=rng_key))(x)
        return jax.grad(value_)(x)
    # @jit
    def solve_(rng_key):
        bisect = jaxopt.Bisection(derivative_,-1.,1.)#,gp=gp,Y=Y,rng_key=rng_key)
        return bisect.run().params
    
    use_scatter = scatter is not None
    gp = build_LSF_GP(LSF_solution, X, Y_err,
                      use_scatter=use_scatter,
                      scatter=list(scatter) if use_scatter else [])
    X_grid  = jnp.linspace(-1,1,100)
    _, cond = gp.condition(Y,X_grid)
    
    
    centres = np.empty(N)
    for i in range(N):
        rng_key = jax.random.PRNGKey(i)
        
        centres[i] = solve_(rng_key)
    mean, sigma = hf.average(centres)
    del(centres); del(X_grid); del(cond)
    gc.collect()
    return -mean, sigma
def estimate_centre_anderson(X,Y,Y_err,LSF_solution):
    
    def value_(x):
        _, cond = gp.condition(Y,jnp.array([x]))
        return cond.mean[0]
    # @partial(gp=cond,Y=Y)
    def derivative_(x):#,gp,Y,rng_key):
        # return jax.grad(partial(value_,gp=gp,Y=Y,rng_key=rng_key))(x)
        return jax.grad(value_)(x)
    # @jit
    gp = build_LSF_GP(LSF_solution,X,Y_err, use_scatter=False,
                      scatter = [])
    
    vn = value_(-0.5)
    vp = value_(+0.5)
    dn = derivative_(-0.5)
    dp = derivative_(+0.5)
    shift_raw = (vp - vn)/(dp - dn)
    
    
    shift = jnp.where(jnp.isfinite(shift_raw), shift_raw, jnp.array(0.0))
    
    return shift, 0.

def estimate_centre_median(X,Y,Y_err,LSF_solution,scatter=None):
    from scipy.special import erfinv
    
    use_scatter = scatter is not None
    gp = build_LSF_GP(LSF_solution, X, Y_err, use_scatter=use_scatter,
                      scatter=list(scatter) if use_scatter else [])
    y = jnp.linspace(-1.0+1e-7, 1.0-1e-7, num=500)
    X_grid_ = jnp.sort(erfinv(y))
    sampled_Xrange = X_grid_.max() - X_grid_.min()
    
    Xmin = X.min()
    Xmax = X.max()
    desired_Xrange=X.max()-X.min()
    
    X_grid_2 = X_grid_/sampled_Xrange*desired_Xrange
    
    offset = X_grid_2.max() - X.max()
    
    
    X_grid   = X_grid_2 - offset
    _, cond  = gp.condition(Y,X_grid)
    # cumsum   = 
    mean_lsf = cond.mean
    cumsum   = np.cumsum(mean_lsf)/np.sum(mean_lsf)-0.5
    
    x1       = np.argmin(np.abs(cumsum))
    x2       = x1+1
    m = (cumsum[x2]-cumsum[x1])/(X_grid[x2]-X_grid[x1])
    b = cumsum[x2] - m*X_grid[x2]
    shift = b/m
    # plt.figure()
    # plt.plot(X_grid,cumsum)
    # plt.scatter([X_grid[x1],X_grid[x2],shift],
    #             [cumsum[x1],cumsum[x2],m*shift+b],
    #             c='r',s=10,marker='o')
    # plt.plot([X_grid[x1],X_grid[x2]],[cumsum[x1],cumsum[x2]])
    # print('linterp',shift)
    # splr  = interpolate.splrep(X_grid,cumsum)
    # shift = brentq(cumsum_at,-0.5,0.5)
    # print('spline',shift)
    return shift, 0.

def estimate_centre_centroid(X,Y,Y_err,LSF_solution,scatter=None):
    use_scatter = scatter is not None
    gp = build_LSF_GP(LSF_solution, X, Y_err, use_scatter=use_scatter,
                      scatter=list(scatter) if use_scatter else [])
    
    X_grid   = jnp.linspace(X.min(),X.max(),1000)
    _, cond  = gp.condition(Y,X_grid)
    mean_lsf = cond.mean
    shift    = -np.average(X_grid,weights=mean_lsf)
    return shift, 0.

def estimate_centre_mean(X,Y,Y_err,LSF_solution,scatter=None):
    use_scatter = scatter is not None
    gp = build_LSF_GP(LSF_solution, X, Y_err, use_scatter=use_scatter,
                      scatter=list(scatter) if use_scatter else [])
    X_grid   = jnp.linspace(X.min(),X.max(),1000)
    _, cond  = gp.condition(Y,X_grid)
    mean_lsf = cond.mean
    shift    = -np.average(X_grid,weights=mean_lsf)
    return shift, 0.