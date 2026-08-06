#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harps/lsf/fit_jax.py

JAX-native, batched replacement for the per-line scipy fit in harps.lsf.fit.line()
(method='scipy', bounded=True, npars=3, weight=True — the only combination
solve_line() actually calls).

Design, informed by harps.lsf.container.interpolate_local():
- Every segment's numerical LSF model in a given order is evaluated on the SAME
  shared, uniform x-grid (see construct.numerical_models: np.linspace(x_min,x_max,npts)).
- interpolate_local() is therefore just an inverse-distance-weighted average of a
  small number of segments' y-curves on that shared grid — a (n_lines, n_segments)
  weight matrix times a (n_segments, n_grid) matrix of curves. No per-line spline
  construction is needed for this step.
- What DOES vary per line is the spline *evaluation* (each line's own x1l - cen,
  scaled by its own fitted width, are different query points). That's implemented
  as a natural cubic spline on the shared grid (see natural_cubic_spline_M /
  eval_spline_batch below), vmapped across lines.
- The bounded fit itself uses jaxopt.LBFGSB minimizing a scalar loss built from
  jax-native residuals, so the gradient comes from autodiff — this also fixes a
  latent bug in the existing scipy path, where lsf_model() divides x by wid but
  the hand-written jacobian_analytical() differentiates as if x were multiplied
  by wid. Autodiff can't have that class of bug: it differentiates whatever the
  forward model actually computes.

This module intentionally supports ONLY npars=3 (amp, cen, wid), weight=True,
bounded=True — the combination solve_line() uses. fit.py's line() is left
untouched as the general-purpose / diagnostic implementation.
"""
import functools
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

jax.config.update("jax_enable_x64", True)

C_KMS = 299792.458  # speed of light, km/s — matches fit.py's constant


# ─────────────────────────────────────────────────────────────────────────────
# Natural cubic spline on a shared, uniform grid (validated against scipy in
# test_spline.py to ~1e-14 for a representative smooth peaked function).
# ─────────────────────────────────────────────────────────────────────────────

def natural_cubic_spline_M(x_grid: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """
    Y: (..., n) batch of curves on the SAME uniform grid x_grid (n,).
    Returns M: (..., n) second derivatives at knots (natural BC: M[0]=M[-1]=0).
    One linear solve for the whole batch (shared tridiagonal system, many RHS
    columns) rather than one solve per curve.
    """
    n = x_grid.shape[0]
    h = x_grid[1] - x_grid[0]

    A = jnp.zeros((n, n))
    idx = jnp.arange(1, n - 1)
    A = A.at[idx, idx].set(4 * h)
    A = A.at[idx, idx - 1].set(h)
    A = A.at[idx, idx + 1].set(h)
    A = A.at[0, 0].set(1.0)
    A = A.at[n - 1, n - 1].set(1.0)

    orig_shape = Y.shape
    Yb = Y.reshape(-1, n)
    rhs_interior = 6.0 / h * (Yb[:, 2:] - 2 * Yb[:, 1:-1] + Yb[:, :-2])
    D = jnp.zeros((Yb.shape[0], n))
    D = D.at[:, 1:-1].set(rhs_interior)
    M = jnp.linalg.solve(A, D.T).T
    return M.reshape(orig_shape)


def _eval_spline_1d(x_grid, Y, M, xq):
    """
    Single spline (1D Y, M), evaluated at query points xq. Returns (value, deriv).

    xq is clamped to the template's own domain [x_grid[0], x_grid[-1]]
    before evaluation. Without this, only the SEGMENT INDEX `i` was being
    clamped (via jnp.clip on the searchsorted result) — the query value xq
    itself was not, so t=(xq-x_i)/h grew without bound for any point
    outside the domain, and the cubic terms (A_**3-A_) etc. blow up
    polynomially in t. Real line windows (14-19 px wide) routinely place
    points 7-10 px from the line centre, outside the template's own +/-6
    domain, so this fired for the majority of lines, not just rare edge
    cases — producing the large spurious negative/positive spikes at
    window edges reported when comparing reconstructed models against the
    data. Clamping xq means points beyond the domain hold at the
    template's boundary value (already small — the LSF is essentially
    flat and near-zero out at +/-6) instead of extrapolating; this is a
    strictly bounded, physically reasonable choice given the template
    defines nothing beyond its own domain, and it changes nothing for any
    query that was already inside [-6, 6].
    """
    n = x_grid.shape[0]
    h = x_grid[1] - x_grid[0]
    xq = jnp.clip(xq, x_grid[0], x_grid[-1])
    i = jnp.clip(jnp.searchsorted(x_grid, xq, side='right') - 1, 0, n - 2)
    x_i = x_grid[i]
    t = (xq - x_i) / h
    Y_i, Y_ip1 = Y[i], Y[i + 1]
    M_i, M_ip1 = M[i], M[i + 1]
    A_, B_ = 1 - t, t
    value = (A_ * Y_i + B_ * Y_ip1
            + (h**2 / 6.0) * ((A_**3 - A_) * M_i + (B_**3 - B_) * M_ip1))
    deriv = ((Y_ip1 - Y_i) / h
            - (h / 6.0) * (3 * A_**2 - 1) * M_i
            + (h / 6.0) * (3 * B_**2 - 1) * M_ip1)
    return value, deriv


# vmap over a batch of (Y, M, xq) triples sharing the same x_grid.
_eval_spline_batch = jax.vmap(_eval_spline_1d, in_axes=(None, 0, 0, 0))


# ─────────────────────────────────────────────────────────────────────────────
# LSF interpolation: (n_lines, n_segments) weights x (n_segments, n_grid) curves.
# Matches harps.lsf.container.get_segment_weights: inverse-distance, masked to
# segments within `segdist*(N-1)` of the line's centre.
# ─────────────────────────────────────────────────────────────────────────────

def segment_interp_weights(centers: jnp.ndarray, segment_centres: jnp.ndarray,
                           N: int = 2) -> jnp.ndarray:
    """
    centers: (n_lines,) each line's bary.
    segment_centres: (n_segments,) centres of every segment in this order,
        i.e. (ledge+redge)/2, matching container.get_segment_centres.
    Returns weights: (n_lines, n_segments), rows sum to 1 over the segments
    actually used (matches container.py's masking exactly, N=2 default).
    """
    segdist = jnp.diff(jnp.sort(segment_centres))[0]  # assumes equally spaced, as container.py does
    distances = jnp.abs(centers[:, None] - segment_centres[None, :])  # (n_lines, n_segments)
    threshold = segdist * (N - 1) if N > 1 else segdist / 2.0
    used = distances < threshold
    inv_dist = jnp.where(used, 1.0 / jnp.maximum(distances, 1e-12), 0.0)
    row_sum = jnp.sum(inv_dist, axis=1, keepdims=True)
    weights = jnp.where(row_sum > 0, inv_dist / row_sum, 0.0)
    return weights


def interpolate_lsf_batch(centers, segment_centres, segment_Y, N=2):
    """
    Returns (n_lines, n_grid) interpolated y-curves, one per line, on the
    shared x_grid (caller already has x_grid; it doesn't change here).
    segment_Y: (n_segments, n_grid).
    """
    W = segment_interp_weights(centers, segment_centres, N=N)  # (n_lines, n_segments)
    return W @ segment_Y  # (n_lines, n_grid)


# ─────────────────────────────────────────────────────────────────────────────
# Weighting / within-limits — ports of fit.py's assign_weights / within_limits,
# expressed as closed-form piecewise-linear functions instead of np.digitize
# (see derivation in conversation: matches the digitize-based version exactly
# for the pixel and velocity bin edges used in fit.py).
# ─────────────────────────────────────────────────────────────────────────────

def assign_weights_pix(dx):
    """dx = x - centre, in pixels. Bin edges [-5,-2.5,2.5,5], matches fit.py."""
    return jnp.clip(1.0 - (jnp.abs(dx) - 2.5) / 2.5, 0.0, 1.0)


def assign_weights_vel(dv):
    """dv = (x-centre)/centre*c, in km/s. Bin edges [-4,-2,2,4], matches fit.py."""
    return jnp.clip(1.0 - (jnp.abs(dv) - 2.0) / 2.0, 0.0, 1.0)


def within_limits_pix(dx):
    return jnp.abs(dx) <= 5.0


def within_limits_vel(dv):
    return jnp.abs(dv) <= 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Model + loss for one line, then vmapped + jaxopt.LBFGSB across a whole order.
# ─────────────────────────────────────────────────────────────────────────────

def _lsf_model_1d(x_grid, Y_line, M_line, x1l, amp, cen, wid, scale_is_pix):
    """
    One line's model, matching fit.py's lsf_model exactly:
        x_test = x1l - cen                          (pixel scale)
                 (x1l - cen) / cen * C_KMS            (velocity scale)
        model  = amp * spline(x_test / wid)
    """
    x_test = jnp.where(scale_is_pix, x1l - cen, (x1l - cen) / cen * C_KMS)
    val, _ = _eval_spline_1d(x_grid, Y_line, M_line, x_test / wid)
    return amp * val, x_test


def _loss_one_line(params, x_grid, Y_line, M_line, x1l, flx1l, err1l, mask, scale_is_pix):
    amp, cen, wid = params['amp'], params['cen'], params['wid']
    wid_pos = jnp.abs(wid)
    model, x_test = _lsf_model_1d(x_grid, Y_line, M_line, x1l, amp, cen, wid_pos, scale_is_pix)
    w = jnp.where(scale_is_pix, assign_weights_pix(x_test), assign_weights_vel(x_test))
    resid = (flx1l - model) * w / err1l
    resid = jnp.where(mask, resid, 0.0)
    return 0.5 * jnp.sum(resid**2)


def _fit_one_line(theta_start, bounds, x_grid, Y_line, M_line, x1l, flx1l, err1l, mask,
                  scale_is_pix, maxiter):
    solver = jaxopt.LBFGSB(
        fun=functools.partial(_loss_one_line, x_grid=x_grid, Y_line=Y_line, M_line=M_line,
                              x1l=x1l, flx1l=flx1l, err1l=err1l, mask=mask,
                              scale_is_pix=scale_is_pix),
        maxiter=maxiter, tol=1e-8,
    )
    result = solver.run(theta_start, bounds=bounds)
    final_loss = _loss_one_line(result.params, x_grid, Y_line, M_line, x1l, flx1l, err1l,
                                mask, scale_is_pix)
    return result.params, final_loss


_fit_batch = jax.vmap(
    _fit_one_line,
    in_axes=(0, 0, None, 0, 0, 0, 0, 0, 0, None, None),
)


def fit_lines_batch(x1l_batch, flx1l_batch, err1l_batch, mask_batch, bary_batch,
                    x_grid, segment_centres, segment_Y, scale, N=2, maxiter=200,
                    amp_rel=0.5, cen_abs=1.0, wid_bounds=(0.1, 5.0)):
    """
    Fit every line in one order in a single vmapped, bound-constrained
    optimization. Padded inputs (x1l_batch etc, shape (n_lines, max_len))
    with mask_batch (same shape, True where real data) let lines of
    different width share one batched call.

    Returns
    -------
    pars   : dict with 'amp','cen','wid' arrays, shape (n_lines,)
    loss   : (n_lines,) final loss (0.5 * sum weighted squared residuals)
    Y_line : (n_lines, n_grid) the interpolated LSF curve actually used per line
             (handy for recomputing the full model/residual array afterwards).
    """
    scale_is_pix = scale[:3] == 'pix'
    n_lines, max_len = x1l_batch.shape

    Y_line = interpolate_lsf_batch(bary_batch, segment_centres, segment_Y, N=N)
    M_line = natural_cubic_spline_M(x_grid, Y_line)

    # Initial guesses, matching fit.py's _prepare_pars (amp ~ 1.1*max flux,
    # cen ~ flux-weighted mean of x, wid = 1.0), computed per line, masked.
    #
    # IMPORTANT: the production LSF templates (segment_Y, as loaded from the
    # pixel_model/velocity_model FITS extensions) are NOT peak-normalized to 1
    # — their integral is held ~constant across segments instead, so peak
    # height varies (shrinks) as the LSF broadens across an order. Since the
    # forward model is `model = amp * spline(...)`, amp's natural scale is
    # "data peak / template peak", not "data peak" alone. Omitting the
    # division here was the root cause of a systematic bad-local-minimum
    # failure (wid collapsing to ~0.55 for every line, chisq/dof in the
    # tens of millions) — see conversation history for the full diagnosis.
    # The jnp.maximum floor guards against a degenerate zero-weight line
    # (interpolate_lsf_batch's segment_interp_weights returning an all-zero
    # row, e.g. a bary_batch value outside every segment's threshold) making
    # template_peak==0 and amp0 blow up to inf.
    template_peak = jnp.maximum(jnp.max(Y_line, axis=1), 1e-12)
    flx_masked = jnp.where(mask_batch, flx1l_batch, 0.0)
    amp0 = 1.1 * jnp.max(flx_masked, axis=1) / template_peak
    x_masked = jnp.where(mask_batch, x1l_batch, 0.0)
    weight_sum = jnp.sum(flx_masked, axis=1)
    cen0 = jnp.where(weight_sum > 0,
                     jnp.sum(x_masked * flx_masked, axis=1) / jnp.maximum(weight_sum, 1e-12),
                     bary_batch)
    wid0 = jnp.full((n_lines,), 1.0)

    theta_start = dict(amp=amp0, cen=cen0, wid=wid0)

    lower = dict(
        amp=amp0 * (1 - amp_rel),
        cen=cen0 - cen_abs,
        wid=jnp.full((n_lines,), wid_bounds[0]),
    )
    upper = dict(
        amp=amp0 * (1 + amp_rel),
        cen=cen0 + cen_abs,
        wid=jnp.full((n_lines,), wid_bounds[1]),
    )
    bounds = (lower, upper)

    err_safe = jnp.where(mask_batch, err1l_batch, 1.0)  # avoid div-by-0 on padding

    pars, loss = _fit_batch(
        theta_start, bounds, x_grid, Y_line, M_line, x1l_batch, flx1l_batch, err_safe,
        mask_batch, scale_is_pix, maxiter,
    )
    return pars, loss, Y_line


def _loss_flat(theta_flat, x_grid, Y_line, M_line, x1l, flx1l, err1l, mask, scale_is_pix):
    """Same loss as _loss_one_line, but taking/returning a flat (3,) array
    instead of a dict — needed because jax.hessian wants an array-valued
    argument to differentiate against, not a dict of scalars."""
    params = dict(amp=theta_flat[0], cen=theta_flat[1], wid=theta_flat[2])
    return _loss_one_line(params, x_grid, Y_line, M_line, x1l, flx1l, err1l, mask, scale_is_pix)


def _param_errors_one_line(theta_flat, x_grid, Y_line, M_line, x1l, flx1l, err1l, mask,
                           scale_is_pix):
    """
    Laplace/Gauss-Newton approximation: the loss here is already
    0.5 * sum((weighted residuals)^2), i.e. half the chi-square, so the
    Hessian of the loss at the best-fit point plays the role of the
    inverse covariance matrix (Fisher information), same as for a proper
    negative-log-likelihood. Cov ≈ H^-1; per-parameter error = sqrt(diag(Cov)).

    Falls back to NaN (rather than a fabricated number) if the Hessian is
    singular or not positive-definite at this point — e.g. a parameter
    that turned out to be unconstrained by the data — since a very large
    or negative "variance" from a bad inverse is worse than admitting we
    don't have a usable error estimate for that line.
    """
    H = jax.hessian(_loss_flat)(theta_flat, x_grid, Y_line, M_line, x1l, flx1l, err1l,
                                mask, scale_is_pix)
    eigvals = jnp.linalg.eigvalsh(H)
    well_posed = jnp.all(eigvals > 1e-8)
    cov = jnp.linalg.pinv(H)
    variances = jnp.diag(cov)
    errors = jnp.where(
        well_posed & (variances > 0),
        jnp.sqrt(jnp.where(variances > 0, variances, jnp.nan)),
        jnp.nan,
    )
    return errors


_param_errors_batch = jax.vmap(
    _param_errors_one_line,
    in_axes=(0, None, 0, 0, 0, 0, 0, 0, None),
)


def estimate_param_errors(pars, x_grid, Y_line, M_line, x1l_batch, flx1l_batch,
                          err1l_batch, mask_batch, scale):
    """
    pars: dict with 'amp','cen','wid', each (n_lines,) — the best-fit
        parameters already found by fit_lines_batch.
    Returns errs: dict with the same keys, each (n_lines,) — per-line,
        per-parameter standard errors (NaN where the Hessian wasn't usable
        for that line, e.g. too few points or a genuinely unconstrained fit).
    """
    scale_is_pix = scale[:3] == 'pix'
    theta_flat = jnp.stack([pars['amp'], pars['cen'], pars['wid']], axis=1)  # (n_lines,3)
    errors = _param_errors_batch(theta_flat, x_grid, Y_line, M_line, x1l_batch,
                                 flx1l_batch, err1l_batch, mask_batch, scale_is_pix)
    return dict(amp=errors[:, 0], cen=errors[:, 1], wid=errors[:, 2])


def compute_model_chisq(pars, x_grid, Y_line, M_line, x1l_batch, flx1l_batch, err1l_batch,
                        mask_batch, scale):
    """
    Recompute the full model array and chi-square/dof per line for the fitted
    parameters — used when writing results back (models array, chisq column).
    """
    scale_is_pix = scale[:3] == 'pix'

    def _one(amp, cen, wid, Y_l, M_l, x1l, flx1l, err1l, mask):
        model, x_test = _lsf_model_1d(x_grid, Y_l, M_l, x1l, amp, cen, jnp.abs(wid), scale_is_pix)
        within = jnp.where(scale_is_pix, within_limits_pix(x_test), within_limits_vel(x_test))
        within = within & mask
        rsd = jnp.where(within, (flx1l - model) / err1l, 0.0)
        chisq = jnp.sum(rsd**2)
        dof = jnp.sum(within) - 3
        model_out = jnp.where(mask, model, 0.0)
        integral = jnp.sum(jnp.where(within, model, 0.0))
        return model_out, chisq, dof, integral

    model_batch, chisq, dof, integral = jax.vmap(_one)(
        pars['amp'], pars['cen'], pars['wid'], Y_line, M_line,
        x1l_batch, flx1l_batch, err1l_batch, mask_batch,
    )
    return model_batch, chisq, dof, integral