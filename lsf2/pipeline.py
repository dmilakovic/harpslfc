"""
The joint outer loop: alternates fitting l_x, the shape-departure grid,
the width grid, the dispersion solution, and (using the freshly-updated
line positions) re-fitting envelope/background -- repeating until all
three of {line_position, width_coeffs, shape_x_length_scale} stop moving
meaningfully step-to-step, or a fixed iteration budget is exhausted.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import LSFConfig
from .data import OrderData
from .dispersion import fit_dispersion, velocity_per_pixel_from_positions
from .forward_model import pixel_model_flux
from .gp import eval_poly_plus_gp, gp_predict_fixed_hyperparams
from .result import LSFOrderResult
from .shape import evaluate_departure, fit_shape_departure, fit_shape_x_length_scale
from .sigma_model import width
from .state import OrderState, initialise_state, refit_envelope_background
from .width import fit_width

log = logging.getLogger(__name__)


def run_order(data: OrderData, cfg: LSFConfig = None, verbose: bool = True,
              return_state: bool = False):
    """ return_state=True additionally returns the converged OrderState
        (flux/model/residuals, envelope/background grids, per-line raw
        GP-training targets, ...) -- not needed for the FITS output, but
        it's what diagnostics.py's plots are built from, since several of
        them (envelope/background, raw-flux residuals) use intermediate
        quantities that LSFOrderResult deliberately does not carry. """
    result, state = _run_order(data, cfg, verbose)
    if return_state:
        return result, state
    return result


def _run_order(data: OrderData, cfg: LSFConfig = None, verbose: bool = True):
    cfg = cfg or LSFConfig()
    state = initialise_state(data, cfg)

    prev_line_position = state.line_position.copy()
    prev_width_coeffs = state.width_coeffs.copy()
    prev_l_x = state.shape_x_length_scale
    converged = False
    n_run = 0

    for iteration in range(cfg.n_outer_iterations):
        n_run = iteration + 1
        v_pix = velocity_per_pixel_from_positions(data.wavelength, state.line_position)

        proposed_l_x = fit_shape_x_length_scale(state, state.line_position, state.width_coeffs, v_pix)
        state.shape_x_length_scale = state.shape_x_length_scale + cfg.shape_x_length_scale_step_size * (
            proposed_l_x - state.shape_x_length_scale)

        state.shape_coeffs = fit_shape_departure(state, state.line_position, state.width_coeffs, v_pix)
        (state.width_coeffs, state.width_gp_fit,
         state.width_raw_target, state.width_raw_target_err) = fit_width(
            state, state.shape_coeffs, state.width_coeffs, state.line_position, v_pix)
        (state.line_position, state.dispersion_gp_fit,
         state.dispersion_raw_target, state.dispersion_raw_target_err) = fit_dispersion(
            state, state.width_coeffs, state.shape_coeffs, state.line_position)

        # Envelope/background refit at the updated positions -- genuinely
        # coupled to the rest of the joint fit (a converged line_position
        # is a better peak-flux read than the catalogue position), not
        # looped for its own sake.
        refit_envelope_background(state)

        line_width = width(state.line_position, state.width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
        position_step = np.sqrt(np.mean((state.line_position - prev_line_position) ** 2))
        width_step = np.sqrt(np.mean((state.width_coeffs - prev_width_coeffs) ** 2))
        l_x_step = abs(state.shape_x_length_scale - prev_l_x)
        step_converged = (position_step < cfg.convergence_tol_position
                           and width_step < cfg.convergence_tol_width
                           and l_x_step < cfg.convergence_tol_lx)

        if verbose:
            log.info(
                "order %d, iteration %d: sigma(v) in [%.4f, %.4f] km/s, "
                "l_x=%.0f px, step: pos=%.2e width=%.2e l_x=%.1f%s",
                data.order, iteration, line_width.min(), line_width.max(),
                state.shape_x_length_scale, position_step, width_step, l_x_step,
                "  [converged]" if step_converged else "",
            )

        prev_line_position = state.line_position.copy()
        prev_width_coeffs = state.width_coeffs.copy()
        prev_l_x = state.shape_x_length_scale

        if step_converged and iteration >= cfg.convergence_min_iterations:
            converged = True
            break

    if not converged and verbose:
        log.warning(
            "order %d: outer loop did NOT meet the convergence tolerance within %d "
            "iterations -- reporting the final state as a possibly-unconverged snapshot.",
            data.order, cfg.n_outer_iterations,
        )

    return _package_result(state, converged, n_run), state


def _package_result(state: OrderState, converged: bool, n_run: int) -> LSFOrderResult:
    data, cfg = state.data, state.cfg
    v_pix = velocity_per_pixel_from_positions(data.wavelength, state.line_position)
    line_width = width(state.line_position, state.width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
    departure = evaluate_departure(state, line_width, state.shape_coeffs, state.line_position)

    # chi2/dof over every fitted pixel window, using the fully-converged model.
    model = np.zeros(data.n_pixels)
    fitted_mask = np.zeros(data.n_pixels, dtype=bool)
    for m in range(data.n_lines):
        idx = state.fit_window(state.line_position[m])
        model[idx] += pixel_model_flux(state.u, state.du, state.line_position[m], idx,
                                        v_pix[m], line_width[m], departure[m])
        fitted_mask[idx] = True
    residual = (state.flux - model) / state.flux_err
    chi2_per_dof = float(np.sum(residual[fitted_mask] ** 2) / fitted_mask.sum())

    # Dense wavelength<->pixel lookup table. Built EXACTLY, not
    # approximately: the polynomial term is a closed-form re-evaluation of
    # the already-fitted Chebyshev coefficients (eval_poly_plus_gp), and
    # the GP residual term is re-evaluated at the already-converged
    # hyperparameters via gp_predict_fixed_hyperparams (no re-
    # optimisation, so this is the identical posterior the last
    # fit_dispersion iteration produced, just evaluated at more points).
    # This is what lets the reconstruction reader use only this dense
    # table and never import jax/tinygp.
    dfit = state.dispersion_gp_fit  # a poly_plus_gp_fit() return dict
    lam_lo, lam_hi = data.wavelength.min(), data.wavelength.max()
    lut_wavelength = np.linspace(lam_lo, lam_hi, cfg.dispersion_lut_n_points)

    poly_term_grid = eval_poly_plus_gp(dfit, lut_wavelength)
    poly_term_train = eval_poly_plus_gp(dfit, data.wavelength)
    residual_train = state.dispersion_raw_target - poly_term_train  # what the residual GP was trained on
    residual_gp_grid = gp_predict_fixed_hyperparams(
        data.wavelength, residual_train, state.dispersion_raw_target_err, lut_wavelength,
        kernel_type=cfg.dispersion_kernel_type,
        length_scale=dfit['gp_length_scale'], signal_std=dfit['gp_signal_std'],
    )
    lut_pixel = poly_term_grid + residual_gp_grid['z_mean']

    return LSFOrderResult(
        order=data.order, n_pixels=data.n_pixels, x_min=data.x_min, x_max=data.x_max,
        width_log_sigma_grid=state.width_coeffs,
        u_inducing=state.u_inducing, x_inducing=state.x_inducing,
        shape_coeffs=state.shape_coeffs,
        shape_x_length_scale=state.shape_x_length_scale,
        shape_u_length_scale_factor=cfg.shape_u_length_scale_factor,
        shape_kappa_sigma0=cfg.shape_kappa_sigma0,
        shape_kappa_sigmaf=cfg.shape_kappa_sigmaf,
        shape_kappa_width_factor=cfg.shape_kappa_width_factor,
        shape_jitter=cfg.shape_jitter,
        u_grid=state.u, du=state.du,
        sigma_ref=float(np.median(line_width)),
        line_wavelength=data.wavelength,
        line_position=state.line_position,
        line_v_pix=v_pix,
        dispersion_lut_wavelength=lut_wavelength,
        dispersion_lut_pixel=lut_pixel,
        dispersion_poly_coeffs=state.dispersion_gp_fit['poly_coeffs'],
        dispersion_poly_coeffs_cov=state.dispersion_gp_fit['poly_coeffs_cov'],
        dispersion_poly_x_lo=state.dispersion_gp_fit['poly_x_lo'],
        dispersion_poly_x_hi=state.dispersion_gp_fit['poly_x_hi'],
        dispersion_poly_degree=state.dispersion_gp_fit['poly_degree'],
        dispersion_gp_kernel_type=cfg.dispersion_kernel_type,
        dispersion_gp_length_scale=state.dispersion_gp_fit['gp_length_scale'],
        dispersion_gp_signal_std=state.dispersion_gp_fit['gp_signal_std'],
        dispersion_train_wavelength=data.wavelength,
        dispersion_train_position=state.dispersion_raw_target,
        dispersion_train_position_err=state.dispersion_raw_target_err,
        n_iterations_run=n_run, converged=converged, chi2_per_dof=chi2_per_dof,
        n_lines=data.n_lines,
    )
