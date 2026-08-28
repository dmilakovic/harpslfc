"""
Width fit: refines the dense log(sigma) grid the same way dispersion.py
refines line position. Each line's pixel window gives a cheap local
correction (to log(sigma), via linearising the model's response to a
small multiplicative change in sigma), and poly_plus_gp_fit fits a smooth
Chebyshev-trend-plus-Matern32-residual curve through
(line_position, log_sigma + correction, uncertainty).
"""
from __future__ import annotations

import numpy as np

from .config import LSFConfig
from .forward_model import pixel_model_flux
from .gp import poly_plus_gp_fit
from .sigma_model import width
from .shape import evaluate_departure

__all__ = ["width", "fit_width"]


def fit_width(state, shape_coeffs, log_sigma_grid_init, line_position, v_pix):
    """ Mutates nothing; returns (log_sigma_grid, width_gp_fit, raw_target,
        raw_target_err). Caller (pipeline.py) assigns these into
        OrderState. """
    cfg: LSFConfig = state.cfg
    data = state.data
    log_sigma_grid = log_sigma_grid_init.copy()
    width_gp_fit = None
    target = target_err = None

    for _ in range(cfg.width_n_outer_steps):
        sigma_current = width(line_position, log_sigma_grid, data.pixel, cfg, state.v_per_pixel_typical)
        log_sigma_current = np.log(sigma_current)
        departure = evaluate_departure(state, sigma_current, shape_coeffs, line_position)

        delta = np.zeros(data.n_lines)
        delta_err = np.zeros(data.n_lines)
        for m in range(data.n_lines):
            idx = state.fit_window(line_position[m])
            model_value = pixel_model_flux(state.u, state.du, line_position[m], idx, v_pix[m],
                                            sigma_current[m], departure[m])

            sigma_perturbed = sigma_current[m] * np.exp(cfg.width_finite_difference_step)
            model_perturbed = pixel_model_flux(state.u, state.du, line_position[m], idx, v_pix[m],
                                                sigma_perturbed, departure[m])
            model_derivative = (model_perturbed - model_value) / cfg.width_finite_difference_step

            weight = state.inverse_variance[idx]
            denominator = np.sum(model_derivative ** 2 * weight)
            if denominator > 1e-12:
                delta[m] = np.sum(model_derivative * weight * (state.flux[idx] - model_value)) / denominator
                delta_err[m] = 1.0 / np.sqrt(denominator)
                # KNOWN, documented limitation (unaddressed, matches
                # fit_dispersion's identical caveat): the naive error above
                # does not account for width_coeffs/shape_coeffs already
                # having been fit to this same window elsewhere in the
                # outer loop, so it understates the true uncertainty. A
                # correct fix needs the joint covariance across all three
                # coupled fits; deliberately left as a documented gap.
            else:
                delta[m] = 0.0
                delta_err[m] = np.inf

        target = log_sigma_current + delta
        target_err = delta_err
        width_fit = poly_plus_gp_fit(
            line_position, target, delta_err, data.pixel.astype(float), n_restarts=3,
            poly_degree=cfg.width_poly_degree,
            length_scale_prior=cfg.width_length_scale_prior,
            kernel_type=cfg.width_kernel_type)
        width_gp_fit = {'z_mean': width_fit['z_mean'], 'z_std': width_fit['z_std'],
                         'length_scale': [width_fit['gp_length_scale']]}
        proposed_grid = width_gp_fit['z_mean']
        log_sigma_grid = log_sigma_grid + cfg.width_step_size * (proposed_grid - log_sigma_grid)

    return log_sigma_grid, width_gp_fit, target, target_err
