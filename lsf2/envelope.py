"""
Envelope E(x) and background B(x): estimated directly from the raw pixel
data, before any LSF or wavelength-solution model is assumed. Entirely in
pixel/flux units -- no velocity dependence here.

fit_envelope_background is called once before the joint outer loop (with
the catalogue line positions) and again after every outer iteration (with
the converged line_position from that iteration), so an improved position
estimate can feed back into a better peak-flux reading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LSFConfig
from .data import OrderData
from .gp import map_gp_fit, poly_plus_gp_fit


def local_extremum(data: OrderData, centres, kind, half_width=2):
    values = np.empty(len(centres))
    for i, c in enumerate(centres):
        lo = max(int(round(c)) - half_width, 0)
        hi = min(int(round(c)) + half_width + 1, data.n_pixels)
        window = data.flux_raw[lo:hi]
        values[i] = window.max() if kind == 'max' else window.min()
    return values


def local_peak_subpixel(data: OrderData, centres, half_width=2):
    """ Sub-pixel corrected (x, y) peak via 3-point parabolic interpolation
        in LOG space around the discrete maximum near each centre: ln(y)
        is exactly quadratic in position for a Gaussian peak, y itself is
        not, so a raw-y parabola systematically under-corrects. dx is
        clipped to +/-0.5 pixels, since the correction is only meaningful
        as a refinement within the sampling interval that produced the
        discrete argmax in the first place. """
    n_pixels = data.n_pixels
    flux_raw = data.flux_raw
    x_peak = np.empty(len(centres))
    y_peak = np.empty(len(centres))
    for i, c in enumerate(centres):
        lo = max(int(round(c)) - half_width, 0)
        hi = min(int(round(c)) + half_width + 1, n_pixels)
        window = flux_raw[lo:hi]
        i_max = np.argmax(window)
        abs_pixel = lo + i_max
        if i_max == 0 or i_max == len(window) - 1:
            x_peak[i] = abs_pixel
            y_peak[i] = window[i_max]
            continue
        y0, y1, y2 = window[i_max - 1], window[i_max], window[i_max + 1]
        if y0 <= 0 or y1 <= 0 or y2 <= 0:
            x_peak[i] = abs_pixel
            y_peak[i] = y1
            continue
        ln_y0, ln_y1, ln_y2 = np.log(y0), np.log(y1), np.log(y2)
        denominator = ln_y2 - 2 * ln_y1 + ln_y0
        if denominator >= 0:
            x_peak[i] = abs_pixel
            y_peak[i] = y1
        else:
            dx = np.clip(0.5 * (ln_y0 - ln_y2) / denominator, -0.5, 0.5)
            x_peak[i] = abs_pixel + dx
            y_peak[i] = np.exp(ln_y1 - (ln_y2 - ln_y0) ** 2 / (8 * denominator))
    return x_peak, y_peak


def background_design_row(data: OrderData, x, e, cfg: LSFConfig):
    """ [1, x~*e, x~^2*e, ..., x~^P*e], with rescaled position x~. """
    x_tilde = data.rescaled_position(x)
    return np.column_stack(
        [np.ones_like(x_tilde)]
        + [x_tilde ** p * e for p in range(0, cfg.background_poly_order)]
    )


@dataclass
class BoundaryPoints:
    pixel: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray


def prepare_boundary(data: OrderData) -> BoundaryPoints:
    boundary_pixel = np.unique(np.concatenate([data.left_edge, data.right_edge]))
    boundary_flux = local_extremum(data, boundary_pixel, 'min')
    boundary_flux_err = data.err_raw[np.round(boundary_pixel).astype(int)]
    return BoundaryPoints(boundary_pixel, boundary_flux, boundary_flux_err)


@dataclass
class EnvelopeBackgroundFit:
    envelope_grid_full: np.ndarray
    background_grid_full: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    env_peak_pixel: np.ndarray
    env_peak_flux: np.ndarray
    background_coeffs: np.ndarray


def fit_envelope_background(data: OrderData, boundary: BoundaryPoints,
                             centre_positions, cfg: LSFConfig) -> EnvelopeBackgroundFit:
    n_pixels = data.n_pixels
    pixel_f = data.pixel.astype(float)

    env_peak_pixel, env_peak_flux = local_peak_subpixel(data, centre_positions)
    env_peak_flux_err = data.err_raw[np.clip(np.round(env_peak_pixel).astype(int), 0, n_pixels - 1)]

    if cfg.envelope_kernel_type == 'locallyperiodic':
        length_scale_prior = [
            (cfg.envelope_periodic_decay_length_frac * n_pixels, cfg.envelope_periodic_decay_length_log_std),
            (cfg.envelope_period_frac * n_pixels, cfg.envelope_period_log_std),
            cfg.envelope_gamma_prior,
        ]
    else:
        length_scale_prior = cfg.envelope_residual_length_scale_prior

    gp_fit = poly_plus_gp_fit(
        env_peak_pixel, env_peak_flux, env_peak_flux_err, pixel_f, n_restarts=3,
        poly_degree=cfg.envelope_poly_degree,
        length_scale_prior=length_scale_prior,
        kernel_type=cfg.envelope_kernel_type,
    )
    envelope_grid = gp_fit['z_mean']
    envelope_std_grid = gp_fit['z_std']

    def envelope(x, _grid=envelope_grid):
        return np.interp(x, pixel_f, _grid)

    def envelope_std(x, _grid=envelope_std_grid):
        return np.interp(x, pixel_f, _grid)

    envelope_at_boundary = envelope(boundary.pixel)
    design = background_design_row(data, boundary.pixel, envelope_at_boundary, cfg)
    xtw = design.T * (1.0 / boundary.flux_err ** 2)
    normal_matrix = xtw @ design
    background_coeffs = np.linalg.solve(normal_matrix, xtw @ boundary.flux)
    background_coeffs_covariance = np.linalg.inv(normal_matrix)

    coupled_prediction_at_boundary = background_design_row(
        data, boundary.pixel, envelope(boundary.pixel), cfg) @ background_coeffs
    boundary_residual = boundary.flux - coupled_prediction_at_boundary

    residual_gp_fit = map_gp_fit(
        boundary.pixel, boundary_residual, boundary.flux_err, pixel_f,
        n_restarts=3, length_scale_priors=[cfg.background_residual_length_scale_prior])
    background_residual_grid = residual_gp_fit['z_mean']
    background_residual_std_grid = residual_gp_fit['z_std']

    def background_residual(x):
        return np.interp(x, pixel_f, background_residual_grid)

    def background_residual_std(x):
        return np.interp(x, pixel_f, background_residual_std_grid)

    def background(x):
        e = envelope(x)
        return background_design_row(data, x, e, cfg) @ background_coeffs + background_residual(x)

    envelope_grid_full = envelope(pixel_f)
    background_grid_full = background(pixel_f)

    # Propagate the envelope's and background's own estimation uncertainty
    # into flux_err (not just err_raw): flux = (F-B)/(E-B), F=flux_raw
    # (known variance err_raw^2), E and B uncertain and correlated (B is
    # built directly from E). Standard multivariate delta method / ratio-
    # of-correlated-variables formula.
    poly_gain = sum(
        background_coeffs[p] * data.rescaled_position(pixel_f) ** p
        for p in range(1, cfg.background_poly_order + 1)
    )
    sigma_E = envelope_std(pixel_f)
    design_row_full = background_design_row(data, pixel_f, envelope_grid_full, cfg)
    coeff_variance_term = np.einsum('ij,jk,ik->i', design_row_full,
                                     background_coeffs_covariance, design_row_full)
    residual_variance_term = background_residual_std(pixel_f) ** 2

    var_D = (1 - poly_gain) ** 2 * sigma_E ** 2 + coeff_variance_term + residual_variance_term
    var_N = data.err_raw ** 2 + poly_gain ** 2 * sigma_E ** 2 + coeff_variance_term + residual_variance_term
    cov_ND = poly_gain * (poly_gain - 1) * sigma_E ** 2 + coeff_variance_term + residual_variance_term

    N_full = data.flux_raw - background_grid_full
    D_full = envelope_grid_full - background_grid_full
    flux_new = N_full / D_full
    flux_err_new = np.abs(flux_new) * np.sqrt(np.maximum(
        var_N / N_full ** 2 + var_D / D_full ** 2 - 2 * cov_ND / (N_full * D_full), 0.0))

    return EnvelopeBackgroundFit(
        envelope_grid_full=envelope_grid_full,
        background_grid_full=background_grid_full,
        flux=flux_new,
        flux_err=flux_err_new,
        env_peak_pixel=env_peak_pixel,
        env_peak_flux=env_peak_flux,
        background_coeffs=background_coeffs,
    )
