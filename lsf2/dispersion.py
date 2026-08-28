"""
Wavelength calibration model. x(lambda) is represented by its value at
the M known comb wavelengths (line_position), refined via the same
per-line-local-correction + smooth-GP pattern used for width.py.

velocity_per_pixel_from_positions gives the LOCAL km/s-per-pixel scale at
every comb line's own wavelength, via a discrete derivative across
neighbouring lines (sorted by wavelength) from the CURRENT position
estimate -- not fit, computed directly, since re-deriving a GP for every
finite-difference perturbation would be far too slow.
"""
from __future__ import annotations

import numpy as np

from .config import LSFConfig, C_LIGHT_KMS
from .forward_model import pixel_model_flux
from .gp import poly_plus_gp_fit
from .sigma_model import width
from .shape import evaluate_departure


def velocity_per_pixel_from_positions(wavelength: np.ndarray, position: np.ndarray) -> np.ndarray:
    order = np.argsort(wavelength)
    lam_sorted = wavelength[order]
    pos_sorted = position[order]
    dx_dlambda_sorted = np.gradient(pos_sorted, lam_sorted)
    v_pix_sorted = C_LIGHT_KMS / dx_dlambda_sorted / lam_sorted
    v_pix = np.empty_like(v_pix_sorted)
    v_pix[order] = v_pix_sorted
    return v_pix


def edge_uncertainty_inflation(wavelength: np.ndarray, cfg: LSFConfig) -> np.ndarray:
    """ Per-line multiplicative factor for delta_err, 1.0 in the interior
        and growing toward cfg.edge_inflation_max_factor for lines within
        cfg.edge_inflation_range line-spacings of either end of the
        wavelength range -- the outermost lines only ever get pulled from
        one side by the GP, so nothing pushes back if their own local
        correction is noisy; this was found to let the edge drift
        (see cfg.max_position_drift) grow unbounded without it. """
    sorted_idx = np.argsort(wavelength)
    sorted_wavelength = wavelength[sorted_idx]
    typical_spacing = np.median(np.diff(sorted_wavelength))
    distance_to_min = sorted_wavelength - sorted_wavelength.min()
    distance_to_max = sorted_wavelength.max() - sorted_wavelength
    distance_to_nearest_edge = np.minimum(distance_to_min, distance_to_max)
    threshold = cfg.edge_inflation_range * typical_spacing
    closeness = np.clip(1.0 - distance_to_nearest_edge / threshold, 0.0, 1.0)
    factor_sorted = 1.0 + (cfg.edge_inflation_max_factor - 1.0) * closeness
    factor = np.empty_like(factor_sorted)
    factor[sorted_idx] = factor_sorted
    return factor


def fit_dispersion(state, width_coeffs, shape_coeffs, line_position_init):
    cfg: LSFConfig = state.cfg
    data = state.data

    if state.edge_inflation_factor is None:
        state.edge_inflation_factor = edge_uncertainty_inflation(data.wavelength, cfg)

    line_position = line_position_init.copy()
    dispersion_gp_fit = None
    target = delta_err = None

    for _ in range(cfg.dispersion_n_outer_steps):
        v_pix = velocity_per_pixel_from_positions(data.wavelength, line_position)
        line_width = width(line_position, width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
        departure = evaluate_departure(state, line_width, shape_coeffs, line_position)

        delta = np.zeros(data.n_lines)
        delta_err = np.zeros(data.n_lines)
        for m in range(data.n_lines):
            idx = state.fit_window(line_position[m])
            model_value = pixel_model_flux(state.u, state.du, line_position[m], idx, v_pix[m],
                                            line_width[m], departure[m])
            model_shifted = pixel_model_flux(
                state.u, state.du, line_position[m] + cfg.dispersion_finite_difference_step,
                idx, v_pix[m], line_width[m], departure[m])
            model_derivative = (model_shifted - model_value) / cfg.dispersion_finite_difference_step

            weight = state.inverse_variance[idx]
            denominator = np.sum(model_derivative ** 2 * weight)
            if denominator > 1e-12:
                delta[m] = np.sum(model_derivative * weight * (state.flux[idx] - model_value)) / denominator
                delta_err[m] = 1.0 / np.sqrt(denominator)
                # Same documented, unaddressed dof-underestimate caveat as
                # width.fit_width -- see that module's note.
            else:
                delta[m] = 0.0
                delta_err[m] = np.inf

        delta_err = delta_err * state.edge_inflation_factor
        target = line_position + delta

        dispersion_fit = poly_plus_gp_fit(
            data.wavelength, target, delta_err, data.wavelength, n_restarts=5,
            poly_degree=cfg.dispersion_poly_degree,
            kernel_type=cfg.dispersion_kernel_type,
            length_scale_prior=cfg.dispersion_length_scale_prior)
        dispersion_gp_fit = dispersion_fit
        proposed = dispersion_fit['z_mean']
        line_position = line_position + cfg.dispersion_step_size * (proposed - line_position)

        # Cap CUMULATIVE drift from the catalogued position (not the
        # per-step delta -- confirmed directly that drift builds up
        # gradually across outer iterations rather than in one jump, so
        # only a cumulative cap actually contains it).
        line_position = data.peak_pixel + np.clip(
            line_position - data.peak_pixel, -cfg.max_position_drift, cfg.max_position_drift)

    return line_position, dispersion_gp_fit, target, delta_err
