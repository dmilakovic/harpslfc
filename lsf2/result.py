"""
LSFOrderResult: everything needed to (a) reconstruct phi(u; x) at an
arbitrary pixel within the order, (b) reconstruct wavelength<->pixel at an
arbitrary point, and (c) judge the fit's quality -- packaged once the
joint outer loop in pipeline.py has converged (or exhausted its
iteration budget). This is what fits_io.py writes to disk and
reconstruct.py reads back.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LSFOrderResult:
    order: int
    n_pixels: int
    x_min: float
    x_max: float

    # --- width(x): sigma [km/s], as a dense log-sigma grid over every
    # pixel 0..n_pixels-1. Exact, not an approximation -- interpolation is
    # all the reader ever needs to do.
    width_log_sigma_grid: np.ndarray

    # --- 2D shape-departure GP: everything evaluate_departure needs
    u_inducing: np.ndarray
    x_inducing: np.ndarray
    shape_coeffs: np.ndarray            # (n_u_inducing, n_x_inducing)
    shape_x_length_scale: float
    shape_u_length_scale_factor: float
    shape_kappa_sigma0: float
    shape_kappa_sigmaf: float
    shape_kappa_width_factor: float
    shape_jitter: float
    u_grid: np.ndarray                  # the fine velocity grid shape_coeffs was fit on
    du: float
    sigma_ref: float                    # median line width at convergence (for provenance)

    # --- dispersion solution
    line_wavelength: np.ndarray         # nm, the M LFC line wavelengths
    line_position: np.ndarray           # final fitted pixel positions
    line_v_pix: np.ndarray              # local km/s/pixel at each line
    dispersion_lut_wavelength: np.ndarray   # dense lookup table
    dispersion_lut_pixel: np.ndarray
    dispersion_poly_coeffs: np.ndarray
    dispersion_poly_coeffs_cov: np.ndarray
    dispersion_poly_x_lo: float
    dispersion_poly_x_hi: float
    dispersion_poly_degree: int
    dispersion_gp_kernel_type: str
    dispersion_gp_length_scale: float
    dispersion_gp_signal_std: float
    dispersion_train_wavelength: np.ndarray   # residual-GP training points
    dispersion_train_position: np.ndarray
    dispersion_train_position_err: np.ndarray

    # --- convergence / quality diagnostics
    n_iterations_run: int
    converged: bool
    chi2_per_dof: float
    n_lines: int

    extra: dict = field(default_factory=dict)
