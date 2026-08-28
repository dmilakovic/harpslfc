"""
All tunable constants for the lsf2 reconstruction pipeline, collected into
one dataclass instead of module-level globals (unlike the original
reconstruct_lsf_master.py script this package replaces). Values are the
same defaults that script used; see its inline comments (preserved in the
docstrings of the functions that consume each parameter) for the reasoning
and empirical checks behind each choice -- that reasoning is not repeated
here, only the numbers.

Passing a `LSFConfig` explicitly through the pipeline (rather than reading
module globals) is what makes it safe to reconstruct several orders in one
process, which the original script never needed to do.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SPEED_OF_LIGHT = 2.99792458e8        # m / s
C_LIGHT_KMS = SPEED_OF_LIGHT / 1e3   # km / s


@dataclass
class LSFConfig:
    # --- background / envelope -------------------------------------------------
    background_poly_order: int = 4
    envelope_poly_degree: int = 5
    envelope_kernel_type: str = 'locallyperiodic'
    envelope_residual_length_scale_prior: tuple = (100, 1.0)      # matern32 fallback
    envelope_periodic_decay_length_frac: float = 3 / 8             # * n_pixels
    envelope_periodic_decay_length_log_std: float = 0.6
    envelope_period_frac: float = 1 / 8                             # * n_pixels
    envelope_period_log_std: float = 0.3
    envelope_gamma_prior: tuple = (2.0, 1.0)
    background_residual_length_scale_prior: tuple = (30, 0.6)

    # --- velocity / pixel grid --------------------------------------------------
    pixel_subsample: int = 31
    grid_min_half_window: int = 8
    grid_safety_margin: float = 1.8   # currently unused directly but kept for parity

    # --- width sigma(x) ----------------------------------------------------------
    width_length_scale_prior: tuple = (20, 2)
    width_poly_degree: int = 5
    width_kernel_type: str = 'matern32'
    width_sigma_floor_frac: float = 0.05   # floor = this * typical v/pixel
    width_n_outer_steps: int = 6
    width_step_size: float = 0.5
    width_finite_difference_step: float = 1e-3

    # --- dispersion / position ----------------------------------------------------
    max_position_drift: float = 1.0           # pixels, cumulative cap
    dispersion_length_scale_prior: tuple = (3, 0.5)   # nm
    dispersion_kernel_type: str = 'expsquared'
    dispersion_poly_degree: int = 9
    dispersion_n_outer_steps: int = 4
    dispersion_step_size: float = 0.5
    dispersion_finite_difference_step: float = 1e-3
    edge_inflation_range: float = 3.0            # line spacings
    edge_inflation_max_factor: float = 5.0

    # --- 2D shape-departure GP -----------------------------------------------------
    n_u_inducing: int = 31
    n_x_inducing: int = 32
    shape_u_length_scale_factor: float = 0.8
    shape_x_length_scale_init: float = 3000.0     # pixels; also the prior mean
    shape_x_length_scale_prior_log_std: float = 1.0
    shape_x_length_scale_bounds: tuple = (200, 10000)
    shape_x_length_scale_step_size: float = 0.5
    shape_kappa_sigma0: float = 0.002
    shape_kappa_sigmaf: float = 0.05
    shape_kappa_width_factor: float = 1.5
    shape_identifiability_weight_shift: float = 1e5
    shape_identifiability_weight_width: float = 1e8
    shape_jitter: float = 1e-3

    # --- joint outer loop -----------------------------------------------------------
    n_outer_iterations: int = 10
    convergence_tol_position: float = 1e-4
    convergence_tol_width: float = 1e-4
    convergence_tol_lx: float = 1.0
    convergence_min_iterations: int = 3

    # --- reconstruction dense grids (written to the FITS file) ---------------------
    dispersion_lut_n_points: int = 2000     # dense wavelength<->pixel lookup table
    dense_width_grid: bool = True            # store the full per-pixel width grid

    extra: dict = field(default_factory=dict)   # escape hatch for anything ad hoc
