"""
OrderState: the mutable state the joint outer loop reads and updates each
iteration (line positions, width grid, shape-departure grid, current
flux/flux_err after the latest envelope/background refit, ...). The
original script kept all of this as module-level globals mutated in
place; here it is one explicit object threaded through width.py,
dispersion.py, shape.py and envelope.py instead, so multiple orders can be
fit in the same process without cross-talk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import LSFConfig, C_LIGHT_KMS
from .data import OrderData
from .envelope import BoundaryPoints, EnvelopeBackgroundFit, prepare_boundary, fit_envelope_background
from .forward_model import get_half_window, make_fit_window


@dataclass
class OrderState:
    data: OrderData
    cfg: LSFConfig

    # velocity grid (fixed for the whole fit)
    u: np.ndarray = None
    du: float = None
    n_grid: int = None
    half_window: int = None
    fit_window: callable = None
    v_per_pixel_typical: float = None

    # boundary points for background fitting (fixed)
    boundary: BoundaryPoints = None

    # updated each outer iteration by the envelope/background refit
    flux: np.ndarray = None
    flux_err: np.ndarray = None
    inverse_variance: np.ndarray = None
    envelope_grid_full: np.ndarray = None
    background_grid_full: np.ndarray = None
    background_coeffs: np.ndarray = None

    # the three jointly-fit quantities
    line_position: np.ndarray = None
    width_coeffs: np.ndarray = None          # log(sigma) dense grid over data.pixel
    shape_coeffs: np.ndarray = None          # (N_U_INDUCING, N_X_INDUCING)
    shape_x_length_scale: float = None       # mutable, refit every outer iteration

    # inducing grids (fixed once set up)
    u_inducing: np.ndarray = None
    x_inducing: np.ndarray = None

    # last GP fit dicts, kept for uncertainty reporting / reconstruction
    dispersion_gp_fit: dict = None
    dispersion_raw_target: np.ndarray = None       # last iteration's GP training target (pixel)
    dispersion_raw_target_err: np.ndarray = None    # last iteration's GP training uncertainty
    width_gp_fit: dict = None
    width_raw_target: np.ndarray = None
    width_raw_target_err: np.ndarray = None

    edge_inflation_factor: np.ndarray = None

    diagnostics: dict = field(default_factory=dict)


def initialise_state(data: OrderData, cfg: LSFConfig) -> OrderState:
    state = OrderState(data=data, cfg=cfg)

    lambda_min, lambda_max = data.wavelength.min(), data.wavelength.max()
    v_per_pixel_typical = (
        C_LIGHT_KMS * (lambda_max - lambda_min)
        / (data.x_max - data.x_min) / np.mean(data.wavelength)
    )
    state.v_per_pixel_typical = v_per_pixel_typical

    state.boundary = prepare_boundary(data)

    # Bootstrap envelope/background from the catalogue positions, purely
    # to get a flux/flux_err good enough to size the velocity grid and
    # measure the LFC's own line spacing.
    eb_fit = fit_envelope_background(data, state.boundary, data.peak_pixel, cfg)
    state.envelope_grid_full = eb_fit.envelope_grid_full
    state.background_grid_full = eb_fit.background_grid_full
    state.flux = eb_fit.flux
    state.flux_err = eb_fit.flux_err
    state.inverse_variance = 1.0 / eb_fit.flux_err ** 2
    state.background_coeffs = eb_fit.background_coeffs

    half_window = get_half_window(state.flux, minimum=cfg.grid_min_half_window)
    state.half_window = int(half_window)
    state.fit_window = make_fit_window(state.half_window, data.n_pixels)

    velocity_half_range = state.half_window * v_per_pixel_typical
    n_velocity_grid = state.half_window * cfg.pixel_subsample * 2 + 1
    state.u = np.linspace(-velocity_half_range, velocity_half_range, n_velocity_grid)
    state.n_grid = len(state.u)
    state.du = state.u[1] - state.u[0]

    initial_width_kms = 1.3 * v_per_pixel_typical
    state.width_coeffs = np.full(data.n_pixels, np.log(initial_width_kms))

    state.u_inducing = np.linspace(state.u.min(), state.u.max(), cfg.n_u_inducing)
    state.x_inducing = np.linspace(data.x_min, data.x_max, cfg.n_x_inducing)
    state.shape_coeffs = np.zeros((cfg.n_u_inducing, cfg.n_x_inducing))
    state.shape_x_length_scale = cfg.shape_x_length_scale_init

    state.line_position = data.peak_pixel.copy()

    return state


def refit_envelope_background(state: OrderState) -> None:
    """ Re-run the envelope/background fit at the current line_position,
        and update state.flux/flux_err/inverse_variance in place. """
    eb_fit = fit_envelope_background(state.data, state.boundary, state.line_position, state.cfg)
    state.envelope_grid_full = eb_fit.envelope_grid_full
    state.background_grid_full = eb_fit.background_grid_full
    state.flux = eb_fit.flux
    state.flux_err = eb_fit.flux_err
    state.inverse_variance = 1.0 / eb_fit.flux_err ** 2
    state.background_coeffs = eb_fit.background_coeffs
