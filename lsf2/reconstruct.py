"""
Reads a FITS file written by fits_io.save_lsf_fits and reconstructs the
LSF phi(u) at an arbitrary pixel or wavelength within an order, or a
weighted composite across every order that covers a given wavelength
(echelle orders overlap in wavelength near their edges).

Deliberately has no jax/tinygp import: evaluate_departure (shape.py) and
the width interpolation are both plain numpy, and the dispersion solution
is read from the dense lookup table saved at write time -- so this module
(and anything built on it) stays lightweight to install on a laptop that
never needs to run the fit itself.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .config import LSFConfig
from .fits_io import load_lsf_fits
from .forward_model import gaussian_mean
from .shape import evaluate_departure


class LSFLibrary:
    """ In-memory view of one lsf2 output FITS file. """

    def __init__(self, path: str):
        self.path = path
        data = load_lsf_fits(path)
        self.header = data['header']
        self._lsf_by_order = {int(row['ORDER']): row for row in data['lsf']}
        self._dispersion_by_order = {int(row['ORDER']): row for row in data['dispersion']}

    def orders(self) -> list[int]:
        return sorted(self._lsf_by_order)

    def pixel_range(self, order: int) -> tuple[float, float, int]:
        row = self._lsf_by_order[order]
        return float(row['X_MIN']), float(row['X_MAX']), int(row['N_PIXELS'])

    def shape_grid(self, order: int):
        """ Returns (u_inducing, x_inducing, shape_coeffs) -- the raw 2D
            departure inducing grid, e.g. for an imshow-style diagnostic. """
        row = self._lsf_by_order[order]
        n_u, n_x = int(row['N_U_INDUCING']), int(row['N_X_INDUCING'])
        shape_coeffs = np.asarray(row['SHAPE_COEFFS']).reshape(n_u, n_x)
        return np.asarray(row['U_INDUCING']), np.asarray(row['X_INDUCING']), shape_coeffs

    # --- dispersion -------------------------------------------------------

    def wavelength_range(self, order: int) -> tuple[float, float]:
        lut_wave = np.asarray(self._dispersion_by_order[order]['LUT_WAVELENGTH'])
        return float(lut_wave.min()), float(lut_wave.max())

    def wavelength_to_pixel(self, order: int, wavelength) -> np.ndarray:
        row = self._dispersion_by_order[order]
        lut_wave = np.asarray(row['LUT_WAVELENGTH'])
        lut_pix = np.asarray(row['LUT_PIXEL'])
        return np.interp(wavelength, lut_wave, lut_pix)

    def pixel_to_wavelength(self, order: int, pixel) -> np.ndarray:
        row = self._dispersion_by_order[order]
        lut_wave = np.asarray(row['LUT_WAVELENGTH'])
        lut_pix = np.asarray(row['LUT_PIXEL'])
        srt = np.argsort(lut_pix)
        return np.interp(pixel, lut_pix[srt], lut_wave[srt])

    def orders_covering_wavelength(self, wavelength: float) -> list[int]:
        covering = []
        for order in self.orders():
            lo, hi = self.wavelength_range(order)
            if lo <= wavelength <= hi:
                covering.append(order)
        return covering

    # --- LSF shape ----------------------------------------------------------

    def sigma_at_pixel(self, order: int, x) -> np.ndarray:
        row = self._lsf_by_order[order]
        pixel = np.arange(int(row['N_PIXELS']), dtype=float)
        log_sigma_grid = np.asarray(row['WIDTH_LOG_SIGMA_GRID'])
        return np.exp(np.interp(x, pixel, log_sigma_grid))

    def _order_namespace(self, order: int):
        row = self._lsf_by_order[order]
        cfg = LSFConfig(
            shape_u_length_scale_factor=float(row['SHAPE_U_LENGTH_SCALE_FACTOR']),
            shape_kappa_sigma0=float(row['SHAPE_KAPPA_SIGMA0']),
            shape_kappa_sigmaf=float(row['SHAPE_KAPPA_SIGMAF']),
            shape_kappa_width_factor=float(row['SHAPE_KAPPA_WIDTH_FACTOR']),
            shape_jitter=float(row['SHAPE_JITTER']),
        )
        ns = SimpleNamespace(
            u=np.asarray(row['U_GRID']),
            u_inducing=np.asarray(row['U_INDUCING']),
            x_inducing=np.asarray(row['X_INDUCING']),
            cfg=cfg,
            shape_x_length_scale=float(row['SHAPE_X_LENGTH_SCALE']),
        )
        n_u, n_x = int(row['N_U_INDUCING']), int(row['N_X_INDUCING'])
        shape_coeffs = np.asarray(row['SHAPE_COEFFS']).reshape(n_u, n_x)
        return ns, shape_coeffs

    def lsf_at_pixel(self, order: int, x: float) -> tuple[np.ndarray, np.ndarray]:
        """ Returns (u [km/s], phi(u)) -- the peak-normalised LSF at pixel
            x within `order`, exactly reconstructed from the saved model
            (Gaussian core + 2D shape-departure GP conditional mean). """
        sigma = float(self.sigma_at_pixel(order, x))
        ns, shape_coeffs = self._order_namespace(order)
        departure = evaluate_departure(ns, np.array([sigma]), shape_coeffs, np.array([x]))[0]
        phi = gaussian_mean(ns.u, sigma) + departure
        return ns.u, phi

    def lsf_at_wavelength(self, order: int, wavelength: float) -> tuple[np.ndarray, np.ndarray]:
        x = float(self.wavelength_to_pixel(order, wavelength))
        return self.lsf_at_pixel(order, x)

    def departure_at_pixel(self, order: int, x: float) -> tuple[np.ndarray, np.ndarray]:
        """ Returns (u, phi(u) - Gaussian(u; sigma(x))) -- the LSF's
            departure from a pure Gaussian core at pixel x. """
        sigma = float(self.sigma_at_pixel(order, x))
        ns, shape_coeffs = self._order_namespace(order)
        departure = evaluate_departure(ns, np.array([sigma]), shape_coeffs, np.array([x]))[0]
        return ns.u, departure

    def fwhm_kms_at_pixel(self, order: int, x: float) -> float:
        return 2.354820045 * float(self.sigma_at_pixel(order, x))

    # --- composite across orders --------------------------------------------

    def composite_lsf_at_wavelength(
        self, wavelength: float, orders: list[int] = None,
        weights: dict | list | np.ndarray = None, u_grid: np.ndarray = None,
    ):
        """ Weighted composite phi(u) at `wavelength`, combining every
            order that covers it (or the given `orders`). weights: dict
            {order: weight}, or a list/array in the same order as
            `orders`; None -> equal weight. Returns (u_grid, phi,
            orders_used). """
        if orders is None:
            orders = self.orders_covering_wavelength(wavelength)
        if not orders:
            raise ValueError(f"No order in this file covers wavelength={wavelength}")

        if weights is None:
            weights = {o: 1.0 for o in orders}
        elif not isinstance(weights, dict):
            weights = dict(zip(orders, weights))
        missing = [o for o in orders if o not in weights]
        if missing:
            raise ValueError(f"No weight given for orders {missing}")
        w_total = sum(weights[o] for o in orders)

        contributions = {o: self.lsf_at_wavelength(o, wavelength) for o in orders}

        if u_grid is None:
            half_range = max(u.max() for u, _ in contributions.values())
            du = min(u[1] - u[0] for u, _ in contributions.values())
            n = int(round(2 * half_range / du)) + 1
            u_grid = np.linspace(-half_range, half_range, n)

        phi = np.zeros_like(u_grid)
        for o in orders:
            u_o, phi_o = contributions[o]
            phi += (weights[o] / w_total) * np.interp(u_grid, u_o, phi_o, left=0.0, right=0.0)

        return u_grid, phi, orders
