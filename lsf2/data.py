"""
Loading one order's worth of data (flux, error, LFC line list) into a
plain, explicit container -- replaces the module-level globals the
original script read (flux_raw, err_raw, line_list, wavelength, ...).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SPEED_OF_LIGHT


@dataclass
class OrderData:
    order: int
    n_pixels: int
    pixel: np.ndarray            # np.arange(n_pixels)
    flux_raw: np.ndarray
    err_raw: np.ndarray
    left_edge: np.ndarray
    peak_pixel: np.ndarray       # catalogue (pre-subpixel-refined) position
    right_edge: np.ndarray
    frequency: np.ndarray        # Hz
    wavelength: np.ndarray       # nm, vacuum
    n_lines: int
    x_min: float
    x_max: float

    def rescaled_position(self, x):
        """ Pixel position rescaled to [-1, 1] over the LFC lines' own
            pixel range -- used by the background design matrix; see
            envelope.py for why this rescaling matters numerically. """
        return 2 * (np.asarray(x) - self.x_min) / (self.x_max - self.x_min) - 1


def load_order_from_spectrum(filename: str, wavereference: str, order: int,
                              f0: float = 7.40e9, fr: float = 18e9,
                              sOrder: int = 40, overwrite: bool = False) -> OrderData:
    """ Load one order directly from an ESPRESSO S2D LFC exposure via
        harps.spectrum, the same entry point the production `harps.lsf`
        pipeline uses. """
    import harps.spectrum as hc

    spec = hc.ESPRESSO(
        filename, f0=f0, fr=fr, overwrite=overwrite, sOrder=sOrder,
        wavereference=wavereference,
    )
    spec.process(fittype='gauss', do_comb_specific=True)

    linelist = spec['linelist']
    flux_raw = np.asarray(spec.flux[order], dtype=float)
    err_raw = np.asarray(spec.error[order], dtype=float)

    cut = np.where(linelist['order'] == order)[0]
    cut2 = np.where(spec.line_positions['order'] == order)[0]
    if len(cut) == 0 or len(cut2) == 0:
        raise ValueError(f"No LFC lines found for order {order} in {filename}")

    line_list = np.array(
        [list(row[0]) + [row[1]]
         for row in np.transpose([spec.line_positions[cut2], linelist[cut]['freq']])],
        dtype=float,
    )
    return _build_order_data(order, flux_raw, err_raw, line_list)


def load_order_from_text(spectrum_file: str, lines_file: str, order: int) -> OrderData:
    """ Load from the plain-text (spectrum, lines) file pair used during
        early development / unit testing -- kept for parity with the
        original script's DATA_SOURCE == 'file' branch. """
    spectrum = np.loadtxt(spectrum_file, comments='#')
    flux_raw, err_raw, _bkg_col, _env_col = spectrum.T
    line_list = np.loadtxt(lines_file, comments='#')
    return _build_order_data(order, flux_raw, err_raw, line_list)


def _build_order_data(order, flux_raw, err_raw, line_list) -> OrderData:
    n_pixels = len(flux_raw)
    pixel = np.arange(n_pixels)

    left_edge, peak_pixel, right_edge, frequency = (
        line_list[:, 1], line_list[:, 2], line_list[:, 3], line_list[:, 4])
    wavelength = SPEED_OF_LIGHT / frequency * 1e9  # vacuum wavelength, nm
    n_lines = len(peak_pixel)

    return OrderData(
        order=order, n_pixels=n_pixels, pixel=pixel,
        flux_raw=flux_raw, err_raw=err_raw,
        left_edge=left_edge, peak_pixel=peak_pixel, right_edge=right_edge,
        frequency=frequency, wavelength=wavelength, n_lines=n_lines,
        x_min=float(peak_pixel.min()), x_max=float(peak_pixel.max()),
    )
