"""
sigma(x): the LSF's Gaussian-core velocity width, as a function of pixel
position, interpolated from a dense log(sigma) grid fit at `pixel` (see
width.fit_width). Kept in its own tiny module (no imports from width.py
or shape.py) so both can depend on it without a circular import --
width.fit_width needs shape.evaluate_departure, and shape.py needs this
function.
"""
from __future__ import annotations

import numpy as np

from .config import LSFConfig


def width(x, log_sigma_grid: np.ndarray, pixel: np.ndarray, cfg: LSFConfig,
          v_per_pixel_typical: float):
    """ sigma(x) [km/s], interpolated from a log(sigma) grid already fit
        at the integer pixel positions `pixel`. Floored at a small
        fraction of the typical velocity-per-pixel scale to keep sigma
        strictly positive under interpolation/extrapolation. """
    floor = cfg.width_sigma_floor_frac * v_per_pixel_typical
    return np.maximum(np.exp(np.interp(x, pixel.astype(float), log_sigma_grid)), floor)
