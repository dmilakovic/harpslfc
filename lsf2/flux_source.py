"""
Lightweight, read-only loader for an S2D exposure's flux (and, if present,
error) extension -- used only to overplot the observed LFC spectrum in
the wavecal GUI's "overlapping orders" tab. Deliberately does NOT go
through harps.spectrum.ESPRESSO's full processing pipeline (background
subtraction, line fitting, ...) since all this needs is the raw counts
per pixel for a visual line-centre check; running the full pipeline just
to look at a spectrum would be needless overhead for a diagnostic tool.

ESO DRS S2D products name the flux extension 'FLUX' (error: 'ERR') by
convention; this falls back to the first 2D image extension if that name
isn't found, so it also works against non-standard files.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits


@dataclass
class FluxSource:
    flux: np.ndarray          # (n_orders, n_pixels)
    error: np.ndarray = None  # same shape, or None if no ERR extension found
    path: str = None

    @classmethod
    def load(cls, path: str) -> "FluxSource":
        with fits.open(path) as hdul:
            flux_hdu = next((h for h in hdul if h.name.upper() == 'FLUX' and h.data is not None), None)
            if flux_hdu is None:
                flux_hdu = next((h for h in hdul if h.data is not None and h.data.ndim == 2), None)
            if flux_hdu is None:
                raise ValueError(f"No FLUX (or any 2D image) extension found in {path}")
            flux = np.asarray(flux_hdu.data, dtype=float)

            err_hdu = next((h for h in hdul if h.name.upper() in ('ERR', 'ERROR') and h.data is not None), None)
            error = np.asarray(err_hdu.data, dtype=float) if err_hdu is not None else None

        return cls(flux=flux, error=error, path=path)

    @property
    def n_orders(self) -> int:
        return self.flux.shape[0]

    @property
    def n_pixels(self) -> int:
        return self.flux.shape[1]

    def order_flux(self, order: int) -> np.ndarray:
        if not (0 <= order < self.n_orders):
            raise ValueError(f"order {order} out of range [0, {self.n_orders})")
        return self.flux[order]
