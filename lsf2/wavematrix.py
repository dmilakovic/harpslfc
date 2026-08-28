"""
Loader for the ESO DRS WAVE_MATRIX product: a dense pixel->wavelength
lookup, one row per S2D "order" (which, for ESPRESSO, already encodes
both the physical echelle order and the image-slicer slice -- see
`physical_order`/`slice_index` below). Row index matches the `order`
argument used throughout the rest of lsf2 (harps.spectrum's
flux[order]/linelist['order'] convention), by construction of the ESO
pipeline: this is the SAME "order" numbering harps.lsf2.data uses to pull
LFC lines out of the S2D file, so row N here lines up with lsf2 output
order N with no re-indexing needed.

Units: ESO WAVE_MATRIX products store wavelength in Angstrom (confirmed
by inspecting a real file -- ESPRESSO's ~380-788nm range appears as
~3772-7908, i.e. x10 nm). lsf2 works in nm throughout, so
WaveMatrix.wavelength_nm() is the one to use when comparing against an
LSFLibrary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits


@dataclass
class WaveMatrix:
    data_angstrom: np.ndarray     # (n_orders, n_pixels), 0 or NaN where invalid
    header: fits.Header
    qc_flag: dict                 # {order: bool}, from 'ESO QC ORDER<N+1> LFC FIT WAVE'
    path: str

    @classmethod
    def load(cls, path: str, angstrom: bool = True) -> "WaveMatrix":
        with fits.open(path) as hdul:
            header = hdul[0].header
            image_hdu = next((h for h in hdul if h.data is not None and h.data.ndim == 2), None)
            if image_hdu is None:
                raise ValueError(f"No 2D image extension found in {path}")
            data = np.asarray(image_hdu.data, dtype=float)

        qc_flag = {}
        n_orders = data.shape[0]
        for row in range(n_orders):
            key = f'ESO QC ORDER{row + 1} LFC FIT WAVE'
            if key in header:
                qc_flag[row] = bool(header[key])

        if not angstrom:
            data = data * 10.0   # nm -> "Angstrom-equivalent" internal storage
        return cls(data_angstrom=data, header=header, qc_flag=qc_flag, path=path)

    @property
    def n_orders(self) -> int:
        return self.data_angstrom.shape[0]

    @property
    def n_pixels(self) -> int:
        return self.data_angstrom.shape[1]

    def orders(self) -> list[int]:
        """ Every row with at least some valid (>0, finite) wavelength. """
        valid_rows = np.where(np.any(np.isfinite(self.data_angstrom) & (self.data_angstrom > 0), axis=1))[0]
        return [int(r) for r in valid_rows]

    def physical_order(self, order: int) -> int:
        """ ESPRESSO convention: two consecutive S2D rows per physical
            echelle order (one per image-slicer slice). """
        return order // 2

    def slice_index(self, order: int) -> int:
        return order % 2

    def _row_nm(self, order: int) -> np.ndarray:
        return self.data_angstrom[order] * 0.1

    def valid_mask(self, order: int) -> np.ndarray:
        row = self.data_angstrom[order]
        return np.isfinite(row) & (row > 0)

    def wavelength_range_nm(self, order: int) -> tuple[float, float]:
        mask = self.valid_mask(order)
        if not mask.any():
            return np.nan, np.nan
        row_nm = self._row_nm(order)
        return float(row_nm[mask].min()), float(row_nm[mask].max())

    def wavelength_nm(self, order: int, pixel) -> np.ndarray:
        """ Interpolated wavelength [nm] at arbitrary (non-integer)
            pixel(s) within `order`, from the dense per-pixel row. """
        mask = self.valid_mask(order)
        pixel_valid = np.arange(self.n_pixels)[mask]
        row_nm = self._row_nm(order)[mask]
        return np.interp(pixel, pixel_valid, row_nm)

    def pixel_at_wavelength(self, order: int, wavelength_nm) -> np.ndarray:
        mask = self.valid_mask(order)
        pixel_valid = np.arange(self.n_pixels)[mask]
        row_nm = self._row_nm(order)[mask]
        srt = np.argsort(row_nm)
        return np.interp(wavelength_nm, row_nm[srt], pixel_valid[srt])

    def orders_covering_wavelength(self, wavelength_nm: float, orders=None) -> list[int]:
        candidates = orders if orders is not None else self.orders()
        covering = []
        for order in candidates:
            lo, hi = self.wavelength_range_nm(order)
            if np.isfinite(lo) and lo <= wavelength_nm <= hi:
                covering.append(order)
        return covering
