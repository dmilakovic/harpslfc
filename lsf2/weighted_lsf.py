"""
Weighted-average LSF over a wavelength RANGE (as opposed to
reconstruct.LSFLibrary.composite_lsf_at_wavelength, which combines
orders at a single point). A "segment" here is one (order, wavelength)
sample -- generate_segments() lays down N evenly-spaced samples per
order across whatever part of [wave_lo, wave_hi] that order covers, so
an order that only partially overlaps the range contributes only from
its overlapping portion, and a range spanning several orders naturally
gets segments from each of them.

Weights are supplied by the caller (e.g. per-segment S/N from the GUI's
table) and normalised to sum to 1 at compute time; they are NOT
re-derived from the data here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits

from .reconstruct import LSFLibrary

C_KMS = 2.99792458e5


def velocity_range_to_wavelength(center_nm: float, velocity_kms_fullwidth: float) -> tuple[float, float]:
    """ [center*(1-v/2c), center*(1+v/2c)] -- velocity_kms_fullwidth is
        the FULL width of the window (i.e. +/- half that in velocity). """
    half = velocity_kms_fullwidth / 2.0
    return center_nm * (1 - half / C_KMS), center_nm * (1 + half / C_KMS)


def generate_order_segments(lib: LSFLibrary, wave_lo: float, wave_hi: float,
                             n_segments_per_order: int = 5, orders: list[int] = None) -> dict:
    """ {order: [{'pixel','wavelength'}, ...]} -- for each covering
        order, n_segments_per_order pixel positions (bin-centred, i.e.
        pixel_lo + (i+0.5)/n * (pixel_hi-pixel_lo)) spanning that order's
        OWN overlap between its LFC-covered pixel range and
        [wave_lo, wave_hi]. Purely a quadrature device: within one order
        these are averaged with equal weight (see compute_weighted_lsf),
        so more segments only resolves how sigma(x)/D(u,x) vary within
        that order's span -- it does not change how much that order
        counts relative to any other order. That's controlled separately,
        by one weight per order (see compute_weighted_lsf's order_weights). """
    if wave_hi < wave_lo:
        wave_lo, wave_hi = wave_hi, wave_lo
    candidates = orders if orders is not None else lib.orders()
    n = max(1, n_segments_per_order)

    order_segments = {}
    for order in candidates:
        lo_o, hi_o = lib.wavelength_range(order)
        wlo, whi = max(lo_o, wave_lo), min(hi_o, wave_hi)
        if wlo >= whi:
            continue
        x_min, x_max, _ = lib.pixel_range(order)
        pixel_lo = float(np.clip(lib.wavelength_to_pixel(order, wlo), x_min, x_max))
        pixel_hi = float(np.clip(lib.wavelength_to_pixel(order, whi), x_min, x_max))
        if pixel_hi < pixel_lo:
            pixel_lo, pixel_hi = pixel_hi, pixel_lo

        fractions = (np.arange(n) + 0.5) / n
        pixels = pixel_lo + fractions * (pixel_hi - pixel_lo)
        order_segments[order] = [
            {'pixel': float(p), 'wavelength': float(lib.pixel_to_wavelength(order, p))} for p in pixels
        ]
    return order_segments


@dataclass
class WeightedLSFResult:
    u: np.ndarray
    phi: np.ndarray
    center_wavelength_nm: float
    velocity_range_kms: float          # NaN if the direct lo/hi input method was used
    wave_lo_nm: float
    wave_hi_nm: float
    segments: list                     # [{'order','pixel','wavelength','order_weight','order_weight_norm','weight_norm'}, ...]
    per_order_phi: dict = field(default_factory=dict)      # {order: (u_grid, phi_order)}, already weight-scaled
    per_segment_phi: list = field(default_factory=list)    # unscaled, one per segment, aligned with `segments`
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S'))


def compute_weighted_lsf(lib: LSFLibrary, order_segments: dict, order_weights: dict,
                          center_wavelength_nm: float = np.nan, velocity_range_kms: float = np.nan,
                          wave_lo_nm: float = np.nan, wave_hi_nm: float = np.nan,
                          u_grid: np.ndarray = None) -> WeightedLSFResult:
    """ Two-level average: within each order, its segments are averaged
        with EQUAL weight (pure quadrature -- see generate_order_segments'
        docstring); across orders, each order's averaged LSF is combined
        using order_weights (one number per order, e.g. S/N), normalised
        to sum to 1. """
    orders = [o for o in order_segments if order_segments[o]]
    if not orders:
        raise ValueError("compute_weighted_lsf: no segments given")
    missing = [o for o in orders if o not in order_weights]
    if missing:
        raise ValueError(f"No weight given for orders {missing}")
    w_total = float(sum(order_weights[o] for o in orders))
    if w_total <= 0:
        raise ValueError("order weights must sum to a positive number")

    all_contributions = {
        order: [lib.lsf_at_pixel(order, seg['pixel']) for seg in order_segments[order]]
        for order in orders
    }

    if u_grid is None:
        all_u = [u for contribs in all_contributions.values() for u, _ in contribs]
        half_range = max(u.max() for u in all_u)
        du = min(u[1] - u[0] for u in all_u)
        n_grid = int(round(2 * half_range / du)) + 1
        u_grid = np.linspace(-half_range, half_range, n_grid)

    phi = np.zeros_like(u_grid)
    per_order_phi = {}
    per_segment_phi = []
    out_segments = []
    for order in orders:
        order_weight_norm = order_weights[order] / w_total
        contribs = all_contributions[order]
        n_seg = len(contribs)

        phi_order_unweighted = np.zeros_like(u_grid)
        for seg, (u, seg_phi) in zip(order_segments[order], contribs):
            interpolated = np.interp(u_grid, u, seg_phi, left=0.0, right=0.0)
            phi_order_unweighted += interpolated / n_seg   # equal weight within the order
            per_segment_phi.append(interpolated)
            out_segments.append({
                **seg, 'order': order, 'order_weight': float(order_weights[order]),
                'order_weight_norm': float(order_weight_norm),
                'weight_norm': float(order_weight_norm / n_seg),
            })

        phi_order_weighted = phi_order_unweighted * order_weight_norm
        per_order_phi[order] = (u_grid, phi_order_weighted)
        phi += phi_order_weighted

    if np.isnan(wave_lo_nm):
        wave_lo_nm = min(s['wavelength'] for s in out_segments)
    if np.isnan(wave_hi_nm):
        wave_hi_nm = max(s['wavelength'] for s in out_segments)
    if np.isnan(center_wavelength_nm):
        center_wavelength_nm = 0.5 * (wave_lo_nm + wave_hi_nm)

    return WeightedLSFResult(
        u=u_grid, phi=phi, center_wavelength_nm=center_wavelength_nm,
        velocity_range_kms=velocity_range_kms, wave_lo_nm=wave_lo_nm, wave_hi_nm=wave_hi_nm,
        segments=out_segments, per_order_phi=per_order_phi, per_segment_phi=per_segment_phi,
    )


def fit_gaussian_to_lsf(u: np.ndarray, phi: np.ndarray) -> dict:
    """ Least-squares Gaussian fit (amplitude, center, sigma all free) to
        an already-computed LSF, for VISUAL comparison only -- this is
        deliberately not part of the saved model (the actual LSF is the
        Gaussian-core-plus-departure-GP construction throughout the rest
        of lsf2; this is just "what single Gaussian looks most like this
        result", for the GUI's departure-from-Gaussian panel). """
    from scipy.optimize import curve_fit

    def gaussian(u, amplitude, center, sigma):
        return amplitude * np.exp(-0.5 * ((u - center) / sigma) ** 2)

    amplitude0 = float(np.max(phi))
    center0 = float(u[np.argmax(phi)])
    weights = np.clip(phi, 0, None)
    if weights.sum() > 0:
        mean_est = np.average(u, weights=weights)
        var_est = np.average((u - mean_est) ** 2, weights=weights)
        sigma0 = float(np.sqrt(max(var_est, 1e-6)))
    else:
        sigma0 = float((u.max() - u.min()) / 6)

    try:
        popt, _ = curve_fit(gaussian, u, phi, p0=[amplitude0, center0, sigma0], maxfev=5000)
    except Exception:
        popt = [amplitude0, center0, sigma0]
    amplitude, center, sigma = (float(v) for v in popt)
    curve = gaussian(u, amplitude, center, sigma)

    return {'amplitude': amplitude, 'center': center, 'sigma': abs(sigma),
            'fwhm': 2.354820045 * abs(sigma), 'curve': curve, 'departure': phi - curve}


def _unique_hdu_name(hdul: fits.HDUList, base_name: str) -> str:
    existing = {h.name for h in hdul}
    if base_name not in existing:
        return base_name
    i = 2
    while f"{base_name}_{i}" in existing:
        i += 1
    return f"{base_name}_{i}"


def save_weighted_lsf_fits(result: WeightedLSFResult, path: str, hdu_name: str = "WEIGHTED_LSF",
                            overwrite_hdu: bool = False) -> tuple[str, str]:
    """ Saves `result` as two extensions -- <hdu_name> (a U_KMS/PHI
        binary table) and <hdu_name>_SEGMENTS (per-segment provenance:
        order, wavelength, weight, normalised weight) -- appended to
        `path` if it already exists, or written to a new file otherwise.
        Returns (path, hdu_name actually used). """
    if os.path.exists(path):
        with fits.open(path) as hdul:
            hdul = fits.HDUList([h.copy() for h in hdul])
    else:
        hdul = fits.HDUList([fits.PrimaryHDU()])

    name = hdu_name if overwrite_hdu else _unique_hdu_name(hdul, hdu_name)
    if overwrite_hdu and name in {h.name for h in hdul}:
        for seg_name in (name, f"{name}_SEGMENTS"):
            if seg_name in hdul:
                del hdul[seg_name]

    lsf_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='U_KMS', format='D', array=result.u),
        fits.Column(name='PHI', format='D', array=result.phi),
    ], name=name)
    lsf_hdu.header['CENTRWL'] = (result.center_wavelength_nm, 'Central wavelength [nm]')
    velrange = result.velocity_range_kms
    lsf_hdu.header['VELRANGE'] = ('N/A' if not np.isfinite(velrange) else velrange,
                                   'Velocity FWHM window [km/s] (N/A: direct range)')
    lsf_hdu.header['WAVE_LO'] = (result.wave_lo_nm, 'Lower wavelength bound [nm]')
    lsf_hdu.header['WAVE_HI'] = (result.wave_hi_nm, 'Upper wavelength bound [nm]')
    lsf_hdu.header['NSEG'] = (len(result.segments), 'Number of contributing segments')
    lsf_hdu.header['DATE'] = result.created_at
    lsf_hdu.header['COMMENT'] = 'Weighted-average LSF built by harps.lsf2.weighted_lsf. See <name>_SEGMENTS.'

    orders = [s['order'] for s in result.segments]
    pixels = [s['pixel'] for s in result.segments]
    wavelengths = [s['wavelength'] for s in result.segments]
    order_weights = [s['order_weight'] for s in result.segments]
    order_weights_norm = [s['order_weight_norm'] for s in result.segments]
    weights_norm = [s['weight_norm'] for s in result.segments]
    segments_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='ORDER', format='J', array=np.array(orders)),
        fits.Column(name='PIXEL', format='D', array=np.array(pixels)),
        fits.Column(name='WAVELENGTH', format='D', array=np.array(wavelengths)),
        fits.Column(name='ORDER_WEIGHT', format='D', array=np.array(order_weights)),
        fits.Column(name='ORDER_WEIGHT_NORM', format='D', array=np.array(order_weights_norm)),
        fits.Column(name='WEIGHT_NORM', format='D', array=np.array(weights_norm)),
    ], name=f"{name}_SEGMENTS")
    segments_hdu.header['COMMENT'] = ('WEIGHT_NORM = ORDER_WEIGHT_NORM / (segments in that order): '
                                       'segments within an order are equal-weighted quadrature points, '
                                       'not independently weighted -- see ORDER_WEIGHT(_NORM) for the '
                                       'one real weight per order (e.g. S/N).')

    hdul.append(lsf_hdu)
    hdul.append(segments_hdu)
    hdul.writeto(path, overwrite=True)
    return path, name


def load_weighted_lsf_fits(path: str, hdu_name: str = "WEIGHTED_LSF") -> WeightedLSFResult:
    with fits.open(path) as hdul:
        lsf_hdu = hdul[hdu_name]
        seg_hdu = hdul[f"{hdu_name}_SEGMENTS"]
        u = np.asarray(lsf_hdu.data['U_KMS'], dtype=float)
        phi = np.asarray(lsf_hdu.data['PHI'], dtype=float)
        header = lsf_hdu.header
        velrange = header.get('VELRANGE', np.nan)
        if isinstance(velrange, str):
            velrange = np.nan
        segments = [
            {'order': int(o), 'pixel': float(p), 'wavelength': float(w),
             'order_weight': float(ow), 'order_weight_norm': float(own), 'weight_norm': float(wn)}
            for o, p, w, ow, own, wn in zip(
                seg_hdu.data['ORDER'], seg_hdu.data['PIXEL'], seg_hdu.data['WAVELENGTH'],
                seg_hdu.data['ORDER_WEIGHT'], seg_hdu.data['ORDER_WEIGHT_NORM'], seg_hdu.data['WEIGHT_NORM'])
        ]
    return WeightedLSFResult(
        u=u, phi=phi, center_wavelength_nm=header.get('CENTRWL', np.nan),
        velocity_range_kms=velrange,
        wave_lo_nm=header.get('WAVE_LO', np.nan), wave_hi_nm=header.get('WAVE_HI', np.nan),
        segments=segments, created_at=header.get('DATE', ''),
    )
