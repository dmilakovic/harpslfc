"""
Compares an lsf2 dispersion solution (from an LSFLibrary, i.e. an
lsf2.cli_run output FITS file) against an independent WAVE_MATRIX
solution, order by order and across the whole detector.

Three kinds of residual are distinguished throughout, since they answer
different questions:
  - "vs theoretical": lsf2's own fitted line_position, converted to
    wavelength by EACH solution, compared to the LFC line's own
    known/theoretical wavelength (line_wavelength) -- this is each
    solution's own absolute accuracy at the LFC line positions.
  - "lsf2 - wave_matrix": the two solutions' wavelength (or velocity)
    difference, evaluated on a dense pixel grid -- a direct, continuous
    comparison that doesn't depend on where the LFC lines happen to fall.
  - "cross-order": the same two comparisons, but for orders/slices whose
    wavelength coverage overlaps, so the *same* wavelength is checked
    against two (or more) independent line measurements.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .reconstruct import LSFLibrary
from .wavematrix import WaveMatrix

C_MS = 2.99792458e8


def common_orders(lib: LSFLibrary, wavemat: WaveMatrix) -> list[int]:
    return sorted(set(lib.orders()) & set(wavemat.orders()))


@dataclass
class OrderComparison:
    order: int
    pixel: np.ndarray
    wavelength_lsf2_nm: np.ndarray
    wavelength_wavemat_nm: np.ndarray
    residual_ms: np.ndarray             # (lsf2 - wave_matrix), m/s, dense curve

    line_wavelength_nm: np.ndarray      # theoretical LFC wavelength, per line
    line_position_lsf2: np.ndarray      # lsf2's own fitted pixel position, per line
    line_residual_lsf2_ms: np.ndarray   # lsf2's own solution vs theoretical, at its own fitted position
    line_residual_wavemat_ms: np.ndarray  # wave_matrix vs theoretical, evaluated AT lsf2's fitted position
    line_continuous_residual_ms: np.ndarray  # (lsf2 - wave_matrix) evaluated AT each line position
    line_position_err_pixels: np.ndarray     # lsf2's own per-line position fit uncertainty
    line_residual_err_ms: np.ndarray         # that uncertainty propagated to m/s via the local dispersion
                                              # (LINE_V_PIX); shared by all three per-line series above,
                                              # since all three differ only in which wavelength solution(s)
                                              # get evaluated AT that (uncertain) position -- see compare_order.
    line_even_mask: np.ndarray               # True for every other line, ordered by wavelength (req. 4)
    line_v_pix_kms: np.ndarray                # local km/s-per-pixel at each line (for exact pixel-unit conversion)
    dense_v_pix_kms: np.ndarray                # same, at each dense-grid point (via gradient)

    rms_ms: float
    median_ms: float
    qc_flag: bool = None


def compare_order(lib: LSFLibrary, wavemat: WaveMatrix, order: int, n_samples: int = 2000) -> OrderComparison:
    x_min, x_max, _ = lib.pixel_range(order)
    wm_lo, wm_hi = wavemat.wavelength_range_nm(order)
    # dense comparison only over the pixel range BOTH solutions actually cover
    pixel_lo = max(x_min, wavemat.pixel_at_wavelength(order, wm_lo))
    pixel_hi = min(x_max, wavemat.pixel_at_wavelength(order, wm_hi))
    if pixel_lo > pixel_hi:
        pixel_lo, pixel_hi = pixel_hi, pixel_lo
    pixel = np.linspace(pixel_lo, pixel_hi, n_samples)

    wavelength_lsf2 = lib.pixel_to_wavelength(order, pixel)
    wavelength_wavemat = wavemat.wavelength_nm(order, pixel)
    residual_ms = C_MS * (wavelength_lsf2 - wavelength_wavemat) / wavelength_wavemat

    disp_row = lib._dispersion_by_order[order]
    line_wavelength = np.asarray(disp_row['LINE_WAVELENGTH'])
    line_position = np.asarray(disp_row['LINE_POSITION'])
    line_v_pix = np.asarray(disp_row['LINE_V_PIX'])                    # km/s per pixel, local scale
    line_position_err_pixels = np.asarray(disp_row['TRAIN_POSITION_ERR'])

    wavelength_lsf2_at_lines = lib.pixel_to_wavelength(order, line_position)
    wavelength_wavemat_at_lines = wavemat.wavelength_nm(order, line_position)
    line_residual_lsf2 = C_MS * (wavelength_lsf2_at_lines - line_wavelength) / line_wavelength
    line_residual_wavemat = C_MS * (wavelength_wavemat_at_lines - line_wavelength) / line_wavelength
    line_continuous_residual = C_MS * (wavelength_lsf2_at_lines - wavelength_wavemat_at_lines) / wavelength_wavemat_at_lines

    # A line's fitted PIXEL position is uncertain (line_position_err_pixels);
    # both "vs theoretical" series above, and the continuous-at-line-position
    # series, are all wavelength solutions evaluated AT that one uncertain
    # x -- so the same position uncertainty, converted through the local
    # dispersion (km/s per pixel -> m/s), is the right error bar for all
    # three. This is not a rigorous re-derivation of each curve's own
    # slope-dependent propagation, just the shared, dominant term -- good
    # enough for a diagnostic plot, not a formal uncertainty budget.
    line_residual_err_ms = np.abs(line_position_err_pixels * line_v_pix) * 1e3

    sort_idx = np.argsort(line_wavelength)
    line_even_mask = np.zeros(len(line_wavelength), dtype=bool)
    line_even_mask[sort_idx[0::2]] = True

    # local km/s-per-pixel, for exact (not globally-approximated) pixel-unit
    # display conversion -- at the dense grid via a numerical gradient of
    # the wave_matrix solution (any real dispersion solution works equally
    # well for this; wave_matrix is arbitrary here), at the lines directly
    # from the stored value already used throughout the pipeline.
    dense_v_pix_kms = (C_MS / 1e3) * np.gradient(wavelength_wavemat, pixel) / wavelength_wavemat

    return OrderComparison(
        order=order, pixel=pixel,
        wavelength_lsf2_nm=wavelength_lsf2, wavelength_wavemat_nm=wavelength_wavemat,
        residual_ms=residual_ms,
        line_wavelength_nm=line_wavelength, line_position_lsf2=line_position,
        line_residual_lsf2_ms=line_residual_lsf2, line_residual_wavemat_ms=line_residual_wavemat,
        line_continuous_residual_ms=line_continuous_residual,
        line_position_err_pixels=line_position_err_pixels, line_residual_err_ms=line_residual_err_ms,
        line_even_mask=line_even_mask, line_v_pix_kms=line_v_pix, dense_v_pix_kms=dense_v_pix_kms,
        rms_ms=float(np.sqrt(np.mean(residual_ms ** 2))), median_ms=float(np.median(residual_ms)),
        qc_flag=wavemat.qc_flag.get(order),
    )


def summary_table(lib: LSFLibrary, wavemat: WaveMatrix, orders: list[int] = None) -> list[dict]:
    """ One row per common order: quick numbers for a sortable overview
        table (req. 3, "accuracy across different slices and orders"). """
    orders = orders if orders is not None else common_orders(lib, wavemat)
    rows = []
    for order in orders:
        cmp = compare_order(lib, wavemat, order)
        lo, hi = wavemat.wavelength_range_nm(order)
        rows.append({
            'order': order,
            'physical_order': wavemat.physical_order(order),
            'slice': wavemat.slice_index(order),
            'wave_lo_nm': lo, 'wave_hi_nm': hi,
            'rms_ms': cmp.rms_ms, 'median_ms': cmp.median_ms,
            'lsf2_line_rms_ms': float(np.sqrt(np.nanmean(cmp.line_residual_lsf2_ms ** 2))),
            'wavemat_line_rms_ms': float(np.sqrt(np.nanmean(cmp.line_residual_wavemat_ms ** 2))),
            'n_lines': len(cmp.line_wavelength_nm),
            'qc_flag': cmp.qc_flag,
        })
    return rows


def two_d_difference(lib: LSFLibrary, wavemat: WaveMatrix, orders: list[int] = None,
                      n_pixel_samples: int = 500):
    """ (pixel_grid, orders, diff_ms[order_index, pixel_index]) -- lsf2
        minus wave_matrix, sampled on a COMMON pixel grid (0..n_pixels-1,
        the physical detector axis shared by every order/slice) so it can
        be shown as a single 2D image across the whole order list. NaN
        where either solution doesn't cover that pixel for that order. """
    orders = sorted(orders) if orders is not None else common_orders(lib, wavemat)
    n_pixels_detector = wavemat.n_pixels
    pixel_grid = np.linspace(0, n_pixels_detector - 1, n_pixel_samples)
    diff = np.full((len(orders), n_pixel_samples), np.nan)

    for i, order in enumerate(orders):
        x_min, x_max, _ = lib.pixel_range(order)
        wm_mask = wavemat.valid_mask(order)
        if not wm_mask.any():
            continue
        wm_pixels = np.arange(n_pixels_detector)[wm_mask]
        wm_lo, wm_hi = wm_pixels.min(), wm_pixels.max()
        lo = max(x_min, wm_lo)
        hi = min(x_max, wm_hi)
        in_range = (pixel_grid >= lo) & (pixel_grid <= hi)
        if not in_range.any():
            continue
        px = pixel_grid[in_range]
        wavelength_lsf2 = lib.pixel_to_wavelength(order, px)
        wavelength_wavemat = wavemat.wavelength_nm(order, px)
        diff[i, in_range] = C_MS * (wavelength_lsf2 - wavelength_wavemat) / wavelength_wavemat

    return pixel_grid, orders, diff


def overlapping_orders_at_wavelength(lib: LSFLibrary, wavemat: WaveMatrix, wavelength_nm: float,
                                      orders: list[int] = None) -> list[int]:
    """ Every common order whose (lsf2 AND wave_matrix) coverage includes
        wavelength_nm -- the group to overplot for req. 2/3 (image-slicer
        slice pairs and/or adjacent overlapping echelle orders). """
    candidates = orders if orders is not None else common_orders(lib, wavemat)
    covering = []
    for order in candidates:
        lib_lo, lib_hi = lib.wavelength_range(order)
        wm_lo, wm_hi = wavemat.wavelength_range_nm(order)
        lo, hi = max(lib_lo, wm_lo), min(lib_hi, wm_hi)
        if lo <= wavelength_nm <= hi:
            covering.append(order)
    return covering


def pooled_percentiles(values) -> tuple:
    """ (median, p16, p84) -- the median and the central-68% interval
        bounds, ignoring non-finite entries. p16/p84 rather than
        mean+/-std since the per-line residuals are not assumed Gaussian
        (and the whole point of showing this is to let genuinely skewed
        or heavy-tailed behaviour show up, not average it away). """
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(values)), float(np.percentile(values, 16)), float(np.percentile(values, 84))


def moving_median_trend(x, y, window: int = 7):
    """ (x_sorted, trend) -- a simple sliding-window median trend line,
        for visualising a systematic even/odd split (req. 4) without
        assuming a functional form or adding a curve-fitting dependency.
        Robust to outliers by construction; window is in POINTS, not a
        physical unit, since points are not evenly spaced in x. """
    x, y = np.asarray(x), np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    n = len(x_s)
    half = max(1, window // 2)
    trend = np.array([np.median(y_s[max(0, i - half):min(n, i + half + 1)]) for i in range(n)])
    return x_s, trend


def suggest_partner_order(lib: LSFLibrary, wavemat: WaveMatrix, order: int):
    """ Best guess at "the other order/slice covering the same
        wavelength range" (req. 3): prefer the ESPRESSO image-slicer
        partner (the other row sharing the same physical_order), and
        fall back to whichever other common order overlaps this one's
        wavelength range the most, if no slicer partner is available
        (e.g. a non-ESPRESSO WAVE_MATRIX, or the partner not present in
        this lsf2 file). Returns None if nothing suitable is found. """
    candidates = [o for o in common_orders(lib, wavemat) if o != order]
    if not candidates:
        return None

    physical = wavemat.physical_order(order)
    same_physical = [o for o in candidates if wavemat.physical_order(o) == physical]
    if same_physical:
        return same_physical[0]

    lo, hi = wavemat.wavelength_range_nm(order)
    best_order, best_overlap = None, 0.0
    for o in candidates:
        o_lo, o_hi = wavemat.wavelength_range_nm(o)
        overlap = min(hi, o_hi) - max(lo, o_lo)
        if overlap > best_overlap:
            best_order, best_overlap = o, overlap
    return best_order
