"""
Saves/loads a list of LSFOrderResult objects to/from a single FITS file.

Layout (chosen with the user, see project discussion):
  - one binary table row per analysed order in each extension (not one
    HDU per order, not fixed-size image cubes), since array lengths
    (n_pixels, n_lines, LUT length) can differ order to order and a
    binary table with variable-length ('P') columns handles that
    natively while staying inspectable with astropy.table.Table.read.
  - the dispersion solution is stored BOTH ways: a dense wavelength<->
    pixel lookup table (for quick, dependency-light reconstruction) AND
    the raw Chebyshev/GP fit products (for anyone who wants to re-
    condition the GP exactly, e.g. to get a posterior variance at an
    arbitrary new wavelength rather than just the mean).

Extension 1, 'LSF': the LSF model (width grid + shape-departure GP) for
every analysed order.
Extension 2, 'DISPERSION': the wavelength solution for the same orders.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits

from .result import LSFOrderResult

_LSF_COLUMN_SPECS = [
    # (name, fits_format_from_field, is_scalar)
    ('ORDER', 'J'),
    ('N_PIXELS', 'J'),
    ('X_MIN', 'D'),
    ('X_MAX', 'D'),
    ('WIDTH_LOG_SIGMA_GRID', 'PD()'),
    ('U_GRID', 'PD()'),
    ('DU', 'D'),
    ('U_INDUCING', 'PD()'),
    ('X_INDUCING', 'PD()'),
    ('N_U_INDUCING', 'J'),
    ('N_X_INDUCING', 'J'),
    ('SHAPE_COEFFS', 'PD()'),   # flattened row-major (n_u_inducing, n_x_inducing); reshape on read
    ('SHAPE_X_LENGTH_SCALE', 'D'),
    ('SHAPE_U_LENGTH_SCALE_FACTOR', 'D'),
    ('SHAPE_KAPPA_SIGMA0', 'D'),
    ('SHAPE_KAPPA_SIGMAF', 'D'),
    ('SHAPE_KAPPA_WIDTH_FACTOR', 'D'),
    ('SHAPE_JITTER', 'D'),
    ('SIGMA_REF', 'D'),
    ('CHI2_PER_DOF', 'D'),
    ('N_ITERATIONS_RUN', 'J'),
    ('CONVERGED', 'L'),
    ('N_LINES', 'J'),
]

_DISPERSION_COLUMN_SPECS = [
    ('ORDER', 'J'),
    ('LUT_WAVELENGTH', 'PD()'),
    ('LUT_PIXEL', 'PD()'),
    ('POLY_COEFFS', 'PD()'),
    ('POLY_COEFFS_COV', 'PD()'),   # flattened (degree+1, degree+1); reshape on read
    ('POLY_X_LO', 'D'),
    ('POLY_X_HI', 'D'),
    ('POLY_DEGREE', 'J'),
    ('GP_KERNEL_TYPE', '16A'),
    ('GP_LENGTH_SCALE_JSON', '64A'),   # json-encoded float or {'length','period','gamma'} dict
    ('GP_SIGNAL_STD', 'D'),
    ('LINE_WAVELENGTH', 'PD()'),
    ('LINE_POSITION', 'PD()'),
    ('LINE_V_PIX', 'PD()'),
    ('TRAIN_WAVELENGTH', 'PD()'),
    ('TRAIN_POSITION', 'PD()'),
    ('TRAIN_POSITION_ERR', 'PD()'),
]


def derive_output_path(input_filename: str, output_root: str = None) -> str:
    """ <output_root or (dirname(input)/lsf_vel)>/<basename>_lsf_vel<ext> """
    directory, filename = os.path.split(input_filename)
    basename, ext = os.path.splitext(filename)
    out_dir = output_root if output_root is not None else os.path.join(directory, 'lsf_vel')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{basename}_lsf_vel{ext}")


def _column(name, fmt, values):
    return fits.Column(name=name, format=fmt, array=values)


def _build_lsf_hdu(results: list[LSFOrderResult]) -> fits.BinTableHDU:
    cols = []
    getters = {
        'ORDER': lambda r: r.order,
        'N_PIXELS': lambda r: r.n_pixels,
        'X_MIN': lambda r: r.x_min,
        'X_MAX': lambda r: r.x_max,
        'WIDTH_LOG_SIGMA_GRID': lambda r: np.asarray(r.width_log_sigma_grid, dtype=float),
        'U_GRID': lambda r: np.asarray(r.u_grid, dtype=float),
        'DU': lambda r: r.du,
        'U_INDUCING': lambda r: np.asarray(r.u_inducing, dtype=float),
        'X_INDUCING': lambda r: np.asarray(r.x_inducing, dtype=float),
        'N_U_INDUCING': lambda r: r.shape_coeffs.shape[0],
        'N_X_INDUCING': lambda r: r.shape_coeffs.shape[1],
        'SHAPE_COEFFS': lambda r: np.asarray(r.shape_coeffs, dtype=float).ravel(),
        'SHAPE_X_LENGTH_SCALE': lambda r: r.shape_x_length_scale,
        'SHAPE_U_LENGTH_SCALE_FACTOR': lambda r: r.shape_u_length_scale_factor,
        'SHAPE_KAPPA_SIGMA0': lambda r: r.shape_kappa_sigma0,
        'SHAPE_KAPPA_SIGMAF': lambda r: r.shape_kappa_sigmaf,
        'SHAPE_KAPPA_WIDTH_FACTOR': lambda r: r.shape_kappa_width_factor,
        'SHAPE_JITTER': lambda r: r.shape_jitter,
        'SIGMA_REF': lambda r: r.sigma_ref,
        'CHI2_PER_DOF': lambda r: r.chi2_per_dof,
        'N_ITERATIONS_RUN': lambda r: r.n_iterations_run,
        'CONVERGED': lambda r: r.converged,
        'N_LINES': lambda r: r.n_lines,
    }
    for name, fmt in _LSF_COLUMN_SPECS:
        values = [getters[name](r) for r in results]
        if fmt == 'PD()':
            arr = np.empty(len(results), dtype=object)
            for i, v in enumerate(values):
                arr[i] = v
            cols.append(_column(name, fmt, arr))
        else:
            cols.append(_column(name, fmt, np.array(values)))
    hdu = fits.BinTableHDU.from_columns(cols, name='LSF')
    hdu.header['COMMENT'] = 'One row per order. SHAPE_COEFFS is flattened row-major (N_U_INDUCING, N_X_INDUCING).'
    return hdu


def _build_dispersion_hdu(results: list[LSFOrderResult]) -> fits.BinTableHDU:
    cols = []
    getters = {
        'ORDER': lambda r: r.order,
        'LUT_WAVELENGTH': lambda r: np.asarray(r.dispersion_lut_wavelength, dtype=float),
        'LUT_PIXEL': lambda r: np.asarray(r.dispersion_lut_pixel, dtype=float),
        'POLY_COEFFS': lambda r: np.asarray(r.dispersion_poly_coeffs, dtype=float),
        'POLY_COEFFS_COV': lambda r: np.asarray(r.dispersion_poly_coeffs_cov, dtype=float).ravel(),
        'POLY_X_LO': lambda r: r.dispersion_poly_x_lo,
        'POLY_X_HI': lambda r: r.dispersion_poly_x_hi,
        'POLY_DEGREE': lambda r: r.dispersion_poly_degree,
        'GP_KERNEL_TYPE': lambda r: r.dispersion_gp_kernel_type,
        'GP_LENGTH_SCALE_JSON': lambda r: json.dumps(r.dispersion_gp_length_scale),
        'GP_SIGNAL_STD': lambda r: r.dispersion_gp_signal_std,
        'LINE_WAVELENGTH': lambda r: np.asarray(r.line_wavelength, dtype=float),
        'LINE_POSITION': lambda r: np.asarray(r.line_position, dtype=float),
        'LINE_V_PIX': lambda r: np.asarray(r.line_v_pix, dtype=float),
        'TRAIN_WAVELENGTH': lambda r: np.asarray(r.dispersion_train_wavelength, dtype=float),
        'TRAIN_POSITION': lambda r: np.asarray(r.dispersion_train_position, dtype=float),
        'TRAIN_POSITION_ERR': lambda r: np.asarray(r.dispersion_train_position_err, dtype=float),
    }
    for name, fmt in _DISPERSION_COLUMN_SPECS:
        values = [getters[name](r) for r in results]
        if fmt == 'PD()':
            arr = np.empty(len(results), dtype=object)
            for i, v in enumerate(values):
                arr[i] = v
            cols.append(_column(name, fmt, arr))
        elif fmt.endswith('A'):
            cols.append(_column(name, fmt, np.array(values, dtype=f'S{fmt[:-1]}')))
        else:
            cols.append(_column(name, fmt, np.array(values)))
    hdu = fits.BinTableHDU.from_columns(cols, name='DISPERSION')
    hdu.header['COMMENT'] = 'One row per order. POLY_COEFFS_COV is flattened (POLY_DEGREE+1, POLY_DEGREE+1).'
    return hdu


def save_lsf_fits(results: list[LSFOrderResult], input_filename: str,
                   output_path: str = None, output_root: str = None) -> str:
    """ Writes `results` to a new FITS file and returns its path.
        output_path overrides the default <basename>_lsf_vel<ext> naming
        entirely; output_root only overrides the destination directory. """
    if not results:
        raise ValueError("save_lsf_fits: no results to save")
    if output_path is None:
        output_path = derive_output_path(input_filename, output_root)

    primary = fits.PrimaryHDU()
    primary.header['ORIGIN'] = 'harps.lsf2'
    primary.header['SRCFILE'] = os.path.basename(input_filename)
    primary.header['DATE'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    primary.header['NORDERS'] = len(results)
    primary.header['ORDERS'] = ','.join(str(r.order) for r in results)

    hdul = fits.HDUList([primary, _build_lsf_hdu(results), _build_dispersion_hdu(results)])
    hdul.writeto(output_path, overwrite=True)
    return output_path


def load_lsf_fits(path: str) -> dict:
    """ Returns {'orders': [...], 'lsf': astropy.table.Table,
        'dispersion': astropy.table.Table, 'header': primary header}.
        See reconstruct.py for a higher-level object built on top of this. """
    from astropy.table import Table

    with fits.open(path) as hdul:
        header = hdul[0].header
        lsf_table = Table.read(hdul['LSF'])
        dispersion_table = Table.read(hdul['DISPERSION'])

    return {
        'orders': list(lsf_table['ORDER']),
        'lsf': lsf_table,
        'dispersion': dispersion_table,
        'header': header,
    }
