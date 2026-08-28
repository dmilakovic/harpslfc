"""
harps.lsf2: LSF and wavelength-solution reconstruction from LFC exposures,
in velocity space. Replaces harps.lsf.

Typical usage
-------------
Run the pipeline for one or more orders and save a FITS file::

    from harps.lsf2 import LSFConfig, load_order_from_spectrum, run_order, save_lsf_fits

    data = load_order_from_spectrum(filename, wavereference, order=85)
    result = run_order(data, cfg=LSFConfig())
    save_lsf_fits([result], filename)

or from the command line::

    python -m harps.lsf2.cli_run FILE.fits --orders 80-95 --wavereference WAVE.fits

Read a saved FITS file back and reconstruct an LSF at an arbitrary pixel
or wavelength::

    from harps.lsf2 import LSFLibrary

    lib = LSFLibrary('FILE_lsf_vel.fits')
    u, phi = lib.lsf_at_pixel(order=85, x=4500.0)
    u, phi = lib.lsf_at_wavelength(order=85, wavelength=550.2)
    u, phi, orders_used = lib.composite_lsf_at_wavelength(550.2)

or from the command line::

    python -m harps.lsf2.cli_reconstruct FILE_lsf_vel.fits --wavelength 550.2 --plot

An interactive GUI (needs PyQt5) is also available::

    python -m harps.lsf2.gui FILE_lsf_vel.fits
"""
from .config import LSFConfig
from .data import OrderData, load_order_from_spectrum, load_order_from_text
from .pipeline import run_order
from .result import LSFOrderResult
from .fits_io import save_lsf_fits, load_lsf_fits, derive_output_path
from .reconstruct import LSFLibrary
from .orders_spec import parse_orders

__all__ = [
    'LSFConfig', 'OrderData', 'load_order_from_spectrum', 'load_order_from_text',
    'run_order', 'LSFOrderResult',
    'save_lsf_fits', 'load_lsf_fits', 'derive_output_path',
    'LSFLibrary', 'parse_orders',
]
