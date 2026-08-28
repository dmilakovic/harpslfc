#!/usr/bin/env python
"""
Command-line entry point for the lsf2 pipeline.

Usage:
    python -m harps.lsf2.cli_run FILENAME --orders 85 --wavereference WAVE.fits
    python -m harps.lsf2.cli_run FILENAME --orders 80-95 --wavereference WAVE.fits
    python -m harps.lsf2.cli_run FILENAME --orders 80,85,90-95 --wavereference WAVE.fits

Saves <output-dir>/<basename>_lsf_vel<ext> (default output-dir:
<dirname(FILENAME)>/lsf_vel), containing the LSF + DISPERSION extensions
for every order that finished successfully.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import LSFConfig
from .data import load_order_from_spectrum
from .fits_io import save_lsf_fits
from .orders_spec import parse_orders
from .pipeline import run_order

log = logging.getLogger('harps.lsf2.cli_run')


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct the ESPRESSO LSF (in velocity space) from an LFC exposure.")
    p.add_argument('filename', help="ESPRESSO S2D LFC exposure (FITS file)")
    p.add_argument('--orders', required=True,
                    help="Orders to analyse: a single order ('85'), a list ('80,85,90'), "
                         "a range ('80-95'), or any comma-separated mixture of these.")
    p.add_argument('--wavereference', required=True,
                    help="Wavelength-reference matrix FITS file for this exposure.")
    p.add_argument('--f0', type=float, default=7.40e9, help="LFC offset frequency, Hz (default: 7.40e9)")
    p.add_argument('--fr', type=float, default=18e9, help="LFC repetition frequency, Hz (default: 18e9)")
    p.add_argument('--sorder', type=int, default=40, help="First usable spectral order (default: 40)")
    p.add_argument('--overwrite', action='store_true', help="Re-process even if cached products exist")
    p.add_argument('-o', '--output', default=None, help="Explicit output FITS path (overrides naming)")
    p.add_argument('--output-dir', default=None,
                    help="Output directory (default: <dirname(filename)>/lsf_vel)")
    p.add_argument('--n-outer-iterations', type=int, default=None,
                    help="Override LSFConfig.n_outer_iterations")
    p.add_argument('--skip-failed', action='store_true',
                    help="On a per-order failure, log and continue instead of aborting the whole run")
    p.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s %(levelname)s %(message)s')

    orders = parse_orders(args.orders)
    log.info("Orders to analyse: %s", orders)

    cfg = LSFConfig()
    if args.n_outer_iterations is not None:
        cfg.n_outer_iterations = args.n_outer_iterations

    results = []
    for order in orders:
        t0 = time.time()
        log.info("=== order %d: loading data ===", order)
        try:
            data = load_order_from_spectrum(
                args.filename, args.wavereference, order,
                f0=args.f0, fr=args.fr, sOrder=args.sorder, overwrite=args.overwrite,
            )
            log.info("order %d: %d pixels, %d LFC lines", order, data.n_pixels, data.n_lines)
            result = run_order(data, cfg=cfg, verbose=True)
        except Exception:
            log.exception("order %d: failed", order)
            if args.skip_failed:
                continue
            raise
        log.info("order %d: done in %.1fs (converged=%s, chi2/dof=%.3f)",
                  order, time.time() - t0, result.converged, result.chi2_per_dof)
        results.append(result)

    if not results:
        log.error("No orders completed successfully -- nothing saved.")
        return 1

    out_path = save_lsf_fits(results, args.filename, output_path=args.output, output_root=args.output_dir)
    log.info("Saved %d order(s) to %s", len(results), out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
