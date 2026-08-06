#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_construct_lsf_espresso.py

Validation + production driver for constructing the ESPRESSO LSF, based on
construct_lsf_ESPRESSO.py.

Two entry points are provided:

1. smoke_test()
   Cheap, fast correctness check. Runs on ONE order, a handful of segments,
   with model_scatter=False and plot=False so it exercises the core fitting
   pipeline (construct.from_spectrum_2d -> model_1d -> model_1s ->
   construct_tinygp -> gp.train_LSF_multistart) without touching the
   scatter-training path or the plotting code (lsfplot.plot_solution),
   which have NOT been re-verified as part of this fix pass.

   Run this FIRST, every time you change anything in harps.lsf, before
   committing to a full multi-order run. It should complete in well under
   a minute on CPU.

2. run_production(...)
   The actual ESPRESSO LSF construction, mirroring construct_from_args()
   in construct_lsf_ESPRESSO.py. Runs both pixel and velocity scales for
   every iteration in [start, stop], with save_fits=True, then calls
   aux.solve(...) to update the linelist.

   NOTE: aux.solve() and the plotting path (plot=True/save_plot=True) were
   NOT part of this bug-fixing pass — they may have their own issues.
   run_production() defaults to plot=False and lets you opt in.

Usage
-----
    python test_construct_lsf_espresso.py --smoke-test
    python test_construct_lsf_espresso.py --production -od 41 42 43 --start 1 --stop 1
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO logs
import warnings
warnings.simplefilter("ignore", FutureWarning)

import argparse
import logging
import sys

import matplotlib
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_construct_lsf_espresso")

# Paths — adjust if your data lives elsewhere
FILENAME = ('/Users/dmilakov/projects/j1333/lfc/LFC_reduced_files/'
           '2023-02-22_ESPRESSO_S2D_LFC_FP_A.fits')
WAVEREFERENCE = ('/Users/dmilakov/projects/j1333/lfc/LFC_reduced_files/'
                 '2023-02-22_ESPRESSO_LFC_WAVE_MATRIX_A.fits')


def _load_spectrum(overwrite=False):
    """Load the ESPRESSO spectrum object once, shared by both entry points."""
    import harps.spectrum as hc
    spec = hc.ESPRESSO(
        FILENAME,
        f0=7.40e9,
        fr=18e9,
        overwrite=overwrite,
        sOrder=40,
        wavereference=WAVEREFERENCE,
    )
    spec.process(fittype='gauss', do_comb_specific=True)
    return spec


def smoke_test(order=41, numseg=4, iter_center=3, overwrite=False):
    """
    Fast correctness check: one order, few segments, no scatter, no plots.

    Returns the fitted lsf2d array on success. Raises on any error — that's
    the point: this should surface bugs in the fitting pipeline quickly,
    on a tiny amount of data, rather than 30 minutes into a full run.
    """
    import harps.lsf.construct as construct

    logger.info(f"=== SMOKE TEST: order={order}, numseg={numseg}, "
               f"iter_center={iter_center}, model_scatter=False, plot=False ===")

    spec = _load_spectrum(overwrite=overwrite)

    lsf2d_pixel = construct.from_spectrum_2d(
        spec,
        orders=[order],
        iteration=1,
        scale='pixel',
        iter_center=iter_center,
        numseg=numseg,
        wavesol_version=700,
        model_scatter=False,     # skip scatter-GP training path for the smoke test
        interpolate=True,
        save_fits=False,         # don't touch disk / fits files for this check
        clobber=False,
        plot=False,              # skip lsfplot.plot_solution (unverified in this pass)
        save_plot=False,
        update_linelist=False,
    )

    n_ok = sum(1 for row in lsf2d_pixel if row is not None)
    logger.info(f"Smoke test complete. {len(lsf2d_pixel)} segments processed, "
               f"pixel-scale lsf2d shape={getattr(lsf2d_pixel, 'shape', None)}.")

    # Sanity checks — fail loudly if the output looks wrong rather than
    # silently returning garbage.
    assert len(lsf2d_pixel) == numseg, (
        f"Expected {numseg} segments in lsf2d, got {len(lsf2d_pixel)}"
    )
    import numpy as np
    finite_amp = np.isfinite(lsf2d_pixel['mf_amp'])
    assert finite_amp.any(), "No segment produced a finite mf_amp — fit failed everywhere."
    logger.info(f"{finite_amp.sum()}/{numseg} segments have a finite mf_amp fit.")

    logger.info("=== SMOKE TEST PASSED ===")
    return lsf2d_pixel


def run_production(orders, start=1, stop=1, numseg=16, scale='both',
                   interpolate=True, model_scatter=True,
                   wavesol_version=700, plot=False, save_plot=False,
                   overwrite=False, run_solve=True, skip_construction=False):
    """
    The actual ESPRESSO LSF construction — mirrors construct_from_args()
    in construct_lsf_ESPRESSO.py.

    Parameters
    ----------
    orders : list[int]
        Echelle orders to fit.
    start, stop : int
        Inclusive range of solve iterations.
    scale : 'pixel', 'velocity', or 'both'
    run_solve : bool
        Whether to call aux.solve(...) after each iteration (this updates
        the linelist with LSF-based fits).
    skip_construction : bool
        If True, skip from_spectrum_2d entirely for every iteration and go
        straight to aux.solve(), using whatever LSF a PRIOR run already
        wrote to lsf_filepath for that iteration/version. Before calling
        solve(), this is validated with aux.lsf_exists_for_version() —
        checking that real (non-default, finite-y) LSF rows actually exist
        for every requested order, not just that the extension is present —
        and raises a clear error immediately if not, rather than silently
        proceeding into solve()'s per-order "no valid LSF model" warnings.
    """
    import harps.lsf.construct as construct
    import harps.lsf.aux as aux
    import harps.inout as hio
    import harps.version as hv

    logger.info(f"=== PRODUCTION RUN: orders={orders}, iterations={start}..{stop}, "
               f"numseg={numseg}, scale={scale}, model_scatter={model_scatter}, "
               f"skip_construction={skip_construction} ===")

    spec = _load_spectrum(overwrite=overwrite)
    lsf_filepath = hio.get_fits_path('lsf', spec.filepath)

    scales = ['pixel', 'velocity'] if scale == 'both' else [scale]
    results = {}

    for it in range(start, stop + 1):
        if skip_construction:
            version = hv.item_to_version(
                dict(iteration=it, model_scatter=model_scatter, interpolate=interpolate),
                ftype='lsf',
            )
            logger.info(f"--- iteration {it}: --skip-construction set, reusing "
                       f"existing LSF (version={version}) ---")
            report = aux.lsf_exists_for_version(lsf_filepath, version, orders,
                                                scales=tuple(scales))
            problems = [f"  {sc}: {rep['reason']}"
                       for sc, rep in report.items() if not rep['ok']]
            if problems:
                raise RuntimeError(
                    "--skip-construction requested, but no valid, previously-"
                    f"fit LSF was found for iteration={it} (version={version}) "
                    f"covering orders={orders}. Run once WITHOUT "
                    "--skip-construction first, or check that --scatter/"
                    "--interpolate match the run that produced the existing "
                    f"LSF (version is derived from those flags).\n"
                    + "\n".join(problems)
                )
            for sc, rep in report.items():
                logger.info(f"    {sc}: {rep['n_valid']}/{rep['n_total']} segments "
                           f"have finite LSF data for the requested orders.")
        else:
            for sc in scales:
                logger.info(f"--- iteration {it}, scale={sc} ---")
                lsf2d = construct.from_spectrum_2d(
                    spec,
                    orders=orders,
                    iteration=it,
                    scale=sc,
                    iter_center=20,
                    numseg=numseg,
                    wavesol_version=wavesol_version,
                    model_scatter=model_scatter,
                    interpolate=interpolate,
                    save_fits=True,
                    clobber=False,
                    plot=plot,
                    save_plot=save_plot,
                    update_linelist=False,
                )
                results[(it, sc)] = lsf2d

        if run_solve:
            logger.info(f"--- solving iteration {it} ---")
            aux.solve(
                spec._outpath, lsf_filepath, iteration=it,
                order=orders,
                model_scatter=model_scatter,
                sOrder=spec.sOrder,
                interpolate=interpolate,
            )

    logger.info("=== PRODUCTION RUN COMPLETE ===")
    return results


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(
        prog='test_construct_lsf_espresso',
        description='Validate and/or run ESPRESSO LSF construction',
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke-test', action='store_true',
                      help='Run the fast single-order correctness check.')
    mode.add_argument('--production', action='store_true',
                      help='Run the real, full LSF construction.')

    parser.add_argument('-od', '--order', nargs='+', type=int,
                        help='Echelle order(s). Required for --production; '
                             'first value used as the order for --smoke-test.')
    parser.add_argument('-or', '--order_range', nargs=2, type=int,
                        help='Range of orders [min, max] (production only).')
    parser.add_argument('-ws', '--wavesol_version', type=int, default=700)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--stop', type=int, default=1)
    parser.add_argument('--scale', type=str, default='both',
                        choices=['pixel', 'velocity', 'both'])
    parser.add_argument('--numseg', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--interpolate', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--scatter', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--plot', type=str2bool, nargs='?',
                        const=False, default=False)
    parser.add_argument('--no-solve', action='store_true',
                        help='Skip the aux.solve(...) step in --production.')
    parser.add_argument('--skip-construction', action='store_true',
                        help='Skip from_spectrum_2d and go straight to '
                             'aux.solve(), reusing a PRIOR run\'s already-'
                             'saved LSF for the same iteration/scatter/'
                             'interpolate settings. Fails loudly if that '
                             'LSF is not actually there and valid.')
    parser.add_argument('--checksum-lsf-ref', type=str, default=None,
                        help='Path to a reference LSF FITS file. After the '
                             'run, compares the produced LSF file against '
                             'it within --checksum-rtol and reports PASS/FAIL.')
    parser.add_argument('--checksum-outpath-ref', type=str, default=None,
                        help='Path to a reference S2D output FITS file '
                             '(linelist/model_lsf). Compared the same way '
                             'as --checksum-lsf-ref.')
    parser.add_argument('--checksum-rtol', type=float, default=1e-6)
    args = parser.parse_args()

    if args.smoke_test:
        order = args.order[0] if args.order else 41
        smoke_test(order=order, numseg=min(args.numseg, 4),
                  overwrite=args.overwrite)
        return

    # --production
    orders_to_fit = list(args.order) if args.order else []
    if args.order_range is not None:
        o_min, o_max = args.order_range
        orders_to_fit.extend(range(o_min, o_max + 1))
    orders = sorted(set(orders_to_fit))
    if not orders:
        parser.error("--production requires -od/--order and/or -or/--order_range")

    run_production(
        orders=orders,
        start=args.start,
        stop=args.stop,
        numseg=args.numseg,
        scale=args.scale,
        interpolate=args.interpolate,
        model_scatter=args.scatter,
        wavesol_version=args.wavesol_version,
        plot=args.plot,
        save_plot=args.plot,
        overwrite=args.overwrite,
        run_solve=not args.no_solve,
        skip_construction=args.skip_construction,
    )

    if args.checksum_lsf_ref or args.checksum_outpath_ref:
        import harps.inout as hio
        from harps.lsf import checksum

        spec = _load_spectrum(overwrite=False)
        all_ok = True

        if args.checksum_lsf_ref:
            lsf_filepath = hio.get_fits_path('lsf', spec.filepath)
            report = checksum.compare_fits_files(
                args.checksum_lsf_ref, lsf_filepath, rtol=args.checksum_rtol
            )
            logger.info("--- checksum: LSF output vs reference ---")
            print(report.summary())
            all_ok = all_ok and report.ok

        if args.checksum_outpath_ref:
            report = checksum.compare_fits_files(
                args.checksum_outpath_ref, spec._outpath, rtol=args.checksum_rtol
            )
            logger.info("--- checksum: S2D output vs reference ---")
            print(report.summary())
            all_ok = all_ok and report.ok

        if not all_ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()