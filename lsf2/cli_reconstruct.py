#!/usr/bin/env python
"""
Command-line entry point for reconstructing an LSF from an lsf2 output
FITS file, at an arbitrary pixel or wavelength.

Usage:
    # at a specific pixel within one order
    python -m harps.lsf2.cli_reconstruct FILE.fits --order 85 --pixel 4500

    # at a specific wavelength -- if only one order covers it, that
    # order's LSF is returned; if several orders overlap there, a
    # composite (equal-weighted by default) is built automatically
    python -m harps.lsf2.cli_reconstruct FILE.fits --wavelength 550.123

    # explicit orders/weights for the composite
    python -m harps.lsf2.cli_reconstruct FILE.fits --wavelength 550.123 \\
        --orders 84,85 --weights 0.3,0.7

    # save the result and/or plot it
    python -m harps.lsf2.cli_reconstruct FILE.fits --wavelength 550.123 \\
        --output lsf_550.123nm.txt --plot
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .orders_spec import parse_orders
from .reconstruct import LSFLibrary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reconstruct an LSF profile from an lsf2 output FITS file.")
    p.add_argument('fitsfile', help="FITS file written by harps.lsf2.cli_run")
    loc = p.add_mutually_exclusive_group(required=True)
    loc.add_argument('--pixel', type=float, help="Reconstruct at this pixel (requires --order)")
    loc.add_argument('--wavelength', type=float, help="Reconstruct at this wavelength (nm)")
    p.add_argument('--order', type=int, default=None, help="Order to use with --pixel")
    p.add_argument('--orders', default=None,
                    help="With --wavelength: restrict the composite to these orders "
                         "(single/list/range, e.g. '84,85' or '80-90'). Default: every "
                         "order in the file that covers this wavelength.")
    p.add_argument('--weights', default=None,
                    help="Comma-separated weights, one per --orders entry (or, with no "
                         "--orders, one per auto-detected covering order in ascending "
                         "order number). Default: equal weight.")
    p.add_argument('--output', '-o', default=None, help="Save the (u, phi) profile to this text file")
    p.add_argument('--plot', action='store_true', help="Show a matplotlib plot of the result")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    lib = LSFLibrary(args.fitsfile)

    if args.pixel is not None:
        if args.order is None:
            print("--pixel requires --order", file=sys.stderr)
            return 2
        u, phi = lib.lsf_at_pixel(args.order, args.pixel)
        orders_used = [args.order]
        label = f"order {args.order}, pixel {args.pixel:.1f}"
    else:
        orders = parse_orders(args.orders) if args.orders else None
        weights = None
        if args.weights is not None:
            w = [float(v) for v in args.weights.split(',')]
            if orders is not None:
                if len(w) != len(orders):
                    print("--weights must have the same length as --orders", file=sys.stderr)
                    return 2
                weights = dict(zip(orders, w))
            else:
                weights = w   # zipped against auto-detected orders inside composite_lsf_at_wavelength
        u, phi, orders_used = lib.composite_lsf_at_wavelength(
            args.wavelength, orders=orders, weights=weights)
        label = f"wavelength {args.wavelength:.4f} nm (orders {orders_used})"

    print(f"Reconstructed LSF at {label}")
    print(f"  n_points = {len(u)}, u range = [{u.min():.3f}, {u.max():.3f}] km/s")
    fwhm_est = _fwhm_from_profile(u, phi)
    if fwhm_est is not None:
        print(f"  approx FWHM = {fwhm_est:.4f} km/s")

    if args.output:
        np.savetxt(args.output, np.transpose([u, phi]), header='u_kms  phi')
        print(f"  saved to {args.output}")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.plot(u, phi, '-')
        plt.axhline(0, color='gray', lw=0.5)
        plt.xlabel('u [km/s]')
        plt.ylabel('phi(u)')
        plt.title(f"LSF: {label}")
        plt.tight_layout()
        plt.show()

    return 0


def _fwhm_from_profile(u, phi):
    half = phi.max() / 2.0
    above = phi >= half
    if not above.any():
        return None
    idx = np.where(above)[0]
    return float(u[idx[-1]] - u[idx[0]])


if __name__ == '__main__':
    sys.exit(main())
