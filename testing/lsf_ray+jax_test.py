#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:34:25 2026

@author: dmilakov


test_single_segment.py

Run this interactively to verify the full fit+recenter loop on one segment
before committing to the full 2000-segment job.
"""
import numpy as np
import jax
import jax.numpy as jnp
import time
import logging
logging.basicConfig(level=logging.INFO)

import harps.spectrum   as hc
import harps.lsf.gp     as gp
import harps.lsf.gp_aux as gp_aux
import harps.lsf.aux    as aux
import harps.lsf.batch  as batch_utils
import harps.version    as hv
import harps.settings   as hs


def test_single_segment(filepath    : str,
                        order       : int = 50,
                        seg_index   : int = 8,
                        numseg      : int = 16,
                        num_starts  : int = 4,
                        numiter     : int = 5,
                        maxiter     : int = 300,
                        scale       : str = 'pixel',
                        ):
    """
    End-to-end test of the iterative GP fit on a single segment.

    Parameters
    ----------
    filepath  : path to a HARPS spectrum FITS file
    order     : echelle order index
    seg_index : which of the 16 segments to test (0-based)
    """
    print(f"\n{'='*60}")
    print(f"Testing single segment: order={order}, seg={seg_index}")
    print(f"{'='*60}\n")

    # -- Settings 
    version = hv.item_to_version(dict(iteration=1,
                                            model_scatter=False,
                                            interpolate=False
                                            ),
                                       ftype='lsf'
                                       )
    
    # ── Load spectrum ─────────────────────────────────────────────────────────
    print("Loading spectrum...")
    if filepath is not None:
        spec = hc.Spectrum(filepath)
    else:
        filename = '/Users/dmilakov/projects/j1333/lfc/LFC_reduced_files/'+\
            '2023-02-22_ESPRESSO_S2D_LFC_FP_A.fits'
        wavereference = '/Users/dmilakov/projects/j1333/lfc/LFC_reduced_files/'+\
            '2023-02-22_ESPRESSO_LFC_WAVE_MATRIX_A.fits'
        #filename = '/home/milakovic/lfc/data/'+\
        #         '2018-12-09/HARPS.2018-12-10T05:25:48.835_e2ds_A.fits'
        spec=hc.ESPRESSO(filename,
                         # f0=2.7e8+(26*250e6),
                         f0 = 7.40e9,
                         fr=18e9,overwrite=False,sOrder=40,
                         wavereference=wavereference)
    pix3d,vel3d,flx3d,err3d,orders_=aux.stack_spectrum(spec,
                                                       version=version,
                                                       wavesol_version=700,
                                                       orders=order,
                                                       subbkg=hs.subbkg,
                                                       divenv=hs.divenv)
    if scale=='pixel':
        x2d = pix3d[:,:,0]
    elif scale=='velocity':
        x2d = vel3d[:,:,0]
    flx2d = flx3d[:,:,0]
    err2d = err3d[:,:,0]
    
    npix   = np.shape(x2d)[1]
    minpix = 0
    maxpix = npix
    seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)

    pixl = seglims[seg_index]
    pixr = seglims[seg_index+1]
    x1s   = np.ravel(x2d  [order, pixl:pixr])
    flx1s = np.ravel(flx2d[order, pixl:pixr])
    err1s = np.ravel(err2d [order, pixl:pixr])

    print(f"Segment pixels: {pixl} -> {pixr}  ({len(x1s)} data points)")
    print(f"Flux range:     {flx1s.min():.2f} -> {flx1s.max():.2f}")

    # ── Build a minimal batch of 1 segment ───────────────────────────────────
    # We reuse make_batch but with just this one segment
    full_batch = batch_utils.make_batch(x2d, flx2d, err2d,
                                        seglims, [order])
    # Pick just the one segment we want
    idx = seg_index   # assuming no empty segments before this one
    single = batch_utils.SegmentBatch(
        x    = full_batch.x   [idx:idx+1],   # shape (1, max_len)
        flx  = full_batch.flx [idx:idx+1],
        err  = full_batch.err [idx:idx+1],
        mask = full_batch.mask[idx:idx+1],
        meta = full_batch.meta[idx:idx+1],
    )
    print(f"Batch shape: {single.x.shape}  (1 segment, padded to {single.x.shape[1]})")

    # ── Generate starts and bounds ────────────────────────────────────────────
    starts = gp.generate_starts_batch(single.x, single.flx,
                                          single.err, num_starts)
    bounds = gp.generate_bounds_batch(single.x, single.flx, single.err)

    print(f"\nStarts (leaf shapes):")
    for k, v in starts.items():
        print(f"  {k}: {v.shape}")
    print(f"\nBounds lower (leaf shapes):")
    for k, v in bounds[0].items():
        print(f"  {k}: {jnp.array(v).shape}")

    # ── Compile the fitter ────────────────────────────────────────────────────
    print("\nCompiling fit_batch (jit+vmap)...")
    t0 = time.time()
    fit_batch = gp.make_batch_fitter(numiter=numiter, maxiter=maxiter)

    # First call triggers compilation
    all_params, all_shifts, all_masks = fit_batch(
        single.x, single.flx, single.err, single.mask, starts, bounds
    )
    jax.block_until_ready(all_params)
    t_compile = time.time() - t0
    print(f"Compilation + first run: {t_compile:.2f}s")

    # ── Second call (no compilation overhead) ─────────────────────────────────
    print("\nRunning again (compiled, no overhead)...")
    t0 = time.time()
    all_params, all_shifts, all_masks = fit_batch(
        single.x, single.flx, single.err, single.mask, starts, bounds
    )
    jax.block_until_ready(all_params)
    t_run = time.time() - t0
    print(f"Pure run time: {t_run:.4f}s")

    # ── Inspect results ───────────────────────────────────────────────────────
    params = jax.tree_util.tree_map(lambda a: float(a[0]), all_params)
    shift  = float(all_shifts[0])
    n_good = int(jnp.sum(all_masks[0]))

    print(f"\n{'─'*40}")
    print(f"Results for order={order}, seg={seg_index}:")
    print(f"  Final shift:    {shift:+.6f} pix")
    print(f"  Good points:    {n_good} / {int(jnp.sum(single.mask[0]))}")
    print(f"  Fitted params:")
    for k, v in params.items():
        print(f"    {k:20s} = {v:+.4f}")

    # ── Quick sanity checks ───────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("Sanity checks:")
    amp_ok    = 0 < params['mf_amp'] < 2 * float(jnp.nanmax(single.flx[0]))
    shift_ok  = abs(shift) < 1.0
    good_ok   = n_good > 10

    print(f"  Amplitude reasonable:  {'✓' if amp_ok   else '✗'} ({params['mf_amp']:.4f})")
    print(f"  Shift < 1 pix:         {'✓' if shift_ok else '✗'} ({shift:+.4f})")
    print(f"  Enough good points:    {'✓' if good_ok  else '✗'} ({n_good})")

    all_ok = amp_ok and shift_ok and good_ok
    print(f"\n{'✓ All checks passed' if all_ok else '✗ Some checks FAILED'}")
    print(f"{'='*60}\n")

    return all_params, all_shifts, all_masks, single


if __name__ == '__main__':
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    test_single_segment(filepath, order=50, seg_index=8)