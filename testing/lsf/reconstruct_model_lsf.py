#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconstruct_model_lsf.py

Rebuilds the model_lsf array for one order OUTSIDE the full
test_construct_lsf_espresso.py --production pipeline, using exactly the
same functions solve_order_jax/_dispatch_orders_checkpointed use
(harps.lsf.fit_jax.interpolate_lsf_batch, natural_cubic_spline_M,
_lsf_model_1d) — just fed by hand from the already-fitted lsf_pix
parameters and the saved LSF templates, instead of running the optimizer
again.

Purpose: isolate whether a mismatch between this reconstruction and the
model_lsf HDU on disk is a genuine bug in the model-evaluation code
itself, vs. something specific to the write path (still-stale extension,
wrong version, import of an unpatched module, etc).

Usage
-----
    python reconstruct_model_lsf.py \\
        --outpath /path/to/2023-02-22_ESPRESSO_S2D_LFC_FP_A.fits \\
        --lsf-filepath /path/to/2023-02-22_ESPRESSO_S2D_LFC_FP_A_lsf.fits \\
        --order 50 --version 111
"""
import argparse
import numpy as np
from fitsio import FITS
import harps.lsf.fit_jax as hfjax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reconstruct_order(outpath, lsf_filepath, order, version, scale='pixel'):
    scl = 'pix' if scale[:3] == 'pix' else 'wav'
    lsf_extname = 'pixel_model' if scale[:3] == 'pix' else 'velocity_model'
    model_extname = 'model_lsf' if scale[:3] == 'pix' else 'model_lsf_vel'

    # --- 1. Read the already-fitted linelist rows for this order ---
    with FITS(outpath, 'r') as hdu:
        linelist = hdu['linelist', version].read()
        flux = hdu['flux'].read()[order]
        bkg = hdu['background'].read()[order]
        try:
            model_on_disk = hdu[model_extname, version].read()[order]
        except Exception as e:
            model_on_disk = None
            print(f"Could not read '{model_extname}' version {version} "
                 f"for order {order}: {e}")

        # Diagnostic: does the scale-specific 'model_lsf_vel' extension
        # exist at all? If this is a 'pixel' scale run and 'model_lsf_vel'
        # is MISSING entirely, that's strong evidence the installed aux.py
        # predates the scale-specific-extension fix (the old code only
        # ever knew about a single shared 'model_lsf' extension, so it
        # would never have created 'model_lsf_vel'), i.e. the running code
        # is not the patched version regardless of what's on disk in your
        # checkout.
        other_extname = 'model_lsf_vel' if scale[:3] == 'pix' else 'model_lsf'
        try:
            other_exists = hdu[other_extname, version].has_data()
        except Exception:
            other_exists = False
        print(f"Diagnostic: extension '{other_extname}' version {version} "
             f"{'exists' if other_exists else 'DOES NOT EXIST'} in {outpath}.")
        if not other_exists:
            print(f"  -> If you've applied the scale-specific-extension fix, "
                 f"'{other_extname}' should exist after any run that fits "
                 f"the '{'velocity' if scale[:3]=='pix' else 'pixel'}' scale "
                 f"too (the default). Its absence suggests the aux.py "
                 f"actually being imported is not the patched one — worth "
                 f"checking harps.lsf.aux.__file__ and "
                 f"inspect.getsource(harps.lsf.aux._dispatch_orders_checkpointed) "
                 f"for 'model_extname'.")

    rows = linelist[linelist['order'] == order]
    n_lines = len(rows)
    if n_lines == 0:
        raise ValueError(f"No linelist rows found for order {order}.")
    print(f"order {order}: {n_lines} lines in linelist")

    # --- 2. Read the LSF templates for this order (same file/version the
    #        fit itself used) ---
    with FITS(lsf_filepath, 'r') as hdu:
        d = hdu[lsf_extname, version].read()
    sub = d[d['order'] == order]
    if len(sub) == 0:
        raise ValueError(f"No {lsf_extname} segments found for order {order} "
                         f"version {version} in {lsf_filepath}.")
    x_grid = jnp.array(sub['x'][0], dtype=jnp.float64)
    segment_Y = jnp.array(np.stack([sub['y'][i] for i in range(len(sub))]),
                          dtype=jnp.float64)
    segment_centres = jnp.array(
        (sub['ledge'].astype(float) + sub['redge'].astype(float)) / 2.0
    )

    # --- 3. Rebuild Y_line/M_line exactly as solve_order_jax does: segment
    #        interpolation keyed on 'bary' (NOT the fitted 'cen') ---
    bary = jnp.array(rows['bary'].astype(float))
    Y_line = hfjax.interpolate_lsf_batch(bary, segment_centres, segment_Y, N=2)
    M_line = hfjax.natural_cubic_spline_M(x_grid, Y_line)

    # --- 4. Evaluate the model for each line at its own pixel window,
    #        using the ALREADY-FITTED amp/cen/wid (lsf_pix or lsf_wav) —
    #        no optimizer involved, just the forward model. ---
    reconstructed = np.zeros_like(flux, dtype=np.float64)
    per_line_report = []
    for k, row in enumerate(rows):
        pixl, pixr = int(row['pixl']), int(row['pixr'])
        amp, cen, wid = row[f'lsf_{scl}'][:3]
        x1l = jnp.arange(pixl, pixr, dtype=jnp.float64)
        model_1l, x_test = hfjax._lsf_model_1d(
            x_grid, Y_line[k], M_line[k], x1l, amp, cen, wid, scale[:3] == 'pix'
        )
        model_1l = np.asarray(model_1l)
        reconstructed[pixl:pixr] = model_1l
        data_1l = flux[pixl:pixr] - bkg[pixl:pixr]
        peak_data = np.max(np.abs(data_1l))
        peak_model = np.max(np.abs(model_1l))
        per_line_report.append((k, pixl, pixr, amp, cen, wid,
                               peak_data, peak_model))

    # --- 5. Compare against what's actually on disk ---
    print("\nFirst 5 lines: reconstructed model vs data vs what's on disk")
    print(f"{'idx':>4} {'pixl':>6} {'amp':>12} {'cen':>10} {'wid':>7} "
         f"{'peak_data':>12} {'peak_recon':>12}")
    for k, pixl, pixr, amp, cen, wid, peak_data, peak_model in per_line_report[:5]:
        print(f"{k:>4} {pixl:>6} {amp:>12.2f} {cen:>10.2f} {wid:>7.4f} "
             f"{peak_data:>12.2f} {peak_model:>12.2f}")

    if model_on_disk is not None:
        diffs = []
        for k, pixl, pixr, *_ in per_line_report:
            seg_recon = reconstructed[pixl:pixr]
            seg_disk = model_on_disk[pixl:pixr]
            if np.allclose(seg_recon, seg_disk, rtol=1e-3, atol=1.0):
                diffs.append(False)
            else:
                diffs.append(True)
        n_mismatch = sum(diffs)
        print(f"\n{n_mismatch}/{n_lines} lines: reconstructed model "
             f"DISAGREES with '{model_extname}' on disk (order {order}, "
             f"version {version})")
        if n_mismatch > 0:
            k0 = diffs.index(True)
            pixl, pixr = per_line_report[k0][1], per_line_report[k0][2]
            print(f"\nFirst mismatching line (idx={k0}, pix {pixl}:{pixr}):")
            print("  reconstructed:", reconstructed[pixl:pixr])
            print("  on disk      :", model_on_disk[pixl:pixr])
            print("  data (flx-bkg):", flux[pixl:pixr] - bkg[pixl:pixr])

    return reconstructed, model_on_disk, flux - bkg, rows


def diagnose_residuals(reconstructed, model_on_disk, rows, order, scale, outdir='.'):
    """
    Quantifies and visualizes the residual between the reconstructed model
    and what's on disk, once they're known to be close (rather than the
    order-of-magnitude-wrong regime the earlier bugs produced).

    Three questions this answers:
      1. How big is the residual, in absolute and relative terms?
      2. Does the relative error look like float32 storage precision
         (model_lsf is saved as float32; the reconstruction here runs in
         float64) — i.e. is it flat at ~1e-7 relative, unstructured?
      3. Or does it have structure — e.g. growing with distance from line
         centre (would point at the spline-domain clamp), or with amp/wid
         (would point at something fit-related) — which would mean it's
         NOT just precision and is worth chasing further.
    """
    scl = 'pix' if scale[:3] == 'pix' else 'wav'
    n_lines = len(rows)

    all_resid = []       # reconstructed - disk, per pixel, all lines concatenated
    all_disk = []        # disk value at that pixel (for relative error)
    all_xtest = []        # (pixel - cen) / wid, i.e. offset from line centre in template units
    per_line_max_abs = np.zeros(n_lines)
    per_line_max_rel = np.zeros(n_lines)

    for k, row in enumerate(rows):
        pixl, pixr = int(row['pixl']), int(row['pixr'])
        amp, cen, wid = row[f'lsf_{scl}'][:3]
        px = np.arange(pixl, pixr)
        resid = reconstructed[pixl:pixr] - model_on_disk[pixl:pixr]
        disk_seg = model_on_disk[pixl:pixr]
        xtest = (px - cen) / wid

        all_resid.append(resid)
        all_disk.append(disk_seg)
        all_xtest.append(xtest)

        per_line_max_abs[k] = np.max(np.abs(resid))
        # relative error only where the model itself isn't near zero,
        # otherwise "relative" is meaningless (dividing by ~0)
        scale_ref = np.maximum(np.abs(disk_seg), 1e-6 * np.max(np.abs(disk_seg)) if np.max(np.abs(disk_seg)) > 0 else 1.0)
        per_line_max_rel[k] = np.max(np.abs(resid) / scale_ref)

    all_resid = np.concatenate(all_resid)
    all_disk = np.concatenate(all_disk)
    all_xtest = np.concatenate(all_xtest)

    # Relative error, computed only where |disk value| is a non-negligible
    # fraction of that LINE's own peak (avoids blowing up near zero-crossings,
    # which is a division artifact, not a real precision problem).
    nonzero = np.abs(all_disk) > 1e-3 * np.max(np.abs(all_disk))
    rel_err = np.abs(all_resid[nonzero]) / np.abs(all_disk[nonzero])

    print("\n=== Residual diagnosis: reconstructed vs on-disk model_"
         f"{scl} (order {order}) ===")
    print(f"max |residual| overall: {np.max(np.abs(all_resid)):.6g}")
    print(f"median |residual| overall: {np.median(np.abs(all_resid)):.6g}")
    print(f"relative error (|resid|/|disk|, where |disk| > 0.1% of that "
         f"line's peak):")
    print(f"  median: {np.median(rel_err):.3e}   90th pct: {np.percentile(rel_err, 90):.3e}"
         f"   max: {np.max(rel_err):.3e}")
    print(f"  (float32 machine epsilon is ~1.19e-07; relative error "
         f"clustering within an order of magnitude or two of that, with "
         f"no dependence on position/amp/wid below, would point at "
         f"float32-storage rounding as the explanation rather than a bug)")

    worst_k = np.argmax(per_line_max_abs)
    print(f"\nWorst single line by max abs residual: idx={worst_k}, "
         f"pix [{int(rows[worst_k]['pixl'])}:{int(rows[worst_k]['pixr'])}], "
         f"max|resid|={per_line_max_abs[worst_k]:.6g}, "
         f"max rel err={per_line_max_rel[worst_k]:.3e}")

    # --- Plot: does the residual have structure? ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.scatter(all_xtest, all_resid, s=2, alpha=0.25, color='tab:blue')
    ax.set_xlabel('(pixel - cen) / wid   [offset from line centre, template units]')
    ax.set_ylabel('reconstructed - disk')
    ax.set_title('Residual vs. offset from line centre\n'
                 '(flat scatter around 0 = no positional structure)')
    ax.axhline(0, color='k', linewidth=0.5)

    ax = axes[0, 1]
    ax.scatter(all_disk[nonzero], all_resid[nonzero], s=2, alpha=0.25, color='tab:green')
    ax.set_xlabel('on-disk model value')
    ax.set_ylabel('reconstructed - disk')
    ax.set_title('Residual vs. local model amplitude\n'
                 '(linear envelope = proportional/precision error)')
    ax.axhline(0, color='k', linewidth=0.5)

    ax = axes[1, 0]
    ax.hist(rel_err, bins=60, color='tab:purple')
    ax.axvline(1.19e-7, color='k', linestyle='--', linewidth=1,
              label='float32 eps (1.19e-7)')
    ax.set_xlabel('relative error |resid| / |disk value|')
    ax.set_ylabel('count (pixels)')
    ax.set_title('Relative error distribution')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    wid_arr = rows[f'lsf_{scl}'][:, 2]
    amp_arr = rows[f'lsf_{scl}'][:, 0]
    sc = ax.scatter(amp_arr, per_line_max_rel, s=10, c=wid_arr, cmap='viridis')
    ax.set_xlabel('fitted amp')
    ax.set_ylabel('max relative error (per line)')
    ax.set_title('Per-line worst relative error vs. amp\n(colour = wid)')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('wid')

    fig.suptitle(f'Order {order}: residual diagnosis (reconstructed vs on-disk model_{scl})')
    fig.tight_layout()
    path = f'{outdir}/reconstruct_model_lsf_order{order}_{scale}_residual_diagnosis.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def make_plots(reconstructed, model_on_disk, data, rows, order, scale,
               outdir='.', n_sample_lines=6):
    """
    Two figures:
      1. per-line panels — for a handful of lines, data / reconstructed
         model (shared, sane y-scale) on top, model_on_disk (its own
         y-scale, since it may be ~1e12 while the others are ~1e5) below.
      2. full-order overview — same idea, across the whole order.
    Saved as PNGs; nothing is shown interactively (Agg backend).
    """
    scl = 'pix' if scale[:3] == 'pix' else 'wav'
    n_lines = len(rows)
    sample_idx = np.linspace(0, n_lines - 1, min(n_sample_lines, n_lines),
                             dtype=int)

    # --- Figure 1: per-line panels ---
    fig, axes = plt.subplots(len(sample_idx), 2, figsize=(11, 2.6 * len(sample_idx)),
                             squeeze=False)
    for row_i, k in enumerate(sample_idx):
        pixl, pixr = int(rows[k]['pixl']), int(rows[k]['pixr'])
        px = np.arange(pixl, pixr)

        ax_left = axes[row_i, 0]
        ax_left.step(px, data[pixl:pixr], where='mid', label='data (flx-bkg)',
                    color='black')
        ax_left.step(px, reconstructed[pixl:pixr], where='mid',
                    label='reconstructed model', color='tab:orange', linestyle='--')
        ax_left.set_ylabel('flux')
        ax_left.set_title(f'line idx={k}  pix [{pixl}:{pixr}]  '
                         f'(data vs reconstructed)')
        ax_left.legend(fontsize=8, loc='upper right')

        ax_right = axes[row_i, 1]
        if model_on_disk is not None:
            ax_right.step(px, model_on_disk[pixl:pixr], where='mid',
                         color='tab:red', label=f"'model_{scl}' on disk")
            ax_right.set_title(f'line idx={k}  on-disk model (own scale)')
            ax_right.legend(fontsize=8, loc='upper right')
        else:
            ax_right.text(0.5, 0.5, 'model on disk not available',
                         ha='center', va='center', transform=ax_right.transAxes)
        ax_right.set_xlabel('pixel')
        ax_left.set_xlabel('pixel')

    fig.suptitle(f'Order {order}, scale={scale}: per-line comparison '
                f'({len(sample_idx)} of {n_lines} lines)')
    fig.tight_layout()
    path1 = f'{outdir}/reconstruct_model_lsf_order{order}_{scale}_per_line.png'
    fig.savefig(path1, dpi=130)
    plt.close(fig)
    print(f"Saved: {path1}")

    # --- Figure 2: full-order overview, two stacked panels sharing x ---
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    px_full = np.arange(len(data))
    ax1.plot(px_full, data, drawstyle='steps-mid', color='black',
            label='data (flx-bkg)', linewidth=0.7)
    ax1.plot(px_full, reconstructed, drawstyle='steps-mid', color='tab:orange',
            label='reconstructed model', linewidth=0.7, linestyle='--')
    ax1.set_ylabel('flux')
    ax1.set_title(f'Order {order}: data vs reconstructed model (sane scale)')
    ax1.legend(fontsize=9)
    # Robust y-limits from the data itself: a handful of badly-fit lines
    # (if any — see the per-line figure) can otherwise blow up the axis
    # and flatten the comparison for every well-behaved line.
    lo, hi = np.percentile(data, [0.5, 99.5])
    pad = 0.15 * (hi - lo)
    ax1.set_ylim(lo - pad, hi + pad)

    if model_on_disk is not None:
        ax2.plot(px_full, model_on_disk, drawstyle='steps-mid', color='tab:red',
                label=f"'model_{scl}' on disk", linewidth=0.7)
        ax2.set_title(f"Order {order}: '{'model_lsf' if scale[:3]=='pix' else 'model_lsf_vel'}' "
                     f"on disk (its own scale — note the y-axis)")
        ax2.legend(fontsize=9)
        lo2, hi2 = np.percentile(model_on_disk, [0.5, 99.5])
        pad2 = 0.15 * (hi2 - lo2) if hi2 > lo2 else 1.0
        ax2.set_ylim(lo2 - pad2, hi2 + pad2)
    else:
        ax2.text(0.5, 0.5, 'model on disk not available',
                ha='center', va='center', transform=ax2.transAxes)
    ax2.set_xlabel('pixel')
    fig2.tight_layout()
    path2 = f'{outdir}/reconstruct_model_lsf_order{order}_{scale}_overview.png'
    fig2.savefig(path2, dpi=130)
    plt.close(fig2)
    print(f"Saved: {path2}")

    return path1, path2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--outpath', required=True)
    p.add_argument('--lsf-filepath', required=True)
    p.add_argument('--order', type=int, required=True)
    p.add_argument('--version', type=int, required=True)
    p.add_argument('--scale', default='pixel', choices=['pixel', 'velocity'])
    p.add_argument('--outdir', default='.',
                   help='Directory to save comparison plots into.')
    p.add_argument('--n-sample-lines', type=int, default=6,
                   help='Number of lines to show in the per-line panel figure.')
    args = p.parse_args()

    reconstructed, model_on_disk, data, rows = reconstruct_order(
        args.outpath, args.lsf_filepath, args.order, args.version, args.scale
    )
    make_plots(reconstructed, model_on_disk, data, rows, args.order, args.scale,
              outdir=args.outdir, n_sample_lines=args.n_sample_lines)
    if model_on_disk is not None:
        diagnose_residuals(reconstructed, model_on_disk, rows, args.order,
                          args.scale, outdir=args.outdir)