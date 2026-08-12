#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_lsf_order_diagnostics.py

For a single echelle order, plots each segment's:
  1. Smooth LSF profile (from the '{scale}_model' FITS extension) overlaid
     with the actual data points used to build it (from '{scale}_gp'),
     distinguishing genuinely-kept points from ones excluded by
     outlier-rejection.
  2. Normalised residuals between the LSF model and that data, computed
     with harps.lsf.gp's own get_residuals / rescale_errors — the same
     procedures construct_tinygp itself uses — not reimplemented here.

One figure per segment (simplest layout for an arbitrary segment count;
see --panels to instead get one multi-panel figure per order).

Requires an environment with `harps` importable (this reuses
harps.lsf.gp.get_residuals/rescale_errors and harps.lsf.gp_aux.parnames_*
directly, per instructions -- it does not reimplement them).

Usage
-----
    python plot_lsf_order_diagnostics.py \\
        --lsf-filepath /path/to/..._lsf.fits \\
        --version 111 --scale pixel \\
        --order 45 --outdir .
"""
import argparse
import numpy as np
import jax.numpy as jnp
from fitsio import FITS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import harps.lsf.gp as gp
import harps.lsf.gp_aux as gp_aux

# Reuse of plot.py's plot_numerical_model (see that module for the fix
# applied to it: the original had a gap where 2-5 segments rendered
# nothing at all). Imported directly rather than copied, per "reuse it"
# -- this only works if harps.lsf.plot itself imports cleanly in your
# environment; if it doesn't (e.g. missing hplot), fall back to the
# local copy in plot_all_lsf_profiles.py instead.
try:
    from harps.lsf.plot import plot_numerical_model
except Exception as e:
    print(f"Could not import harps.lsf.plot.plot_numerical_model ({e}); "
         f"falling back to a local copy.")
    def plot_numerical_model(ax, nummodel, *args, **kwargs):
        x = nummodel['x']; y = nummodel['y']
        numseg_sent, npts = np.shape(x)
        user_label = kwargs.pop('label', None)
        colors = plt.cm.jet(np.linspace(0, 1, max(numseg_sent, 2)))
        for i, (x_, y_) in enumerate(zip(x, y)):
            label = user_label if numseg_sent == 1 else f'Segment {i+1}'
            ax.plot(x_, y_, *args, color=colors[i], label=label, **kwargs)
        return ax


EXCLUDED_ERR_THRESHOLD = 1e8   # matches construct_tinygp's Y_err_fit=1e9
                                # inflation for outlier-rejected points


def build_theta(row, sct=False):
    theta = {p: float(row[p]) for p in gp_aux.parnames_lfc}
    if sct:
        theta_sct = {p: float(row[p]) for p in gp_aux.parnames_sct}
        return theta, theta_sct
    return theta, None


def segment_data(row, has_scatter_fields):
    """
    Recovers, from one row of the '{scale}_gp' extension:
      X, Y  : this segment's true-length data (padding-to-fixed-length
              slots, where data_yerr is exactly 0, dropped)
      kept  : boolean mask, True = genuinely used in the fit,
              False = excluded by outlier-rejection (data_yerr==1e9)
      Y_err : the RAW (possibly outlier-inflated) error, as stored

    No separate mask field is stored on disk (containers.lsf's dtype
    doesn't define one) -- it's fully recoverable from data_yerr alone,
    exactly as construct_tinygp writes it.
    """
    x_full = np.asarray(row['data_x'], dtype=float)
    y_full = np.asarray(row['data_y'], dtype=float)
    e_full = np.asarray(row['data_yerr'], dtype=float)
    valid_slot = e_full > 0   # drop unused padding-to-fixed-length slots
    X = x_full[valid_slot]
    Y = y_full[valid_slot]
    Yerr = e_full[valid_slot]
    kept = Yerr < EXCLUDED_ERR_THRESHOLD
    return X, Y, Yerr, kept


def residuals_for_segment(row, has_scatter_fields):
    theta, theta_sct = build_theta(row, sct=has_scatter_fields)
    X, Y, Yerr, kept = segment_data(row, has_scatter_fields)

    scatter_is_active = False
    if has_scatter_fields:
        sct_x = jnp.asarray(row['sct_x'], dtype=float)
        sct_y = jnp.asarray(row['sct_y'], dtype=float)
        sct_yerr = jnp.asarray(row['sct_yerr'], dtype=float)
        # Require genuinely finite, nonzero scatter data before attempting
        # rescaling. `sct_x != 0` alone is NOT a safe "is this populated"
        # check: NaN != 0 evaluates to True for every element, so a
        # segment whose scatter fields are entirely NaN (seen in practice
        # in this file -- construct_tinygp's own scatter-GP training
        # evidently produced NaN for it at some point) would otherwise
        # look "active", and NaN silently propagates through
        # rescale_errors/get_residuals with no error at all -- an empty-
        # looking residual panel, not a crash.
        all_finite = bool(jnp.all(jnp.isfinite(sct_x))
                          and jnp.all(jnp.isfinite(sct_y))
                          and jnp.all(jnp.isfinite(sct_yerr)))
        scatter_is_active = all_finite and bool(jnp.any(sct_x != 0) or jnp.any(sct_y != 0))
        if has_scatter_fields and not all_finite:
            print(f"  segm {int(row['segm'])}: stored scatter data contains "
                 f"NaN/Inf; treating as no scatter model for this segment "
                 f"(raw data_yerr residuals).")

    X_j = jnp.asarray(X, dtype=float)
    Yerr_j = jnp.asarray(Yerr, dtype=float)

    if scatter_is_active:
        scatter = [theta_sct, sct_x, sct_y, sct_yerr]
        try:
            S, S_var = gp.rescale_errors(True, scatter, X_j, Yerr_j)
            Yerr_for_rsd = np.asarray(S)
        except Exception as e:
            # Defensive fallback for genuinely unexpected failures (this
            # is not working around a known bug -- gp.build_scatter_GP's
            # scalar-predicate issue, hit here previously, is fixed).
            print(f"  order/segm scatter rescaling raised unexpectedly "
                 f"({e!r}); falling back to raw data_yerr for residuals.")
            scatter = None
            Yerr_for_rsd = Yerr
    else:
        scatter = None
        Yerr_for_rsd = Yerr

    rsd = np.asarray(gp.get_residuals(X_j, jnp.asarray(Y, dtype=float),
                                      jnp.asarray(Yerr_for_rsd, dtype=float),
                                      theta, scatter=scatter))
    return X, Y, Yerr, kept, rsd


def plot_segment(model_row, gp_row, has_scatter_fields, order, scale, outdir):
    segm = int(gp_row['segm'])
    X, Y, Yerr, kept, rsd = residuals_for_segment(gp_row, has_scatter_fields)

    fig, (ax_top, ax_rsd) = plt.subplots(
        2, 1, figsize=(7, 6.5), sharex=True,
        gridspec_kw=dict(height_ratios=[2.5, 1]),
        layout='constrained',
    )

    # --- top panel: smooth LSF model (reused plot_numerical_model) + data ---
    nummodel = {'x': model_row['x'][None, :], 'y': model_row['y'][None, :]}
    plot_numerical_model(ax_top, nummodel, lw=2, zorder=5)
    ax_top.get_lines()[-1].set_label('LSF model')

    ax_top.errorbar(X[kept], Y[kept], Yerr[kept], marker='.', ms=5, ls='',
                    color='k', capsize=2, label='data (used)', zorder=3)
    if np.any(~kept):
        ax_top.errorbar(X[~kept], Y[~kept], None, marker='x', ms=5, ls='',
                        color='0.6', label='data (outlier-rejected)', zorder=2)
    ax_top.axhline(0, ls=':', c='grey', lw=0.8, zorder=0)
    ax_top.set_ylabel('flux (normalised)')
    ax_top.set_title(f'Order {order}, segment {segm} ({scale})')
    ax_top.legend(fontsize=8, loc='upper right')

    # --- bottom panel: normalised residuals ---
    ax_rsd.axhline(0, ls=':', c='grey', lw=0.8)
    for lvl, style in [(1, '--'), (3, ':')]:
        ax_rsd.axhline(lvl, ls=style, c='0.7', lw=0.8)
        ax_rsd.axhline(-lvl, ls=style, c='0.7', lw=0.8)
    ax_rsd.plot(X[kept], rsd[kept], '.', ms=5, color='k', zorder=3)
    if np.any(~kept):
        ax_rsd.plot(X[~kept], rsd[~kept], 'x', ms=5, color='0.6', zorder=2)
    ax_rsd.set_xlabel('x [pixel]' if scale[:3] == 'pix' else 'x [km/s]')
    ax_rsd.set_ylabel('normalised\nresidual')
    finite_rsd = rsd[np.isfinite(rsd)]
    if finite_rsd.size:
        ylim = max(4.0, np.percentile(np.abs(finite_rsd), 99) * 1.2)
        ax_rsd.set_ylim(-ylim, ylim)

    path = f'{outdir}/lsf_diag_order{order}_segm{segm:02d}_{scale}.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_segment_panel(model_row, gp_row, has_scatter_fields, ax_top, ax_rsd,
                       order, scale):
    """Same content as plot_segment, drawn into existing axes (for --panels mode)."""
    segm = int(gp_row['segm'])
    X, Y, Yerr, kept, rsd = residuals_for_segment(gp_row, has_scatter_fields)

    nummodel = {'x': model_row['x'][None, :], 'y': model_row['y'][None, :]}
    plot_numerical_model(ax_top, nummodel, lw=1.5, zorder=5)
    ax_top.errorbar(X[kept], Y[kept], Yerr[kept], marker='.', ms=3, ls='',
                    color='k', capsize=1.5, zorder=3)
    if np.any(~kept):
        ax_top.errorbar(X[~kept], Y[~kept], None, marker='x', ms=4, ls='',
                        color='0.6', zorder=2)
    ax_top.axhline(0, ls=':', c='grey', lw=0.6, zorder=0)
    ax_top.set_title(f'segm {segm}', fontsize=9)

    ax_rsd.axhline(0, ls=':', c='grey', lw=0.6)
    ax_rsd.plot(X[kept], rsd[kept], '.', ms=3, color='k', zorder=3)
    if np.any(~kept):
        ax_rsd.plot(X[~kept], rsd[~kept], 'x', ms=4, color='0.6', zorder=2)


def run(lsf_filepath, order, version, scale, outdir, panels=False):
    ext_model = 'pixel_model' if scale[:3] == 'pix' else 'velocity_model'
    ext_gp = 'pixel_gp' if scale[:3] == 'pix' else 'velocity_gp'

    with FITS(lsf_filepath, 'r') as hdu:
        d_model = hdu[ext_model, version].read()
        d_gp = hdu[ext_gp, version].read()

    has_scatter_fields = 'sct_log_amp' in d_gp.dtype.names

    sub_model = d_model[d_model['order'] == order]
    sub_gp = d_gp[d_gp['order'] == order]
    # keep only segments with a genuine fit (numlines>0), matched by segm
    valid_segm = sub_gp['segm'][sub_gp['numlines'] > 0]
    if valid_segm.size == 0:
        print(f"No valid (numlines>0) segments for order {order} in {ext_gp}.")
        return []

    model_by_segm = {int(r['segm']): r for r in sub_model}
    gp_by_segm = {int(r['segm']): r for r in sub_gp}

    paths = []
    if not panels:
        for segm in sorted(valid_segm):
            if segm not in model_by_segm:
                print(f"segm {segm}: missing from {ext_model}, skipping.")
                continue
            path = plot_segment(model_by_segm[segm], gp_by_segm[segm],
                               has_scatter_fields, order, scale, outdir)
            paths.append(path)
            print(f"Saved: {path}")
    else:
        n = len(valid_segm)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows * 2, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
            gridspec_kw=dict(height_ratios=[2.5, 1] * nrows, wspace=0.3),
            squeeze=False,
            layout='constrained',
        )
        for i, segm in enumerate(sorted(valid_segm)):
            r, c = divmod(i, ncols)
            ax_top = axes[2 * r, c]
            ax_rsd = axes[2 * r + 1, c]
            if segm not in model_by_segm:
                ax_top.axis('off'); ax_rsd.axis('off')
                continue
            plot_segment_panel(model_by_segm[segm], gp_by_segm[segm],
                              has_scatter_fields, ax_top, ax_rsd, order, scale)
        # hide any unused trailing axes
        for j in range(len(valid_segm), nrows * ncols):
            r, c = divmod(j, ncols)
            axes[2 * r, c].axis('off')
            axes[2 * r + 1, c].axis('off')
        fig.suptitle(f'Order {order}, all segments ({scale})')
        path = f'{outdir}/lsf_diag_order{order}_allsegm_{scale}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
        print(f"Saved: {path}")

    return paths


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lsf-filepath', required=True)
    p.add_argument('--version', type=int, required=True)
    p.add_argument('--scale', default='pixel', choices=['pixel', 'velocity'])
    p.add_argument('--order', type=int, required=True)
    p.add_argument('--outdir', default='.')
    p.add_argument('--panels', action='store_true',
                   help='One multi-panel figure for the whole order instead '
                        'of a separate figure per segment.')
    args = p.parse_args()

    run(args.lsf_filepath, args.order, args.version, args.scale, args.outdir,
        panels=args.panels)