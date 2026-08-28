"""
Diagnostic plots and validation analyses for a converged order fit.
Ported from reconstruct_lsf_master.py's six plot windows plus its
per-line free-width, continuous-residual-profile, and l_x-evidence-scan
checks -- the parts of that script NOT needed to reconstruct the LSF
itself (see result.py / fits_io.py for that), kept here as an optional,
separately-importable module so a plain `import harps.lsf2` for
production runs never has to pull in matplotlib.

Every function takes the (result, state) pair `pipeline.run_order(...,
return_state=True)` returns. `run_all` produces every plot in one call
and, optionally, saves each to `output_dir` as a PNG.
"""
from __future__ import annotations

import os

import numpy as np

from .config import C_LIGHT_KMS
from .dispersion import velocity_per_pixel_from_positions
from .forward_model import gaussian_mean, pixel_model_flux
from .result import LSFOrderResult
from .shape import evaluate_departure, shape_departure_log_evidence
from .sigma_model import width
from .state import OrderState

EXPECTED_R = 145000
EXPECTED_FWHM_KMS = C_LIGHT_KMS / EXPECTED_R


def _save(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=130)
        print(f"  saved {save_path}")


def _cmap_for(state: OrderState):
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('nipy_spectral')
    norm = matplotlib.colors.Normalize(vmin=state.data.x_min, vmax=state.data.x_max)
    return cmap, norm


# =============================================================================
# Window 1: envelope + background
# =============================================================================
def plot_envelope_background(state: OrderState, save_path: str = None):
    import matplotlib.pyplot as plt
    data = state.data

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    ax.plot(data.pixel, data.flux_raw, lw=0.5, color='0.6', label='raw flux')
    ax.plot(data.pixel, state.envelope_grid_full, 'r-', lw=1, label='envelope E(x)')
    ax.plot(data.pixel, state.background_grid_full, 'b-', lw=1, label='background B(x)')
    ax.set_ylim(0, np.nanmax(state.envelope_grid_full) * 1.15)
    ax.legend(fontsize=8)
    ax.set_title(f'Order {data.order}: envelope and background')

    ax = axes[1]
    ax.plot(data.pixel, state.background_grid_full / state.envelope_grid_full, color='purple', lw=1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title('Background / envelope ratio')
    ax.set_xlabel('pixel')
    axes[0].set_xlim(data.x_min - 50, data.x_max + 50)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Window 2 (matches the project's existing "order N, all segments" diagnostic
# PNGs): one panel per position bin, all lines in that bin overlaid, model
# curve on top, residuals in a panel below. x-axis in raw pixel offset by
# default (matching those PNGs); pass axis='velocity' for km/s instead.
# =============================================================================
def plot_all_segments(state: OrderState, result: LSFOrderResult, n_segments: int = 16,
                       axis: str = 'pixel', save_path: str = None):
    import matplotlib.pyplot as plt
    data, cfg = state.data, state.cfg

    v_pix = velocity_per_pixel_from_positions(data.wavelength, state.line_position)
    line_width = width(state.line_position, state.width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
    departure = evaluate_departure(state, line_width, state.shape_coeffs, state.line_position)

    order_by_pos = np.argsort(state.line_position)
    bin_edges = np.linspace(0, data.n_lines, n_segments + 1).astype(int)
    n_cols = 4
    n_rows = int(np.ceil(n_segments / n_cols))

    fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                              sharex=True, gridspec_kw={'height_ratios': [3, 1] * n_rows})

    for i in range(n_segments):
        bin_idx = order_by_pos[bin_edges[i]:bin_edges[i + 1]]
        row, col = divmod(i, n_cols)
        ax_top, ax_bot = axes[2 * row, col], axes[2 * row + 1, col]

        U, F, M = [], [], []
        for m in bin_idx:
            idx = state.fit_window(state.line_position[m])
            offset = idx - state.line_position[m]
            x_axis_vals = offset * v_pix[m] if axis == 'velocity' else offset
            model_vals = pixel_model_flux(state.u, state.du, state.line_position[m], idx,
                                           v_pix[m], line_width[m], departure[m])
            U.append(x_axis_vals)
            F.append(state.flux[idx])
            M.append(model_vals)
        if not U:
            continue
        U, F, M = np.concatenate(U), np.concatenate(F), np.concatenate(M)
        srt = np.argsort(U)

        ax_top.plot(U, F, '.', ms=3, color='k', alpha=0.6)
        ax_top.plot(U[srt], M[srt], '-', color='tab:blue', lw=1)
        ax_top.axhline(0, color='gray', lw=0.4, ls=':')
        ax_top.set_title(f'segm {i}', fontsize=9)

        ax_bot.plot(U, F - M, '.', ms=2, color='k', alpha=0.5)
        ax_bot.axhline(0, color='gray', lw=0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel('pixel' if axis == 'pixel' else 'u [km/s]')
    fig.suptitle(f'Order {data.order}, all segments ({axis})')
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Window 3: wavelength calibration residuals
# =============================================================================
def plot_dispersion_residuals(state: OrderState, save_path: str = None):
    import matplotlib.pyplot as plt
    data = state.data
    v_pix = velocity_per_pixel_from_positions(data.wavelength, state.line_position)

    target = state.dispersion_raw_target
    target_err = state.dispersion_raw_target_err
    dlambda_dx = data.wavelength * v_pix / C_LIGHT_KMS   # nm per pixel, local dispersion
    wavelength_residual = (target - state.line_position) * dlambda_dx
    wavelength_residual_err = target_err * np.abs(dlambda_dx)
    velocity_residual = 2.99792458e8 * (wavelength_residual / data.wavelength)
    velocity_residual_err = 2.99792458e8 * (wavelength_residual_err / data.wavelength)
    dispersion_std_ms = 2.99792458e8 * (state.dispersion_gp_fit['z_std'] * np.abs(dlambda_dx) / data.wavelength)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.errorbar(data.wavelength, velocity_residual, yerr=velocity_residual_err,
                fmt='.', ms=4, elinewidth=0.5, capsize=0, label='per-line residual')
    srt = np.argsort(data.wavelength)
    ax.fill_between(data.wavelength[srt], -dispersion_std_ms[srt], dispersion_std_ms[srt],
                     color='tab:green', alpha=0.25, label='dispersion GP posterior std')
    ax.axhline(0, color='gray', lw=0.5)
    ax.legend(fontsize=8)
    ax.set_title(f'Order {data.order}: wavelength calibration residuals')
    ax.set_xlabel('wavelength [nm]')
    ax.set_ylabel('residual [m/s]')
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Window 4: FWHM(x), and FWHM vs. local background/envelope ratio
# =============================================================================
def plot_width(state: OrderState, save_path: str = None):
    import matplotlib.pyplot as plt
    data, cfg = state.data, state.cfg

    fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(16, 6))
    sigma_grid = width(data.pixel.astype(float), state.width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
    fwhm_grid = 2.355 * sigma_grid
    width_std = state.width_gp_fit['z_std'] if state.width_gp_fit else np.zeros_like(sigma_grid)
    fwhm_upper = 2.355 * np.exp(state.width_coeffs + width_std)
    fwhm_lower = 2.355 * np.exp(state.width_coeffs - width_std)

    ax.plot(data.pixel, fwhm_grid, color='tab:orange', lw=1.3, label='FWHM(x)')
    if state.width_raw_target is not None:
        ax.errorbar(state.line_position, 2.355 * np.exp(state.width_raw_target),
                     yerr=2.355 * np.exp(state.width_raw_target) * state.width_raw_target_err,
                     fmt='.', ms=4, elinewidth=0.5, capsize=0, alpha=0.5, label='per-line target (pre-GP)')
    ax.fill_between(data.pixel, fwhm_lower, fwhm_upper, color='tab:orange', alpha=0.3, label='GP posterior std')
    ax.axhline(EXPECTED_FWHM_KMS, color='k', ls='--', lw=1,
               label=f'R={EXPECTED_R} ({EXPECTED_FWHM_KMS:.3f} km/s)')
    ax.legend(fontsize=8)
    ax.set_xlim(data.x_min - 50, data.x_max + 50)
    ax.set_title(f'Order {data.order}: FWHM(x) [km/s]')
    ax.set_xlabel('pixel')
    ax.set_ylabel('FWHM [km/s]')

    if state.width_raw_target is not None:
        be_ratio = (np.interp(state.line_position, data.pixel, state.background_grid_full)
                    / np.interp(state.line_position, data.pixel, state.envelope_grid_full))
        fwhm_lines = 2.355 * np.exp(state.width_raw_target)
        cmap, norm = _cmap_for(state)
        ax_b.scatter(be_ratio, fwhm_lines, c=cmap(norm(state.line_position)), s=10, alpha=0.6)
        finite = np.isfinite(be_ratio) & np.isfinite(fwhm_lines)
        if finite.sum() > 2:
            r = np.corrcoef(be_ratio[finite], fwhm_lines[finite])[0, 1]
            slope, intercept = np.polyfit(be_ratio[finite], fwhm_lines[finite], 1)
            xs = np.linspace(be_ratio[finite].min(), be_ratio[finite].max(), 50)
            ax_b.plot(xs, slope * xs + intercept, 'k--', lw=1, label=f'linear fit, r={r:.3f}')
            ax_b.legend(fontsize=8)
        ax_b.set_title('Per-line FWHM vs. local background/envelope ratio')
        ax_b.set_xlabel('B(x) / E(x)')
        ax_b.set_ylabel('FWHM [km/s]')

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Window 5: full raw-flux model overlay + normalised residuals
# =============================================================================
def plot_full_model(state: OrderState, save_path: str = None):
    import matplotlib.pyplot as plt
    data, cfg = state.data, state.cfg

    v_pix = velocity_per_pixel_from_positions(data.wavelength, state.line_position)
    line_width = width(state.line_position, state.width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
    departure = evaluate_departure(state, line_width, state.shape_coeffs, state.line_position)

    model = np.zeros(data.n_pixels)
    fitted_mask = np.zeros(data.n_pixels, dtype=bool)
    for m in range(data.n_lines):
        idx = state.fit_window(state.line_position[m])
        model[idx] += pixel_model_flux(state.u, state.du, state.line_position[m], idx,
                                        v_pix[m], line_width[m], departure[m])
        fitted_mask[idx] = True
    residual_norm = (state.flux - model) / state.flux_err
    model_raw = model * (state.envelope_grid_full - state.background_grid_full) + state.background_grid_full

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax = axes[0]
    ax.errorbar(data.pixel[fitted_mask], data.flux_raw[fitted_mask], data.err_raw[fitted_mask],
                marker='.', ms=8, alpha=0.5, label='data (raw)')
    ax.plot(data.pixel[fitted_mask], model_raw[fitted_mask], 'r-', lw=0.8, label='model (raw)')
    ax.plot(data.pixel, state.envelope_grid_full, color='tab:orange', lw=1, alpha=0.8, label='envelope E(x)')
    ax.plot(data.pixel, state.background_grid_full, color='tab:blue', lw=1, alpha=0.8, label='background B(x)')
    ax.legend(fontsize=8)
    ax.set_title(f'Order {data.order}: all lines, data + forward model (raw flux units)')
    ax.set_ylabel('flux [counts]')
    axes[0].set_xlim(data.x_min - 50, data.x_max + 50)

    ax = axes[1]
    ax.plot(data.pixel[fitted_mask], residual_norm[fitted_mask], '.', ms=2, alpha=0.4)
    ax.axhline(0, color='r', lw=0.8)
    ax.set_ylim(-30, 30)
    ax.set_title('Normalised residuals, (flux - model) / error')
    ax.set_xlabel('pixel')
    fig.tight_layout()
    _save(fig, save_path)
    return fig, model, fitted_mask, residual_norm


# =============================================================================
# Window 6: LSF departure from a Gaussian, across the order
# =============================================================================
def plot_shape_departure(state: OrderState, save_path: str = None, n_positions: int = 16):
    import matplotlib.pyplot as plt
    data, cfg = state.data, state.cfg

    x_pos_array = np.linspace(data.x_min, data.x_max, n_positions)
    cmap, norm = _cmap_for(state)
    colours = cmap(norm(x_pos_array))
    sigma_x = width(x_pos_array, state.width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
    departure_x = evaluate_departure(state, sigma_x, state.shape_coeffs, x_pos_array)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for x_pos, colour, dep in zip(x_pos_array, colours, departure_x):
        ax.plot(state.u, dep, color=colour, label=f'{x_pos:.0f}')
    ax.axhline(0, color='gray', lw=0.5)
    ax.legend(fontsize=6, ncol=2, title='pixel')
    ax.set_title('Departure from Gaussian: phi(u;x) - G(u;sigma(x))')
    ax.set_xlabel('u [km/s]')
    ax.set_ylabel('departure')

    ax = axes[1]
    im = ax.imshow(state.shape_coeffs.T, aspect='auto', origin='lower', cmap='RdBu_r',
                    extent=[state.u_inducing.min(), state.u_inducing.max(),
                            state.x_inducing.min(), state.x_inducing.max()],
                    vmin=-np.max(np.abs(state.shape_coeffs)), vmax=np.max(np.abs(state.shape_coeffs)))
    fig.colorbar(im, ax=ax, label='departure')
    ax.set_title('2D GP departure grid D(u,x)')
    ax.set_xlabel('u [km/s]')
    ax.set_ylabel('pixel')

    fig.suptitle(f'Order {data.order}: LSF shape departure from a Gaussian')
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =============================================================================
# Per-line free width: releases sigma_m independently per line (no
# smoothness across lines at all) and compares against the smooth width(x)
# model, both directly (FWHM vs pixel) and via a shoulder/core/far-wing
# stacked-residual comparison.
# =============================================================================
def fit_per_line_width(state: OrderState) -> np.ndarray:
    from scipy.optimize import minimize_scalar
    data, cfg = state.data, state.cfg
    v_pix = velocity_per_pixel_from_positions(data.wavelength, state.line_position)
    sigma_init = width(state.line_position, state.width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
    departure_all = evaluate_departure(state, sigma_init, state.shape_coeffs, state.line_position)

    sigma_per_line = np.empty(data.n_lines)
    for m in range(data.n_lines):
        idx = state.fit_window(state.line_position[m])
        weight = state.inverse_variance[idx]
        departure_m = departure_all[m]
        target = state.flux[idx]

        def neg_log_likelihood(log_sigma, idx=idx, weight=weight, departure_m=departure_m,
                                target=target, m=m):
            sigma_m = np.exp(log_sigma)
            model = pixel_model_flux(state.u, state.du, state.line_position[m], idx,
                                      v_pix[m], sigma_m, departure_m)
            resid = target - model
            return 0.5 * np.sum(weight * resid ** 2)

        result = minimize_scalar(neg_log_likelihood,
                                  bounds=(np.log(0.3 * sigma_init[m]), np.log(3 * sigma_init[m])),
                                  method='bounded', options={'xatol': 1e-5})
        sigma_per_line[m] = np.exp(result.x)
    return sigma_per_line


def stacked_residual_bins(state: OrderState, sigma_array: np.ndarray, departure_array: np.ndarray):
    """ Weighted-mean residual in three fixed u bins (core/shoulder/
        far-wing), plus chi2 and point count. Used both for the per-line
        free-width comparison and the l_x length-scale comparison below. """
    data = state.data
    v_pix = velocity_per_pixel_from_positions(data.wavelength, state.line_position)
    u_all, resid_all, weight_all = [], [], []
    for m in range(data.n_lines):
        idx = state.fit_window(state.line_position[m])
        model = pixel_model_flux(state.u, state.du, state.line_position[m], idx,
                                  v_pix[m], sigma_array[m], departure_array[m])
        u_all.append((idx - state.line_position[m]) * v_pix[m])
        resid_all.append(state.flux[idx] - model)
        weight_all.append(state.inverse_variance[idx])
    u_all, resid_all, weight_all = np.concatenate(u_all), np.concatenate(resid_all), np.concatenate(weight_all)

    def wmean(mask):
        return np.average(resid_all[mask], weights=weight_all[mask]) if mask.sum() > 0 else np.nan

    peak_mask = np.abs(u_all) < 0.5
    shoulder_mask = (u_all >= 0.5) & (u_all < 1.5)
    far_mask = (u_all >= 2.0) & (u_all < 3.0)
    chi2 = float(np.sum(weight_all * resid_all ** 2))
    return {'peak': wmean(peak_mask), 'shoulder': wmean(shoulder_mask), 'far_wing': wmean(far_mask),
            'chi2': chi2, 'n_points': len(resid_all)}


def plot_per_line_width(state: OrderState, save_path: str = None):
    import matplotlib.pyplot as plt
    data, cfg = state.data, state.cfg

    sigma_per_line = fit_per_line_width(state)
    line_width = width(state.line_position, state.width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
    departure_all = evaluate_departure(state, line_width, state.shape_coeffs, state.line_position)

    order_by_pos = np.argsort(state.line_position)
    fwhm_sorted = 2.355 * sigma_per_line[order_by_pos]
    position_sorted = state.line_position[order_by_pos]
    odd_mean, even_mean = np.mean(fwhm_sorted[1::2]), np.mean(fwhm_sorted[0::2])
    print(f"Per-line width, odd/even split: even={even_mean:.4f} km/s, odd={odd_mean:.4f} km/s, "
          f"|diff|={abs(even_mean - odd_mean) * 1000:.2f} m/s")

    smooth = stacked_residual_bins(state, line_width, departure_all)
    free = stacked_residual_bins(state, sigma_per_line, departure_all)
    print(f"Shoulder-excess, smooth width(x):  peak={smooth['peak']:+.5f}, "
          f"shoulder={smooth['shoulder']:+.5f}, far_wing={smooth['far_wing']:+.5f}")
    print(f"Shoulder-excess, per-line free width: peak={free['peak']:+.5f}, "
          f"shoulder={free['shoulder']:+.5f}, far_wing={free['far_wing']:+.5f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    smooth_fwhm_curve = 2.355 * width(data.pixel.astype(float), state.width_coeffs, data.pixel, cfg,
                                       state.v_per_pixel_typical)
    ax1.plot(data.pixel, smooth_fwhm_curve, color='tab:orange', lw=1.5, label='smooth width(x)', zorder=3)
    ax1.scatter(position_sorted[0::2], fwhm_sorted[0::2], s=14, marker='^', color='tab:green',
                label='even index', zorder=4)
    ax1.scatter(position_sorted[1::2], fwhm_sorted[1::2], s=14, marker='v', color='tab:red',
                label='odd index', zorder=4)
    ax1.set_xlim(data.x_min - 50, data.x_max + 50)
    ax1.set_xlabel('pixel')
    ax1.set_ylabel('FWHM [km/s]')
    ax1.set_title('Per-line free width vs. smooth width(x)')
    ax1.legend(fontsize=7)

    x_pos, bar_w = np.arange(3), 0.35
    keys = ['peak', 'shoulder', 'far_wing']
    ax2.bar(x_pos - bar_w / 2, [smooth[k] for k in keys], bar_w, label='smooth width(x)',
            color='tab:orange', alpha=0.8)
    ax2.bar(x_pos + bar_w / 2, [free[k] for k in keys], bar_w, label='per-line free width',
            color='tab:blue', alpha=0.8)
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(keys)
    ax2.set_ylabel('weighted mean residual (flux - model)')
    ax2.set_title('Shoulder-excess bins: smooth vs. per-line free width')
    ax2.legend(fontsize=8)

    fig.suptitle(f'Order {data.order}: releasing the Gaussian core width per line')
    fig.tight_layout()
    _save(fig, save_path)
    return fig, sigma_per_line


# =============================================================================
# Continuous residual profile vs. u -- finer-grained than the 3-bin summary
# above, plus a lag-1 autocorrelation check (noise-covariance vs genuine
# shape mismatch).
# =============================================================================
def plot_continuous_residual_profile(state: OrderState, n_bins: int = 30, save_path: str = None):
    import matplotlib.pyplot as plt
    data, cfg = state.data, state.cfg

    v_pix = velocity_per_pixel_from_positions(data.wavelength, state.line_position)
    line_width = width(state.line_position, state.width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)
    departure_all = evaluate_departure(state, line_width, state.shape_coeffs, state.line_position)

    u_stack, resid_stack, weight_stack = [], [], []
    for m in range(data.n_lines):
        idx = state.fit_window(state.line_position[m])
        model_window = pixel_model_flux(state.u, state.du, state.line_position[m], idx,
                                         v_pix[m], line_width[m], departure_all[m])
        u_stack.append((idx - state.line_position[m]) * v_pix[m])
        resid_stack.append(state.flux[idx] - model_window)
        weight_stack.append(state.inverse_variance[idx])
    u_stack, resid_stack, weight_stack = (np.concatenate(u_stack), np.concatenate(resid_stack),
                                           np.concatenate(weight_stack))
    srt = np.argsort(u_stack)
    u_stack, resid_stack, weight_stack = u_stack[srt], resid_stack[srt], weight_stack[srt]

    bin_edges = np.linspace(u_stack.min(), u_stack.max(), n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_mean = np.full(n_bins, np.nan)
    bin_err = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (u_stack >= bin_edges[i]) & (u_stack < bin_edges[i + 1])
        if mask.sum() > 5:
            bin_mean[i] = np.average(resid_stack[mask], weights=weight_stack[mask])
            bin_err[i] = 1 / np.sqrt(np.sum(weight_stack[mask]))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.errorbar(bin_centres, bin_mean, yerr=bin_err, fmt='o-', ms=4, capsize=2, color='tab:blue')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('u [km/s]')
    ax.set_ylabel('weighted mean residual (flux - model)')
    ax.set_title(f'Order {data.order}: continuous residual profile vs. u ({n_bins} bins)')
    fig.tight_layout()
    _save(fig, save_path)

    if len(resid_stack) > 1:
        lag1 = np.corrcoef(resid_stack[:-1], resid_stack[1:])[0, 1]
        print(f"Stacked-residual (u-sorted) lag-1 autocorrelation: {lag1:+.4f}")

    return fig, bin_centres, bin_mean, bin_err


# =============================================================================
# l_x identifiability scan -- is the fitted SHAPE_X_LENGTH_SCALE close to the
# closed-form evidence peak, or is the profile flat (meaning a free
# optimiser could wander)?
# =============================================================================
def lx_evidence_scan(state: OrderState, n_candidates: int = 20,
                      lx_bounds: tuple = (200, 10000)):
    data = state.data
    v_pix = velocity_per_pixel_from_positions(data.wavelength, state.line_position)
    candidates = np.logspace(np.log10(lx_bounds[0]), np.log10(lx_bounds[1]), n_candidates)
    log_evidence = np.array([
        shape_departure_log_evidence(lx, state, state.line_position, state.width_coeffs, v_pix)
        for lx in candidates
    ])
    return candidates, log_evidence


def plot_lx_evidence_scan(state: OrderState, save_path: str = None, **kwargs):
    import matplotlib.pyplot as plt
    candidates, log_evidence = lx_evidence_scan(state, **kwargs)
    peak_idx = int(np.argmax(log_evidence))
    spread = float(log_evidence.max() - log_evidence.min())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(candidates, log_evidence - log_evidence.max(), 'o-')
    ax.axvline(state.shape_x_length_scale, color='tab:red', ls='--', lw=1.5,
               label=f'converged value ({state.shape_x_length_scale:.0f} px)')
    ax.axvline(state.cfg.shape_x_length_scale_init, color='k', ls=':', lw=1,
               label=f'prior centre ({state.cfg.shape_x_length_scale_init:.0f} px)')
    ax.set_xscale('log')
    ax.set_xlabel('candidate l_x [pixels]')
    ax.set_ylabel('log evidence, relative to max')
    ax.set_title(f'Order {state.data.order}: l_x identifiability -- evidence peak at '
                 f'{candidates[peak_idx]:.0f} px, spread={spread:.1f} nats')
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    print(f"l_x evidence scan: peak at {candidates[peak_idx]:.0f} px, spread={spread:.3f} nats "
          f"({'identifiable' if spread > 5 else 'nearly flat -- treat l_x as weakly constrained'})")
    return fig, candidates, log_evidence


# =============================================================================
# Orchestrator
# =============================================================================
def run_all(state: OrderState, result: LSFOrderResult, output_dir: str = None,
            show: bool = True, n_segments: int = 16):
    """ Produces every diagnostic plot above. If output_dir is given, each
        is also saved as order<N>_<name>.png there. """
    import matplotlib.pyplot as plt

    def path(name):
        if output_dir is None:
            return None
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"order{state.data.order}_{name}.png")

    plot_envelope_background(state, save_path=path('envelope_background'))
    plot_all_segments(state, result, n_segments=n_segments, save_path=path('all_segments_pixel'))
    plot_dispersion_residuals(state, save_path=path('dispersion_residuals'))
    plot_width(state, save_path=path('width'))
    plot_full_model(state, save_path=path('full_model'))
    plot_shape_departure(state, save_path=path('shape_departure'))
    plot_per_line_width(state, save_path=path('per_line_width'))
    plot_continuous_residual_profile(state, save_path=path('residual_profile'))
    plot_lx_evidence_scan(state, save_path=path('lx_evidence_scan'))

    if show:
        plt.show()
