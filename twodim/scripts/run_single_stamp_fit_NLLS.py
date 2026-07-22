#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:01:13 2026

@author: dmilakov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified version of run_single_stamp_fit.py.

Uses the non-linear least squares 2D Gaussian fitter that already lives in 
EchelleAnalyzer._process_single_stamp(). Recovers sigma_x and sigma_y with 
formal 1-sigma uncertainties derived from
the curve_fit covariance matrix.

Usage is identical to the original: pass a JSON config file.
The JSON no longer needs an 'mcmc_params' section; it is silently ignored
if present so existing configs stay compatible.
"""

import time
import json
import re
import warnings
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Package imports  (same as original)
# ---------------------------------------------------------------------------
from harps.twodim.analyzer import EchelleAnalyzer   # new (current) analyzer
# We also need the fitting function to rebuild the model for residual plots
from harps.twodim.fitting import twoD_Gaussian

GAUSS_PARAMS_NAME = [
    'AMPLITUDE', 'CEN_X', 'CEN_Y', 'SIGMA_X', 'SIGMA_Y', 'THETA', 'CONST'
]


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------

def parse_zernike_string(zern_str):
    """Parses string like '(n1,m1) (n2,m2)...' into list of tuples."""
    if not zern_str:
        return None
    pattern = re.compile(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)')
    matches = pattern.findall(zern_str)
    if not matches:
        raise ValueError(
            f"Could not parse Zernike string: '{zern_str}'. "
            "Expected format like '(n,m) (n,m)'."
        )
    return [(int(n), int(m)) for n, m in matches]


# ---------------------------------------------------------------------------
# New: thin wrapper that calls the existing NLLS fitter on one stamp
# ---------------------------------------------------------------------------

def fit_stamp_nlls(stamp_data, stamp_half_width, fit_threshold_snr=5.0):
    """
    Fit a 2D Gaussian to *stamp_data* using non-linear least squares
    (scipy.optimize.curve_fit / Levenberg-Marquardt via 'trf').

    This is essentially _process_single_stamp() pulled out of EchelleAnalyzer
    so it can be called on an already-extracted stamp without needing the full
    analyzer image loaded.

    Parameters
    ----------
    stamp_data : 2-D ndarray
        Raw (un-normalised) pixel values, shape (2*sw+1, 2*sw+1).
    stamp_half_width : int
        Half-width of the stamp (sw).  Used for initial guesses and bounds.
    fit_threshold_snr : float
        Minimum amplitude / amplitude_error to accept a fit.

    Returns
    -------
    result : dict or None
        Keys:
          'popt'      – array of 7 best-fit parameters
                        [amplitude_norm, x_centre_stamp, y_centre_stamp,
                         sigma_x, sigma_y, theta_rad, offset_norm]
          'perr'      – 1-sigma formal uncertainties (same order)
          'pcov'      – full 7×7 covariance matrix
          'chi2_red'  – reduced chi-squared
          'stamp_norm'– normalised stamp (what was actually fitted)
          'model_norm'– best-fit model evaluated on the stamp grid
          'residuals' – normalised residuals map, shape = stamp_data.shape
          'param_names' – list of parameter name strings
        Returns None if the fit failed any quality gate.
    """
    sw = stamp_half_width
    total_flux = np.sum(stamp_data)
    if total_flux <= 1e-9:
        print("  [NLLS] Rejected: total flux ≤ 0")
        return None

    stamp_norm = stamp_data / total_flux
    stamp_h, stamp_w = stamp_norm.shape

    # Pixel coordinate grids (stamp-local, origin at top-left corner)
    y_sg, x_sg = np.meshgrid(
        np.arange(stamp_h), np.arange(stamp_w), indexing='ij'
    )

    # ---- Initial parameter guess ----------------------------------------
    p0 = (
        np.max(stamp_norm),   # amplitude (normalised)
        stamp_w / 2.0,        # x centre
        stamp_h / 2.0,        # y centre
        sw / 3.0,             # sigma_x  (pixels)
        sw / 3.0,             # sigma_y  (pixels)
        0.0,                  # theta    (radians)
        np.min(stamp_norm),   # constant background offset
    )

    # ---- Parameter bounds -----------------------------------------------
    # Keep sigmas physical and centroids inside the stamp
    bounds = (
        [0,          -stamp_w,  -stamp_h,  0.1, 0.1, -np.pi, -np.inf],
        [np.inf,  2*stamp_w, 2*stamp_h, stamp_w, stamp_h,  np.pi,  np.inf],
    )

    # ---- Fit -------------------------------------------------------------
    try:
        popt, pcov = curve_fit(
            twoD_Gaussian,
            (x_sg, y_sg),
            stamp_norm.ravel(),
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"  [NLLS] curve_fit failed: {exc}")
        return None

    # ---- Quality gates ---------------------------------------------------
    if not np.all(np.isfinite(pcov)):
        print("  [NLLS] Rejected: non-finite covariance matrix")
        return None

    perr = np.sqrt(np.diag(pcov))

    amp_snr = popt[0] / perr[0] if perr[0] > 1e-12 else np.inf
    if amp_snr < fit_threshold_snr:
        print(f"  [NLLS] Rejected: amplitude S/N = {amp_snr:.2f} < {fit_threshold_snr}")
        return None

    if not np.all(np.isfinite(popt)):
        print("  [NLLS] Rejected: non-finite best-fit parameters")
        return None

    sigma_x, sigma_y = popt[3], popt[4]
    if sigma_x <= 0 or sigma_y <= 0:
        print(f"  [NLLS] Rejected: non-positive sigma (sx={sigma_x:.3f}, sy={sigma_y:.3f})")
        return None

    # ---- Residuals and goodness-of-fit ----------------------------------
    model_norm = twoD_Gaussian((x_sg, y_sg), *popt)
    res_norm_1d = stamp_norm.ravel() - model_norm

    # Poisson-like variance in the normalised domain:  Var(d/F) ≈ max(d,1)/F²
    var_norm = np.maximum(stamp_data.ravel(), 1e-9) / (total_flux ** 2)
    res_div_sig = res_norm_1d / np.sqrt(var_norm)

    chi2 = np.sum(res_div_sig ** 2)
    dof  = max(1, stamp_norm.size - len(popt))
    chi2_red = chi2 / dof

    return {
        'popt':        popt,
        'perr':        perr.astype('f4'),
        'pcov':        pcov,
        'chi2_red':    chi2_red,
        'stamp_norm':  stamp_norm,
        'model_norm':  model_norm.reshape(stamp_h, stamp_w),
        'residuals':   res_div_sig.reshape(stamp_h, stamp_w),
        'param_names': GAUSS_PARAMS_NAME,
    }


def print_nlls_results(fit_result):
    """Pretty-print the NLLS fit results."""
    if fit_result is None:
        print("  No valid fit result to display.")
        return

    popt = fit_result['popt']
    perr = fit_result['perr']
    names = fit_result['param_names']

    print("\n--- NLLS Fit Results ---")
    print(f"  {'Parameter':<14}  {'Value':>12}  {'±1σ':>12}")
    print(f"  {'-'*14}  {'-'*12}  {'-'*12}")
    for name, val, err in zip(names, popt, perr):
        print(f"  {name:<14}  {val:>12.6f}  {err:>12.6f}")

    print(f"\n  sigma_x = {popt[3]:.4f} ± {perr[3]:.4f}  px")
    print(f"  sigma_y = {popt[4]:.4f} ± {perr[4]:.4f}  px")
    print(f"  theta   = {np.degrees(popt[5]):.3f} ± {np.degrees(perr[5]):.3f}  deg")
    print(f"  Red. χ² = {fit_result['chi2_red']:.4f}")


def plot_nlls_overview(stamp_data, fit_result, title_prefix="", filename=None):
    """
    Three-panel diagnostic plot:
      left   – raw stamp data
      centre – best-fit 2D Gaussian model
      right  – normalised residuals  (data − model) / σ_pixel
    """
    if fit_result is None:
        print("  Cannot plot: no fit result.")
        return

    stamp_norm  = fit_result['stamp_norm']
    model_norm  = fit_result['model_norm']
    residuals   = fit_result['residuals']
    popt        = fit_result['popt']
    chi2_red    = fit_result['chi2_red']

    fig = plt.figure(figsize=(13, 4))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    vmin, vmax = stamp_norm.min(), stamp_norm.max()
    res_lim = max(3.0, np.nanpercentile(np.abs(residuals), 99))

    # --- Panel 1: data ---
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(stamp_norm, origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    ax0.set_title("Data (normalised)")
    plt.colorbar(im0, ax=ax0, fraction=0.046)

    # overlay ellipse
    from matplotlib.patches import Ellipse
    sx, sy, th = popt[3], popt[4], popt[5]
    xc, yc     = popt[1], popt[2]
    ell = Ellipse(
        xy=(xc, yc),
        width=4 * sx, height=4 * sy,
        angle=np.degrees(th),
        edgecolor='red', facecolor='none', lw=1.5, label='2σ ellipse'
    )
    ax0.add_patch(ell)
    ax0.plot(xc, yc, 'r+', ms=8, mew=1.5)

    # --- Panel 2: model ---
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(model_norm, origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
    ax1.set_title(
        f"Model\n"
        f"σ_x={sx:.3f}±{fit_result['perr'][3]:.3f}  "
        f"σ_y={sy:.3f}±{fit_result['perr'][4]:.3f} px"
    )
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # --- Panel 3: residuals ---
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(
        residuals, origin='lower',
        vmin=-res_lim, vmax=res_lim, cmap='RdBu_r'
    )
    ax2.set_title(f"Residuals (data−model)/σ\nRed. χ²={chi2_red:.3f}")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    fig.suptitle(title_prefix, fontsize=11, y=1.01)
    plt.tight_layout()

    if filename:
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"  Saved overview plot to {filename}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    # --- Argument parser (identical interface to original) ----------------
    parser = argparse.ArgumentParser(
        description="Fit a single LFC stamp with NLLS 2D Gaussian (replaces MCMC)."
    )
    parser.add_argument("config_file", type=str,
                        help="Path to the JSON configuration file.")
    parser.add_argument("-ord", "--order",    type=int)
    parser.add_argument("-img", "--image",    type=str, choices=['A', 'B'])
    parser.add_argument("-seg", "--segment",  type=int)
    parser.add_argument("-p",   "--peak_index", type=int)
    parser.add_argument("--show_plots", action='store_true')
    args = parser.parse_args()

    # --- Load JSON config -------------------------------------------------
    config_path = Path(args.config_file)
    if not config_path.is_file():
        print(f"Error: config file not found at {config_path}")
        return
    with open(config_path) as f:
        config = json.load(f)
    print(f"Loaded config: {config_path}")

    # Ignore 'mcmc_params' silently so old configs stay valid
    if 'mcmc_params' in config:
        warnings.warn(
            "'mcmc_params' found in config but will be ignored — "
            "this script uses NLLS, not MCMC.",
            UserWarning
        )

    # --- File paths -------------------------------------------------------
    results_fits_path = Path(config.get('results_fits_file', ''))
    lfc_file_path     = Path(config.get('lfc_source_file', ''))
    bias_file_path    = Path(config['bias_file']) if config.get('bias_file') else None

    plot_settings = config.get('plot_settings', {})
    plot_dir      = Path(plot_settings.get('plot_dir', './single_stamp_plots_nlls'))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Target selection ------------------------------------------------
    tgt = config.get('target', {})
    target_order      = args.order      if args.order      is not None else tgt.get('order')
    target_image_str  = (args.image     if args.image      is not None else tgt.get('image', 'A')).upper()
    target_segment    = args.segment    if args.segment    is not None else tgt.get('segment')
    target_peak_index = args.peak_index if args.peak_index is not None else tgt.get('peak_index')

    if None in [target_order, target_segment, target_peak_index]:
        print("Error: order, segment, and peak_index must all be set.")
        return
    if target_image_str not in ['A', 'B']:
        print("Error: image must be 'A' or 'B'.")
        return
    img_type_int = 0 if target_image_str == 'A' else 1

    # --- NLLS fitter parameters (from config or defaults) ----------------
    # These come from 'fitter_params' in the JSON (same key as before),
    # but now only the two NLLS-relevant ones are used.
    fitter_cfg        = config.get('fitter_params', {})
    fit_threshold_snr = fitter_cfg.get('fit_threshold_snr', 5.0)
    stamp_hw_override = fitter_cfg.get('stamp_half_width', None)  # optional override

    # --- Plotting flags ---------------------------------------------------
    save_plots   = plot_settings.get('save_plots', True)
    if args.show_plots:
        save_plots = False
    enable_plots = not plot_settings.get('no_plots', False)
    plot_format  = plot_settings.get('plot_format', 'png').lower()
    if plot_format not in ['pdf', 'png']:
        plot_format = 'png'

    # --- Initialise EchelleAnalyzer as a reader --------------------------
    # (identical to original — we only use it for stamp extraction)
    print(f"\nInitialising reader for results file: {results_fits_path}")
    reader = EchelleAnalyzer(
        lfc_filename=lfc_file_path or 'dummy.fits',
        bias_filename=bias_file_path,
        output_dir = config.get("output_dir")
    )
    reader.output_fits_path = results_fits_path
    # reader._ensure_fits_structure()

    # Read metadata to discover detector type and stamp_half_width used
    metadata = reader.get_fits_metadata(results_fits_path)
    if metadata is None:
        return
    detector  = metadata.get('detector', 'red')
    detector = "red"
    print(f"\nDetector = {detector}")
    stamp_hw  = stamp_hw_override if stamp_hw_override is not None \
                else metadata.get('params', {}).get('stamp_half_width', 5)

    # Load image data
    print(f"\nLoading image data ({detector} detector)...")
    reader.lfc_path = lfc_file_path
    if not reader.load_data(detector=detector):
        print("Failed to load image data.")
        return

    # --- Read peak catalog and select target peak -----------------------
    print("\nReading peak catalog...")
    peak_catalog = reader.read_peak_catalog(results_fits_path)
    reader._write_peak_catalog()
    if peak_catalog is None:
        return

    print(f"Selecting target: Order={target_order}, Image={target_image_str}, "
          f"Segment={target_segment}, PeakIndex={target_peak_index}")

    seg_mask = (
        (peak_catalog['ORDER_NUM'] == target_order)  &
        (peak_catalog['IMGTYPE']   == img_type_int)  &
        (peak_catalog['SEGMENT']   == target_segment)
    )
    peaks_in_seg = peak_catalog[seg_mask]

    if len(peaks_in_seg) == 0:
        print("Error: no peaks found for specified target.")
        return
    if not (0 <= target_peak_index < len(peaks_in_seg)):
        print(f"Error: peak_index {target_peak_index} out of range "
              f"(segment has {len(peaks_in_seg)} peaks).")
        return

    target_peak = peaks_in_seg[target_peak_index]
    peak_x = int(target_peak['PEAK_X'])
    peak_y = int(target_peak['PEAK_Y'])
    print(f"Target peak at pixel ({peak_x}, {peak_y})")

    # --- Extract stamp ---------------------------------------------------
    print(f"Extracting {2*stamp_hw+1}×{2*stamp_hw+1} stamp...")
    stamp_data = reader.get_stamp_data(peak_x, peak_y, stamp_half_width=stamp_hw)
    if stamp_data is None:
        print("Error: could not extract stamp (peak too close to edge?).")
        return
    print(f"Stamp shape: {stamp_data.shape}")

    # --- Run NLLS fit ----------------------------------------------------
    print("\nRunning NLLS 2D Gaussian fit...")
    t0 = time.time()
    fit_result = fit_stamp_nlls(stamp_data, stamp_hw, fit_threshold_snr=fit_threshold_snr)
    dt = time.time() - t0
    print(f"Fit completed in {dt*1e3:.1f} ms.")

    if fit_result is None:
        print("\nNLLS fit failed — no result to display.")
        return

    # --- Report results --------------------------------------------------
    print_nlls_results(fit_result)

    # Convenience: pull out the two numbers you care about most
    popt = fit_result['popt']
    perr = fit_result['perr']
    sigma_x      = popt[3];  sigma_x_err = perr[3]
    sigma_y      = popt[4];  sigma_y_err = perr[4]
    theta_deg    = np.degrees(popt[5])
    theta_deg_err= np.degrees(perr[5])
    chi2_red     = fit_result['chi2_red']

    print(f"\n{'='*50}")
    print(f"  sigma_x = {sigma_x:.4f} ± {sigma_x_err:.4f} px")
    print(f"  sigma_y = {sigma_y:.4f} ± {sigma_y_err:.4f} px")
    print(f"  theta   = {theta_deg:.3f} ± {theta_deg_err:.3f} deg")
    print(f"  Red. χ² = {chi2_red:.4f}")
    print(f"{'='*50}")

    # --- Plot ------------------------------------------------------------
    if enable_plots:
        base = (f"O{target_order}{target_image_str}"
                f"_S{target_segment}_P{target_peak_index}")
        title_prefix = (f"NLLS fit  |  O={target_order}{target_image_str}  "
                        f"S={target_segment}  P={target_peak_index}  "
                        f"({peak_x},{peak_y})")

        fname_overview = (
            plot_dir / f"{base}_nlls_overview.{plot_format}"
            if save_plots else None
        )
        try:
            plot_nlls_overview(
                stamp_data, fit_result,
                title_prefix=title_prefix,
                filename=fname_overview,
            )
        except Exception as exc:
            print(f"Error generating overview plot: {exc}")

    print("\nSingle stamp NLLS fitting finished.")
    return fit_result   # useful when called programmatically


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from scipy.optimize import curve_fit   # only needed at runtime
    main()