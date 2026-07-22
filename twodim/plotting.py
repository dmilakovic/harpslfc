#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotting.py

Created on Thu Apr 24 14:19:12 2025

@author: dmilakov
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata # For 2D interpolation/gridding
import matplotlib.colors as colors # For MidpointNormalize
import warnings

def plot_gaussian_ellipse(ax, xo, yo, sigma_x, sigma_y, theta_rad, n_std=1.0, **kwargs):
    """Plots an ellipse representing the n_std contour of a 2D Gaussian."""
    width = 2 * n_std * sigma_x
    height = 2 * n_std * sigma_y
    angle_deg = np.degrees(theta_rad)
    if sigma_x <= 0 or sigma_y <= 0: return # Skip invalid
    ellipse_kwargs = {'edgecolor': 'red', 'facecolor': 'none', 'lw': 1}
    ellipse_kwargs.update(kwargs)
    ellipse = Ellipse(xy=(xo, yo), width=width, height=height, angle=angle_deg, **ellipse_kwargs)
    ax.add_patch(ellipse)

def plot_raw_data_stamps(processed_stamps_info, segment_id=None, filename=None):
    """Plots the raw data for each processed stamp with overlays in a grid layout."""
    num_plots = len(processed_stamps_info)
    if num_plots == 0:
        print("Plotting: No processed stamps with raw data info to plot.")
        return

    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig = plt.figure(figsize=(cols * 2.2 + 1.0, rows * 2.2))
    fig.suptitle(f"Raw 2D Gauss Fitted Stamps (Segment: {segment_id})", fontsize=14)
    gs = gridspec.GridSpec(rows, cols + 1, figure=fig, width_ratios=[1]*cols + [0.1], wspace=0.3, hspace=0.4)
    cax = fig.add_subplot(gs[:, cols])

    all_stamp_data = [info['stamp_data'] for info in processed_stamps_info]
    with warnings.catch_warnings(): # Suppress warnings for all-NaN slices if they occur
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vmin = np.nanmin([np.nanmin(s) for s in all_stamp_data if s is not None and s.size > 0]) if any(s is not None and s.size > 0 for s in all_stamp_data) else 0
        vmax = np.nanmax([np.nanmax(s) for s in all_stamp_data if s is not None and s.size > 0]) if any(s is not None and s.size > 0 for s in all_stamp_data) else 1
    if vmin == vmax: vmin -= 0.1; vmax += 0.1

    im = None
    for idx, info in enumerate(processed_stamps_info):
        r, c = idx // cols, idx % cols
        ax = fig.add_subplot(gs[r, c])
        stamp_data = info['stamp_data']
        if stamp_data is None: continue # Should not happen if list is filtered, but safety check
        xo, yo = info['xo_stamp'], info['yo_stamp']
        sx, sy = info['sigma_x'], info['sigma_y']
        theta = info['theta_rad']
        red_chi2 = info.get('chi2_reduced', np.nan)

        extent = [-0.5, stamp_data.shape[1] - 0.5, -0.5, stamp_data.shape[0] - 0.5]
        im = ax.imshow(stamp_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        ax.plot(xo, yo, '+', color='red', markersize=6, markeredgewidth=1.0)
        plot_gaussian_ellipse(ax, xo, yo, sx, sy, theta, n_std=1.0, edgecolor='red', ls='-', lw=0.8)
        plot_gaussian_ellipse(ax, xo, yo, sx, sy, theta, n_std=2.0, edgecolor='red', ls='--', lw=0.8)
        ax.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}", transform=ax.transAxes, color='white', backgroundcolor='black', fontsize=6, ha='right', va='top', alpha=0.7)
        ax.set_title(f"Peak {info['peak_index']}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

    for idx in range(num_plots, rows * cols):
        r, c = idx // cols, idx % cols
        try: fig.add_subplot(gs[r, c]).axis('off')
        except ValueError: pass

    if im: fig.colorbar(im, cax=cax, label="Raw Pixel Value")
    fig.subplots_adjust(top=0.92)

    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        except Exception as e: print(f"ERROR saving raw stamp plot to {filename}: {e}")
        finally: plt.close(fig)
    else: plt.show()

def plot_normalized_residuals_stamps(processed_stamps_info, segment_id=None, filename=None):
    """Plots the normalized residuals for each processed stamp in a grid layout."""
    num_plots = len(processed_stamps_info)
    if num_plots == 0:
        print("Plotting: No processed stamps with residual info to plot.")
        return

    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig = plt.figure(figsize=(cols * 2.2 + 1.0, rows * 2.2))
    fig.suptitle(f"Normalized Residuals (Segment: {segment_id}, {num_plots} plotted)", fontsize=14)
    gs = gridspec.GridSpec(rows, cols + 1, figure=fig, width_ratios=[1]*cols + [0.1], wspace=0.3, hspace=0.4)
    cax = fig.add_subplot(gs[:, cols])

    all_residuals = [info['residuals_norm_2d'] for info in processed_stamps_info]
    valid_residuals = [res for res in all_residuals if res is not None and res.size > 0 and np.all(np.isfinite(res))]
    if not valid_residuals:
         print("Plotting: No valid residuals found to determine color limits.")
         max_abs_res = 1.0
    else:
         with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=RuntimeWarning)
             max_abs_res = np.nanmax([np.nanmax(np.abs(res)) for res in valid_residuals])

    clim_val = min(max(max_abs_res, 1.0), 10.0) # Cap at +/- 10 sigma, ensure at least +/- 1
    vmin, vmax = -clim_val, clim_val

    im = None
    for idx, info in enumerate(processed_stamps_info):
        r, c = idx // cols, idx % cols
        ax = fig.add_subplot(gs[r, c])
        residuals_norm_2d = info.get('residuals_norm_2d')
        if residuals_norm_2d is None:
             ax.text(0.5, 0.5, 'Residuals\nN/A', ha='center', va='center', transform=ax.transAxes, fontsize=8)
             ax.set_title(f"Peak {info['peak_index']}", fontsize=7)
             ax.set_xticks([]); ax.set_yticks([])
             continue

        xo, yo = info['xo_stamp'], info['yo_stamp']
        sx, sy = info['sigma_x'], info['sigma_y']
        theta = info['theta_rad']
        red_chi2 = info.get('chi2_reduced', np.nan)

        extent = [-0.5, residuals_norm_2d.shape[1] - 0.5, -0.5, residuals_norm_2d.shape[0] - 0.5]
        im = ax.imshow(residuals_norm_2d, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        plot_gaussian_ellipse(ax, xo, yo, sx, sy, theta, n_std=1.0, edgecolor='black', ls='-', lw=0.6, alpha=0.6)
        plot_gaussian_ellipse(ax, xo, yo, sx, sy, theta, n_std=2.0, edgecolor='black', ls='--', lw=0.6, alpha=0.6)
        ax.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}", transform=ax.transAxes, color='black', backgroundcolor='white', fontsize=6, ha='right', va='top', alpha=0.8)
        ax.set_title(f"Peak {info['peak_index']}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

    for idx in range(num_plots, rows * cols):
        r, c = idx // cols, idx % cols
        try: fig.add_subplot(gs[r, c]).axis('off')
        except ValueError: pass

    if im: fig.colorbar(im, cax=cax, label="Normalized Residual ($\sigma$)")
    fig.subplots_adjust(top=0.92)

    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        except Exception as e: print(f"ERROR saving residual stamp plot to {filename}: {e}")
        finally: plt.close(fig)
    else: plt.show()


def plot_coefficient_vs_segment(results_list, zernike_indices, n, m, title=None, filename=None):
    """Plots a specific Zernike coefficient value (+/- error) against segment index."""
    if not zernike_indices: return # Cannot plot if indices unknown
    coeff_idx = -1
    try: coeff_idx = zernike_indices.index((n, m))
    except ValueError:
        print(f"Plotting: Coefficient Z({n},{m}) not found in indices. Cannot plot.")
        return

    segments, coeffs, errors = [], [], []
    results_list.sort(key=lambda r: r.get('segment_index', -1))

    for result in results_list:
        c = result.get('COEFFS')
        e = result.get('ERR_COEFFS')
        if c is not None and coeff_idx < len(c):
            segments.append(result.get('SEGMENT', len(segments))) # Use SEGMENT if available
            coeffs.append(c[coeff_idx])
            err_val = e[coeff_idx] if e is not None and coeff_idx < len(e) and np.isfinite(e[coeff_idx]) else 0
            errors.append(err_val)

    if not segments:
        print(f"Plotting: No valid data points found for coefficient Z({n},{m}) plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(segments, coeffs, yerr=errors, fmt='-o', capsize=5, markersize=5)
    ax.set_xlabel("Segment Index")
    ax.set_ylabel(f"Zernike Coefficient Z({n},{m}) Value")
    default_title = f"Zernike Coefficient Z({n},{m}) vs. Segment Index"
    ax.set_title(title if title else default_title)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Ensure integer ticks
    plt.tight_layout()

    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        except Exception as e: print(f"ERROR saving coeff vs seg plot to {filename}: {e}")
        finally: plt.close(fig)
    else: plt.show()

def plot_adjacent_segment_comparison(results_list, metric='rmse_diff', title=None, filename=None):
    """Compares coefficient vectors of adjacent segments."""
    if len(results_list) < 2: return
    results_list.sort(key=lambda r: r.get('SEGMENT', -1))
    pairs, values = [], []

    for i in range(len(results_list) - 1):
        res1, res2 = results_list[i], results_list[i+1]
        s1, s2 = res1.get('SEGMENT'), res2.get('SEGMENT')
        c1, c2 = res1.get('COEFFS'), res2.get('COEFFS')
        if s1 is None or s2 is None or c1 is None or c2 is None or s2 != s1 + 1: continue

        if metric == 'rmse_diff':
            value = np.sqrt(np.nanmean((c1 - c2)**2)) # Use nanmean
            ylabel = "RMSE Difference of Coefficients"
        elif metric == 'cosine_similarity':
            norm1, norm2 = np.linalg.norm(c1), np.linalg.norm(c2)
            value = np.dot(c1, c2) / (norm1 * norm2) if norm1 > 1e-9 and norm2 > 1e-9 else np.nan
            ylabel = "Cosine Similarity of Coefficients"
        else: print(f"Error: Unknown metric '{metric}'."); return

        if np.isfinite(value): pairs.append(f"{s1}-{s2}"); values.append(value)

    if not pairs: return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pairs, values, '-o', markersize=5)
    ax.set_xlabel("Adjacent Segment Pair Index")
    ax.set_ylabel(ylabel)
    default_title = f"{ylabel} vs. Segment Pair"
    ax.set_title(title if title else default_title)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        except Exception as e: print(f"ERROR saving adjacent seg plot to {filename}: {e}")
        finally: plt.close(fig)
    else: plt.show()


def plot_coefficient_heatmap_vs_segment(results_list, zernike_indices, value_type='coeff', title=None, filename=None):
    """Plots a heatmap showing Zernike coefficients variation across segments."""
    if not results_list or not zernike_indices: return
    results_list.sort(key=lambda r: r.get('SEGMENT', -1))
    segment_indices = [r.get('SEGMENT') for r in results_list]
    if any(s is None for s in segment_indices): return # Need segment index
    n_segments = len(segment_indices)
    n_coeffs = len(zernike_indices)
    data_matrix = np.full((n_coeffs, n_segments), np.nan) # Initialize with NaN

    for seg_idx, result in enumerate(results_list):
        coeffs = result.get('COEFFS')
        if coeffs is not None and len(coeffs) == n_coeffs:
            if value_type == 'abs': data_matrix[:, seg_idx] = np.abs(coeffs)
            elif value_type == 'sq': data_matrix[:, seg_idx] = coeffs**2
            else: data_matrix[:, seg_idx] = coeffs

    if np.all(np.isnan(data_matrix)): return
    fig, ax = plt.subplots(figsize=(max(6, n_segments * 0.5), max(6, n_coeffs * 0.25)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if value_type == 'coeff':
            cmap = 'coolwarm'; max_abs = np.nanmax(np.abs(data_matrix)); vmin, vmax = -max_abs, max_abs
            cbar_label = "Coefficient Value"
        else:
            cmap = 'viridis'; vmin = 0; vmax = np.nanmax(data_matrix)
            cbar_label = "Coefficient Abs Value" if value_type == 'abs' else "Coefficient Squared"
        if not np.isfinite(vmax) or vmax==vmin: vmin, vmax = 0, 1 # Fallback limits


    im = ax.imshow(data_matrix, aspect='auto', cmap=cmap, origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_xlabel("Segment Index"); ax.set_ylabel("Zernike Index (l)")
    ax.set_xticks(np.arange(n_segments)); ax.set_xticklabels(segment_indices)
    ytick_labels = [f"{l}\n({n},{m})" for l, (n, m) in enumerate(zernike_indices)]
    ax.set_yticks(np.arange(n_coeffs)); ax.set_yticklabels(ytick_labels, fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar_label, pad=0.02)
    default_title = f"Zernike Coeffs vs Segments ({value_type.capitalize()})"
    ax.set_title(title if title else default_title)
    plt.tight_layout()

    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        except Exception as e: print(f"ERROR saving heatmap plot to {filename}: {e}")
        finally: plt.close(fig)
    else: plt.show()
    
class MidpointNormalize(colors.Normalize):
    """ Normalize the threshold to the center """
    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Normalize values based on the midpoint.
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        # Ensure input is array and handle potential mask
        value_ma = np.ma.masked_array(value, np.isnan(value))
        if value_ma.mask.all(): # Handle case where all values are masked
            return value_ma
        return np.ma.masked_array(np.interp(value_ma, x, y), value_ma.mask)


def plot_coefficient_on_detector(results_table, zernike_indices, coeff_n, coeff_m,
                                 value_type='coeff', title=None, filename=None,
                                 img_shape=None, grid_resolution=(100,100)):
    """
    Plots the spatial variation of a specific Zernike coefficient across the detector.

    Uses the median X,Y position of the *original peaks* within each segment.

    Args:
        results_table (np.ndarray): Structured array read from the results FITS file.
                                    Must contain 'ORDER_NUM', 'IMGTYPE', 'SEGMENT',
                                    'COEFFS', and potentially 'X_STACK', 'Y_STACK' (or precomputed medians).
                                    It's better to add median positions during saving.
        zernike_indices (list): List of (n,m) Zernike indices corresponding to COEFFS column.
        coeff_n (int): Radial order (n) of the coefficient to plot.
        coeff_m (int): Azimuthal order (m) of the coefficient to plot.
        value_type (str): 'coeff', 'abs', or 'sq'. Defaults to 'coeff'.
        title (str, optional): Plot title.
        filename (str or Path, optional): File path to save plot. Shows if None.
        img_shape(tuple, optional): (height, width) of original detector for plot limits.
        grid_resolution(tuple): Resolution (nx, ny) for interpolation grid.
    """
    if results_table is None or len(results_table) == 0:
        print("Plotting: No results data provided for detector map.")
        return
    if not zernike_indices:
        print("Plotting: Zernike indices needed for detector map.")
        return

    # --- Find coefficient index ---
    coeff_idx = -1
    try: coeff_idx = zernike_indices.index((coeff_n, coeff_m))
    except ValueError:
        print(f"Plotting: Z({coeff_n},{coeff_m}) not found in indices.")
        return

    # --- Extract relevant data ---
    segment_x_median = []
    segment_y_median = []
    coeff_values = []
    valid_segment_mask = np.zeros(len(results_table), dtype=bool)

    # Check if median positions are already stored (RECOMMENDED)
    if 'MEDIAN_X' in results_table.dtype.names and 'MEDIAN_Y' in results_table.dtype.names:
        print("Plotting: Using pre-calculated MEDIAN_X, MEDIAN_Y columns.")
        segment_x_median = results_table['MEDIAN_X']
        segment_y_median = results_table['MEDIAN_Y']
        valid_position_mask = np.isfinite(segment_x_median) & np.isfinite(segment_y_median)
    elif 'X_STACK' in results_table.dtype.names and 'Y_STACK' in results_table.dtype.names:
         warnings.warn("Plotting: MEDIAN_X/Y not found. Calculating approximate medians from first/last pixel in X_STACK/Y_STACK - THIS IS INEFFICIENT AND APPROXIMATE. Add median calculation during saving.")
         # This is a fallback and likely inaccurate representation of segment center
         # It also requires reading potentially large variable arrays
         x_coords, y_coords = [], []
         for i in range(len(results_table)):
              x_pix = results_table['X_STACK'][i] # This is the DEROTATED data! NOT original pixels
              y_pix = results_table['Y_STACK'][i]
              if x_pix is not None and len(x_pix) > 0:
                   # Cannot easily get back original median from derotated stack
                   # Placeholder: use center of derotated coords - HIGHLY APPROXIMATE
                   x_coords.append(np.median(x_pix)) # Median of derotated X
                   y_coords.append(np.median(y_pix)) # Median of derotated Y
              else:
                   x_coords.append(np.nan)
                   y_coords.append(np.nan)
         segment_x_median = np.array(x_coords)
         segment_y_median = np.array(y_coords)
         valid_position_mask = np.isfinite(segment_x_median) & np.isfinite(segment_y_median)
         print("WARNING: Using approximate median positions derived from *derotated* stacked data. Results may be misleading.")
    else:
        print("Plotting Error: Cannot determine segment positions. Need 'MEDIAN_X'/'MEDIAN_Y' or 'X_STACK'/'Y_STACK' columns.")
        return


    # Extract coefficient values
    if 'COEFFS' in results_table.dtype.names:
         all_coeffs = results_table['COEFFS']
         # Ensure it's a 2D array even if only one row/coeff
         if all_coeffs.ndim == 1 and len(zernike_indices) > 0:
             all_coeffs = all_coeffs.reshape(-1, len(zernike_indices))

         if coeff_idx < all_coeffs.shape[1]:
             coeff_values = all_coeffs[:, coeff_idx]
             valid_coeff_mask = np.isfinite(coeff_values) & results_table['FIT_SUCCESS'] # Only use successful fits
         else: # Should not happen if index found, but safety check
             valid_coeff_mask = np.zeros(len(results_table), dtype=bool)
             coeff_values = np.full(len(results_table), np.nan)
    else:
        print("Plotting Error: 'COEFFS' column not found.")
        return

    # Combine masks
    valid_mask = valid_position_mask & valid_coeff_mask
    print(np.sum(valid_mask), len(valid_mask))
    if not np.any(valid_mask):
        print(f"Plotting: No valid data points found for Z({coeff_n},{coeff_m}).")
        return

    # Get valid data points
    x_valid = segment_x_median[valid_mask]
    y_valid = segment_y_median[valid_mask]
    c_valid = coeff_values[valid_mask]

    # Apply value transformation if needed
    if value_type == 'abs':
        plot_values = np.abs(c_valid)
        cbar_label = f"|Z({coeff_n},{coeff_m})|"
        cmap = 'viridis'
        norm = colors.Normalize(vmin=0, vmax=np.nanmax(plot_values) if np.any(plot_values) else 1)
    elif value_type == 'sq':
        plot_values = c_valid**2
        cbar_label = f"Z({coeff_n},{coeff_m})^2"
        cmap = 'viridis'
        norm = colors.Normalize(vmin=0, vmax=np.nanmax(plot_values) if np.any(plot_values) else 1)
    else: # 'coeff'
        plot_values = c_valid
        cbar_label = f"Z({coeff_n},{coeff_m})"
        cmap = 'coolwarm'
        max_abs = np.nanmax(np.abs(plot_values)) if np.any(np.abs(plot_values) > 1e-9) else 1.0
        # Use MidpointNormalize to center colormap at 0
        norm = MidpointNormalize(vmin=-max_abs, vmax=max_abs, midpoint=0)


    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 8)) # Square aspect often good for detector plots

    # Option 1: Scatter plot
    sc = ax.scatter(x_valid, y_valid, c=plot_values, s=50, 
                    cmap=cmap, norm=norm, edgecolors='k', linewidths=0.5,
                    linestyle='-')
    cbar = fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)

    # Option 2: Interpolated Grid (requires griddata) - uncomment to use
    # try:
    #     xi = np.linspace(np.min(x_valid), np.max(x_valid), grid_resolution[0])
    #     yi = np.linspace(np.min(y_valid), np.max(y_valid), grid_resolution[1])
    #     xi, yi = np.meshgrid(xi, yi)
    #     zi = griddata((x_valid, y_valid), plot_values, (xi, yi), method='linear') #'cubic', 'nearest'
    #     im = ax.imshow(zi, extent=(np.min(x_valid), np.max(x_valid), np.min(y_valid), np.max(y_valid)),
    #                    origin='lower', cmap=cmap, norm=norm, aspect='auto', interpolation='bilinear')
    #     cbar = fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
    #     # Overlay original points for context?
    #     ax.scatter(x_valid, y_valid, s=10, c='k', alpha=0.3)
    # except Exception as e:
    #     print(f"Warning: Interpolation failed ({e}), falling back to scatter plot.")
    #     sc = ax.scatter(x_valid, y_valid, c=plot_values, s=50, cmap=cmap, norm=norm, edgecolors='k', linewidths=0.5)
    #     cbar = fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)


    # --- Labels, Title, Limits ---
    ax.set_xlabel("Detector X pixel")
    ax.set_ylabel("Detector Y pixel")
    if img_shape:
         ax.set_xlim(0, img_shape[1])
         ax.set_ylim(0, img_shape[0])
         ax.set_aspect('equal', adjustable='box') # Maintain aspect ratio if shape known
    else:
         ax.set_aspect('equal', adjustable='datalim')


    default_title = f"Spatial Variation of Z({coeff_n},{coeff_m}) ({value_type.capitalize()})"
    ax.set_title(title if title else default_title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Save or Show ---
    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        except Exception as e: print(f"ERROR saving detector map plot to {filename}: {e}")
        finally: plt.close(fig)
    else: plt.show()
    
def plot_stamp_fit_overview(
    stamp_data,          # Original 2D stamp data
    model_mean_2d,       # Mean posterior model prediction (2D)
    mean_residuals_2d,   # Raw residuals (Data - Model) (2D)
    norm_mean_residuals_2d, # Normalized residuals (Data - Model) / Sigma (2D)
    fit_stats,           # Dictionary containing fit stats (e.g., red_chi2)
    posterior_samples,   # Dictionary of posterior samples (for dx, dy)
    stamp_coords,        # Tuple (grid_x, grid_y) for extent calculation
    title_prefix="",     # String to prepend to the main title
    filename=None
    ):
    """
    Plots a 4-panel overview of the stamp fit: Data, Model, Raw Residuals, Norm Residuals.
    """
    if stamp_data is None or model_mean_2d is None or mean_residuals_2d is None or norm_mean_residuals_2d is None:
        print("Plotting Error: Missing required data arrays for overview plot.")
        return
    if stamp_data.shape != model_mean_2d.shape:
         print("Plotting Error: Data and Model shape mismatch.")
         return

    grid_x, grid_y = stamp_coords
    # Ensure grid_x/y are numpy arrays for min/max
    grid_x_np = np.asarray(grid_x)
    grid_y_np = np.asarray(grid_y)
    extent = [-0.5 + grid_x_np.min(), 0.5 + grid_x_np.max(),
              -0.5 + grid_y_np.min(), 0.5 + grid_y_np.max()]

    # Get context info from posterior samples/fit_stats if available
    xo = float(np.mean(posterior_samples.get('dx', 0.0))) if posterior_samples else 0.0
    yo = float(np.mean(posterior_samples.get('dy', 0.0))) if posterior_samples else 0.0
    red_chi2 = fit_stats.get('red_chi2', np.nan) if fit_stats else np.nan

    # --- Create Figure (2x2 grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5), sharex=True, sharey=True)
    fig.suptitle(f"{title_prefix} Stamp Fit Overview", fontsize=14)

    # --- Determine common color limits for Data and Model ---
    with warnings.catch_warnings(): # Suppress warnings for all-NaN slices
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vmin_data = np.nanpercentile(stamp_data, 2)
        vmax_data = np.nanpercentile(stamp_data, 98)
        vmin_model = np.nanpercentile(model_mean_2d, 2)
        vmax_model = np.nanpercentile(model_mean_2d, 98)
    vmin = min(vmin_data, vmin_model)
    vmax = max(vmax_data, vmax_model)
    if not (np.isfinite(vmin) and np.isfinite(vmax)): vmin, vmax = 0, 1 # Fallback
    if vmin == vmax: vmin -= 0.1; vmax += 0.1

    # --- Panel 1: Data (Top-Left) ---
    ax = axes[0, 0]
    im1 = ax.imshow(stamp_data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
    ax.plot(xo, yo, '+', color='red', markersize=6, markeredgewidth=1.0, label='Fitted Center')
    ax.set_title("Data")
    ax.set_ylabel("Relative Y (pixels)")

    # --- Panel 2: Model (Top-Right) ---
    ax = axes[0, 1]
    im2 = ax.imshow(model_mean_2d, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
    ax.plot(xo, yo, '+', color='red', markersize=6, markeredgewidth=1.0)
    ax.set_title("Model (Mean Posterior)")
    # Add colorbar attached specifically to this axis, adjust fraction/pad
    # Use aspect to control height relative to axis, shrink controls overall size
    cbar_data = fig.colorbar(im2, ax=ax, shrink=0.80, aspect=20, pad=0.04, label="Data/Model Value")

    # --- Panel 3: Raw Residuals (Bottom-Left) ---
    ax = axes[1, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_abs_raw_res = np.nanmax(np.abs(mean_residuals_2d))
    clim_raw = max(max_abs_raw_res, 1e-9) # Avoid zero range if residuals are zero
    if not np.isfinite(clim_raw): clim_raw = 1.0 # Fallback if all NaN
    norm_raw = MidpointNormalize(vmin=-clim_raw, vmax=clim_raw, midpoint=0)
    im3 = ax.imshow(mean_residuals_2d, cmap='coolwarm', origin='lower', extent=extent, norm=norm_raw, interpolation='nearest')
    ax.plot(xo, yo, 'x', color='black', markersize=8, alpha=0.7)
    ax.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}", transform=ax.transAxes, color='black', backgroundcolor='white', fontsize=8, ha='right', va='top', alpha=0.8)
    ax.set_title("Raw Residuals (Data-Model)")
    ax.set_xlabel("Relative X (pixels)")
    ax.set_ylabel("Relative Y (pixels)")
    # Add colorbar attached specifically to this axis
    cbar_res = fig.colorbar(im3, ax=ax, shrink=0.80, aspect=20, pad=0.04, label="Residual Value")

    # --- Panel 4: Normalized Residuals (Bottom-Right) ---
    ax = axes[1, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_abs_norm_res = np.nanmax(np.abs(norm_mean_residuals_2d))
    clim_norm = min(max(max_abs_norm_res, 1.0), 5.0) # Cap plotting range (+/- 5 sigma)
    if not np.isfinite(clim_norm): clim_norm = 1.0 # Fallback
    norm_norm = MidpointNormalize(vmin=-clim_norm, vmax=clim_norm, midpoint=0)
    im4 = ax.imshow(norm_mean_residuals_2d, cmap='coolwarm', origin='lower', extent=extent, norm=norm_norm, interpolation='nearest')
    ax.plot(xo, yo, 'x', color='black', markersize=8, alpha=0.7)
    ax.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}", transform=ax.transAxes, color='black', backgroundcolor='white', fontsize=8, ha='right', va='top', alpha=0.8)
    ax.set_title("Normalized Residuals ($\sigma$)")
    ax.set_xlabel("Relative X (pixels)")
    # Add colorbar attached specifically to this axis
    cbar_norm = fig.colorbar(im4, ax=ax, shrink=0.80, aspect=20, pad=0.04, label="Norm Residual ($\sigma$)")

    # Apply aspect ratio AFTER plotting and colorbars
    for ax_row in axes:
        for ax_i in ax_row:
            ax_i.set_aspect('equal', adjustable='box')

    # Use constrained_layout first for better spacing attempt
    # try:
    #      plt.constrained_layout(pad=0.4, w_pad=0.5, h_pad=0.2)
    # except ValueError:
    #      # Fallback to tight_layout if constrained_layout fails (e.g., older matplotlib)
    #      print("Warning: constrained_layout failed, using tight_layout.")
    #      plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Adjust top slightly for suptitle if needed AFTER layout manager
    fig.subplots_adjust(top=0.93)


    # Save or Show
    if filename:
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Overview plot saved to {filename}")
        except Exception as e: print(f"ERROR saving overview plot: {e}")
        finally: plt.close(fig)
    else: plt.show()


# def plot_stamp_fit_overview(
#     stamp_data,          # Original 2D stamp data
#     model_mean_2d,       # Mean posterior model prediction (2D)
#     mean_residuals_2d,   # Raw residuals (Data - Model) (2D)
#     norm_mean_residuals_2d, # Normalized residuals (Data - Model) / Sigma (2D)
#     fit_stats,           # Dictionary containing fit stats (e.g., red_chi2)
#     posterior_samples,   # Dictionary of posterior samples (for dx, dy)
#     stamp_coords,        # Tuple (grid_x, grid_y) for extent calculation
#     title_prefix="",     # String to prepend to the main title
#     filename=None
#     ):
#     """
#     Plots a 4-panel overview of the stamp fit: Data, Model, Raw Residuals, Norm Residuals.
#     """
#     if stamp_data is None or model_mean_2d is None or mean_residuals_2d is None or norm_mean_residuals_2d is None:
#         print("Plotting Error: Missing required data arrays for overview plot.")
#         return
#     if stamp_data.shape != model_mean_2d.shape:
#          print("Plotting Error: Data and Model shape mismatch.")
#          return

#     grid_x, grid_y = stamp_coords
#     extent = [-0.5 + grid_x.min(), 0.5 + grid_x.max(),
#               -0.5 + grid_y.min(), 0.5 + grid_y.max()] # Use relative coords

#     # Get context info from posterior samples/fit_stats if available
#     xo = float(np.mean(posterior_samples.get('dx', 0.0))) if posterior_samples else 0.0
#     yo = float(np.mean(posterior_samples.get('dy', 0.0))) if posterior_samples else 0.0
#     red_chi2 = fit_stats.get('red_chi2', np.nan) if fit_stats else np.nan

#     # --- Create Figure (2x2 grid) ---
#     fig, axes = plt.subplots(2, 2, figsize=(9, 9.5), sharex=True, sharey=True) # Increased height slightly for title
#     fig.suptitle(f"{title_prefix} Stamp Fit Overview", fontsize=14)
#     axes_flat = axes.ravel() # Flatten for easier iteration if needed, but use 2D index

#     # --- Determine common color limits for Data and Model ---
#     with warnings.catch_warnings(): # Suppress warnings for all-NaN slices
#         warnings.simplefilter("ignore", category=RuntimeWarning)
#         # Use percentiles for robustness against single hot/cold pixels
#         vmin_data = np.nanpercentile(stamp_data, 2)
#         vmax_data = np.nanpercentile(stamp_data, 98)
#         vmin_model = np.nanpercentile(model_mean_2d, 2)
#         vmax_model = np.nanpercentile(model_mean_2d, 98)
#     # Use the wider range, ensuring non-zero span
#     vmin = min(vmin_data, vmin_model)
#     vmax = max(vmax_data, vmax_model)
#     if vmin == vmax: vmin -= 0.1; vmax += 0.1


#     # --- Panel 1: Data (Top-Left) ---
#     ax = axes[0, 0]
#     im1 = ax.imshow(stamp_data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, extent=extent)
#     ax.plot(xo, yo, '+', color='red', markersize=6, markeredgewidth=1.0, label='Fitted Center')
#     ax.set_title("Data")
#     ax.set_ylabel("Relative Y (pixels)")
#     # fig.colorbar(im1, ax=ax, shrink=0.7, label="Pixel Value") # Colorbar for all data/model later?

#     # --- Panel 2: Model (Top-Right) ---
#     ax = axes[0, 1]
#     im2 = ax.imshow(model_mean_2d, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, extent=extent)
#     ax.plot(xo, yo, '+', color='red', markersize=6, markeredgewidth=1.0)
#     ax.set_title("Model (Mean Posterior)")
#     # Add a single colorbar for data/model - place it next to this axis
#     cbar_data = fig.colorbar(im2, ax=axes[0,:].ravel().tolist(), shrink=0.7, label="Data/Model Value", pad=0.03)


#     # --- Panel 3: Raw Residuals (Bottom-Left) ---
#     ax = axes[1, 0]
#     max_abs_raw_res = np.nanmax(np.abs(mean_residuals_2d)); clim_raw = max(max_abs_raw_res, 1e-9)
#     norm_raw = MidpointNormalize(vmin=-clim_raw, vmax=clim_raw, midpoint=0)
#     im3 = ax.imshow(mean_residuals_2d, cmap='coolwarm', origin='lower', extent=extent, norm=norm_raw)
#     ax.plot(xo, yo, 'x', color='black', markersize=8, alpha=0.7)
#     ax.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}", transform=ax.transAxes, color='black', backgroundcolor='white', fontsize=8, ha='right', va='top', alpha=0.8)
#     ax.set_title("Raw Residuals (Data-Model)")
#     ax.set_xlabel("Relative X (pixels)")
#     ax.set_ylabel("Relative Y (pixels)")
#     # fig.colorbar(im3, ax=ax, shrink=0.7, label="Residual Value") # Separate colorbar below


#     # --- Panel 4: Normalized Residuals (Bottom-Right) ---
#     ax = axes[1, 1]
#     max_abs_norm_res = np.nanmax(np.abs(norm_mean_residuals_2d)); clim_norm = min(max(max_abs_norm_res, 1.0), 5.0)
#     norm_norm = MidpointNormalize(vmin=-clim_norm, vmax=clim_norm, midpoint=0)
#     im4 = ax.imshow(norm_mean_residuals_2d, cmap='coolwarm', origin='lower', extent=extent, norm=norm_norm)
#     ax.plot(xo, yo, 'x', color='black', markersize=8, alpha=0.7)
#     ax.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}", transform=ax.transAxes, color='black', backgroundcolor='white', fontsize=8, ha='right', va='top', alpha=0.8)
#     ax.set_title("Normalized Residuals ($\sigma$)")
#     ax.set_xlabel("Relative X (pixels)")
#     # Add separate colorbars for residuals, placed carefully
#     cbar_res = fig.colorbar(im3, ax=axes[1,0], shrink=0.7, label="Residual Value", location='bottom', pad=0.15)
#     cbar_norm = fig.colorbar(im4, ax=axes[1,1], shrink=0.7, label="Norm Residual ($\sigma$)", location='bottom', pad=0.15)


#     # Apply aspect ratio and maybe share limits after plotting
#     for ax_row in axes:
#         for ax_i in ax_row:
#             ax_i.set_aspect('equal', adjustable='box')

#     plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout for suptitle and bottom colorbars

#     # Save or Show
#     if filename:
#         try:
#             plt.savefig(filename, dpi=150, bbox_inches='tight')
#             print(f"Overview plot saved to {filename}")
#         except Exception as e: print(f"ERROR saving overview plot: {e}")
#         finally: plt.close(fig)
#     else: plt.show()
