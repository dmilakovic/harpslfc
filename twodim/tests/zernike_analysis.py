#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:12:16 2025

@author: dmilakov
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
from fitsio import FITS
import warnings
import itertools
import traceback
import time
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec # Import GridSpec


# Assuming zernike_fitter.py contains your ZernikeFitter class
try:
    from harps.twodim.zernike_fitter import ZernikeFitter
    ZERNPY_AVAILABLE = True
except ImportError:
    print("Warning: ZernikeFitter class not found. Zernike fitting part will be skipped.")
    print("Make sure zernike_fitter.py is in the same directory or your Python path.")
    ZERNPY_AVAILABLE = False
    class ZernikeFitter: # Dummy class if not available
          def __init__(self, *args, **kwargs): pass
          def fit(self, *args, **kwargs): print("ZernikeFitter not available."); return False
          def plot_fit_comparison(self, *args, **kwargs): pass
          def plot_fit_residuals(self, *args, **kwargs): pass


# --- Configuration ---
STAMP_HALF_WIDTH = 5   # +/- pixels around the peak to extract (total size = 2*HALF_WIDTH + 1)
PEAK_MIN_DISTANCE = 5 # min_distance for peak_local_max
CLUSTER_EPS_X = 0.004  # DBSCAN epsilon for normalized X (adjust based on peak density/spacing)
CLUSTER_EPS_Y = 0.008 # DBSCAN epsilon for normalized Y (smaller for tight horizontal lines)
CLUSTER_MIN_SAMPLES = 5 # DBSCAN min_samples
GAUSSIAN_FIT_THRESHOLD_SNR = 5.0 # Minimum required Amplitude/Noise ratio for a successful Gaussian fit

# --- Helper Functions (Partly from your code) ---

def load_echelle_data(fits_path, bias_path=None, hdu_index=1):
    """Loads FITS data, optionally bias subtracting."""
    fits_file = Path(fits_path)
    if not fits_file.is_file():
        raise FileNotFoundError(f"Data file not found: {fits_file}")

    with FITS(fits_file, 'r') as hdul:
        data = hdul[hdu_index].read() # Assuming data is in the second HDU (index 1)

    if bias_path:
        bias_file = Path(bias_path)
        if not bias_file.is_file():
            warnings.warn(f"Bias file not found: {bias_file}. Proceeding without bias subtraction.")
            bias = 0
        else:
             with FITS(bias_file, 'r') as hdul_b:
                 # Try matching HDU index, fallback to 1 or 0 if needed
                 try:
                     bias = hdul_b[hdu_index].read()
                 except IndexError:
                     warnings.warn(f"Bias HDU index {hdu_index} not found. Trying index 1.")
                     try:
                         bias = hdul_b[1].read()
                     except IndexError:
                          warnings.warn("Bias HDU index 1 not found. Trying index 0.")
                          bias = hdul_b[0].read()

                 if bias.shape != data.shape:
                     warnings.warn(f"Bias shape {bias.shape} does not match data shape {data.shape}. Skipping bias subtraction.")
                     bias = 0

        data = data.astype(float) - bias.astype(float) # Ensure float for subtraction
    else:
        data = data.astype(float) # Ensure float anyway

    print(f"Loaded data from {fits_file}, shape: {data.shape}")
    return data

def find_peaks(image_data, min_distance, threshold_abs=None, threshold_rel=None,
               hard_cut = 320):
    """Finds local peaks in the image data."""
    local_image_data = np.copy(image_data)
    if threshold_abs is None and threshold_rel is None:
         # Default threshold: slightly above median
         threshold_abs = np.median(local_image_data) + 1 * np.std(local_image_data)
         print(f"Using default absolute threshold for peak finding: {threshold_abs:.2f}")
    
    if hard_cut is not None:
        cut = np.where((local_image_data <= hard_cut))
        local_image_data[cut]=0.
    
    coordinates = peak_local_max(
        local_image_data,
        min_distance=min_distance,
        # threshold_abs=threshold_abs,
        # threshold_rel=threshold_rel,
        # exclude_border=True # Important to avoid issues with stamp extraction
    )
    # peak_local_max returns (row, col) which corresponds to (y, x)
    # Swap columns to get (x, y) convention
    coordinates_xy = coordinates[:, ::-1]
    print(f"Found {len(coordinates_xy)} peaks.")
    return coordinates_xy

def normalize_coordinates(points_int, shape):
    """Normalizes integer pixel coordinates to [0, 1] range."""
    points_flt = points_int.astype(float)
    points_flt[:, 0] /= shape[1]  # Normalize x by width
    points_flt[:, 1] /= shape[0]  # Normalize y by height
    return points_flt

def inverse_normalize_coordinates(points_flt, shape):
    """Converts normalized coordinates back to integer pixels."""
    points_int = np.zeros_like(points_flt, dtype=int)
    points_int[:, 0] = np.round(points_flt[:, 0] * shape[1]).astype(int)
    points_int[:, 1] = np.round(points_flt[:, 1] * shape[0]).astype(int)
    return points_int

def cluster_peaks_to_orders(peaks_xy, image_shape, eps_x, eps_y, min_samples):
    """Clusters peaks into orders using DBSCAN and identifies A/B pairs."""
    if len(peaks_xy) == 0:
        return [], {}

    points_normalized = normalize_coordinates(peaks_xy, image_shape)

    # Scale normalized coordinates by inverse epsilon for DBSCAN
    points_scaled = np.copy(points_normalized)
    points_scaled[:, 0] /= eps_x
    points_scaled[:, 1] /= eps_y

    # Apply DBSCAN
    clustering = DBSCAN(eps=1, min_samples=min_samples).fit(points_scaled)
    labels = clustering.labels_

    # Extract clusters (potential orders) and sort points within each cluster by Y
    unique_labels = sorted(list(set(labels) - {-1})) # Ignore noise (-1)
    raw_orders = []
    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_points_xy = peaks_xy[cluster_mask]
        # Sort by x-coordinate
        cluster_points_sorted = cluster_points_xy[np.argsort(cluster_points_xy[:, 1])]
        raw_orders.append(cluster_points_sorted)

    if not raw_orders:
        print("No clusters found.")
        return [], {}

    # --- Identify A/B pairs ---
    # Calculate average X for each raw order
    avg_x_coords = [np.mean(order[:, 0]) for order in raw_orders]
    # Sort orders by their average Y coordinate
    sorted_order_indices = np.argsort(avg_x_coords)

    paired_orders = {}
    order_counter = 0
    paired_status = [False] * len(raw_orders)

    print(f"Found {len(raw_orders)} raw orders. Attempting to pair A/B...")

    # Define a threshold for pairing based on typical Y separation
    # This might need tuning based on your specific instrument/data
    # Let's use the median difference between adjacent sorted orders as a guide
    if len(sorted_order_indices) > 1:
        x_diffs = np.diff(np.array(avg_x_coords)[sorted_order_indices])
        # Filter out large gaps, focus on small separations typical of A/B pairs
        # typical_x_sep = np.median(x_diffs[x_diffs < np.percentile(x_diffs, 75)]) # Heuristic
        typical_x_sep = np.median(x_diffs)
        # More robust: Use a fraction of cluster_eps_y in pixel space
        pairing_x_threshold = (CLUSTER_EPS_X * 5) * image_shape[1] # Tolerate 1.5x eps_y separation
        # plt.figure()
        # plt.hist(x_diffs,bins=20)
        # plt.axvline(typical_x_sep,c='r')
        # plt.axvline(pairing_x_threshold,c='b',ls=':')
        # plt.show()
        print(f"Using X pairing threshold: {pairing_x_threshold:.2f} pixels (based on eps_x)")
    else:
        pairing_x_threshold = np.inf # Cannot pair if only one order


    for i in range(len(sorted_order_indices)):
        if paired_status[sorted_order_indices[i]]:
            continue # Already paired

        current_order_idx = sorted_order_indices[i]
        current_order_x = avg_x_coords[current_order_idx]
        order_counter += 1 # Assign a new order number

        # Look for the next unpaired order below it
        found_pair = False
        if i + 1 < len(sorted_order_indices):
            next_order_idx = sorted_order_indices[i+1]
            if not paired_status[next_order_idx]:
                next_order_x = avg_x_coords[next_order_idx]

                # Check if they are close enough in Y to be a pair
                if abs(next_order_x - current_order_x) < pairing_x_threshold:
                    # Pair found! Assign A and B based on Y coordinate
                    # Assuming A is typically above B (lower Y index)
                    paired_orders[order_counter] = {
                        'B': raw_orders[current_order_idx],
                        'A': raw_orders[next_order_idx]
                    }
                    paired_status[current_order_idx] = True
                    paired_status[next_order_idx] = True
                    found_pair = True
                    print(f"  Paired Order {order_counter}: Raw indices {current_order_idx}(A) and {next_order_idx}(B)")


        # If no pair was found below, it's a single order (or the last one)
        if not found_pair:
             paired_orders[order_counter] = {
                 'B': raw_orders[current_order_idx],
                 'A': None # Mark B as missing
             }
             paired_status[current_order_idx] = True
             print(f"  Assigned Order {order_counter}: Raw index {current_order_idx}(A), no B pair found.")


    print(f"Clustering complete. Identified {len(paired_orders)} potential echelle orders (some might be single A/B).")
    return raw_orders, paired_orders # Return both raw and paired for inspection

# --- Gaussian Fitting Function (from your code, slightly modified) ---

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta_rad, offset):
    """
    2D Gaussian function for fitting.
    theta_rad: rotation angle in radians.
    """
    (x, y) = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta_rad)**2)/(2*sigma_x**2) + (np.sin(theta_rad)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta_rad))/(4*sigma_x**2) + (np.sin(2*theta_rad))/(4*sigma_y**2)
    c = (np.sin(theta_rad)**2)/(2*sigma_x**2) + (np.cos(theta_rad)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()



def plot_gaussian_ellipse(ax, xo, yo, sigma_x, sigma_y, theta_rad, n_std=1.0, **kwargs):
    """
    Plots an ellipse representing the n_std contour of a 2D Gaussian.

    Args:
        ax: The matplotlib axes to plot on.
        xo, yo: Center of the Gaussian.
        sigma_x, sigma_y: Standard deviations along the principal axes.
        theta_rad: Rotation angle of the ellipse in radians.
        n_std: The number of standard deviations for the contour (e.g., 1.0 for 1-sigma).
        **kwargs: Additional keyword arguments passed to matplotlib.patches.Ellipse
                  (e.g., color, lw, ls).
    """
    # Width and height of the ellipse correspond to 2 * n_std * sigma
    width = 2 * n_std * sigma_x
    height = 2 * n_std * sigma_y
    angle_deg = np.degrees(theta_rad) # Ellipse angle expects degrees

    # Ensure sigmas are positive for valid ellipse
    if sigma_x <= 0 or sigma_y <= 0:
        print(f"Warning: Non-positive sigma ({sigma_x}, {sigma_y}) for ellipse plot. Skipping ellipse.")
        return

    # Default ellipse style
    ellipse_kwargs = {'edgecolor': 'red', 'facecolor': 'none', 'lw': 1}
    ellipse_kwargs.update(kwargs) # Override defaults if provided

    ellipse = Ellipse(xy=(xo, yo), width=width, height=height, angle=angle_deg,
                      **ellipse_kwargs)
    ax.add_patch(ellipse)
    

def plot_normalized_residuals_stamps(processed_stamps_info, segment_id=None, filename=None):
    """
    Plots the normalized residuals for each processed stamp in a grid layout.

    Args:
        processed_stamps_info (list): List of dictionaries, each containing info
                                      about a successfully fitted stamp, including
                                      'residuals_norm_2d', 'xo_stamp', 'yo_stamp',
                                      'sigma_x', 'sigma_y', 'theta_rad',
                                      'chi2_reduced', 'peak_index'.
        segment_id (int | str | None): Identifier for the current segment (for plot title).
    """
    num_plots = len(processed_stamps_info)
    if num_plots == 0:
        print("No processed stamps with residual info to plot.")
        return

    # Determine grid size for stamps (try to make it square)
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    # Create figure
    fig = plt.figure(figsize=(cols * 2.2 + 1.0, rows * 2.2)) # Adjust size + space for cbar
    fig.suptitle(f"Normalized Residuals (Segment: {segment_id}, {num_plots} plotted)", fontsize=14)

    # Use GridSpec for layout
    gs = gridspec.GridSpec(rows, cols + 1, figure=fig,
                           width_ratios=[1] * cols + [0.1], # Width ratio for cbar column
                           wspace=0.3, hspace=0.4) # Adjust spacing

    # Create the colorbar axis
    cax = fig.add_subplot(gs[:, cols]) # Span all rows in the last column

    # Determine shared symmetric color limits based on normalized residuals
    all_residuals = [info['residuals_norm_2d'] for info in processed_stamps_info]
    max_abs_res = np.max([np.max(np.abs(res)) for res in all_residuals if res is not None])
    # Set symmetric limits, e.g., +/- 3 sigma or based on max absolute value
    clim_val = min(max(max_abs_res, 1.0), 10.0) # Cap at +/- 5 sigma, ensure at least +/- 1
    vmin, vmax = -clim_val, clim_val

    im = None # To hold the last AxesImage for colorbar

    # Plot each residual stamp in the main grid
    for idx, info in enumerate(processed_stamps_info):
        # Calculate subplot row and column
        r = idx // cols
        c = idx % cols
        ax = fig.add_subplot(gs[r, c]) # Add subplot using GridSpec indices

        residuals_norm_2d = info.get('residuals_norm_2d') # Use .get for safety
        if residuals_norm_2d is None:
             ax.text(0.5, 0.5, 'Residuals\nN/A', ha='center', va='center', transform=ax.transAxes)
             ax.set_xticks([])
             ax.set_yticks([])
             continue # Skip if residuals aren't available

        xo, yo = info['xo_stamp'], info['yo_stamp']
        sx, sy = info['sigma_x'], info['sigma_y']
        theta = info['theta_rad']
        red_chi2 = info['chi2_reduced']

        # Display the normalized residuals
        extent = [-0.5, residuals_norm_2d.shape[1] - 0.5, -0.5, residuals_norm_2d.shape[0] - 0.5]
        # Use a diverging colormap centered at 0
        im = ax.imshow(residuals_norm_2d, cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax, extent=extent)

        # Plot fitted center (optional, maybe distracting on residuals)
        # ax.plot(xo, yo, '+', color='black', markersize=6, markeredgewidth=1.0, alpha=0.5)

        # Plot 1-sigma ellipse outline for context (optional)
        plot_gaussian_ellipse(ax, xo, yo, sx, sy, theta, n_std=1.0, edgecolor='black', ls='-', lw=1, alpha=0.6)
        plot_gaussian_ellipse(ax, xo, yo, sx, sy, theta, n_std=2.0, edgecolor='black', ls='--', lw=1, alpha=0.6)


        # Add reduced chi-squared text
        ax.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}",
                transform=ax.transAxes, color='black', backgroundcolor='white',# Better visibility
                fontsize=6, ha='right', va='top', alpha=0.8)
        ax.set_title(f"Peak {info['peak_index']}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

    # Turn off unused subplots
    for idx in range(num_plots, rows * cols):
        r = idx // cols
        c = idx % cols
        try:
             ax_to_turn_off = fig.add_subplot(gs[r, c])
             ax_to_turn_off.axis('off')
        except ValueError: pass

    # Add shared colorbar to the dedicated axis
    if im:
        fig.colorbar(im, cax=cax, label="Normalized Residual ($\sigma$)")

    fig.subplots_adjust(top=0.92)
    if filename:
        try: # Add try-except for saving robustness
            plt.savefig(filename, dpi=300, bbox_inches='tight') # Use tight bbox
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"ERROR saving plot to {filename}: {e}")
        finally:
            plt.close(fig) # Close the figure after attempting to save
    else:
        plt.show()
    
    
    
# --- Main Processing Function ---

# Assume previous imports and helper functions (plot_gaussian_ellipse, twoD_Gaussian) are present

def process_echelle_order(selected_peaks_xy, image_data, stamp_half_width,
                          fit_threshold_snr, plot_stamps=False, segment_id=None,
                          stamps_save_path=None, residuals_save_path=None):
    """
    Processes peaks from a selected order/segment: extracts stamps, fits Gaussians,
    normalizes, calculates DEROTATED centered coordinates, and stacks results.
    Optionally plots original stamps and normalized residual stamps.

    Args:
        selected_peaks_xy (np.ndarray): Array of (x, y) peak coordinates.
        image_data (np.ndarray): The full 2D image data.
        stamp_half_width (int): Half-width of the square stamp to extract.
        fit_threshold_snr (float): Minimum Amplitude/Error ratio for Gaussian fit acceptance.
        plot_stamps (bool): If True, generates plots for original stamps and residuals.
        segment_id (int | str | None): Identifier for the current segment.

    Returns:
        tuple: (X_stack, Y_stack, Z_stack, fit_details)
               Stacked 1D arrays of DEROTATED centered X, Y coords and normalized Z flux.
               fit_details: List of dictionaries containing fit results.
               Returns (None, None, None, []) if processing fails.
    """
    all_rel_x_rotated_list = []
    all_rel_y_rotated_list = []
    all_norm_z_list = []
    fit_details = []
    processed_stamps_info_for_plotting = [] # Store info ONLY if plot_stamps is True
    img_h, img_w = image_data.shape

    print(f"\nProcessing {len(selected_peaks_xy)} peaks for segment {segment_id}...")

    for i, (peak_x, peak_y) in enumerate(selected_peaks_xy):
        # --- Define stamp boundaries ---
        x_min = max(0, peak_x - stamp_half_width)
        x_max = min(img_w, peak_x + stamp_half_width + 1)
        y_min = max(0, peak_y - stamp_half_width)
        y_max = min(img_h, peak_y + stamp_half_width + 1)

        # --- Check stamp validity ---
        if (x_max - x_min) != (2 * stamp_half_width + 1) or \
           (y_max - y_min) != (2 * stamp_half_width + 1):
             # No print here, handled later if needed
             continue

        # --- Extract raw stamp data ---
        stamp_data = image_data[y_min:y_max, x_min:x_max]

        # --- Normalization ---
        total_flux = np.sum(stamp_data)
        if total_flux <= 1e-9:
            continue
        stamp_norm = stamp_data / total_flux

        # --- Gaussian Fitting ---
        stamp_h, stamp_w = stamp_norm.shape
        y_stamp_grid, x_stamp_grid = np.meshgrid(np.arange(stamp_h), np.arange(stamp_w), indexing='ij')
        initial_guess = (np.max(stamp_norm), stamp_w/2.0, stamp_h/2.0,
                         stamp_half_width/3.0, stamp_half_width/3.0, 0.0, np.min(stamp_norm))
        bounds = ([0, -stamp_w, -stamp_h, 0.1, 0.1, -np.pi, -np.inf],
                  [np.inf, 2*stamp_w, 2*stamp_h, stamp_w, stamp_h, np.pi, np.inf])

        try:
            popt, pcov = curve_fit(twoD_Gaussian, (x_stamp_grid, y_stamp_grid),
                                   stamp_norm.ravel(), p0=initial_guess, bounds=bounds, maxfev=5000)

            if not np.all(np.isfinite(pcov)): raise RuntimeError("Non-finite covariance")
            perr = np.sqrt(np.diag(pcov))
            amplitude_snr = popt[0] / perr[0] if perr[0] > 1e-12 else np.inf

            if amplitude_snr < fit_threshold_snr or not np.all(np.isfinite(popt)) \
               or popt[3] <= 0 or popt[4] <= 0:
                 continue

            # --- Store Successful Fit Results ---
            fit_result_dict = { # Store essential results for everyone
                'peak_xy': (peak_x, peak_y), 'params': popt.tolist(), 'errors': perr.tolist(),
                'param_names': ['amplitude', 'xo_stamp', 'yo_stamp', 'sigma_x', 'sigma_y', 'theta_rad', 'offset'],
                'total_flux_raw': total_flux, 'amplitude_snr': amplitude_snr
            }
            fit_details.append(fit_result_dict)

            # --- Calculate Derotated Coordinates ---
            amplitude_norm, fit_center_x_stamp, fit_center_y_stamp, sigma_x_fit, sigma_y_fit, theta_fit, offset_norm = popt
            fit_center_x_abs = x_min + fit_center_x_stamp
            fit_center_y_abs = y_min + fit_center_y_stamp
            y_abs_grid, x_abs_grid = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
            rel_x = x_abs_grid - fit_center_x_abs
            rel_y = y_abs_grid - fit_center_y_abs
            if sigma_x_fit >= sigma_y_fit: theta_major_axis = theta_fit
            else: theta_major_axis = theta_fit + np.pi / 2.0
            angle_to_rotate_by = -theta_major_axis
            cos_rot, sin_rot = np.cos(angle_to_rotate_by), np.sin(angle_to_rotate_by)
            rel_x_rotated = rel_x * cos_rot + rel_y * sin_rot
            rel_y_rotated = -rel_x * sin_rot + rel_y * cos_rot

            all_rel_x_rotated_list.append(rel_x_rotated.ravel())
            all_rel_y_rotated_list.append(rel_y_rotated.ravel())
            all_norm_z_list.append(stamp_norm.ravel())

            # --- Calculate Residuals and Store Plotting Info IF needed ---
            if plot_stamps:
                # Calculate model and residuals
                model_norm = twoD_Gaussian((x_stamp_grid, y_stamp_grid), *popt)
                residuals_norm_1d = stamp_norm.ravel() - model_norm
                variance_norm = np.maximum(stamp_data.ravel(), 1e-9) / (total_flux**2)
                # Normalized residuals = (data - model) / sqrt(variance)
                residuals_div_sigma = residuals_norm_1d / np.sqrt(variance_norm)
                chi2_val = np.sum(residuals_div_sigma**2) # Use residual/sigma for chi2
                dof = len(stamp_norm.ravel()) - len(popt)
                red_chi2 = chi2_val / dof if dof > 0 else np.inf

                # Store info required by BOTH plotting functions
                processed_stamps_info_for_plotting.append({
                     'stamp_data': stamp_data,          # For raw plot
                     'residuals_norm_2d': residuals_div_sigma.reshape(stamp_h, stamp_w), # For residual plot
                     'xo_stamp': fit_center_x_stamp,   # For both
                     'yo_stamp': fit_center_y_stamp,   # For both
                     'sigma_x': sigma_x_fit,           # For both
                     'sigma_y': sigma_y_fit,           # For both
                     'theta_rad': theta_fit,           # For both
                     'chi2_reduced': red_chi2,         # For both
                     'peak_index': i                   # For both
                 })

        # --- Handle Fit Errors ---
        except (RuntimeError, ValueError) as e: # Catch fit failures
             # No print here, summary printed later if needed
             continue
        except Exception as e: # Catch unexpected errors
             print(f"  Peak {i} ({peak_x}, {peak_y}): Unexpected error: {e}")
             traceback.print_exc()
             continue

    # --- Summary Messages ---
    num_processed = len(fit_details)
    num_attempted = len(selected_peaks_xy)
    num_skipped = num_attempted - num_processed
    print(f"Segment {segment_id}: Attempted {num_attempted} peaks. Successfully processed {num_processed}. Skipped {num_skipped}.")


    # --- Plotting Section (Calls external functions with save paths) ---
    if plot_stamps and processed_stamps_info_for_plotting:
        # Determine if showing or saving based on provided paths
        show_plots = stamps_save_path is None and residuals_save_path is None
        if not show_plots:
             print(f"Segment {segment_id}: Saving plots for {len(processed_stamps_info_for_plotting)} stamps...")
        else:
             print(f"Segment {segment_id}: Generating plots for {len(processed_stamps_info_for_plotting)} stamps...")

        # --- Plot 1: Raw Data Stamps ---
        try:
            plot_raw_data_stamps(processed_stamps_info_for_plotting,
                                 segment_id=segment_id,
                                 filename=stamps_save_path) # Pass save path
        except Exception as e:
             print(f"Error generating/saving raw data stamp plot: {e}")
             traceback.print_exc()

        # --- Plot 2: Normalized Residual Stamps ---
        try:
            plot_normalized_residuals_stamps(processed_stamps_info_for_plotting,
                                             segment_id=segment_id,
                                             filename=residuals_save_path) # Pass save path
        except Exception as e:
             print(f"Error generating/saving residual stamp plot: {e}")
             traceback.print_exc()

    elif plot_stamps and not processed_stamps_info_for_plotting:
         print(f"Segment {segment_id}: Plotting requested, but no stamps were successfully processed for plotting.")



    # --- Final Stacking and Return ---
    if not all_rel_x_rotated_list:
        # This message now accurately reflects that no fits succeeded for stacking
        # print(f"Segment {segment_id}: No valid peaks processed successfully for stacking.")
        return None, None, None, [] # Return empty results consistent with fit_details

    X_stack = np.concatenate(all_rel_x_rotated_list)
    Y_stack = np.concatenate(all_rel_y_rotated_list)
    Z_stack = np.concatenate(all_norm_z_list)

    print(f"Stacked data shape (derotated): X={X_stack.shape}, Y={Y_stack.shape}, Z={Z_stack.shape}")

    return X_stack, Y_stack, Z_stack, fit_details


# --- IMPORTANT: Define the Raw Data Plotting Function ---
# You need to take the plotting logic from the previous version of process_echelle_order
# and put it into its own function, similar to plot_normalized_residuals_stamps.
# Let's call it plot_raw_data_stamps:

def plot_raw_data_stamps(processed_stamps_info, segment_id=None, filename=None):
    """
    Plots the raw data for each processed stamp with overlays in a grid layout.
    (Based on the plotting logic from the previous version)
    """
    num_plots = len(processed_stamps_info)
    if num_plots == 0:
        print("No processed stamps with raw data info to plot.")
        return

    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig = plt.figure(figsize=(cols * 2.2 + 1.0, rows * 2.2))
    fig.suptitle(f"Raw 2D Gauss Fitted Stamps (Segment: {segment_id})", fontsize=14)
    gs = gridspec.GridSpec(rows, cols + 1, figure=fig,
                           width_ratios=[1] * cols + [0.1], wspace=0.3, hspace=0.4)
    cax = fig.add_subplot(gs[:, cols])

    all_stamp_data = [info['stamp_data'] for info in processed_stamps_info]
    vmin = np.min([np.min(s) for s in all_stamp_data])
    vmax = np.max([np.max(s) for s in all_stamp_data])
    if vmin == vmax: vmin -= 0.1; vmax += 0.1

    im = None
    for idx, info in enumerate(processed_stamps_info):
        r, c = idx // cols, idx % cols
        ax = fig.add_subplot(gs[r, c])

        stamp_data = info['stamp_data']
        xo, yo = info['xo_stamp'], info['yo_stamp']
        sx, sy = info['sigma_x'], info['sigma_y']
        theta = info['theta_rad']
        red_chi2 = info.get('chi2_reduced', np.nan) # Get chi2 if available

        extent = [-0.5, stamp_data.shape[1] - 0.5, -0.5, stamp_data.shape[0] - 0.5]
        im = ax.imshow(stamp_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, extent=extent) # gray cmap
        ax.plot(xo, yo, '+', color='red', markersize=6, markeredgewidth=1.0)
        plot_gaussian_ellipse(ax, xo, yo, sx, sy, theta, n_std=1.0, edgecolor='red', ls='-', lw=0.8)
        plot_gaussian_ellipse(ax, xo, yo, sx, sy, theta, n_std=2.0, edgecolor='red', ls='--', lw=0.8)

        ax.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}",
                transform=ax.transAxes, color='white', backgroundcolor='black',
                fontsize=6, ha='right', va='top', alpha=0.7)
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
        try: # Add try-except for saving robustness
            plt.savefig(filename, dpi=300, bbox_inches='tight') # Use tight bbox
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"ERROR saving plot to {filename}: {e}")
        finally:
            plt.close(fig) # Close the figure after attempting to save
    else:
        plt.show()
    
    
def find_coeff_index(zernike_indices, n, m):
    """Finds the index of a specific Zernike term (n, m) in the list."""
    try:
        return zernike_indices.index((n, m))
    except ValueError:
        return -1 # Not found

def plot_coefficient_vs_segment(results_list, zernike_indices, n, m, title=None,
                                filename=None):
    """
    Plots a specific Zernike coefficient value (+/- error) against the segment index.

    Args:
        results_list (list): List of result dictionaries from successful segment fits.
                             Each dict must contain 'segment_index', 'fitted_coeffs', 'err_coeffs'.
        zernike_indices (list): The list of (n, m) tuples corresponding to the coefficients.
        n (int): Radial order of the coefficient to plot.
        m (int): Azimuthal order of the coefficient to plot.
        title (str, optional): Title for the plot.
    """
    coeff_idx = find_coeff_index(zernike_indices, n, m)
    if coeff_idx == -1:
        print(f"Warning: Coefficient Z(n={n}, m={m}) not found in the fitted indices. Cannot plot.")
        return

    segments = []
    coeffs = []
    errors = []

    # Sort results by segment index just in case
    results_list.sort(key=lambda r: r['segment_index'])

    for result in results_list:
        segments.append(result['segment_index'])
        coeffs.append(result['fitted_coeffs'][coeff_idx])
        # Handle potential NaN or missing errors
        err = result.get('err_coeffs')
        if err is not None and coeff_idx < len(err) and np.isfinite(err[coeff_idx]):
             errors.append(err[coeff_idx])
        else:
             errors.append(0) # Plot without error bar if error is invalid

    if not segments:
        print(f"No data points found for coefficient Z(n={n}, m={m}) plot.")
        return

    fig = plt.figure(figsize=(10, 5))
    plt.errorbar(segments, coeffs, yerr=errors, fmt='-o', capsize=5, markersize=5)
    plt.xlabel("Segment Index (Approx. along X)")
    plt.ylabel(f"Zernike Coefficient Z(n={n}, m={m}) Value")
    if title is None:
        title = f"Zernike Coefficient Z(n={n}, m={m}) vs. Segment Index"
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(segments) # Ensure ticks are at segment indices
    plt.tight_layout()
    if filename:
        try: # Add try-except for saving robustness
            plt.savefig(filename, dpi=300, bbox_inches='tight') # Use tight bbox
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"ERROR saving plot to {filename}: {e}")
        finally:
            plt.close(fig) # Close the figure after attempting to save
    else:
        plt.show()


def plot_adjacent_segment_comparison(results_list, metric='rmse_diff', title=None,
                                     filename=None):
    """
    Compares coefficient vectors of adjacent segments using a specified metric.

    Args:
        results_list (list): List of result dictionaries from successful segment fits.
                             Each dict must contain 'segment_index', 'fitted_coeffs'.
        metric (str): Comparison metric ('rmse_diff' or 'cosine_similarity').
        title (str, optional): Title for the plot.
    """
    if len(results_list) < 2:
        print("Need at least two successful segment fits to compare adjacent segments.")
        return

    # Sort results by segment index
    results_list.sort(key=lambda r: r['segment_index'])

    segment_pairs = [] # e.g., "0-1", "1-2"
    comparison_values = []

    for i in range(len(results_list) - 1):
        res1 = results_list[i]
        res2 = results_list[i+1]

        # Check if segment indices are indeed adjacent
        if res2['segment_index'] != res1['segment_index'] + 1:
             print(f"Warning: Segments {res1['segment_index']} and {res2['segment_index']} are not adjacent. Skipping comparison.")
             continue

        coeffs1 = res1['fitted_coeffs']
        coeffs2 = res2['fitted_coeffs']

        if metric == 'rmse_diff':
            # Calculate Root Mean Square Difference between coefficient vectors
            value = np.sqrt(np.mean((coeffs1 - coeffs2)**2))
            ylabel = "RMSE Difference of Coefficients"
        elif metric == 'cosine_similarity':
            # Calculate Cosine Similarity (1 = identical direction, 0 = orthogonal, -1 = opposite)
            norm1 = np.linalg.norm(coeffs1)
            norm2 = np.linalg.norm(coeffs2)
            if norm1 > 1e-9 and norm2 > 1e-9: # Avoid division by zero for null vectors
                 value = np.dot(coeffs1, coeffs2) / (norm1 * norm2)
            else:
                 value = np.nan # Undefined if one vector is zero
            ylabel = "Cosine Similarity of Coefficients"
        else:
            print(f"Error: Unknown comparison metric '{metric}'.")
            return

        if np.isfinite(value):
             segment_pairs.append(f"{res1['segment_index']}-{res2['segment_index']}")
             comparison_values.append(value)

    if not segment_pairs:
        print(f"No valid adjacent segment comparisons found for metric '{metric}'.")
        return

    fig = plt.figure(figsize=(10, 5))
    plt.plot(segment_pairs, comparison_values, '-o', markersize=5)
    plt.xlabel("Adjacent Segment Pair Index")
    plt.ylabel(ylabel)
    if title is None:
        title = f"{ylabel} vs. Segment Pair"
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if filename:
        try: # Add try-except for saving robustness
            plt.savefig(filename, dpi=300, bbox_inches='tight') # Use tight bbox
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"ERROR saving plot to {filename}: {e}")
        finally:
            plt.close(fig) # Close the figure after attempting to save
    else:
        plt.show()
        
def plot_coefficient_heatmap_vs_segment(results_list, zernike_indices, value_type='coeff', title=None, filename=None):
    """
    Plots a heatmap showing Zernike coefficients variation across segments.

    Args:
        results_list (list): List of result dictionaries from successful segment fits.
                             Each dict must contain 'segment_index', 'fitted_coeffs'.
        zernike_indices (list): The list of (n, m) tuples corresponding to the coefficients.
        value_type (str): Type of value to plot: 'coeff' (raw value), 'abs' (absolute value),
                          'sq' (squared value). Defaults to 'coeff'.
        title (str, optional): Title for the plot.
        filename (str or Path, optional): Path to save the figure. If None, shows interactively.
    """
    if not results_list:
        print("Warning: No segment results provided for coefficient heatmap.")
        return

    # Sort results by segment index
    results_list.sort(key=lambda r: r['segment_index'])

    segment_indices = [r['segment_index'] for r in results_list]
    n_segments = len(segment_indices)
    n_coeffs = len(zernike_indices)

    # Create data matrix: rows = Zernike coeffs, cols = segments
    data_matrix = np.zeros((n_coeffs, n_segments))

    for seg_idx, result in enumerate(results_list):
        coeffs = result.get('fitted_coeffs')
        if coeffs is not None and len(coeffs) == n_coeffs:
            if value_type == 'abs':
                data_matrix[:, seg_idx] = np.abs(coeffs)
            elif value_type == 'sq':
                 data_matrix[:, seg_idx] = coeffs**2
            else: # Default 'coeff'
                 data_matrix[:, seg_idx] = coeffs
        else:
            data_matrix[:, seg_idx] = np.nan # Mark missing data

    if np.all(np.isnan(data_matrix)):
         print("Warning: All coefficient data is missing or NaN for heatmap.")
         return


    fig, ax = plt.subplots(figsize=(max(6, n_segments * 0.5), max(6, n_coeffs * 0.25)))

    # Determine color limits and colormap based on value_type
    if value_type == 'coeff':
        cmap = 'coolwarm' # Diverging for signed values
        # Find symmetric limit around 0, ignoring NaNs
        max_abs_val = np.nanmax(np.abs(data_matrix))
        if not np.isfinite(max_abs_val) or max_abs_val < 1e-9: max_abs_val = 1.0 # Default if all zero/NaN
        vmin, vmax = -max_abs_val, max_abs_val
        cbar_label = "Coefficient Value"
    else:
        cmap = 'viridis' # Sequential for positive values
        vmin = 0
        vmax = np.nanmax(data_matrix)
        if not np.isfinite(vmax) or vmax < 1e-9: vmax = 1.0 # Default if all zero/NaN
        if value_type == 'sq':
            cbar_label = "Coefficient Squared (Power)"
        else: # abs
             cbar_label = "Coefficient Absolute Value"


    im = ax.imshow(data_matrix, aspect='auto', cmap=cmap, origin='lower',
                   interpolation='nearest', vmin=vmin, vmax=vmax)

    # Set ticks and labels
    ax.set_xlabel("Segment Index")
    ax.set_ylabel("Zernike Index (l)")

    # X ticks: Segment indices
    ax.set_xticks(np.arange(n_segments))
    ax.set_xticklabels(segment_indices)

    # Y ticks: Zernike indices l and (n,m)
    ytick_labels = [f"{l}\n({n},{m})" for l, (n, m) in enumerate(zernike_indices)]
    ax.set_yticks(np.arange(n_coeffs))
    ax.set_yticklabels(ytick_labels, fontsize=8)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=cbar_label, pad=0.02)

    if title is None:
        title = f"Zernike Coefficient Variation Across Segments ({value_type.capitalize()} Value)"
    ax.set_title(title)

    plt.tight_layout()

    if filename:
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        except Exception as e:
            print(f"ERROR saving plot to {filename}: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()
        
        
        

# --- Example Workflow ---
if __name__ == "__main__":
    # --- Inputs ---
    base_dir = Path('/Users/dmilakov/projects/lfc/data/harps/raw/4bruce/') # Adjust if needed
    
    lfc_filename = base_dir / 'HARPS.2015-04-17T00:00:41.445_lfc.fits'
    bias_filename = base_dir / 'HARPS.2015-04-16T17:40:29.876_bias.fits'
    bias_filename = None
    detector = 'red'
    # detector = 'blue'
    hdu_idx = 2 if detector == 'red' else 1
    hard_flux_cut = 320 if detector == 'red' else 420
    
    # --- Plotting Directory ---
    plot_dir = Path('/Users/dmilakov/projects/spectroperf/dataprod/plots/') / time.strftime('%Y-%m-%dT%H-%M-%S')  # Define output directory
    plot_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
    print(f"Plots will be saved to: {plot_dir.resolve()}")
    save_plots_to_file = True
    
    # --- Plotting Control ---
    # Set this to True to generate plots for the selected segment(s)
    # If TARGET_SEGMENT_INDEX is None, plots will be generated and SAVED for ALL segments.
    # If TARGET_SEGMENT_INDEX is set, plots will be generated and SAVED for only that segment.
    GENERATE_STAMP_PLOTS = True
    # Generate Zernike fit plots (comparison and residuals)?
    GENERATE_ZERNIKE_PLOTS = True
    # Generate Zernike power-spectrum
    GENERATE_SPECTRUM_PLOTS = True
    # Generage plots comparing coefficients
    GENERATE_COMPARISON_PLOTS = True
    # Generate a 2d heatmap for coefficients
    GENERATE_COEFF_HEATMAP = True
    
    # Mark all identified peaks
    visualise_peaks = False
    # Mark all identified echelle orders
    visualise_clusters = False

    # --- Segmentation & Selection ---
    NUM_SEGMENTS = 16   # <<< Number of segments to divide the order into
    TARGET_ORDER_NUMBER = 4
    TARGET_IMAGE_TYPE = 'A'
    # Set to a specific segment index (0 to NUM_SEGMENTS-1) to process only one,
    # or set to None to process and analyze all segments.
    TARGET_SEGMENT_INDEX = None # Process all segments
    # TARGET_SEGMENT_INDEX = 2

    # Zernike Fitting Parameters (per segment)
    N_MAX_ZERN = 6
    R_MAX_ZERN = 5.0

    # --- Peak Finding and Clustering Parameters (Keep as before) ---
    STAMP_HALF_WIDTH = 5
    PEAK_MIN_DISTANCE = 5
    CLUSTER_EPS_X = 0.002
    CLUSTER_EPS_Y = 0.02
    CLUSTER_MIN_SAMPLES = 5
    GAUSSIAN_FIT_THRESHOLD_SNR = 5.0
    threshold = 350 # Example threshold

    # --- 1. Load Data ---
    try:
        image_data = load_echelle_data(lfc_filename, bias_path = bias_filename, 
                                       hdu_index=hdu_idx)
        image_shape = image_data.shape
    except FileNotFoundError as e:
        print(e)
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        # exit()


    # --- 2. Find Peaks ---
    # Optional: Pre-filter data slightly to help peak finding
    # data_filtered = ndi.gaussian_filter(image_data, sigma=0.5)
    # Use a reasonable absolute threshold based on data properties
    # threshold = np.median(image_data) + 5 * np.std(image_data[image_data > np.median(image_data)]) # Threshold above background
    # threshold = 350 # Or set manually based on inspection
    print(f"Using peak threshold: {threshold}")
    all_peaks_xy = find_peaks(image_data, min_distance=PEAK_MIN_DISTANCE, 
                              hard_cut=hard_flux_cut)

    # Optional: Visualize peaks on image
    
    if visualise_peaks:
        plt.figure(figsize=(12, 8))
        plt.imshow(image_data, cmap='gray', origin='lower', vmax=np.percentile(image_data, 99.5))
        plt.scatter(all_peaks_xy[:, 0], all_peaks_xy[:, 1], s=10, facecolors='none', edgecolors='r', alpha=0.7)
        plt.title("Detected Peaks")
        plt.show()


    # --- 3. Cluster Peaks ---
    # (Keep the cluster_peaks_to_orders call and visualization as before)
    raw_orders, paired_orders_dict = cluster_peaks_to_orders(
        all_peaks_xy, image_shape, CLUSTER_EPS_X, CLUSTER_EPS_Y, CLUSTER_MIN_SAMPLES
    )
    
    if visualise_clusters:
        # Optional: Visualize clusters
        plt.figure(figsize=(12, 8))
        plt.imshow(image_data, cmap='gray', origin='lower', vmax=np.percentile(image_data, 99)) # Show background too
        # plt.scatter(all_peaks_xy[:, 0], all_peaks_xy[:, 1], s=5, c='gray', alpha=0.1, label='All Peaks')
        colors = plt.cm.viridis(np.linspace(0, 1, len(paired_orders_dict)))
        for i, (order_num, pair_info) in enumerate(paired_orders_dict.items()):
            color = colors[i]
            if pair_info['A'] is not None:
                plt.plot(pair_info['A'][:, 0], pair_info['A'][:, 1], 'o-', color=color, markersize=3, lw=1, label=f'Order {order_num}A')
            if pair_info['B'] is not None:
                plt.plot(pair_info['B'][:, 0], pair_info['B'][:, 1], 's--', color=color, markersize=3, lw=1, label=f'Order {order_num}B')
        plt.title("Clustered Echelle Orders (A=solid/o, B=dashed/s)")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.show()
    # input(a,'press any key to continue')

    # --- 4. Select Order ---
    # (Keep the order selection logic as before)
    if TARGET_ORDER_NUMBER not in paired_orders_dict:
        print(f"Error: Target order number {TARGET_ORDER_NUMBER} not found.")
        exit()
    selected_order_pair = paired_orders_dict[TARGET_ORDER_NUMBER]
    if TARGET_IMAGE_TYPE not in ['A', 'B']:
        print(f"Error: Target image type must be 'A' or 'B'.")
        exit()
    selected_peaks_full_order = selected_order_pair.get(TARGET_IMAGE_TYPE)
    if selected_peaks_full_order is None or len(selected_peaks_full_order) == 0:
        print(f"Error: No peaks found for Order {TARGET_ORDER_NUMBER}, Image {TARGET_IMAGE_TYPE}.")
        exit()

    print(f"\nSelected Order {TARGET_ORDER_NUMBER}, Image {TARGET_IMAGE_TYPE} with {len(selected_peaks_full_order)} peaks.")

    # --- 5. Segmentation ---
    # Sort peaks by X coordinate (already done in clustering, but double-check)
    selected_peaks_full_order = selected_peaks_full_order[np.argsort(selected_peaks_full_order[:, 0])]

    # Split the sorted peaks into segments
    # np.array_split tries to make segments as equal size as possible
    segmented_peaks = np.array_split(selected_peaks_full_order, NUM_SEGMENTS)

    print(f"Divided selected order into {len(segmented_peaks)} segments.")
    for iseg, seg_peaks in enumerate(segmented_peaks):
        print(f"  Segment {iseg}: {len(seg_peaks)} peaks.")

    # --- 6. Process Segment(s) and Fit ---
    all_segment_results = [] # Store results for each segment
    common_fit_info = {}
    
    # Determine which segments to process
    if TARGET_SEGMENT_INDEX is None:
        segments_to_process = range(NUM_SEGMENTS)
        print("\nProcessing all segments...")
    elif 0 <= TARGET_SEGMENT_INDEX < NUM_SEGMENTS:
        segments_to_process = [TARGET_SEGMENT_INDEX]
        print(f"\nProcessing only segment {TARGET_SEGMENT_INDEX}...")
    else:
        print(f"Error: Invalid TARGET_SEGMENT_INDEX {TARGET_SEGMENT_INDEX}. Must be between 0 and {NUM_SEGMENTS-1} or None.")
        exit()


    # --- Loop through the segments to process ---
    for segment_idx in segments_to_process:
        segment_peaks_xy = segmented_peaks[segment_idx]
        print(f"\n--- Processing Segment {segment_idx} ---")

        if len(segment_peaks_xy) < CLUSTER_MIN_SAMPLES: # Need enough peaks for stable processing/fit
            print(f"Skipping Segment {segment_idx}: Too few peaks ({len(segment_peaks_xy)}).")
            continue
        
        # --- Construct filenames IF plotting is enabled for this segment ---
        stamps_filepath = None
        residuals_filepath = None
        # Decide if plotting/saving should happen for this specific segment iteration
        should_plot_this_segment = GENERATE_STAMP_PLOTS and \
                                   (TARGET_SEGMENT_INDEX is None or segment_idx == TARGET_SEGMENT_INDEX)

        if should_plot_this_segment:
             stamps_dir = plot_dir / f"Order{TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}" / "stamps"
             stamps_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
             stamps_filepath = stamps_dir / f"Order{TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}_seg{segment_idx}_stamps.pdf"
             residuals_filepath = stamps_dir / f"Order{TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}_seg{segment_idx}_norm_residuals.pdf"
        
        # Process the peaks within this segment
        X_stack, Y_stack, Z_stack, fit_details = process_echelle_order(
            segment_peaks_xy,
            image_data,
            STAMP_HALF_WIDTH,
            GAUSSIAN_FIT_THRESHOLD_SNR,
            plot_stamps=should_plot_this_segment,
            segment_id=segment_idx,
            stamps_save_path=stamps_filepath,       # Pass the constructed path or None
            residuals_save_path=residuals_filepath # Pass the constructed path or None
        )

        # Fit Zernike polynomials if processing was successful and Zernpy is available
        if ZERNPY_AVAILABLE and X_stack is not None and len(X_stack) > (N_MAX_ZERN + 1)**2 + 2 : # Check if enough points for fit

            print(f"\n--- Starting Zernike Fit for Segment {segment_idx} ---")
            # Instantiate a new fitter for each segment to ensure clean state
            fitter = ZernikeFitter(n_max=N_MAX_ZERN, r_max=R_MAX_ZERN)

            initial_guess = {'xc': 0.0, 'yc': 0.0}
            bounds = {'lower': [-0.5, -0.5] + [-np.inf]*fitter.n_coeffs,
                      'upper': [ 0.5,  0.5] + [ np.inf]*fitter.n_coeffs}

            fit_success = fitter.fit(X_stack, Y_stack, Z_stack,
                                     initial_guess=initial_guess,
                                     bounds=bounds,
                                     verbose=False) # Less verbose fitting in the loop

            if fit_success:
                # fitter.plot_fit_comparison(X_stack, Y_stack, Z_stack, title=f"Order {TARGET_ORDER_NUMBER} Slice {TARGET_IMAGE_TYPE} Segment {segment_idx:02d}")
                segment_result = fitter.get_results(include_coeffs_table=False) # Get dict results
                segment_result['segment_index'] = segment_idx
                segment_result['num_peaks_processed'] = len(fit_details)
                segment_result['num_pixels_stacked'] = len(X_stack)
                # Store common info once if needed (use results from first successful fit)
                if not common_fit_info:
                     common_fit_info = {
                         'zernike_indices': segment_result['zernike_indices'],
                         'n_max_zern': N_MAX_ZERN,
                         'r_max_zern': R_MAX_ZERN
                     }
                all_segment_results.append(segment_result)
                print(f"Segment {segment_idx}: Zernike fit successful. RMSE={segment_result['rmse']:.4f}, R^2={segment_result['r_squared']:.4f}")

                # --- Plot/Save Zernike Results IF enabled ---
                should_plot_zernike = GENERATE_ZERNIKE_PLOTS and \
                                      (TARGET_SEGMENT_INDEX is None or segment_idx == TARGET_SEGMENT_INDEX)

                if should_plot_zernike:
                    # Construct Zernike plot filenames
                    zernike_dir = plot_dir / f"Order{TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}" / "zernike"
                    zernike_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
                    
                    zernike_comp_filepath = zernike_dir / f"O{TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}_S{segment_idx}_zernike_fit.pdf"
                    zernike_res_filepath = zernike_dir / f"O{TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}_S{segment_idx}_zernike_residuals.pdf"

                    print(f"Segment {segment_idx}: Saving Zernike plots...")
                    try:
                         # Pass the filename argument to the fitter's plot methods
                         fitter.plot_fit_comparison(X_stack, Y_stack, Z_stack,
                                                     title=f"Zernike Fit (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}, Seg {segment_idx})",
                                                     filename=zernike_comp_filepath)
                    except Exception as e:
                         print(f"ERROR generating/saving Zernike comparison plot: {e}")
                         traceback.print_exc()

                    try:
                         fitter.plot_fit_residuals(X_stack, Y_stack, Z_stack,
                                                    title=f"Zernike Residuals (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}, Seg {segment_idx})",
                                                    filename=zernike_res_filepath)
                    except Exception as e:
                         print(f"ERROR generating/saving Zernike residual plot: {e}")
                         traceback.print_exc()
                # --- Plot/Save Zernike Power Spectrum ---
                should_plot_spectrum = GENERATE_SPECTRUM_PLOTS and \
                                       (TARGET_SEGMENT_INDEX is None or segment_idx == TARGET_SEGMENT_INDEX)
                if should_plot_spectrum:
                    # Construct Zernike plot filenames
                    zernike_dir = plot_dir / f"Order{TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}" / "zernike"
                    zernike_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
                    
                    spectrum_filepath = zernike_dir / f"O{TARGET_ORDER_NUMBER}_{TARGET_IMAGE_TYPE}_S{segment_idx}_zernike_spectrum.pdf"
                    print(f"Segment {segment_idx}: Saving Zernike spectrum plot...")
                    try:
                        # Call the method on the fitter instance
                        fitter.plot_power_spectrum(
                            title=f"Zernike Spectrum (coeff) - Seg {segment_idx}",
                            filename=spectrum_filepath,
                            plot_type='coeff' # 'abs' or 'sq' or 'coeff'
                        )
                    except Exception as e:
                        print(f"ERROR saving Zernike spectrum plot: {e}")
                        traceback.print_exc()
                # --- End Zernike Plotting Block ---

            else:
                print(f"Segment {segment_idx}: Zernike fit failed. Message: {fitter.message}")

        elif X_stack is None:
            print(f"Segment {segment_idx}: Skipped Zernike fitting because data processing failed or yielded no data.")
        elif len(X_stack) <= (N_MAX_ZERN + 1)**2 + 2:
             print(f"Segment {segment_idx}: Skipped Zernike fitting because not enough stacked pixels ({len(X_stack)}) for the number of parameters.")
        elif not ZERNPY_AVAILABLE:
             print(f"Segment {segment_idx}: Skipped Zernike fitting because ZernikeFitter class is not available.")

    # --- End of Segment Loop ---

    # --- 7. Analyze Results Across Segments (if multiple segments processed) ---
    if len(all_segment_results) > 1:
        print("\n--- Analyzing Zernike Coefficients Across Segments ---")

        if not common_fit_info:
             print("No successful segment fits with common info to analyze.")
        else:
             zernike_indices = common_fit_info['zernike_indices']
             comparison_dir = plot_dir / f"Order{TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}" / "comparison"
             comparison_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
             # Construct filenames for analysis plots (always construct path objects)
             defocus_plot_path = comparison_dir / f"O{TARGET_ORDER_NUMBER}_{TARGET_IMAGE_TYPE}_analysis_defocus_vs_segment.pdf"
             astig_plot_path = comparison_dir / f"O{TARGET_ORDER_NUMBER}_{TARGET_IMAGE_TYPE}_analysis_astigmatism_vs_segment.pdf"
             astig2_plot_path = comparison_dir / f"O{TARGET_ORDER_NUMBER}_{TARGET_IMAGE_TYPE}_analysis_secondary_astigmatism_vs_segment.pdf"
             primary_spherical_plot_path = comparison_dir / f"O{TARGET_ORDER_NUMBER}_{TARGET_IMAGE_TYPE}_analysis_primary_spherical_vs_segment.pdf"
             coma_plot_path = comparison_dir / f"O{TARGET_ORDER_NUMBER}_{TARGET_IMAGE_TYPE}_analysis_coma_vs_segment.pdf"
             rmse_diff_plot_path = comparison_dir / f"O{TARGET_ORDER_NUMBER}_{TARGET_IMAGE_TYPE}_analysis_rmse_diff_vs_segment.pdf"
             cosine_sim_plot_path = comparison_dir / f"O{TARGET_ORDER_NUMBER}_{TARGET_IMAGE_TYPE}_analysis_cosine_sim_vs_segment.pdf"

             # --- Determine if analysis plots should be SAVED ---
             # Save ONLY if flag is True AND we processed all segments
             save_analysis_plots = GENERATE_COMPARISON_PLOTS and (TARGET_SEGMENT_INDEX is None)
             # --- Generate Analysis Plots (Conditionally Save or Show) ---
             # Only proceed if the flag is True
             if GENERATE_COMPARISON_PLOTS:
                 if save_plots_to_file:
                     print("Saving analysis plots...")
                 else:
                     # Print message only if flag is True but we didn't process all segments
                     if TARGET_SEGMENT_INDEX is not None:
                         print("Generating analysis plots (will show interactively as only one segment was targeted for processing)...")

                 # Pass filename ONLY if save_analysis_plots is True
                 fname_defocus = defocus_plot_path if save_plots_to_file else None
                 try:
                     plot_coefficient_vs_segment(all_segment_results, zernike_indices, n=2, m=0,
                                                title=f"Defocus Z(2,0) vs Segment (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE})",
                                                filename=fname_defocus)
                 except Exception as e: print(f"ERROR generating/saving defocus plot: {e}")

                 fname_astig = astig_plot_path if save_plots_to_file else None
                 try:
                     plot_coefficient_vs_segment(all_segment_results, zernike_indices, n=2, m=2,
                                                title=f"Astigmatism Z(2,2) vs Segment (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE})",
                                                filename=fname_astig)
                 except Exception as e: print(f"ERROR generating/saving astigmatism plot: {e}")

                 fname_coma = coma_plot_path if save_plots_to_file else None
                 try:
                     plot_coefficient_vs_segment(all_segment_results, zernike_indices, n=3, m=1,
                                                title=f"Coma Z(3,1) vs Segment (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE})",
                                                filename=fname_coma)
                 except Exception as e: print(f"ERROR generating/saving coma plot: {e}")
                 
                 fname_astig2 = astig2_plot_path if save_plots_to_file else None
                 try:
                     plot_coefficient_vs_segment(all_segment_results, zernike_indices, n=4, m=2,
                                                title=f"Secondary stigmatism Z(4,2) vs Segment (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE})",
                                                filename=fname_astig2)
                 except Exception as e: print(f"ERROR generating/saving secondary astigmatism plot: {e}")
                 
                 fname_primary_spherical = primary_spherical_plot_path if save_plots_to_file else None
                 try:
                     plot_coefficient_vs_segment(all_segment_results, zernike_indices, n=4, m=0,
                                                title=f"Primary spherical Z(4,0) vs Segment (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE})",
                                                filename=fname_primary_spherical)
                 except Exception as e: print(f"ERROR generating/saving primary spherical plot: {e}")

                 fname_rmse = rmse_diff_plot_path if save_plots_to_file else None
                 try:
                     plot_adjacent_segment_comparison(all_segment_results, metric='rmse_diff',
                                                       title=f"RMSE Diff Between Adj Seg Coeffs (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE})",
                                                       filename=fname_rmse)
                 except Exception as e: print(f"ERROR generating/saving RMSE diff plot: {e}")

                 fname_cosine = cosine_sim_plot_path if save_plots_to_file else None
                 try:
                     plot_adjacent_segment_comparison(all_segment_results, metric='cosine_similarity',
                                                       title=f"Cosine Sim Between Adj Seg Coeffs (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE})",
                                                       filename=fname_cosine)
                 except Exception as e: print(f"ERROR generating/saving cosine similarity plot: {e}")

             else:
                 # If flag is False, print message
                 print("Analysis plot generation skipped (GENERATE_COMPARISON_PLOTS=False).")
             # --- Coefficient Heatmap ---
             save_heatmap = GENERATE_COEFF_HEATMAP and (TARGET_SEGMENT_INDEX is None) and save_plots_to_file
             if GENERATE_COEFF_HEATMAP:
                 
                 comparison_dir = plot_dir / f"Order{TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE}" / "comparison"
                 comparison_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
                 
                 heatmap_filepath = comparison_dir / f"O{TARGET_ORDER_NUMBER}_{TARGET_IMAGE_TYPE}_analysis_coeff_heatmap.pdf"
                 fname_heatmap = heatmap_filepath if save_heatmap else None

                 if save_heatmap: print("Saving coefficient heatmap...")
                 else: print("Generating coefficient heatmap (will show interactively)...")

                 try:
                      plot_coefficient_heatmap_vs_segment(
                          results_list=all_segment_results,
                          zernike_indices=zernike_indices,
                          value_type='coeff', # Or 'abs' or 'sq'
                          title=f"Zernike Coeffs vs Segments (Order {TARGET_ORDER_NUMBER}{TARGET_IMAGE_TYPE})",
                          filename=fname_heatmap
                      )
                 except Exception as e:
                      print(f"ERROR generating/saving coefficient heatmap: {e}")
                      traceback.print_exc()

             else: print("Coefficient heatmap generation skipped (GENERATE_COEFF_HEATMAP=False).")

    elif len(all_segment_results) == 1 and GENERATE_COMPARISON_PLOTS:
         print("\nOnly one segment processed or successfully fitted. Cannot generate comparison plots.")
         # Print the single result details if needed
         print("\nOnly one segment processed or successfully fitted. No cross-segment analysis possible.")
         # Optionally print the single result details here
         print("\n--- Zernike Fit Result (Single Segment) ---")
         res = all_segment_results[0]
         print(f"Segment Index: {res['segment_index']}")
         print(f"Success: {res['success']}")
         print(f"Fitted Centroid Xc = {res['fitted_xc']:.4f} +/- {res['err_xc']:.4f}")
         print(f"Fitted Centroid Yc = {res['fitted_yc']:.4f} +/- {res['err_yc']:.4f}")
         print(f"RMSE: {res['rmse']:.4f}")
         print(f"R-squared: {res['r_squared']:.4f}")
         print("Fitted Zernike Coefficients:")
         zernike_indices = res['zernike_indices']
         for i, (n, m) in enumerate(zernike_indices):
              coeff_val = res['fitted_coeffs'][i]
              err_val = res['err_coeffs'][i] if res['err_coeffs'] is not None and np.isfinite(res['err_coeffs'][i]) else np.nan
              print(f"  Z(n={n}, m={m}): {coeff_val:.5f} +/- {err_val:.5f}")
    elif not GENERATE_COMPARISON_PLOTS:
        pass # Don't print anything if comparison plots were explicitly disabled
    else: # No segments successfully fitted
        print("\nNo segments were successfully processed and fitted.")

   

    print("\nWorkflow finished.")