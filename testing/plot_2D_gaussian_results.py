#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 23:59:16 2025

@author: dmilakov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to read GAUSS2D FITS extension and plot 3D scatter plots of Gaussian parameters.
"""
import argparse
import numpy as np
import fitsio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting
from pathlib import Path

# Indices for Gaussian parameters within the GAUSS_PARAMS vector column
# Based on ['amplitude', 'xo_stamp', 'yo_stamp', 'sigma_x', 'sigma_y', 'theta_rad', 'offset']
IDX_SIGMA_X = 3
IDX_SIGMA_Y = 4
IDX_THETA_RAD = 5

def plot_gaussian_param_3d(peak_x, peak_y, param_z_values, param_name, aspect_ratio_values, title_suffix=""):
    """
    Creates a 3D scatter plot of (peak_x, peak_y, param_z_values).
    Points are colored by the aspect_ratio_values.

    Args:
        peak_x (np.ndarray): X coordinates of the peaks.
        peak_y (np.ndarray): Y coordinates of the peaks.
        param_z_values (np.ndarray): Z values for the scatter plot (e.g., sigma_x).
        param_name (str): Name of the Z parameter (for Z-axis label).
        aspect_ratio_values (np.ndarray): Values used for coloring the points.
        title_suffix (str, optional): Suffix to add to the plot title.
    """
    if not all(len(arr) == len(peak_x) for arr in [peak_y, param_z_values, aspect_ratio_values]):
        print("Error: Input arrays must have the same length.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Handle potential NaNs or Infs in aspect_ratio for coloring
    # Replace NaNs with a neutral value (e.g., 1) or filter them out
    # For color mapping, it's often good to clip extreme values too.
    valid_color_mask = np.isfinite(aspect_ratio_values)
    if not np.all(valid_color_mask):
        print(f"Warning: Found {np.sum(~valid_color_mask)} non-finite aspect ratio values. They will be plotted with a default color or skipped if all are non-finite.")
        # Option: assign a default color, or filter data
        # For now, let matplotlib handle it, but better to be explicit

    # Determine color limits for aspect ratio
    # Exclude NaNs/Infs from percentile calculation
    finite_aspect_ratios = aspect_ratio_values[valid_color_mask]
    if len(finite_aspect_ratios) > 0:
        c_min = np.percentile(finite_aspect_ratios, 5)
        c_max = np.percentile(finite_aspect_ratios, 95)
        # Ensure c_min is not equal to c_max
        if c_min == c_max:
            c_min -= 0.1
            c_max += 0.1
    else: # Fallback if all are NaN/Inf
        c_min, c_max = 0, 2


    scatter = ax.scatter(peak_x, peak_y, param_z_values,
                         c=aspect_ratio_values, cmap='viridis',
                         s=10, alpha=0.7, vmin=c_min, vmax=c_max)

    ax.set_xlabel("Peak X (pixel)")
    ax.set_ylabel("Peak Y (pixel)")
    ax.set_zlabel(param_name)
    ax.set_title(f"3D Scatter of {param_name} vs. Peak Position {title_suffix}")

    # Add colorbar
    cbar = fig.colorbar(scatter, shrink=0.7, aspect=20)
    cbar.set_label("Aspect Ratio (Sigma_X / Sigma_Y)")
    
    # Improve layout if possible (might need manual adjustment for 3D)
    # fig.tight_layout() # Often tricky with 3D colorbars

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot 3D scatter of Gaussian fit parameters from a FITS file.")
    parser.add_argument("fits_file", type=str, help="Path to the input FITS file (e.g., *_zernike_results.fits)")
    parser.add_argument("--extname", type=str, default="GAUSS2D", help="Name of the FITS extension containing Gaussian parameters (default: GAUSS2D)")
    parser.add_argument("--order", type=int, default=None, help="Optional: Filter by specific echelle order number.")
    parser.add_argument("--imgtype", type=int, default=None, choices=[0, 1], help="Optional: Filter by image type (0 for A, 1 for B).")
    parser.add_argument("--segment", type=int, default=None, help="Optional: Filter by specific segment number.")

    args = parser.parse_args()

    fits_path = Path(args.fits_file)
    if not fits_path.is_file():
        print(f"Error: FITS file not found at {fits_path}")
        return

    print(f"Reading data from: {fits_path}, extension: {args.extname}")

    try:
        with fitsio.FITS(fits_path, 'r') as fits:
            if args.extname not in fits:
                print(f"Error: Extension '{args.extname}' not found in FITS file.")
                available_exts = [hdu.get_extname() for hdu in fits]
                print(f"Available extensions: {available_exts}")
                return
            
            # Read necessary columns
            # Check if columns exist before trying to read them all
            hdu = fits[args.extname]
            header = hdu.read_header()
            available_cols = [key for key in header.keys() if key.startswith('TTYPE')]
            col_names_in_hdu = [header[key] for key in available_cols]

            required_cols_for_data = ['PEAK_X', 'PEAK_Y', 'CEN_X', 'CEN_Y',
                                      'SIGMA_X', 'SIGMA_Y', 'THETA','AMPLITUDE','CONST']
            required_cols_for_filter = []
            if args.order is not None: required_cols_for_filter.append('ORDER_NUM')
            if args.imgtype is not None: required_cols_for_filter.append('IMGTYPE')
            if args.segment is not None: required_cols_for_filter.append('SEGMENT')

            all_needed_cols = list(set(required_cols_for_data + required_cols_for_filter))

            missing_cols = [col for col in all_needed_cols if col not in col_names_in_hdu]
            if missing_cols:
                print(f"Error: Required columns missing from HDU '{args.extname}': {missing_cols}")
                print(f"Available columns: {col_names_in_hdu}")
                return

            data = hdu.read(columns=all_needed_cols)
            print(f"Read {len(data)} rows from extension '{args.extname}'.")

    except Exception as e:
        print(f"Error reading FITS file: {e}")
        traceback.print_exc()
        return

    if len(data) == 0:
        print("No data found in the extension.")
        return

    # --- Filtering based on arguments ---
    mask = np.ones(len(data), dtype=bool)
    title_suffix = ""
    if args.order is not None:
        mask &= (data['ORDER_NUM'] == args.order)
        title_suffix += f" (Order {args.order})"
    if args.imgtype is not None:
        mask &= (data['IMGTYPE'] == args.imgtype)
        img_char = 'A' if args.imgtype == 0 else 'B'
        title_suffix += f" (Img {img_char})"
    if args.segment is not None:
        mask &= (data['SEGMENT'] == args.segment)
        title_suffix += f" (Seg {args.segment})"

    filtered_data = data[mask]

    if len(filtered_data) == 0:
        print("No data matches the specified filters.")
        return
    print(f"Plotting {len(filtered_data)} filtered data points.")

    # Extract parameters
    peak_x = filtered_data['CEN_X'][:,0]
    peak_y = filtered_data['CEN_Y'][:,0]
    # gauss_params_all = filtered_data[['AMPLITUDE',
    #                                   'CEN_X',
    #                                   'CEN_Y',
    #                                   'SIGMA_X',
    #                                   'SIGMA_Y',
    #                                   'THETA',
    #                                   'CONST']] # This is an array of arrays

    # Ensure gauss_params_all is 2D
    # if gauss_params_all.ndim == 1 and isinstance(gauss_params_all[0], np.ndarray):
    #     try:
    #         gauss_params_all = np.vstack(gauss_params_all)
    #     except ValueError as e:
    #         print(f"Error reshaping GAUSS_PARAMS. Ensure it's a uniform array of parameter vectors: {e}")
    #         return

    # if gauss_params_all.shape[1] <= max(IDX_SIGMA_X, IDX_SIGMA_Y, IDX_THETA_RAD):
    #     print(f"Error: GAUSS_PARAMS column does not have enough elements. Expected at least {max(IDX_SIGMA_X, IDX_SIGMA_Y, IDX_THETA_RAD)+1}, found {gauss_params_all.shape[1]}.")
    #     return


    sigma_x_values = filtered_data['SIGMA_X'][:,0]
    sigma_y_values = filtered_data['SIGMA_Y'][:,0]
    theta_rad_values = filtered_data['THETA'][:,0]
    theta_deg_values = np.degrees(theta_rad_values) # Often more intuitive
    print(theta_deg_values)
    # Calculate aspect ratio for coloring, handle division by zero
    aspect_ratio = np.full_like(sigma_x_values, np.nan) # Initialize with NaN
    valid_sigma_y_mask = sigma_y_values != 0
    aspect_ratio[valid_sigma_y_mask] = sigma_x_values[valid_sigma_y_mask] / sigma_y_values[valid_sigma_y_mask]


    # Create plots
    print("Plotting Sigma_X...")
    plot_gaussian_param_3d(peak_x, peak_y, sigma_x_values, "Sigma_X (pixels)", aspect_ratio, title_suffix)

    print("Plotting Sigma_Y...")
    plot_gaussian_param_3d(peak_x, peak_y, sigma_y_values, "Sigma_Y (pixels)", aspect_ratio, title_suffix)

    print("Plotting Theta (degrees)...")
    plot_gaussian_param_3d(peak_x, peak_y, theta_deg_values, "Theta (degrees)", aspect_ratio, title_suffix)

    print("All plots generated.")

if __name__ == "__main__":
    main()