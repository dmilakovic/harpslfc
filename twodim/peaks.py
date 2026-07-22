#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
peaks.py

Created on Thu Apr 24 14:18:16 2025

@author: dmilakov
"""

import numpy as np
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt # Import matplotlib
import warnings

def find_peaks(image_data, min_distance, 
               # threshold_abs=None, threshold_rel=None,
               hard_cut=None, plot_interactive=False): # Added plot_interactive flag
    """
    Finds local peaks in the image data. Includes optional hard flux cut and
    optional interactive plot of found peaks.

    Args:
        image_data (np.ndarray): Input 2D image data.
        min_distance (int): Minimum number of pixels separating peaks.
        threshold_abs (float, optional): Minimum intensity of peaks.
        threshold_rel (float, optional): Minimum intensity relative to max value.
        hard_cut (float, optional): If set, pixels <= this value are set to 0
                                    before peak finding. Defaults to None.
        plot_interactive (bool): If True, display an interactive plot showing the
                                 original image and the detected peaks. Defaults to False.

    Returns:
        np.ndarray: Array of (x, y) coordinates of detected peaks.
                    Returns empty array if no peaks found.
    """
    local_image_data = np.copy(image_data) # Work on a copy

    # --- Thresholding ---
    # if threshold_abs is None and threshold_rel is None:
    #      # Default threshold logic (improved robustness)
    #      with warnings.catch_warnings(): # Suppress warnings for empty slices
    #           warnings.simplefilter("ignore", category=RuntimeWarning)
    #           valid_pixels = local_image_data[local_image_data > 0]
    #           if len(valid_pixels) > 10: # Require a minimum number for robust stats
    #                threshold_abs = np.nanmedian(valid_pixels) + 1.0 * np.nanstd(valid_pixels)
    #           else:
    #                threshold_abs = np.nanpercentile(local_image_data, 75) if local_image_data.size > 0 else 1.0 # Use percentile if few positive pixels
    #      print(f"Peak finding: Using default absolute threshold: {threshold_abs:.2f}")

    # --- Hard Cut ---
    if hard_cut is not None:
        print(f"Peak finding: Applying hard cut <= {hard_cut}")
        cut_mask = (local_image_data <= hard_cut)
        local_image_data[cut_mask] = 0.0 # Set to float zero

    # --- Peak Finding ---
    print("Peak finding: Running peak_local_max...")
    # Note: peak_local_max returns (row, col) i.e., (y, x)
    coordinates = peak_local_max(
        local_image_data, # Use the potentially modified image for finding
        min_distance=min_distance,
        # threshold_abs=threshold_abs,
        # threshold_rel=threshold_rel,
        # exclude_border=min_distance # Exclude border pixels
    )

    if coordinates.size == 0:
        print("Peak finding: No peaks found.")
        coordinates_xy = np.array([], dtype=int).reshape(0, 2) # Return empty array
    else:
        # Swap columns to get (x, y) convention
        coordinates_xy = coordinates[:, ::-1]
        print(f"Peak finding: Found {len(coordinates_xy)} peaks.")

    # --- Optional Interactive Plot ---
    if plot_interactive:
        print("Peak finding: Displaying interactive plot...")
        fig, ax = plt.subplots(figsize=(10, 8)) # Adjust size as needed

        # Determine robust color limits for display
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Show original image data
            vmin = np.nanpercentile(image_data, 5)
            vmax = np.nanpercentile(image_data, 99.5) # Avoid extreme outliers saturating scale
            if vmin == vmax: vmin -= 0.1; vmax += 0.1 # Handle constant image case

        # Display the *original* image data for context
        im = ax.imshow(image_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax,
                       interpolation='nearest')

        # Overlay detected peaks
        if coordinates_xy.size > 0:
            ax.scatter(coordinates_xy[:, 0], coordinates_xy[:, 1],
                       s=20, facecolors='none', edgecolors='red', linewidths=0.8,
                       label=f'Detected Peaks ({len(coordinates_xy)})')
            ax.legend(fontsize='small')

        ax.set_title('Detected Peaks Overlay')
        ax.set_xlabel('X pixel')
        ax.set_ylabel('Y pixel')
        fig.colorbar(im, ax=ax, shrink=0.7, label='Pixel Value')
        plt.tight_layout()
        plt.show() # Display interactively

    return coordinates_xy

def find_peaks2(image_data, min_distance, threshold_abs=None, threshold_rel=None,
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