#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clustering.py

Created on Thu Apr 24 14:18:37 2025

@author: dmilakov
"""

import numpy as np
from sklearn.cluster import DBSCAN
import warnings
import matplotlib.pyplot as plt

def normalize_coordinates(points_int, shape):
    """Normalizes integer pixel coordinates to [0, 1] range."""
    points_flt = points_int.astype(float)
    # Handle potential division by zero if shape has zero dimension
    width = shape[1] if shape[1] > 0 else 1
    height = shape[0] if shape[0] > 0 else 1
    points_flt[:, 0] /= width   # Normalize x by width
    points_flt[:, 1] /= height  # Normalize y by height
    return points_flt

# inverse_normalize_coordinates is not needed by clustering but keep if used elsewhere
# def inverse_normalize_coordinates(points_flt, shape): ...

def cluster_peaks_to_orders(peaks_xy, image_shape, eps_x, eps_y, min_samples,
                            plot_interactive=False, image_data=None):
    """
    Clusters peaks into orders using DBSCAN and identifies A/B pairs based on X proximity.

    Args:
        peaks_xy (np.ndarray): Array of (x, y) coordinates of found peaks.
        image_shape (tuple): Shape of the original image (height, width).
        eps_x (float): DBSCAN epsilon for normalized X coordinate.
        eps_y (float): DBSCAN epsilon for normalized Y coordinate.
        min_samples (int): DBSCAN min_samples parameter.
        plot_interactive (bool): If True, display an interactive plot showing the
                                 original peaks and the clustered orders. Defaults to False.

    Returns:
        tuple: (raw_orders, paired_orders_dict)
               raw_orders: List of arrays, each containing sorted peaks for a raw cluster.
               paired_orders_dict: Dictionary where keys are assigned order numbers (1, 2, ...)
                                   and values are dicts {'A': peaks_A, 'B': peaks_B}.
                                   'A' or 'B' can be None if no pair was found.
    """
    if peaks_xy is None or len(peaks_xy) == 0:
        print("Clustering: No peaks provided.")
        return [], {}

    print("Clustering: Normalizing coordinates...")
    points_normalized = normalize_coordinates(peaks_xy, image_shape)

    # Scale normalized coordinates by inverse epsilon for DBSCAN
    # Avoid division by zero if eps is zero or negative
    scale_x = 1.0 / eps_x if eps_x > 1e-9 else 1e9
    scale_y = 1.0 / eps_y if eps_y > 1e-9 else 1e9
    points_scaled = np.copy(points_normalized)
    points_scaled[:, 0] *= scale_x
    points_scaled[:, 1] *= scale_y

    print("Clustering: Running DBSCAN...")
    # Apply DBSCAN (eps=1 because data is scaled)
    try:
        clustering = DBSCAN(eps=1, min_samples=min_samples).fit(points_scaled)
        labels = clustering.labels_
    except Exception as e:
        print(f"ERROR during DBSCAN fitting: {e}")
        return [], {}

    # Extract clusters (potential orders) and sort points within each cluster by Y (trace direction)
    unique_labels = sorted(list(set(labels) - {-1})) # Ignore noise (-1)
    raw_orders = []
    print(f"Clustering: Found {len(unique_labels)} raw clusters (excluding noise).")
    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_points_xy = peaks_xy[cluster_mask]
        # Sort by Y-coordinate (assuming this follows the echelle trace)
        cluster_points_sorted = cluster_points_xy[np.argsort(cluster_points_xy[:, 1])]
        raw_orders.append(cluster_points_sorted)

    if not raw_orders:
        print("Clustering: No valid clusters formed.")
        return [], {}

    # --- Identify A/B pairs based on proximity in X ---
    # Calculate average X for each raw order cluster
    avg_x_coords = [np.mean(order[:, 0]) for order in raw_orders]
    # Sort orders based on their average X coordinate
    sorted_order_indices = np.argsort(avg_x_coords)

    paired_orders = {}
    order_counter = 0
    paired_status = [False] * len(raw_orders)

    print(f"Clustering: Attempting to pair A/B based on X proximity...")

    # Define a threshold for pairing based on typical X separation in pixels
    # Uses a multiple of the normalized epsilon converted back to pixels
    pairing_x_threshold_pixels = (eps_x * 5.0) * image_shape[1] # Tolerate ~5x eps_x separation
    print(f"Clustering: Using X pairing threshold: {pairing_x_threshold_pixels:.2f} pixels.")

    for i in range(len(sorted_order_indices)):
        current_sorted_idx_list_pos = i # Position in the sorted list
        original_raw_order_idx = sorted_order_indices[current_sorted_idx_list_pos]

        if paired_status[original_raw_order_idx]:
            continue # This raw order was already assigned to a pair

        current_order_avg_x = avg_x_coords[original_raw_order_idx]
        order_counter += 1 # Assign a new final order number

        # Look for the *next* unpaired order in the X-sorted list
        found_pair = False
        for j in range(current_sorted_idx_list_pos + 1, len(sorted_order_indices)):
            next_original_raw_order_idx = sorted_order_indices[j]

            if not paired_status[next_original_raw_order_idx]: # Check if this one is available
                next_order_avg_x = avg_x_coords[next_original_raw_order_idx]

                # Check if they are close enough in X to be a pair
                if abs(next_order_avg_x - current_order_avg_x) < pairing_x_threshold_pixels:
                    # Pair found! Assign A and B based on X coordinate
                    # Assume A has larger X than B (adjust if necessary)
                    # DM 2025-05-09: A has smaller X than B (determined by examining a raw LFC frame with two combs)
                    # DM 2025-05-09: Code changed accordingly
                    if current_order_avg_x < next_order_avg_x:
                         idx_A, idx_B = original_raw_order_idx, next_original_raw_order_idx
                    else:
                         idx_B, idx_A = original_raw_order_idx, next_original_raw_order_idx

                    paired_orders[order_counter] = {
                        'A': raw_orders[idx_A],
                        'B': raw_orders[idx_B]
                    }
                    paired_status[idx_A] = True
                    paired_status[idx_B] = True
                    found_pair = True
                    print(f"  Paired Order {order_counter}: Raw indices {idx_A}(A) and {idx_B}(B)")
                    break # Stop searching for a pair for the current order

        # If no pair was found after checking subsequent orders
        if not found_pair:
             # Assign as pair 'B', assuming 'A' might be off-chip or not clustered
             # Or decide based on some absolute position? Let's default to B.
             # Check if it is the leftmost or rightmost based on average X? Let's assume B if alone.
             paired_orders[order_counter] = {
                 'A': None,
                 'B': raw_orders[original_raw_order_idx]
             }
             paired_status[original_raw_order_idx] = True
             print(f"  Assigned Order {order_counter}: Raw index {original_raw_order_idx}(B), no A pair found.")

    print(f"Clustering complete. Identified {len(paired_orders)} final echelle orders.")
    
    # --- Optional Interactive Plot ---
    if plot_interactive:
        print("Clustering: Displaying interactive plot...")
        fig, ax = plt.subplots(figsize=(10, 8)) # Adjust size as needed
        
        # Plot data if provided
        if image_data is not None: 
            finite_data = image_data[np.isfinite(image_data)]
            if finite_data.size > 0:
                vmin_img = np.percentile(finite_data, 5)
                vmax_img = np.percentile(finite_data, 99)
                if vmin_img == vmax_img:
                    vmin_img = vmin_img - 1 if vmin_img > 0 else -1
                    vmax_img = vmax_img + 1
            else:
                vmin_img = 0
                vmax_img = 1

            ax.imshow(image_data, cmap='gray', origin='lower',
                      vmin=vmin_img, vmax=vmax_img, aspect='auto')
        
        # Scatter plot of ALL original peaks (useful context)
        if peaks_xy.size > 0:
            ax.scatter(peaks_xy[:, 0], peaks_xy[:, 1], s=5, c='gray', alpha=0.2, label='All Peaks')

        # Plot noise points if any
        noise_mask = (labels == -1)
        if np.any(noise_mask):
             ax.scatter(peaks_xy[noise_mask, 0], peaks_xy[noise_mask, 1], s=8, c='black', marker='x', label='Noise')

        # Plot clustered orders (A/B pairs)
        if paired_orders:
            colors = plt.cm.coolwarm(np.linspace(0, 1, len(paired_orders)))
            for i, (order_num, pair_info) in enumerate(paired_orders.items()):
                color = colors[i % len(colors)] # Cycle colors if needed

                peaks_A = pair_info.get('A')
                peaks_B = pair_info.get('B')

                if peaks_A is not None and len(peaks_A) > 0:
                    ax.plot(peaks_A[:, 0], peaks_A[:, 1], 'o-', color=color, markersize=3, lw=1,
                            label=f'Order {order_num}A' if i < 10 else None) # Limit legend entries
                if peaks_B is not None and len(peaks_B) > 0:
                     ax.plot(peaks_B[:, 0], peaks_B[:, 1], 's--', color=color, markersize=3, lw=1,
                             label=f'Order {order_num}B' if i < 10 else None)

        ax.set_title("Clustered Echelle Orders (A=solid/o, B=dashed/s)")
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        # Optionally set limits based on image_shape or data range
        # ax.set_xlim(0, image_shape[1])
        # ax.set_ylim(0, image_shape[0])
        # ax.invert_yaxis() # Often useful for image coordinates
        ax.set_aspect('equal')
        # ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5)) # Place legend outside
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        plt.show() # Display interactively


    # Return raw_orders for debugging if needed, but paired_orders_dict is primary output
    return raw_orders, paired_orders