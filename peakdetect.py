#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 18:06:41 2025

@author: dmilakov
"""

import numpy as np
import matplotlib.pyplot as plt
from harps.functions import math as mathfunc
import harps.progress_bar as progress_bar
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.cluster import DBSCAN
from collections import Counter
from matplotlib.lines import Line2D
from sklearn.linear_model import RANSACRegressor, HuberRegressor
import scipy
from scipy.signal import welch, windows, convolve, wiener, nuttall # windows, convolve used in placeholders


__all__ = [
        "get_window",
        "get_window_robust",
        "detect_initial_extrema_candidates",
        "filter_extrema_with_dbscan",
        "apply_clustering_filter_to_extrema",
        "plot_smoothing_and_derivatives",
        "plot_clustering_2d_results",
        "plot_feature_space_3d",
        "process_spectrum_for_lfc_lines",
        ]


# ==============================================================================
# == SECTION 1: USER-PROVIDED HELPER FUNCTIONS (YOU NEED TO FILL THESE IN) ====
# ==============================================================================


def _datacheck(x_axis, y_axis):
    if x_axis is None:
        x_axis = np.arange(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")
    
    # cut = np.where(y_axis!=0.0)[0]
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis
    

def _rebin( a, newshape ):
    '''Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape), ValueError(f"Array ndim {a.ndim} does not match newshape len {len(newshape)}")

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]

def _smooth(x, window_len=11, window="hanning", mode="valid"):
    """
    smooth the data using a window of the requested size.
    
    This method is based on the convolution of a scaled window on the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    
    keyword arguments:
    x -- the input signal 
    
    window_len -- the dimension of the smoothing window; should be an odd
        integer (default: 11)
    
    window -- the type of window from 'flat', 'hanning', 'hamming', 
        'bartlett', 'blackman', where flat is a moving average
        (default: 'hanning')

    
    return: the smoothed signal
        
    example:

    t = linspace(-2,2,0.1)
    x = sin(t)+randn(len(t))*0.1
    y = _smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, 
    numpy.convolve, scipy.signal.lfilter 
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    if window_len<3:
        return x
    #declare valid windows in a dictionary
    window_funcs = {
        "flat": lambda _len: np.ones(_len, "d"),
        "hanning": np.hanning,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
        "nuttall": nuttall
        }
    
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    try:
        w = window_funcs[window](window_len)
    except KeyError:
        raise ValueError(
            "Window is not one of '{0}', '{1}', '{2}', '{3}', '{4}'".format(
            *window_funcs.keys()))
    
    y = np.convolve(w / w.sum(), s, mode = mode)
    
    if mode=='valid':
        trim = (window_len - 1) // 2
        return y [trim : -trim]
    elif mode=='same':
        pad = window_len - 1
        return y[pad : pad + len(x)]

def get_window(y_axis,plot=False,minimum=8):
    if len(y_axis) == 0: return minimum
    
    freq0, P0    = welch(y_axis,nperseg=min(len(y_axis), 512))
    cut = np.where(freq0>0.02)[0]
    if len(cut) == 0 and len(freq0) > 0 : # If cut removes everything, take the first non-DC if available
        if freq0[0] == 0 and len(freq0)>1: cut = [1]
        elif freq0[0] > 0: cut = [0]
        else: return minimum # No usable frequencies
    elif len(cut) == 0 and len(freq0) == 0: # No frequencies at all
         return minimum
     
    freq, P = freq0[cut], P0[cut]
    maxind     = np.argmax(P)
    maxfreq    = freq[maxind]
    if plot:
        import matplotlib.ticker as ticker
        def one_over(x):
            """Vectorized 1/x, treating x==0 manually"""
            x = np.array(x, float)
            near_zero = np.isclose(x, 0)
            x[near_zero] = np.inf
            x[~near_zero] = 1 / x[~near_zero]
            return x
        
        fig, ax = plt.subplots(1)
        ax.semilogy(1/freq,P)
        ax.semilogy(1/maxfreq,P[maxind],marker='x',c='C1')
        ax2 = ax.secondary_xaxis('top',functions=(one_over,one_over))
        # ax2.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=10))
        ax.set_xlabel("Period (pix)")
        ax2.set_xlabel("Frequency (1/pix)")
        ax.set_ylabel("Power")
        # ax.set_xlim(1e-3,0.51)
        
        
    if maxfreq <= 1e-9: return minimum
    window = mathfunc.round_down_to_odd(1. / maxfreq)
    return window if window > minimum else minimum


# ==============================================================================
# == SECTION 2: ROBUST HELPER FUNCTIONS (Likely from our previous discussions) =
# ==============================================================================

def get_window_robust(y_axis, sampling_rate=1.0, plot=False,
                      default_window_period=15,
                      min_period_pixels=3, max_period_pixels=None,
                      nperseg_min_periods=4):
    if not isinstance(y_axis, np.ndarray): y_axis = np.asarray(y_axis)
    if len(y_axis) < 2 * default_window_period:
        return mathfunc.round_down_to_odd(max(3, len(y_axis) // 5))

    if max_period_pixels is None: max_period_pixels = len(y_axis) / 3.0
    max_period_pixels = max(max_period_pixels, min_period_pixels * 2.0)

    nperseg_val = min(len(y_axis), max(256, int(nperseg_min_periods * max_period_pixels)))
    if nperseg_val <=0: nperseg_val = min(len(y_axis), 256) # Safety for nperseg

    try:
        freqs, Pxx = welch(y_axis, fs=sampling_rate, nperseg=nperseg_val, scaling='density', window='hann')
    except ValueError:
        return mathfunc.round_down_to_odd(default_window_period)

    min_freq_of_interest = 1.0 / (max_period_pixels + 1e-9)
    max_freq_of_interest = 1.0 / (min_period_pixels + 1e-9)
    
    # Exclude DC (freqs[0] can be zero) and frequencies outside our band
    # Ensure freqs are positive before taking reciprocal for period
    valid_indices = np.where((freqs > min_freq_of_interest) & 
                             (freqs < max_freq_of_interest) & 
                             (freqs > 1e-9))[0]

    dominant_freq = 0
    if len(valid_indices) == 0 or len(Pxx[valid_indices]) == 0:
        period = default_window_period
    else:
        peak_idx_in_valid = np.argmax(Pxx[valid_indices])
        dominant_freq = freqs[valid_indices[peak_idx_in_valid]]
        if dominant_freq <= 1e-9: period = default_window_period
        else: period = 1.0 / dominant_freq

    period = np.clip(period, min_period_pixels, max_period_pixels)
    window_len = mathfunc.round_down_to_odd(int(period))
    window_len = max(3, window_len)

    if plot:
        fig, ax = plt.subplots(1, figsize=(8,5))
        ax.semilogy(freqs[freqs>1e-9], Pxx[freqs>1e-9], label='Full Spectrum (Welch)', color='grey', alpha=0.7)
        if len(valid_indices) > 0:
             ax.semilogy(freqs[valid_indices], Pxx[valid_indices], label='Considered Band', color='orange')
             if dominant_freq > 1e-9:
                ax.semilogy(dominant_freq, Pxx[valid_indices[peak_idx_in_valid]], 'rx', markersize=10,
                             label=f'Dominant (Period ~{period:.1f}px)')
        ax.set_xlabel(f'Frequency (1/pixels, fs={sampling_rate})')
        ax.set_ylabel('PSD'); ax.legend()
        ax.set_title(f'Power Spectrum for Window Estimation (Est. Window: {window_len})')
        ax.grid(True, linestyle=':'); plt.tight_layout(); plt.show()
    return window_len

def get_window_robust_targeted(
    y_axis,
    sampling_rate=1.0,
    plot=False,
    # Target period range for the desired window length
    target_min_period_pixels=10,
    target_max_period_pixels=20,
    # Fallback if no suitable peak in target range
    default_window_period=15,
    # Broader constraints for Welch and initial period estimation
    overall_min_period_pixels=3, # Smallest possible period to consider in PSD
    overall_max_period_pixels=None, # Largest, e.g., for nperseg estimation
    nperseg_min_periods=4,
    verbose=False
):
    if not isinstance(y_axis, np.ndarray):
        y_axis = np.asarray(y_axis)

    if len(y_axis) == 0: # Handle empty input
        print("Warning: y_axis is empty. Returning default_window_period.")
        return mathfunc.round_down_to_odd(default_window_period)
        
    if len(y_axis) < 2 * target_max_period_pixels : # Heuristic if signal is too short for target
        print(f"Warning: y_axis (len {len(y_axis)}) too short for target period {target_max_period_pixels}. "
              f"Using a fraction of y_axis length or default.")
        # Fallback to a smaller window if signal is very short
        # Or simply use default_window_period if that's preferred
        return mathfunc.round_down_to_odd(max(overall_min_period_pixels, min(target_min_period_pixels, len(y_axis) // 5)))


    # Determine overall_max_period_pixels for nperseg calculation if not provided
    if overall_max_period_pixels is None:
        overall_max_period_pixels = len(y_axis) / 3.0
    # Ensure overall_max is at least as large as target_max
    overall_max_period_pixels = max(overall_max_period_pixels, target_max_period_pixels, overall_min_period_pixels * 2.0)


    nperseg_val = min(len(y_axis), max(256, int(nperseg_min_periods * overall_max_period_pixels)))
    if nperseg_val <= 0:
        nperseg_val = min(len(y_axis), 256) # Safety for nperseg
    if nperseg_val > len(y_axis): # nperseg cannot be greater than signal length
        nperseg_val = len(y_axis)


    try:
        freqs, Pxx = welch(y_axis, fs=sampling_rate, nperseg=nperseg_val, scaling='density', window='hann')
    except ValueError as e:
        print(f"ValueError during Welch: {e}. Returning default_window_period.")
        return mathfunc.round_down_to_odd(default_window_period)

    if freqs.size == 0: # Welch returned no frequencies
        print("Warning: Welch returned no frequencies. Returning default_window_period.")
        return mathfunc.round_down_to_odd(default_window_period)

    # Define the target frequency range based on the target period range
    # Add epsilon to avoid division by zero or issues with exact boundaries
    target_max_freq = 1.0 / (target_min_period_pixels - 1e-9) if target_min_period_pixels > 1e-9 else np.inf
    target_min_freq = 1.0 / (target_max_period_pixels + 1e-9) if target_max_period_pixels > 1e-9 else 0

    # Find indices of frequencies within our *target* band
    # Also ensure freqs > 0 to avoid issues with 1/freq
    target_band_indices = np.where(
        (freqs >= target_min_freq) & (freqs <= target_max_freq) & (freqs > 1e-9)
    )[0]

    period_in_target_range_found = False
    selected_period = default_window_period # Initialize with fallback
    dominant_freq_in_target_band = 0

    if target_band_indices.size > 0 and Pxx[target_band_indices].size > 0:
        # Find the peak power within this specific target frequency band
        peak_idx_in_target_band = np.argmax(Pxx[target_band_indices])
        dominant_freq_in_target_band = freqs[target_band_indices[peak_idx_in_target_band]]

        if dominant_freq_in_target_band > 1e-9:
            selected_period = 1.0 / dominant_freq_in_target_band
            period_in_target_range_found = True
            if verbose: print(f"Found dominant peak in target range: Period = {selected_period:.2f} pixels (Freq = {dominant_freq_in_target_band:.3f})")
        else:
            print("Dominant frequency in target band is too low or zero.")
    else:
        print(f"No significant power or no frequencies found in the target period range ({target_min_period_pixels}-{target_max_period_pixels} pixels).")

    if not period_in_target_range_found:
        print(f"Falling back to default_window_period: {default_window_period} pixels.")
        selected_period = default_window_period
        # Optional: could also fall back to overall max power if default is not preferred,
        # but that might defeat the purpose of the target range.
        # For example, to use the previous logic as a fallback:
        # overall_min_freq = 1.0 / (overall_max_period_pixels + 1e-9)
        # overall_max_freq = 1.0 / (overall_min_period_pixels + 1e-9)
        # overall_valid_indices = np.where((freqs > overall_min_freq) & (freqs < overall_max_freq) & (freqs > 1e-9))[0]
        # if overall_valid_indices.size > 0 and Pxx[overall_valid_indices].size > 0:
        #     overall_peak_idx = np.argmax(Pxx[overall_valid_indices])
        #     overall_dominant_freq = freqs[overall_valid_indices[overall_peak_idx]]
        #     if overall_dominant_freq > 1e-9: selected_period = 1.0 / overall_dominant_freq


    # Final window length must still be an odd integer and within overall sensible bounds
    # Clip the selected_period to be within the target range itself, or if not found,
    # it will be the default_window_period (which should ideally be within or near target range).
    # The clipping below to overall_min/max_period_pixels is more of a safety for the final window_len.
    final_selected_period = np.clip(selected_period, target_min_period_pixels, target_max_period_pixels)
    # However, if the peak was outside, and you MUST use the target range, the above clip is correct.
    # If selected_period was a fallback (like default_window_period), this clip ensures it's within target.

    # If the goal was "if there's a peak in target, use it, otherwise use global max (clipped to overall)",
    # then the final_selected_period logic would be:
    # if period_in_target_range_found:
    #    final_selected_period = selected_period # Already from target band
    # else:
    #    # Recalculate global max period based on wider overall_min/max_period_pixels
    #    # This part is essentially your original function's logic
    #    overall_min_f = 1.0 / (overall_max_period_pixels + 1e-9)
    #    overall_max_f = 1.0 / (overall_min_period_pixels + 1e-9)
    #    overall_indices = np.where((freqs > overall_min_f) & (freqs < overall_max_f) & (freqs > 1e-9))[0]
    #    if overall_indices.size > 0:
    #        glob_peak_idx = np.argmax(Pxx[overall_indices])
    #        glob_dom_freq = freqs[overall_indices[glob_peak_idx]]
    #        final_selected_period = 1.0 / glob_dom_freq if glob_dom_freq > 1e-9 else default_window_period
    #    else:
    #        final_selected_period = default_window_period
    # final_selected_period = np.clip(final_selected_period, overall_min_period_pixels, overall_max_period_pixels)
    # The above commented block is if you want global peak as fallback.
    # The current simpler logic (using default_window_period as fallback and clipping to target) is:
    final_selected_period = np.clip(selected_period, target_min_period_pixels, target_max_period_pixels)


    window_len = mathfunc.round_down_to_odd(int(final_selected_period))
    window_len = max(3, window_len) # Ensure at least 3

    if plot:
        fig, ax = plt.subplots(1, figsize=(10, 6)) # Increased figure size for more details
        ax.semilogy(freqs[freqs > 1e-9], Pxx[freqs > 1e-9], label='Full Spectrum (Welch)', color='grey', alpha=0.5, zorder=1)

        # Highlight the target frequency band
        ax.axvspan(target_min_freq, target_max_freq, alpha=0.2, color='lightgreen', label=f'Target Freq. Band ({target_min_period_pixels}-{target_max_period_pixels} px period)')

        if target_band_indices.size > 0:
            ax.semilogy(freqs[target_band_indices], Pxx[target_band_indices], label='Power in Target Band', color='orange', zorder=2,linewidth=1.5)
            if dominant_freq_in_target_band > 1e-9 and period_in_target_range_found:
                ax.semilogy(dominant_freq_in_target_band, Pxx[target_band_indices[peak_idx_in_target_band]],
                            'rx', markersize=12, mew=2,
                            label=f'Peak in Target (Period ~{1.0/dominant_freq_in_target_band:.1f}px)', zorder=3)
        
        # Optionally, plot the global peak if it was different
        # This requires calculating it if not period_in_target_range_found
        # For simplicity, this part is omitted but could be added for diagnostics

        ax.set_xlabel(f'Frequency (1/pixels, fs={sampling_rate})')
        ax.set_ylabel('PSD'); ax.legend(loc='best') # Changed legend location
        ax.set_title(f'Window Estimation (Targeted: {target_min_period_pixels}-{target_max_period_pixels} px). Final Est. Window: {window_len} px', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)

        # Secondary x-axis for period
        def freq_to_period(f_arr):
            # Ensure f_arr is an array for vectorized operations
            f_arr = np.asarray(f_arr)
            # Handle potential zero or negative frequencies to avoid warnings/errors
            periods = np.full_like(f_arr, np.inf, dtype=float)
            valid_f = f_arr > 1e-9
            periods[valid_f] = 1.0 / f_arr[valid_f]
            return periods

        # secax = ax.secondary_xaxis('top', functions=(freq_to_period, freq_to_period))
        # secax.set_xlabel('Period (pixels)')

        # This section tries to set specific ticks, but the problem is often with the auto-derived limits before this.
        common_periods = np.array([overall_min_period_pixels, target_min_period_pixels, (target_min_period_pixels+target_max_period_pixels)/2 , target_max_period_pixels, overall_max_period_pixels])
        common_periods = np.unique(common_periods[common_periods > 0])
        # If common_periods is empty or results in inf after transformation by the axis itself, it can be an issue.
        # However, the error "Axis limits cannot be NaN or Inf" points to the automatic limit setting.
        # secax.set_xticks(common_periods)
        # secax.set_xticklabels([f"{p:.0f}" for p in common_periods])

        # plt.tight_layout(); 
        plt.show()
    return window_len

def robust_noise_std(y_data):
    """Estimates noise standard deviation using Median Absolute Deviation."""
    y_data = np.asarray(y_data)
    if y_data.size == 0:
        return 1e-6 # Avoid zero if empty
    # scale='normal' makes MAD comparable to Gaussian std
    mad_val = scipy.stats.median_abs_deviation(y_data, scale='normal')
    return mad_val if mad_val > 1e-9 else 1e-9 # Floor value

def robust_local_noise_std(y_data, center_idx, window_half_width, fallback_noise_std):
    """Estimates local noise std around center_idx, excluding a central region."""
    if y_data is None or len(y_data) == 0: return fallback_noise_std
    exclude_half_width = window_half_width // 3
    left_s=max(0,center_idx-window_half_width); left_e=max(0,center_idx-exclude_half_width)
    right_s=min(len(y_data),center_idx+exclude_half_width+1); right_e=min(len(y_data),center_idx+window_half_width+1)
    noise_samp=[];
    if left_e>left_s: noise_samp.extend(y_data[left_s:left_e])
    if right_e>right_s: noise_samp.extend(y_data[right_s:right_e])
    if len(noise_samp)<5: return fallback_noise_std # Need enough samples for MAD
    # Use nan_policy='omit' if y_data might contain NaNs
    mad_val=scipy.stats.median_abs_deviation(noise_samp,scale='normal', nan_policy='omit')
    return mad_val if mad_val>1e-9 else fallback_noise_std # Return fallback if MAD is effectively zero


def _fit_spacing_polynomial(extrema_x_coords, poly_degree=2):
    """Fits a polynomial S_exp(x_midpoint) = a*x_mid^2 + b*x_mid + c."""
    extrema_x_coords = np.asarray(extrema_x_coords)
    if len(extrema_x_coords) < poly_degree + 2 : return None, None # Need enough points
    spacings = np.diff(extrema_x_coords)
    if spacings.size == 0: return None, None # Need at least one spacing
    x_midpoints = (extrema_x_coords[:-1] + extrema_x_coords[1:]) / 2.0
    if len(x_midpoints) < poly_degree + 1: return None, None

    poly_features = PolynomialFeatures(degree=poly_degree, include_bias=True)
    try:
        X_poly = poly_features.fit_transform(x_midpoints.reshape(-1, 1))
    except ValueError: return None, None # If x_midpoints is problematic

    # Choose robust regressor based on number of points
    if len(X_poly) > X_poly.shape[1] * 2 and len(X_poly) > 5 :
        model = RANSACRegressor(random_state=42,
                                min_samples=max(X_poly.shape[1]+1, int(len(X_poly)*0.2)), # Ensure enough samples for model
                                residual_threshold=np.std(spacings)*0.75 if spacings.std() > 1e-6 else 1.0)
    else: model = HuberRegressor(epsilon=1.35)
    try: model.fit(X_poly, spacings); return model, poly_features
    except ValueError: return None, None


def _calculate_features_for_extrema_v3(
    all_extrema_x, all_extrema_y, all_extrema_types,
    original_y_data_for_noise, original_x_data_for_noise,
    noise_estimation_window_pixels, global_fallback_noise_std,
    spacing_poly_degree=2, spacing_uncertainty_const=2.0
):
    num_total_extrema = len(all_extrema_x)
    if num_total_extrema == 0: return np.array([]).reshape(0, 3)

    feature_x_coords = np.asarray(all_extrema_x)
    feature_prom_depth_snr = np.zeros(num_total_extrema)
    feature_spacing_dev_norm = np.zeros(num_total_extrema)
    raw_prom_depth_values = np.zeros(num_total_extrema)

    for i in range(num_total_extrema):
        x_i, y_i, type_i = all_extrema_x[i], all_extrema_y[i], all_extrema_types[i]
        opposite_type_mask = (all_extrema_types == -type_i)
        y_brackets = []
        if np.any(opposite_type_mask):
            opposite_x_cand = all_extrema_x[opposite_type_mask]; opposite_y_cand = all_extrema_y[opposite_type_mask]
            left_opp_indices = np.where(opposite_x_cand < x_i)[0]
            right_opp_indices = np.where(opposite_x_cand > x_i)[0]
            if left_opp_indices.size > 0:
                closest_left_idx_in_opposites = left_opp_indices[np.argmax(opposite_x_cand[left_opp_indices])]
                y_brackets.append(opposite_y_cand[closest_left_idx_in_opposites])
            if right_opp_indices.size > 0:
                closest_right_idx_in_opposites = right_opp_indices[np.argmin(opposite_x_cand[right_opp_indices])]
                y_brackets.append(opposite_y_cand[closest_right_idx_in_opposites])
        if y_brackets:
            raw_pd = (y_i - np.max(y_brackets)) if type_i == 1 else (np.min(y_brackets) - y_i)
            raw_prom_depth_values[i] = max(0, raw_pd)
        
        local_noise = global_fallback_noise_std
        if original_x_data_for_noise is not None and original_x_data_for_noise.size > 0:
            center_idx_in_orig = np.argmin(np.abs(original_x_data_for_noise - x_i))
            if original_y_data_for_noise is not None:
                local_noise = robust_local_noise_std(original_y_data_for_noise, center_idx_in_orig, 
                                                     noise_estimation_window_pixels // 2, global_fallback_noise_std)
        feature_prom_depth_snr[i] = raw_prom_depth_values[i] / local_noise if local_noise > 1e-9 else raw_prom_depth_values[i] / 1e-9

    sorted_indices_spacing = np.argsort(all_extrema_x)
    all_extrema_x_sorted = all_extrema_x[sorted_indices_spacing]
    spacing_model, spacing_poly_transformer = _fit_spacing_polynomial(all_extrema_x_sorted, poly_degree=spacing_poly_degree)

    if spacing_model is not None:
        temp_spacing_dev_norm_sorted = np.zeros(num_total_extrema)
        for i_sorted in range(num_total_extrema):
            x_i_s = all_extrema_x_sorted[i_sorted]; devs_norm_list = []
            if i_sorted > 0:
                s_bwd_act = x_i_s - all_extrema_x_sorted[i_sorted-1]; x_mid_bwd = (x_i_s + all_extrema_x_sorted[i_sorted-1])/2.0
                s_bwd_exp = spacing_model.predict(spacing_poly_transformer.transform(np.array([[x_mid_bwd]])))[0]
                devs_norm_list.append((s_bwd_act - s_bwd_exp) / spacing_uncertainty_const)
            if i_sorted < num_total_extrema - 1:
                s_fwd_act = all_extrema_x_sorted[i_sorted+1] - x_i_s; x_mid_fwd = (x_i_s + all_extrema_x_sorted[i_sorted+1])/2.0
                s_fwd_exp = spacing_model.predict(spacing_poly_transformer.transform(np.array([[x_mid_fwd]])))[0]
                devs_norm_list.append((s_fwd_act - s_fwd_exp) / spacing_uncertainty_const)
            if devs_norm_list: temp_spacing_dev_norm_sorted[i_sorted] = np.mean(np.abs(devs_norm_list))
        original_order_indices = np.argsort(sorted_indices_spacing)
        feature_spacing_dev_norm = temp_spacing_dev_norm_sorted[original_order_indices]
    else: print("    Warning: Spacing model fit failed. Spacing deviation feature will be zero.")
    return np.vstack([feature_x_coords, feature_prom_depth_snr, feature_spacing_dev_norm]).T

def _calculate_features_for_one_type( # NEW: Calculates features for one type (P or V)
    extrema_x_of_type, extrema_y_of_type_refined, # Extrema we are calculating features for
    opposite_extrema_x, opposite_extrema_y_refined, # Their opposites for prom/depth
    is_peak_type, # Boolean: True if extrema_x_of_type are peaks
    original_y_data_for_noise, original_x_data_for_noise,
    noise_estimation_window_pixels, global_fallback_noise_std,
    spacing_poly_degree=2, spacing_uncertainty_const=2.0
):
    num_extrema = len(extrema_x_of_type)
    if num_extrema == 0: return np.array([]).reshape(0, 3)

    feature_x_coords = np.asarray(extrema_x_of_type)
    feature_prom_depth_snr = np.zeros(num_extrema)
    feature_spacing_dev_norm = np.zeros(num_extrema)

    # Calculate Prominence/Depth S/N
    for i in range(num_extrema):
        x_i, y_i_refined = extrema_x_of_type[i], extrema_y_of_type_refined[i]
        
        raw_pd_val = 0
        if opposite_extrema_x.size > 0: # Check if opposites exist
            left_opp_indices = np.where(opposite_extrema_x < x_i)[0]
            right_opp_indices = np.where(opposite_extrema_x > x_i)[0]
            y_brackets_refined = []
            if left_opp_indices.size > 0:
                closest_left_idx = left_opp_indices[np.argmax(opposite_extrema_x[left_opp_indices])]
                y_brackets_refined.append(opposite_extrema_y_refined[closest_left_idx])
            if right_opp_indices.size > 0:
                closest_right_idx = right_opp_indices[np.argmin(opposite_extrema_x[right_opp_indices])]
                y_brackets_refined.append(opposite_extrema_y_refined[closest_right_idx])
            
            if y_brackets_refined:
                raw_pd_val = (y_i_refined - np.max(y_brackets_refined)) if is_peak_type else \
                             (np.min(y_brackets_refined) - y_i_refined)
                raw_pd_val = max(0, raw_pd_val)
        
        local_noise = global_fallback_noise_std
        if original_x_data_for_noise is not None and original_x_data_for_noise.size > 0:
            center_idx_in_orig = np.argmin(np.abs(original_x_data_for_noise - x_i))
            if original_y_data_for_noise is not None:
                local_noise = robust_local_noise_std(original_y_data_for_noise, center_idx_in_orig, 
                                                     noise_estimation_window_pixels // 2, global_fallback_noise_std)
        feature_prom_depth_snr[i] = raw_pd_val / local_noise if local_noise > 1e-9 else raw_pd_val / 1e-9

    # Calculate Spacing Deviation (using only extrema_x_of_type)
    spacing_model, spacing_poly_transformer = _fit_spacing_polynomial(extrema_x_of_type, poly_degree=spacing_poly_degree)
    if spacing_model is not None:
        for i in range(num_extrema):
            x_i = extrema_x_of_type[i]; devs_norm_list = []
            if i > 0:
                s_bwd_act = x_i - extrema_x_of_type[i-1]; x_mid_bwd = (x_i + extrema_x_of_type[i-1])/2.0
                s_bwd_exp_arr = predict_spacing(x_mid_bwd, spacing_model, spacing_poly_transformer)
                if s_bwd_exp_arr is not None and np.isfinite(s_bwd_exp_arr[0]): devs_norm_list.append(abs(s_bwd_act - s_bwd_exp_arr[0]) / spacing_uncertainty_const)
            if i < num_extrema - 1:
                s_fwd_act = extrema_x_of_type[i+1] - x_i; x_mid_fwd = (x_i + extrema_x_of_type[i+1])/2.0
                s_fwd_exp_arr = predict_spacing(x_mid_fwd, spacing_model, spacing_poly_transformer)
                if s_fwd_exp_arr is not None and np.isfinite(s_fwd_exp_arr[0]): devs_norm_list.append(abs(s_fwd_act - s_fwd_exp_arr[0]) / spacing_uncertainty_const)
            if devs_norm_list: feature_spacing_dev_norm[i] = np.mean(np.abs(devs_norm_list))
    else: print(f"    Warning: Spacing model fit failed for {'peaks' if is_peak_type else 'valleys'}. Spacing dev will be zero.")
    return np.vstack([feature_x_coords, feature_prom_depth_snr, feature_spacing_dev_norm]).T


def _ensure_alternation(peaks_xy_tuple, valleys_xy_tuple):
    """
    Ensures strict P-V-P-V alternation from separated peak/valley inputs.
    Input: peaks_xy_tuple = (peaks_x_array, peaks_y_array)
           valleys_xy_tuple = (valleys_x_array, valleys_y_array)
    Output: (alt_peaks_x, alt_peaks_y, alt_valleys_x, alt_valleys_y)
    """
    # Unpack inputs, handling potential empty lists/arrays gracefully
    if peaks_xy_tuple and len(peaks_xy_tuple) == 2 and hasattr(peaks_xy_tuple[0], '__len__'):
        peaks_x_np, peaks_y_np = np.asarray(peaks_xy_tuple[0]), np.asarray(peaks_xy_tuple[1])
    else:
        peaks_x_np, peaks_y_np = np.array([]), np.array([])

    if valleys_xy_tuple and len(valleys_xy_tuple) == 2 and hasattr(valleys_xy_tuple[0], '__len__'):
        valleys_x_np, valleys_y_np = np.asarray(valleys_xy_tuple[0]), np.asarray(valleys_xy_tuple[1])
    else:
        valleys_x_np, valleys_y_np = np.array([]), np.array([])

    if not peaks_x_np.size and not valleys_x_np.size:
        return np.array([]), np.array([]), np.array([]), np.array([])
    if not peaks_x_np.size:
        return np.array([]), np.array([]), valleys_x_np, valleys_y_np
    if not valleys_x_np.size:
        return peaks_x_np, peaks_y_np, np.array([]), np.array([])

    # Combine all features with a type indicator (1 for peak, -1 for valley)
    all_x = np.concatenate((peaks_x_np, valleys_x_np))
    all_y = np.concatenate((peaks_y_np, valleys_y_np))
    all_types = np.concatenate((np.ones(peaks_x_np.size), -np.ones(valleys_x_np.size)))

    # Sort by x-coordinate
    sorted_indices = np.argsort(all_x)
    sorted_x = all_x[sorted_indices]
    sorted_y = all_y[sorted_indices]
    sorted_types = all_types[sorted_indices]

    final_x_list, final_y_list, final_types_list = [], [], []

    if not sorted_x.size: # Should not happen if checks above passed
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Add the first feature
    final_x_list.append(sorted_x[0])
    final_y_list.append(sorted_y[0])
    final_types_list.append(sorted_types[0])

    for i in range(1, sorted_x.size):
        current_x, current_y, current_type = sorted_x[i], sorted_y[i], sorted_types[i]
        last_x, last_y, last_type = final_x_list[-1], final_y_list[-1], final_types_list[-1]

        if current_type != last_type: # Different type, so alternate; add it
            final_x_list.append(current_x)
            final_y_list.append(current_y)
            final_types_list.append(current_type)
        else: # Same type as previous (e.g., P-P or V-V)
            if current_type == 1: # Both are peaks
                if current_y > last_y: # Current peak is higher, replace previous
                    final_x_list[-1], final_y_list[-1] = current_x, current_y
            else: # Both are valleys
                if current_y < last_y: # Current valley is lower, replace previous
                    final_x_list[-1], final_y_list[-1] = current_x, current_y
    
    final_x_arr = np.array(final_x_list)
    final_y_arr = np.array(final_y_list)
    final_types_arr = np.array(final_types_list)

    # Separate back into peaks and valleys
    out_peaks_x = final_x_arr[final_types_arr == 1]
    out_peaks_y = final_y_arr[final_types_arr == 1]
    out_valleys_x = final_x_arr[final_types_arr == -1]
    out_valleys_y = final_y_arr[final_types_arr == -1]
    
    return out_peaks_x, out_peaks_y, out_valleys_x, out_valleys_y

def _ensure_alternation_and_tag(peaks_x, peaks_y, valleys_x, valleys_y):
    """
    Ensures strict P-V-P-V alternation AND returns a combined list with type tags,
    as well as separated lists.
    Type tags: +1 for peak, -1 for valley.
    """
    peaks_x_np, peaks_y_np = np.asarray(peaks_x), np.asarray(peaks_y)
    valleys_x_np, valleys_y_np = np.asarray(valleys_x), np.asarray(valleys_y)

    if not peaks_x_np.size and not valleys_x_np.size:
        return np.array([]), np.array([]), np.array([]), np.array([]), \
               np.array([]), np.array([]), np.array([])

    all_x_coords, all_y_coords, all_types_list = [], [], []
    if peaks_x_np.size > 0:
        all_x_coords.extend(peaks_x_np); all_y_coords.extend(peaks_y_np)
        all_types_list.extend([1] * peaks_x_np.size)
    if valleys_x_np.size > 0:
        all_x_coords.extend(valleys_x_np); all_y_coords.extend(valleys_y_np)
        all_types_list.extend([-1] * valleys_x_np.size)
    
    all_x, all_y, all_types = np.array(all_x_coords), np.array(all_y_coords), np.array(all_types_list)

    if not all_x.size:
        return np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

    sorted_indices = np.argsort(all_x)
    sorted_x, sorted_y, sorted_types = all_x[sorted_indices], all_y[sorted_indices], all_types[sorted_indices]

    final_x_list, final_y_list, final_types_list_out = [], [], []
    if sorted_x.size > 0:
        final_x_list.append(sorted_x[0]); final_y_list.append(sorted_y[0]); final_types_list_out.append(sorted_types[0])
        for i in range(1, sorted_x.size):
            cx,cy,ct = sorted_x[i],sorted_y[i],sorted_types[i]
            lt = final_types_list_out[-1]
            if ct != lt:
                final_x_list.append(cx); final_y_list.append(cy); final_types_list_out.append(ct)
            else:
                if ct == 1 and cy > final_y_list[-1]: final_x_list[-1],final_y_list[-1]=cx,cy
                elif ct == -1 and cy < final_y_list[-1]: final_x_list[-1],final_y_list[-1]=cx,cy
    
    combined_x_alt = np.array(final_x_list)
    combined_y_alt = np.array(final_y_list)
    combined_types_alt = np.array(final_types_list_out)
    
    out_peaks_x = combined_x_alt[combined_types_alt == 1]
    out_peaks_y = combined_y_alt[combined_types_alt == 1]
    out_valleys_x = combined_x_alt[combined_types_alt == -1]
    out_valleys_y = combined_y_alt[combined_types_alt == -1]
    
    return out_peaks_x, out_peaks_y, out_valleys_x, out_valleys_y, \
           combined_x_alt, combined_y_alt, combined_types_alt


def fit_envelope_interp_smooth(
    extrema_x, # Sorted x-coordinates of peaks or valleys
    extrema_y, # Corresponding y-values (refined to original data)
    full_x_axis, # The x-axis for which the envelope should be defined (e.g., original_x_axis)
    smooth_window_len_pixels=51,
    smooth_window_type='nuttall'
):
    """
    Estimates an envelope by:
    1. Linearly interpolating between the given extrema points.
    2. Smoothing the resulting interpolated function.

    Args:
        extrema_x (np.array): X-coordinates of the extrema (peaks or valleys), must be sorted.
        extrema_y (np.array): Y-values of the extrema.
        full_x_axis (np.array): The complete x-axis over which to calculate the envelope.
        smooth_window_len_pixels (int): Window length for smoothing the interpolated envelope.
                                        Should be odd.
        smooth_window_type (str): Type of window for smoothing (e.g., 'nuttall').

    Returns:
        np.array: The smoothed envelope y-values corresponding to full_x_axis.
                  Returns None if input is insufficient.
    """
    extrema_x = np.asarray(extrema_x)
    extrema_y = np.asarray(extrema_y)
    full_x_axis = np.asarray(full_x_axis)

    if len(extrema_x) < 2: # Need at least two points for interpolation
        # print("Warning: Not enough extrema to create interpolated envelope. Returning flat line or NaNs.")
        # Fallback: could be a constant (median of extrema_y) or NaNs
        if len(extrema_y) > 0:
            return np.full_like(full_x_axis, np.median(extrema_y), dtype=float)
        else:
            return np.full_like(full_x_axis, np.nan, dtype=float)

    # Ensure extrema_x are sorted for np.interp
    sort_indices = np.argsort(extrema_x)
    sorted_extrema_x = extrema_x[sort_indices]
    sorted_extrema_y = extrema_y[sort_indices]
    
    # Prevent issues if full_x_axis extends beyond sorted_extrema_x range by specifying left/right fill
    # Use the y-value of the first/last extremum for extrapolation
    jagged_envelope = np.interp(
        full_x_axis,
        sorted_extrema_x,
        sorted_extrema_y,
        left=sorted_extrema_y[0], # Fill value for x < min(sorted_extrema_x)
        right=sorted_extrema_y[-1] # Fill value for x > max(sorted_extrema_x)
    )

    # Ensure smooth_window_len is odd and appropriate
    # window_len = mathfunc.round_down_to_odd(smooth_window_len_pixels)
    # Use your actual mathfunc or a direct odd check:
    window_len = int(smooth_window_len_pixels)
    if window_len % 2 == 0: window_len +=1
    window_len = max(3, window_len)


    if len(jagged_envelope) < window_len :
        # print(f"Warning: Jagged envelope length ({len(jagged_envelope)}) < smooth window ({window_len}). Returning unsmoothed.")
        return jagged_envelope # Not enough points to smooth properly

    smoothed_envelope = _smooth(jagged_envelope, window_len=window_len, window=smooth_window_type, mode="same")
    
    return smoothed_envelope
from scipy.signal import savgol_filter # Add this import

def fit_envelope_savgol(
    extrema_x, extrema_y,
    full_x_axis, # Envelope will be defined on this grid
    savgol_window_len_pixels=51,
    savgol_polyorder=3,
    verbose=False
):
    extrema_x = np.asarray(extrema_x); extrema_y = np.asarray(extrema_y)
    full_x_axis = np.asarray(full_x_axis)

    if len(extrema_x) < 2: # Need at least two points for robust interpolation
        if verbose: print(f"  Warning: Not enough extrema ({len(extrema_x)}) to create Savgol envelope. Returning flat line or NaNs.")
        return np.full_like(full_x_axis, np.median(extrema_y) if len(extrema_y) > 0 else np.nan, dtype=float)

    # Ensure extrema_x are sorted for np.interp
    sort_indices = np.argsort(extrema_x)
    sorted_extrema_x = extrema_x[sort_indices]; sorted_extrema_y = extrema_y[sort_indices]
    
    # 1. Linear Interpolation to get a jagged envelope on the full_x_axis grid
    jagged_envelope = np.interp(
        full_x_axis, sorted_extrema_x, sorted_extrema_y,
        left=sorted_extrema_y[0], right=sorted_extrema_y[-1]
    )

    # 2. Smooth with Savitzky-Golay
    window_len = mathfunc.round_down_to_odd(int(savgol_window_len_pixels))
    # Savgol window must be odd and greater than polyorder
    if window_len <= savgol_polyorder:
        window_len = savgol_polyorder + 1 if savgol_polyorder % 2 == 0 else savgol_polyorder + 2
        window_len = mathfunc.round_down_to_odd(window_len) # Ensure odd
    window_len = max(3, window_len)


    if len(jagged_envelope) < window_len :
        if verbose: print(f"  Warning: Jagged envelope length ({len(jagged_envelope)}) < Savgol window ({window_len}). Returning unsmoothed jagged envelope.")
        return jagged_envelope

    try:
        # mode="interp" can sometimes be better for edges if data is not periodic
        # mode="mirror" is also a common choice
        smoothed_envelope = savgol_filter(jagged_envelope, window_length=window_len, polyorder=savgol_polyorder, mode="mirror")
    except ValueError as e:
        if verbose: print(f"  Warning: Savgol filter failed ({e}). Returning jagged envelope.")
        return jagged_envelope 
        
    return smoothed_envelope

def fit_envelope_polynomial(x_coords, y_coords, degree=3, robust=True):
    x_coords_arr = np.asarray(x_coords).reshape(-1, 1) # Ensure 2D for scaler
    y_coords_arr = np.asarray(y_coords)

    if len(x_coords_arr) < degree + 1:
        print(f"Warning: Not enough points ({len(x_coords_arr)}) to fit envelope of degree {degree}. Returning None.")
        return None, None, None # Model, PolyFeatures, Scaler

    # --- Normalize x_coords ---
    # MinMaxScaler to [0, 1] or [-1, 1] is common
    # StandardScaler (mean 0, std 1) also works well
    # Let's use MinMaxScaler to [-1, 1] for better behavior around origin if poly has offset
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = x_scaler.fit_transform(x_coords_arr)
    # --- End Normalization ---

    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    # Fit PolynomialFeatures on SCALED x data
    X_poly_scaled = poly_features.fit_transform(x_scaled)

    if robust:
        model = HuberRegressor(epsilon=1.35, max_iter=500) # Increase max_iter
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(fit_intercept=False) # PolynomialFeatures includes bias

    try:
        model.fit(X_poly_scaled, y_coords_arr)
        return model, poly_features, x_scaler # Return the scaler
    except ValueError as e:
        print(f"Error fitting envelope polynomial: {e}")
        return None, None, None
    except Exception as e_gen: # Catch other potential convergence issues
        print(f"General error/convergence failure fitting envelope: {e_gen}")
        return None, None, None


def predict_envelope(x_coords_new, model, poly_features_transformer, x_scaler):
    """Predicts envelope y-values for new x-coordinates, using the trained scaler."""
    if model is None or poly_features_transformer is None or x_scaler is None:
        return np.full_like(np.asarray(x_coords_new), np.nan)

    x_coords_new_arr = np.asarray(x_coords_new).reshape(-1, 1)
    # --- Scale new x_coords using the *same* scaler from fitting ---
    x_new_scaled = x_scaler.transform(x_coords_new_arr)
    # --- End Scaling ---

    X_new_poly_scaled = poly_features_transformer.transform(x_new_scaled)
    return model.predict(X_new_poly_scaled)
    
def predict_spacing(x_midpoints_new, spacing_model, spacing_poly_transformer):
    """Predicts expected spacing for new x_midpoints."""
    if spacing_model is None or spacing_poly_transformer is None:
        return np.full_like(np.asarray(x_midpoints_new), np.nan) # Default if no model
    X_new_poly = spacing_poly_transformer.transform(np.asarray(x_midpoints_new).reshape(-1, 1))
    return spacing_model.predict(X_new_poly)



def remove_false_triplets_v2( # Renamed for clarity
    all_extrema_x_in, all_extrema_y_in, all_extrema_types_in, # Combined, sorted, alternated
    max_triplet_x_span_pixels,
    min_prom_depth_factor_center, # For check against immediate opposites
    y_consistency_factor=0.2,     # For new check: y_center vs (y_prev_same, y_next_same)
                                  # y_center must be within [min - factor*diff, max + factor*diff]
    verbose=False
):
    """
    Identifies and removes spurious triplets (P-V-P or V-P-V).
    A triplet's central extremum is removed if:
    1. Its x-span is small AND
    2. Its prominence/depth relative to immediate opposites is small OR
    3. Its y-value is inconsistent with its further same-type neighbors.
    """
    if len(all_extrema_x_in) < 3:
        return all_extrema_x_in, all_extrema_y_in, all_extrema_types_in # No change, return as is

    # Work on copies
    current_x = list(all_extrema_x_in)
    current_y = list(all_extrema_y_in)
    current_types = list(all_extrema_types_in)
    
    indices_to_remove = [] # Store indices of central elements of false triplets
    
    # fig = plt.figure()
    # ax = fig.add_subplot()
    
    # Iterate up to the point where a triplet can still be formed
    for i in range(len(current_x) - 2):
        x1, y1, t1 = current_x[i], current_y[i], current_types[i]
        x2, y2, t2 = current_x[i+1], current_y[i+1], current_types[i+1]
        x3, y3, t3 = current_x[i+2], current_y[i+2], current_types[i+2]

        # Check for an alternating triplet (P-V-P or V-P-V)
        # This should always be true if input is already alternated, but good check.
        if not (t1 == t3 and t1 != t2):
            continue 

        triplet_x_span = x3 - x1
        is_potential_false_triplet = False

        if triplet_x_span < max_triplet_x_span_pixels:
            # --- Check 1: Prominence/Depth of central E2 relative to E1, E3 ---
            center_is_weak = False
            if t1 == 1: # P1-V2-P3, check depth of V2
                prominence_of_valley = min(y1 - y2, y3 - y2)
                amplitude_span_outer_peaks = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3), 1e-6)
                if prominence_of_valley < min_prom_depth_factor_center * amplitude_span_outer_peaks or prominence_of_valley <= 0:
                    center_is_weak = True
                    if verbose: print(f"  Triplet P-V-P at x~{x2:.1f} (span {triplet_x_span:.1f}): Center V2 (y={y2:.2f}) weak prom={prominence_of_valley:.2f} rel to P1={y1:.2f}, P3={y3:.2f}.")
            else: # V1-P2-V3, check prominence of P2
                effective_prominence = min(y2 - y1, y2 - y3)
                amplitude_span_outer_valleys = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3), 1e-6)
                if effective_prominence < min_prom_depth_factor_center * amplitude_span_outer_valleys or effective_prominence <=0:
                    center_is_weak = True
                    if verbose: print(f"  Triplet V-P-V at x~{x2:.1f} (span {triplet_x_span:.1f}): Center P2 (y={y2:.2f}) weak prom={effective_prominence:.2f} rel to V1={y1:.2f}, V3={y3:.2f}.")

            # --- Check 2: Y-value consistency of E2 with further same-type neighbors ---
            center_y_inconsistent = False
            if not center_is_weak: # Only do this check if it wasn't already flagged as weak
                # Find E0 (same type as E2, before E1) and E4 (same type as E2, after E3)
                # This requires searching the *original unfiltered list* before any removals in this loop
                
                # Find previous same-type extremum (E0)
                e0_idx = -1
                for k in range(i - 1, -1, -1): # Search backwards from E1
                    if all_extrema_types_in[k] == t2: # t2 is type of central element E2
                        e0_idx = k
                        break
                
                # Find next same-type extremum (E4)
                e4_idx = -1
                for k in range(i + 3, len(all_extrema_x_in)): # Search forwards from E3
                    if all_extrema_types_in[k] == t2:
                        e4_idx = k
                        break
                
                if e0_idx != -1 and e4_idx != -1: # Both E0 and E4 exist
                    y0 = all_extrema_y_in[e0_idx]
                    y4 = all_extrema_y_in[e4_idx]
                    
                    min_y_outer_same = min(y0, y4)
                    max_y_outer_same = max(y0, y4)
                    y_diff_outer_same = abs(y0 - y4)
                    
                    lower_bound = min_y_outer_same - y_consistency_factor * y_diff_outer_same
                    upper_bound = max_y_outer_same + y_consistency_factor * y_diff_outer_same
                    
                    if not (lower_bound <= y2 <= upper_bound):
                        center_y_inconsistent = True
                        if verbose: print(f"  Triplet ...E2... at x~{x2:.1f}: Center E2 (y={y2:.2f}, type={t2}) y-inconsistent "
                                          f"with same-type neighbors E0(y={y0:.2f}), E4(y={y4:.2f}). Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                # else: Not enough same-type neighbors to perform this check, so it passes this sub-check.

            if center_is_weak or center_y_inconsistent:
                is_potential_false_triplet = True

            if is_potential_false_triplet:
                # Mark the central element for removal
                # We use original indices from all_extrema_x_in to mark on a boolean mask
                # Find the index in all_extrema_x_in that corresponds to current_x[i+1]
                # This is tricky if current_x has already been modified.
                # Better to operate on a boolean mask of the original input.
                indices_to_remove.append(i+1) # This index is for the 'current_x' list *as it's being processed*
                                              # This needs to be carefully handled if list is modified in place.
                                              # Using a boolean mask on original is safer.

    # --- Safer removal using a boolean mask on the input arrays ---
    if len(all_extrema_x_in) == 0 : # Should have been caught earlier
        return np.array([]), np.array([]), np.array([]), np.array([]) \
               ,np.array([]), np.array([]), np.array([])


    indices_to_keep_mask = np.ones(len(all_extrema_x_in), dtype=bool)
    
    # Re-iterate for removal logic, this time operating on the original input via mask
    # This is safer than modifying the list being iterated over.
    current_x_orig = all_extrema_x_in
    current_y_orig = all_extrema_y_in
    current_types_orig = all_extrema_types_in

    processed_central_indices = set() # To avoid re-evaluating a removed center

    for i in range(len(current_x_orig) - 2):
        if i+1 in processed_central_indices:
            continue

        x1, y1, t1 = current_x_orig[i], current_y_orig[i], current_types_orig[i]
        x2, y2, t2 = current_x_orig[i+1], current_y_orig[i+1], current_types_orig[i+1]
        x3, y3, t3 = current_x_orig[i+2], current_y_orig[i+2], current_types_orig[i+2]

        if not (t1 == t3 and t1 != t2):
            continue

        triplet_x_span = x3 - x1
        remove_center = False

        if triplet_x_span < max_triplet_x_span_pixels:
            center_is_weak = False
            if t1 == 1: # P1-V2-P3
                prom_of_V2 = min(y1 - y2, y3 - y2)
                amp_span_P = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3), 1e-6)
                if prom_of_V2 < min_prom_depth_factor_center * amp_span_P or prom_of_V2 <= 0:
                    center_is_weak = True
            else: # V1-P2-V3
                eff_prom_P2 = min(y2 - y1, y2 - y3)
                amp_span_V = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3), 1e-6)
                if eff_prom_P2 < min_prom_depth_factor_center * amp_span_V or eff_prom_P2 <=0:
                    center_is_weak = True
            
            if center_is_weak:
                remove_center = True
                if verbose: print(f"  Triplet at x~{x2:.1f}: Center E2 (y={y2:.2f}) weak prominence/depth. Marked for removal.")
            else: # Center not weak by first criterion, check y-consistency
                e0_idx, e4_idx = -1, -1
                # Search in original full list (all_extrema_x_in) for same-type neighbors
                # This search needs to be based on the original indices, not 'i'
                # Find original index of current_x_orig[i]
                original_idx_of_e1 = np.where(all_extrema_x_in == x1)[0] # Assuming x are unique enough
                original_idx_of_e3 = np.where(all_extrema_x_in == x3)[0]
                if not (original_idx_of_e1.size > 0 and original_idx_of_e3.size > 0): continue # Should not happen

                for k in range(original_idx_of_e1[0] - 1, -1, -1):
                    if all_extrema_types_in[k] == t2: e0_idx = k; break
                for k in range(original_idx_of_e3[0] + 1, len(all_extrema_x_in)):
                    if all_extrema_types_in[k] == t2: e4_idx = k; break
                
                if e0_idx != -1 and e4_idx != -1:
                    y0, y4 = all_extrema_y_in[e0_idx], all_extrema_y_in[e4_idx]
                    min_y_outer, max_y_outer = min(y0,y4), max(y0,y4)
                    y_diff_outer = abs(y0-y4)
                    lower_b = min_y_outer - y_consistency_factor * y_diff_outer
                    upper_b = max_y_outer + y_consistency_factor * y_diff_outer
                    if not (lower_b <= y2 <= upper_b):
                        remove_center = True
                        if verbose: print(f"  Triplet at x~{x2:.1f}: Center E2 (y={y2:.2f}) y-inconsistent. Marked for removal.")
            
            if remove_center:
                # [ax.scatter(x_,y_,c='k',marker='.') for (x_,y_) in [(x1,y1),(x2,y2),(x3,y3)]]
                # ax.scatter(x2,y2,marker='x',c='r')
                indices_to_keep_mask[np.where(all_extrema_x_in == x2)[0][0]] = False # Mark central for removal
                processed_central_indices.add(i+1) # Mark this original central index as processed

    filtered_x_final = all_extrema_x_in[indices_to_keep_mask]
    filtered_y_final = all_extrema_y_in[indices_to_keep_mask]
    filtered_types_final = all_extrema_types_in[indices_to_keep_mask]

    # Final alternation pass
    # The _ensure_alternation_and_tag also separates, so we can use its separated output
    peaks_x_out, peaks_y_out, valleys_x_out, valleys_y_out, \
    comb_x_out, comb_y_out, comb_types_out = _ensure_alternation_and_tag(
        filtered_x_final[filtered_types_final==1], filtered_y_final[filtered_types_final==1],
        filtered_x_final[filtered_types_final==-1], filtered_y_final[filtered_types_final==-1]
    )
    
    return peaks_x_out, peaks_y_out, valleys_x_out, valleys_y_out, \
           comb_x_out, comb_y_out, comb_types_out
          


def filter_lfc_extrema_v8_final_rules_with_plots(
    peaks_x_cand, peaks_y_cand_refined,
    valleys_x_cand, valleys_y_cand_refined,
    original_x_axis, original_y_axis,
    # Parameters
    spacing_poly_degree=2, 
    spacing_uncertainty_const=2.0, 
    spacing_max_dev_factor=3.0,
    envelope_savgol_window=51, 
    envelope_savgol_polyorder=3,
    ratio_trend_savgol_window=101, 
    ratio_trend_savgol_polyorder=2,
    envelope_ratio_max_abs_dev_from_trend = 0.15,
    peak_snr_min_thresh_poisson=3.0,
    noise_estimation_window_pixels=15, 
    
    global_fallback_noise_std=None,
    plot_filter_diagnostics=False, 
    verbose=False
):
    # --- 0. Initial Setup & Alternation ---
    if global_fallback_noise_std is None:
        global_fallback_noise_std = robust_noise_std(original_y_axis - np.median(original_y_axis))

    current_peaks_x, current_peaks_y, \
    current_valleys_x, current_valleys_y = _ensure_alternation(
        (peaks_x_cand, peaks_y_cand_refined),
        (valleys_x_cand, valleys_y_cand_refined)
    )
    
            
    
    if verbose: print(f"  RuleFilter v8 Input: {len(current_peaks_x)}P, {len(current_valleys_x)}V (Y refined)")

    plot_cand_peaks_x_alt = np.copy(current_peaks_x)
    plot_cand_peaks_y_alt = np.copy(current_peaks_y)
    plot_cand_valleys_x_alt = np.copy(current_valleys_x)
    plot_cand_valleys_y_alt = np.copy(current_valleys_y)

    if not current_peaks_x.size or not current_valleys_x.size:
        print("  Not enough peaks OR valleys after initial alternation for v8 filtering.")
        plot_data_ph = (None,None,None, np.array([]),np.array([]),np.array([]))
        return current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y, plot_data_ph

    # --- 1. Fit Global Spacing Models ---
    spacing_model_p, spacing_poly_trans_p = _fit_spacing_polynomial(current_peaks_x, poly_degree=spacing_poly_degree)
    spacing_model_v, spacing_poly_trans_v = _fit_spacing_polynomial(current_valleys_x, poly_degree=spacing_poly_degree)

    # --- 2. Fit Global Savgol Envelopes (on original_x_axis grid) ---
    peak_envelope = fit_envelope_savgol(current_peaks_x, current_peaks_y, original_x_axis, envelope_savgol_window, envelope_savgol_polyorder, verbose)
    valley_envelope = fit_envelope_savgol(current_valleys_x, current_valleys_y, original_x_axis, envelope_savgol_window, envelope_savgol_polyorder, verbose)
    if peak_envelope is None and verbose: print("  Warning: Failed to create peak envelope.")
    if valley_envelope is None and verbose: print("  Warning: Failed to create valley envelope.")
    
    # --- 3. Calculate and Smooth the V_env(x) / P_env(x) Ratio Trend ---
    R_exp_trend = None
    if peak_envelope is not None and valley_envelope is not None and \
       np.all(np.isfinite(peak_envelope)) and np.all(np.isfinite(valley_envelope)):
        P_env_for_ratio = np.maximum(peak_envelope, 1e-6 * (np.nanmax(peak_envelope) if np.any(np.isfinite(peak_envelope)) and np.nanmax(peak_envelope) > 0 else 1.0) )
        R_observed_globally = valley_envelope / P_env_for_ratio
        
        ratio_win = mathfunc.round_down_to_odd(int(ratio_trend_savgol_window))
        ratio_poly = int(ratio_trend_savgol_polyorder)
        if ratio_win <= ratio_poly: ratio_win = ratio_poly + (2 - ratio_poly % 2)
        
        if len(R_observed_globally) >= ratio_win:
            try: R_exp_trend = savgol_filter(R_observed_globally, window_length=ratio_win, polyorder=ratio_poly, mode="mirror")
            except ValueError: R_exp_trend = R_observed_globally 
        else: R_exp_trend = R_observed_globally
    if R_exp_trend is None and verbose: print("  Warning: Could not compute R_exp_trend for envelope ratio criterion.")

    # --- Data structures for storing metrics ---
    peak_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'env_ratio_exp':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}
    valley_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'env_ratio_exp':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}
    
    # --- Filter Peaks ---
    for i in range(len(current_peaks_x)):
        xp, yp = current_peaks_x[i], current_peaks_y[i]
        peak_metrics['x'].append(xp); peak_metrics['y'].append(yp)
        pass_pos, pass_snr_p, pass_spacing_p, pass_ratio_p = True, True, True, True
        if yp <= 1e-9: pass_pos = False
        current_snr_P_val, current_spacing_dev_p_val, obs_r_peak_val, exp_r_peak_val = np.nan, np.nan, np.nan, np.nan

        if pass_pos: # S/N for positive peaks
            
            signal_P = yp - np.interp(xp, original_x_axis, valley_envelope) 
            signal_P = max(0,signal_P)
            
            # Define a window size over which to calculate S/N
            wsize = 15
            y_in_window = original_y_axis[xp - wsize : xp + wsize+1]
            mask        = mathfunc.sigclip1d_biased_low(y_in_window,
                                                        sigma_lower=3, 
                                                        sigma_upper=1.5,
                                                        plot=False)
            y_clipped   = y_in_window[mask]
            
            noise_approx_P = np.std(y_clipped)
            
            
            if noise_approx_P>1e-9:
                current_snr_P_val = signal_P/noise_approx_P
            else: current_snr_P_val = signal_P/1e-9
            # print(f'{xp:<4d}{signal_P=:>12.3f}\t{noise_approx_P=:>12.3f}{current_snr_P_val=:>8.3f}')
            if current_snr_P_val < peak_snr_min_thresh_poisson: pass_snr_p = False
        else: pass_snr_p = False
        peak_metrics['snr'].append(current_snr_P_val)

        if spacing_model_p: # Spacing
            devs=[];s_b_e,s_f_e=np.nan,np.nan
            if i>0:
                s_b_a=xp-current_peaks_x[i-1];
                xm_b=(xp+current_peaks_x[i-1])/2;
                s_b_e_arr=predict_spacing(xm_b,spacing_model_p,spacing_poly_trans_p)
                if s_b_e_arr is not None and np.isfinite(s_b_e_arr[0]):
                    s_b_e=s_b_e_arr[0];
                    devs.append(abs(s_b_a-s_b_e)/spacing_uncertainty_const)
            if i<len(current_peaks_x)-1:
                s_f_a=current_peaks_x[i+1]-xp;
                xm_f=(xp+current_peaks_x[i+1])/2;
                s_f_e_arr=predict_spacing(xm_f,spacing_model_p,spacing_poly_trans_p);
                if s_f_e_arr is not None and np.isfinite(s_f_e_arr[0]):
                    s_f_e=s_f_e_arr[0];
                    devs.append(abs(s_f_a-s_f_e)/spacing_uncertainty_const)
            if devs:current_spacing_dev_p_val=np.mean(devs)
            else:current_spacing_dev_p_val=0 
            if current_spacing_dev_p_val > spacing_max_dev_factor: pass_spacing_p=False
        peak_metrics['spacing_dev'].append(current_spacing_dev_p_val)
        
        if R_exp_trend is not None and peak_envelope is not None and valley_envelope is not None: # Envelope Ratio
            P_env_val = np.interp(xp, original_x_axis, peak_envelope)
            V_env_val = np.interp(xp, original_x_axis, valley_envelope)
            exp_r_peak_val = np.interp(xp, original_x_axis, R_exp_trend)
            if np.isfinite(P_env_val) and np.isfinite(V_env_val) and np.isfinite(exp_r_peak_val):
                if yp <= V_env_val : pass_ratio_p = False
                if pass_ratio_p and P_env_val > V_env_val and yp < (V_env_val + 0.3 * (P_env_val - V_env_val)): pass_ratio_p = False
                if pass_ratio_p and abs(P_env_val) > 1e-6:
                    obs_r_peak_val = V_env_val / P_env_val
                    if abs(obs_r_peak_val - exp_r_peak_val) > envelope_ratio_max_abs_dev_from_trend: pass_ratio_p = False
                elif pass_ratio_p : pass_ratio_p = False
            else: pass_ratio_p = False
        else: pass_ratio_p = True # Cannot test
        peak_metrics['env_ratio_obs'].append(obs_r_peak_val); peak_metrics['env_ratio_exp'].append(exp_r_peak_val)
        peak_metrics['pass_snr'].append(pass_pos and pass_snr_p); peak_metrics['pass_spacing'].append(pass_spacing_p); peak_metrics['pass_ratio'].append(pass_ratio_p)
        peak_metrics['pass_all_rules'].append(pass_pos and pass_snr_p and pass_spacing_p and pass_ratio_p)

    # Filter Valleys (No S/N, uses R_exp_trend for ratio check)
    # ... (Valley filtering logic similar to peaks, but no S/N check. Populates valley_metrics) ...
    for i in range(len(current_valleys_x)):
        xv, yv = current_valleys_x[i], current_valleys_y[i]
        valley_metrics['x'].append(xv); valley_metrics['y'].append(yv)
        pass_spacing_v, pass_ratio_v = True, True
        valley_metrics['snr'].append(np.nan); valley_metrics['pass_snr'].append(True) # No S/N for valleys
        current_spacing_dev_v_val=np.nan; obs_r_valley_val=np.nan; exp_r_valley_val=np.nan
        if spacing_model_v: # Spacing logic for valleys
            devs_v=[];s_b_e_v,s_f_e_v=np.nan,np.nan
            if i>0:
                s_b_a=xv-current_valleys_x[i-1];
                xm_b=(xv+current_valleys_x[i-1])/2;
                s_e_arr=predict_spacing(xm_b,spacing_model_v,spacing_poly_trans_v);
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):
                    s_b_e_v=s_e_arr[0];
                    devs_v.append(abs(s_b_a-s_b_e_v)/spacing_uncertainty_const)
            if i<len(current_valleys_x)-1:
                s_f_a=current_valleys_x[i+1]-xv;
                xm_f=(xv+current_valleys_x[i+1])/2;
                s_e_arr=predict_spacing(xm_f,spacing_model_v,spacing_poly_trans_v);
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):
                    s_f_e_v=s_e_arr[0];
                    devs_v.append(abs(s_f_a-s_f_e_v)/spacing_uncertainty_const)
            if devs_v:current_spacing_dev_v_val=np.mean(devs_v)
            else:current_spacing_dev_v_val=0
            if current_spacing_dev_v_val > spacing_max_dev_factor: pass_spacing_v=False
        valley_metrics['spacing_dev'].append(current_spacing_dev_v_val)
        if R_exp_trend is not None and peak_envelope is not None and valley_envelope is not None: # Envelope Ratio for Valleys
            P_env_val = np.interp(xv, original_x_axis, peak_envelope)
            V_env_val = np.interp(xv, original_x_axis, valley_envelope)
            exp_r_valley_val = np.interp(xv, original_x_axis, R_exp_trend)
            if np.isfinite(P_env_val) and np.isfinite(V_env_val) and np.isfinite(exp_r_valley_val):
                if yv >= P_env_val: pass_ratio_v = False
                if pass_ratio_v and P_env_val > V_env_val and yv > (V_env_val + 0.7*(P_env_val-V_env_val)): pass_ratio_v = False
                if pass_ratio_v and abs(P_env_val) > 1e-6:
                    obs_r_valley_val = V_env_val / P_env_val
                    if abs(obs_r_valley_val - exp_r_valley_val) > envelope_ratio_max_abs_dev_from_trend: pass_ratio_v = False
                elif pass_ratio_v: pass_ratio_v = False
            else: pass_ratio_v = False
        else: pass_ratio_v = True
        valley_metrics['env_ratio_obs'].append(obs_r_valley_val); valley_metrics['env_ratio_exp'].append(exp_r_valley_val if 'exp_r_valley_val' in locals() and np.isfinite(exp_r_valley_val) else np.nan)
        valley_metrics['pass_spacing'].append(pass_spacing_v); valley_metrics['pass_ratio'].append(pass_ratio_v)
        valley_metrics['pass_all_rules'].append(pass_spacing_v and pass_ratio_v) # Only spacing and ratio for valleys

    for key in peak_metrics: peak_metrics[key] = np.array(peak_metrics[key])
    for key in valley_metrics: valley_metrics[key] = np.array(valley_metrics[key])

    rule_kept_peaks_x = peak_metrics['x'][peak_metrics['pass_all_rules']]
    rule_kept_peaks_y = peak_metrics['y'][peak_metrics['pass_all_rules']]
    rule_kept_valleys_x = valley_metrics['x'][valley_metrics['pass_all_rules']]
    rule_kept_valleys_y = valley_metrics['y'][valley_metrics['pass_all_rules']]
    if verbose: print(f"  After rule-based pre-filter: {len(rule_kept_peaks_x)}P, {len(rule_kept_valleys_x)}V")

    # --- Diagnostic Plots for Rule-Based Filter ---
    if plot_filter_diagnostics:
        fig, axs = plt.subplots(4, 1, figsize=(15,16), sharex=True)
        fig.suptitle("Rule-Based Filter Diagnostics (Savgol Ratio Trend)", fontsize=16)
        xpr = original_x_axis

        # Panel 1: Data, Envelopes, Final Selection
        axs[0].plot(original_x_axis,original_y_axis,c='grey',alpha=0.4,label="Orig")
        if peak_envelope is not None:axs[0].plot(original_x_axis,peak_envelope,'b-',alpha=0.6,lw=1.5,label="P Env (Savgol)")
        if valley_envelope is not None:axs[0].plot(original_x_axis,valley_envelope,'g-',alpha=0.6,lw=1.5,label="V Env (Savgol)")
        axs[0].scatter(plot_cand_peaks_x_alt,plot_cand_peaks_y_alt,c='lightcoral',marker='^',s=20,alpha=0.5,label="P Cands")
        axs[0].scatter(plot_cand_valleys_x_alt,plot_cand_valleys_y_alt,c='lightgreen',marker='v',s=20,alpha=0.5,label="V Cands")
        if peak_metrics.get('x',np.array([])).size>0 and peak_metrics.get('pass_all_rules',np.array([])).size>0 and np.any(peak_metrics['pass_all_rules']):axs[0].scatter(peak_metrics['x'][peak_metrics['pass_all_rules']],peak_metrics['y'][peak_metrics['pass_all_rules']],c='blue',marker='^',s=40,label="Rule-Kept P")
        if valley_metrics.get('x',np.array([])).size>0 and valley_metrics.get('pass_all_rules',np.array([])).size>0 and np.any(valley_metrics['pass_all_rules']):axs[0].scatter(valley_metrics['x'][valley_metrics['pass_all_rules']],valley_metrics['y'][valley_metrics['pass_all_rules']],c='darkgreen',marker='v',s=40,label="Rule-Kept V")
        axs[0].set_title("Data, Envelopes, Rule-Filtered");axs[0].set_ylabel("Flux");axs[0].legend(fontsize='small',loc='upper right');axs[0].grid(True,ls=':')
        
        # Panel 2: S/N (Peaks only)
        if peak_metrics.get('x',np.array([])).size>0:
            axs[1].scatter(peak_metrics['x'][peak_metrics['pass_snr']],peak_metrics['snr'][peak_metrics['pass_snr']],c='b',marker='^',s=30,alpha=0.7,label="P S/N Pass")
            axs[1].scatter(peak_metrics['x'][~peak_metrics['pass_snr']],peak_metrics['snr'][~peak_metrics['pass_snr']],c='r',marker='x',s=50,label="P S/N Fail")
        axs[1].axhline(peak_snr_min_thresh_poisson,c='k',ls='--',label="P S/N Thresh");axs[1].set_title("Peak S/N ((yp - Venv) / noise_approx)");axs[1].set_ylabel("S/N");axs[1].legend(fontsize='small');axs[1].grid(True,ls=':');axs[1].set_yscale('symlog',linthresh=1,linscale=0.5)
        
        # Panel 3: Spacing Deviation
        if peak_metrics.get('x',np.array([])).size>0:axs[2].scatter(peak_metrics['x'][peak_metrics['pass_spacing']],peak_metrics['spacing_dev'][peak_metrics['pass_spacing']],c='b',marker='^',s=30,alpha=0.7,label="P Spacing Pass");axs[2].scatter(peak_metrics['x'][~peak_metrics['pass_spacing']],peak_metrics['spacing_dev'][~peak_metrics['pass_spacing']],c='r',marker='x',s=50,label="P Spacing Fail")
        if valley_metrics.get('x',np.array([])).size>0:axs[2].scatter(valley_metrics['x'][valley_metrics['pass_spacing']],valley_metrics['spacing_dev'][valley_metrics['pass_spacing']],c='g',marker='v',s=30,alpha=0.7,label="V Spacing Pass");axs[2].scatter(valley_metrics['x'][~valley_metrics['pass_spacing']],valley_metrics['spacing_dev'][~valley_metrics['pass_spacing']],c='m',marker='x',s=50,label="V Spacing Fail")
        axs[2].axhline(spacing_max_dev_factor,c='k',ls='--',label="Max Spacing Dev");axs[2].set_title("Norm Spacing Dev");axs[2].set_ylabel("Norm Spacing Dev");axs[2].legend(fontsize='small');axs[2].grid(True,ls=':');axs[2].set_yscale('log')
        
        # Panel 4: Envelope Ratio Deviation
        if R_exp_trend is not None and original_x_axis.size > 0: axs[3].plot(original_x_axis, R_exp_trend,'k:',alpha=0.8,label="Smoothed Venv/Penv Trend (R_exp)")
        if peak_metrics.get('x',np.array([])).size > 0 :
            valid_p_ratios = np.isfinite(peak_metrics['env_ratio_obs']) & np.isfinite(peak_metrics['env_ratio_exp'])
            if np.any(valid_p_ratios): # Only plot if there's valid data
                peak_ratio_devs = np.abs(peak_metrics['env_ratio_obs'][valid_p_ratios] - peak_metrics['env_ratio_exp'][valid_p_ratios])
                axs[3].scatter(peak_metrics['x'][valid_p_ratios][peak_metrics['pass_ratio'][valid_p_ratios]], peak_ratio_devs[peak_metrics['pass_ratio'][valid_p_ratios]], c='b', marker='^', s=30, alpha=0.7, label="P Ratio Pass")
                axs[3].scatter(peak_metrics['x'][valid_p_ratios][~peak_metrics['pass_ratio'][valid_p_ratios]], peak_ratio_devs[~peak_metrics['pass_ratio'][valid_p_ratios]],c='r', marker='x', s=50, label="P Ratio Fail")
        if valley_metrics.get('x',np.array([])).size > 0:
            valid_v_ratios = np.isfinite(valley_metrics['env_ratio_obs']) & np.isfinite(valley_metrics['env_ratio_exp'])
            if np.any(valid_v_ratios):
                valley_ratio_devs = np.abs(valley_metrics['env_ratio_obs'][valid_v_ratios] - valley_metrics['env_ratio_exp'][valid_v_ratios])
                axs[3].scatter(valley_metrics['x'][valid_v_ratios][valley_metrics['pass_ratio'][valid_v_ratios]], valley_ratio_devs[valley_metrics['pass_ratio'][valid_v_ratios]], c='g', marker='v', s=30, alpha=0.7, label="V Ratio Pass")
                axs[3].scatter(valley_metrics['x'][valid_v_ratios][~valley_metrics['pass_ratio'][valid_v_ratios]], valley_ratio_devs[~valley_metrics['pass_ratio'][valid_v_ratios]],c='m', marker='x', s=50, label="V Ratio Fail")
        axs[3].axhline(envelope_ratio_max_abs_dev_from_trend, c='k', ls='--', label="Max Ratio Dev")
        axs[3].set_title("Abs Dev from Smoothed Venv/Penv Ratio Trend");axs[3].set_ylabel("|Obs V/P - R_exp|");axs[3].legend(fontsize='small');axs[3].grid(True,ls=':');axs[3].set_xlabel("X-coord")
        
        plt.tight_layout(rect=[0,0,1,0.97]);plt.show()


    dbscan_peaks_plot_data=(None,None,None, np.copy(rule_kept_peaks_x), np.copy(rule_kept_peaks_y), np.ones_like(rule_kept_peaks_x))
    dbscan_valleys_plot_data=(None,None,None, np.copy(rule_kept_valleys_x), np.copy(rule_kept_valleys_y), -np.ones_like(rule_kept_valleys_x))

    fpxa,fpya,fvxa,fvya = rule_kept_peaks_x, rule_kept_peaks_y, \
                          rule_kept_valleys_x, rule_kept_valleys_y
    plot_data_for_orchestrator_final = (dbscan_peaks_plot_data, dbscan_valleys_plot_data)
    return fpxa,fpya,fvxa,fvya, plot_data_for_orchestrator_final





def detect_initial_extrema_candidates(y_smoothed, x_coords_for_smoothed,
                                      deriv_method='coeff'):
    """
    Detects initial peak and valley candidates from smoothed data using derivatives.
    """
    if not isinstance(y_smoothed, np.ndarray): y_smoothed = np.asarray(y_smoothed)
    if not isinstance(x_coords_for_smoothed, np.ndarray): x_coords_for_smoothed = np.asarray(x_coords_for_smoothed)

    if len(y_smoothed) < 3: # Need at least 3 points for derivatives
        print("Warning in detect_initial_extrema_candidates: Smoothed data too short.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    derivative1st = mathfunc.derivative1d(y_smoothed, x=None, order=1, method=deriv_method)
    derivative2nd = mathfunc.derivative1d(y_smoothed, x=None, order=2, method=deriv_method)

    if derivative1st.size < 2: # Need at least 2 points for np.diff
        print("Warning in detect_initial_extrema_candidates: First derivative too short for sign change detection.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Threshold for first derivative being "close to zero" - currently used for plotting, not direct candidate selection here
    # deriv1_thresh = first_deriv_zero_threshold_factor * np.max(np.abs(derivative1st)) if derivative1st.size > 0 else 1e-9
    
    candidate_indices = []
    sign_changes = np.diff(np.sign(derivative1st))
    crossings_idx = np.where(np.abs(sign_changes) == 2)[0] # Indices *before* the sign change

    for idx in crossings_idx:
        # Refine to point with derivative1st closer to zero
        if idx + 1 < derivative1st.size: # Ensure idx+1 is a valid index
            if np.abs(derivative1st[idx]) < np.abs(derivative1st[idx + 1]):
                candidate_indices.append(idx)
            else:
                candidate_indices.append(idx + 1)
        else: # If idx is the second to last point, only idx can be the extremum from this crossing
            candidate_indices.append(idx)
            
    candidate_indices = sorted(list(set(candidate_indices))) # Unique sorted indices
    candidate_indices = np.array(candidate_indices, dtype=int)

    if candidate_indices.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Filter by 2nd derivative sign & ensure indices are valid
    valid_candidate_indices = candidate_indices[
        (candidate_indices < len(derivative2nd)) &
        (candidate_indices < len(x_coords_for_smoothed)) & # Also check against x_coords length
        (candidate_indices < len(y_smoothed)) # And y_smoothed length
    ]
    
    if valid_candidate_indices.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    max_ind = valid_candidate_indices[derivative2nd[valid_candidate_indices] < 0]
    min_ind = valid_candidate_indices[derivative2nd[valid_candidate_indices] > 0]

    peaks_x = x_coords_for_smoothed[max_ind] if max_ind.size > 0 else np.array([])
    peaks_y = y_smoothed[max_ind] if max_ind.size > 0 else np.array([])
    valleys_x = x_coords_for_smoothed[min_ind] if min_ind.size > 0 else np.array([])
    valleys_y = y_smoothed[min_ind] if min_ind.size > 0 else np.array([])
    
    return peaks_x, peaks_y, valleys_x, valleys_y

# --- SECTION 3: REFINEMENT TO ORIGINAL DATA (Parabolic/Weighted Mean Y) ---
def refine_extrema_to_original_pixel_grid( # Renamed for clarity
    extrema_x_from_smoothed, # X-positions from smoothed data (physical scale)
    original_x_axis,         # Original physical x-coordinates of the spectrum
    original_y_axis,         # Original y-flux values of the spectrum
    y_refinement_method='weighted_mean', # 'direct', 'weighted_mean', 'parabolic_vertex', 'parabolic_at_barycenter'
    y_refinement_weights=None  # For 'weighted_mean', e.g., [0.25, 0.5, 0.25]
):
    """
    Refines extrema by:
    1. X-coordinate: Finding the *integer pixel index* in original_x_axis closest
       to each x_from_smoothed.
    2. Y-coordinate: Calculating a refined y-value from original_y_axis centered
       at this integer pixel index, using the specified y_refinement_method.

    Args:
        extrema_x_from_smoothed (np.array): X-coords of extrema from smoothed data (physical scale).
        original_x_axis (np.array): The original physical x-coordinates of the spectrum (must be sorted).
        original_y_axis (np.array): The original y-flux values of the spectrum.
        y_refinement_method (str): Method for y-value refinement.
        y_refinement_weights (list/np.array, optional): Weights for 'weighted_mean'.

    Returns:
        tuple: (refined_x_indices, refined_y_values)
               refined_x_indices: NumPy array of integer pixel indices on the original grid.
               refined_y_values: NumPy array of corresponding refined y-values from original_y_axis.
    """
    original_x_axis = np.asarray(original_x_axis)
    original_y_axis = np.asarray(original_y_axis)
    extrema_x_input = np.asarray(extrema_x_from_smoothed) # Ensure it's an array

    if not original_x_axis.size or not original_y_axis.size or not extrema_x_input.size:
        # Return empty arrays of correct types if no input or no data to refine
        return np.array([], dtype=int), np.array([], dtype=float)

    # Prepare output arrays
    refined_x_indices_list = []
    refined_y_values_list = []

    if y_refinement_method == 'weighted_mean':
        if y_refinement_weights is None:
            y_refinement_weights = np.array([1/3., 1/3., 1/3.]) # Simple average
        else:
            y_refinement_weights = np.asarray(y_refinement_weights)
            if len(y_refinement_weights) != 3:
                print("Warning: y_refinement_weights must be length 3 for 'weighted_mean'. Using simple average.")
                y_refinement_weights = np.array([1/3., 1/3., 1/3.])
            elif not np.isclose(np.sum(y_refinement_weights), 1.0):
                print("Warning: y_refinement_weights do not sum to 1. Normalizing.")
                y_refinement_weights = y_refinement_weights / np.sum(y_refinement_weights)
    
    for x_val_smoothed in extrema_x_input:
        # Find index of closest point in original_x_axis
        # np.searchsorted finds insertion point.
        # If original_x_axis itself represents pixel indices (0, 1, 2,...),
        # then rounding x_val_smoothed (if it was supersampled index) and clamping
        # might be an alternative. But since original_x_axis can be physical units,
        # finding the closest point is more general.
        idx_map = np.searchsorted(original_x_axis, x_val_smoothed, side='left')
        
        center_idx_orig = 0 # Default for safety
        if idx_map == 0:
            center_idx_orig = 0
        elif idx_map == len(original_x_axis):
            center_idx_orig = len(original_x_axis) - 1
        else:
            # Compare distance to original_x_axis[idx_map-1] and original_x_axis[idx_map]
            if abs(original_x_axis[idx_map-1] - x_val_smoothed) < abs(original_x_axis[idx_map] - x_val_smoothed):
                center_idx_orig = idx_map - 1
            else:
                center_idx_orig = idx_map
        
        refined_x_indices_list.append(center_idx_orig) # Store the integer pixel index

        # --- Get refined Y value based on method, centered at center_idx_orig ---
        y_c_val_orig = original_y_axis[center_idx_orig] # Value at the chosen original pixel
        
        refined_y = y_c_val_orig # Default to direct value

        if y_refinement_method != 'direct':
            idx_l, idx_r = center_idx_orig - 1, center_idx_orig + 1
            if idx_l >= 0 and idx_r < len(original_y_axis): # Enough points for 3-point methods
                y_l_val_orig = original_y_axis[idx_l]
                y_r_val_orig = original_y_axis[idx_r]

                if y_refinement_method == 'weighted_mean':
                    refined_y = (y_refinement_weights[0] * y_l_val_orig +
                                 y_refinement_weights[1] * y_c_val_orig +
                                 y_refinement_weights[2] * y_r_val_orig)
                
                elif y_refinement_method == 'parabolic_vertex' or \
                     y_refinement_method == 'parabolic_at_barycenter':
                    
                    # Parabola: y_rel = a * x_rel^2 + b * x_rel + c, for x_rel = -1, 0, 1
                    # Coefficients relative to center_idx_orig as x_rel=0
                    c_parab = y_c_val_orig
                    b_parab = (y_r_val_orig - y_l_val_orig) / 2.0
                    a_parab = (y_l_val_orig + y_r_val_orig - 2.0 * y_c_val_orig) / 2.0

                    if abs(a_parab) < 1e-9: # Denominator 2a for vertex is near zero (linear/flat)
                        refined_y = y_c_val_orig
                    else:
                        if y_refinement_method == 'parabolic_vertex':
                            # y_vertex = c - b^2 / (4a) but using our a,b,c: a0 - a1^2/(4*a2)
                            y_vertex = c_parab - (b_parab**2) / (4 * a_parab)
                            refined_y = np.clip(y_vertex, 
                                                min(y_l_val_orig, y_c_val_orig, y_r_val_orig),
                                                max(y_l_val_orig, y_c_val_orig, y_r_val_orig))
                        
                        elif y_refinement_method == 'parabolic_at_barycenter':
                            flux_offset = 0.0
                            min_local_flux = min(y_l_val_orig, y_c_val_orig, y_r_val_orig)
                            if min_local_flux <= 0:
                                flux_offset = abs(min_local_flux) + 1e-6 # Ensure positive weights
                            
                            yl_w = y_l_val_orig + flux_offset
                            yc_w = y_c_val_orig + flux_offset
                            yr_w = y_r_val_orig + flux_offset
                            sum_fluxes = yl_w + yc_w + yr_w
                            
                            if abs(sum_fluxes) < 1e-9: x_bary_shift = 0.0
                            else: x_bary_shift = (yl_w * (-1) + yr_w * (1)) / sum_fluxes # Relative to center pixel
                            
                            y_at_bary = c_parab + b_parab * x_bary_shift + a_parab * (x_bary_shift**2)
                            refined_y = np.clip(y_at_bary,
                                                min(y_l_val_orig, y_c_val_orig, y_r_val_orig),
                                                max(y_l_val_orig, y_c_val_orig, y_r_val_orig))
            # else: refined_y remains y_c_val_orig (not enough points for 3-point window)
        
        refined_y_values_list.append(refined_y)

    return np.array(refined_x_indices_list, dtype=int), np.array(refined_y_values_list, dtype=float)


def select_bracketed_peaks(
    peaks_x,      # Final lists of peaks
    valleys_x,       # Final lists of valleys
    mindist = 5,
    asymmetry_factor=0.5 # e.g., |(xp-xvl) - (xvr-xp)| / ((xp-xvl) + (xvr-xp)) < 0.5
):
    """
    Selects peaks that are bracketed by two valleys with reasonably symmetric spacing.

    Args:
        peaks_x (np.array): X-coordinates of final peaks.
        peaks_y (np.array): Y-coordinates of final peaks.
        valleys_x (np.array): X-coordinates of final valleys.
        valleys_y (np.array): Y-coordinates of final valleys.
        max_dx_asymmetry_factor (float): Maximum allowed relative asymmetry in spacing
                                         between a peak and its left/right valleys.
                                         Factor = |dx_left - dx_right| / (dx_left + dx_right).
                                         A value of 0.5 means one side can be up to 3x the other.
                                         A value of 0.2 means one side can be up to 1.5x the other.

    Returns:
        np.ndarray: A structured array with columns 'pixl' (left valley x),
                    'pixc' (peak x), 'pixr' (right valley x), 'yc' (peak y),
                    'yl' (left valley y), 'yr' (right valley y) for the selected peaks.
                    Returns an empty structured array if no peaks meet the criteria.
    """
    if not peaks_x.size or valleys_x.size < 2: # Need at least one peak and two valleys
        # Define the dtype for the structured array even if empty
        dtype_out = [('pixl', float), ('pixc', float), ('pixr', float),
                     ('yc', float), ('yl', float), ('yr', float)]
        return np.array([], dtype=dtype_out)

    # Ensure valleys are sorted by x for efficient searching
    sort_v_idx = np.argsort(valleys_x)
    sorted_valleys_x = valleys_x[sort_v_idx]
    # sorted_valleys_y = valleys_y[sort_v_idx]

    selected_peak_data = []

    for i in range(len(peaks_x)):
        xp = peaks_x[i]

        # Find the closest valley to the left
        left_valley_indices = np.where(sorted_valleys_x < xp)[0]
        if not left_valley_indices.size:
            continue # No valley to the left
        idx_vl = left_valley_indices[-1] # Index in sorted_valleys_x
        xvl = sorted_valleys_x[idx_vl]

        # Find the closest valley to the right
        right_valley_indices = np.where(sorted_valleys_x > xp)[0]
        if not right_valley_indices.size:
            continue # No valley to the right
        idx_vr = right_valley_indices[0] # Index in sorted_valleys_x
        xvr = sorted_valleys_x[idx_vr]

        # Check if these valleys are indeed the *immediate* bracketing ones
        # (i.e., no other peak between xp and xvl, or xp and xvr)
        # This is implicitly handled if the input peaks/valleys are already well-alternated.
        # For robustness, one could check:
        if np.any((peaks_x > xvl) & (peaks_x < xp)) or \
            np.any((peaks_x < xvr) & (peaks_x > xp)):
            continue # Not truly bracketing

        dx_left = xp - xvl
        dx_right = xvr - xp

        if dx_left <= 0 or dx_right <= 0: # Should not happen if valleys truly bracket
            continue
        if np.abs(dx_left) <= mindist or np.abs(dx_right) <= mindist:
            continue

        # Asymmetry check
        asymmetry = abs(dx_left - dx_right) / (dx_left + dx_right)

        if asymmetry <= asymmetry_factor:
            selected_peak_data.append((xvl, xp, xvr))

    dtype_out = [('pixl', int), ('pixc', int), ('pixr', int),]
    return np.array(selected_peak_data, dtype=dtype_out)

# --- SECTION 4: MAIN ORCHESTRATING FUNCTION (No Segmentation - v3) ---

def process_spectrum2d(flux2d,
                       sOrder=None,
                       x_axis_orig_input=None,
                       super_sample_factor=3,
                       snr_threshold = 3, 
                       window_len_method_options: dict = None,
                       deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
                       # Parameters for Sav-Gol filter
                       savgol_params: dict = None, 
                       # Consolidated parameters for the hybrid filter (rules + separate DBSCAN)
                       hybrid_filter_params: dict = None,
                       triplet_removal_params: dict = None,
                       y_refinement_on_candidates_params: dict = None,
                       plot_main_details=False,
                       plot_deriv_stage_details=True, # Plot for initial smoothing/derivatives
                       plot_filter_stage_details=True,  # Plot for rule-based filter diagnostics
                       verbose=False
                   ):

    """
    Processes a 2D spectrum array by applying process_spectrum to each row (spectrum).

    Iterates over the rows (axis=0) of the flux2d array, runs process_spectrum
    on each row, and aggregates the results.

    Args:
        flux2d (np.ndarray): The 2D spectrum array, shape (n_rows, n_pixels).
        x_axis_orig_input (np.ndarray, optional): The common x-axis (e.g., pixels, wavelengths)
            corresponding to the columns of flux2d. If None, pixel indices (0 to n_pixels-1) are used.
        super_sample_factor (int): Super-sampling factor for internal processing.
        snr_threshold (float): Signal-to-noise threshold used within filtering stages.
        window_len_method_options (dict, optional): Options for determining the smoothing window length.
        deriv_method (str): Method for calculating derivatives ('coeff' or 'sg').
        first_deriv_zero_threshold_factor (float): Threshold factor for derivative zero-crossing detection.
        savgol_params (dict, optional): Parameters for Savitzky-Golay filtering used in envelope fitting etc.
        hybrid_filter_params (dict, optional): Parameters for the hybrid peak/valley filtering stage.
        triplet_removal_params (dict, optional): Parameters for the false triplet removal stage.
        y_refinement_on_candidates_params (dict, optional): Parameters for refining y-values of candidates.
        plot_main_details (bool): If True, generate summary plots for each row processed.
        plot_deriv_stage_details (bool): If True (and plot_main_details=True), plot derivative stage details.
        plot_filter_stage_details (bool): If True (and plot_main_details=True), plot filtering stage details.
        verbose (bool): If True, print verbose processing information for each row.

    Returns:
        tuple: A tuple containing:
            - line_positions (np.ndarray): A structured NumPy array with fields 'x_pos' (float)
              and 'row_index' (int), containing the x-position (along the common x-axis)
              and originating row index for every detected line across all rows.
            - peak_envelope (np.ndarray): A 2D array with the same shape as flux2d, where
              each row contains the fitted peak envelope for the corresponding input row.
            - valley_envelope (np.ndarray): A 2D array with the same shape as flux2d, where
              each row contains the fitted valley envelope for the corresponding input row.
    """
    if not isinstance(flux2d, np.ndarray) or flux2d.ndim != 2:
        raise ValueError("flux2d must be a 2D NumPy array.")

    n_rows, n_pixels = flux2d.shape # Axis 0 is rows, Axis 1 is pixels/columns

    if n_rows == 0 or n_pixels == 0:
        print("Warning: flux2d is empty. Returning empty results.")
        # Ensure dtype matches the expected output even when empty
        line_dtype = [('x_pos', 'f8'), ('row_index', 'i4')]
        return np.array([], dtype=line_dtype), np.zeros_like(flux2d), np.zeros_like(flux2d)

    # Prepare the x-axis once - it corresponds to the columns
    if x_axis_orig_input is None:
        x_axis_common = np.arange(n_pixels)
    else:
        x_axis_common = np.asarray(x_axis_orig_input)
        if len(x_axis_common) != n_pixels:
            raise ValueError(f"Provided x_axis_orig_input length ({len(x_axis_common)}) "
                             f"does not match flux2d pixel dimension (columns) ({n_pixels})")

    # Initialize containers for results
    all_lines_records = [] # List to collect (x_pos, row_idx) tuples
    # Envelopes will have the same shape as input
    peak_envelope_all = np.zeros_like(flux2d)
    valley_envelope_all = np.zeros_like(flux2d)

    # Define the structured array dtype expected from process_spectrum
    expected_line_dtype_names = ['pixl', 'pixc', 'pixr']
    # Define the final structured array dtype including the row index
    final_line_dtype = [('order', 'i4'), ('pixl', '<i4'), 
                        ('pixc', '<i4'), ('pixr', '<i4')]

    if verbose:
        print(f"Processing {n_rows} rows (spectra) with {n_pixels} pixels each.")

    # Iterate over each row (axis=0)
    for row_idx in range(n_rows):
        progress_bar.update(row_idx/(n_rows-1),f'Peakdetect {row_idx}/{n_rows}')
        if row_idx<sOrder:
            continue
        if verbose:
            print(f"\n--- Processing Row {row_idx}/{n_rows-1} ---")

        y_row = flux2d[row_idx, :] # Extract the current row

        # Handle potential completely flat or NaN rows if necessary
        if np.all(np.isnan(y_row)) or (y_row.size > 0 and np.all(y_row == y_row[0])):
             if verbose:
                 print(f"  Skipping row {row_idx} due to NaNs or being flat.")
             # Fill envelopes with NaN or a constant if appropriate for this row
             peak_envelope_all[row_idx, :] = np.nan
             valley_envelope_all[row_idx, :] = np.nan
             continue # Skip to the next row

        try:
            # Run process_spectrum for the current row
            lines_row, peak_env_row, valley_env_row = process_spectrum(
                y_axis_orig_input=y_row,          # Pass the row data
                x_axis_orig_input=x_axis_common,  # Pass the common x-axis
                super_sample_factor=super_sample_factor,
                snr_threshold=snr_threshold,
                window_len_method_options=window_len_method_options,
                deriv_method=deriv_method,
                first_deriv_zero_threshold_factor=first_deriv_zero_threshold_factor,
                savgol_params=savgol_params,
                hybrid_filter_params=hybrid_filter_params,
                triplet_removal_params=triplet_removal_params,
                y_refinement_on_candidates_params=y_refinement_on_candidates_params,
                plot_main_details=plot_main_details,
                plot_deriv_stage_details=plot_deriv_stage_details,
                plot_filter_stage_details=plot_filter_stage_details,
                verbose=verbose # Pass verbosity down
            )

            # Store the results for this row
            # --- Line handling ---
            if lines_row is not None and len(lines_row) > 0:
                # Check if lines_row has the expected structure
                if not (hasattr(lines_row, 'dtype') and all(name in lines_row.dtype.names for name in expected_line_dtype_names)):
                     if verbose:
                         print(f"  Warning: lines output for row {row_idx} is not a structured array with expected fields {expected_line_dtype_names}. Skipping lines for this row.")
                         # Still process envelopes below
                else:
                    # Iterate through each record in the structured array from process_spectrum
                    for line_record in lines_row:
                        # Create a tuple matching the final_line_dtype order
                        record = (row_idx, line_record['pixl'], line_record['pixc'], line_record['pixr'])
                        all_lines_records.append(record)
                    if verbose:
                        print(f"  Found {len(lines_row)} lines in row {row_idx}.")


            # Store the envelopes in the corresponding row
            if peak_env_row is not None and len(peak_env_row) == n_pixels:
                peak_envelope_all[row_idx, :] = peak_env_row
            elif verbose:
                 print(f"  Warning: Peak envelope for row {row_idx} has unexpected length ({len(peak_env_row) if peak_env_row is not None else 'None'}) vs expected ({n_pixels}). Filling with NaNs.")
                 peak_envelope_all[row_idx, :] = np.nan


            if valley_env_row is not None and len(valley_env_row) == n_pixels:
                valley_envelope_all[row_idx, :] = valley_env_row
            elif verbose:
                 print(f"  Warning: Valley envelope for row {row_idx} has unexpected length ({len(valley_env_row) if valley_env_row is not None else 'None'}) vs expected ({n_pixels}). Filling with NaNs.")
                 valley_envelope_all[row_idx, :] = np.nan


        except Exception as e:
            print(f"Error processing row {row_idx}: {e}")
            # Optionally decide how to handle errors: skip, fill with NaN, re-raise
            # Filling envelopes with NaN for robustness
            peak_envelope_all[row_idx, :] = np.nan
            valley_envelope_all[row_idx, :] = np.nan
            # Continue to the next row
            continue

    # Convert the collected line data into a structured NumPy array
    line_positions = np.array(all_lines_records, dtype=final_line_dtype)
    if verbose:
        print(f"\n--- Finished processing all rows ---")
        print(f"Total lines found across all rows: {len(line_positions)}")

    return line_positions, peak_envelope_all, valley_envelope_all    

def process_spectrum(
    y_axis_orig_input, x_axis_orig_input=None,
    super_sample_factor=3,
    snr_threshold = 3, 
    window_len_method_options: dict = None,
    deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
    # Parameters for Sav-Gol filter
    savgol_params: dict = None, 
    # Consolidated parameters for the hybrid filter (rules + separate DBSCAN)
    hybrid_filter_params: dict = None,
    triplet_removal_params: dict = None,
    y_refinement_on_candidates_params: dict = None,
    plot_main_details=False,
    plot_deriv_stage_details=True, # Plot for initial smoothing/derivatives
    plot_filter_stage_details=True,  # Plot for rule-based filter diagnostics
    verbose=False
):
    x_axis_orig, y_axis_orig = _datacheck(x_axis_orig_input, y_axis_orig_input)
    if verbose: print(f"Processing spectrum (len {len(y_axis_orig)}) with Hybrid Filter v7 (Separate DBSCAN)")

    # --- Default parameters ---
    if window_len_method_options is None:
        window_len_method_options = {'method':'auto_robust',
                                     'user_val':None,
                                     'gw_params':{'target_min_period_pixels':10,
                                                  'target_max_period_pixels':35,
                                                  'default_window_period':15,
                                                  'overall_min_period_pixels':5,
                                                  'verbose':verbose}}
        
    def_savgol_window = 51
    def_savgol_polyorder = 3
    if savgol_params is None:
        savgol_params = dict(
            window = def_savgol_window,
            polyorder = def_savgol_polyorder,
            )
    for key, val in zip(['window', 'polyorder'],
                        [def_savgol_window, def_savgol_polyorder]):
        savgol_params.setdefault(key, val)
    
    
    global_noise_val = robust_noise_std(y_axis_orig - np.median(y_axis_orig))
    if hybrid_filter_params is None:
        hybrid_filter_params = {
            'spacing_poly_degree': 1, 
            'spacing_uncertainty_const': 2.0,
            'spacing_max_dev_factor': 3.0, 
            'envelope_savgol_window': savgol_params['window'],
            'envelope_savgol_polyorder': savgol_params['polyorder'], #'max_allowed_Venv_Penv_ratio': 0.35,
            'ratio_trend_savgol_window':101, 'ratio_trend_savgol_polyorder': 2,
            'peak_snr_min_thresh_poisson': snr_threshold,
            'noise_estimation_window_pixels': 15,
            'global_fallback_noise_std': global_noise_val,
            'plot_filter_diagnostics': plot_main_details and plot_filter_stage_details, # For rules part
            'verbose': verbose
        }
    # Ensure critical params have defaults if dict is partially provided
    hybrid_filter_params.setdefault('global_fallback_noise_std', global_noise_val)
    hybrid_filter_params.setdefault('plot_filter_diagnostics', plot_main_details and plot_filter_stage_details)
    hybrid_filter_params.setdefault('verbose', verbose)


    if y_refinement_on_candidates_params is None:
        y_refinement_on_candidates_params = {'y_refinement_method': 'weighted_mean'}

    # --- Stage 1: Initial Peak/Valley X (from smooth) and Y (from smooth) Detection ---
    if verbose: print("--- Stage 1: Initial Peak/Valley X and Smoothed Y ---")
    if super_sample_factor > 1 and len(y_axis_orig) > 0 :
        y_rebinned = _rebin(y_axis_orig, newshape=(int(super_sample_factor * len(y_axis_orig)),))
        x_rebinned = np.linspace(np.min(x_axis_orig), np.max(x_axis_orig), len(y_rebinned))
    else: y_rebinned = np.copy(y_axis_orig); x_rebinned = np.copy(x_axis_orig)
    if len(y_rebinned) == 0: print("Rebinned data empty."); return ([[], []], [[], []])

    plot_gw = plot_main_details and plot_deriv_stage_details # Control get_window plot
    win_method = window_len_method_options['method']; gw_p = window_len_method_options.get('gw_params', {})
    gw_p['verbose'] = verbose # Pass verbose to get_window
    if win_method == 'auto_robust': window_len_on_orig_scale = get_window_robust_targeted(y_axis_orig, plot=plot_gw, **gw_p)
    # ... (other window selection methods as before) ...
    else: window_len_on_orig_scale = gw_p.get('default_window_period',15)
    
    actual_smoothing_window_len = mathfunc.round_down_to_odd(int(window_len_on_orig_scale * super_sample_factor))
    if len(y_rebinned) > 0:
        max_poss_win = mathfunc.round_down_to_odd(len(y_rebinned)//2-1 if len(y_rebinned)//2-1>=1 else 1)
        if max_poss_win < 3 and len(y_rebinned) >=3 : max_poss_win = 3
        actual_smoothing_window_len = min(actual_smoothing_window_len, max_poss_win)
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    if verbose: print(f"  Smooth window: {actual_smoothing_window_len} (on rebinned)")
    
    y_smoothed = _smooth(y_rebinned, window_len=actual_smoothing_window_len, window='nuttall', mode="same")
    x_coords_for_smoothed = x_rebinned
    if len(y_smoothed) < 3: print("Smoothed data too short."); return ([[], []], [[], []])

    initial_peaks_x, initial_peaks_y_smoothed, \
    initial_valleys_x, initial_valleys_y_smoothed = \
        detect_initial_extrema_candidates(y_smoothed, x_coords_for_smoothed, deriv_method)
    if verbose: print(f"  Initial derivative cands: {len(initial_peaks_x)}P, {len(initial_valleys_x)}V (Y from smooth).")
    if not initial_peaks_x.size and not initial_valleys_x.size: print("No initial cands."); return ([[], []], [[], []])

    # --- Stage 1.5: Refine Y-values of initial candidates using ORIGINAL data ---
    if verbose: print("\n--- Stage 1.5: Refining Y-values of initial candidates on original data ---")
    initial_peaks_x, initial_peaks_y_refined_orig = refine_extrema_to_original_pixel_grid(
        initial_peaks_x, x_axis_orig, y_axis_orig, **y_refinement_on_candidates_params)
    initial_valleys_x, initial_valleys_y_refined_orig = refine_extrema_to_original_pixel_grid(
        initial_valleys_x, x_axis_orig, y_axis_orig, **y_refinement_on_candidates_params)
    if verbose: print(f"    Y-values refined. {len(initial_peaks_y_refined_orig)}P, {len(initial_valleys_y_refined_orig)}V.")
    # Note: initial_peaks_x_ref is same as initial_peaks_x (x-pos not changed by y-refinement func)


    # --- NEW Stage 1.75: Remove False Triplets ---
    # First, combine peaks and valleys from Stage 1.5 using _ensure_alternation_and_tag
    # to get a single list suitable for remove_false_triplets
    # This also ensures initial strict alternation before triplet removal.

    # Parameters for triplet removal (should come from a new param dict or defaults)
    if triplet_removal_params is None:
        triplet_removal_params = {
            'max_triplet_x_span_pixels': window_len_on_orig_scale * 0.3, # e.g., 30% of typical period
            'min_prom_depth_factor_center': 2, # Center must be at least 10% of outer span/height
            'y_consistency_factor':0.2,
        }
    
    # max_triplet_x_span = triplet_removal_params.get('max_triplet_x_span_pixels', window_len_on_orig_scale * 0.3)
    # min_prom_depth_factor_triplet = triplet_removal_params.get('min_prominence_depth_factor_for_center', 0.1)
    if verbose: print("\n--- Stage 1.75: Removing False Triplets ---")
    
    _, _, _, _, \
    combined_x_for_triplet_filt, \
    combined_y_for_triplet_filt, \
    combined_types_for_triplet_filt = _ensure_alternation_and_tag(
        initial_peaks_x, initial_peaks_y_refined_orig, # Using X from smooth, Y from original
        initial_valleys_x, initial_valleys_y_refined_orig
    )

    if combined_x_for_triplet_filt.size > 0:
        # remove_false_triplets returns separated lists AND combined lists
        peaks_after_triplet_filt_x, peaks_after_triplet_filt_y, \
        valleys_after_triplet_filt_x, valleys_after_triplet_filt_y, \
        _, _, _ = remove_false_triplets_v2( # We only need the separated lists from its return
            combined_x_for_triplet_filt,
            combined_y_for_triplet_filt,
            combined_types_for_triplet_filt,
            **triplet_removal_params,
            verbose=verbose
        )
        if verbose: print(f"    After triplet removal: {len(peaks_after_triplet_filt_x)}P, {len(valleys_after_triplet_filt_x)}V.")
    else: # No candidates to feed to triplet filter
        peaks_after_triplet_filt_x, peaks_after_triplet_filt_y = initial_peaks_x, initial_peaks_y_refined_orig
        valleys_after_triplet_filt_x, valleys_after_triplet_y = initial_valleys_x, initial_valleys_y_refined_orig
        
    # --- Stage 2: Apply Hybrid Rule-Based Pre-Filter + Separate DBSCAN ---
    if verbose: print(f"\n--- Stage 2: Filtering peaks and valleys ---")
    final_peaks_x_arr, final_peaks_y_arr, \
    final_valleys_x_arr, final_valleys_y_arr, \
    plot_data_output_hybrid = filter_lfc_extrema_v8_final_rules_with_plots( 
        peaks_after_triplet_filt_x, peaks_after_triplet_filt_y,
        valleys_after_triplet_filt_x, valleys_after_triplet_filt_y,
        x_axis_orig, y_axis_orig,
        **hybrid_filter_params # Pass all hybrid filtering parameters
    )
    # Y-values from here (final_peaks_y_arr, etc.) are already refined to original.
    # X-values are from the smoothed data.
    
    # ---- Stage 3: Select bracketed lines
    if verbose: print(f"\n--- Stage 2: Detecting lines ---")
    lines = select_bracketed_peaks(final_peaks_x_arr, 
                                   final_valleys_x_arr,
                                   mindist = 0.3 * window_len_on_orig_scale
                                   )
    
    peak_envelope   = fit_envelope_savgol(final_peaks_x_arr, 
                                          final_peaks_y_arr, 
                                          x_axis_orig, 
                                          savgol_params['window'], 
                                          savgol_params['polyorder'], 
                                          verbose)
    valley_envelope = fit_envelope_savgol(final_valleys_x_arr, 
                                          final_valleys_y_arr, 
                                          x_axis_orig, 
                                          savgol_params['window'], 
                                          savgol_params['polyorder'], 
                                          verbose)
    
    if verbose: 
        print(f"\n--- Overall Processing Complete ---")
        print(f"Found total {len(final_peaks_x_arr)} peaks & {len(final_valleys_x_arr)} valleys, with {len(lines)} lines.")

    # --- Plotting ---
    if plot_main_details:
        # plot_data_output_hybrid = (peaks_dbscan_plot_info, valleys_dbscan_plot_info)
        # each _plot_info = (unscaled_feat, scaled_feat, labels, cand_x, cand_y, types_array_for_those_cands)
        peaks_plot_info = plot_data_output_hybrid[0]
        valleys_plot_info = plot_data_output_hybrid[1]

        if plot_deriv_stage_details:
            deriv1_plot = mathfunc.derivative1d(y_smoothed, order=1, method=deriv_method)
            deriv2_plot = mathfunc.derivative1d(y_smoothed, order=2, method=deriv_method)
            deriv1_abs_thresh_plot = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1_plot)) if deriv1_plot.size > 0 else 0
            plot_smoothing_and_derivatives(
                x_axis_orig, y_axis_orig, x_coords_for_smoothed, y_smoothed,
                deriv1_plot, deriv2_plot,
                initial_peaks_x, initial_peaks_y_smoothed,
                initial_valleys_x, initial_valleys_y_smoothed,
                actual_smoothing_window_len, deriv1_abs_thresh_plot
            )
        
        # The filter_lfc_extrema_v7_hybrid_sep_dbscan handles its own rule-diagnostic plots
        # if hybrid_filter_params['plot_filter_diagnostics'] is True.

        # Plot final summary (candidates to DBSCAN vs. final DBSCAN output)
        if plot_filter_stage_details: # Reuse this flag for the final summary after DBSCAN
            print("\nPlotting final summary of filtering:")
            
            # Candidates that went into DBSCAN for peaks (i.e., rule-survivors for peaks)
            cand_p_x_dbscan = peaks_plot_info[3]
            cand_p_y_dbscan = peaks_plot_info[4] # These Ys are already refined
            # Candidates that went into DBSCAN for valleys
            cand_v_x_dbscan = valleys_plot_info[3]
            cand_v_y_dbscan = valleys_plot_info[4] # These Ys are already refined
            
            plot_filtering_2d_results( # Your generic 2D results plotter
                x_axis_orig, y_axis_orig,
                x_coords_for_smoothed, y_smoothed, # Smoothed context
                cand_p_x_dbscan, cand_p_y_dbscan, 
                cand_v_x_dbscan, cand_v_y_dbscan,
                final_peaks_x_arr, final_peaks_y_arr, 
                final_valleys_x_arr, final_valleys_y_arr,
                lines = lines,
                peak_envelope = peak_envelope,
                valley_envelope = valley_envelope,
                title_suffix="Full Spectrum (Final after Hybrid Filter)"
            )


    # return [final_peaks_x_arr.tolist(), final_peaks_y_arr.tolist()], \
    #        [final_valleys_x_arr.tolist(), final_valleys_y_arr.tolist()]
    return lines, peak_envelope, valley_envelope


# --- Plotting Utilities (Updated for consistency) ---

def plot_smoothing_and_derivatives(
    x_axis_orig, y_axis_orig, x_coords_for_smoothed, y_smoothed,
    derivative1st, derivative2nd, # Pass these in if calculated in orchestrator
    initial_derivative_peaks_x, initial_derivative_peaks_y,
    initial_derivative_valleys_x, initial_derivative_valleys_y,
    actual_smoothing_window_len, deriv1_abs_thresh # Pass absolute threshold
):
    fig_sd, (ax_sd1, ax_sd2, ax_sd3) = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
    fig_sd.suptitle("Spectrum Smoothing and Derivatives", fontsize=14)
    ax_sd1.plot(x_axis_orig, y_axis_orig, label='Original Data', color='lightgray', alpha=0.7)
    ax_sd1.plot(x_coords_for_smoothed, y_smoothed, label=f'Smoothed (Win={actual_smoothing_window_len})', color='blue')
    ax_sd1.scatter(initial_derivative_peaks_x, initial_derivative_peaks_y, marker='o', edgecolor='magenta', facecolor='none', s=40, label='Initial Peak Deriv. Cand.', zorder=3, alpha=0.7)
    ax_sd1.scatter(initial_derivative_valleys_x, initial_derivative_valleys_y, marker='o', edgecolor='cyan', facecolor='none', s=40, label='Initial Valley Deriv. Cand.', zorder=3, alpha=0.7)
    ax_sd1.legend(loc='upper right'); ax_sd1.set_ylabel("Flux")

    ax_sd2.plot(x_coords_for_smoothed, derivative1st, label='1st Derivative')
    ax_sd2.axhline(0, color='gray', linestyle='--'); ax_sd2.axhline(deriv1_abs_thresh, color='orange', linestyle=':', label=f'D1 Thresh'); ax_sd2.axhline(-deriv1_abs_thresh, color='orange', linestyle=':')
    ax_sd2.legend(loc='upper right'); ax_sd2.set_ylabel("1st Deriv")

    ax_sd3.plot(x_coords_for_smoothed, derivative2nd, label='2nd Derivative')
    ax_sd3.axhline(0, color='gray', linestyle='--')
    ax_sd3.legend(loc='upper right'); ax_sd3.set_ylabel("2nd Deriv")
    ax_sd3.set_xlabel("X-coordinate")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
    
def plot_filtering_2d_results( # Keep the content same, just rename for context
    x_axis_orig, y_axis_orig,
    x_coords_smoothed, y_values_smoothed,
    cand_peaks_x, cand_peaks_y_refined_orig,
    cand_valleys_x, cand_valleys_y_refined_orig,
    final_peaks_x, final_peaks_y_refined,
    final_valleys_x, final_valleys_y_refined,
    lines = None,
    peak_envelope = None,
    valley_envelope = None,
    title_suffix=""
):
    fig = plt.figure(figsize=(15, 7))
    ax  = fig.add_subplot()
    main_title = "Filtering Results" # Generic title
    if title_suffix: 
        main_title += f" - {title_suffix}"
    plt.title(main_title, fontsize=14)
    ax.plot(x_axis_orig,y_axis_orig,label='Original Data',color='lightgrey',alpha=0.7,zorder=1)
    if x_coords_smoothed is not None and y_values_smoothed is not None and x_coords_smoothed.size>0:
        ax.plot(x_coords_smoothed,y_values_smoothed,label='Smoothed Data (Context)',color='gray',alpha=0.4,linestyle=':',zorder=1)
    cpk_x,cpk_y=np.asarray(cand_peaks_x),np.asarray(cand_peaks_y_refined_orig)
    cvl_x,cvl_y=np.asarray(cand_valleys_x),np.asarray(cand_valleys_y_refined_orig)
    fpk_x,fpk_y=np.asarray(final_peaks_x),np.asarray(final_peaks_y_refined)
    fvl_x,fvl_y=np.asarray(final_valleys_x),np.asarray(final_valleys_y_refined)
    if cpk_x.size>0:
        plt.scatter(cpk_x,cpk_y,color='pink',marker='o',
                    s=50,label='Peak Cand. (to Filter)',zorder=2,alpha=0.6)
    if cvl_x.size>0:
        plt.scatter(cvl_x,cvl_y,color='lightblue',marker='o',
                    s=50,label='Valley Cand. (to Filter)',zorder=2,alpha=0.6)
    sfpk_x=set(np.round(fpk_x,decimals=5))
    sfvl_x=set(np.round(fvl_x,decimals=5))
    false_pk_x,false_pk_y=[],[]
    if cpk_x.size>0:
        for x,y in zip(cpk_x,cpk_y):
            if round(x,5) not in sfpk_x:
                false_pk_x.append(x);
                false_pk_y.append(y)
    false_vl_x,false_vl_y=[],[]
    if cvl_x.size>0:
        for x,y in zip(cvl_x,cvl_y):
            if round(x,5) not in sfvl_x:false_vl_x.append(x);false_vl_y.append(y)
    ax.scatter(false_pk_x,false_pk_y,color='red',marker='x',s=100,label='Discarded Peaks (Filter)',zorder=3)
    ax.scatter(false_vl_x,false_vl_y,color='magenta',marker='x',s=100,label='Discarded Valleys (Filter)',zorder=3)
    if fpk_x.size>0:
        ax.scatter(fpk_x,fpk_y,edgecolor='blue',facecolor='none',marker='o',
                    s=120,label='Final Peaks',zorder=4,linewidth=1.5)
    if fvl_x.size>0:
        ax.scatter(fvl_x,fvl_y,edgecolor='green',facecolor='none',marker='o',
                    s=120,label='Final Valleys',zorder=4,linewidth=1.5)
        
    ymin_ = np.percentile(y_axis_orig, 2.5)
    ymax_ = np.max(y_axis_orig)
    yspan_ = ymax_ - ymin_
    
    ymin = max(ymin_ - 0.15*yspan_, -1000)
    ymax = ymax_ + 0.15*yspan_
        
    if lines is not None:
        yspan = ymax - ymin
        y0 = ymin + 0.05 * yspan
        f = 0.03
        for line in lines:
            x_cen = line['pixc']
            ax.plot([x_cen, x_cen], [y0, y0 + f*yspan], c='k', alpha=0.7)
            
    if peak_envelope is not None:
        ax.plot(x_axis_orig, peak_envelope, 'b-')
    if valley_envelope is not None:
        ax.plot(x_axis_orig, valley_envelope, 'g-')
    ax.set(xlabel='X (Orig Scale)',
           ylabel='Flux (Orig Scale)',
           ylim = (ymin, ymax))
    ax.legend()
    ax.grid(True,ls=':',alpha=0.7)
    plt.show()