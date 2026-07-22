#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io.py


Created on Thu Apr 24 14:17:57 2025

@author: dmilakov
"""

import warnings
from pathlib import Path
import numpy as np
from fitsio import FITS

def load_echelle_data(fits_path, bias_path=None, hdu_index=1):
    """Loads FITS data, optionally bias subtracting."""
    fits_file = Path(fits_path)
    if not fits_file.is_file():
        raise FileNotFoundError(f"Data file not found: {fits_file}")

    print(f"Loading data from: {fits_file} (HDU {hdu_index})")
    with FITS(fits_file, 'r') as hdul:
        # Ensure data exists before reading
        if len(hdul) <= hdu_index:
             raise IndexError(f"HDU index {hdu_index} out of range for file {fits_file} (has {len(hdul)} HDUs).")
        data = hdul[hdu_index].read()

    if bias_path:
        bias_file = Path(bias_path)
        print(f"Attempting bias subtraction using: {bias_file}")
        if not bias_file.is_file():
            warnings.warn(f"Bias file not found: {bias_file}. Proceeding without bias subtraction.")
            bias = 0
        else:
             with FITS(bias_file, 'r') as hdul_b:
                 bias = None
                 # Try matching HDU index first
                 if len(hdul_b) > hdu_index:
                     bias = hdul_b[hdu_index].read()
                     print(f"  Found bias in HDU {hdu_index}.")
                 # Fallback if needed
                 elif len(hdul_b) > 1 and hdu_index != 1:
                     warnings.warn(f"Bias HDU index {hdu_index} not found. Trying index 1.")
                     bias = hdul_b[1].read()
                     print(f"  Found bias in HDU 1.")
                 elif len(hdul_b) > 0 and hdu_index != 0:
                     warnings.warn(f"Bias HDU index {hdu_index} or 1 not found. Trying index 0.")
                     bias = hdul_b[0].read()
                     print(f"  Found bias in HDU 0.")

                 if bias is None:
                      warnings.warn(f"Could not read bias frame from {bias_file}. Skipping subtraction.")
                      bias = 0
                 elif bias.shape != data.shape:
                     warnings.warn(f"Bias shape {bias.shape} does not match data shape {data.shape}. Skipping bias subtraction.")
                     bias = 0
                 else:
                      print("  Bias frame loaded and shapes match.")


        # Perform subtraction ensuring float type
        data = data.astype(float) - bias.astype(float)
        print("  Bias subtraction performed.")
    else:
        data = data.astype(float) # Ensure float anyway
        print("  No bias frame provided.")

    print(f"Data loading complete. Final shape: {data.shape}")
    return data