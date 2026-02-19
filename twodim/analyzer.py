#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:19:51 2025

@author: dmilakov
"""
import numpy as np
import fitsio
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import traceback
import warnings
import time
import os # For file locking check (optional but good practice)

# Use relative imports within the package
from .io import load_echelle_data
from .peaks import find_peaks
from .clustering import cluster_peaks_to_orders
from .fitting import twoD_Gaussian
from .plotting import (plot_gaussian_ellipse, plot_raw_data_stamps,
                       plot_normalized_residuals_stamps) # Keep per-segment plotters here

GAUSS_PARAMS_NAME = ['AMPLITUDE',
                     'CEN_X',
                     'CEN_Y',
                     'SIGMA_X',
                     'SIGMA_Y',
                     'THETA',
                     'CONST']

try:
    from .zernike_fitter import ZernikeFitter, ZERNPY_AVAILABLE, generate_zernike_indices
except ImportError:
    warnings.warn("zernike_fitter module not found. Zernike fitting will be skipped.")
    ZERNPY_AVAILABLE = False
    class ZernikeFitter: # Dummy
         def __init__(self, *args, **kwargs): self.n_coeffs=0; self.zernike_indices = []
         def fit(self, *args, **kwargs): self.success=False; self.message="Fitter unavailable"; return False
         def get_results(self, *args, **kwargs): return {'success': False}
         def plot_power_spectrum(self, *args, **kwargs): pass
         def plot_fit_comparison(self, *args, **kwargs): pass
         def plot_fit_residuals(self, *args, **kwargs): pass
    def generate_zernike_indices(n_max): return []

class EchelleAnalyzer:
    """
    Analyzes echelle spectrograph data to fit Zernike polynomials to stacked PSF profiles.
    Handles data loading, peak finding, order clustering, segmented analysis,
    and saving results to a FITS file using variable-length arrays.
    """
    def __init__(self, lfc_filename, bias_filename=None, output_dir='./echelle_analysis_output',
                 peak_min_distance=5, peak_threshold_abs=None, peak_threshold_rel=None, peak_hard_cut=None,
                 cluster_eps_x=0.002, cluster_eps_y=0.02, cluster_min_samples=5,
                 stamp_half_width=5, fit_threshold_snr=5.0,
                 num_segments=16, n_max_zern=6, r_max_zern=5.0,
                 output_suffix='_zernike_results.fits', clobber_output=True):

        self.lfc_path = Path(lfc_filename)
        self.bias_path = Path(bias_filename) if bias_filename else None
        self.output_dir = Path(output_dir)
        self.output_fits_path = None # Defined in load_data
        self.output_suffix = output_suffix
        # CLOBBER only applies if file *doesn't exist* at start or if explicitly requested
        # The update logic handles existing files.
        self.clobber_allowed = clobber_output
    
        self.params = { # Store parameters for saving in header
            'peak_min_distance': peak_min_distance, 'peak_threshold_abs': peak_threshold_abs,
            'peak_threshold_rel': peak_threshold_rel, 'peak_hard_cut': peak_hard_cut,
            'cluster_eps_x': cluster_eps_x, 'cluster_eps_y': cluster_eps_y,
            'cluster_min_samples': cluster_min_samples, 'stamp_half_width': stamp_half_width,
            'fit_threshold_snr': fit_threshold_snr, 'num_segments': num_segments,
            'n_max_zern': n_max_zern, 'r_max_zern': r_max_zern,
        }

        # Runtime attributes
        self.image_data = None; self.image_shape = None; self.detector = None
        self.hdu_index = None; self.all_peaks_xy = None; self.paired_orders_dict = None
        self.median_segment_positions = {}
        self.zernike_indices = None
        self.results_table_dtype = None # Numpy dtype
        self.fits_column_formats = None # Explicit FITS TFORM strings (needed for fitsio empty write)
        self._num_coeffs_for_fits = 0 # Store coeff number for fixed vector formats
        self._fits_file_created = False # Flag to track if FITS file structure exists
        
        self.peak_catalog_dtype = None # Initialize
        self.peak_catalog_fits_formats = None # Initialize
        self._define_results_dtype_and_formats() # Define ZERNIKE dtype
        self._define_peak_catalog_dtype()      # Define PEAK_CATALOG dtype
        self._define_gauss2d_dtype_and_formats() # Define GAUSS2D dtype



    def _define_results_dtype_and_formats(self):
        """Defines the numpy dtype AND the explicit FITS TFORM format strings for fitsio."""
        num_coeffs = 0
        try:
             from .zernike_fitter import generate_zernike_indices
             self.zernike_indices = generate_zernike_indices(self.params['n_max_zern'])
             num_coeffs = len(self.zernike_indices)
        except (NameError, ImportError): self.zernike_indices = []

        self._num_coeffs_for_fits = num_coeffs

        # Define numpy dtype parts using 'O' for variable columns
        base_dtype_list = [
            ('ORDER_NUM', 'i4'), ('IMGTYPE', 'i2'), ('SEGMENT', 'i4'),
            ('N_MAX_ZERN', 'i2'), ('R_MAX_ZERN', 'f4'),
            ('NUM_PEAKS_PROC', 'i4'), ('NUM_PIX_STACKED', 'i4'),
            ('MEDIAN_X', 'f4'), ('MEDIAN_Y', 'f4'),
            ('FIT_SUCCESS', 'bool'), # Use 'bool' instead of '?' for clarity
            ('RMSE', 'f4'), ('R_SQUARED', 'f4')
        ]
        coeff_dtype_list = []
        if num_coeffs > 0:
             # Define fixed vector dtypes for numpy
             coeff_dtype_list = [('COEFFS', f'{num_coeffs}f4'), ('ERR_COEFFS', f'{num_coeffs}f4')]

        varlen_dtype_list = [('X_STACK', 'O'), ('Y_STACK', 'O'), ('Z_STACK', 'O')]

        # Combine for numpy dtype list representation
        full_dtype_list_repr = base_dtype_list + coeff_dtype_list + varlen_dtype_list
        # Create the final numpy dtype
        self.results_table_dtype = np.dtype(full_dtype_list_repr)

        # --- Generate corresponding FITS TFORM format strings ---
        self.fits_column_formats = []
        # Iterate through the FINAL dtype's fields this time
        for name in self.results_table_dtype.names:
            field_info = self.results_table_dtype.fields[name]
            dtype = field_info[0] # The numpy dtype object for this field
            kind = dtype.kind
            itemsize = dtype.itemsize

            # *** CORRECTED LOGIC: Check for subdtype first for fixed vectors ***
            if dtype.subdtype is not None:
                 # This indicates a fixed-size array within the structured type (like COEFFS)
                 base_dtype, shape = dtype.subdtype
                 base_kind = base_dtype.kind
                 base_itemsize = base_dtype.itemsize
                 vec_len = shape[0] # Assumes 1D vector, adjust if multi-dimensional

                 if base_kind == 'f':
                      base_format = 'E' if base_itemsize == 4 else 'D'
                      self.fits_column_formats.append(f'{vec_len}{base_format}')
                 elif base_kind == 'i':
                      if base_itemsize == 2: base_format = 'I'
                      elif base_itemsize == 4: base_format = 'J'
                      else: base_format = 'K' # Assume 64-bit
                      self.fits_column_formats.append(f'{vec_len}{base_format}')
                 # Add other base kinds if needed (e.g., bool 'L', string 'A')
                 else:
                      warnings.warn(f"Unhandled base dtype kind '{base_kind}' in subdtype for column '{name}'. Using default vector format '{vec_len}A'.")
                      self.fits_column_formats.append(f'{vec_len}A')

            # --- Handle Scalar Types ---
            elif kind in 'iu':
                if itemsize == 2: self.fits_column_formats.append('I')
                elif itemsize == 4: self.fits_column_formats.append('J')
                elif itemsize == 8: self.fits_column_formats.append('K')
                else: self.fits_column_formats.append('J')
            elif kind == 'f': # Scalar float
                 self.fits_column_formats.append('E' if itemsize == 4 else 'D')
            elif kind == 'b' or kind == '?': # Boolean (kind can be '?' sometimes)
                 self.fits_column_formats.append('L')
            elif kind == 'O': # Object -> Assume Variable Length Array of float32
                 self.fits_column_formats.append('PE') # Use 'PD' if underlying data is f8/float64
            # Kind 'V' (void) represents the structured type itself or fixed bytes,
            # it shouldn't be hit here when checking individual fields *unless*
            # it represents a fixed-size byte array - handle if needed.
            # Kind 'S'/'U' for strings -> 'A'
            elif kind in 'SU':
                 # fitsio typically handles strings automatically, but 'A' is the explicit char format
                 # Need length for 'nA' format if writing fixed-width strings
                 # For simplicity here, assuming scalar strings map to 'A' if needed
                 # If fitsio handles them without explicit format, this might not be needed
                 self.fits_column_formats.append('A') # Or determine length n -> 'nA'
                 warnings.warn(f"Assuming 'A' format for string column '{name}'. Check fitsio documentation if fixed width needed.")

            else:
                 warnings.warn(f"Unhandled numpy dtype kind '{kind}' for column '{name}'. Using default FITS format 'A'.")
                 self.fits_column_formats.append('A')

        # Verification
        if len(self.fits_column_formats) != len(self.results_table_dtype.names):
             # This should not happen now, but good to keep check
             raise ValueError("Internal Error: Mismatch between number of generated FITS formats and dtype fields.")

        # print(f"Defined results table dtype and FITS formats ({len(self.fits_column_formats)} columns).")
        # print(f"  Formats: {self.fits_column_formats}") # Debug print

    def _define_peak_catalog_dtype(self):
        """Defines the numpy dtype for the PEAK_CATALOG table."""
        # Define columns for the peak catalog
        # Essential: Order, ImgType, Segment, Original Peak X, Original Peak Y
        # Useful additions: Peak Raw Flux, Peak Local S/N (if calculable), Cluster Label (from DBSCAN)
        peak_dtype_list = [
            ('ORDER_NUM', 'i4'),    # Assigned final order number
            ('IMGTYPE', 'i2'),    # 0=A, 1=B
            ('SEGMENT', 'i4'),    # Assigned segment index within order/image
            ('PEAK_X', 'i4'),     # Original integer X pixel coord of the peak
            ('PEAK_Y', 'i4'),     # Original integer Y pixel coord of the peak
            ('RAW_FLUX', 'f4'),   # Raw flux value at the peak pixel (useful for filtering)
            # Optional: Add more?
            # ('CLUSTER_LABEL', 'i4'), # Raw DBSCAN label (might be useful for diagnostics)
            # ('PEAK_SNR', 'f4')      # Estimated local S/N (harder to calculate robustly here)
        ]
        self.peak_catalog_dtype = np.dtype(peak_dtype_list)
        # FITS formats for this simple table (no var-len)
        self.peak_catalog_fits_formats = ['J', 'I', 'J', 'J', 'J', 'E'] # Adjust if optional cols added
        
    def _define_gauss2d_dtype_and_formats(self):
        """Defines the numpy dtype and FITS TFORM for the GAUSS2D parameters table."""
        # Parameters from twoD_Gaussian: amplitude, xo, yo, sigma_x, sigma_y, theta_rad, offset
        num_gauss_params = 7
        

        gauss2d_dtype_list = [
            ('ORDER_NUM', 'i4'),      # Assigned final order number
            ('IMGTYPE', 'i2'),      # 0=A, 1=B
            ('SEGMENT', 'i4'),      # Assigned segment index within order/image
            ('IDX', 'i4'),# Original index of the peak within the segment's input list
            ('PEAK_X', 'i4'),    # Original integer X pixel coord of the peak
            ('PEAK_Y', 'i4'),    # Original integer Y pixel coord of the peak
            # ('GAUSS_PARAMS', f'{num_gauss_params}f4'), # amplitude, xo_s, yo_s, sx, sy, th_r, off_n
            # ('GAUSS_ERRORS', f'{num_gauss_params}f4')
            *((name,'f4',(2,)) for name in GAUSS_PARAMS_NAME)
            # Optional: Add reduced_chi2 from the fit?
            # ('RED_CHI2', 'f4')
        ]
        self.gauss2d_table_dtype = np.dtype(gauss2d_dtype_list)

        # --- Generate corresponding FITS TFORM format strings ---
        self.gauss2d_fits_formats = []
        for name in self.gauss2d_table_dtype.names:
            field_info = self.gauss2d_table_dtype.fields[name]
            dtype = field_info[0]
            kind = dtype.kind
            itemsize = dtype.itemsize

            if dtype.subdtype is not None: # Fixed vector (GAUSS_PARAMS, GAUSS_ERRORS)
                 base_dtype, shape = dtype.subdtype
                 base_kind = base_dtype.kind
                 vec_len = shape[0]
                 base_format = 'E' if base_kind == 'f' else 'J' # Assuming float or int
                 self.gauss2d_fits_formats.append(f'{vec_len}{base_format}')
            elif kind in 'iu':
                if itemsize == 2: self.gauss2d_fits_formats.append('I')
                elif itemsize == 4: self.gauss2d_fits_formats.append('J')
                else: self.gauss2d_fits_formats.append('K')
            elif kind == 'f':
                 self.gauss2d_fits_formats.append('E' if itemsize == 4 else 'D')
            else: # Fallback, should not be hit for this dtype
                 warnings.warn(f"Unhandled kind '{kind}' for GAUSS2D column '{name}'. Using 'A'.")
                 self.gauss2d_fits_formats.append('A')

        if len(self.gauss2d_fits_formats) != len(self.gauss2d_table_dtype.names):
             raise ValueError("Internal Error: Mismatch in GAUSS2D FITS formats and dtype fields.")
        # print("Defined GAUSS2D table dtype and FITS formats.")
        
    def _ensure_fits_structure(self):
        """
        Ensures the FITS file exists and has a Primary HDU.
        Does NOT create the ZERNIKE HDU here.
        """
        # This flag now only tracks if the file exists physically,
        # not necessarily if the ZERNIKE HDU is present.
        self._fits_file_created = self.output_fits_path.exists()

        if not self.output_fits_path: print("Error: Output FITS path not set."); return False

        # If file doesn't exist OR clobber is allowed and it *does* exist
        needs_creation = not self._fits_file_created or (self._fits_file_created and self.clobber_allowed)

        if needs_creation:
             if self._fits_file_created and self.clobber_allowed:
                  print(f"Output file {self.output_fits_path} exists. Overwriting (clobber=True).")
                  try: self.output_fits_path.unlink()
                  except OSError as e: print(f"Warning: Could not delete existing file: {e}")
                  self._fits_file_created = False

             print(f"Creating new FITS file with Primary HDU: {self.output_fits_path}")
             # --- Prepare Primary HDU Header ---
             primary_hdr = fitsio.FITSHDR()
             # ... (populate primary_hdr as before) ...
             primary_hdr['ORIGFILE'] = self.lfc_path.name
             primary_hdr['BIASFILE'] = self.bias_path.name if self.bias_path else 'None'
             primary_hdr['DETECTOR'] = self.detector if self.detector else 'Unknown'
             primary_hdr['IMG_H'] = self.image_shape[0] if self.image_shape else -1
             primary_hdr['IMG_W'] = self.image_shape[1] if self.image_shape else -1
             primary_hdr['DATE'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
             primary_hdr['AUTHOR'] = 'harps.twodim.analyzer'
             for key, val in self.params.items():
                 hdr_key = key.upper()[:8]; val_str = str(val)
                 if len(val_str) < 68: primary_hdr[hdr_key] = val if isinstance(val,(int,float,bool,str)) else val_str

             # --- Write ONLY Primary HDU ---
             try:
                 with fitsio.FITS(self.output_fits_path, 'rw', clobber=True) as fits: # Clobber must be true here
                     fits.write(None, header=primary_hdr)
                 self._fits_file_created = True
                 self.clobber_allowed = False # Disable clobber for subsequent operations
                 print("  Successfully wrote Primary HDU.")
                 return True
             except Exception as e:
                 print(f"Error creating initial FITS file with Primary HDU: {e}")
                 traceback.print_exc()
                 return False
        else:
            # File exists and clobber is false
            print(f"Output file {self.output_fits_path} exists (clobber=False).")
            # We don't need to validate the ZERNIKE HDU here, _update_fits_file will handle it.
            self._fits_file_created = True # Confirm file exists
            self.clobber_allowed = False
            return True




    def _update_fits_file(self, new_results_list):
        """
        Writes or appends results to the ZERNIKE BINTABLE in the FITS file.
        Creates the HDU on the first write operation for this file handle.
        """
        if not new_results_list: return
        # Ensure Primary HDU exists (creates file if needed)
        if not self._ensure_fits_structure():
             print("Error: Cannot ensure base FITS file structure exists.")
             return
        if self.results_table_dtype is None or self.fits_column_formats is None:
             print("Error: Results table dtype or FITS formats not defined."); return False

        print(f"Updating FITS file {self.output_fits_path} with {len(new_results_list)} result(s)...")
        extname = 'ZERNIKE'
        col_names = list(self.results_table_dtype.names)
        col_formats = self.fits_column_formats

        try:
            # Open the FITS file in read-write mode
            with fitsio.FITS(self.output_fits_path, 'rw') as fits:

                # --- Check if ZERNIKE HDU exists ---
                hdu_exists = extname in fits

                # --- Prepare Data to Write/Append ---
                rows_to_write_or_append = []
                existing_keys = set()

                if hdu_exists:
                    # Read existing keys ONLY if appending
                    try:
                        existing_data = fits[extname].read(columns=['ORDER_NUM', 'IMGTYPE', 'SEGMENT'])
                        if existing_data is not None and len(existing_data) > 0:
                             existing_keys = set(zip(existing_data['ORDER_NUM'], existing_data['IMGTYPE'], existing_data['SEGMENT']))
                        print(f"  Read {len(existing_keys)} existing segment keys for duplicate check.")
                    except Exception as e:
                        print(f"Warning: Could not read existing keys from {extname}: {e}. Proceeding without duplicate check for this batch.")
                        existing_keys = set() # Cannot check duplicates reliable

                    # Filter new results if appending
                    num_duplicates = 0
                    for result_dict in new_results_list:
                        key = (result_dict['ORDER_NUM'], result_dict['IMGTYPE'], result_dict['SEGMENT'])
                        if key not in existing_keys: rows_to_write_or_append.append(result_dict)
                        else: num_duplicates += 1
                    if num_duplicates > 0: print(f"  Skipped {num_duplicates} duplicate segment entries.")

                else:
                    # If HDU doesn't exist, all results in this batch are new
                    rows_to_write_or_append = new_results_list
                    print(f"  {extname} HDU not found. Will create and write {len(rows_to_write_or_append)} new rows.")


                if not rows_to_write_or_append:
                     print("  No new unique results to write/append.")
                     return # Nothing more to do

                # --- Convert list of dicts to structured array ---
                num_write = len(rows_to_write_or_append)
                output_struct = np.zeros(num_write, dtype=self.results_table_dtype)
                valid_output_rows = 0
                num_coeffs = self._num_coeffs_for_fits

                for i, row_dict in enumerate(rows_to_write_or_append):
                     try:
                         for name in self.results_table_dtype.names:
                              if name in row_dict:
                                  value = row_dict[name]
                                  if name in ['X_STACK', 'Y_STACK', 'Z_STACK']: output_struct[i][name] = np.asarray(value, dtype='f4')
                                  elif name in ['COEFFS', 'ERR_COEFFS']:
                                      expected_len = num_coeffs
                                      current_val = np.asarray(value, dtype='f4')
                                      output_struct[i][name] = current_val if current_val.shape == (expected_len,) else np.full(expected_len, np.nan, dtype='f4')
                                  else: output_struct[i][name] = value
                         valid_output_rows += 1
                     except Exception as e: print(f"Error preparing row {i} for writing: {e}")

                if valid_output_rows == 0: print("Error: No valid rows prepared for writing."); return

                # --- Perform Write or Append ---
                if hdu_exists:
                    # Append to existing HDU
                    print(f"  Appending {valid_output_rows} rows to {extname} HDU...")
                    fits[extname].append(output_struct[:valid_output_rows])
                else:
                    # Write the new HDU for the first time with data
                    print(f"  Writing new {extname} HDU with {valid_output_rows} rows...")
                    results_hdr = fitsio.FITSHDR() # Create header for the new HDU
                    results_hdr['EXTNAME'] = extname
                    if self.zernike_indices: # Add Zernike indices
                        indices_str = ",".join([f"{n}_{m}" for n, m in self.zernike_indices])
                        max_len = 68; parts = [indices_str[i:i+max_len] for i in range(0, len(indices_str), max_len)]
                        for idx, part in enumerate(parts): results_hdr[f'ZNIND{idx}'] = part
                    # Use write_table, providing explicit formats as it might still be needed
                    # by fitsio even when writing non-empty data with var-len columns
                    fits.write_table(
                        data=output_struct[:valid_output_rows],
                        names=col_names,
                        formats=col_formats,
                        header=results_hdr,
                        extname=extname
                    )

                print(f"  Successfully wrote/appended {valid_output_rows} results.")

        except Exception as e:
            print(f"Error writing/appending to FITS file: {e}")
            traceback.print_exc()
            
    def _update_gauss2d_fits_file(self, new_gauss_fit_list):
        """
        Writes or appends individual Gaussian fit results to the GAUSS2D BINTABLE.
        """
        if not new_gauss_fit_list: return
        if not self._ensure_fits_structure():
             print("Error: Cannot ensure base FITS file structure for GAUSS2D.")
             return
        if self.gauss2d_table_dtype is None or self.gauss2d_fits_formats is None:
             print("Error: GAUSS2D table dtype or FITS formats not defined."); return

        print(f"Updating FITS file {self.output_fits_path} with {len(new_gauss_fit_list)} GAUSS2D candidate result(s)...")
        extname = 'GAUSS2D'
        col_names = list(self.gauss2d_table_dtype.names)
        col_formats = self.gauss2d_fits_formats

        try:
            with fitsio.FITS(self.output_fits_path, 'rw') as fits:
                hdu_exists = extname in fits
                rows_to_write_or_append = []
                existing_keys = set()

                if hdu_exists:
                    try:
                        existing_data = fits[extname].read(columns=['ORDER_NUM', 'IMGTYPE', 'SEGMENT', 'IDX'])
                        if existing_data is not None and len(existing_data) > 0:
                             existing_keys = set(zip(existing_data['ORDER_NUM'], existing_data['IMGTYPE'],
                                                     existing_data['SEGMENT'], existing_data['IDX']))
                        # print(f"  Read {len(existing_keys)} existing GAUSS2D keys for duplicate check.") # Less verbose
                    except Exception as e:
                        print(f"Warning: Could not read existing keys from {extname}: {e}. Proceeding without duplicate check for this batch.")
                        existing_keys = set()

                    num_duplicates = 0
                    valid_new_entries = [] # Filter for non-None entries first
                    for item in new_gauss_fit_list:
                        if isinstance(item, dict): # Ensure item is a dictionary
                            valid_new_entries.append(item)
                        else:
                            print(f"Warning: Encountered a non-dictionary item in new_gauss_fit_list: {type(item)}. Skipping.")
                    
                    for result_dict in valid_new_entries:
                        try: # Add try-except for key access, though 'isinstance' should prevent None
                            key = (result_dict['ORDER_NUM'], result_dict['IMGTYPE'],
                                   result_dict['SEGMENT'], result_dict['IDX'])
                            if key not in existing_keys:
                                rows_to_write_or_append.append(result_dict)
                            else:
                                num_duplicates +=1
                        except TypeError as te: # Should be caught by isinstance now
                            print(f"Warning: TypeError accessing keys in GAUSS2D result_dict (was it None?): {result_dict}. Error: {te}")
                        except KeyError as ke:
                            print(f"Warning: Missing key in GAUSS2D result_dict: {ke}. Dict: {result_dict}")

                    if num_duplicates > 0: print(f"  Skipped {num_duplicates} duplicate GAUSS2D entries.")
                else:
                    # If HDU doesn't exist, filter for valid dictionaries before assigning
                    for item in new_gauss_fit_list:
                        if isinstance(item, dict):
                            rows_to_write_or_append.append(item)
                        else:
                            print(f"Warning: Encountered a non-dictionary item when creating GAUSS2D HDU: {type(item)}. Skipping.")
                    if rows_to_write_or_append:
                        print(f"  {extname} HDU not found. Will create and write {len(rows_to_write_or_append)} new rows.")

                if not rows_to_write_or_append:
                     print("  No new unique GAUSS2D results to write/append.")
                     return

                # Convert list of dicts to structured array
                output_struct = np.zeros(len(rows_to_write_or_append), dtype=self.gauss2d_table_dtype)
                valid_rows_for_struct = 0
                for i, row_dict in enumerate(rows_to_write_or_append):
                    # At this point, row_dict should absolutely be a dictionary.
                    # The error location suggests it is not.
                    # The added isinstance checks above should prevent Nones from reaching here.
                    if not isinstance(row_dict, dict): # Defensive check, should have been filtered
                        print(f"Critical Warning: row_dict at index {i} is not a dict: {type(row_dict)}. This should not happen.")
                        continue 

                    try:
                        for name in self.gauss2d_table_dtype.names:
                            if name in row_dict: # This is where the original error was
                                output_struct[valid_rows_for_struct][name] = row_dict[name]
                            # else: # Optional: Warning for missing keys, but might be too verbose
                            #    print(f"Debug: Key '{name}' not in row_dict for GAUSS2D struct. Keys: {list(row_dict.keys())}")
                        valid_rows_for_struct += 1
                    except Exception as e:
                        print(f"Error populating structured array for GAUSS2D row {i} (data: {row_dict}): {e}")
                        # Optionally, fill this row with NaNs or skip it
                        # For now, we just don't increment valid_rows_for_struct if there's an error for THIS row

                if valid_rows_for_struct == 0:
                    print("  No valid rows could be prepared for GAUSS2D structured array.")
                    return
                
                # Slice output_struct to only include validly populated rows
                final_output_struct = output_struct[:valid_rows_for_struct]


                if hdu_exists:
                    if len(final_output_struct) > 0:
                        print(f"  Appending {len(final_output_struct)} rows to {extname} HDU...")
                        fits[extname].append(final_output_struct)
                    else:
                        print(f"  No valid GAUSS2D rows to append after final checks.")
                else:
                    if len(final_output_struct) > 0:
                        print(f"  Writing new {extname} HDU with {len(final_output_struct)} rows...")
                        gauss2d_hdr = fitsio.FITSHDR()
                        gauss2d_hdr['EXTNAME'] = extname
                        gauss2d_hdr['COMMENT'] = 'Parameters of 2D Gaussian fits to individual lines.'
                        gauss2d_hdr['COMMENT'] = 'GAUSS_PARAMS: amplitude, xo_stamp, yo_stamp, sigma_x, sigma_y, theta_rad, offset'
                        fits.write_table(
                            data=final_output_struct,
                            names=col_names,
                            formats=col_formats,
                            header=gauss2d_hdr,
                            extname=extname
                        )
                    else:
                        print(f"  No valid GAUSS2D rows to write for new HDU after final checks.")
                
                if len(final_output_struct) > 0:
                    print(f"  Successfully wrote/appended {len(final_output_struct)} GAUSS2D results.")
        except Exception as e:
            print(f"Error writing/appending GAUSS2D to FITS file: {e}")
            traceback.print_exc()
            
    def load_data(self, detector='red'):
        """Loads data for the specified detector and checks output FITS status."""
        self.detector = detector.lower()
        self.hdu_index = 2 if self.detector == 'red' else 1
        self.params['peak_hard_cut'] = 310 if detector=='red' else 420
        print(f"\nLoading data for detector: {self.detector} (HDU {self.hdu_index})...")
        try:
            # --- Load Image Data ---
            self.image_data = load_echelle_data(self.lfc_path, self.bias_path, hdu_index=self.hdu_index)
            self.image_shape = self.image_data.shape
            print(f"Data loaded successfully. Shape: {self.image_shape}")

            # --- Define Output Path ---
            self.output_fits_path = self.output_dir / f"{self.lfc_path.stem}_{self.detector}{self.output_suffix}"
            self.output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
            print(f"Output FITS defined as: {self.output_fits_path}")

            # --- Check Existing Output FITS File ---
            self._fits_file_created = self.output_fits_path.exists() # Does it exist at all?
            extname_to_check = 'ZERNIKE' # Use the target extension name

            if self._fits_file_created:
                 print(f"Checking existing FITS file: {self.output_fits_path}")
                 # Open in read-write ('rw') to check structure but allow appending later
                 try:
                     with fitsio.FITS(self.output_fits_path, 'rw') as fits:
                         print(fits)
                         if extname_to_check in fits and fits[extname_to_check].get_exttype() == 'BINTABLE':
                              # Valid structure found
                              print("  Existing file is valid for appending.")
                              # If valid, we *do not* want clobber, force append mode later
                              self.clobber_allowed = False # Ensure subsequent writes append
                         else:
                              # File exists but lacks the required HDU/type
                              print(f"  Existing file exists but lacks a valid {extname_to_check} BINTABLE HDU.")
                              if self.clobber_allowed:
                                   print("  Clobber is allowed, will overwrite when writing.")
                                   # We don't delete here, let _ensure_fits_structure handle it if needed
                                   self._fits_file_created = False # Mark that structure needs creation
                              else:
                                   # Cannot proceed if file is invalid and clobber is off
                                   print("  Clobber is FALSE, cannot proceed with invalid existing file.")
                                   self.image_data = None # Invalidate loaded data
                                   return False
                 except Exception as e:
                      # Error opening/reading the existing file
                      print(f"  Error reading/validating existing FITS file {self.output_fits_path}: {e}.")
                      if self.clobber_allowed:
                           print("  Clobber is allowed, attempting to overwrite.")
                           self._fits_file_created = False # Mark for recreation
                      else:
                           print("  Clobber is FALSE, cannot proceed with problematic existing file.")
                           self.image_data = None # Invalidate loaded data
                           return False
            else:
                 # File does not exist
                 print("  Output file does not exist. Will create new.")
                 # Clobber is effectively true for initial creation
                 self.clobber_allowed = True # Ensure it's set correctly
                 self._fits_file_created = False # Structure needs creation

            # If we reach here, loading was successful and FITS status is determined
            return True

        except FileNotFoundError as e:
            print(f"Error: {e}")
            self.image_data = None
            return False
        except Exception as e:
            print(f"Error during data loading or FITS check: {e}")
            traceback.print_exc()
            self.image_data = None
            return False

    def find_all_peaks(self, plot_interactive=False, detector='red'): # Accept flag
        """Finds peaks in the loaded image data."""
        if self.image_data is None: 
            print("Error: Image data not loaded. Loading now")
            try:
                self.load_data(detector=detector)
            except:
                print("Failed loading data. Return False")
                return False
        print("\nFinding peaks...")
        try:
            self.all_peaks_xy = find_peaks(
                self.image_data,
                min_distance=self.params['peak_min_distance'],
                # threshold_abs=self.params['peak_threshold_abs'],
                # threshold_rel=self.params['peak_threshold_rel'],
                hard_cut=self.params['peak_hard_cut'],
                plot_interactive=plot_interactive # Pass flag
            )
            if self.all_peaks_xy is None or len(self.all_peaks_xy) == 0: print("Warning: No peaks found."); return False
            print(f"Found {len(self.all_peaks_xy)} peaks.")
            return True
        except Exception as e: print(f"Error during peak finding: {e}"); traceback.print_exc(); self.all_peaks_xy = None; return False


    def cluster_orders(self, plot_interactive=False): # Accept flag
        """Clusters peaks and calculates median segment positions."""
        if self.all_peaks_xy is None: print("Error: Peaks not found."); return False
        if self.image_shape is None: 
            print("Error: Image shape not known.")
            try:
                self.load_data()
                if not self.all_peaks_xy:
                    self.find_all_peaks()
            except:
                return False
        if plot_interactive and self.image_data is None: # Check for image_data if plotting
            print("Error: image_data not loaded, cannot plot overlay. Load data first.")
            return False # This return False seems fine.
    
        print("\nClustering peaks into orders...")
        try:
            _, self.paired_orders_dict = cluster_peaks_to_orders(
                self.all_peaks_xy, self.image_shape,
                eps_x=self.params['cluster_eps_x'], eps_y=self.params['cluster_eps_y'],
                min_samples=self.params['cluster_min_samples'],
                plot_interactive=plot_interactive, # Pass flag
                image_data=self.image_data if plot_interactive else None # Pass image_data only if plotting
            )
            if not self.paired_orders_dict: print("Warning: No orders/clusters found."); return False
            print(f"Found {len(self.paired_orders_dict)} potential order pairs.")

            print("Calculating median positions for segments...")
            self.median_segment_positions = {}
            num_seg = self.params['num_segments']
            for order_num, pair_info in self.paired_orders_dict.items():
                 for img_type_str, peaks_full_order in pair_info.items():
                     if peaks_full_order is None or len(peaks_full_order) == 0: continue
                     img_type_int = 0 if img_type_str == 'A' else 1

                     # --- Peaks are already sorted by Y (or along trace) by cluster_peaks_to_orders ---
                     # peaks_in_order = peaks_full_order # Use the already Y-sorted list

                     # --- Segment by index along the Y-sorted list ---
                     # This assumes the Y-sort reasonably follows the trace path
                     num_peaks_in_order = len(peaks_full_order)
                     indices_for_order = np.arange(num_peaks_in_order)
                     # Split the *indices* into N segments
                     segmented_peak_indices_list = np.array_split(indices_for_order, num_seg)

                     # --- Calculate median position for each segment ---
                     for seg_idx, indices_in_segment in enumerate(segmented_peak_indices_list):
                          if len(indices_in_segment) == 0:
                               self.median_segment_positions[(order_num, img_type_int, seg_idx)] = (np.nan, np.nan)
                               continue

                          # Get the actual peak data for this segment using the indices
                          seg_peaks = peaks_full_order[indices_in_segment]

                          # Calculate median X and Y for this segment
                          # Now, this represents the median position of a contiguous group *along the trace*
                          if len(seg_peaks) > 0:
                               med_x = np.median(seg_peaks[:, 0])
                               med_y = np.median(seg_peaks[:, 1])
                          else: # Should not happen, but safety
                                med_x, med_y = np.nan, np.nan

                          # --- Store the calculated median position ---
                          self.median_segment_positions[(order_num, img_type_int, seg_idx)] = (med_x, med_y)

            print("Median segment positions calculated (using median of peaks segmented along Y-sorted trace).")
            return True
        except Exception as e: print(f"Error during clustering/median calc: {e}"); traceback.print_exc(); self.paired_orders_dict = None; return False
        
    def _process_single_stamp(self, peak_xy):
        """ Internal helper: Processes a single peak. Returns dict or None. """
        peak_x, peak_y = peak_xy
        sw = self.params['stamp_half_width']
        snr_thresh = self.params['fit_threshold_snr']
        img_h, img_w = self.image_shape

        x_min, x_max = max(0, peak_x - sw), min(img_w, peak_x + sw + 1)
        y_min, y_max = max(0, peak_y - sw), min(img_h, peak_y + sw + 1)
        if (x_max - x_min) != (2*sw + 1) or (y_max - y_min) != (2*sw + 1): return None

        stamp_data = self.image_data[y_min:y_max, x_min:x_max]
        total_flux = np.sum(stamp_data)
        if total_flux <= 1e-9: return None
        stamp_norm = stamp_data / total_flux

        stamp_h, stamp_w = stamp_norm.shape
        y_sg, x_sg = np.meshgrid(np.arange(stamp_h), np.arange(stamp_w), indexing='ij')
        p0 = (np.max(stamp_norm), stamp_w/2.0, stamp_h/2.0, sw/3.0, sw/3.0, 0.0, np.min(stamp_norm))
        bounds = ([0, -stamp_w, -stamp_h, 0.1, 0.1, -np.pi, -np.inf],
                  [np.inf, 2*stamp_w, 2*stamp_h, stamp_w, stamp_h, np.pi, np.inf])
        try:
            popt, pcov = curve_fit(twoD_Gaussian, (x_sg, y_sg), stamp_norm.ravel(), p0=p0, bounds=bounds, maxfev=5000)
            if not np.all(np.isfinite(pcov)): raise RuntimeError("Non-finite cov")
            perr = np.sqrt(np.diag(pcov))
            amp_snr = popt[0] / perr[0] if perr[0] > 1e-12 else np.inf
            if amp_snr < snr_thresh or not np.all(np.isfinite(popt)) or popt[3] <= 0 or popt[4] <= 0: return None
        except (RuntimeError, ValueError): return None

        amp_n, xc_s, yc_s, sx, sy, th, off_n = popt
        xc_abs, yc_abs = x_min + xc_s, y_min + yc_s
        y_ag, x_ag = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
        rel_x, rel_y = x_ag - xc_abs, y_ag - yc_abs
        th_major = th if sx >= sy else th + np.pi / 2.0
        rot_ang = -th_major
        cos_r, sin_r = np.cos(rot_ang), np.sin(rot_ang)
        rel_x_rot = rel_x * cos_r + rel_y * sin_r
        rel_y_rot = -rel_x * sin_r + rel_y * cos_r

        model_norm = twoD_Gaussian((x_sg, y_sg), *popt)
        res_norm_1d = stamp_norm.ravel() - model_norm
        var_norm = np.maximum(stamp_data.ravel(), 1e-9) / (total_flux**2)
        res_div_sig = res_norm_1d / np.sqrt(var_norm)
        chi2 = np.sum(res_div_sig**2)
        dof = max(1, len(stamp_norm.ravel()) - len(popt))
        red_chi2 = chi2 / dof

        return {
            'x_rotated': rel_x_rot.ravel().astype('f4'),
            'y_rotated': rel_y_rot.ravel().astype('f4'),
            'z_norm': stamp_norm.ravel().astype('f4'),
            'pars': np.asarray([amp_n, xc_abs, yc_abs, sx, sy, th_major, off_n]),
            'perr': perr.astype('f4'),
            'plot_info': {
                'stamp_data': stamp_data, 'residuals_norm_2d': res_div_sig.reshape(stamp_h, stamp_w),
                'xo_stamp': xc_s, 'yo_stamp': yc_s, 'sigma_x': sx, 'sigma_y': sy,
                'theta_rad': th_major, 'chi2_reduced': red_chi2
            }
        }



    def _analyze_segment_compute_only(self, segment_peaks_xy, order_num, img_type_int, segment_idx, plot_config=None):
        """
        Computes results for a single segment without saving. Handles plotting.
        Returns the results dictionary.
        """
        if plot_config is None: plot_config = {}
        save_plots = plot_config.get('save_plots', False)
        base_plot_dir = Path(plot_config.get('plot_dir', self.output_dir / self.lfc_path.name / "plots"))

        stamps_config = plot_config.get('stamps', {})
        zernike_config = plot_config.get('zernike', {})
        spectrum_config = plot_config.get('spectrum', {})

        plot_stamps = stamps_config.get('enable', False)
        plot_zernike = zernike_config.get('enable', False)
        plot_spectrum = spectrum_config.get('enable', False)

        print(f"-- Computing: Order {order_num}, ImgType {img_type_int}, Segment {segment_idx} ({len(segment_peaks_xy)} peaks) --")
        segment_results_x, segment_results_y, segment_results_z = [], [], []
        # segment_results_pars, segment_results_errs = []
        individual_gauss_fits = []
        plotting_info_list = []
        num_peaks_processed = 0

        for i, peak_xy in enumerate(segment_peaks_xy):
            processed_stamp = self._process_single_stamp(peak_xy)
            if processed_stamp:
                segment_results_x.append(processed_stamp['x_rotated'])
                segment_results_y.append(processed_stamp['y_rotated'])
                segment_results_z.append(processed_stamp['z_norm'])
                # --- Collect Gaussian fit data for GAUSS2D table ---
                segment_results_dict = {
                    'ORDER_NUM': order_num,
                    'IMGTYPE': img_type_int,
                    'SEGMENT': segment_idx,
                    'IDX': i, # Index within the input segment_peaks_xy
                    'PEAK_X': peak_xy[0],
                    'PEAK_Y': peak_xy[1],
                    'CHISQ': processed_stamp['plot_info']['chi2_reduced'],
                    'THETA_MAJOR': processed_stamp['plot_info']['theta_rad']
                    }
                parameters_dict = dict((name,((processed_stamp['pars'][k], 
                                               processed_stamp['perr'][k])))
                    for k,name in enumerate(GAUSS_PARAMS_NAME))
                segment_results_dict.update(parameters_dict)
                individual_gauss_fits.append(segment_results_dict)
                
                num_peaks_processed += 1
                if plot_stamps:
                     processed_stamp['plot_info']['peak_index'] = i
                     plotting_info_list.append(processed_stamp['plot_info'])

        # --- Handle No Processed Peaks ---
        median_pos = self.median_segment_positions.get((order_num, img_type_int, segment_idx), (np.nan, np.nan))
        median_x, median_y = median_pos
        coeffs_nan = np.full(len(self.zernike_indices) if self.zernike_indices else 0, np.nan, dtype='f4')
        base_output = {
            'ORDER_NUM': order_num, 
            'IMGTYPE': img_type_int, 
            'SEGMENT': segment_idx,
            'N_MAX_ZERN': self.params['n_max_zern'], 
            'R_MAX_ZERN': self.params['r_max_zern'],
            'NUM_PEAKS_PROC': num_peaks_processed, 
            'NUM_PIX_STACKED': 0,
            'MEDIAN_X': median_x, 
            'MEDIAN_Y': median_y, 
            'FIT_SUCCESS': False,
            'RMSE': np.nan, 
            'R_SQUARED': np.nan,
            'COEFFS': coeffs_nan, 
            'ERR_COEFFS': coeffs_nan,
            'X_STACK': np.array([], dtype='f4'), 
            'Y_STACK': np.array([], dtype='f4'), 
            'Z_STACK': np.array([], dtype='f4')
        }
        if not segment_results_x: print(f"Segment {segment_idx}: No peaks processed."); return base_output

        X_stack = np.concatenate(segment_results_x)
        Y_stack = np.concatenate(segment_results_y)
        Z_stack = np.concatenate(segment_results_z)
        num_pixels_stacked = len(X_stack)
        print(f"Segment {segment_idx}: Processed {num_peaks_processed} peaks, {num_pixels_stacked} stacked pixels.")

        # Update output dict with stacked data
        base_output.update({'NUM_PIX_STACKED': num_pixels_stacked,
                           'X_STACK': X_stack, 
                           'Y_STACK': Y_stack, 
                           'Z_STACK': Z_stack,
                           }
                           )


        # --- Plotting Stamps ---
        img_type_str = 'A' if img_type_int == 0 else 'B'
        if plot_stamps and plotting_info_list:
             custom_subdir = stamps_config.get('subdir')
             stamps_final_dir = base_plot_dir / custom_subdir if custom_subdir else base_plot_dir / f"Order{order_num}{img_type_str}" / "stamps"
             stamps_final_dir.mkdir(parents=True, exist_ok=True)
             stamps_filename = stamps_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_stamps.pdf" if save_plots else None
             residuals_filename = stamps_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_norm_residuals.pdf" if save_plots else None
             print(f"Segment {segment_idx}: Generating stamp plots (Save: {save_plots}, Dir: {stamps_final_dir})...")
             try: plot_raw_data_stamps(plotting_info_list, segment_idx, filename=stamps_filename)
             except Exception as e: print(f"ERROR plotting raw stamps: {e}")
             try: plot_normalized_residuals_stamps(plotting_info_list, segment_idx, filename=residuals_filename)
             except Exception as e: print(f"ERROR plotting residual stamps: {e}")


        # --- Zernike Fitting ---
        if ZERNPY_AVAILABLE and num_pixels_stacked > len(base_output['COEFFS']) + 2:
            fitter = ZernikeFitter(n_max=self.params['n_max_zern'], r_max=self.params['r_max_zern'])
            if self.zernike_indices and fitter.zernike_indices != self.zernike_indices:
                 fitter.zernike_indices = self.zernike_indices; fitter.n_coeffs = len(fitter.zernike_indices)

            ig = {'xc': 0.0, 'yc': 0.0}; bnds = {'lower': [-0.5, -0.5] + [-np.inf]*fitter.n_coeffs, 'upper': [0.5, 0.5] + [np.inf]*fitter.n_coeffs}
            fit_success = fitter.fit(X_stack, Y_stack, Z_stack, initial_guess=ig, bounds=bnds, verbose=False)
            base_output['FIT_SUCCESS'] = fit_success

            if fit_success:
                results = fitter.get_results(include_coeffs_table=False)
                base_output.update({
                    'RMSE': results.get('rmse', np.nan), 'R_SQUARED': results.get('r_squared', np.nan),
                    'COEFFS': results.get('fitted_coeffs', coeffs_nan), 'ERR_COEFFS': results.get('err_coeffs', coeffs_nan)
                })
                if base_output['COEFFS'] is None or not isinstance(base_output['COEFFS'], np.ndarray): base_output['COEFFS'] = coeffs_nan
                if base_output['ERR_COEFFS'] is None or not isinstance(base_output['ERR_COEFFS'], np.ndarray): base_output['ERR_COEFFS'] = coeffs_nan
                np.nan_to_num(base_output['COEFFS'], copy=False, nan=np.nan)
                np.nan_to_num(base_output['ERR_COEFFS'], copy=False, nan=np.nan)
                print(f"Segment {segment_idx}: Zernike fit successful. RMSE={base_output['RMSE']:.4f}, R^2={base_output['R_SQUARED']:.4f}")

                # --- Plotting Zernike Results ---
                if plot_zernike or plot_spectrum:
                     custom_subdir = zernike_config.get('subdir')
                     zernike_final_dir = base_plot_dir / custom_subdir if custom_subdir else base_plot_dir / f"Order{order_num}{img_type_str}" / "zernike"
                     zernike_final_dir.mkdir(parents=True, exist_ok=True)
                     print(f"Segment {segment_idx}: Generating Zernike plots (Save: {save_plots}, Dir: {zernike_final_dir})...")
                     if plot_zernike:
                          zcomp_fn = zernike_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_zernike_fit.pdf" if save_plots else None
                          zres_fn = zernike_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_zernike_residuals.pdf" if save_plots else None
                          title_fit = f"Zernike Fit (O={order_num}{img_type_str}, S={segment_idx})"; title_res = f"Zernike Residuals (O={order_num}{img_type_str}, S={segment_idx})"
                          try: fitter.plot_fit_comparison(X_stack, Y_stack, Z_stack, title=title_fit, filename=zcomp_fn)
                          except Exception as e: print(f"ERROR plotting Zernike fit: {e}")
                          try: fitter.plot_fit_residuals(X_stack, Y_stack, Z_stack, title=title_res, filename=zres_fn)
                          except Exception as e: print(f"ERROR plotting Zernike residuals: {e}")
                     if plot_spectrum:
                           spec_type = spectrum_config.get('spectrum_type', 'abs')
                           zspec_fn = zernike_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_zernike_spectrum_{spec_type}.pdf" if save_plots else None
                           title_spec = f"Zernike Spectrum ({spec_type}) (O={order_num}{img_type_str}, S={segment_idx})"
                           try: fitter.plot_power_spectrum(title=title_spec, filename=zspec_fn, plot_type=spec_type)
                           except Exception as e: print(f"ERROR plotting Zernike spectrum: {e}")
            else: print(f"Segment {segment_idx}: Zernike fit failed. {fitter.message}")
        # else: (Handle other skipped cases if needed)

        return base_output, individual_gauss_fits # Return computed results

    



    def analyze_and_update(self, orders_to_process=None, image_types_to_process=['A', 'B'], plot_config=None):
        """
        Analyzes specified orders/images and updates the FITS file incrementally.
        Also saves the peak catalog if not already present.
        """
        if self.paired_orders_dict is None: print("Error: Orders not clustered."); return False
        if orders_to_process is None: orders_to_process = sorted(list(self.paired_orders_dict.keys()))

        overall_success = True
        total_segments_computed = 0
        start_time = time.time()
        print(f"\n<<<<< Starting Incremental Analysis for Orders: {orders_to_process} >>>>>")

        # --- Ensure FITS file structure exists (Primary HDU) ---
        # This only creates the primary HDU if the file is new
        if not self._ensure_fits_structure():
            print("Error: Could not initialize base FITS file structure. Aborting analysis.")
            return False

        # --- Write/Verify PEAK_CATALOG HDU (Do this ONCE) ---
        if not self._write_peak_catalog():
             print("Error or Warning: Could not write or verify PEAK_CATALOG HDU. Proceeding without it.")
             # Continue analysis, but peak retrieval later might fail

        all_gauss_fits_for_batch_update = []
        
        # --- Loop through orders and images for Zernike analysis ---
        for order_num in orders_to_process:
             if order_num not in self.paired_orders_dict: print(f"Skipping order {order_num} - not found."); continue
             for image_type in image_types_to_process:
                 img_type_str = image_type.upper()
                 img_type_int = 0 if img_type_str == 'A' else 1
                 if img_type_str not in ['A', 'B']: continue

                 peaks_full_order = self.paired_orders_dict[order_num].get(img_type_str)
                 if peaks_full_order is None or len(peaks_full_order) == 0:
                     print(f"No peaks to process for Order {order_num}, Image {img_type_str}.")
                     continue

                 print(f"\n===== Analyzing & Updating ZERNIKE for Order {order_num}, Image {img_type_str} =====")
                 peaks_sorted = peaks_full_order[np.argsort(peaks_full_order[:, 0])] # Sort by X
                 segmented_peaks_list = np.array_split(peaks_sorted, self.params['num_segments']) # List of arrays

                 results_for_zernike_batch = []
                 results_for_gauss2d_batch = []
                 for segment_idx, segment_peaks in enumerate(segmented_peaks_list):
                      if len(segment_peaks) < self.params['cluster_min_samples']:
                          print(f"--- Skipping Zernike Analysis for Segment {segment_idx} (Too few peaks: {len(segment_peaks)}) ---")
                          continue

                      # --- Compute results for the segment (Zernike fit, etc.) ---
                      zernike_segment_summary, individual_gauss_fits_for_segment = \
                          self._analyze_segment_compute_only(
                              segment_peaks, order_num, img_type_int, segment_idx, plot_config
                          )
                      if zernike_segment_summary:
                          results_for_zernike_batch.append(zernike_segment_summary)
                          # total_segments_computed only counts Zernike results
                          total_segments_computed += 1 if zernike_segment_summary['FIT_SUCCESS'] or zernike_segment_summary['NUM_PEAKS_PROC'] > 0 else 0
                      if individual_gauss_fits_for_segment:
                          self._update_gauss2d_fits_file(individual_gauss_fits_for_segment)
                          

                 # --- Update ZERNIKE HDU after processing all segments for this order/image ---
                 if results_for_zernike_batch:
                      self._update_fits_file(results_for_zernike_batch) # Appends/writes to ZERNIKE HDU
                      
                      

                 print(f"===== Finished ZERNIKE update for Order {order_num}, Image {img_type_str} =====")

        end_time = time.time()
        print(f"\n<<<<< Incremental Analysis Complete ({total_segments_computed} segments processed/attempted) Duration: {end_time - start_time:.2f} sec >>>>>")
        return overall_success
    
    
    def _write_peak_catalog(self):
        """
        Writes the PEAK_CATALOG HDU to the FITS file if it doesn't exist.
        Populates it with all clustered peaks.
        Returns True on success or if already exists, False on error.
        """
        if not self.output_fits_path or not self.output_fits_path.exists():
            print("Error: Output FITS file does not exist. Cannot write peak catalog.")
            return False
        if self.paired_orders_dict is None:
            print("Warning: No clustered orders available to create peak catalog.")
            return True # Not an error, just nothing to write
        if self.peak_catalog_dtype is None:
             print("Error: Peak catalog dtype not defined.")
             return False

        extname = 'PEAK_CATALOG'
        print(f"\nChecking/Writing {extname} HDU...")

        try:
            with fitsio.FITS(self.output_fits_path, 'rw') as fits:
                if extname in fits:
                    print(f"  {extname} HDU already exists.")
                    return True # Assume it's correct if it exists

                # --- HDU does not exist, create and populate ---
                print(f"  Creating and populating {extname} HDU...")
                peak_catalog_list = []
                num_seg = self.params['num_segments']

                for order_num, pair_info in self.paired_orders_dict.items():
                     for img_type_str, peaks_full_order in pair_info.items():
                         if peaks_full_order is None or len(peaks_full_order) == 0: continue
                         img_type_int = 0 if img_type_str == 'A' else 1
                         # Sort by X to match segmentation used in analysis
                         peaks_sorted = peaks_full_order[np.argsort(peaks_full_order[:, 0])]
                         segmented_peaks_indices = np.array_split(np.arange(len(peaks_sorted)), num_seg)

                         for seg_idx, indices_in_segment in enumerate(segmented_peaks_indices):
                              if len(indices_in_segment) == 0: continue
                              segment_peaks = peaks_sorted[indices_in_segment]

                              # Add each peak in the segment to the list
                              for peak_idx_in_segment, peak_xy in enumerate(segment_peaks):
                                   peak_x, peak_y = peak_xy # Original integer coords
                                   # Get raw flux value (handle potential out-of-bounds)
                                   raw_flux = 0.0
                                   if 0 <= peak_y < self.image_shape[0] and 0 <= peak_x < self.image_shape[1]:
                                        raw_flux = self.image_data[peak_y, peak_x]

                                   peak_catalog_list.append((
                                       order_num, img_type_int, seg_idx,
                                       peak_x, peak_y, float(raw_flux) # Ensure float for flux
                                       # Add other optional info here if dtype includes it
                                   ))

                if not peak_catalog_list:
                     print("  No peaks found to write to catalog.")
                     # Optionally write an empty HDU? Or just return True? Let's return True.
                     return True

                # Convert list of tuples to structured array
                catalog_data = np.array(peak_catalog_list, dtype=self.peak_catalog_dtype)

                # Prepare header
                catalog_hdr = fitsio.FITSHDR()
                catalog_hdr['EXTNAME'] = extname
                catalog_hdr['COMMENT'] = 'Catalog of peaks identified and segmented.'
                catalog_hdr['COMMENT'] = 'IMGTYPE: 0=A, 1=B'
                # Add relevant parameters used for finding/clustering?
                catalog_hdr['PK_MINDS'] = (self.params['peak_min_distance'], 'Peak finding min_distance')
                catalog_hdr['PK_HCUT'] = (self.params['peak_hard_cut'] if self.params['peak_hard_cut'] is not None else -1, 'Peak finding hard cut')
                catalog_hdr['CL_EPSX'] = (self.params['cluster_eps_x'], 'Clustering eps_x (norm)')
                catalog_hdr['CL_EPSY'] = (self.params['cluster_eps_y'], 'Clustering eps_y (norm)')
                catalog_hdr['CL_MINS'] = (self.params['cluster_min_samples'], 'Clustering min_samples')
                catalog_hdr['N_SEG'] = (self.params['num_segments'], 'Number of segments used')


                # Write the new HDU using write_table
                # No var-len issues here, so write_table is fine
                fits.write_table(catalog_data, header=catalog_hdr, extname=extname)
                print(f"  Successfully wrote {len(catalog_data)} peaks to {extname} HDU.")
                return True

        except Exception as e:
            print(f"Error writing {extname} HDU: {e}")
            traceback.print_exc()
            return False

    # --- Reading Methods --- (Keep as before)
    def read_results_table(self, fits_path=None):
        """Reads the ZERNIKE table from the output FITS file."""
        path = Path(fits_path) if fits_path else self.output_fits_path
        extname = 'ZERNIKE'
        if not path or not path.is_file(): print(f"Error: Results FITS file not found at {path}"); return None
        try:
            with fitsio.FITS(path, 'r') as fits:
                 if extname in fits:
                      data = fits[extname].read()
                      print(f"Read {extname} table with {len(data)} rows from {path}.")
                      return data
                 else: print(f"Error: HDU '{extname}' not found in {path}"); return None
        except Exception as e: print(f"Error reading FITS file {path}: {e}"); return None



    def get_segment_data(self, order_num, img_type_int, segment_idx, fits_path=None):
        """Reads the stacked X, Y, Z data for a specific segment from the FITS file."""
        results_table = self.read_results_table(fits_path=fits_path)
        if results_table is None: return None, None, None
        mask = (results_table['ORDER_NUM'] == order_num) & (results_table['IMGTYPE'] == img_type_int) & (results_table['SEGMENT'] == segment_idx)
        match_indices = np.where(mask)[0]
        if len(match_indices) == 0: print(f"Data not found for O={order_num},T={img_type_int},S={segment_idx}"); return None, None, None
        segment_row = results_table[match_indices[0]]
        # fitsio reads var-len back as arrays
        return segment_row['X_STACK'], segment_row['Y_STACK'], segment_row['Z_STACK']


    def get_segment_coeffs(self, order_num, img_type_int, segment_idx, fits_path=None):
        """Reads the fitted coefficients and errors for a specific segment."""
        results_table = self.read_results_table(fits_path=fits_path)
        if results_table is None: return None, None
        mask = (results_table['ORDER_NUM'] == order_num) & (results_table['IMGTYPE'] == img_type_int) & (results_table['SEGMENT'] == segment_idx)
        match_indices = np.where(mask)[0]
        if len(match_indices) == 0: print(f"Coeffs not found for O={order_num},T={img_type_int},S={segment_idx}"); return None, None
        segment_row = results_table[match_indices[0]]
        if 'COEFFS' not in results_table.dtype.names: return None, None
        coeffs = segment_row['COEFFS']; errs = segment_row.get('ERR_COEFFS')
        if coeffs is None or (isinstance(coeffs, np.ndarray) and np.all(np.isnan(coeffs))): return None, None # Check for NaN array
        return coeffs, errs

    def get_fits_metadata(self, fits_path=None):
        """Reads metadata (params, indices, shape) from the FITS file headers."""
        path = Path(fits_path) if fits_path else self.output_fits_path
        extname = 'ZERNIKE'
        if not path or not path.is_file(): print(f"Error: FITS file not found at {path}"); return None

        metadata = {'params': {}, 'zernike_indices': None, 'image_shape': None}
        try:
            with fitsio.FITS(path, 'r') as fits:
                if len(fits) > 0:
                     phdr = fits[0].read_header()
                     metadata['image_shape'] = (phdr.get('IMG_H', -1), phdr.get('IMG_W', -1))
                     metadata['detector'] = phdr.get('DETECTOR','Unknown')
                     # Read back known parameters from self.params
                     for key in self.params.keys():
                          hdr_key = key.upper()[:8]
                          if hdr_key in phdr:
                               # Attempt to convert back to original type if possible
                               original_val = self.params[key]
                               saved_val = phdr[hdr_key]
                               if isinstance(original_val, bool): metadata['params'][key] = str(saved_val).lower() in ['true', 't', '1']
                               elif isinstance(original_val, int): metadata['params'][key] = int(saved_val)
                               elif isinstance(original_val, float): metadata['params'][key] = float(saved_val)
                               elif original_val is None: metadata['params'][key] = None if str(saved_val).lower() == 'none' else saved_val
                               else: metadata['params'][key] = saved_val # Keep as string otherwise
                # Read Zernike indices from the ZERNIKE HDU header
                if extname in fits:
                     rhdr = fits[extname].read_header()
                     indices_str_parts = []
                     idx = 0
                     while f'ZNIND{idx}' in rhdr: indices_str_parts.append(rhdr[f'ZNIND{idx}']); idx += 1
                     indices_str = "".join(indices_str_parts)
                     if indices_str:
                         try: metadata['zernike_indices'] = [(int(n), int(m)) for n, m in (pair.split('_') for pair in indices_str.split(','))]
                         except Exception: print("Warning: Could not parse Zernike indices from header.")

            print(f"Read metadata from {path}.")
            return metadata
        except Exception as e: 
            print(f"Error reading metadata from FITS file {path}: {e}")
            return None
        
    # --- Add New Reading Method for Peak Catalog ---
    def read_peak_catalog(self, fits_path=None, save_internally=False):
        """Reads the PEAK_CATALOG table from the output FITS file."""
        path = Path(fits_path) if fits_path else Path(self.output_fits_path)
        extname = 'PEAK_CATALOG'
        if not path or not path.is_file(): print(f"Error: FITS file not found at {path}"); return None
        try:
            with fitsio.FITS(path, 'r') as fits:
                 if extname in fits:
                      data = fits[extname].read()
                      print(f"Read {extname} table with {len(data)} rows from {path}.")
                      if save_internally: 
                          self.all_peaks_xy = np.vstack([data['PEAK_X'],data['PEAK_Y']]).T
                          print("Saved the catalog internally into .all_peaks_xy (any old value is overwritten)")
                      return data
                 else: print(f"Error: HDU '{extname}' not found in {path}"); return None
        except Exception as e: print(f"Error reading FITS file {path}: {e}"); return None

    def get_peaks_for_segment(self, order_num, img_type_int, segment_idx, fits_path=None):
        """Retrieves the (X,Y) coordinates of peaks for a specific segment from the catalog."""
        catalog = self.read_peak_catalog(fits_path=fits_path)
        if catalog is None: return None

        mask = (catalog['ORDER_NUM'] == order_num) & \
               (catalog['IMGTYPE'] == img_type_int) & \
               (catalog['SEGMENT'] == segment_idx)

        segment_peaks = catalog[mask]
        if len(segment_peaks) == 0:
             # print(f"No peaks found in catalog for O={order_num}, T={img_type_int}, S={segment_idx}")
             return np.array([], dtype=int).reshape(0, 2) # Return empty array

        return np.column_stack((segment_peaks['PEAK_X'], segment_peaks['PEAK_Y']))


    def get_stamp_data(self, peak_x, peak_y, stamp_half_width=None):
         """Extracts stamp data around a given peak coordinate."""
         if self.image_data is None: print("Error: Image data not loaded."); return None
         if stamp_half_width is None: stamp_half_width = self.params['stamp_half_width']
         img_h, img_w = self.image_shape

         x_min = max(0, peak_x - stamp_half_width)
         x_max = min(img_w, peak_x + stamp_half_width + 1)
         y_min = max(0, peak_y - stamp_half_width)
         y_max = min(img_h, peak_y + stamp_half_width + 1)

         # Check if requested stamp size is possible
         expected_size = 2 * stamp_half_width + 1
         if (x_max - x_min) != expected_size or (y_max - y_min) != expected_size:
              # warnings.warn(f"Peak ({peak_x},{peak_y}) too close to edge for full {expected_size}x{expected_size} stamp.")
              # Return None or partial stamp? Let's return None for simplicity.
              return None

         return self.image_data[y_min:y_max, x_min:x_max]