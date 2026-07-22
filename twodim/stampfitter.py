#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:06:54 2025

@author: dmilakov
"""
# harps/twodim/stampfitter.py

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood as numpyro_log_likelihood # Import log_likelihood
from numpyro.infer.initialization import init_to_uniform, init_to_median, init_to_mean,  init_to_value
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors # For MidpointNormalize
import traceback
from .plotting import MidpointNormalize, plot_stamp_fit_overview


# Try importing zernpy, essential for Zernike basis
try:
    import zernpy
    HAS_ZERNPY = True
except ImportError:
    warnings.warn("zernpy library not found. Zernike component requires 'pip install zernpy'")
    HAS_ZERNPY = False


# Configure JAX for 64-bit precision if needed (recommended for stability)
# from jax import config
# config.update("jax_enable_x64", True)

# --- Helper Functions ---

def create_stamp_grid(stamp_shape):
    """Creates relative coordinate grids for a stamp."""
    height, width = stamp_shape
    x_coords = np.arange(width) - (width - 1.0) / 2.0
    y_coords = np.arange(height) - (height - 1.0) / 2.0
    grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing='ij')
    return grid_x, grid_y

def calculate_zernike_basis_stamp(grid_x, grid_y, n_max_stamp, indices_to_use=None):
    """ Calculates Zernike basis matrix, returning selected columns and indices. """
    if not HAS_ZERNPY: raise ImportError("zernpy required.")

    max_radius = jnp.sqrt(jnp.max(grid_x**2 + grid_y**2)); max_radius = jnp.maximum(max_radius, 1e-6)
    x_norm, y_norm = grid_x / max_radius, grid_y / max_radius
    rho, theta = jnp.sqrt(x_norm**2 + y_norm**2), jnp.arctan2(y_norm, x_norm)

    all_indices = []
    for n in range(n_max_stamp + 1):
        for m in range(-n, n + 1, 2): all_indices.append((n, m))

    n_points, n_zernikes_all = grid_x.size, len(all_indices)
    full_basis_matrix_np = np.zeros((n_points, n_zernikes_all))
    valid_mask = (rho <= 1.0).flatten()
    rho_valid, theta_valid = rho.flatten()[valid_mask], theta.flatten()[valid_mask]

    if rho_valid.size > 0:
        temp_basis_valid = np.zeros((rho_valid.size, n_zernikes_all))
        rho_np, theta_np = np.array(rho_valid), np.array(theta_valid)
        for i, (n, m) in enumerate(all_indices):
            try: temp_basis_valid[:, i] = zernpy.ZernPol(m=m, n=n).polynomial_value(rho_np, theta_np)
            except Exception as e: warnings.warn(f"Zernpy error (n={n},m={m}): {e}")
        full_basis_matrix_np[valid_mask, :] = temp_basis_valid

    if indices_to_use is not None:
        selected_indices_list, selected_columns = [], []
        for i, idx in enumerate(all_indices):
            if idx in indices_to_use: selected_indices_list.append(idx); selected_columns.append(i)
        if not selected_columns: return jnp.zeros((n_points, 0)), [], all_indices
        selected_basis_matrix_jax = jnp.array(full_basis_matrix_np[:, selected_columns])
        return selected_basis_matrix_jax, selected_indices_list, all_indices
    else:
        return jnp.array(full_basis_matrix_np), all_indices, all_indices


def continuum_model_2d_poly(grid_x, grid_y, coeffs, center_x, center_y, theta):
    """ Models the continuum with a rotated 2D polynomial. """
    x_shifted, y_shifted = grid_x - center_x, grid_y - center_y
    cos_t, sin_t = jnp.cos(-theta), jnp.sin(-theta)
    x_rot, y_rot = x_shifted * cos_t + y_shifted * sin_t, -x_shifted * sin_t + y_shifted * cos_t
    val = coeffs[0] # c00
    if len(coeffs) > 1: val += coeffs[1] * x_rot   # c10
    if len(coeffs) > 2: val += coeffs[2] * y_rot   # c01
    if len(coeffs) > 3: val += coeffs[3] * x_rot**2 # c20
    if len(coeffs) > 4: val += coeffs[4] * x_rot * y_rot # c11
    if len(coeffs) > 5: val += coeffs[5] * y_rot**2 # c02
    return val.flatten()


def get_prior_params(param_name, global_x, global_y, prior_info):
    """ Placeholder: Retrieves prior parameters based on global coordinates. """
    # --- Replace with your actual spatial model lookup/evaluation ---
    default_sd = 1.0
    if 'zern_coeff' in param_name: default_sd = 0.5
    if 'amp' in param_name.lower(): default_sd = 10.0 # Needs careful scaling based on data normalization/units
    if 'cont_coeff' in param_name: default_sd = 0.1
    if 'bias' in param_name.lower(): default_sd = 5.0 # Scale for bias offset
    if 'read_noise_var' in param_name.lower(): default_sd = 5.0**2 # Variance scale
    return {'mean': 0.0, 'sd': default_sd}


class MidpointNormalize(colors.Normalize): # Keep for plotting
    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint; colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# --- Numpyro Model Definition ---
def bayesian_stamp_model(raw_data_flat, # <<< Input RAW data
                           grid_x, grid_y,
                           zernike_basis_fitted, n_zernikes_fitted, fitted_zernike_indices,
                           n_cont_coeffs, global_x, global_y,
                           # Pass prior *parameters* directly now
                           bias_log_offset_loc, bias_log_offset_scale,
                           line_log_amp_loc, line_log_amp_scale,
                           rnv_log_loc, rnv_log_scale,
                           # Pass other priors if needed, e.g., for coeffs based on global_x/y
                           prior_info=None): # Prior_info might still hold spatial models
    """ Defines the Bayesian model using Gaussian approximation for noise on RAW data. """
    if prior_info is None: prior_info = {}

    # --- Priors ---
    # Bias: Directly use passed prior hyperparameters for log(offset)
    log_bias_offset = numpyro.sample("log_bias_offset", dist.Normal(loc=bias_log_offset_loc, scale=bias_log_offset_scale))
    bias_offset = jnp.exp(log_bias_offset)
    # Bias relative to zero now, not min_stamp_value (min_stamp_value affects prior loc calc)
    bias = numpyro.sample("bias_base", dist.Normal(350, 50)) # Prior directly on base bias around expected value
    # bias = numpyro.deterministic("bias", bias_base + bias_offset) # Or just sample bias directly?
    # Let's try sampling bias directly with a sensible prior informed by data min
    # bias_loc = float(jnp.min(raw_data_flat)) + 5.0 # Slightly above min
    # bias_scale = float(jnp.sqrt(jnp.maximum(bias_loc, 1.0))) * 2.0 # Generous scale
    # bias = numpyro.sample("bias", dist.Normal(bias_loc, bias_scale))


    # LFC Line Amplitude: Use passed prior hyperparameters for log(Amplitude)
    log_line_amp = numpyro.sample("log_A_line", dist.Normal(loc=line_log_amp_loc, scale=line_log_amp_scale))
    line_amp = numpyro.deterministic("A_line", jnp.exp(log_line_amp))

    # Read Noise Variance: Use passed prior hyperparameters for log(RNV)
    log_read_noise_var = numpyro.sample("log_read_noise_var", dist.Normal(loc=rnv_log_loc, scale=rnv_log_scale))
    read_noise_var = numpyro.deterministic("read_noise_var", jnp.exp(log_read_noise_var))


    # --- Other Priors (dx, dy, Zernike, Continuum - keep non-centered, use generic defaults for now) ---
    dx_params = get_prior_params('dx', global_x, global_y, prior_info); dy_params = get_prior_params('dy', global_x, global_y, prior_info)
    loc_dx = jnp.array(dx_params.get('mean', 0.0), dtype=jnp.float32); scale_dx = jnp.maximum(jnp.array(dx_params.get('sd', 0.5), dtype=jnp.float32), 1e-6)
    loc_dy = jnp.array(dy_params.get('mean', 0.0), dtype=jnp.float32); scale_dy = jnp.maximum(jnp.array(dy_params.get('sd', 0.5), dtype=jnp.float32), 1e-6)
    dx = numpyro.sample("dx", dist.Normal(loc=loc_dx, scale=scale_dx))
    dy = numpyro.sample("dy", dist.Normal(loc=loc_dy, scale=scale_dy))

    zern_prior_means = jnp.zeros(n_zernikes_fitted, dtype=jnp.float32)
    zern_prior_sds = jnp.maximum(jnp.full(n_zernikes_fitted, 0.2, dtype=jnp.float32), 1e-6) # Prior on STD NORMAL offset -> Zernike coeffs will have units of Line Amp
    with numpyro.plate("zernike_coeffs", n_zernikes_fitted):
        z_offset = numpyro.sample("z_offset", dist.Normal(loc=0.0, scale=1.0))
    # Scale offsets by amplitude? No, Zernikes modify shape, overall scale is A_line. Coeffs are relative.
    # Need to think about scale of Zernike coefficients relative to A_line.
    # Let's assume the coeffs are roughly order unity relative to peak A_line for now.
    # This means prior_sd for z_offset should be appropriate.
    zern_coeffs_array = zern_prior_means + z_offset * zern_prior_sds # These are unitless factors?
    zern_coeffs_final = numpyro.deterministic("zernike_coeffs_final", zern_coeffs_array)

    # Continuum needs scaling too - relative to what? Let's keep ratio for now
    ratio_params = get_prior_params('cont_ratio', global_x, global_y, prior_info)
    alpha = jnp.maximum(jnp.array(ratio_params.get('alpha', 1.0), dtype=jnp.float32), 1e-6)
    beta_val = jnp.maximum(jnp.array(ratio_params.get('beta', 15.7), dtype=jnp.float32), 1e-6) # Prior mean ratio ~ 1/(1+15.7) ~ 6%
    cont_ratio = numpyro.sample("cont_ratio", dist.Beta(concentration1=alpha, concentration0=beta_val))
    cont_amp = numpyro.deterministic("A_cont", line_amp * cont_ratio) # Continuum amp linked to line amp

    cont_theta_params = get_prior_params('cont_theta', global_x, global_y, prior_info)
    loc_theta = jnp.array(cont_theta_params.get('mean', 0.0), dtype=jnp.float32); scale_theta = jnp.maximum(jnp.array(cont_theta_params.get('sd', np.pi/8), dtype=jnp.float32), 1e-6)
    cont_theta = numpyro.sample("cont_theta", dist.Normal(loc=loc_theta, scale=scale_theta))
    cont_cx, cont_cy = 0.0, 0.0
    cont_prior_means = jnp.zeros(n_cont_coeffs, dtype=jnp.float32)
    cont_prior_sds = jnp.maximum(jnp.full(n_cont_coeffs, 0.1, dtype=jnp.float32), 1e-6) # Coeffs define shape variation
    with numpyro.plate("cont_coeffs", n_cont_coeffs): cont_offset = numpyro.sample("cont_offset", dist.Normal(loc=0.0, scale=1.0))
    cont_coeffs_array = cont_prior_means + cont_offset * cont_prior_sds
    cont_coeffs_final = numpyro.deterministic("cont_coeffs_final", cont_coeffs_array)

    # --- Model Calculations ---
    # Zernike basis defines shape, amplitude A_line scales it
    line_profile = jnp.dot(zernike_basis_fitted, zern_coeffs_final) if n_zernikes_fitted > 0 else jnp.zeros_like(raw_data_flat)
    # Continuum model defines shape, amplitude A_cont scales it
    continuum_profile = continuum_model_2d_poly(grid_x, grid_y, cont_coeffs_final, cont_cx, cont_cy, cont_theta)
    # Combine components
    mu = bias + line_amp * line_profile + cont_amp * continuum_profile
    safe_mu = jnp.maximum(mu, 1e-6) # Ensure mean signal >= 0 for variance calc

    # --- Likelihood (Gaussian Approximation on RAW counts) ---
    variance_eff = read_noise_var + safe_mu # Variance = RNV + Signal
    sigma_eff = jnp.sqrt(variance_eff)
    safe_sigma_eff = jnp.maximum(sigma_eff, 1e-6) # Ensure std dev > 0

    numpyro.sample("obs", dist.Normal(loc=mu, scale=safe_sigma_eff), obs=raw_data_flat) # Observe RAW data
    
# --- Main Fitter Class ---

class BayesianStampFitter:
    """ Fits a single echelle stamp using a Bayesian model with numpyro. """

    def __init__(self, n_max_zern_stamp=4, n_poly_cont=6,
                 zernike_indices_to_fit=None):
        # ... (Initialization logic remains the same as previous version) ...
        self.n_max_zern_stamp = n_max_zern_stamp if HAS_ZERNPY else -1
        self.n_poly_cont = n_poly_cont
        self.all_zernike_indices_stamp = []
        if HAS_ZERNPY:
            for n in range(self.n_max_zern_stamp + 1):
                for m in range(-n, n + 1, 2): self.all_zernike_indices_stamp.append((n, m))
        if zernike_indices_to_fit is None and HAS_ZERNPY:
            self.zernike_indices_to_fit = [idx for idx in self.all_zernike_indices_stamp if idx != (0, 0)]
            if not self.all_zernike_indices_stamp: self.zernike_indices_to_fit = []
        elif not HAS_ZERNPY: self.zernike_indices_to_fit = []
        else:
             self.zernike_indices_to_fit = [idx for idx in zernike_indices_to_fit if idx in self.all_zernike_indices_stamp]
             if len(self.zernike_indices_to_fit) != len(zernike_indices_to_fit): warnings.warn("Some requested Zernike indices are invalid.")
        self.n_zernikes_fitted = len(self.zernike_indices_to_fit)
        print(f"Stamp fitter initialized. Fitting {self.n_zernikes_fitted} Zernike terms: {self.zernike_indices_to_fit}")
        self.stamp_shape = None; self.grid_x, self.grid_y = None, None
        self.zernike_basis_fitted = None; self.mcmc_result = None
        self.posterior_samples = None; self.prior_info = None
        self.fit_stats = dict(red_chi2=np.nan, aic=np.nan, aicc=np.nan, bic=np.nan,
                              loglik=np.nan, k=0, n=0)


    def _prepare_for_fit(self, stamp_data_shape, prior_info=None):
        """ Pre-calculates grid and the *selected* Zernike basis. """
        # ... (Logic remains the same as previous version) ...
        if self.stamp_shape == stamp_data_shape and self.zernike_basis_fitted is not None:
            self.prior_info = prior_info if prior_info is not None else {}
            return
        print(f"Preparing grid and basis for stamp shape: {stamp_data_shape}")
        self.stamp_shape = stamp_data_shape
        self.grid_x, self.grid_y = create_stamp_grid(self.stamp_shape)
        self.prior_info = prior_info if prior_info is not None else {}
        if self.n_zernikes_fitted > 0:
            try:
                selected_basis, selected_idx, _ = calculate_zernike_basis_stamp(
                    self.grid_x, self.grid_y, self.n_max_zern_stamp,
                    indices_to_use=self.zernike_indices_to_fit )
                if len(selected_idx) != self.n_zernikes_fitted or set(selected_idx) != set(self.zernike_indices_to_fit):
                     warnings.warn("Mismatch between requested and returned Zernike indices/basis columns.")
                     self.zernike_indices_to_fit = selected_idx # Update to actual fitted indices
                     self.n_zernikes_fitted = len(selected_idx)
                self.zernike_basis_fitted = selected_basis
                print(f"Calculated Zernike basis with {self.n_zernikes_fitted} selected terms.")
            except Exception as e: print(f"Error calculating Zernike basis: {e}"); self.zernike_basis_fitted = None
        else: self.zernike_basis_fitted = None


    def fit(self, stamp_data, global_x, global_y, prior_info=None,
            num_warmup=500, num_samples=1000, seed=0,
            # *** Set defaults directly to {} ***
            kernel_kwargs={}, mcmc_kwargs={},
            init_strategy='median'):
        """ Performs MCMC fitting using the numpyro model on RAW data with data-informed priors. """
        # No longer need these checks due to default args:
        # if kernel_kwargs is None: kernel_kwargs = {}
        # if mcmc_kwargs is None: mcmc_kwargs = {}

        if stamp_data.shape[0] != stamp_data.shape[1] or stamp_data.ndim != 2:
             raise ValueError("Stamp data must be 2D square.")

        # --- Prepare Data and Calculate Data-Informed Prior Hyperparameters ---
        raw_data_flat = stamp_data.flatten()
        min_stamp_val = float(np.min(raw_data_flat))
        max_stamp_val = float(np.max(raw_data_flat))
        approx_peak_height = max(1.0, max_stamp_val - min_stamp_val)
        print(f"  Stamp min={min_stamp_val:.1f}, max={max_stamp_val:.1f}, approx peak={approx_peak_height:.1f}")

        safe_min_val = max(min_stamp_val, 1.0)
        bias_log_offset_loc = 0.5 * np.log(safe_min_val)
        bias_log_offset_scale = 1.5

        line_log_amp_loc = np.log(approx_peak_height)
        line_log_amp_scale = 1.5

        # Use prior_info dict passed in, default expected RNV is 25 if not found
        if prior_info is None: prior_info = {} # Ensure prior_info is a dict
        rnv_log_loc = np.log(max(prior_info.get('expected_rnv', 25.0), 1.0))
        rnv_log_scale = 1.0

        print(f"  Prior Hyperparams (Log-Space):")
        print(f"    log_bias_offset: loc={bias_log_offset_loc:.2f}, scale={bias_log_offset_scale:.2f}")
        print(f"    log_A_line:      loc={line_log_amp_loc:.2f}, scale={line_log_amp_scale:.2f}")
        print(f"    log_RNV:         loc={rnv_log_loc:.2f}, scale={rnv_log_scale:.2f}")

        # Prepare grid/basis
        self._prepare_for_fit(stamp_data.shape, prior_info) # Pass prior_info here too
        if self.n_zernikes_fitted > 0 and self.zernike_basis_fitted is None:
            print("Error: Zernike basis failed preparation."); return None
        if self.n_zernikes_fitted > 0 and self.zernike_basis_fitted.shape[1] != self.n_zernikes_fitted:
            print("Error: Basis shape mismatch."); return None

        # --- Initialize Kernel & MCMC ---
        # Select Initialization Function Explicitly
        if init_strategy == 'uniform': init_func = init_to_uniform
        elif init_strategy == 'median': init_func = init_to_median
        elif init_strategy == 'mean': init_func = init_to_mean
        # elif init_strategy == 'prior': init_func = init_to_prior
        else:
            warnings.warn(f"Invalid init_strategy '{init_strategy}'. Using default 'uniform'.")
            init_strategy = 'uniform'
            init_func = init_to_uniform

        # Pass init_func and **kernel_kwargs (now guaranteed to be a dict)
        kernel = NUTS(bayesian_stamp_model,
                      init_strategy=init_func,
                      **kernel_kwargs)

        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    progress_bar=True,
                    **mcmc_kwargs) # Unpack mcmc_kwargs (also guaranteed dict)

        rng_key = jax.random.PRNGKey(seed)
        print(f"Starting MCMC (Warmup={num_warmup}, Samples={num_samples}, Init={init_strategy})...")

        # --- Reset results attributes ---
        self.mcmc_result = None
        self.posterior_samples = None
        self.fit_stats = dict(red_chi2=np.nan, aic=np.nan, aicc=np.nan, bic=np.nan,
                              loglik=np.nan, k=0, n=0) # Initialize correctly

        try:
            # Prepare ALL arguments needed by the model
            model_args = {
                 "raw_data_flat": jnp.array(raw_data_flat), # Pass RAW data
                 "grid_x": jnp.array(self.grid_x), "grid_y": jnp.array(self.grid_y),
                 "zernike_basis_fitted": self.zernike_basis_fitted if self.n_zernikes_fitted > 0 else jnp.zeros((stamp_data.size, 0)),
                 "n_zernikes_fitted": self.n_zernikes_fitted,
                 "fitted_zernike_indices": self.zernike_indices_to_fit,
                 "n_cont_coeffs": self.n_poly_cont,
                 "global_x": float(global_x), "global_y": float(global_y),
                 # Pass computed prior hyperparameters
                 "bias_log_offset_loc": bias_log_offset_loc, "bias_log_offset_scale": bias_log_offset_scale,
                 "line_log_amp_loc": line_log_amp_loc, "line_log_amp_scale": line_log_amp_scale,
                 "rnv_log_loc": rnv_log_loc, "rnv_log_scale": rnv_log_scale,
                 "prior_info": self.prior_info # Pass the potentially updated prior_info
            }
            # Run MCMC, passing all arguments the model needs
            mcmc.run(rng_key, **model_args)

            print("MCMC finished.")
            self.mcmc_result = mcmc
            self.posterior_samples = mcmc.get_samples()
            self.calculate_fit_statistics(raw_data_flat) # Pass RAW data for stats
            return mcmc
        except Exception as e:
            print(f"Error during MCMC run: {e}"); traceback.print_exc(); return None

    def calculate_fit_statistics(self, data_flat):
        """Calculates reduced chi2, AIC, AICc, BIC based on posterior."""
        if self.posterior_samples is None:
            print("No posterior samples available for statistics.")
            self.fit_stats = {'red_chi2': np.nan, 'aic': np.nan, 'aicc': np.nan, 'bic': np.nan, 'loglik': np.nan, 'k': 0, 'n': len(data_flat)}
            return

        data_flat_jnp = jnp.array(data_flat)
        n_data = len(data_flat_jnp)
        self.fit_stats['n'] = n_data

        # --- Count Parameters (k) ---
        k = 0
        if self.posterior_samples:
             # Count based on sampled variables (sites with _offset or specific names)
             # This count needs to be accurate based on the *sampled* sites in the model
             sampled_vars = list(self.posterior_samples.keys())
             k = len(sampled_vars)
             # Adjust for plates IF the offset names were used within plates
             # This count might be slightly off if deterministic sites are included in len(),
             # A more robust way is to count non-deterministic sites in the trace, but this is complex.
             # Let's manually count based on known sampled variables:
             k_manual = 0
             sampled_site_names = ["log_bias_offset", "log_A_line", "dx", "dy",
                                  "z_offset", "cont_ratio", "cont_theta",
                                  "cont_offset", "log_read_noise_var"]
             for key in sampled_site_names:
                  if key in self.posterior_samples:
                       # Add 1 for scalar, or size for plate
                       sample_shape = self.posterior_samples[key].shape
                       if len(sample_shape) > 1: # Plate dimension
                            k_manual += sample_shape[1] # Add number of elements in plate
                       else:
                            k_manual += 1
             k = k_manual # Use manually counted k

        self.fit_stats['k'] = k
        if k <= 0: print("Warning: Parameter count (k) is zero."); return # Handle k=0

        print(f"Calculating fit statistics (n={n_data}, k={k})...")

        # --- Log Likelihood ---
        loglik = np.nan
        try:
            # Re-calculate prior hyperparameters needed by model signature
            min_stamp_val_for_loglik = float(np.min(data_flat))
            approx_peak_height_loglik = max(1.0, float(np.max(data_flat)) - min_stamp_val_for_loglik)
            safe_min_val_ll = max(min_stamp_val_for_loglik, 1.0)
            bias_log_offset_loc_ll = 0.5 * np.log(safe_min_val_ll)
            line_log_amp_loc_ll = np.log(approx_peak_height_loglik)
            rnv_log_loc_ll = np.log(max(self.prior_info.get('expected_rnv', 25.0), 1.0))

            model_args_for_loglik = {
                "grid_x": jnp.array(self.grid_x), "grid_y": jnp.array(self.grid_y),
                "zernike_basis_fitted": self.zernike_basis_fitted if self.n_zernikes_fitted > 0 else jnp.zeros((n_data, 0)),
                "n_zernikes_fitted": self.n_zernikes_fitted, "fitted_zernike_indices": self.zernike_indices_to_fit,
                "n_cont_coeffs": self.n_poly_cont, "global_x": float(self.prior_info.get('global_x', 0.0)),
                "global_y": float(self.prior_info.get('global_y', 0.0)), "prior_info": self.prior_info,
                # Pass hyperparameters used in model's priors
                "bias_log_offset_loc": bias_log_offset_loc_ll, "bias_log_offset_scale": 1.5, # Match scale used in fit
                "line_log_amp_loc": line_log_amp_loc_ll, "line_log_amp_scale": 1.5, # Match scale used in fit
                "rnv_log_loc": rnv_log_loc_ll, "rnv_log_scale": 1.0, # Match scale used in fit
                # *** REMOVED "min_stamp_value" from args ***
            }
            # log_likelihood needs model, samples, model args (incl data for obs site)
            log_lik_samples_dict = numpyro.infer.log_likelihood(
                model=bayesian_stamp_model,
                posterior_samples=self.posterior_samples,
                **model_args_for_loglik,
                raw_data_flat=data_flat_jnp # Match obs site name from model!
            )

            if 'obs' not in log_lik_samples_dict: raise ValueError("Log likelihood missing 'obs' site.")
            log_lik_per_sample = jnp.sum(log_lik_samples_dict['obs'], axis=-1)
            mean_log_lik = jnp.mean(log_lik_per_sample); loglik = float(mean_log_lik)
            self.fit_stats['loglik'] = loglik; print(f"  Mean Log Likelihood: {loglik:.4f}")
        except Exception as e: print(f"  Error calculating log likelihood: {e}"); traceback.print_exc(); loglik = np.nan; self.fit_stats['loglik'] = loglik

        # --- AIC, AICc, BIC ---
        if np.isfinite(loglik):
             # ... (AIC/AICc/BIC calculations same as before) ...
             self.fit_stats['aic'] = -2 * loglik + 2 * k
             if (n_data - k - 1) > 0: self.fit_stats['aicc'] = self.fit_stats['aic'] + (2*k*(k+1)) / (n_data - k - 1)
             else: self.fit_stats['aicc'] = np.inf
             if n_data > 0: self.fit_stats['bic'] = -2 * loglik + k * np.log(n_data)
             else: self.fit_stats['bic'] = np.nan
             print(f"  AIC={self.fit_stats['aic']:.2f}, AICc={self.fit_stats['aicc']:.2f}, BIC={self.fit_stats['bic']:.2f}")
        else: print("  Cannot calculate AIC/AICc/BIC."); self.fit_stats.update({'aic': np.nan, 'aicc': np.nan, 'bic': np.nan})

        # --- Reduced Chi-squared ---
        # --- Reduced Chi-squared ---
        red_chi2 = np.nan
        try:
            samples = self.posterior_samples
            # *** Check required SAMPLED keys exist before accessing ***
            required_sampled_keys = ['log_bias_offset', 'log_A_line', 'cont_ratio',
                                     'cont_theta', 'log_read_noise_var']
            if self.n_zernikes_fitted > 0: required_sampled_keys.append('z_offset')
            if self.n_poly_cont > 0: required_sampled_keys.append('cont_offset')
            if not all(key in samples for key in required_sampled_keys):
                 missing = [key for key in required_sampled_keys if key not in samples]
                 raise KeyError(f"Missing required sampled keys: {missing}")

            # *** Manually compute deterministic values from SAMPLES ***
            min_val_jnp = jnp.array(np.min(data_flat_jnp), dtype=jnp.float32)
            bias_s = min_val_jnp + jnp.exp(samples['log_bias_offset']) # Calculate bias_s
            line_amp_s = jnp.exp(samples['log_A_line'])             # Calculate A_line_s
            read_noise_var_s = jnp.exp(samples['log_read_noise_var']) # Calculate RNV_s
            cont_ratio_s = samples['cont_ratio']
            cont_amp_s = line_amp_s * cont_ratio_s                   # Calculate A_cont_s
            cont_theta_s = samples['cont_theta']                     # Get theta_s

            # Reconstruct final coefficients from offsets
            zern_prior_means = jnp.zeros(self.n_zernikes_fitted, dtype=jnp.float32); zern_prior_sds = jnp.maximum(jnp.full(self.n_zernikes_fitted, 0.2, dtype=jnp.float32), 1e-6)
            zern_coeffs_s = zern_prior_means[None,:] + samples['z_offset'] * zern_prior_sds[None,:] if 'z_offset' in samples else None

            cont_prior_means = jnp.zeros(self.n_poly_cont, dtype=jnp.float32); cont_prior_sds = jnp.maximum(jnp.full(self.n_poly_cont, 0.1, dtype=jnp.float32), 1e-6)
            cont_coeffs_s = cont_prior_means[None,:] + samples['cont_offset'] * cont_prior_sds[None,:] if 'cont_offset' in samples else None

            # --- Calculate mu_mean (vectorized) ---
            num_samples_total = bias_s.shape[0] # Now bias_s is defined
            if zern_coeffs_s is not None and self.n_zernikes_fitted > 0: term_line = line_amp_s[:, None] * jnp.einsum('ik,jk->ji', self.zernike_basis_fitted, zern_coeffs_s)
            else: term_line = jnp.zeros((num_samples_total, n_data))
            term_cont_samples = []
            for i in range(num_samples_total):
                 coeffs_c_i = cont_coeffs_s[i] if cont_coeffs_s is not None else np.zeros(self.n_poly_cont); theta_c_i = cont_theta_s[i]
                 cont_prof_i = continuum_model_2d_poly(self.grid_x, self.grid_y, coeffs_c_i, 0.0, 0.0, theta_c_i)
                 term_cont_samples.append(cont_amp_s[i] * cont_prof_i)
            term_cont = jnp.stack(term_cont_samples)
            mu_posterior = bias_s[:, None] + term_line + term_cont
            mu_mean = jnp.mean(mu_posterior, axis=0)

            # Calculate Effective Sigma using read_noise_var
            read_noise_var_mean = jnp.mean(read_noise_var_s)
            variance_eff = read_noise_var_mean + jnp.maximum(mu_mean, 1e-6)
            sigma_eff_mean = jnp.sqrt(variance_eff); sigma_eff_mean = jnp.maximum(sigma_eff_mean, 1e-6)

            # Calculate Chi2
            residuals = data_flat_jnp - mu_mean
            chi2 = jnp.sum((residuals / sigma_eff_mean)**2)
            dof = max(1, n_data - k)
            red_chi2 = float(chi2 / dof)
            self.fit_stats['red_chi2'] = red_chi2
            print(f"  Reduced Chi2: {self.fit_stats['red_chi2']:.4f} (DOF={dof})")

        except KeyError as e: print(f"  Error calculating reduced chi-squared: Missing key {e}."); red_chi2 = np.nan
        except Exception as e: print(f"  Error calculating reduced chi-squared: {e}"); traceback.print_exc(); red_chi2 = np.nan
        self.fit_stats['red_chi2'] = red_chi2


    def plot_residuals(self, stamp_data, filename=None):
        """ Plots the raw mean residuals and normalized mean residuals side-by-side. """
        if self.posterior_samples is None: print("No posterior samples available for residual plot."); return
        if stamp_data.shape != self.stamp_shape: print("Error: Provided stamp_data shape differs."); return

        print("Generating residual plots (raw and normalized)...")
        n_data = stamp_data.size
        data_flat_jnp = jnp.array(stamp_data.flatten()) # Use JAX array

        try:
            samples = self.posterior_samples
            # *** Check required SAMPLED keys exist before accessing ***
            required_sampled_keys = ['log_bias_offset', 'log_A_line', 'cont_ratio',
                                     'cont_theta', 'log_read_noise_var']
            if self.n_zernikes_fitted > 0: required_sampled_keys.append('z_offset')
            if self.n_poly_cont > 0: required_sampled_keys.append('cont_offset')
            if not all(key in samples for key in required_sampled_keys):
                 missing = [key for key in required_sampled_keys if key not in samples]
                 raise KeyError(f"Missing required sampled keys in posterior samples: {missing}")

            # --- Manually compute deterministic values from samples ---
            # *** CORRECTED: Use data_flat_jnp which is defined in this scope ***
            min_val_jnp = jnp.array(np.min(data_flat_jnp), dtype=jnp.float32)
            bias_s = min_val_jnp + jnp.exp(samples['log_bias_offset'])
            line_amp_s = jnp.exp(samples['log_A_line'])
            read_noise_var_s = jnp.exp(samples['log_read_noise_var'])
            cont_ratio_s = samples['cont_ratio']
            cont_amp_s = line_amp_s * cont_ratio_s # Calculate A_cont
            cont_theta_s = samples['cont_theta']

            zern_prior_means = jnp.zeros(self.n_zernikes_fitted, dtype=jnp.float32)
            zern_prior_sds = jnp.maximum(jnp.full(self.n_zernikes_fitted, 0.2, dtype=jnp.float32), 1e-6)
            zern_coeffs_s = zern_prior_means[None,:] + samples['z_offset'] * zern_prior_sds[None,:] if 'z_offset' in samples and self.n_zernikes_fitted > 0 else None

            cont_prior_means = jnp.zeros(self.n_poly_cont, dtype=jnp.float32)
            cont_prior_sds = jnp.maximum(jnp.full(self.n_poly_cont, 0.1, dtype=jnp.float32), 1e-6)
            cont_coeffs_s = cont_prior_means[None,:] + samples['cont_offset'] * cont_prior_sds[None,:] if 'cont_offset' in samples and self.n_poly_cont > 0 else None


            # --- Calculate mu_mean ---
            num_samples_total = bias_s.shape[0]
            if zern_coeffs_s is not None and self.n_zernikes_fitted > 0:
                line_prof_samples = jnp.einsum('ik,jk->ji', self.zernike_basis_fitted, zern_coeffs_s)
                term_line = line_amp_s[:, None] * line_prof_samples
            else: term_line = jnp.zeros((num_samples_total, n_data))
            term_cont_samples = []
            for i in range(num_samples_total):
                 coeffs_c_i = cont_coeffs_s[i] if cont_coeffs_s is not None else np.zeros(self.n_poly_cont)
                 theta_c_i = cont_theta_s[i]; cont_prof_i = continuum_model_2d_poly(self.grid_x, self.grid_y, coeffs_c_i, 0.0, 0.0, theta_c_i)
                 term_cont_samples.append(cont_amp_s[i] * cont_prof_i)
            term_cont = jnp.stack(term_cont_samples)
            mu_posterior = bias_s[:, None] + term_line + term_cont
            mu_mean = jnp.mean(mu_posterior, axis=0)

            # Calculate Residuals and Normalization Sigma
            mean_residuals = data_flat_jnp - mu_mean
            read_noise_var_mean = jnp.mean(read_noise_var_s)
            variance_eff = read_noise_var_mean + jnp.maximum(mu_mean, 1e-6)
            sigma_eff_mean = jnp.sqrt(variance_eff); sigma_eff_mean = jnp.maximum(sigma_eff_mean, 1e-6)
            norm_mean_residuals = mean_residuals / sigma_eff_mean

            mean_residuals_2d = np.array(mean_residuals.reshape(self.stamp_shape))
            norm_mean_residuals_2d = np.array(norm_mean_residuals.reshape(self.stamp_shape))

            # Get context info
            xo = float(np.mean(samples.get('dx', 0.0)))
            yo = float(np.mean(samples.get('dy', 0.0)))
            red_chi2 = self.fit_stats.get('red_chi2', np.nan)

        except KeyError as e: print(f"Error calculating residuals for plot: Missing key {e}."); return
        except Exception as e: print(f"Error calculating residuals for plot: {e}"); traceback.print_exc(); return

        # --- Create Residual Plot ---
        # (Plotting code with ax1, ax2, imshows, colorbars, etc. remains the same)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
        fig.suptitle("Fit Residuals (Relative Coords)", fontsize=14)
        # Plot 1: Raw Mean Residuals
        ax1 = axes[0]; max_abs_raw_res = np.nanmax(np.abs(mean_residuals_2d)); clim_raw = max(max_abs_raw_res, 1e-9)
        norm_raw = MidpointNormalize(vmin=-clim_raw, vmax=clim_raw, midpoint=0)
        extent = [-0.5 + self.grid_x.min(), 0.5 + self.grid_x.max(), -0.5 + self.grid_y.min(), 0.5 + self.grid_y.max()]
        im1 = ax1.imshow(mean_residuals_2d, cmap='coolwarm', origin='lower', extent=extent, norm=norm_raw)
        fig.colorbar(im1, ax=ax1, shrink=0.7, label="Mean Residual (Data Units)")
        ax1.plot(xo, yo, 'x', color='black', markersize=8, alpha=0.7)
        ax1.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}", transform=ax1.transAxes, color='black', backgroundcolor='white', fontsize=8, ha='right', va='top', alpha=0.8)
        ax1.set_title("Raw Mean Residuals"); ax1.set_xlabel("Relative X (pixels)"); ax1.set_ylabel("Relative Y (pixels)")
        ax1.set_aspect('equal', adjustable='box')
        # Plot 2: Normalized Mean Residuals
        ax2 = axes[1]; max_abs_norm_res = np.nanmax(np.abs(norm_mean_residuals_2d)); clim_norm = min(max(max_abs_norm_res, 1.0), 5.0)
        norm_norm = MidpointNormalize(vmin=-clim_norm, vmax=clim_norm, midpoint=0)
        im2 = ax2.imshow(norm_mean_residuals_2d, cmap='coolwarm', origin='lower', extent=extent, norm=norm_norm)
        fig.colorbar(im2, ax=ax2, shrink=0.7, label="Mean Normalized Residual ($\sigma$)")
        ax2.plot(xo, yo, 'x', color='black', markersize=8, alpha=0.7)
        ax2.text(0.97, 0.97, r"$\chi^2_\nu$=" + f"{red_chi2:.2f}", transform=ax2.transAxes, color='black', backgroundcolor='white', fontsize=8, ha='right', va='top', alpha=0.8)
        ax2.set_title("Normalized Mean Residuals"); ax2.set_xlabel("Relative X (pixels)")
        ax2.set_aspect('equal', adjustable='box')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save or Show
        if filename: # ... (save/close logic) ...
            try: plt.savefig(filename, dpi=150, bbox_inches='tight')
            except Exception as e: print(f"ERROR saving residual plot: {e}")
            finally: plt.close(fig)
        else: plt.show()
        
    def plot_fit_overview(self, stamp_data, filename=None, title_prefix=""):
        """ Generates and saves/shows the 4-panel stamp fit overview plot. """
        if self.posterior_samples is None: print("No posterior samples for overview plot."); return
        if stamp_data.shape != self.stamp_shape: print("Error: Data shape mismatch."); return
        if self.grid_x is None or self.grid_y is None: print("Error: Grid not prepared."); return

        print("Generating fit overview plot...")
        n_data = stamp_data.size
        data_flat_jnp = jnp.array(stamp_data.flatten())

        try:
            samples = self.posterior_samples
            # *** Manually compute deterministic values from SAMPLES ***
            required_sampled_keys = ['log_bias_offset', 'log_A_line', 'cont_ratio', 'cont_theta', 'log_read_noise_var']
            # ... (Add checks for z_offset, cont_offset if needed) ...
            if not all(key in samples for key in required_sampled_keys):
                 missing = [key for key in required_sampled_keys if key not in samples]
                 raise KeyError(f"Missing required sampled keys: {missing}")

            min_val_jnp = jnp.array(np.min(data_flat_jnp), dtype=jnp.float32); bias_s = min_val_jnp + jnp.exp(samples['log_bias_offset'])
            line_amp_s = jnp.exp(samples['log_A_line']); read_noise_var_s = jnp.exp(samples['log_read_noise_var'])
            cont_ratio_s = samples['cont_ratio']; cont_amp_s = line_amp_s * cont_ratio_s; cont_theta_s = samples['cont_theta']
            zern_prior_means = jnp.zeros(self.n_zernikes_fitted, dtype=jnp.float32); zern_prior_sds = jnp.maximum(jnp.full(self.n_zernikes_fitted, 0.2, dtype=jnp.float32), 1e-6)
            zern_coeffs_s = zern_prior_means[None,:] + samples['z_offset'] * zern_prior_sds[None,:] if 'z_offset' in samples and self.n_zernikes_fitted > 0 else None
            cont_prior_means = jnp.zeros(self.n_poly_cont, dtype=jnp.float32); cont_prior_sds = jnp.maximum(jnp.full(self.n_poly_cont, 0.1, dtype=jnp.float32), 1e-6)
            cont_coeffs_s = cont_prior_means[None,:] + samples['cont_offset'] * cont_prior_sds[None,:] if 'cont_offset' in samples and self.n_poly_cont > 0 else None

            # Calculate mu_mean
            num_samples_total = bias_s.shape[0]
            if zern_coeffs_s is not None and self.n_zernikes_fitted > 0: term_line = line_amp_s[:, None] * jnp.einsum('ik,jk->ji', self.zernike_basis_fitted, zern_coeffs_s)
            else: term_line = jnp.zeros((num_samples_total, n_data))
            term_cont_samples = []
            for i in range(num_samples_total):
                 coeffs_c_i = cont_coeffs_s[i] if cont_coeffs_s is not None else np.zeros(self.n_poly_cont); theta_c_i = cont_theta_s[i]
                 cont_prof_i = continuum_model_2d_poly(self.grid_x, self.grid_y, coeffs_c_i, 0.0, 0.0, theta_c_i)
                 term_cont_samples.append(cont_amp_s[i] * cont_prof_i)
            term_cont = jnp.stack(term_cont_samples)
            mu_posterior = bias_s[:, None] + term_line + term_cont
            mu_mean = jnp.mean(mu_posterior, axis=0)

            # Calculate Residuals and Normalization Sigma
            mean_residuals = data_flat_jnp - mu_mean
            read_noise_var_mean = jnp.mean(read_noise_var_s)
            variance_eff = read_noise_var_mean + jnp.maximum(mu_mean, 1e-6)
            sigma_eff_mean = jnp.sqrt(variance_eff); sigma_eff_mean = jnp.maximum(sigma_eff_mean, 1e-6)
            norm_mean_residuals = mean_residuals / sigma_eff_mean

            # Reshape for plotting
            model_mean_2d = np.array(mu_mean.reshape(self.stamp_shape))
            mean_residuals_2d = np.array(mean_residuals.reshape(self.stamp_shape))
            norm_mean_residuals_2d = np.array(norm_mean_residuals.reshape(self.stamp_shape))

            # Get context info
            xo = float(np.mean(samples.get('dx', 0.0)))
            yo = float(np.mean(samples.get('dy', 0.0)))
            red_chi2 = self.fit_stats.get('red_chi2', np.nan)

        except KeyError as e: print(f"Error calculating data for plot: Missing key {e}."); return
        except Exception as e: print(f"Error calculating data for plot: {e}"); traceback.print_exc(); return

        # --- Call the plotting function from plotting module ---
        # Ensure plot_stamp_fit_overview is imported
        plot_stamp_fit_overview(
             stamp_data=stamp_data, # Pass original 2D data
             model_mean_2d=model_mean_2d,
             mean_residuals_2d=mean_residuals_2d,
             norm_mean_residuals_2d=norm_mean_residuals_2d,
             fit_stats=self.fit_stats,
             posterior_samples=self.posterior_samples, # Pass samples for dx,dy if needed
             stamp_coords=(self.grid_x, self.grid_y),
             title_prefix=title_prefix,
             filename=filename
        )


    def get_results_summary(self, prob=0.9):
        """Returns a summary dictionary including fit statistics."""
        if self.mcmc_result is None: return None
        summary_dict = {}
        try:
            summary = numpyro.diagnostics.summary(self.mcmc_result.get_samples(group_by_chain=True), prob=prob, group_by_chain=False)
            for key, stats in summary.items():
                 summary_dict[key] = {k: np.array(v) for k, v in stats.items()} # Convert to numpy
            # Add our calculated stats
            summary_dict['fit_stats'] = self.fit_stats
            return summary_dict
        except Exception as e: print(f"Error generating summary: {e}"); return {'fit_stats': self.fit_stats}
        
    def print_summary(self, prob=0.9):
        """Prints the numpyro summary table."""
        if self.mcmc_result is None:
            print("Fit not performed or failed. No summary available.")
            return
        print("\nMCMC Posterior Summary:")
        self.mcmc_result.print_summary(prob=prob)


    


    def plot_posteriors(self, params_to_plot=None, filename=None):
         """ Plot posterior distributions for selected parameters """
         if self.mcmc_result is None: print("No MCMC results to plot."); return
         try:
              import arviz as az
              # --- CORRECTED: Use filter_vars instead of var_names ---
              # Or convert all and filter the InferenceData object after creation
              az_data = az.from_numpyro(self.mcmc_result) # Convert the whole object first

              if params_to_plot:
                  # Filter the InferenceData object
                  available_vars = list(az_data.posterior.data_vars)
                  valid_params = [p for p in params_to_plot if p in available_vars]
                  if not valid_params: print(f"Warning: None of requested params found in posterior."); return
                  # Create a new InferenceData object with only the selected variables
                  # Using dataset selection syntax
                  az_data_filtered = az_data.posterior[valid_params]
              else:
                  az_data_filtered = az_data # Use all variables if none specified


              # Create the posterior plot from filtered data
              az.plot_posterior(az_data_filtered)
              fig = plt.gcf() # Get current figure

              if filename:
                  plt.savefig(filename); print(f"Posteriors plot saved to {filename}"); plt.close(fig)
              else: plt.show()
         except ImportError: print("Plotting posteriors requires 'arviz'. Install with 'pip install arviz'")
         except Exception as e: print(f"Error plotting posteriors: {e}"); traceback.print_exc()


    # Add methods for plotting data vs model, residuals, corner plots etc. as needed
    # Example:
    def plot_corner(self, params_to_plot, filename=None):
         """ Plot corner plot for selected parameters """
         if self.mcmc_result is None: print("No MCMC results for corner plot."); return
         try:
             import corner # Requires corner
             samples = self.mcmc_result.get_samples()
             samples_array = np.stack([samples[k] for k in params_to_plot if k in samples], axis=-1)
             if samples_array.ndim == 2 and samples_array.shape[1] > 0:
                 fig = corner.corner(samples_array, labels=params_to_plot, show_titles=True)
                 if filename:
                     plt.savefig(filename); print(f"Corner plot saved to {filename}"); plt.close(fig)
                 else: plt.show()
             else: print("Could not extract valid samples for corner plot.")
         except ImportError: print("Plotting corner plot requires 'corner'. Install with 'pip install corner'")
         except Exception as e: print(f"Error generating corner plot: {e}")