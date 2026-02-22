#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:00:06 2023

@author: dmilakov
"""
import numpy as np
import jax.numpy as jnp
import harps.lsf.read as hread
import harps.lsf.gp as hlsfgp
import harps.containers as container
import harps.progress_bar as progress_bar
import harps.plotter as hplt
import matplotlib.pyplot as plt
import logging
from scipy.optimize import leastsq, least_squares
import scipy.interpolate as interpolate
from scipy.interpolate import splrep, splev

from matplotlib import ticker

from typing import Tuple, Optional, Any, Callable


def residuals(pars,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y,
              x,scale,weight=False,obs=None,obs_err=None):
    amp, cen, wid, m, y0 = _unpack_pars(pars)
    splr_lsf      = interpolate.splrep(lsf_loc_x , lsf_loc_y / np.max(lsf_loc_y)) 
    model_data    = lsf_model(splr_lsf, pars, x, scale) + \
                    m * (x-cen) + y0 
                    
    
    splr_sct  = interpolate.splrep(sct_loc_x, sct_loc_y)
    model_scatter = sct_model(splr_sct, pars,x,scale)
    if not weight:
        within  = within_limits(x,cen,scale)
        weights = np.zeros_like(x)
        weights[within] = 1.
    else:
        weights  = assign_weights(x,cen,scale)
    
    
    if obs is not None:
        resid = (obs - model_data) * weights
        if obs_err is not None:
            rescaled_error = obs_err 
            # rescaled_error = err1l*model_scatter
            resid = resid / rescaled_error 
        return resid.astype(np.float64)
    else:
        return model_data
def residuals2(pars,splr_lsf,
              x,scale,weight=False,obs=None,obs_err=None):
    # amp, cen, wid, m, y0 = _unpack_pars(pars)
    cen = _unpack_pars(pars)[1]
    within  = within_limits(x,cen,scale)
    outside = ~within
    
    weights_line = np.zeros_like(x)
    weights_line[outside] = 1.
    if not weight:
        weights_lsf = np.zeros_like(x)
        weights_lsf[within] = 1.
    else:
        weights_lsf  = assign_weights(x,cen,scale)
    # splr_lsf   = interpolate.splrep(lsf_loc_x , lsf_loc_y / np.max(lsf_loc_y)) 
    model_lsf  = lsf_model(splr_lsf, pars, x, scale)# * weights_lsf
    # model_scatter = sct_model(sct_loc_x,sct_loc_y,pars,x,scale)
    # 
    
    model_data   = model_lsf #+ model_line
    if len(pars)>3:
        amp, cen, wid, m, y0 = _unpack_pars(pars)
        model_line   = (m * (x - cen) + y0)# * weights_line
        model_data = model_lsf + model_line
    # model_data   = np.vstack([model_lsf, model_line ])
    
    if obs is not None:
        resid = (obs - model_data) * weights_lsf
        # resid = np.vstack([(obs-model_lsf),
        #                    (obs-model_line)])
        if obs_err is not None:
            rescaled_error = obs_err 
            # rescaled_error = err1l*model_scatter
            resid = resid / rescaled_error 
        return resid.flatten()
    else:
        return model_data
    
def line_old(x1l,flx1l,err1l,bary,LSF1d,scale,weight=True,interpolate=True,
        output_model=False,output_rsd=False,plot=False,save_fig=None,
        rsd_range=None,npars=None,method='lmfit',bounded=False,
        *args,**kwargs):
    """
    lsf1d must be an instance of LSF class and contain only one segment 
    (see harps.lsf)
    """
    
        
    # def residuals_lmfit(pars,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y):
    #     model_data    = lsf_model(lsf_loc_x,lsf_loc_y,pars,x1l,scale)
    #     if weight:
    #         weights  = assign_weights(x1l,pars['cen'],scale)
    #     else:
    #         weights = np.ones_like(x1l)
        
        
    #     rescaled_error = err1l 
    #     resid = (flx1l - model_data) / rescaled_error * weights
    #     return resid
    
    
    
    N = 2 if interpolate else 1
    lsf_loc_x,lsf_loc_y = LSF1d.interpolate_lsf(bary,N)
    sct_loc_x,sct_loc_y = LSF1d.interpolate_scatter(bary,N)
    
    npars = npars if npars is not None else container.npars
    guess_amp = np.max(flx1l)
    guess_cen = np.average(x1l,weights=flx1l)
    p0 = _prepare_pars(npars,method,x1l,flx1l)
    if method=='scipy':
        
        if not bounded:
            popt,pcov,infodict,errmsg,ier = leastsq(residuals2,x0=p0,
                                            args=(lsf_loc_x,lsf_loc_y,
                                                  sct_loc_x,sct_loc_y,
                                                  x1l,scale,weight,
                                                  flx1l,err1l),
                                            ftol=1e-12,
                                            Dfun=jacobian_analytical,
                                            full_output=True)
        else:
            bounds = np.array([(0.8*guess_amp,1.2*guess_amp),
                      (guess_cen-0.5,guess_cen+0.5),
                      (0.9,1.1),
                      (-1e3,1e3),
                      (-1e3,1e3)])
            print(bounds)
            result = least_squares(residuals, x0=p0,
                                    bounds = bounds[:npars].T,
                                    jac=jacobian_analytical,
                                    args=(lsf_loc_x,lsf_loc_y,
                                          sct_loc_x,sct_loc_y),
                                    )
            pars = result['x']
            ier = [1]
            errmsg = ''
            print(result)
        if ier not in [1, 2, 3, 4]:
            
            print("Optimal parameters not found: " + errmsg)
            popt = np.full_like(p0,np.nan)
            pcov = None
            success = False
        else:
            success = True
        
        if success:
            # amp, cen, wid, a0, a1 = popt
            cost = np.sum(infodict['fvec']**2)
            dof  = (len(x1l) - len(popt))
            if pcov is not None:
                pcov = pcov*cost/dof
            else:
                pcov = np.diag([np.nan for i in range(len(popt))])
            #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
            rsd = infodict['fvec']
            
        else:
            popt = np.full_like(p0,np.nan)
            pcov = np.diag([np.nan for i in range(len(popt))])
            cost = np.nan
            dof  = (len(x1l) - len(popt))
            success=False
        pars    = popt
        errors  = np.sqrt(np.diag(pcov))
        # print(pars,errors)
    # chisqnu = cost/dof
    elif method=='lmfit':
        from lmfit import create_params, fit_report, minimize, Model
        # # fit_params = create_params(amp=guess_amp, cen=guess_cen, 
        # #                            wid=1., slope=0.0, offset=0)
        
        result = minimize(residuals2, p0, 
                        kws=dict(
                            lsf_loc_x=lsf_loc_x,
                            lsf_loc_y=lsf_loc_y,
                            sct_loc_x=sct_loc_x,
                            sct_loc_y=sct_loc_y,
                            x = x1l,
                            scale = scale,
                            obs = flx1l,
                            obs_err = err1l
                            ))
        # model = Model(model_profile) + Model(model_line)
        # result = model.fit(flx1l,params=p0,weights=1./err1l,
        #                    x_array=x1l,
        #                    lsf_loc_x=lsf_loc_x,
        #                    lsf_loc_y=lsf_loc_y,
        #                    scale=scale
        #                    )
        # print(fit_report(result))
        pars_obj = result.params
        pars = _unpack_pars(pars_obj)
        success = result.success
        covar = result.covar
        errors = np.sqrt(np.diag(covar))
        cost   = np.sum(result.residual**2)
        rsd    = result.residual
    # model    = lsf_model(lsf_loc_x,lsf_loc_y,pars,x1l,scale)
    model    = residuals2(pars,lsf_loc_x,lsf_loc_y,sct_loc_x,sct_loc_y,
                         x=x1l,scale=scale,weight=weight,
                         obs=None,obs_err=None)
    within   = within_limits(x1l,pars[1],scale)
    
    chisq, dof = get_chisq_dof(x1l,flx1l,err1l,model,pars,scale)
    chisqnu = chisq / dof
    # _        = np.where((rsd_norm<10)&(within))
    # print(_)
    if len(np.shape(model))>1:
        integral = np.sum(model,axis=0)[within]
    else:
        integral = np.sum(model[within])
    output_tuple = (success, pars, errors, cost, chisqnu, integral)
    # print(cost,(len(x1l) - len(popt)),cost/(len(x1l) - len(popt)),chisq,dof,chisq/dof)
    if plot:
        fig = plot_fit(x1l,flx1l,err1l,model=model,#rsd_norm=rsd_norm,
                       pars=pars,scale=scale,
                       lsf_loc_x=lsf_loc_x,lsf_loc_y=lsf_loc_y,
                       rsd_range=rsd_range,
                       **kwargs)
    if output_model:  
        output_tuple =  output_tuple + (model,)
    if output_rsd:  
        output_tuple =  output_tuple + (rsd,)
    if plot:
        output_tuple =  output_tuple + (fig,)
        
    return output_tuple

# from scipy.optimize import leastsq, least_squares
# lmfit is imported conditionally later
# Assuming helper functions (residuals2, _prepare_pars, etc.) and LSF1d class are defined as provided.

# For type hinting if not already imported in the scope of helpers
# from .LSF_handling_module import LSF1d # Example of how LSF1d might be imported
# from .fitting_helpers import ( # Example imports
#     residuals2, _prepare_pars, _unpack_pars, within_limits,
#     get_chisq_dof, plot_fit
# )

# Define constants for parameter names if they are fixed
# PARAM_NAMES = ['amp', 'cen', 'wid', 'slope', 'offset']
# SPEED_OF_LIGHT_KM_S = 299792.458 # Used in lsf_model, get_binlimits

def line(
    x1l: np.ndarray,
    flx1l: np.ndarray,
    err1l: Optional[np.ndarray],
    bary: float,
    LSF1d_obj: 'LSF1d', # Use forward reference for LSF1d if defined later or imported
    scale: str,
    npars: int = 3, # Default to 3 parameters (amp, cen, wid)
    weight: bool = True,
    interpolate: bool = True,
    output_model: bool = False,
    output_rsd: bool = False,
    plot: bool = False,
    # save_fig: Optional[str] = None, # Parameter was unused, consider if plot_fit handles it
    rsd_range: Optional[Tuple[float, float]] = None,
    method: str = 'scipy',
    bounded: bool = True,
    bounds_config: Optional[dict] = None, # More flexible bounds
    ftol_scipy: float = 1e-7,
    lmfit_options: Optional[dict] = None, # Options for lmfit.minimize
    verbose: bool = False, # Added for more feedback
    **plot_kwargs: Any # Pass extra kwargs to plot_fit
) -> Tuple:
    """
    Fits a model profile (LSF + optional linear baseline) to an observed 1D line.

    Parameters
    ----------
    x1l, flx1l, err1l : np.ndarray
        Observed data: x-coordinates, flux, and errors. err1l can be None.
    bary : float
        Barycentric velocity or similar for LSF interpolation.
    LSF1d_obj : LSF1d
        Instance of LSF class containing LSF data for the relevant segment.
    scale : str
        Coordinate scale ('pix' or 'vel').
    npars : int, optional
        Number of parameters to fit (3, 4, or 5).
        3: amp, cen, wid
        4: amp, cen, wid, offset (y0)
        5: amp, cen, wid, slope (m), offset (y0)
        Default is 3.
    weight : bool, optional
        If True, applies custom weighting to residuals. Default is True.
    interpolate : bool, optional
        If True, interpolates LSF to 2 points around bary. Default is True.
    output_model, output_rsd, plot : bool, optional
        Flags to include model, residuals, or plot figure in output.
    rsd_range : tuple, optional
        Range for plotting residuals.
    method : str, optional
        Optimization method: 'scipy' or 'lmfit'. Default is 'lmfit'.
    bounded : bool, optional
        If True and method='scipy', uses bounded least-squares. Default is False.
    bounds_config : dict, optional
        Configuration for parameter bounds if `bounded=True` and `method='scipy'`.
        E.g., {'amp_rel': 0.2, 'cen_abs': 0.5, 'wid_rel': 0.1,
               'slope_abs': 1e3, 'offset_abs': 1e3}
        Rel factors are relative to initial guess, Abs are absolute deviations.
    ftol_scipy : float, optional
        Tolerance for scipy.optimize.leastsq. Default is 1e-12.
    lmfit_options : dict, optional
        Additional options to pass to lmfit.minimize (e.g., method, fit_kws).
    verbose : bool, optional
        If True, prints more detailed information during fitting.
    **plot_kwargs : Any
        Additional keyword arguments passed to the `plot_fit` function.

    Returns
    -------
    tuple
        Contains: (success, pars_array, errors_array, cost, chisq_reduced, integral,
                   [model_array], [residuals_array], [figure_object])
        Optional elements depend on output_model, output_rsd, plot flags.
    """
    if not (3 <= npars <= 5):
        raise ValueError(f"npars must be 3, 4, or 5, got {npars}")
    if err1l is None and verbose:
        print("Warning: err1l is None. Residuals will not be properly scaled by errors.")

    # --- 1. LSF Preparation ---
    num_interp_points = 2 if interpolate else 1
    print(bary, len(LSF1d_obj))
    lsf_loc_x, lsf_loc_y = LSF1d_obj.interpolate_lsf(bary, num_interp_points)
    sct_loc_x, sct_loc_y = LSF1d_obj.interpolate_scatter(bary, num_interp_points) # sct_loc currently unused by residuals2
    
    splr_lsf  = splrep(lsf_loc_x , lsf_loc_y / np.max(lsf_loc_y)) 
    
    
    # --- 2. Initial Parameter Guessing ---
    p0_obj = _prepare_pars(npars, method, x1l, flx1l) # Returns tuple for scipy, Parameters for lmfit

    # Unpack initial guesses for bounds if needed (scipy bounded)
    # Ensure _unpack_pars can handle both scipy tuple and lmfit Parameters
    initial_guesses_tuple = _unpack_pars(p0_obj if method == 'lmfit' else tuple(val for val in p0_obj if val is not None))


    # --- 3. Fitting ---
    pars_arr: np.ndarray = np.full(npars, np.nan)
    errors_arr: np.ndarray = np.full(npars, np.nan)
    cost: float = np.nan
    success: bool = False
    residuals_fit: Optional[np.ndarray] = None # Store residuals from the fit
    covariance_matrix: Optional[np.ndarray] = None

    if method == 'scipy':
        p0_scipy_tuple = p0_obj # _prepare_pars already returns tuple for scipy

        args_for_residuals = (
            splr_lsf,
            # sct_loc_x, sct_loc_y, # sct_loc currently unused by residuals2
            x1l, scale, weight,
            flx1l, err1l
        )
        if not bounded:
            popt, pcov, infodict, errmsg, ier = leastsq(
                residuals2, x0=p0_scipy_tuple, args=args_for_residuals,
                Dfun=jacobian_analytical,
                ftol=ftol_scipy, full_output=True
            )
            success = ier in [1, 2, 3, 4]
            if not success and verbose:
                print(f"SciPy (leastsq) optimization failed: {errmsg} (ier={ier})")
            if success:
                pars_arr = np.asarray(popt)
                residuals_fit = infodict['fvec'] # These are weighted and error-scaled residuals
                cost = np.sum(residuals_fit**2)
                covariance_matrix = pcov
        else: # Bounded scipy fit
            # More flexible bounds setup
            default_bounds_config = {
                'amp_rel': 0.5, 'cen_abs': 1.0, 'wid_abs': (0.1, 5.0), # wid can be (min_abs, max_abs)
                'slope_abs': 1e3, 'offset_abs': np.inf # Effectively unbounded if Inf
            }
            current_bounds_config = {**default_bounds_config, **(bounds_config or {})}
            scipy_bounds = [[-np.inf] * npars, [np.inf] * npars] # Default to (-inf, inf)

            # Amp
            scipy_bounds[0][0] = initial_guesses_tuple[0] * (1 - current_bounds_config['amp_rel'])
            scipy_bounds[1][0] = initial_guesses_tuple[0] * (1 + current_bounds_config['amp_rel'])
            if scipy_bounds[0][0] < 0 and initial_guesses_tuple[0] >=0 : scipy_bounds[0][0] = 0 # common for amp

            # Cen
            scipy_bounds[0][1] = initial_guesses_tuple[1] - current_bounds_config['cen_abs']
            scipy_bounds[1][1] = initial_guesses_tuple[1] + current_bounds_config['cen_abs']
            
            # Wid
            if isinstance(current_bounds_config['wid_abs'], tuple):
                scipy_bounds[0][2], scipy_bounds[1][2] = current_bounds_config['wid_abs']
            else: # Assume relative if not tuple
                scipy_bounds[0][2] = initial_guesses_tuple[2] * (1 - current_bounds_config.get('wid_rel',0.5))
                scipy_bounds[1][2] = initial_guesses_tuple[2] * (1 + current_bounds_config.get('wid_rel',0.5))
            if scipy_bounds[0][2] <=0 : scipy_bounds[0][2] = 1e-6 # Width must be positive

            if npars >= 4: # Offset (y0)
                # Note: _unpack_pars puts slope (m) at index 3, offset (y0) at index 4 for 5 params
                # For 4 params, it puts offset (y0) at index 3
                offset_idx = 3 if npars == 4 else 4
                scipy_bounds[0][offset_idx] = initial_guesses_tuple[offset_idx] - current_bounds_config['offset_abs']
                scipy_bounds[1][offset_idx] = initial_guesses_tuple[offset_idx] + current_bounds_config['offset_abs']
            if npars == 5: # Slope (m)
                slope_idx = 3
                scipy_bounds[0][slope_idx] = initial_guesses_tuple[slope_idx] - current_bounds_config['slope_abs']
                scipy_bounds[1][slope_idx] = initial_guesses_tuple[slope_idx] + current_bounds_config['slope_abs']
            
            # Transpose for least_squares format
            final_scipy_bounds_transposed = np.array(scipy_bounds)
            # print("Attempting to fit")
            
            try:
                
                result = least_squares(
                    residuals2, x0=p0_scipy_tuple,
                    bounds=final_scipy_bounds_transposed,
                    jac=jacobian_analytical,
                    x_scale='jac',
                    args=args_for_residuals, 
                    # kwargs={'obs': flx1l, 'obs_err': err1l}, # Pass obs and obs_err via kwargs for least_squares
                    ftol=ftol_scipy,
                )
                success = result.success
                if success:
                    pars_arr = result.x
                    residuals_fit = result.fun # fun is the residual vector
                    cost = np.sum(residuals_fit**2) # or result.cost * 2 (cost is 0.5 * sumsq)
                    
                    # Estimate covariance for least_squares
                    # Based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
                    # H = J^T J. Cov = (H)^-1 * (chi^2 / dof) if errors are scaled.
                    # If residuals are already properly scaled ( (data-model)/err ), then Cov = (J^T J)^-1
                    if result.jac is not None:
                        jac = result.jac
                        hess_approx = jac.T @ jac
                        try:
                            covariance_matrix = np.linalg.inv(hess_approx)
                        except np.linalg.LinAlgError:
                            if verbose: print("SciPy (least_squares): Could not invert Jacobian product for covariance.")
                            covariance_matrix = None
                else:
                    if verbose: print(f"SciPy (least_squares) optimization failed: {result.message}")

            except Exception as e:
                if verbose: print(f"Error during SciPy (least_squares) bounded fit: {e}")
                success = False
            # print("Fit success = ", success)
        # Post-process covariance for both scipy methods if successful
        if success and covariance_matrix is not None:
            dof_fit = max(1, len(x1l) - len(pars_arr)) # Ensure dof > 0
            # If residuals_fit were (data-model)/err, then cost is chi2
            # If errors were not used or misestimated, scale pcov by reduced_chi2
            # For now, assume residuals_fit are properly scaled by error if err1l was provided
            # If err1l was None, residuals_fit are (data-model)*weights, so cost is not chi2
            # Let's assume `cost` is sum of ( (obs-model)*weight/err )^2
            if err1l is not None: # Only scale if errors were used to make cost ~ chi2
                 # This scaling is often debated. If errors are correct, scaling isn't needed.
                 # If errors are relative, scaling by reduced_chi2 can be appropriate.
                 # For simplicity, let's take scipy's `pcov` as is from leastsq,
                 # and calculated one from jacobian for least_squares.
                 # Scaling pcov by cost/dof for leastsq:
                 if not bounded: # leastsq provided pcov needs scaling
                    covariance_matrix = covariance_matrix * cost / dof_fit
            try:
                errors_arr = np.sqrt(np.diag(covariance_matrix))
            except (ValueError, TypeError): # e.g. if covariance_matrix has negatives, or is None
                if verbose: print("SciPy: Could not compute errors from covariance matrix.")
                errors_arr = np.full(npars, np.nan)


    elif method == 'lmfit':
        try:
            from lmfit import minimize, fit_report
        except ImportError:
            raise ImportError("lmfit library is required for method='lmfit'. Please install it.")

        p0_lmfit_params = p0_obj # _prepare_pars returns Parameters object for lmfit
        minimize_kws = dict(
            splr_lsf=splr_lsf,
            # lsf_loc_x=lsf_loc_x, lsf_loc_y=lsf_loc_y,
            # sct_loc_x=sct_loc_x, sct_loc_y=sct_loc_y, # sct_loc currently unused
            x=x1l, scale=scale, weight=weight,
            obs=flx1l, obs_err=err1l
        )
        current_lmfit_options = lmfit_options or {}

        try:
            result = minimize(
                residuals2, params=p0_lmfit_params, kws=minimize_kws, **current_lmfit_options
            )
            success = result.success
            
            if success:
                pars_arr = np.array(list(result.params.valuesdict().values())) # Ensure order
                # errors_arr = np.array([result.params[pname].stderr for pname in result.params if result.params[pname].stderr is not None] 
                #                      + [np.nan]*(npars - len([p for p in result.params if result.params[pname].stderr is not None])))
                errors_arr = np.array(
                        [result.params[pname].stderr for pname in result.params if result.params[pname].stderr is not None] + 
                        [np.nan] * (npars - len([pname for pname in result.params if result.params[pname].stderr is not None]))
                    )
                if len(errors_arr) != npars : errors_arr = np.full(npars, np.nan) # Fallback
                
                cost = result.chisqr if result.chisqr is not None else np.sum(result.residual**2)
                residuals_fit = result.residual # These are weighted and error-scaled by residuals2
                covariance_matrix = result.covar
            else:
                if verbose:
                    print(f"LMFIT optimization failed:")
                    # print(fit_report(result)) # Can be very verbose
                    print(f"  Message: {result.message}")
        except Exception as e:
            if verbose: print(f"Error during LMFIT minimization: {e}")
            success = False
            
    else:
        raise ValueError(f"Unknown fitting method: {method}. Choose 'scipy' or 'lmfit'.")

    if not success: # Ensure NaNs if fit failed completely
        pars_arr = np.full(npars, np.nan)
        errors_arr = np.full(npars, np.nan)
        cost = np.nan
        residuals_fit = None

    # --- 4. Post-Fitting Calculations ---
    # Reconstruct the model using the fitted parameters (or NaNs if failed)
    # Important: Pass weight=False (or ensure residuals2 handles it) if model shouldn't be weighted
    # As analyzed, residuals2 does not apply weights_lsf to model_data when obs is None, so this is okay.
    # print(pars_arr)
    model_array = residuals2(
        pars_arr,splr_lsf ,
        #lsf_loc_x, lsf_loc_y, sct_loc_x, sct_loc_y,
        x=x1l, scale=scale, weight=weight, # Pass original weight for consistency in how cen might be used
        obs=None, obs_err=None
    )
    if np.all(np.isnan(pars_arr)): # If pars are NaN, model will be NaN
        model_array = np.full_like(x1l, np.nan)


    # Chi-squared and integral
    chisq, dof = np.nan, 0
    chisq_reduced = np.nan
    integral = np.nan

    if success and not np.all(np.isnan(model_array)): # Only if fit was successful and model is not all NaN
        #chisq and dof are calculated based on the "within" region.
        chisq, dof = get_chisq_dof(x1l, flx1l, err1l, model_array, pars_arr, scale)
        if dof > 0:
            chisq_reduced = chisq / dof
        else:
            chisq_reduced = np.nan # Or np.inf, depending on convention
            if verbose and len(x1l)>0 : print(f"Warning: Degrees of freedom ({dof}) is not positive for chi-squared calculation.")

        # Integral calculation (assuming model_array is 1D as per residuals2)
        # The `within_limits` function uses the *fitted* center from `pars_arr`.
        if not np.isnan(pars_arr[1]): # Ensure center parameter is not NaN
            within_idx = within_limits(x1l, pars_arr[1], scale)
            if np.any(within_idx): # Check if any points are within limits
                 integral = np.sum(model_array[within_idx])
            else:
                 if verbose: print("Warning: No data points within integration limits after fit.")
        else:
            if verbose: print("Warning: Fitted center is NaN, cannot calculate integral.")


    # --- 5. Output ---
    output_tuple = (success, pars_arr, errors_arr, cost, chisq_reduced, integral)

    if output_model:
        output_tuple += (model_array,)
    if output_rsd:
        # If residuals_fit is None (e.g. fit failed early), create NaN array
        if residuals_fit is None:
            residuals_fit_out = np.full_like(x1l, np.nan, dtype=float)
        else:
            residuals_fit_out = residuals_fit
        output_tuple += (residuals_fit_out,)

    fig_object = None
    if plot:
        if success or not np.all(np.isnan(model_array)): # Plot even if fit "failed" but produced a model
            try:
                # Ensure plot_fit can handle potentially NaN parameters gracefully
                fig_object = plot_fit(
                    x1l, flx1l, err1l, model=model_array,
                    pars=pars_arr, scale=scale,
                    lsf_loc_x=lsf_loc_x, lsf_loc_y=lsf_loc_y, # For plotting LSF component
                    rsd_range=rsd_range,
                    title=f"Line Fit ({method})",
                    **plot_kwargs
                )
                output_tuple += (fig_object,)
            except Exception as e:
                if verbose: print(f"Error during plotting: {e}")
                if 'fig_object' in locals() and fig_object is not None: # If fig was created but error later
                     output_tuple += (fig_object,) # Still add it, if it exists
                else:
                     output_tuple += (None,) # Add None if fig creation failed entirely
        else:
            if verbose: print("Skipping plot as fit failed and no model could be generated.")
            output_tuple += (None,)


    return output_tuple

def jacobian_analytical(pars,splr_lsf,
              x,scale,weight=False,obs=None,obs_err=None):
    amp, cen, wid = _unpack_pars(pars)[:3]
    wid = np.abs(wid)
    # 1. Transform x-coordinates to the eIP frame (delta_x)
    if scale[:3] == 'pix':
        dx_vals = x - cen
        dt_dcen = -1.0
    elif scale[:3] == 'vel':
        dx_vals = (x - cen) / cen * 299792.458
        dt_dcen = -299792.458 * x / (cen**2)

    # 2. Evaluate the eIP (psi) and its derivative (psi_prime)
    # Using the pre-calculated spline representation
    psi       = interpolate.splev(dx_vals * wid, splr_lsf, der=0)
    psi_prime = interpolate.splev(dx_vals * wid, splr_lsf, der=1)
    
    # 3. Model derivatives (dModel/dp) [12, 13]
    dM_dA   = psi
    # Applying Chain Rule: dM/dcen = A * d_psi/dx * dx/dcen
    dM_dcen = amp * psi_prime * (1.0 / wid) * dt_dcen
    # Applying Chain Rule: dM/dwid = A * d_psi/dx * dx/dwid
    dM_dwid = amp * psi_prime * (-dx_vals / (wid**2))
    
    # 4. Handle Weighting [14, 15]
    # Use the same weighting logic as residuals2 to ensure consistency
    if weight:
        w_array = assign_weights(x, cen, scale)
    else:
        # 'within_limits' logic from HARPS source [5, 16]
        w_array = np.zeros_like(x)
        w_array[within_limits(x, cen, scale)] = 1.0

    # 5. Stack and return residual derivatives: - (1/sigma) * dModel/dp [17, 18]
    # The negative sign is essential because residuals = (Obs - Model) / Err
    jac_matrix = np.stack([dM_dA, dM_dcen, dM_dwid], axis=1)
    weighted_jac = jac_matrix * w_array[:, np.newaxis]
    
    return -weighted_jac / obs_err[:, np.newaxis]

def get_chisq_dof(x1l,flx1l,err1l,model,pars,scale):
    if len(np.shape(model))>1:
        rsd_norm = (np.sum(model,axis=0)-flx1l)/err1l
    else:
        rsd_norm = (model-flx1l)/err1l
    cen  = _unpack_pars(pars)[1]
    within   = within_limits(x1l,cen,scale)
    dof      = len(within)-len(pars)
    chisq    = np.sum(rsd_norm[within]**2)
    chisqnu  = chisq/dof
    return chisq,dof

def line_gauss(x1l,flx1l,err1l,bary,LSF1d,scale,interpolate=True,
        output_model=False,output_rsd=False,plot=False,save_fig=None,
        rsd_range=None,
        *args,**kwargs):
    from harps.fit import gauss as fit_gauss
    
    output_gauss = fit_gauss(x1l, flx1l, err1l, 
                       model='SimpleGaussian', 
                       xscale=scale,
                       output_model=False)
    success, pars, errors, chisq, chisqnu,integral = output_gauss
    output_tuple = (success, pars, errors, chisq, chisqnu, integral)
    if len(pars)==5:
        A, mu, sigma, m, y0 = pars
    elif len(pars)==4:
        A, mu, sigma, y0 = pars
        m = 0.
    elif len(pars)==3:
        A, mu, sigma = pars
        m  = 0.
        y0 = 0.
    
    model = A*np.exp(-0.5*(x1l-mu)**2/sigma**2)
    try:
        model += m*(x1l-mu) + y0
    except:
        pass
    label = r'Gaussian IP'
    rsd_norm = np.abs((model-flx1l)/err1l)
    if plot:
        fig = plot_fit(x1l,flx1l,err1l,model,pars,scale,
                       # rsd_norm=rsd_norm,
                       rsd_range=rsd_range,
                       is_gaussian=True,
                       **kwargs)
    if output_model:  
        output_tuple =  output_tuple + (model,)
    if output_rsd:  
        output_tuple =  output_tuple + ((flx1l-model)/err1l,)
    if plot:
        output_tuple =  output_tuple + (fig,)
        
    return output_tuple

def plot_fit(x1l,flx1l,err1l,model,pars,scale,is_gaussian=False,
             rsd_norm=None,
             rsd_range=None,**kwargs):
    def func_pixel(x):
        return x - pars[1]
    def inverse_pixel(x):
        return x + pars[1]
    def func_velocity(x):
        return (x/pars[1] - 1)*299792.458
    def inverse_velocity(x):
        return pars[1]*(1+x/299792.458)
    axes = kwargs.pop('axes',None)
    ax_sent = True if axes is not None else False
    if ax_sent:
        ax1, ax2 = axes
    else:
        default_args = dict(figsize=(5,4.5),height_ratios=[3,1],
                           left=0.12,
                           bottom=0.12,
                           top=0.9,
                           hspace=0.02
            )
        fig_args = {**default_args,**kwargs}
        fig = hplt.Figure2(2,1,**fig_args)
        ax1 = fig.add_subplot(0,1,0,1)
        ax2 = fig.add_subplot(1,2,0,1,sharex=ax1)
    
    
    
    within   = within_limits(x1l,pars[1],scale)
    
    ax1.errorbar(x1l,flx1l,err1l,drawstyle='steps-mid',marker='.',
                 label='Data')
    if len(np.shape(model))>1:
        for array,cut in zip(model,[within,~within]):
            ax1.plot(x1l[cut],array[cut],
                     drawstyle='steps-mid',marker='x',lw=3,
                     label='Model')
    else:
        ax1.plot(x1l[within],model[within],drawstyle='steps-mid',marker='x',lw=3,
             label='Model')
    ax1.axvline(pars[1],ls=':',c='k',lw=2)
    
    
    
    # rsd_norm = ((flx1l-model)/err1l)#[within]
    # dof = np.sum(within)-len(pars)
    # chisq = np.sum(rsd_norm**2)
    # chisqnu = chisq/dof
    rsd_norm = rsd_norm if rsd_norm is not None else (model-flx1l)/err1l
    # dof  = (len(x1l) - len(pars))
    # within   = within_limits(x1l,pars[1],scale)
    # chisq = np.sum(rsd_norm[within]**2)
    # chisqnu = chisq/dof
    chisq, dof = get_chisq_dof(x1l,flx1l,err1l,model,pars,scale)
    chisqnu = chisq / dof
    # print(chisq,dof,chisqnu)
    ax1.text(0.95,0.9,r'$\chi^2_\nu=$'+f'{chisqnu:8.2f}',
             ha='right',va='baseline',
             transform=ax1.transAxes)
    if scale[:3]=='pix':
        dx1, dx2 = 5, 2.5
    elif scale[:3]=='vel':
        dv     = np.array([2,4]) # units km/s
        dx1, dx2 = pars[1] *  dv/299792.458
    ax1.axvspan(pars[1]-dx1,pars[1]+dx1,alpha=0.1)
    ax1.axvspan(pars[1]-dx2,pars[1]+dx2,alpha=0.1)
    ax1.xaxis.tick_bottom()
    
    if 'pix' in scale:
        functions = (func_pixel,inverse_pixel)
        for ax in [ax1,ax2]:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5,integer=True))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    elif 'vel' in scale or 'wav' in scale:
        functions = (func_velocity,inverse_velocity)
        for ax in [ax1,ax2]:
            ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax_top = ax1.secondary_xaxis('top', functions=functions)
    
    ax_top.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_top.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax_top.set_xlabel(r'$\Delta x$'+f' ({scale[:3]})',labelpad=3)
    
    # ax2.scatter(x1l,infodict['fvec'],label='infodict')
    xgrid = np.linspace(x1l.min(), x1l.max(), 300)
    lsf_loc_x = kwargs.pop('lsf_loc_x',None)
    lsf_loc_y = kwargs.pop('lsf_loc_y',None)
    if lsf_loc_x is not None and lsf_loc_y is not None:
        splr  = interpolate.splrep(lsf_loc_x , lsf_loc_y / np.max(lsf_loc_y)) 
        ygrid = lsf_model(splr, pars, xgrid, scale)
        label = r'$\psi(\Delta x)$'
    if is_gaussian:
        pars = _unpack_pars(pars)
        A, mu, sigma = pars[:3]
        m = 0; y0 = 0
        if len(pars)==4:
            y0 = pars[-1]
        elif len(pars)==5:
            m, y0 = pars[3:]
        
        ygrid = A*np.exp(-0.5*(xgrid-mu)**2/sigma**2) + m*(xgrid-mu) + y0
        
        label = r'Gaussian IP'
    ax1.plot(xgrid,ygrid,c='grey',lw=2,ls='--',label=label)    
    # if len(pars)==5:
    #     ax1.plot(xgrid,(xgrid-pars[1])*pars[3] + pars[4],lw=2,ls='--',label='linear')
    ax1.legend(loc='upper left')
    weights = assign_weights(x1l,pars[1],scale)[within]
    # rsd  = (flx1l-model)/err1l
    if len(np.shape(rsd_norm))>1:
        for array,cut in zip(rsd_norm,[within,~within]):
            ax2.scatter(x1l[cut],array[cut],label='rsd',
                        edgecolor='k',color='w')
            ax2.scatter(x1l[cut],array[cut],label='rsd',marker='o',
                        alpha=weights)
    else:
        ax2.scatter(x1l[within],rsd_norm[within],label='rsd',
                    edgecolor='k',color='w')
        ax2.scatter(x1l[within],rsd_norm[within],label='rsd',marker='o',
                    alpha=weights)
    
    
    ax2.axhspan(-1,1,color='grey',alpha=0.3)
    # ylim = np.min([1.5*np.nanpercentile(np.abs(rsd_norm),95),10.3])
    # ylim = 1.8*np.max(np.abs(rsd_norm))
    if len(np.shape(rsd_norm))>1:
        default_ylim = np.max(np.abs(rsd_norm[0][within]))
    else:
        default_ylim = np.max(np.abs(rsd_norm[within]))
    ylim = rsd_range if rsd_range is not None else 1.8*default_ylim
    # rsd_range = kwargs.pop('rsd_range',False)
    # print(ylim,rsd_range)
    # if rsd_range:
        # ylim = rsd_range
    ax2.set_ylim(-ylim,ylim)
    if 'pix' in scale:
        ax2.set_xlabel('Pixel')
    elif 'vel' in scale or 'wav' in scale:
        ax2.set_xlabel(r'Wavelength (\AA)')
    # ax2.set_xlabel(f"{scale.capitalize()}")
    ax1.set_ylabel(r"Intensity ($e^-$)")
    ax2.set_ylabel("Residuals "+r"($\sigma$)")
    
    if len(np.shape(rsd_norm))>1:
        rsd_use = rsd_norm[0]
    else:
        rsd_use = rsd_norm
    for x,r,w in zip(x1l[within],rsd_use[within],weights):
        ax2.text(x,r+0.1*ylim*2,f'{w:.2f}',
                  horizontalalignment='center',
                  verticalalignment='center',
                  fontsize=8)
    if not ax_sent:
        fig.ticks_('major', 1,'y',ticknum=3)
        fig.scinotate(0,'y',)
    # ax2.legend()
        return fig
    else:
        return ax1,ax2
def _prepare_pars(npars,method,x,y):
    assert npars>2
    guess_amp = 1.1*np.max(y)
    guess_cen = np.average(x,weights=y)
    guess_wid = 1.0
    if method=='scipy':
        p0 = (guess_amp,guess_cen,guess_wid)
        if npars==4:
            p0 = (p0)+(0.,)
        elif npars==5:
            p0 = (p0)+(0.,0.)
    elif method=='lmfit':
        from lmfit import Parameters
        parameters = [('amp',    guess_amp, True, None,None,None,None),
                      ('cen',    guess_cen, True, None,None,None,None),
                      ('wid',    guess_wid, True, None,None,None,None),
                      ('slope',        0.0, True, None,None,None,None),
                      ('offset',       0.0, True, None,None,None,None),
                      ]
        p0 = Parameters()
        # parameter tuples (name, value, vary, min, max, expr, brute_step).
        p0.add_many(*parameters[:npars])
        
    return p0
    
def _unpack_pars(pars):
    try:
        vals = pars.valuesdict()
    except:
        vals = pars
    npars = len(pars)
    if isinstance(vals,dict):
        m = 0.
        y0 = 0.
        if len(vals)==3:
            amp = vals['amp']
            cen = vals['cen']
            wid = vals['wid']
        elif len(vals)==4:
            amp = vals['amp']
            cen = vals['cen']
            wid = vals['wid']
            y0  = vals['offset']
        elif len(vals)==5:
            amp = vals['amp']
            cen = vals['cen']
            wid = vals['wid']
            y0  = vals['offset']
            m   = vals['slope']
        elif len(vals)==2:
            amp = vals['amp']
            cen = vals['cen']
            wid = 1.
    else:
        amp = vals[0]
        cen = vals[1]
        wid = vals[2]
        y0  = 0.
        m   = 0.
        if len(vals)==4:
            y0 = vals[3]
        if len(vals)==5:
            m  = vals[3]
            y0 = vals[4]
    return (amp,cen,wid,m,y0)[:npars]
def lsf_model(splr, pars, xarray, scale):
    """
    Returns the model of the data from the LSF and parameters provided. 
    Does not include the background.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    amp,cen,wid = _unpack_pars(pars)[:3]
    
    wid   = np.abs(wid)
    # x     = lsf_loc_x * wid
    # y     = lsf_loc_y / np.max(lsf_loc_y) 
    # print(pars)
    # splr  = interpolate.splrep(x,y) 
    
    if scale[:3]=='pix':
        x_test = xarray-cen
    elif scale[:3]=='vel':
        x_test = (xarray-cen)/cen*299792.458
    # model = amp * (m*x_test + y0 + interpolate.splev(x_test,splr))
    model = amp * interpolate.splev(x_test / wid,splr) 
    return model

def sct_model(splr, pars,xarray,scale):
    """
    Returns the model of the data from the LSF and parameters provided. 
    Does not include the background.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    amp,cen,wid,m,y0 = _unpack_pars(pars)
    wid   = np.abs(wid)
    
    
    if scale[:3]=='pix':
        x_test = xarray-cen
    elif scale[:3]=='vel':
        x_test = (xarray-cen)/cen*299792.458
    model = interpolate.splev(x_test / wid, splr)
    return np.exp(model/2.)

def within_limits(xarray,center,scale):
    '''
    Returns a boolean array of length len(xarray), indicating whether the 
    xarray values are within fitting limits.

    Parameters
    ----------
    xarray : array-like
        x-coordinates of the line.
    center : scalar
        centre of the line.
    scale : string
        'pixel or 'velocity'.

    Returns
    -------
    array-like
        A boolean array of length len(xarray). Elements equals True when 
        xarray values are within fitting limits.

    '''
    binlims = get_binlimits(xarray, center, scale)
    low  = np.min(binlims)
    high = np.max(binlims)
    return (xarray>=low)&(xarray<=high)

def get_binlimits(xarray,center,scale):
    if scale[:3]=='pix':
        dx = np.array([-5,-2.5,2.5,5]) # units pix
        binlims = dx + center
    elif scale[:3]=='vel':
        varray = (xarray-center)/center * 299792.458 # units km/s
        # dv     = np.array([-5,-2.5,2.5,5]) # units km/s
        dv = np.array([-4,-2,2,4])
        binlims = center * (1 + dv/299792.458) # units wavelength
    return binlims

def assign_weights(xarray,center,scale):
    def f(x,x1,x2): 
        # a linear function going through x1 and x2
        return np.abs((x-x1)/(x2-x1))
    
    weights  = np.zeros_like(xarray,dtype=np.float64)
    binlims = get_binlimits(xarray, center, scale)
        
    idx      = np.digitize(xarray,binlims)
    cut1     = np.where(idx==2)[0]
    cutl     = np.where(idx==1)[0]
    cutr     = np.where(idx==3)[0]
    # ---------------
    # weights are:
    #  = 1,           -2.5<=x<=2.5
    #  = 0,           -5.0>=x & x>=5.0
    #  = linear[0-1]  -5.0<x<-2.5 & 2.5>x>5.0 
    # ---------------
    weights[cutl] = f(xarray[cutl],binlims[0],binlims[1])
    weights[cutr] = f(xarray[cutr],binlims[3],binlims[2])
    weights[cut1] = 1
    return weights