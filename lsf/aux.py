#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
import harps.functions.math as mathfunc
import harps.functions.outliers as outlier
import harps.functions.aux as auxfunc
import harps.functions.spectral as specfunc


import harps.containers as container
import harps.lsf.fit as hlsfit
import harps.version as hv
import harps.progress_bar as progress_bar
import harps.lines_aux as laux
import harps.settings as hs
import harps.wavesol as ws

import gc
import numpy as np
import harps.lsf.inout as io
import hashlib
import sys
import logging
import jax
import jax.numpy as jnp
import time
import multiprocessing
import copy
from   functools import partial
import traceback

from scipy import interpolate
from scipy.optimize import brentq
import scipy.stats as stats

from fitsio import FITS

import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List, Any



def prepare_array(array):
    if array is not None:
        dim = len(np.shape(array))
        if dim == 3:
            output = np.atleast_3d(array)
        elif dim<3:
            output = np.moveaxis(np.atleast_3d(array),-1,0)
        else:
            raise Exception("Array has more than 3 dimensions.")
    else:
        output = None
    return output





def stack(*args,**kwargs):
    return stack_subbkg_divenv(*args,**kwargs)



def stack_subbkg_divenv(fittype,linelists,flx3d_in,x3d_in,err3d_in,
          env3d_in,bkg3d_in,orders=None,subbkg=hs.subbkg,divenv=hs.divenv):
    # numex = np.shape(linelists)[0]
    
    logging.info(f'subbkg, divenv = {subbkg}, {divenv}')
    
    ftpix = '{}_pix'.format(fittype)
    ftwav = '{}_wav'.format(fittype)
    
    x3d_in   = prepare_array(x3d_in)
    flx3d_in = prepare_array(flx3d_in)
    bkg3d_in = prepare_array(bkg3d_in)
    env3d_in = prepare_array(env3d_in)
    err3d_in = prepare_array(err3d_in)
    
    numex, numord, numpix = np.shape(flx3d_in)
    pix3d = np.zeros((numord,numpix,numex))
    flx3d = np.zeros((numord,numpix,numex))
    err3d = np.zeros((numord,numpix,numex))   
    vel3d = np.zeros((numord,numpix,numex)) 
    
    
    
    
    data, data_error, bkg_norm = laux.prepare_data(flx3d_in,env3d_in,bkg3d_in, 
                                         subbkg=subbkg, divenv=divenv)
    
    
    linelists = np.atleast_2d(linelists)
    for exp,linelist in enumerate(linelists):
        progress_bar.update((exp+1)/len(linelists),"Stack")
        if orders is not None:
            orders = np.atleast_1d(orders)
        else:
            orders = np.unique(linelist['order'])
        # print(orders)
        for j,line in enumerate(linelist):
            od       = line['order']
            if od not in orders:
                continue
            pixl     = line['pixl']
            pixr     = line['pixr']
            # print(pixl,pixr)
            f_star = line[f'{ftpix}_integral']
            x_star = line[ftpix][1]
            wav1l = x3d_in[exp,od,pixl:pixr]
            vel1l = (wav1l - line[ftwav][1])/line[ftwav][1]*299792.458 #km/s
            # print(j, np.all(np.isfinite(vel1l)))
            # if any of vel1l are nan or have velocities larger than 8 km/s
            # then set the entire velocity array to nan, it will be removed
            # later by ``clean_input''
            cond = np.any(np.isnan(vel1l)) #or np.any(np.abs(vel1l)>8.0)
            if cond:
                vel1l = np.full_like(wav1l, np.nan)
            outside = np.abs(vel1l)>8
            vel1l[outside] = 0.
            # flux is Poissonian distributed, P(nu),  mean = variance = nu
            # Sum of fluxes is also Poissonian, P(sum(nu))
            #           mean     = sum(nu)
            #           variance = sum(nu)
            # C_flux = np.sum(flx1l)
            C_flux = f_star
            C_flux_err = np.sqrt(C_flux)
            # C_flux_err = 0.
            
            data1l = data[exp,od,pixl:pixr]
            data1l_var_tmp = (data_error[exp,od,pixl:pixr])**2
            # data1l_var = laux.quotient_variance(data1l, data1l_var_tmp, 
            #                                     f_star, np.sqrt(f_star))
            flx1l = data1l/f_star
            p = flx1l
            data1l_var = p*(1-p)/f_star 
            data1l_err = np.sqrt(data1l_var)
            
            
            pix3d[od,pixl:pixr,exp] = np.arange(pixl,pixr) - x_star
            vel3d[od,pixl:pixr,exp] = vel1l
            flx3d[od,pixl:pixr,exp] = flx1l
            # error propagation from normalisation
            # N = F/C_flux
            # sigma_N = 1/C_flux * np.sqrt(sigma_F**2 + (N * C_flux_err)**2)
            err3d[od,pixl:pixr,exp] = data_error[exp,od,pixl:pixr]/f_star
            # err3d[od,pixl:pixr,exp] = data1l_err
            # err3d[od,pixl:pixr,exp] = 1./f_star*np.sqrt(data_error[exp,od,pixl:pixr]**2 + \
            #                                 data[exp,od,pixl:pixr]*f_star)
            
    pix3d = jnp.array(pix3d)
    vel3d = jnp.array(vel3d)
    flx3d = jnp.array(flx3d*100)
    err3d = jnp.array(err3d*100)
    
    
    return pix3d,vel3d,flx3d,err3d,orders


def stack_outpath(outpath,version,wavesol_version,orders=None,subbkg=hs.subbkg,divenv=hs.subbkg,
                   **kwargs):
    # wav2d = spec.wavereference
    # wav2d = spec['wavesol_gauss',701] # this should be changed to a new wsol every iteration
    item,fittype  = get_linelist_item_fittype(version)
    logging.info(f"Stacking {item}, {fittype}")
    with FITS(outpath) as hdul:
        flx2d = hdul['flux'].read()
        bkg2d = hdul['background'].read()
        env2d = hdul['envelope'].read()
        err2d = np.sqrt(np.abs(flx2d+bkg2d))
        llist = hdul[item].read()
        wref  = hdul['wavereference'].read()
    nord, npix = np.shape(flx2d)
    if version//100==1:
        wav2d = wref
    else:
        wav2d = ws.comb_dispersion(linelist=llist, 
                                   version=wavesol_version, 
                                   fittype=fittype,
                                   npix=npix, 
                                   nord=nord)
    # orders = spec.prepare_orders(order)
    return stack_subbkg_divenv(fittype,llist,flx2d,wav2d,err2d,env2d,bkg2d,
                               orders=orders,subbkg=subbkg,divenv=divenv,
                               **kwargs) 

def stack_spectrum(spec,version,wavesol_version,orders=None,subbkg=hs.subbkg,divenv=hs.subbkg,
                   **kwargs):
    # wav2d = spec.wavereference
    # wav2d = spec['wavesol_gauss',701] # this should be changed to a new wsol every iteration
    flx2d = spec['flux']
    bkg2d = spec['background']
    env2d = spec['envelope']
    err2d = np.sqrt(np.abs(flx2d)+np.abs(bkg2d))
    
    item,fittype  = get_linelist_item_fittype(version)
    print(item,fittype)
    logging.info(f"Stacking {item}, {fittype}")
    llist = spec[item]
    nord, npix = np.shape(flx2d)
    if version//100==1:
        wav2d = spec['wavereference']
    else:
        wav2d = ws.comb_dispersion(linelist=llist, 
                                   version=wavesol_version, 
                                   fittype=fittype,
                                   npix=npix, 
                                   nord=nord)
    # orders = spec.prepare_orders(order)
    return stack_subbkg_divenv(fittype,llist,flx2d,wav2d,err2d,env2d,bkg2d,
                               orders=orders,subbkg=subbkg,divenv=divenv,
                               **kwargs) 

def _prepare_lsf1s(n_data,n_sct,pars):
    lsf1s = get_empty_lsf(1,n_data,n_sct,pars)#[0]
    return lsf1s

def _calculate_shift(y,x):
    return -mathfunc.derivative_zero(y,x,-1,1)

# @jax.jit
# def loss_(theta,X,Y,Y_err):
#     gp = build_gp(theta,X,Y_err)
#     return -gp.log_probability(Y)




    
def bin_means(x,y,xbins,minpts=10,value='mean',kind='spline',y_err=None,
              remove_outliers=False,return_population_variance=False,
              return_population_kurtosis=False,
              return_variance_variance=False):
    
    if return_variance_variance:
        calc_pop_var = True
        calc_pop_kurt = True
        calc_var_var = True
    if return_population_kurtosis:
        calc_pop_var = True
        calc_pop_kurt = True
    if return_population_variance:
        calc_pop_var = True
    def interpolate_bins(means,missing_xbins,kind):
        
        x = xbins[idx]
        y = means[idx]
        if kind == 'spline':
            splr  = interpolate.splrep(x,y)
            model = interpolate.splev(missing_xbins,splr)
        else:
            model = np.interp(missing_xbins,x,y)
        return model
   # which pixels have at least minpts points in them?
    hist, edges = np.histogram(x,xbins)
    bins  = np.where(hist>=minpts)[0]+1
    # sort the points into bins and use only the ones with at least minpts
    inds  = np.digitize(x,xbins,right=False)
    means = np.zeros(len(xbins))
    stds  = np.zeros(len(xbins))
    var_pop = np.zeros(len(xbins))
    kurt  = np.zeros(len(xbins))
    var_var = np.zeros(len(xbins))
    idx   = bins
    # first calculate means for bins in which data exists
    for i in idx:
        # skip if more right than the rightmost bin
        if i>=len(xbins):
            continue
        # select the points in the bin
        cut = np.where(inds==i)[0]
        if len(cut)<1:
            print("Deleting bin ",i)
            continue
        y1  = y[cut]
        
        if remove_outliers == True:
            outliers = outlier.is_outlier(y1)
            y1=y1[~outliers]
        
        if value=='mean':
            means[i] = np.nanmean(y1)
            stds[i]  = np.nanstd(y1)
        elif value=='median':
            means[i] = np.nanmedian(y1)
            stds[i]  = np.nanstd(y1)
        elif value=='weighted_mean':
            assert y_err is not None
            means[i],stds[i] = mathfunc.wmean(y1,y_err[cut])
        if calc_pop_var:
            var_pop[i] = np.sum((y1-np.mean(y1))**2)/(len(y1)-1)
        if calc_pop_kurt:
            # first calculate the biased sample 4th moment and correct for bias
            n        = len(y1)
            mom4_sam = stats.moment(y1,moment=4,nan_policy='omit') 
            mom4_pop = n/(n-1)*mom4_sam
            # kurtosis is the 4th population moment / standard deviation**4
            kurt[i]  = mom4_pop / np.power(var_pop[i],2.)
        if calc_var_var:
            n = len(y1)
            var_var[i] = (kurt[i] - (n-3)/(n-1))*np.power(var_pop[i],2.)/n
    # go back and interpolate means for empty bins
    idy   = auxfunc.find_missing(idx)
    # interpolate if no points in the bin, but only pixels -5 to 5
    if len(idy)>0:
        idy = np.atleast_1d(idy)
        means[idy] = interpolate_bins(means,xbins[idy],kind)
        
    return_tuple = (means,stds,hist)
    if return_population_variance: # this is really variance, not standard dev
        return_tuple = return_tuple + (var_pop,)
    if return_population_kurtosis:   
        return_tuple = return_tuple + (kurt,)
    if return_variance_variance:   
        return_tuple = return_tuple + (var_var,)
    return return_tuple

def bin_statistics(x,y,minpts=10,
                   calculate = ['mean','std','pop_variance','pop_kurtosis',
                                'pop_variance_variance'],
                   remove_outliers=False):
    
    counts, bin_edges = bin_optimally(x,minpts)
    # means, stds = get_bin_mean_std(x, y, bin_edges)
    
    arrays = get_bin_stat(x, y, bin_edges,calculate=calculate, 
                          remove_outliers=remove_outliers)
    
    
    return arrays
    
def bin_optimally(
    x: Union[List[float], np.ndarray],
    minpts_per_bin: int = 10,
    initial_nbins: int = 50,
    min_allowed_nbins: int = 5,
    max_abs_val_for_range: Optional[float] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Attempts to find an "optimal" number of bins for a 1D histogram of x,
    such that all non-empty bins contain at least `minpts_per_bin` data points.

    It starts with `initial_nbins` and iteratively reduces the number of bins
    until the condition is met or `min_allowed_nbins` is reached.
    The histogram range is symmetric around zero, determined by `max_abs_val_for_range`
    or the maximum absolute value in x.

    Parameters
    ----------
    x : array-like
        Input 1D data array.
    minpts_per_bin : int, optional
        The minimum number of data points required in each non-empty bin.
        Default is 10.
    initial_nbins : int, optional
        The number of bins to start the search with. Default is 50.
    min_allowed_nbins : int, optional
        The minimum number of bins to try. If the number of bins drops
        below this, the search stops. Default is 3.
    max_abs_val_for_range : float, optional
        If provided, this value is used to set the histogram range as
        (-max_abs_val_for_range, max_abs_val_for_range).
        If None, it's determined from `np.max(np.abs(x_arr))`. Default is None.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        - counts : np.ndarray or None
            The histogram counts for the "optimal" binning, or the counts from
            the last attempt if no optimal solution was found according to criteria.
            Returns None if input x is empty and no valid range can be determined.
        - bin_edges : np.ndarray or None
            The bin edges for the "optimal" binning, or edges from the last
            attempt. Returns None if input x is empty.
    """
    x_arr = np.asarray(x)

    if x_arr.size == 0:
        # print("Warning: Input array 'x' is empty.")
        return np.array([]), np.array([]) # Consistent with np.histogram on empty

    if minpts_per_bin <= 0:
        raise ValueError("minpts_per_bin must be positive.")
    if initial_nbins < min_allowed_nbins:
        # print(f"Warning: initial_nbins ({initial_nbins}) is less than "
        #       f"min_allowed_nbins ({min_allowed_nbins}). "
        #       f"Starting with min_allowed_nbins.")
        current_nbins = min_allowed_nbins
    else:
        current_nbins = initial_nbins
    
    if min_allowed_nbins < 1:
        raise ValueError("min_allowed_nbins must be at least 1.")


    if max_abs_val_for_range is not None:
        if max_abs_val_for_range < 0:
            raise ValueError("max_abs_val_for_range must be non-negative.")
        xlim = max_abs_val_for_range
    else:
        xlim = np.max(np.abs(x_arr))

    # Handle case where all x are zero or xlim is effectively zero
    if np.isclose(xlim, 0):
        # If all data is at/near zero, create a single bin centered at zero
        # if it meets the minpts_per_bin criterion.
        if x_arr.size >= minpts_per_bin:
            # np.histogram with range=(0,0) and nbins=1 might be problematic for some numpy versions
            # Manually create the single bin result
            # A single bin from -epsilon to +epsilon or just a conceptual bin at 0
            # For simplicity, let's return one bin that covers "0"
            # The exact edges are less critical if all data is at 0.
            return np.array([x_arr.size]), np.array([-1e-9, 1e-9]) # Small range around 0
        else:
            # Cannot satisfy minpts_per_bin with a single bin for all-zero data
            # print("Warning: All data is at zero, but not enough points to satisfy minpts_per_bin.")
            return np.array([x_arr.size]), np.array([-1e-9, 1e-9]) # Return the counts anyway, user can check

    last_counts: Optional[np.ndarray] = None
    last_bin_edges: Optional[np.ndarray] = None
    found_optimal = False

    while current_nbins >= min_allowed_nbins:
        try:
            counts, bin_edges = np.histogram(x_arr, bins=current_nbins, range=(-xlim, xlim))
        except ValueError as e: # e.g. if range max < range min due to xlim issues (should be caught by xlim>=0)
            print(f"Error during histogram calculation with nbins={current_nbins}: {e}")
            current_nbins -= 1
            continue
        except Exception as e: # Catch other unexpected np.histogram errors
            print(f"An unexpected error occurred with np.histogram (nbins={current_nbins}): {e}")
            # Fallback: store current attempt and break or reduce nbins
            last_counts, last_bin_edges = (np.array([]), np.array([])) # Indicate failure
            current_nbins -= 1 # Or break, depending on desired strictness
            continue


        last_counts, last_bin_edges = counts, bin_edges # Store current attempt

        # Consider only bins with actual data points
        non_empty_bin_counts = counts[counts > 0]

        if non_empty_bin_counts.size == 0 and x_arr.size > 0:
            # All bins are empty, but data exists. This might happen if xlim
            # is too restrictive (though it's derived from data).
            # Or if data points fall exactly on edges in a way that np.histogram misses them.
            # This indicates the current binning is not useful.
            all_bins_sufficient = False
        elif non_empty_bin_counts.size == 0 and x_arr.size == 0: # Already handled
             all_bins_sufficient = True # Vacuously true
        else:
            all_bins_sufficient = np.all(non_empty_bin_counts >= minpts_per_bin)

        if all_bins_sufficient:
            found_optimal = True
            break  # Found a suitable binning

        current_nbins -= 1

    # If no optimal solution was found, last_counts and last_bin_edges
    # will hold the results from the attempt with min_allowed_nbins.
    # This matches the original function's implicit behavior on failure.
    # if not found_optimal and x_arr.size > 0:
    #     print(f"Warning: Could not find an optimal binning satisfying minpts_per_bin={minpts_per_bin}. "
    #           f"Returning results for nbins={min_allowed_nbins} (or last valid attempt).")

    return last_counts, last_bin_edges

def bin_optimally_old(x,minpts=10):
    MIN_NBINS=3
    # determine histogram limits, symmetric around zero
    # xmin = np.abs(np.min(x))
    # xmax = np.abs(np.max(x))
    xlim = np.max(np.abs(x))
    # choose a starting value for nbins and count the number of points in bins
    nbins = 50
    
    
    # stopping condition is defined as False for first iteration
    condition = False
    while condition == False:
        try:
            counts ,bin_edges = np.histogram(x,nbins,range=(-xlim,xlim))
        except Exception as e:
            # Catch other potential errors
            print(f"An unexpected error occurred : {e}")
            print(x, nbins,xlim)
        # remove bins with no points from consideration (there may be gaps in data)
        cut = np.where(counts!=0)[0]
        counts_ = counts[cut]
        condition = np.all(counts_>minpts)
        if not condition: nbins = nbins - 1
        if nbins<MIN_NBINS:
            break
    return counts, bin_edges
        


        
def get_bin_stat(x,y,bin_edges,calculate=['mean','std'],remove_outliers=True,
                 usejax=False):
    allowed = ['mean','std','sam_variance','sav_variance_variance',
               'pop_variance','pop_kurtosis','pop_variance_variance']
    if usejax:
        import jax.numpy as np
    else:
        import numpy as np
    if isinstance(calculate,list):
        pass
    else:
        calculate = np.atleast_1d(calculate)
    # assert calculate in allowed, 'input not recognised'
    indices = np.digitize(x,bin_edges)
    nbins  = len(bin_edges)-1
    bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
    arrays = {name:np.full(nbins,np.nan) for name in calculate}
    arrays.update({'bin_centres':bin_centres})
    functions = {'mean':np.nanmean,
                 'std':np.nanstd,
                 'sam_variance':np.nanvar,
                 'pop_variance':np.nanvar,
                 'sam_variance_variance':get_samvar_variance,
                 'pop_kurtosis':get_kurtosis,
                 'pop_variance_variance':get_popvar_variance}
    arguments = {'mean':{},
                 'std':{},
                 'sam_variance':{},
                 'sam_variance_variance':{},
                 'pop_variance':{'ddof':1},
                 'pop_kurtosis':{},
                 'pop_variance_variance':{}}
    # means = np.zeros(nbins)
    # stds  = np.zeros(nbins)
    # plt.scatter(x,y)
    # [plt.axvline(pos,ls=':',c='k') for pos in bin_edges]
    flagged = np.zeros(nbins,dtype=bool)
    for i in range(nbins):
        try:
            cut = np.where(indices==i+1)[0]
        except:
            import jax.numpy as jnp
            cut = jnp.where(indices==i+1,size=len(indices),fill_value=False)[0]
        if len(cut)>0: 
            pass
        else:
            flagged[i]=True
            continue
        y_i = y[cut] 
        if remove_outliers == True:
            outliers = outlier.is_outlier(y_i)
            try:
                y_i=y_i[~outliers]
            except:
                pass
        for name in calculate:
            try:
                val = functions[name](y_i,**arguments[name])
                if np.isfinite(val):
                    arrays[name][i] = val
                else:
                    # arrays[name][i] = np.nan
                    flagged[i] = True 
            except:
                val = functions[name](y_i,**arguments[name])
                arrays[name].at[i].set(val)
    output_dict = dict()
    for key,array in arrays.items():
        output_dict[key] = array[~flagged]
    # plt.errorbar((bin_edges[1:]+bin_edges[:-1])/2,means,stds,ls='',marker='s',c='r')
    return output_dict

def get_kurtosis_old(x,*args,**kwargs):
    n        = len(x)
    diff     = x-jnp.nanmean(x)
    # mom4_sam = 1./n * jnp.nansum(expt_rec(diff,4))
    mom4_sam = stats.moment(x,moment=4,nan_policy='omit') 
    mom4_pop = n/(n-1)*mom4_sam
    var_pop  = jnp.nanvar(x,ddof=1)
    # kurtosis is the 4th population moment / standard deviation**4
    return mom4_pop / jnp.power(var_pop,2.)


def get_kurtosis(
    x: Union[jnp.ndarray, np.ndarray, List, Tuple],
    fisher: bool = True,
    ddof_method: str = 'n-1',
    min_points_for_kurtosis: int = 4, # More explicit minimum for stable kurtosis
    **kwargs: Any # To catch unused kwargs
) -> jnp.ndarray:
    """
    Calculates kurtosis using JAX NumPy.

    Parameters
    ----------
    x : array-like
        Input data. Should be convertible to a JAX array.
    fisher : bool, optional
        If True (default), calculates excess kurtosis (subtracts 3).
        If False, calculates Pearson's moment kurtosis.
    ddof_method : str, optional
        Determines the denominator for moment calculations.
        - 'n': Uses N_eff (count of non-NaNs) as the denominator.
        - 'n-1': Uses N_eff - 1 as the denominator.
    min_points_for_kurtosis : int, optional
        The minimum number of non-NaN data points required for a meaningful
        kurtosis calculation. Defaults to 4, as unbiased kurtosis estimators
        often require at least 4 points. If N_eff < this, NaN is returned.
    **kwargs : Any
        Catches any other keyword arguments from the original signature.

    Returns
    -------
    jnp.ndarray
        The calculated kurtosis (scalar JAX array), or NaN if insufficient data.
    """
    x_jnp = jnp.asarray(x)
    n_eff = jnp.sum(jnp.isfinite(x_jnp))

    # Check if enough data points for meaningful kurtosis
    if n_eff < min_points_for_kurtosis:
        return jnp.nan

    # Additional checks based on ddof_method
    if ddof_method == 'n-1':
        if n_eff < 2: # N_eff - 1 would be < 1
            return jnp.nan
    elif ddof_method == 'n':
        if n_eff < 1: # Should be covered by min_points_for_kurtosis already
            return jnp.nan
    else:
        # This case should ideally raise an error or be handled,
        # but for JAX JIT, returning NaN might be preferred over raising errors.
        # However, an invalid parameter should ideally be caught early.
        # For now, let's assume valid ddof_method or rely on downstream errors.
        # raise ValueError("ddof_method must be 'n' or 'n-1'")
        pass


    mean_x = jnp.nanmean(x_jnp)
    deviations = x_jnp - mean_x

    if ddof_method == 'n-1':
        var_x = jnp.nanvar(x_jnp, ddof=1)
        sum_dev4 = jnp.nansum(deviations**4)
        # Ensure n_eff - 1 is not zero before division
        mom4_x = jnp.where(n_eff > 1, sum_dev4 / (n_eff - 1.0), jnp.nan)
    elif ddof_method == 'n':
        var_x = jnp.nanvar(x_jnp, ddof=0)
        # Ensure n_eff is not zero before division
        mom4_x = jnp.where(n_eff > 0, jnp.nansum(deviations**4) / n_eff, jnp.nan)
        # Simpler: mom4_x = jnp.nanmean(deviations**4) if n_eff > 0 else jnp.nan

    if jnp.isnan(var_x) or var_x == 0 or jnp.isnan(mom4_x):
        return jnp.nan

    pearson_kurtosis = mom4_x / (var_x**2)

    if fisher:
        return pearson_kurtosis - 3.0
    else:
        return pearson_kurtosis

def get_samvar_variance(x,*args,**kwargs):
    '''
    Returns variance on the sample variance.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    n        = len(x)
    mom4_sam = stats.moment(x,moment=4,nan_policy='omit') 
    # diff     = x-jnp.nanmean(x)
    # mom4_sam = 1./n * jnp.nansum(expt_rec(diff,4))
    var      = jnp.nanvar(x)
    
    return mom4_sam/n - var**2 * (n-3)/(n*(n-1))

def expt_rec(a, b):
    if b == 0:
        return 1
    elif b % 2 == 1:
        return a * expt_rec(a, b - 1)
    else:
        p = expt_rec(a, b / 2)
        return p * p

def get_popvar_variance(
    x: Union[jnp.ndarray, np.ndarray, List, Tuple],
    kurtosis_function: callable = get_kurtosis, # Allow injecting kurtosis func
    **kurtosis_kwargs: Any # Pass kwargs to the kurtosis function
) -> jnp.ndarray:
    """
    Estimates a quantity related to the variance of the sample variance.
    The formula used is ( (kappa - (n_eff-3)/(n_eff-1)) * var_pop^2 ) / n_eff,
    which is an estimator for the variance of s^2 under normality assumptions,
    where n_eff is the number of non-NaN observations.

    Parameters
    ----------
    x : array-like
        Input data.
    kurtosis_function : callable, optional
        The function to use for calculating kurtosis.
        Defaults to `get_kurtosis_jax`.
    **kurtosis_kwargs : Any
        Keyword arguments to pass to the `kurtosis_function`.
        Common ones for `get_kurtosis_jax` would be `fisher` and `ddof_method`.

    Returns
    -------
    jnp.ndarray
        The calculated value, or NaN if inputs are invalid or insufficient.
    """
    x_jnp = jnp.asarray(x)
    n_eff = jnp.sum(jnp.isfinite(x_jnp))

    # The formula (n-3)/(n-1) requires n-1 != 0 and ideally n >= 3 for meaningfulness.
    # For the overall formula, n_eff must be > 0.
    # Let's set a practical minimum for n_eff. For the term (n_eff-3)/(n_eff-1),
    # n_eff should be at least 2 to avoid division by zero (n_eff-1).
    # If n_eff = 2, (n_eff-3)/(n_eff-1) = -1.
    # If n_eff = 1, n_eff-1 = 0 (division by zero).
    # If n_eff = 0, division by n_eff.

    if n_eff < 2: # Need at least 2 points for (n_eff-1) term.
                  # Also, division by n_eff requires n_eff > 0.
        return jnp.nan

    var_pop_ddof1 = jnp.nanvar(x_jnp, ddof=1) # Uses n_eff-1 in denominator

    # If variance is NaN (e.g., n_eff < 2 for ddof=1) or zero, result is likely NaN or problematic
    if jnp.isnan(var_pop_ddof1):
        return jnp.nan
    # If var_pop_ddof1 is 0, the numerator becomes kurt * 0 - term * 0 = 0.
    # Then 0 / n_eff. This is okay unless n_eff is also 0 (caught above).

    # Ensure default kurtosis parameters if not provided
    k_kwargs = {'fisher': False, 'ddof_method': 'n-1', **kurtosis_kwargs}
    kurt = kurtosis_function(x_jnp, **k_kwargs)


    if jnp.isnan(kurt):
        return jnp.nan

    # Calculate the (n_eff-3)/(n_eff-1) term safely
    # n_eff is already guaranteed to be >= 2 here.
    n_factor_numerator = n_eff - 3.0
    n_factor_denominator = n_eff - 1.0
    
    # This check is now redundant due to n_eff < 2 check above, but good for clarity
    # if n_factor_denominator == 0:
    #     return jnp.nan
    n_factor = n_factor_numerator / n_factor_denominator

    numerator = (kurt - n_factor) * jnp.power(var_pop_ddof1, 2.)
    
    # Final division by n_eff (guaranteed > 0 here due to n_eff < 2 check)
    result = numerator / n_eff

    return result

def get_popvar_variance_old(x,*args,**kwargs):
    n = len(x)
    var_pop = jnp.nanvar(x,ddof=1)
    kurt    = get_kurtosis(x,*args,**kwargs)
    return (kurt - (n-3)/(n-1))*jnp.power(var_pop,2.)/n

# def solve_1d(lsf2d,linelist1d,x1d,flx1d,bkg1d,err1d,fittype,scale='pix',
#              interpolate=False):
    
#     tot = len(linelist1d)
#     scl = f"{scale[:3]}"
#     for i, line in enumerate(linelist1d):
#         od   = line['order']
        
#         lpix = line['pixl']
#         rpix = line['pixr']
#         flx  = flx1d[lpix:rpix]
#         x    = x1d[lpix:rpix]
#         bkg  = bkg1d[lpix:rpix]
#         err  = err1d[lpix:rpix]
#         try:
#             lsf1d  = lsf2d[od].values
#         except:
#             continue
#         if len(lsf1d)>len(np.unique(lsf1d['segm'])):
#             realnseg = len(lsf1d)
#             expnseg  = len(np.unique(lsf1d['segm']))
#             raise ValueError(f"Expected {expnseg} segments, got {realnseg}")
#         success = False
#         # print(x,flx,bkg,err)
#         try:
            
#             output = hfit.lsf(x,flx,bkg,err,lsf1d,interpolate=interpolate,
#                               output_model=False)
#             success, pars, errs, chisq, chisqnu = output
#         except:
#             pass
#         # sys.exit()
#         print('\nline',i,success,pars,chisq, chisqnu)
#         # sys.exit()
#         if not success:
#             print('fail')
#             pars = np.full(3,np.nan)
#             errs = np.full(3,np.nan)
#             chisq = np.nan
#             continue
#         else:
#             pars[1] = pars[1] 
#             line[f'lsf_{scl}']     = pars
#             line[f'lsf_{scl}_err'] = errs
#             line[f'lsf_{scl}_chisq']  = chisq
#             line[f'lsf_{scl}_chisqnu']  = chisqnu
        
#         progress_bar.update((i+1)/tot,"Solve")
#     return linelist1d

def get_linelist_item_fittype(version,fittype=None):
    if version==1:
        item = ('linelist',version)
        default_fittype = 'lsf'
    elif version>1 and version<=200:
        item = 'linelist'
        default_fittype = 'gauss'
    else:
        item = ('linelist',version-100)
        default_fittype = 'lsf'
    
        
    fittype = fittype if fittype is not None else default_fittype
    return item,fittype

def read_outfile4solve(out_filepath,version,scale):
    with FITS(out_filepath,'rw',clobber=False) as hdu:
        item,fittype = get_linelist_item_fittype(version)
        # print(item,fittype)
        # centres = hdu[item].read(columns=f'{fittype}_{scale[:3]}')[:,1]
            # linelist_im1 = hdu['linelist',iteration-1].read()
        linelist = hdu[item].read()
        flx2d = hdu['flux'].read()
        err2d = hdu['error'].read()
        env2d = hdu['envelope'].read()
        bkg2d = hdu['background'].read()
        
        nbo,npix = np.shape(flx2d)
        x2d   = np.vstack([np.arange(npix) for od in range(nbo)])
    return x2d,flx2d,err2d,env2d,bkg2d,linelist
        
def solve(out_filepath,lsf_filepath,iteration,order,force_version=None,
          model_scatter=False,interpolate=False,scale=['pixel','velocity'],
          npars = None,
          subbkg=hs.subbkg,divenv=hs.divenv,save2fits=True,logger=None):
    from fitsio import FITS
    from harps.lsf.container import LSF2d
    
    def bulk_fit(function):
        manager = multiprocessing.Manager()
        inq = manager.Queue()
        outq = manager.Queue()
    
        # construct the workers
        nproc = multiprocessing.cpu_count()
        logger.info(f"Using {nproc} workers")
        workers = [LineSolver(str(name+1), function,inq, outq) 
                   for name in range(nproc)]
        for worker in workers:
            worker.start()
    
        # add data to the queue for processing
        work_len = tot
        for item in cut:
            inq.put(item)
    
        while outq.qsize() != work_len:
            # waiting for workers to finish
            done = outq.qsize()
            progress = done/(work_len-1)
            time_elapsed = time.time() - time_start
            progress_bar.update(progress,name='lsf.aux.solve',
                               time=time_elapsed,
                               logger=None)
            
            time.sleep(1)
    
        # clean up
        for worker in workers:
            worker.terminate()
    
        # print the outputs
        results = []
        while not outq.empty():
            results.append(outq.get())
        return results
    
    if logger is not None:
        logger = logger.getChild('solve')
    else:
        logger = logging.getLogger(__name__).getChild('solve')
    # abbreviations
    # scl = f'{scale[:3]}'
    if force_version is not None:
        version = force_version
    else:
        version = hv.item_to_version(dict(iteration=iteration,
                                        model_scatter=model_scatter,
                                        interpolate=interpolate
                                        ),
                                   ftype='lsf'
                                   )
    scale = np.atleast_1d(scale)
    logger.info(f'version : {version}')
    # READ LSF
    with FITS(lsf_filepath,'r',clobber=False) as hdu:
        if 'pixel' in scale:
            lsf2d_pix = hdu['pixel_model',version].read()
            LSF2d_nm_pix = LSF2d(lsf2d_pix)
        if 'velocity' in scale:
            lsf2d_vel = hdu['velocity_model',version].read()
            LSF2d_nm_vel = LSF2d(lsf2d_vel)
    # lsf2d_gp = LSF2d_gp[order].values
    # lsf2d_numerical = hlsfit.numerical_model(lsf2d_gp,xrange=(-8,8),subpix=11)
    # LSF2d_numerical = LSF(lsf2d_numerical)
    
    
    # COPY LINELIST 
    io.copy_linelist_inplace(out_filepath, version)
    
    # READ OLD LINELIST AND DATA
    x2d,flx2d,err2d,env2d,bkg2d,linelist = read_outfile4solve(out_filepath,
                                                        version,
                                                        scale='pixel')
    flx_norm, err_norm, bkg_norm  = laux.prepare_data(flx2d,env2d,bkg2d, 
                                         subbkg=subbkg, divenv=divenv)
    
    
    # MAKE MODEL EXTENSION
    io.make_extension(out_filepath, 'model_lsf', version, flx2d.shape)
    
    nbo,npix = np.shape(flx2d)
    orders = specfunc.prepare_orders(order, nbo, sOrder=39, eOrder=None)
    
    # firstrow = int(1e6)
    cut_ = [np.ravel(np.where(linelist['order']==od)[0]) for od in orders]
    cut = np.hstack(cut_)
    tot = len(cut)
    logger.info(f"Number of lines to fit : {tot}")
    # new_linelist = []
    # model2d = np.zeros_like(flx2d)
    # def get_iterable()
    # lines = (line for line in linelist)
    time_start = time.time()
    
    option = 2
    if 'pixel' in scale:
        partial_function_pix = partial(solve_line,
                                       linelist=linelist,
                                       x2d=x2d,
                                       flx2d=flx_norm,
                                       err2d=err_norm,
                                       LSF2d_nm=LSF2d_nm_pix,
                                       ftype='lsf',
                                       scale='pixel',
                                       interpolate=interpolate,
                                       npars=npars)
        if option==1:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial_function_pix, cut)
        elif option==2:
            results = bulk_fit(partial_function_pix)
        print(np.shape(np.asarray(results,dtype="object")))
        new_llist, models = np.transpose(np.asarray(results,dtype="object"))
    
        linelist[cut] = new_llist
    # delete these lines later. These were put in to skip re-doing the entire
    # calculations for pixel when also creating velocity models
    # with FITS(out_filepath,'r') as hdul:
        # linelist = hdul['linelist',version].read()
    # fit for wavelength positions
    if 'velocity' in scale:
        lsf_wavesol = ws.comb_dispersion(linelist, version=701, fittype='lsf', 
                                         npix=npix, 
                                         nord=nbo,
                                         ) 
        
        partial_function_vel= partial(solve_line,
                                       linelist=linelist,
                                       x2d=lsf_wavesol,
                                       flx2d=flx_norm,
                                       err2d=err_norm,
                                       LSF2d_nm=LSF2d_nm_vel,
                                       ftype='lsf',
                                       scale='velocity',
                                       interpolate=interpolate,
                                       npars=npars)
        
        if option==1:
            with multiprocessing.Pool() as pool:
                results = pool.map(partial_function_pix, cut)
        elif option==2:
            results = bulk_fit(partial_function_vel)
        new_llist, models = np.transpose(np.asarray(results,dtype="object"))
        linelist[cut] = new_llist
    worktime = (time.time() - time_start)
    h, m, s  = progress_bar.get_time(worktime)
    logger.info(f"Total time elapsed : {h:02d}h {m:02d}m {s:02d}s")
    
    if save2fits:
        for i,(ll,mod) in enumerate(zip(new_llist,models)):
            od   = ll['order']
            pixl = ll['pixl']
            row  = cut[i]
            with FITS(out_filepath,'rw',clobber=False) as hdu:
                hdu['model_lsf',version].write(np.array(mod),start=[od,pixl])
                hdu['linelist',version].write(np.atleast_1d(ll),firstrow=row)
        # if 'velocity' in scale:
        #     for i,(mod) in enumerate(zip(new_llist,models)):
        #         od   = ll['order']
        #         pixl = ll['pixl']
        #         row  = cut[i]
        #         with FITS(out_filepath,'rw',clobber=False) as hdu:
        #             hdu['model_lsf',version].write(np.array(mod),start=[od,pixl])
        #             # hdu['linelist',version].write(np.atleast_1d(ll),firstrow=row)
           
        with FITS(out_filepath,'rw',clobber=False) as hdu:
            hdu['linelist',version].write_key('ITER', iteration)
            hdu['linelist',version].write_key('SCT', model_scatter)
            hdu['linelist',version].write_key('INTP', interpolate)
    return linelist

class LineSolver(multiprocessing.Process):
    """
    Simple worker.
    """

    def __init__(self, name, function, in_queue, out_queue):
        super(LineSolver, self).__init__()
        self.name = name
        self.function = function
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.logger = logging.getLogger("worker_"+name)

    def run(self):
        while True:
            # grab work; do something to it (+1); then put the result on the output queue
            item = self.in_queue.get()
            # print(f'item after queue.get = {item}')
            result = self.function(item,logger=self.logger)
            self.out_queue.put(result)
            
def solve_line(i,linelist,x2d,flx2d,err2d,LSF2d_nm,ftype='gauss',scale='pix',
                interpolate=False,npars=None,logger=None):
    
    logger = logger if logger is not None else logging.getLogger(__name__)
    
    if scale[:3] =='pix':
        scl = 'pix'
    elif scale[:3]=='vel':
        scl = 'wav'
    line   = linelist[i]
    od     = line['order']
    lpix   = line['pixl']
    rpix   = line['pixr']
    bary   = line['bary']
    # cent   = line[f'{ftype}_{scl}'][1]
    flx1l  = flx2d[od,lpix:rpix]
    x1l    = x2d[od,lpix:rpix]
    err1l  = err2d[od,lpix:rpix]
    npars  = npars if npars is not None else hs.npars
    
    
    try: 
        LSF1d  = LSF2d_nm[od]
    except:
        logger.warning("LSF not found")
        return None
    
    success = False
    pars = "None"
    chisq = np.nan
    
    
    try:
        # logger.info(lsf1d.interpolate(bary))
        # output = hfit.lsf(x1l,flx1l,bkg1l,err1l,lsf1d,
        #                   interpolate=interpolate,
        #                   output_model=True)
        output = hlsfit.line(
            x1l,flx1l,err1l,
            bary=bary,
            LSF1d_obj = LSF1d, # Use forward reference for LSF1d if defined later or imported
            scale = scale,
            npars = 3, # Default to 3 parameters (amp, cen, wid)
            weight = True,
            interpolate = interpolate,
            output_model = True,
            method = 'scipy',
            bounded = True,
            )
        
        success, pars, errs, chisq, chisqnu, integral, model1l = output
    except Exception as e:
        print(f"Exception {e}")
        traceback.print_exc(limit=2, file=sys.stdout)
        # logger.critical("failed")
        pass
    print('line',i,success,pars,chisq)
    
    if not success:
        logger.critical('FAILED TO FIT LINE')
        logger.warning([i,od,bary,x1l,flx1l,err1l])
        # return x1l,flx1l,err1l,LSF1d,interpolate
        # sys.exit()
        pars = np.full(npars,np.nan)
        errs = np.full(npars,np.nan)
        chisq = np.nan
        chisqnu = np.nan
        integral = np.nan
        model1l = np.zeros_like(flx1l)
    # else:
        # pars[1] = pars[1] 
        # new_line = copy.deepcopy(line)
    line[f'lsf_{scl}'][:npars]     = pars[:npars]
    line[f'lsf_{scl}_err'][:npars] = errs[:npars]
    line[f'lsf_{scl}_chisq']       = chisq
    line[f'lsf_{scl}_chisqnu']     = chisqnu
    line[f'lsf_{scl}_integral']    = integral
    
    return line, model1l
    
def shift_anderson(lsfx,lsfy):
    deriv = mathfunc.derivative1d(lsfy,lsfx)
    
    left  = np.where(lsfx==-0.5)[0]
    right = np.where(lsfx==0.5)[0]
    elsf_neg     = lsfy[left]
    elsf_pos     = lsfy[right]
    elsf_der_neg = deriv[left]
    elsf_der_pos = deriv[right]
    shift        = float((elsf_pos-elsf_neg)/(elsf_der_pos-elsf_der_neg))
    return shift
def shift_zeroder(lsfx,lsfy):
    shift = -brentq(mathfunc.derivative_eval,-1,1,args=(lsfy,lsfx))
    return shift    

    
def get_empty_lsf(numsegs,n_data,n_sct,pars=None):
    '''
    Returns an empty array for LSF model.
    
    Args:
    ----
        method:    string ('analytic','spline','gp')
        numsegs:   int, number of segments per range modelled
        n:         int, number of parameters (20 for analytic, 160 for spline, 2 for gp)
        pixcens:   array of pixel centers to save to field 'x'
    '''
    lsf_cont = container.lsf(numsegs,n_data,n_sct,pars)
        
    return lsf_cont

from typing import Tuple, List, Union, Optional, Any

# It's good practice to define these at the module level if they are fixed
# or make them configurable if they might change.
# For now, using the hardcoded value from the original code.
X_ABS_LIMIT = 10.0
PLOT_BIN_EDGES_DEFAULT = np.arange(-8, 8 + 0.5, 0.5)


def clean_input(
    x1s: Union[List, np.ndarray],
    flx1s: Union[List, np.ndarray],
    err1s: Optional[Union[List, np.ndarray]] = None,
    filter_n_elements: Optional[int] = None, # Renamed for clarity from 'filter'
    # xrange: Optional[Tuple[float, float]] = None, # Unused in current active logic
    # binsize: Optional[float] = None, # Unused in current active logic
    sort: bool = True,
    verbose: bool = False,
    plot: bool = False,
    rng_key_input: Optional[Any] = None, # Renamed from rng_key for clarity
) -> Tuple[np.ndarray, ...]:
    """
    Cleans input arrays by removing NaNs, Infs, zero fluxes, and values
    where abs(x) is not less than X_ABS_LIMIT.
    Optionally sorts the data by x-values and/or randomly subsamples it.

    Parameters
    ----------
    x1s : array-like
        X-axis array (e.g., pixels, velocities).
    flx1s : array-like
        Flux array.
    err1s : array-like, optional
        Error array. Default is None.
    filter_n_elements : int, optional
        If provided, data is randomly subsampled. The resulting number of
        elements will be approximately len(x) // filter_n_elements.
        Requires JAX. Default is None (no subsampling).
    xrange : tuple, optional
        Currently UNUSED in the active filtering logic. Intended for x-axis range.
    binsize : float, optional
        Currently UNUSED. Intended for binning.
    sort : bool, optional
        If True (default), sorts the output arrays by the x-axis values.
    verbose : bool, optional
        If True, prints messages about discarded points. Default is False.
    plot : bool, optional
        If True, generates a diagnostic plot. Requires Matplotlib. Default is False.
    rng_key_input : JAX PRNGKey, optional
        A JAX PRNGKey for reproducible random subsampling if `filter_n_elements`
        is used. If None, a default key is used. Default is None.

    Returns
    -------
    Tuple[np.ndarray, ...]
        A tuple containing the cleaned (and optionally sorted/filtered) arrays.
        (x_cleaned, flx_cleaned) or (x_cleaned, flx_cleaned, err_cleaned).
    """
    # --- 1. Initial Conversion and Sorting ---
    x1s_arr = np.ravel(np.asarray(x1s))
    flx1s_arr = np.ravel(np.asarray(flx1s))
    err1s_arr: Optional[np.ndarray] = None

    if x1s_arr.size == 0: # Handle empty input early
        if verbose:
            print("Input x1s array is empty. Returning empty arrays.")
        empty_res = (x1s_arr, flx1s_arr)
        if err1s is not None:
            empty_res += (np.ravel(np.asarray(err1s)),)
        return empty_res

    # Initial sort by x-values 
    initial_sorter = np.argsort(x1s_arr)
    x1s_sorted = x1s_arr[initial_sorter]
    flx1s_sorted = flx1s_arr[initial_sorter]

    if err1s is not None:
        err1s_arr = np.ravel(np.asarray(err1s))[initial_sorter]

    numpts_initial = x1s_sorted.size

    # --- 2. Define Conditions for Good Data Points ---
    conditions = [
        np.isfinite(x1s_sorted),
        np.abs(x1s_sorted) < X_ABS_LIMIT,  # Corrected logic
        np.isfinite(flx1s_sorted),
        flx1s_sorted != 0,
    ]
    if err1s_arr is not None:
        conditions.append(np.isfinite(err1s_arr))

    # Combine conditions: all must be true for a point to be kept
    finite_mask = np.logical_and.reduce(conditions)

    # --- 3. Optional Plotting of Initial State and Filtering ---
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(x1s_sorted, flx1s_sorted, marker='o', label='Original (Sorted)', alpha=0.6, s=30)
        plt.scatter(
            x1s_sorted[~finite_mask], flx1s_sorted[~finite_mask],
            marker='x', c='red', label=f'Rejected (NaN/Inf/0-flux/|x|>={X_ABS_LIMIT})', s=50
        )
        # Vertical lines as in original code
        for edge in PLOT_BIN_EDGES_DEFAULT:
            plt.axvline(edge, ls=':', color='gray', alpha=0.7)
        plt.xlabel("X-values (pixels/velocity)")
        plt.ylabel("Flux")
        plt.title("Data Cleaning: Initial Filtering")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # --- 4. Apply the Mask ---
    x_clean = x1s_sorted[finite_mask]
    flx_clean = flx1s_sorted[finite_mask]

    # Store results in a list to easily append error array
    result_arrays: List[np.ndarray] = [x_clean, flx_clean]
    if err1s_arr is not None:
        err_clean = err1s_arr[finite_mask]
        result_arrays.append(err_clean)

    # --- 5. Optional Subsampling (using JAX) ---
    if filter_n_elements is not None:
        if not isinstance(filter_n_elements, int) or filter_n_elements <= 0:
            if verbose:
                print(f"Warning: Invalid 'filter_n_elements' value ({filter_n_elements}). Skipping subsampling.")
        else:
            current_len = len(result_arrays[0])
            if current_len == 0:
                if verbose:
                    print("No data points after initial cleaning. Skipping subsampling.")
            else:
                try:
                    import jax
                    import jax.random as jr
                    import jax.numpy as jnp # JAX arrays are needed for indexing with JAX choice

                    key_to_use = rng_key_input if rng_key_input is not None else jr.PRNGKey(55873)
                    
                    # Convert to JAX arrays for JAX operations
                    jax_result_arrays = [jnp.asarray(arr) for arr in result_arrays]

                    num_to_select = current_len // filter_n_elements

                    if num_to_select <= 0 :
                        if verbose:
                            print(f"Warning: 'filter_n_elements' ({filter_n_elements}) results in <=0 elements "
                                  f"to select from {current_len} points. Resulting array will be empty or very small.")
                        # Ensure num_to_select isn't negative if current_len is 0
                        num_to_select = max(0, num_to_select)


                    if num_to_select == 0 and current_len > 0:
                         # If we must select 0 items, create empty arrays of the same type
                        if verbose:
                             print(f"Subsampling with filter_n_elements={filter_n_elements} results in 0 elements selected. Returning empty arrays.")
                        result_arrays = [arr_jax[0:0] for arr_jax in jax_result_arrays]

                    elif num_to_select > 0 :
                        # JAX choice requires num_to_select <= current_len
                        if num_to_select > current_len:
                            if verbose:
                                print(f"Warning: num_to_select ({num_to_select}) > current_len ({current_len}). "
                                      f"Selecting all {current_len} elements.")
                            num_to_select = current_len
                        
                        choice_indices = jr.choice(key_to_use, jnp.arange(current_len), 
                                                   shape=(num_to_select,), replace=False)
                        result_arrays = [np.asarray(arr_jax[choice_indices]) for arr_jax in jax_result_arrays]

                except ImportError:
                    if verbose:
                        print("JAX not found. Skipping subsampling ('filter_n_elements').")
                except Exception as e: # Catch other JAX related errors
                    if verbose:
                        print(f"Error during JAX subsampling: {e}. Skipping subsampling.")


    # --- 6. Optional Final Sort ---
    # This sort is useful if subsampling (which shuffles) was performed,
    # or if the user explicitly wants sorted output even if no subsampling occurred.
    if sort:
        if len(result_arrays[0]) > 0: # Can only sort non-empty arrays
            final_sorter = np.argsort(result_arrays[0])
            result_arrays = [arr[final_sorter] for arr in result_arrays]

    # --- 7. Verbose Output ---
    numpts_final = len(result_arrays[0])
    if verbose:
        diff = numpts_initial - numpts_final
        if numpts_initial > 0:
            kept_frac = numpts_final / numpts_initial
            disc_frac = diff / numpts_initial
            print(f"{numpts_final:5d}/{numpts_initial:5d} ({kept_frac:5.2%}) kept; "
                  f"{diff:5d}/{numpts_initial:5d} ({disc_frac:5.2%}) discarded")
        else:
            print("Initial array was empty, 0 points kept.")

    return tuple(result_arrays)

def clean_input_old(x1s,flx1s,err1s=None,filter=None,xrange=None,binsize=None,
                sort=True,verbose=False,plot=False,rng_key=None):
    '''
    Removes infinities, NaN and zeros from the arrays. If sort=True, sorts the
    data by pixel number. If filter is given, removes every nth element from
    the array, where n=filter value.
    

    Parameters
    ----------
    x1s : array-like
        X-axis array, either pixels or velocities.
    flx1s : array-like
        Flux array.
    err1s : array-like, optional
        Error array. The default is None.
    verbose : boolean, optional
        Prints messages. The default is False.
    sort : boolean, optional
        Sorts the array. The default is True.
    filter_every : int, optional
        Filters every int element from the arrays. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    x1s    = np.ravel(x1s)
    sorter = np.argsort(x1s)
    x1s    = x1s[sorter]
    flx1s  = np.ravel(flx1s)[sorter]
    if err1s is not None:
        err1s = np.ravel(err1s)[sorter]
    # remove infinites, nans, zeros and outliers
    arr = np.array([np.isfinite(x1s),
                    np.abs(x1s)<10,
                    np.isfinite(flx1s),
                    np.isfinite(err1s),
                    flx1s!=0,
                    # np.abs(x1s)<8.,
                    ])
    finite  = np.logical_and.reduce(arr)
    cut     = np.where(finite_)[0]
    # optimal binning and outlier detection    
    # counts, bin_edges = bin_optimally(x1s[finite_],minpts=5)
    bin_edges = np.arange(-8,8+0.5,0.5)
    # counts, edges = np.histogram(x1s[finite_],bins=bin_edges)
    # print(counts,bin_edges)
    # idx     = np.digitize(x1s[finite_],bin_edges)
    # identify outliers and remove them
    # keep   = ~hf.is_outlier_from_linear(x1s[finite_],
    #                                     flx1s[finite_],
    #                                     idx,
    #                                     yerrs=err1s[finite_],
    #                                     thresh=3.5)
    # keep  = ~hf.is_outlier_bins(flx1s[finite_],idx,thresh=3.5)
    # finite  = cut[keep]
    # uncomment next line if no outliers should be removed
    # finite  = finite_
    if plot:
        # import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x1s,flx1s,marker='o')
        plt.scatter(x1s[~finite_],flx1s[~finite_],marker='x',c='g')
        # plt.scatter(x1s[cut[~keep]],flx1s[cut[~keep]],marker='x',c='r')
        [plt.axvline(edge,ls=':') for edge in bin_edges]
    numpts  = np.size(flx1s)
    
     
    x      = x1s[finite]
    flx    = flx1s[finite]
    res    = (x,flx)
    if err1s is not None:
        err = np.ravel(err1s)[finite]
        res = res + (err,)
    
    if filter:
        rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(55873)
        shape   = (len(x)//filter,)
        choice  = jax.random.choice(rng_key,np.arange(len(x)),shape,False)
        res = tuple(array[choice] for array in res)
        # res = (array[::filter] for array in res)
    if sort:
        sorter = np.argsort(res[0])
        res = (array[sorter] for array in res)
    res = tuple(res)    
    if verbose:
        diff  = numpts-len(res[0])
        print("{0:5d}/{1:5d} ({2:5.2%}) kept ; ".format(len(res[0]),numpts,
                                                      len(res[0])/numpts) +\
              "{0:5d}/{1:5d} ({2:5.2%}) discarded".format(diff,numpts,
                                                        diff/numpts))
    return tuple(res)   

def lin2log(values,errors):
    '''
    Transforms the values and the errors from linear into log space. 

    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    errors : TYPE
        DESCRIPTION.

    Returns
    -------
    log_values : TYPE
        DESCRIPTION.
    err_log_values : TYPE
        DESCRIPTION.

    '''
    log_values = jnp.log(values)
    err_log_values = jnp.abs(1./values * errors)
    return log_values, err_log_values

def log2lin(values,errors):
    '''
    Transforms the values and the errors from log into linear space. 

    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    errors : TYPE
        DESCRIPTION.

    Returns
    -------
    lin_values : TYPE
        DESCRIPTION.
    err_lin_values : TYPE
        DESCRIPTION.

    '''
    lin_values = jnp.exp(values)
    err_lin_values = jnp.abs(values) * errors
    return lin_values, err_lin_values
    
def get_checksum(X,Y,Y_err,uniqueid=None):
    if uniqueid is not None:
        uniqueid = uniqueid 
    else:
        import random
        import time
        seed= random.seed(time.time())
        uniqueid = random.random()
    _ = np.sum([X,Y,Y_err]) + np.sum(np.atleast_1d(uniqueid))
    return hashlib.md5(_).hexdigest()
    

if __name__ == '__main__':
    # Example data
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, 0])
    data_jax = jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., jnp.nan, 0.])
    data_const = np.array([5., 5., 5., 5., np.nan])
    data_const_jax = jnp.array([5., 5., 5., 5., jnp.nan])
    data_short_jax = jnp.array([1.,2.])


    print("--- SciPy Version ---")
    k_scipy_fisher_biased = get_kurtosis_old(data)
    print(f"SciPy (Fisher, biased): {k_scipy_fisher_biased}") # Default scipy.stats.kurtosis
    
    k_scipy_pearson_biased = get_kurtosis_old(data)
    print(f"SciPy (Pearson, biased): {k_scipy_pearson_biased}")
    
    k_scipy_fisher_unbiased = get_kurtosis_old(data)
    print(f"SciPy (Fisher, unbiased): {k_scipy_fisher_unbiased}")

    k_scipy_const = get_kurtosis_old(data_const)
    print(f"SciPy (Constant data, Pearson, biased): {k_scipy_const}") # Scipy returns -3 for fisher=True, 0 for fisher=False for constant data

    print("\n--- JAX Version ---")
    k_jax_fisher_n1 = get_kurtosis(data_jax, fisher=True, ddof_method='n-1')
    print(f"JAX (Fisher, ddof='n-1'): {k_jax_fisher_n1}") # Closest to original effective calculation + Fisher

    k_jax_pearson_n1 = get_kurtosis(data_jax, fisher=False, ddof_method='n-1')
    print(f"JAX (Pearson, ddof='n-1'): {k_jax_pearson_n1}") # Closest to original effective calculation

    k_jax_fisher_n = get_kurtosis(data_jax, fisher=True, ddof_method='n')
    print(f"JAX (Fisher, ddof='n'): {k_jax_fisher_n}")

    k_jax_pearson_n = get_kurtosis(data_jax, fisher=False, ddof_method='n')
    print(f"JAX (Pearson, ddof='n'): {k_jax_pearson_n}")
    
    k_jax_const = get_kurtosis(data_const_jax, fisher=False, ddof_method='n-1')
    print(f"JAX (Constant data, Pearson, ddof='n-1'): {k_jax_const}") # Should be NaN due to 0/0

    k_jax_short = get_kurtosis(data_short_jax, fisher=False, ddof_method='n-1')
    print(f"JAX (Short data [1,2], Pearson, ddof='n-1'): {k_jax_short}") # Should be NaN because n_eff-1 = 1
    
    popvar_var_old = get_popvar_variance_old(data_jax)
    print(f"OLD popvar variance : {popvar_var_old}")
    
    popvar_var_new = get_popvar_variance(data_jax)
    print(f"NEW popvar variance : {popvar_var_new}")

    # Original function for comparison (if you define `jnp` and `stats`)
    # To run the original, you'd need:
    # import jax.numpy as jnp
    # import scipy.stats as stats
    # def get_kurtosis_original(x,*args,**kwargs): # Renamed original
    #     n        = len(x) # This 'n' is the issue if NaNs are present
    #     # For a fair comparison, n_eff should be used in the correction factor
    #     x_np = np.asarray(x) # For stats.moment
    #     n_eff = np.sum(np.isfinite(x_np))
    #     if n_eff < 2: return np.nan # Cannot do N_eff / (N_eff - 1)
    #
    #     diff     = x-jnp.nanmean(x)
    #     mom4_sam = stats.moment(x_np,moment=4,nan_policy='omit')
    #     mom4_pop = n_eff/(n_eff-1.0)*mom4_sam if n_eff > 1 else np.nan # Corrected n
    #     var_pop  = jnp.nanvar(x,ddof=1)
    #     if var_pop == 0 : return np.nan
    #     return mom4_pop / jnp.power(var_pop,2.)

    # print("\n--- Original (Corrected 'n' for comparison) ---")
    # k_original_corrected_n = get_kurtosis_original(data_jax)
    # print(f"Original (Pearson, ddof='n-1' effectively, corrected 'n'): {k_original_corrected_n}")
    # print(f"This should be close to JAX (Pearson, ddof='n-1'): {k_jax_pearson_n1}")