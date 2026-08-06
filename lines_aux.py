#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:46:49 2023

@author: dmilakov
"""
import numpy as np
import matplotlib.pyplot as plt

def prepare_data(flx,env,bkg,subbkg,divenv):
    assert np.shape(flx)==np.shape(env)==np.shape(bkg)
    f        = flx
    var_data = flx
    
    # var_env  = env
    var_env = var_data
    
    if subbkg:
        b  = bkg
        var_bkg  = bkg
        e  = env - bkg
    else:
        b  = np.zeros_like(f)   
        var_bkg = np.zeros_like(f)  
        e  = env
    if divenv:
        data_norm = (f-b) / e
    else:
        data_norm = f - b

    if not subbkg and not divenv:
        var = var_data
        bkg_norm = bkg
    elif subbkg and not divenv:
        var = np.abs(flx) + np.abs(bkg)
        bkg_norm  = np.zeros_like(bkg)
    elif subbkg and divenv:
        var = 1./e**2 * var_data + ((f-e)/e**2)**2 * var_bkg + \
            ((b-f)/e**2)**2 * var_env
        bkg_norm  = np.zeros_like(bkg)
    elif not subbkg and divenv:
        var = var_data/e**2 + f**2/e**4 * var_env
        bkg_norm = bkg/e
    error_norm   =  np.sqrt(var)
    
    # plt.figure()
    # # plt.plot(f/np.sqrt(var_data),label='data')
    # # plt.plot(e/np.sqrt(var_env),label='env')
    # # plt.plot(b/np.sqrt(var_bkg),label='bkg')
    # plt.plot(data/data_error,label='final')
    # plt.legend()
    return data_norm, error_norm, bkg_norm

def quotient_variance(x_mean,x_var,y_mean,y_var,corr=None):
    '''
    Returns the quotient variance:
        variance = (x_mean/y_mean)^2 * ((x_var/x_mean)^2 + (y_var/y_mean)^2)
        

    Parameters
    ----------
    x_mean : TYPE
        DESCRIPTION.
    y_mean : TYPE
        DESCRIPTION.
    x_var : TYPE
        DESCRIPTION.
    y_var : TYPE
        DESCRIPTION.
    corr : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    variance

    '''
    
    return (x_mean/y_mean)**2 * ((x_var/x_mean)**2 + (y_var/y_mean)**2)