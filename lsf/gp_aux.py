#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:25:51 2023

@author: dmilakov
"""

parnames_lfc = ['mf_amp','mf_loc','mf_log_sig','mf_const',
                'gp_log_amp','gp_log_scale','log_var_add']
parnames_sct = ['sct_log_amp','sct_log_scale','sct_log_const']
parnames_all = parnames_lfc + parnames_sct

import numpy as np
import harps.lsf.read as hread
import harps.lsf.gp as hlsfgp
import harps.lsf.aux as aux
import harps.lines_aux as laux
import harps.settings as hs
import jax
import jaxopt
import jax.numpy as jnp
from functools import partial 
from scipy.optimize import leastsq
import logging

N_interpolate = 2


def evaluate_GP(GP,y_data,x_test):
    _, cond = GP.condition(y_data,x_test)
    
    mean = cond.mean
    var  = jnp.sqrt(cond.variance)
    
    return mean, var

def build_scatter_GP_from_lsf1s(lsf1s):
    scatter    = hread.scatter_from_lsf1s(lsf1s)
    scatter_gp = hlsfgp.build_scatter_GP(scatter[0],
                                         X=scatter[1],
                                         Y_err=scatter[3])
    return scatter_gp

def evaluate_scatter_GP_from_lsf1s(lsf1s,x_test):
    theta_sct, sct_x, sct_y, sct_yerr  = hread.scatter_from_lsf1s(lsf1s)
    sct_gp = hlsfgp.build_scatter_GP(theta_sct,sct_x,sct_yerr)
   
    return evaluate_GP(sct_gp, sct_y, x_test)


    
def build_LSF_GP_from_lsf1s(lsf1s,return_scatter=False):
    theta_LSF, data_x, data_y, data_yerr = hread.LSF_from_lsf1s(lsf1s)
    scatter  = hread.scatter_from_lsf1s(lsf1s)
    LSF_gp = hlsfgp.build_LSF_GP(theta_LSF, data_x, data_y, data_yerr,
                                 scatter=scatter)
    if return_scatter:
        return LSF_gp, scatter
    else:
        return LSF_gp

def evaluate_LSF_GP_from_lsf1s(lsf1s,x_test):
    theta_LSF, data_x, data_y, data_yerr = hread.LSF_from_lsf1s(lsf1s)
    scatter = hread.scatter_from_lsf1s(lsf1s)
    LSF_gp = hlsfgp.build_LSF_GP(theta_LSF, data_x, data_y, data_yerr,
                                 scatter=scatter)
    
    return evaluate_GP(LSF_gp, data_y, x_test)


def get_likelihood_from_lsf1s(lsf1s):
    theta_LSF, data_x, data_y, data_yerr = hread.LSF_from_lsf1s(lsf1s)
    scatter = hread.scatter_from_lsf1s(lsf1s)
    LSF_gp = hlsfgp.build_LSF_GP(theta_LSF, data_x, data_y, data_yerr,
                                 scatter=scatter)
    return LSF_gp.log_probability(data_y)


def evaluate_lsf1s(lsf1s,x_test):
    return evaluate_LSF_GP_from_lsf1s(lsf1s,x_test)

def get_segment_centres(lsf1d):
    segcens = (lsf1d['ledge']+lsf1d['redge'])/2
    return segcens


def get_segment_weights(center,lsf1d,N=2):
    sorter=np.argsort(lsf1d['segm'])
    segcens   = get_segment_centres(lsf1d[sorter])
    # print(segcens)
    segdist   = jnp.diff(segcens)[0] # assumes equally spaced segment centres
    distances = jnp.abs(center-segcens)
    if N>1:
        used  = distances<segdist*(N-1)
    else:
        used  = distances<segdist/2.
    inv_dist  = 1./distances
    
    segments = jnp.where(used)[0]
    weights  = inv_dist[used]/np.sum(inv_dist[used])
    
    return segments, weights

def helper_calculate_average(list_array,weights,N):
    weights_= jnp.vstack([jnp.full(N,w,dtype='float32') for w in weights])
    average = jnp.average(list_array,axis=0,weights=weights_) 
    return average

def helper_extract_params(theta):
    # print(theta)
    try:
        amp = theta['amp']
        cen = theta['cen']
        wid = jnp.abs(theta['wid'])
    except:
        # amp, cen = theta
        amp, cen, wid = theta
        # wid = jnp.abs(wid)
    return amp,cen,wid
    # return amp,cen
def helper_rescale_errors(theta,x_test,y_err,sct_data,weights):
    
    amp,cen,wid = helper_extract_params(theta)
    x   = jnp.array((x_test-cen) * wid)
    # amp,cen = helper_extract_params(theta)
    # x   = jnp.array((x_test-cen))
    
    S_list = []
    for scatter in sct_data:
        S, S_var = hlsfgp.rescale_errors(scatter,x,y_err,plot=False)
        S_list.append(S)
    average = helper_calculate_average(jnp.array(S_list),
                                       weights,len(x_test))   
     
    return average

def helper_rescale_xarray(theta,x_test):
    '''
    Transforms the x-coordinate array as:
        x_transformed = (x_test - cen)*wid
    
    where cen and wid are contained in the dictionary theta.

    Parameters
    ----------
    theta : dictionary
        Contains parameters {amp, cen, wid}.
    x_test : array-like
        The x-coordinate array.

    Returns
    -------
    x : array
        The transformed x-coordinate array.

    '''
    amp,cen,wid = helper_extract_params(theta)
    x   = jnp.array((x_test-cen) * wid)
    # amp,cen = helper_extract_params(theta)
    # x   = jnp.array((x_test-cen))
    return x
    
def rsd(theta,x_test,y_data,y_err,LSF_data,sct_data,weights):
    model,mod_err = get_model(theta,x_test,LSF_data,weights)
    
    return (y_data-model)/y_err

def residuals_(theta,x_test,y_data,y_err,lsf1d,N):
    bary = jnp.average(x_test,weights=y_data)
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    rescaled_yerr = helper_rescale_errors(theta, x_test, y_err, sct_data, weights)
    return rsd(theta,x_test,y_data,rescaled_yerr,LSF_data,sct_data,weights)
    
@jax.jit
def loss_jitted(*args,**kwargs):
    loss(*args,**kwargs)
    
def loss(theta,x_test,y_data,y_err,lsf1d,N):
    bary = jnp.average(x_test,weights=y_data)
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    rescaled_yerr = helper_rescale_errors(theta, x_test, y_err, sct_data, weights)
    chisq  = jnp.sum(rsd(theta,x_test,y_data,rescaled_yerr,
                         LSF_data,sct_data,weights)**2)
    
    return chisq



def get_params_scipy(lsf1d,x_test,y_data,y_err,interpolate=False):
    
    bary = np.average(x_test,weights=y_data)
    N = N_interpolate if interpolate == True else 1
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    # theta = (np.max(y_data),bary)
    theta = (np.max(y_data),bary,1.)
    rescaled_yerr = helper_rescale_errors(theta, x_test, y_err, 
                                          sct_data, weights)
    pars,pcov,infodict,errmsg,ier = leastsq(rsd,x0=theta,
                                               args=(x_test,y_data,rescaled_yerr,
                                                     LSF_data,sct_data,weights),
                                               full_output=True)
    optpars = dict(
        amp=pars[0],
        cen=pars[1],
        wid=pars[2]
        # wid=1.0
        )
    # logging.info([pcov,infodict,errmsg,ier])
    if ier not in [1, 2, 3, 4]:
        print("Optimal parameters not found: " + errmsg)
        optpars = np.full_like(theta,np.nan)
        pcov = None
        success = False
    else:
        success = True
    dof = len(x_test) - len(optpars)
    chisq= loss(optpars,x_test,y_data,y_err,lsf1d,N)
    return optpars, pcov, chisq, dof

# def get_parameters(lsf1d,x_test,y_data,y_err,interpolate=False):
#     bary = np.average(x_test,weights=y_data)
#     N = N_interpolate if interpolate == True else 1
    
def get_model(theta,x_test,LSF_data,weights):
    amp,cen,wid = helper_extract_params(theta)
    # amp,cen = helper_extract_params(theta)
    x = helper_rescale_xarray(theta, x_test)
    
    model_list = []
    error_list = []
    M = len(x)
    N = len(LSF_data)
    for i in range(N):
        LSF_theta, LSF_x, LSF_y, LSF_yerr = LSF_data[i]
        mean, error = hlsfgp.get_model(x,LSF_x,LSF_y,LSF_yerr,LSF_theta,
                                     scatter=None)
        model_list.append(mean)
        error_list.append(error)
        
    weights_= jnp.vstack([jnp.full(M,w,dtype='float32') \
                          for w in weights])
    
    model_  = jnp.average(jnp.array(model_list),
                          axis=0,
                          weights=weights_
                          ) 
    
    error_ = jnp.sqrt(jnp.sum(jnp.power(jnp.array(error_list),2.),
                              axis=0)
                      )
    # incorrect, for testing:
    normalisation = amp / jnp.max(model_) 
    # correct, to be used; the integral over the area is one
    # normalisation = amp / jnp.sum(model_) 
    # 
    model = model_ * normalisation 
    # logging.info([x_test,model_,model_*normalisation])
    mod_err = error_ * normalisation 
    return model, mod_err    

def get_parameters(lsf1d,x_test,y_data,y_err,interpolate=False):
    # bary = jnp.average(x_test,weights=y_data)
    
    N = N_interpolate if interpolate == True else 1
    bary = jnp.average(x_test,weights=y_data)
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    
    # logging.info([interpolate,N,weights])
    
    theta = dict(
        # amp = jnp.max(y_data)/jnp.sum(y_data),
        # amp = jnp.sum(y_data),
        amp = jnp.max(y_data)*1.0,
        cen = bary,
        wid = 1.0
        )
    lower_bounds = dict(
        # amp = jnp.max(y_data)/jnp.sum(y_data)*0.5,
        # amp = jnp.sum(y_data)*0.95,
        amp = jnp.max(y_data)*0.80,
        cen = bary-0.25,
        wid = 0.98
        )
    upper_bounds = dict(
        # amp = jnp.max(y_data)/jnp.sum(y_data)*1.5,
        # amp = jnp.sum(y_data)*2,
        amp = jnp.max(y_data)*2,
        cen = bary+0.25,
        wid = 1.02,
        )
    bounds = (lower_bounds, upper_bounds)
    rescaled_yerr = helper_rescale_errors(theta, x_test, y_err, 
                                          sct_data, weights)
    
    @jax.jit
    def loss_(theta):
        residuals = rsd(theta,x_test,y_data,y_err,LSF_data,sct_data,weights)
        return jnp.sum(residuals**2)
        
    
    fun1 = loss_
    fun2 = partial(loss,
                  x_test=x_test,
                  y_data=y_data,
                  y_err=rescaled_yerr,
                  lsf1d={segm:val for segm,val in enumerate(lsf1d)},
                  N=N)
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=fun1,method="l-bfgs-b")
    solution = lbfgsb.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
    
    # solver = jaxopt.GradientDescent(fun=partial(fun1,))
    # solution = solver.run(jax.tree_map(jnp.asarray, theta))
    
    optpars = solution.params
    dof = len(x_test) - len(optpars)
    chisq= fun1(optpars)
    pcov = None
    return optpars, pcov, chisq, dof

def get_parameters_opt(lsf1d,x_test,y_data,y_err,interpolate=False):
    import scipy.optimize as optimize
    bary = np.average(x_test,weights=y_data)
    N = N_interpolate if interpolate == True else 1
    # LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    # sct_data,weights = extract_lists('scatter',bary,lsf1d,N=N)
    
    
    theta = dict(
        amp = np.max(y_data)*1.00,
        cen = bary,
        wid = 1.0
        )
    lower_bounds = dict(
        amp = np.max(y_data)*0.8,
        cen = bary-1.0,
        wid = 0.98
        )
    upper_bounds = dict(
        amp = np.max(y_data)*1.2,
        cen = bary+1.0,
        wid = 1.02,
        )
    bounds = (lower_bounds, upper_bounds)
    
    kwargs = dict(
        x_test=np.array(x_test,dtype='float32'),
        y_data=np.array(y_data),
        y_err=np.array(y_err),
        lsf1d=lsf1d,
        N=N
        )
    fun = partial(loss,
                  x_test=x_test,
                  y_data=y_data,
                  y_err=y_err,
                  lsf1d=lsf1d,
                  N=N)
    optimize.minimize(fun, x0=np.fromiter(theta.values(),dtype='float32'), 
                       # args=tuple(kwargs.values()),
                      # args=kwargs,
                      method='BFGS')
    dof = len(x_test) - len(theta)
    chisq= fun(theta)
    pcov = None
    return theta, pcov, chisq,dof 





def extract_LSF_lists(center,lsf1d,N=2):
    return extract_lists('LSF',center,lsf1d,N)

def extract_scatter_lists(center,lsf1d,N=2):
    return extract_lists('scatter',center,lsf1d,N)

def extract_lists(what,center,lsf1d,N=2):
    assert what in ['LSF','scatter']
    segments, weights = get_segment_weights(center,lsf1d,N)
    ll = []
    for segm in segments:
        data = hread.from_lsf1s(lsf1d[segm],what)
        ll.append(data)
    return tuple(ll),tuple(weights)

def get_integral(optpars,x1l,flx1l,lsf1d,interpolate,M=100):
    N = N_interpolate if interpolate == True else 1
    bary   = jnp.average(x1l,weights=flx1l)
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    x_test = jnp.linspace(x1l.min(), x1l.max(),M)
    model, model_err = get_model(optpars,x_test,LSF_data,weights)
    
    return jnp.trapz(model,x_test)

def fit_lsf2line(x1l,flx1l,err1l,lsf1d,interpolate=True,
        output_model=False,plot=False,*args,**kwargs):
    
    bary   = np.average(x1l,weights=flx1l)
    x_test = jnp.array(x1l,dtype=jnp.float32)
    y_data = jnp.array(flx1l,dtype=jnp.float32)
    y_err  = jnp.array(err1l,dtype=jnp.float32)
    
    # optpars, chisq, dof = get_parameters(lsf1d,x_test,y_data,y_err,
    #                                       interpolate=interpolate)
    try:
        # optpars, chisq, dof = get_parameters(lsf1d,x_test,y_data,y_err,
        #                                       interpolate=interpolate)
        optpars, cov_x, chisq, dof = get_params_scipy(lsf1d,x_test,y_data,y_err,
                                                interpolate=interpolate)
        # pcov = np.zeros((3,3))
        # pcov[:cov_x.shape[0],:cov_x.shape[1]] = cov_x
        # pcov[cov_x.shape[0]:,cov_x.shape[1]:] = 1.
        pcov = cov_x
        success = True
    except:
        optpars = dict(amp=np.nan,cen=np.nan,wid=np.nan)
        # optpars = dict(amp=np.nan,cen=np.nan)
        success = False
    
    if success:   
        amp = optpars['amp']
        cen = optpars['cen']
        # wid = 1.0
        wid = optpars['wid']
        
        # chisq   = loss(optpars, x_test,y_data,y_err,
        #                theta_LSF,LSF_x, LSF_y, LSF_yerr)
        # dof  = len(x1l) - (len(optpars)+len(theta_LSF))
        if pcov is not None:
            pcov = pcov*chisq/dof
        else:
            pcov = np.array([[np.inf,0,0],[0,np.inf,0],[0,0,np.inf]])
        #print((3*("{:<3d}")).format(*idx),popt, type(pcov))
    else:
        nparams = 3
        popt = np.full(nparams,np.nan)
        amp, cen, wid = popt
        # amp, cen = popt
        pcov = np.array([[np.inf,0],[0,np.inf,0],[0,0,np.inf]])
        chisq = np.nan
        success=False
        dof  = len(x1l)
    pars    = np.array([amp, cen, wid])
    errors  = np.sqrt(np.diag(pcov))
    chisqnu = chisq/dof
    
    N = N_interpolate if interpolate == True else 1
    LSF_data,weights = extract_lists('LSF',bary,lsf1d,N=N)
    model, model_err = get_model(optpars,x_test,LSF_data,weights)
    
    integral = np.sum(model)
    output_tuple = (success, pars, errors, chisq, chisqnu, integral)
    if plot:
        plot_result(optpars,lsf1d,x1l,flx1l,err1l)
    if output_model:  
        output_tuple = output_tuple + (model,)
    return output_tuple
    
def plot_result(optpars,lsf1d,pix,flux,error,interpolate=True):
    import matplotlib.pyplot as plt
    pix = jnp.array(pix)
    
    # theta_LSF, LSF_x, LSF_y, LSF_yerr = hread.LSF_from_lsf1s(lsf1s)
    # print('plot',*[np.shape(_) for _ in [optpars,LSF_x,LSF_y,LSF_yerr]])
    # model   = return_model(optpars,pix,lsf1s)
    bary = np.average(pix,weights=flux)
    N = N_interpolate if interpolate == True else 1
    LSF_data, weights = extract_lists('LSF',bary,lsf1d,N=N)
    sct_data, weights = extract_lists('scatter',bary,lsf1d,N=N)
    
    
    model,model_err = get_model(optpars,pix,LSF_data,weights)
    rescaled_yerr = helper_rescale_errors(optpars, pix, error,sct_data,weights)
    # full_error    = jnp.sqrt(jnp.sum(jnp.power(jnp.array([model_err,
    #                                                       rescaled_yerr]),2.),
    #                                  axis=0))
    # residuals = rsd(optpars,pix,flux-background,error,LSF_data,sct_data,weights)
    # residuals = ((flux-background)-model)/error
    residuals = ((flux)-model)/rescaled_yerr
    dof = len(pix)-len(optpars)-1
    chisq = np.sum(residuals**2)
    
    # print(model)
    # plotter = Figure2(2,1,height_ratios=[3,1])
    # ax0     = plotter.add_subplot(0,1,0,1)
    # ax1     = plotter.add_subplot(1,2,0,1,sharex=ax0)
    fig, (ax1,ax2) = plt.subplots(2,1)
    
    ax1.set_title('fit.lsf')
    ax1.errorbar(pix,flux,error,label='Flux',drawstyle='steps-mid')
    ax1.plot(pix,model,label='Model',drawstyle='steps-mid')
    ax1.axvline(bary,ls=":")
    ax1.axvline(optpars['cen'],ls="--",c='k')
    ax1.axhline(np.max(flux),ls=":")
    ax1.axhline(optpars['amp'],ls='--',c='k')
    ax1.axvspan(bary-0.5,bary+0.5,alpha=0.1)
    ax1.text(x=0.1,y=0.5,s=r'$\chi^2_\nu$='+f'{chisq/dof:8.2f}',
             transform=ax1.transAxes)
    # ax1.axhspan()
    # ax1.axhspan(np.max(flux-background)*0.9,
    #             np.max(flux-background)*1.1,
    #             alpha=0.1)
    
    
    x_grid = np.linspace(pix.min(),pix.max(),400)
    model_grid,model_grid_err = get_model(optpars,x_grid,LSF_data,weights)
    ax1.plot(x_grid,model_grid,lw=2)
    
    ax2.scatter(pix,residuals,marker='s')
    [ax2.axhline(i,ls='--',lw=1) for i in [-1,0,1]]
    # ax2.set_ylim(-5,5)
    ax1.legend()
    

def format_as_lsf1s(theta_i, n_data=600, n_sct=40):
    """
    Transforms optimized JAX hyperparameters into a structured NumPy LSF model.
    
    Parameters:
    -----------
    theta_i : dict
        A dictionary containing optimized scalars for a single segment.
    n_data : int
        Size of the data buffer (default 600) [11].
    n_sct : int
        Size of the scatter buffer (default 40) [12].
        
    Returns:
    --------
    lsf1s : np.ndarray
        A structured NumPy array containing hyperparameters and metadata slots.
    """
    # Use existing utility to create the structured container [9, 13]
    parnames = parnames_lfc.copy()
    lsf1s = aux._prepare_lsf1s(n_data=n_data, n_sct=n_sct, pars=parnames)
    
    # Map the dictionary values to the structured array fields [14, 15]
    for name in parnames:
        if name in theta_i:
            # JAX device arrays are converted to NumPy scalars here
            lsf1s[name] = float(theta_i[name])
            
    return lsf1s