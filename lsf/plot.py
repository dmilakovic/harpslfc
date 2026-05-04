#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
from harps import functions as hf
from harps import plotter as hplot
from harps.settings import version as hs_version
import harps.lsf.aux as aux
import harps.lsf.gp as lsfgp

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as  plt
import matplotlib.offsetbox as offsetbox
from matplotlib import ticker

import os
import logging
from datetime import datetime

import jax
import jax.numpy as jnp

from scipy.optimize import  curve_fit


import hashlib


topsavedir = os.environ['LSFPLOTDIR'] 
plt.style.use('stamp')
math_ff='cm'
fontsize=10
def plot_variance_modification(ax,X,Y,Y_err,scatter=None,mcmc_posterior=None):
    
    import matplotlib.ticker as ticker
    if ax is not None:
        pass
    else:
        fig, ax = plt.subplots(1)
        
    if scatter is not None:
        params_sct, logvar_x, logvar_y, logvar_error = scatter
        sct_gp = lsfgp.build_scatter_GP(params_sct, logvar_x, logvar_error)
        
    X_grid = jnp.linspace(X.min(),X.max(),200)
    
    
    _, sct_cond_grid = sct_gp.condition(logvar_y,X_grid)
    F_mean_grid  = sct_cond_grid.mean
    F_sigma_grid = jnp.sqrt(sct_cond_grid.variance)
    # print(np.shape(F_mean_grid));sys.exit()
    # f_grid, f_var_grid = transform(X_grid,np.full_like(X_grid,1.),
    #                                 F_mean_grid,F_sigma_grid,
    #                                 sct_gp,logvar_y)
    # logvar_grid_y, logvar_grid_err = aux.lin2log(f_grid, np.sqrt(f_var_grid))
    
    
    linvar_y, linvar_err = aux.log2lin(logvar_y, logvar_error)
    ax.errorbar(logvar_x,logvar_y,
                logvar_error,ls='',capsize=2,marker='s',
                label='Binned residuals')
    
    
    
    ax.plot(X_grid,F_mean_grid,'-C0',label=r'$g(\Delta x;\phi_g)$')
    ax.fill_between(X_grid,
                    F_mean_grid + F_sigma_grid, 
                    F_mean_grid - F_sigma_grid, 
                    color='C0',
                    alpha=0.3)
    if mcmc_posterior is not None:
        quantiles = [16, 50, 84]
        q = np.percentile(mcmc_posterior["g"], quantiles, axis=0)-np.log(2)
        ax.plot(X,q[1],'-C1',label=r'MCMC')
        ax.plot(X,q[0],'--C1')
        ax.plot(X,q[2],'--C1')
    # ax.scatter(X,(S/Y_err)**2.,c='r',s=2)
    ax.set_ylabel(r'$\log(S^2)$')
    # ax.set_yscale('log')
    ax.set_xlabel('Distance from centre (pix)')
    ax.set_ylim(-1.5, 3.5)
    ax.yaxis.tick_left()
    # ax.yaxis.set_ticks_position('left')
    axr = ax.secondary_yaxis('right', functions=(lambda x: np.exp(x), 
                                                  lambda x: np.log(x)))
    axr.yaxis.set_major_locator(ticker.FixedLocator([1,5,10,15,20]))
    axr.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    axr.set_ylabel(r'$S^2$',labelpad=-3)
    # axr.set_yticks([1, 5, 10,20])
    # axr.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.legend(fontsize=fontsize)
    return ax
    
    
def plot_tinygp_model(x,y,y_err,solution,ax,scatter=None,
                      plot_mean_function=True,plot_g=True,plot_gaussian=False):
    X = jnp.array(x)
    Y        = jnp.array(y)
    Y_err    = jnp.array(y_err)
    X_grid = jnp.linspace(X.min(),X.max(),400)
    rsd = {}
    
    gp = lsfgp.build_LSF_GP(solution,X,Y,Y_err,scatter)
    # condition on data and calculate residuals
    _, cond = gp.condition(Y, X)
    if scatter:
        Y_err_, _ = lsfgp.rescale_errors(scatter, X, Y_err)
    else:
        Y_err_ = Y_err
    
    rsd['gp'] = (Y - cond.loc ) / Y_err_
    
    # condition and plot on a fine grid
    _, cond = gp.condition(Y, X_grid)

    mu = cond.loc
    std = np.sqrt(cond.variance)
    ax.errorbar(X, Y, Y_err, marker='.', c='k', label="Data",ls='')
    ax.plot(X_grid, mu, 
            # label='Empirical IP',
            label=r"Empirical IP, $\psi$", 
            lw=2, c='C1',zorder=10)
    for i in [1]:
        ax.fill_between(X_grid, mu + i*std, mu - i*std, color="C1", alpha=0.3)
    

    if plot_mean_function:
        ax.plot(X_grid, jax.vmap(gp.mean_function)(X_grid), c="C2",lw=2,ls='--',
                label=r"Mean function, $\mathbf{m}$")
    # Separate mean and GP
    if plot_g:
        _, cond_nomean = gp.condition(Y, X_grid, include_mean=False)
    
        mu_nomean = cond_nomean.loc #+ soln.params["mf_amps"][0] # second term is for nicer plot
        std_nomean = np.sqrt(cond_nomean.variance)
    
        # plt.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
        
        ax.plot(X_grid, mu_nomean, c='C7', ls=':', lw=2, 
                label=r"$\psi-\mathbf{m}$")
    if plot_gaussian:
        # First panel: Fitted Gaussian
        p0 = (np.max(Y),0,np.std(X),0)
        popt,pcov = curve_fit(hf.gauss4p,X._value,Y._value,sigma=Y_err._value,
                              absolute_sigma=False,p0=p0)
        gauss_Y = hf.gauss4p(X_grid,*popt)
        gauss_Y_err = hf.error_from_covar(hf.gauss4p, popt, pcov, X_grid)
        ax.plot(X_grid, gauss_Y, c="C2",ls='--',
                 label="Gaussian IP",lw=2,zorder=5)
        # for i in [1,3]:
        #     upper = gauss_Y + i*gauss_Y_err
        #     lower = gauss_Y - i*gauss_Y_err
        #     ax.fill_between(X_grid,upper,lower,
        #                         color='C3',alpha=0.3,zorder=0)
    
        gauss_mu_predict  = hf.gauss4p(X, *popt)
        gauss_std_predict = hf.error_from_covar(hf.gauss4p, popt, pcov, X)
        gauss_rsd = (Y-gauss_mu_predict)/gauss_std_predict
        rsd['gauss']=gauss_rsd

    ax.axhline(0,ls=':',lw=1)
    ax.legend(loc=1,fontsize=fontsize)
    ax.set_xlabel('Distance from centre (pix)')
    ax.set_ylabel("Intensity (arbitrary)")
    return rsd

def plot_solution(pix1s,flx1s,err1s,params_LSF,scatter,metadata,shift,
                  save=False,debug=False,logger=None,**kwargs):
    plot_subpix_grid = kwargs.pop('plot_subpix',False)
    plot_model       = kwargs.pop('plot_model',True)
    rasterized       = kwargs.pop('rasterized',False)
    plot_sigma       = kwargs.pop('plot_sigma',[1,3])
    plot_gaussian    = kwargs.pop('plot_gaussian',False)
    
    scale_name       = metadata['scale'][:3]
    
    # Determine the scale of the x-axis, for labeling
    scale_unit   = {'pix':'pix', 'vel':r'kms^{-1}'}
    scale_label  = {'pix':'pix', 'mps':r'ms$^{-1}$', 
                    'kmps':r'kms$^{-1}$', 'pix':'pix'}
    centre_factor = {'pix':1.,
                     'mpix':1000.,
                     'vel':1000.,
                     'ms^{-1}':1000.}
    centre_unit   = {'pix':'pix','vel':'ms^{-1}'}
    
    cu = centre_unit[scale_name]
    cf = centre_factor[cu]
    
    xaxis_unit    = {'pix':'pix','vel':'kmps'}
    xaxis_label   = "Distance from centre " + \
                    f"({scale_label[xaxis_unit[scale_name]]})"
    
    
    # params_LSF = dictionary['solution_LSF']
    N_params   = len(params_LSF)
    full_theta = params_LSF
    
    X        = jnp.array(pix1s)
    Y        = jnp.array(flx1s)
    Y_err    = jnp.array(err1s)
    
    if scatter is not None:
        params_sct, logvar_x, logvar_y, logvar_error = scatter
        # solution_scatter = dictionary['solution_scatter']
        # params_sct = solution_scatter[0]#.params
        # logvar_x   = solution_scatter[1]
        # logvar_y   = solution_scatter[2] 
        # logvar_error = solution_scatter[3] 
        # scatter    = (params_sct,logvar_x,logvar_y,logvar_error)
        N_params = N_params + len(params_sct)
        full_theta.update(params_sct)
        
        gp_scatter = lsfgp.build_scatter_GP(params_sct, logvar_x, logvar_error)
        _, gp_sct_cond = gp_scatter.condition(logvar_y,X)
        # var_scatter = jnp.exp(gp_sct_cond.loc) * Y_err**2
        new_err, new_err_var = lsfgp.rescale_errors(scatter,X,Y_err,plot=False)
        var_data = new_err**2
        
    else:
        scatter = None
        # var_scatter = jnp.zeros_like(Y_err)
    # mode, mode_err = lsfgp.estimate_centre(X,Y,Y_err,params_LSF,
    #                                            scatter=scatter,N=10)
    # model_scatter = dictionary['model_scatter']
    # scatter = "True" if model_scatter==True else None
    # calculate all variances 
    Y_data_err = Y_err
    if scatter is not None:
        S, S_var = lsfgp.rescale_errors(scatter, X, Y_err)
        Y_data_err = S
        
    if debug:
        logger = logger if logger is not None else logging.getLogger(__name__)
        for (p,v) in full_theta.items():
            logger.info(f"{p:<20s} = {v:>8.3f}")
    
    
    
    
    # Condition the model on a dense grid in X
    X_grid = jnp.linspace(X.min(),X.max(),400)
    gp = lsfgp.build_LSF_GP(params_LSF,X,Y,
                      Y_err=Y_err,
                      # Y_err=jnp.zeros_like(Y),
                      scatter=scatter,
                      #scatter=None
                      )
    _, cond = gp.condition(Y, X_grid)
    
    mu = cond.loc
    std = np.sqrt(cond.variance)
    
    plotter = hplot.Figure2(4,2, 
                            figsize=(10,9),
                            # figsize=(7,6.7),
                            left=0.08,right=0.93,
                            bottom=0.08,top=0.98,
                        height_ratios=[2.5,1,1,1],width_ratios=[5,1],
                        enforce_figsize=True)
    
    ax_obs = plotter.add_subplot(0,1,0,1)
    ax_gp  = plotter.add_subplot(1,2,0,1,sharex=ax_obs)
    ax_var = plotter.add_subplot(2,3,0,1,sharex=ax_obs)
    ax_rsd = plotter.add_subplot(3,4,0,1,sharex=ax_obs)
    ax_hst = plotter.add_subplot(3,4,1,2)
    
    for ax in plotter.axes:
        ax.axhline(0,ls=':',c='grey',zorder=5)
    
    # First panel: data, full model and the mean function
    # First panel: Y_err is original
    ax_obs.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
    ax_obs.plot(X_grid, mu, label=r"Empirical IP, $\psi(x;\theta,\phi)$",
                c='C1',lw=2,zorder=5)
    for i in np.atleast_1d(plot_sigma):
        ax_obs.fill_between(X_grid, mu + i*std, mu - i*std, color="C1", 
                            alpha=0.15)
    ax_obs.plot(X_grid, jax.vmap(gp.mean_function)(X_grid), c="C2",ls='--',
             label=r"Mean function, ${\bf m}(x)$",lw=2,zorder=4)   
    
    
    # First panel: random samples from GP posterior 
    # rng_key = jax.random.PRNGKey(55873)
    # sampled_f = cond.sample(rng_key,(20,))
    # for f in sampled_f:
    #     ax_obs.plot(X_grid,f,c='C0',lw=0.5)
    
    # Second panel: the Gaussian process + residuals from Gaussian model
    _, cond_nomean = gp.condition(Y, X_grid, include_mean=False)
    
    mu_nomean = cond_nomean.loc 
    std_nomean = np.sqrt(cond_nomean.variance)
    ax_gp.plot(X_grid, mu_nomean, c='C1', ls='-', 
               label=r"$(\psi - \mathbf{m})(x)$",zorder=5)
    y2lims = [100,-100] # saves y limits for the middle panel
    for i in np.atleast_1d(plot_sigma):
        upper = mu_nomean + i*std_nomean
        lower = mu_nomean - i*std_nomean
        if np.max(lower)<y2lims[0]:
            y2lims[0]=np.min(lower)
        if np.max(upper)>y2lims[1]:
            y2lims[1]=np.max(upper)
        ax_gp.fill_between(X_grid, upper, lower,
                         color="C1", alpha=0.3,zorder=0)
    # Second panel: random samples from GP posterior , no mean function
    # rng_key = jax.random.PRNGKey(55873)
    # sampled_f = cond_nomean.sample(rng_key,(20,))
    # for f in sampled_f:
    #     ax_gp.plot(X_grid,f,c='C1',lw=0.5,alpha=0.4)
        
    # Second panel: residuals from gaussian model
    # _, cond_nomean_predict = gp.condition(Y, X, include_mean=False)
    # std_nomean_predict = np.sqrt(cond_nomean_predict.variance)
    cond_nomean_predict, std_nomean_predict = gp.predict(Y,X,
                                                         include_mean=False,
                                                         return_var=True)
    Y_gauss_rsd = Y - jax.vmap(gp.mean_function)(X)
    Y_gauss_err = Y_err
    # Y_gauss_err = jnp.sqrt(Y_err**2 + std_nomean_predict**2)
    ax_gp.errorbar(X, Y_gauss_rsd, Y_err, marker='.',ms=4,
                   color='k',ls='',capsize=2,label=r"data $- {\bf m}(x)$")
    if scatter is not None:
        ax_gp.errorbar(X, Y_gauss_rsd, Y_data_err, marker='',ms=0,
                       color='grey',ls='',alpha=0.5,capsize=4,zorder=0)
    
    # Third panel: variances
    ax_var = plot_variances(ax_var, X,Y,Y_err,params_LSF,scatter=scatter,
                            yscale='log')
    # plotter.ticks(2,'y',ticknum=3,)
    ax_var.legend(loc='upper left',bbox_to_anchor=(1.02, 0.9),
                  fontsize=fontsize)
    
    # Fourth left panel: normalised residuals for Gaussian Process
    _, cond_predict = gp.condition(Y, X, include_mean=True)
    mu_model = cond_predict.loc # second term is for nicer plot
    
    
    Y_mod_err  = np.sqrt(cond_predict.variance)
    Y_tot_err = Y_data_err
    # Y_tot_err  = jnp.sqrt(np.sum(np.power([Y_data_err,Y_mod_err],2.),axis=0))
    # rsd        = (Y - Y_pred)/Y_err
    # # rsd        = (Y - Y_pred)/Y_tot_err
    rsd        = lsfgp.get_residuals(X, Y, Y_tot_err, params_LSF)
    # # Y_tot = Y_err
    # # Y_tot = jnp.sqrt(var_predict)
    # rsd = (mu_model-Y)/Y_tot
    # snr = Y/Y_tot
    ax_rsd.scatter(X,rsd,marker='.',c='k')
    # ax_obs.plot(X,snr)
    
    rsd_to_plot = [rsd]
    # ---------- Single gaussian fit (optional):
    if plot_gaussian:
        # First panel: Fitted Gaussian
        p0 = (np.max(Y),0,np.std(X),0)
        popt,pcov = curve_fit(hf.gauss4p,X._value,Y._value,sigma=Y_err._value,
                              absolute_sigma=False,p0=p0)
        gauss_Y = hf.gauss4p(X_grid,*popt)
        gauss_Y_err = hf.error_from_covar(hf.gauss4p, popt, pcov, X_grid)
        ax_obs.plot(X_grid, gauss_Y, c="C3",ls=':',
                 label="Gaussian IP",lw=2,zorder=3)
        for i in [1,3]:
            upper = gauss_Y + i*gauss_Y_err
            lower = gauss_Y - i*gauss_Y_err
            ax_obs.fill_between(X_grid,upper,lower,
                                color='C3',alpha=0.3,zorder=0)
    
        # Fourth left panel: normalised residuals for a single Gaussian function
        gauss_mu_predict  = hf.gauss4p(X, *popt)
        gauss_std_predict = hf.error_from_covar(hf.gauss4p, popt, pcov, X)
        gauss_rsd = (Y-gauss_mu_predict)/gauss_std_predict
        ax_rsd.scatter(X,gauss_rsd,marker='.',c='C3')    
        rsd_to_plot.append(gauss_rsd)
    
    # ---------- Histogram of residuals
    ax3_ylims = ax_rsd.get_ylim()
    colors = ['C0','C3']
    
    for i,rsd_arr in enumerate(rsd_to_plot):
        y_pos = 0.9-0.12*i
        color = colors[i]
        median,upper,lower=plot_histogram(ax_hst, rsd_arr, color, y_pos,
                                          range=ax3_ylims)
    
        for ax in [ax_rsd,ax_hst]:
            # [ax.axhline(val,ls=(0,(10,5,10,5)),color=color,lw=0.8) for val in [upper,lower]]
            [ax.axhspan(-1,1,color=color,alpha=0.2)]
    ax_hst.set_ylim(ax3_ylims)
    chisq = np.sum(rsd**2)
    dof   = len(rsd)-N_params
    aicc  = chisq + 2*len(Y)*N_params / (len(Y)-N_params-1)
    items = ['A','mu','sigma','y0','a','l','logvar','logL','N','nu',
             'chisq','chisqnu','shift']
    labels = dict(
                  A='A',
                  mu=r'\mu',
                  sigma=r'\tau', 
                  y0=r'y_0',
                  a=r'a', 
                  l=r'l', 
                  logvar= r'\log(\sigma_0)',
                  #logyerr= r'log$_{10}$(<Y_err>)',
                  N='N',
                  nu=r'\nu',
                  chisq=r'\chi^2',
                  chisqnu=r'\chi^2_\nu',
                  #AICc='AICc',
                  logL=r'\log(\mathcal{L})',
                  mode=r'\mathrm{Mode}',
                  mode_err='error',
                  shift=r'\mathrm{Shift}'
              )
    
    values = dict(
        A=params_LSF['mf_amp'], 
        mu=params_LSF['mf_loc'],
        sigma=np.exp(params_LSF['mf_log_sig']), 
        y0=params_LSF['mf_const'],
        a=np.exp(params_LSF['gp_log_amp']), 
        l=np.exp(params_LSF['gp_log_scale']), 
        logvar=params_LSF['log_var_add'],
        logyerr=np.log10(np.mean(Y_err**2)),
        N=len(Y),
        nu=dof,
        chisq=chisq,
        chisqnu=chisq/dof,
        AICc=aicc,
        logL=-lsfgp.loss_LSF(params_LSF,X,Y,Y_err,scatter),
        shift=shift*cf,
        # mode=mode*cf,
        # mode_err=mode_err*cf,
        # 'Gaus centre':None,
        )
    units = dict(
        A='arb.', 
        mu=scale_unit[scale_name],
        sigma=scale_unit[scale_name], 
        y0='arb.',
        a='arb.',
        l=scale_unit[scale_name], 
        logvar='',
        logyerr='',
        N='',
        nu='',
        chisq='',
        chisqnu='',
        AICc='',
        logL='',
        shift=cu,
        mode='',#centre_unit[scale_name],
        mode_err=cu,
        )
    formats = dict(
        A='9.3f', 
        mu='9.3f',
        sigma='9.3f', 
        y0='9.3f',
        a='9.3f', 
        l='9.3f', 
        logvar='9.3f',
        logyerr='9.3f',
        N='5d',
        nu='5d',
        chisq='9.3f',
        chisqnu='9.3f',
        AICc='9.3f',
        logL='9.3f',
        shift='+9.3f',
        mode='+9.1f',
        mode_err='9.1f',
        )
    if plot_gaussian:
        items.append('mu_gauss')
        labels.update({'mu_gauss':'Gauss centre'})
        values.update({'mu_gauss':popt[1]*centre_factor[scale_name]})
        units.update({'mu_gauss':centre_unit[scale_name]})
        formats.update({'mu_gauss':'+9.3f'})
    text_list = [r'\begin{eqnarray*} ']
    for i,key in enumerate(items):
        l = labels[key]
        # print(i,l)
        v = values[key]
        m = formats[key]
        u = units[key]
        if key!='mode_err':
            text = (f"{l:<10} &=& {v:>{m}}")
        else:
            text = (f"&&\pm{v:>{m}}")
        if len(u)>0:
            text+=r'\;(\mathrm{'+ f'{u}' + r'})'
        
        # if l==r'\mathrm{Mode}':
        #     text+=r'\pm '
        # else:
        text+=r'\\ '
            
        # else:
        #     pass
        text_list.append(text)
        # ax_obs.text(1.04,0.9-i*0.08,text,
        #           horizontalalignment='left',
        #           verticalalignment='top', 
        #           transform=ax_obs.transAxes)
        # print(text)
    text_list.append('\end{eqnarray*}')
    text_aligned = ''.join(text_list)
    if debug:
        # logger = logger if logger is not None else logging.getLogger(__name__)
        logger.info(text_aligned)
    with mpl.rc_context({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"}):
        
        ax_obs.text(1.01,0.9,text_aligned,fontsize=11,
                    transform=ax_obs.transAxes)
    # ob = offsetbox.AnchoredText(text, pad=1, loc=6, prop=dict(size=8))
    # ob.patch.set(alpha=0.85)
    # ax_obs.add_artist(ob)
    
    Xlarger = np.max([X.max(), np.abs(X.min())])
    Xlimit = 1.025*Xlarger
    ax_obs.set_xlim(-Xlimit, Xlimit)
    
    try:
        combined_values = np.concatenate([np.abs(np.asarray(y2lims)).ravel(),
                                          np.asarray(Y_gauss_rsd).ravel()])
    except ValueError as e:
        # Handle cases where one or both might be empty or have incompatible shapes
        # For example, if one is empty, just use the other.
        print(f"Error during concatenation for y2lim: {e}")
        list_for_max = []
        if hasattr(y2lims, '__len__') and len(y2lims) > 0:
            list_for_max.extend(np.abs(np.asarray(y2lims)).ravel())
        if hasattr(Y_gauss_rsd, '__len__') and len(Y_gauss_rsd) > 0:
            list_for_max.extend(np.asarray(Y_gauss_rsd).ravel())
        
        if not list_for_max: # If both were empty or unsuitable
            y2lim = 1.0 # Default sensible limit if no data
            print("Warning: y2lims and Y_gauss_rsd result in no valid data for y2lim. Setting to 1.0")
        else:
            combined_values = np.array(list_for_max)
    
    
    if combined_values.size == 0:
        y2lim = 0.5 # Default if no valid numeric data
        print("Warning: No valid numeric data to determine y2lim. Setting to 1.0.")
    else:
        # First, handle potential Infs separately if nanmax doesn't do what you want with them
        # np.nanmax will return inf if inf is present.
        # Filter out NaNs first for np.max if you need specific inf handling,
        # or just use np.nanmax and then check its result.
    
        finite_values = combined_values[np.isfinite(combined_values)]
        if finite_values.size == 0:
            # All values were NaN or Inf, or the array was empty after concatenation
            # Check if Infs were present in the original combined_values
            if np.isinf(combined_values).any():
                y2lim = np.inf # Propagate Inf if that's desired, otherwise handle
                print("Warning: y2lim calculation resulted in Inf due to Inf values in input.")
            else:
                y2lim = 1.0 # Default if only NaNs or empty
                print("Warning: y2lim calculation resulted in only NaNs or empty. Setting to 1.0.")
        else:
            y2lim = np.max(finite_values) # Max of finite values
    
    # Final check on y2lim itself before setting axis limits
    if not np.isfinite(y2lim) or y2lim == 0: # Also handle if max finite is 0
        print(f"Warning: Calculated y2lim ({y2lim}) is not finite or is zero. Setting a default y-limit (e.g., 1.0).")
        y2lim = 0.5 # Or another sensible default like a small positive number

    
    ax_gp.set_ylim(-1.5*y2lim,1.5*y2lim)
    # ax1.set_ylim(-0.05,0.25)
    plotter.axes[-1].set_xlabel("x "+r'(kms$^{-1}$)')
    ax_obs.set_ylabel(r"Intensity $I$ (arbitrary)")
    ax_gp.set_ylabel(r"$I$ (arb.)")
    ax_rsd.set_ylabel("Residuals "+r"($\sigma$)")
    ax_rsd.set_xlabel(f"{xaxis_label}")
    ax_hst.set_yticklabels([])
    ax_hst.set_xlabel(r'\#')
    for ax in [ax_obs,ax_gp,ax_var]:
        ax.tick_params(labelbottom=False)
    
    _ = ax_obs.legend(fontsize=fontsize)
    _ = ax_gp.legend(loc='upper left',fontsize=fontsize)
    
    plotter.figure.align_ylabels()
    
    if save:
        
        figmetadata=dict(
            Author = 'Dinko Milakovic',
            Creator = "harps.lsf.plot",
            Title = f"Order = {metadata['order']}; "+\
                f"pixl = {metadata['ledge']}; "+\
                f"pixr = {metadata['redge']}; "+\
                f"Scale = {metadata['scale']}; "+\
                f"Model scatter = {metadata['model_scatter']}; " +\
                f"Iteration = {metadata['iteration']}",
            
            )
        try:
            checksum = metadata['checksum']
        except:
            checksum = aux.get_checksum(X,Y,Y_err)
        name = get_figure_name(metadata,scatter=scatter)
        today = datetime.today().strftime('%Y-%m-%d')
        savedir = os.path.join(*[topsavedir,hs_version,today])
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        figname = os.path.join(*[savedir,
                                 f"{scale_name}_{name}_{checksum}.pdf"])
        plotter.save(figname,metadata=figmetadata)
        _ = plt.close(plotter.figure)  
        del(plotter)
        return None
    else:
        return plotter
def plot_histogram(ax,rsd_arr,color,text_yposition,range=None):
    # Fourth right panel: a histogram of normalised residuals
    n, bins, _ = ax.hist(np.ravel(rsd_arr),bins=10,range=range,
                          color=color,orientation='horizontal',
                          histtype='step',lw=2)
    # Fourth right panel: print and plot horizontal lines for quantiles 
    median    = np.median(np.ravel(rsd_arr))
    quantiles = np.quantile(np.ravel(rsd_arr),[0.05,0.95])
    lower, upper = quantiles - median
    ax.text(0.85,text_yposition,#
             r'${{{0:+3.1f}}}_{{{1:+3.1f}}}^{{{2:+3.1f}}}$'.format(median,lower,upper),
             horizontalalignment='right',
             verticalalignment='center', 
             transform=ax.transAxes, 
              fontsize=fontsize,
             color=color)
    # [ax.axhline(val,ls=(0,(10,5,10,5)),color='grey',lw=0.8) for val in [upper,lower]]
    return median,upper,lower 

def plot_variances(ax, X,Y,Y_err,theta,scatter=None,yscale='log'):
    var_data = jnp.power(Y_err,2)
    var_add = jnp.broadcast_to(jnp.exp(theta['log_var_add']),Y_err.shape)
    var_tot = var_add + var_data
    
    gp = lsfgp.build_LSF_GP(theta,X,Y,Y_err,scatter=scatter)
    _, cond = gp.condition(Y,X,include_mean=True)
    var_mod = cond.variance
    if scatter is not None:
        S, S_var = lsfgp.rescale_errors(scatter, X, Y_err)
        var_new   = jnp.power(S,2.)    
        var_new_err = 2*S*jnp.sqrt(S_var)   
        var_tot   = var_add + var_new
        
            
        
        ax.scatter(X,var_new,label='Modified data',#+r'$g \sigma_{\hat{\boldsymbol{\psi}}}$',
                   marker='.',s=8,c='C5')
        
        ax.fill_between(X,
                        var_new+var_new_err,
                        var_new-var_new_err,
                        color='C5',alpha=0.3)
        
    ax.scatter(X,var_data,label='Original data',#+r'$\sigma_{\hat{\boldsymbol{\psi}}}$',
               marker='.',c='grey',s=6)
    # ax.plot(X,var_add,label=r'$\sigma_0$',ls=(0,(1,2,1,2)),c='C3')
    ax.scatter(X,var_tot,label=r'$\sigma_{tot}$',s=6,c='C0')
    ax.plot(X,var_mod,label=r'${\rm diag}\;{\bf K}_{i,j}$',ls='-',c='C1',lw=1.)
    ax.legend(fontsize=fontsize)
    yscale_kwargs={}
    if yscale=='log':
        yscale_kwargs.update(dict(nonpositive='clip'))
    ax.set_yscale(yscale,**yscale_kwargs)
    ax.set_ylabel("Variances")
    return ax

def plot_analytic_lsf(values,ax,title=None,saveto=None,**kwargs):
    nitems = len(values.shape)
    npts   = 500
    x = np.linspace(-6,6,npts)
    plot_components=kwargs.pop('plot_components',False)
    if nitems>0:
        numvals = len(values)
        colors = plt.cm.jet(np.linspace(0,1,numvals))
        for j,item in enumerate(values):
            y = hf.gaussP(x,*item['pars'])
            ax.plot(x,y,lw=2,c=colors[j])
            if plot_components:
                ylist = hf.gaussP(x,*item['pars'],return_components=True)
                [ax.plot(x,y_,lw=0.6,ls='--',c=colors[j]) for y_ in ylist]
    else:            
        y = hf.gaussP(x,*values['pars'])
        ax.plot(x,y,lw=2)
    return ax
    
def plot_gp_lsf(values,ax,title=None,saveto=None,**kwargs):
    nitems = len(values.shape)
    npts   = 500
    x      = np.linspace(-10,10,npts)
    if nitems>0:
        numvals = len(values)
        colors = plt.cm.jet(np.linspace(0,1,numvals))
        for j, item in enumerate(values):
            pass

    return ax

def plot_scatter(scatter,x_test):
    theta_scatter, logvar_x, logvar_y, logvar_err = scatter
    sct_gp        = lsfgp.build_scatter_GP(theta_scatter,logvar_x,logvar_err)
    _, sct_cond   = sct_gp.condition(logvar_y,logvar_x)
    
    linvar_y, linvar_err = aux.log2lin(logvar_y, logvar_err)
    
    
    F_mean  = sct_cond.mean
    F_sigma = jnp.sqrt(sct_cond.variance)
    
    # S, S_var = transform(X,Y_err,F_mean,F_sigma)
    # if plot:
        # X_grid = jnp.linspace(X.min(),X.max(),200)
        
    # x_test = logvar_x
    _, sct_cond_grid = sct_gp.condition(logvar_y,x_test)
    F_mean_grid  = sct_cond_grid.mean
    F_sigma_grid = jnp.sqrt(sct_cond_grid.variance)
    # print(np.shape(F_mean_grid));sys.exit()
    # f_grid, f_var_grid = transform(x_test,np.ones_like(x_test),
                                    # F_mean_grid,F_sigma_grid)
    fig, ax = plt.subplots(1,1)
    ax.errorbar(logvar_x,logvar_y,
                logvar_err,ls='',capsize=2,marker='x',
                label='F(x) bins')
    ax.plot(x_test,F_mean_grid,'-k',label='F(x) GP')
    ax.fill_between(x_test,
                    F_mean_grid+F_sigma_grid,
                    F_mean_grid-F_sigma_grid,
                    color='k',
                    alpha=0.3)
    # ax.scatter(X,(S/Y_err)**2.,c='r',s=2)
    ax.legend(fontsize=fontsize)
    
def get_figure_name(metadata,scatter=None):
    
    keys = ['order','segment','scale','model_scatter','iteration','interpolate']
    text = dict(
        order='od',
        segment='seg',
        scale='scale',
        model_scatter='scatter',
        iteration='iter',
        interpolate='interp'
        )
    figname = ""
    for key in keys:
        try:
            val = metadata[key]
            if type(val)==int:
                value = f'{val:02d}'
            else:
                value = val
            
            if key=='model_scatter':
                value = True if scatter is not None else False
            figname = figname + f"{text[key]}={value}_"
        except:
            continue
    return figname
    
    # f"order_segment={metadata['order']}_{metadata['segment']}_"+\
        # f"{metadata['scale']}_scatter={metadata['model_scatter']}"
        
def plot_numerical_model(ax,nummodel,*args,**kwargs):
    if ax is not None:
        pass
    else:
        figure = hplot.Figure2(1,1,figsize=(5,4))
        ax = figure.add_subplot(0,1,0,1)
    x = nummodel['x']
    y = nummodel['y']
    numseg_sent,npts = np.shape(x)
    if numseg_sent==1:
        x = x[0]
        y = y[0]
        ax.plot(x,y,*args,**kwargs)
    if numseg_sent>5:
        colors = plt.cm.jet(np.linspace(0, 1, numseg_sent))
        for i,(x_,y_) in enumerate(zip(x,y)):
            ax.plot(x_,y_,color=colors[i],*args,**kwargs,label=f'Segment {i+1}')
    
    return ax
    
        