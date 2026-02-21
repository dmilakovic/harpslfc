#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:45:47 2019

@author: dmilakov
"""
import harps.functions as hf
from   harps.core import np, plt
import harps.lsf.aux as aux
import jax
import jax.numpy as jnp
import jaxopt
from   tinygp import kernels, GaussianProcess, noise
from   functools import partial 
import gc
from   scipy.optimize import curve_fit, brentq
import scipy.interpolate as interpolate
import logging
import ray


@jax.jit
def loss_LSF(theta,X,Y,Y_err,scatter=None):
    gp = build_LSF_GP(theta,X,Y,Y_err,scatter)
    return -gp.log_probability(Y)
@jax.jit
def loss_scatter(theta,X,Y,Y_err):
    gp = build_scatter_GP(theta,X,Y_err)
    return -gp.log_probability(Y)

def _check_arrays(*args):
    conditions = np.array([np.isfinite(_) for _ in args])
    finite  = np.logical_and.reduce(conditions)
    cut     = np.where(finite)[0]
    return cut


def get_lsf_bounds(X, Y, Y_err):
    """
    Calculates boxed constraints for GP hyperparameter optimization 
    based on the data and an initial Gaussian fit.
    """

    # Initial guess for the mean function (A, mu, sigma, offset)
    p0 = (np.max(Y), 0, np.std(X), 0)
    cut = _check_arrays(X, Y, Y_err)
    if len(cut)<2:
        logger.warning('Data is wrong!')
        print(X,Y,Y_err)
    X = X[cut]
    Y = Y[cut]
    Y_err = Y_err[cut]
    print('GET_LSF_BOUNDS data cleaning ok')
    # Perform a preliminary Gaussian fit to establish constraint center
    # Logic follows gp.txt [3]
    popt, pcov = curve_fit(hf.gauss4p, X, Y, sigma=Y_err, absolute_sigma=False, p0=p0)
    print("GET_LSF_BOUNDS Gaussian fit ok")
    perr = np.sqrt(np.diag(pcov))
    
    # The 'kappa' factor determines the width of the search space 
    # around the initial fit. gp.txt uses 5 [1].
    kappa = 5
    
    lower_bounds = dict(
        mf_amp       = popt[0]-kappa*perr[0],
        mf_loc       = popt[1]-kappa*perr[1],
        mf_log_sig   = np.log(popt[2]-kappa*perr[2]),
        mf_const     = popt[3]-kappa*perr[3],
        gp_log_amp   = -4., #popt[0]/3.-kappa*perr[0],
        gp_log_scale = -1.,
        log_var_add  = -15.,
    )
    upper_bounds = dict(
        mf_amp       = popt[0]+kappa*perr[0],
        mf_loc       = popt[1]+kappa*perr[1],
        mf_log_sig   = np.log(popt[2]+kappa*perr[2]),
        mf_const     = popt[3]+kappa*perr[3],
        gp_log_amp   = 4., # popt[0]/3.+kappa*perr[0],
        gp_log_scale = 1.,
        log_var_add  = 1.5,
    )
    
    
    return (lower_bounds, upper_bounds)

@ray.remote
def run_lsf_optimization_task(theta_start, X, Y, Y_err, scatter, bounds):
    """
    Ray Task: Performs a single L-BFGS-B optimization.
    JAX JIT is utilized within the loss_LSF function.
    """
    # Import within task to ensure availability on remote workers
    from harps.lsf.gp import loss_LSF 
    
    lbfgsb = jaxopt.ScipyBoundedMinimize(
        fun=partial(loss_LSF, X=X, Y=Y, Y_err=Y_err, scatter=scatter),
        method="l-bfgs-b"
    )
    
    # Run the optimization
    try:
        solution = lbfgsb.run(theta_start, bounds=bounds)
        # Return the params and the final loss value for comparison
        return solution.params, solution.state.fun_val
    except Exception as e:
        return None, jnp.inf
    
def train_LSF_multistart_ray(X, Y, Y_err, scatter=None, num_starts=4):
    """
    Coordinating function for multi-start GP training.
    """
    
    # 1. Put data in Ray Object Store to prevent redundant copying
    X_ref = ray.put(X)
    Y_ref = ray.put(Y)
    Y_err_ref = ray.put(Y_err)
    
    # 2. Generate a list of diverse starting guesses (modifying lengths/amps)
    starts = generate_starting_guesses(X, Y, Y_err, num_starts)
    bounds = get_lsf_bounds(X, Y, Y_err) 

    # 3. Launch parallel optimization tasks
    futures = [
        run_lsf_optimization_task.remote(s, X_ref, Y_ref, Y_err_ref, scatter, bounds) 
        for s in starts
    ]
    
    # 4. Collect results and choose the one with the minimum negative log-likelihood
    results = ray.get(futures)
    
    # Filter out failures
    valid_results = [res for res in results if res is not None]
    if not valid_results:
        raise RuntimeError("All hyperparameter optimization starts failed.")
        
    best_params = min(valid_results, key=lambda x: x[5])
    return best_params

def generate_starting_guesses(X, Y, Y_err, n):
    """
    Generates variations of the initial hyperparameters to explore
    different regions of the likelihood surface.
    """
    
    # Get the standard initial guess from existing logic
    base_theta = train_LSF_tinygp(X, Y, Y_err, return_only_init=True)
    
    guesses = []
    # Seed 0 is always the base guess
    guesses.append(base_theta)
    
    # Add variations for other starts
    # We vary gp_log_scale (correlation length) and gp_log_amp
    for i in range(1, n):
        new_theta = base_theta.copy()
        # Perturb length-scale log-space by +/- 0.5
        new_theta['gp_log_scale'] += np.random.uniform(-0.5, 0.5)
        # Perturb GP amplitude
        new_theta['gp_log_amp'] += np.random.uniform(-1.0, 1.0)
        guesses.append(new_theta)
        
    return guesses
    
def train_LSF_tinygp(X,Y,Y_err,scatter=None, return_only_init=False):
    '''
    Returns parameters which minimise the loss function defined below.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    Y_err : TYPE
        DESCRIPTION.
    scatter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    solution : TYPE
        DESCRIPTION.

    '''
    p0 = (np.max(Y),0,np.std(X),0)
    # plt.errorbar(X,Y,Y_err,marker='o',ls='')
    
    logger = logging.Logger(__name__+'gp.train_LSF')
    logger.info('Found popt')
    cut = _check_arrays(X, Y, Y_err)
    if len(cut)<2:
        logger.warning('Data is wrong!')
        print(X,Y,Y_err)
    X = X[cut]
    Y = Y[cut]
    Y_err = Y_err[cut]
    
    # print(len(X),len(Y),len(Y_err))
    popt,pcov = curve_fit(hf.gauss4p,X._value,Y._value,sigma=Y_err._value,
                          absolute_sigma=False,p0=p0)
    
    perr = np.sqrt(np.diag(pcov))
    
    theta = dict(
        mf_amp        = popt[0],
        mf_loc        = popt[1],
        mf_log_sig    = jnp.log(popt[2]),
        mf_const      = popt[3],
        gp_log_amp    = 1.,#popt[0]/5.,
        gp_log_scale  = 0.,
        log_var_add   = -5.,
    )
    if return_only_init:
        return theta
    
    kappa = 5
    lower_bounds = dict(
        mf_amp       = popt[0]-kappa*perr[0],
        mf_loc       = popt[1]-kappa*perr[1],
        mf_log_sig   = np.log(popt[2]-kappa*perr[2]),
        mf_const     = popt[3]-kappa*perr[3],
        gp_log_amp   = -4., #popt[0]/3.-kappa*perr[0],
        gp_log_scale = -1.,
        log_var_add  = -15.,
    )
    upper_bounds = dict(
        mf_amp       = popt[0]+kappa*perr[0],
        mf_loc       = popt[1]+kappa*perr[1],
        mf_log_sig   = np.log(popt[2]+kappa*perr[2]),
        mf_const     = popt[3]+kappa*perr[3],
        gp_log_amp   = 4., # popt[0]/3.+kappa*perr[0],
        gp_log_scale = 1.,
        log_var_add  = 1.5,
    )
    # print(popt); print(perr); print(theta)#; sys.exit()
    bounds = (lower_bounds, upper_bounds)
    # print(bounds)
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=partial(loss_LSF,
                                                      X=X,
                                                      Y=Y,
                                                      Y_err=Y_err,
                                                      scatter=scatter),
                                          method="l-bfgs-b")
    solution = lbfgsb.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
    
    # solver = jaxopt.GradientDescent(fun=partial(loss_LSF,
    #                                           X=X,
    #                                           Y=Y,
    #                                           Y_err=Y_err,
    #                                           scatter=scatter
    #                                           ))
    # solution = solver.run(jax.tree_map(jnp.asarray, theta))
    # try:
    #     print(f"Best fit parameters: {solution.params}")
    # except: pass
    # debug = False
    # logger = logging.getLogger(__name__)
    # if debug:
    #     try:
    #         logger.info(f"Final -log(L): {solution.state.fun_val}")
    #     except: pass
    return solution.params

        
    
def get_model(x_test,X,Y,Y_err,theta,scatter=None):
    # print('get_model',*[np.shape(_) for _ in [x_test,X,Y,Y_err]])
    # print('get_model',theta)
    gp = build_LSF_GP(theta,X,Y,Y_err,scatter=None)
    _, cond = gp.condition(Y,x_test)
    model = cond.mean
    var   = jnp.sqrt(cond.variance)
    return model, var
    
    
def get_residuals(X,Y,Y_err,theta,scatter=None):
    '''
    Returns the residuals to the LSF model

    Parameters
    ----------
    X : array-like
        x-coordinates values.
    Y : array-like
        y-coordinates values.
    Y_err : array-like
        Standard deviation (error) on the y-coordinate values.
    theta : dictionary
        Parameters of the LSF model.
    scatter : tuple, optional
        Output of train_scatter_gp. The default is None.

    Returns
    -------
    rsd : TYPE
        Normalised residuals of the data to the model. 
        No rescaling is done internally on the errors. One may modify the
        Y_err before passing it to this function.

    '''
    model, variance = get_model(X, X, Y, Y_err, theta, None)
    rsd = jnp.array((Y - model)/Y_err)
    return rsd
    
def estimate_variance(X,Y,Y_err,theta,minpts,plot=False,ax=None):
    """
    Estimates the variance based on the residuals to the provided GP parameters
    
    The returned variance is in units of data variance! 
    One should multiply this variance with the variance on the data to get
    accurate results. 

    Parameters
    ----------
    X : jax array
        Contains the x-coordinates
    Y : jax array
        Contains the y-coordinates
    Y_err : jax array
        Contains the error on the y-coordinates.
    theta : dictionary
        Contains the LSF hyper-parameters.
    scale : TYPE
        DESCRIPTION.
    step : TYPE
        DESCRIPTION.
    minpts : TYPE
        DESCRIPTION.

    Returns
    -------
    logvar_x : TYPE
        DESCRIPTION.
    logvar_y : TYPE
        DESCRIPTION.

    """
    
    
    # Optimally bin the counts    
    counts, bin_edges = aux.bin_optimally(X,minpts)
    # Define bin centres
    # bin_cens = jnp.array((bin_edges[1:]+bin_edges[:-1])/2.)
    
    rsd = get_residuals(X,Y,Y_err,theta)
    # Calculate the relevant statistics
    calculate=['mean','std','sam_variance','sam_variance_variance',
               'pop_variance','pop_variance_variance']
    arrays = aux.get_bin_stat(X, rsd, bin_edges,calculate=calculate,
                              remove_outliers=True)
    # means = arrays['mean']
    # stds  = arrays['std']
    bin_cens = arrays['bin_centres']
    sam_var_ = arrays['sam_variance']
    sam_var_var = arrays['sam_variance_variance']
    # pop_var_ = arrays['pop_variance']
    # pop_var_var = arrays['pop_variance_variance']
    
    # Remove empty bins
    # try:
    #     cut = np.where(sam_var_!=0)[0]
    # except:
    # cut = jnp.where(sam_var_!=0,size=len(sam_var_))
    # cut = jnp.isfinite(sam_var_)
    # print(sam_var_)
    x_array     = bin_cens
    # pop_var     = pop_var_[cut]
    # pop_err     = jnp.sqrt(pop_var)  # error = sqrt of population variance
    sam_var     = sam_var_
    sam_err     = jnp.sqrt(sam_var)
    # pop_var_err = jnp.sqrt(pop_var_var[cut])
    sam_var_err = jnp.sqrt(sam_var_var)
    log_sam_var, log_sam_var_err = aux.lin2log(sam_var,sam_var_err)
    
    
    # plot_flag = plot | (ax is not None)
    # if plot_flag:
    #     if ax is not None:
    #         pass
    #     else:
    #         fig, ax = plt.subplots(1)
    #     ax.scatter(X,rsd,marker='o',s=3,label='rsd')
    #     ax.errorbar(bin_cens[cut],
    #                  np.zeros_like(bin_cens[cut]),
    #                  # means[cut],
    #                  sam_err[cut],
    #                  marker='s',ls='',c='red',
    #                  label = 'means')
    #     ax.errorbar(bin_cens[cut],sam_var,sam_var_err,marker='x',ls='',
    #                 capsize=2,c='C1')
    #     # for i in [-1,1]:
    #     #     for j in [1]:
    #     #         ax.plot(x_array, j*sam_var,color='r',lw=2)
            
    #     #         ax.fill_between(x_array, 
    #     #                           j*sam_var + i*sam_var_err,
    #     #                           j*sam_var - i*sam_var_err, 
    #     #                           color='red',alpha=0.3,zorder=10)
    #     # plt.xlim(-8,7)
    #     ax.set_xlabel("Distance from centre (pix)")
    #     ax.set_ylabel(r"$S^2 (\sigma^2)$")
    #     ax.legend()
        
    return x_array, log_sam_var, log_sam_var_err

def train_scatter_tinygp(X,Y,Y_err,theta_lsf,minpts=15,
                         include_error=True):
    '''
    Based on Kersting et al. 2007 :
        Most Likely Heteroscedastic Gaussian Process Regression

    '''

    x_array, log_var, err_log_var_ = estimate_variance(X,Y,Y_err,
                                                          theta_lsf,
                                                          minpts,plot=False)
    # print(x_array,log_var,err_log_var_)
    # err_log_var = None
    # if include_error:
    err_log_var = err_log_var_
    
    # print(f"Optimizing scatter parameters, err_log_variance = {err_log_variance}")
    theta = dict(
        sct_log_const  = -5.0,
        sct_log_amp    = -0.2,
        sct_log_scale  = 0.0,
        sct_log_epsilon0 = -3.,
        )
    lower_bounds = dict(
        sct_log_const  =-10.0,
        sct_log_amp    =-3.0,
        sct_log_scale  =-1.0,
        sct_log_epsilon0 = -15.,
        )
    upper_bounds = dict(
        sct_log_const  = 5.0,
        sct_log_amp    = 3.0,
        sct_log_scale  = 3.0,
        sct_log_epsilon0 = 3.,
        )
    bounds = (lower_bounds, upper_bounds)
    lbfgsb = jaxopt.ScipyBoundedMinimize(fun=partial(loss_scatter,
                                                      X=x_array,
                                                      Y=log_var,
                                                      Y_err=err_log_var),
                                          method="l-bfgs-b")
    solution = lbfgsb.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
    # solver = jaxopt.GradientDescent(fun=partial(loss_scatter,
    #                                           X=x_array,
    #                                           Y=log_var,
    #                                           Y_err=err_log_var,
    #                                           ))
    # solution = solver.run(jax.tree_map(jnp.asarray, theta))
    # print("Scatter solution:",solution.params)
    # print(f"Scatter final negative log likelihood: {solution.state.fun_val}")
    return solution.params, x_array, log_var, err_log_var

def get_scatter_covar(X,Y,Y_err,theta_lsf):
    gp = build_LSF_GP(theta_lsf,X,Y,Y_err,scatter=None)
    _, cond = gp.condition(Y,X,include_mean=False)
    # mean_lsf = cond.loc
    # plt.plot(X,mean_lsf)
    return cond.covariance


    


def rescale_errors(scatter,X,Y_err,plot=False,ax=None):
    '''
    Performs error rescaling, as determined by the scatter parameters

    Parameters
    ----------
    scatter : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    Y_err : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    
    
    theta_scatter, logvar_x, logvar_y, logvar_err = scatter
    
    
    sct_gp        = build_scatter_GP(theta_scatter,logvar_x,logvar_err)
    _, sct_cond   = sct_gp.condition(logvar_y,X)
    F_mean  = sct_cond.mean
    F_sigma = jnp.sqrt(sct_cond.variance)
    
    S, S_var = transform(X,Y_err,F_mean,F_sigma,sct_gp,logvar_y)
    return S, S_var

def plot_variance_GP(scatter,X,Y_err,plot=False,ax=None):
    theta_scatter, logvar_x, logvar_y, logvar_err = scatter
    
    
    sct_gp        = build_scatter_GP(theta_scatter,logvar_x,logvar_err)
    _, sct_cond   = sct_gp.condition(logvar_y,X)
    F_mean  = sct_cond.mean
    F_sigma = jnp.sqrt(sct_cond.variance)
    
    S, S_var = transform(X,Y_err,F_mean,F_sigma,sct_gp,logvar_y)
    
    plot_flag = plot | (ax is not None)
    if plot_flag:
        import matplotlib.ticker as ticker
        if ax is not None:
            pass
        else:
            fig, ax = plt.subplots(1)
        X_grid = jnp.linspace(X.min(),X.max(),200)
        
        
        _, sct_cond_grid = sct_gp.condition(logvar_y,X_grid)
        F_mean_grid  = sct_cond_grid.mean
        F_sigma_grid = jnp.sqrt(sct_cond_grid.variance)
        # print(np.shape(F_mean_grid));sys.exit()
        # f_grid, f_var_grid = transform(X_grid,np.full_like(X_grid,1.),
        #                                 F_mean_grid,F_sigma_grid,
        #                                 sct_gp,logvar_y)
        # logvar_grid_y, logvar_grid_err = aux.lin2log(f_grid, np.sqrt(f_var_grid))
        
        
        linvar_y, linvar_err = aux.log2lin(logvar_y, logvar_err)
        ax.errorbar(logvar_x,logvar_y,
                    logvar_err,ls='',capsize=2,marker='s',
                    label='binned')
        
        
        
        ax.plot(X_grid,F_mean_grid,'-C0',label=r'$g(\Delta x;\phi_g)$')
        ax.fill_between(X_grid,
                        F_mean_grid + F_sigma_grid, 
                        F_mean_grid - F_sigma_grid, 
                        color='C0',
                        alpha=0.3)
        # ax.scatter(X,(S/Y_err)**2.,c='r',s=2)
        ax.set_ylabel(r'$\log(\frac{S^2}{\sigma^2})$')
        # ax.set_yscale('log')
        ax.set_xlabel('Distance from centre (pix)')
        ax.set_ylim(-1.5, 3.5)
        ax.yaxis.tick_left()
        # ax.yaxis.set_ticks_position('left')
        axr = ax.secondary_yaxis('right', functions=(lambda x: np.exp(x), 
                                                      lambda x: np.log(x)))
        axr.yaxis.set_major_locator(ticker.FixedLocator([1,5,10,15,20]))
        axr.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axr.set_ylabel(r'$S^2 (\sigma^2)$',labelpad=-3)
        # axr.set_yticks([1, 5, 10,20])
        # axr.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.legend()
    return ax
    
    
    

def F(x,gp,logvar_y):
    '''
    For a scalar input x and a Gaussian Process GP(mu,sigma**2), 
    returns a scalar output GP(mu (x))
    
    Parameters
    ----------
        x : float32
    Output:
        value : float32
    '''
    
    value = gp.condition(logvar_y,jnp.atleast_1d(x))[1].mean
    return value[0]

def transform(x, sigma, GP_mean, GP_sigma, GP, logvar_y):
    '''
    Rescales the old error value at x-coordinate x using the GP mean 
    and sigma evaluated at x.
    
    F ~ GP(mean, sigma^2)
    F(x=x_i) = log( S_i^2 / sigma_i^2 )
    ==> S_i = sqrt( exp( F(x=x_i) ) ) * sigma_i
            = sqrt( exp( GP_mean) ) * sigma_i
        because of the property of logarithms:
            = exp (GP_mean/2.) * sigma_i
    
    
    Propagation of error gives:
    sigma(S_i) = | S_i / 2 * d(F)/dx|_{x_i} * GP_sigma |
    
    where
    GP_mean = F(x=x_i)
    GP_sigma = sigma(F(x=x_i)) 

    Parameters
    ----------
    x : float32, array_like
        x-coordinate.
    sigma : float32, array_like
        error on the y-coordinate value at x.
    GP_mean : float32, array_like
        mean of the GP evaluated at x.
    GP_sigma : float32, array_like
        sigma of the GP evaluated at x.

    Returns
    -------
    S : float32, array_like
        rescaled error on the y-coordinate at x.
    S_var : float32, array_like
        variance on the rescaled error due to uncertainty on the GP mean.

    '''
    deriv = jax.grad(partial(F,gp=GP,logvar_y=logvar_y))
    dFdx  = jax.vmap(deriv)(x)
    S = sigma * jnp.exp(GP_mean/2.)
    S_var = jnp.power(S / 2. * dFdx * GP_sigma,2.)
    return S, S_var
def gaussian_mean_function(theta, X):
    '''
    Returns the Gaussian profile with parameters encapsulated in dictionary
    theta, evaluated a points in X

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    mean  = theta["mf_loc"]
    sigma = jnp.exp(theta["mf_log_sig"])
    gauss = jnp.exp(-0.5 * jnp.square((X - mean)/sigma)) \
            / jnp.sqrt(2*jnp.pi) / sigma
    beta = jnp.array([gauss,1])
    
    return jnp.array([theta['mf_amp'],theta['mf_const']]) @ beta

def build_scatter_GP(theta,X,Y_err=None):
    '''
    Returns Gaussian Process for the intrinsic scatter of points (beyond noise)

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    sct_const  = jnp.exp(theta['sct_log_const'])
    sct_amp    = jnp.exp(theta['sct_log_amp'])
    sct_scale  = jnp.exp(theta['sct_log_scale'])
    pred = Y_err!=None
    
    def true_func():
        
        return noise.Diagonal(jnp.power(Y_err,2.))
    def false_func():
        val = 1e-8
        # val = jnp.exp(theta['sct_log_epsilon0'])
        return noise.Diagonal(jnp.full_like(X,val))
    noise1d = jax.lax.cond(pred,true_func,false_func)
    sct_kernel = sct_amp * kernels.ExpSquared(sct_scale) #+ kernels.Constant(sct_const)
    # sct_kernel = sct_amp * kernels.Matern52(sct_scale) #+ kernels.Constant(sct_const)
    return GaussianProcess(
        sct_kernel,
        X,
        noise= noise1d,
        mean = sct_const
    )

def build_LSF_GP(theta_lsf,X,Y=None,Y_err=None,scatter=None):
    '''
    Returns a Gaussian Process for the LSF. If scatter is not None, tries to 
    include a second GP for the intrinsic scatter of datapoints beyond the
    error on each individual point.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    Y_err : TYPE
        DESCRIPTION.
    scatter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    gp_amp   = jnp.exp(theta_lsf['gp_log_amp'])
    gp_scale = jnp.exp(theta_lsf["gp_log_scale"])
    kernel = gp_amp * kernels.ExpSquared(gp_scale) # LSF kernel
    # Various variances (obs=observed, add=constant random noise, tot=total)
    var_add = jnp.exp(theta_lsf['log_var_add']) 
    
    if scatter is not None:   
        S, S_var = rescale_errors(scatter, X, Y_err)
        var_data  = jnp.power(S,2.)
    else:
        _pred_ = Y_err!=None
        def _true_func():
            return jnp.power(Y_err,2.)
        def _false_func():
            return jnp.full_like(X, 1e-8)
        var_data = jax.lax.cond(_pred_,_true_func,_false_func)
    var_tot = var_data + var_add
    noise2d = jnp.diag(var_tot)
    Noise2d = noise.Dense(noise2d)
    
    
    return GaussianProcess(
        kernel,
        X,
        noise = noise2d,
        mean=partial(gaussian_mean_function, theta_lsf),
    )

def estimate_centre_numerically(X,Y,Y_err,LSF_solution,scatter=None,N=10):
    gp = build_LSF_GP(LSF_solution,X,Y,Y_err,scatter)
    rng_key = jax.random.PRNGKey(1234)
    X_grid  = jnp.linspace(-1,1,100)
    _, cond = gp.condition(Y,X_grid)
    samples = cond.sample(rng_key,shape=(N,))
    der=hf.derivative(samples,X_grid,order=1,accuracy=8)

def estimate_centre(X,Y,Y_err,LSF_solution,scatter=None,N=10):
    
    def value_(x):
        _, cond = gp.condition(Y,jnp.array([x]))
        sample = cond.sample(rng_key,shape=())
        return sample[0]
    # @partial(gp=cond,Y=Y)
    def derivative_(x):#,gp,Y,rng_key):
        # return jax.grad(partial(value_,gp=gp,Y=Y,rng_key=rng_key))(x)
        return jax.grad(value_)(x)
    # @jit
    def solve_(rng_key):
        bisect = jaxopt.Bisection(derivative_,-1.,1.)#,gp=gp,Y=Y,rng_key=rng_key)
        return bisect.run().params
    
    gp = build_LSF_GP(LSF_solution,X,Y,Y_err,scatter)
    X_grid  = jnp.linspace(-1,1,100)
    _, cond = gp.condition(Y,X_grid)
    
    
    centres = np.empty(N)
    for i in range(N):
        rng_key = jax.random.PRNGKey(i)
        
        centres[i] = solve_(rng_key)
    mean, sigma = hf.average(centres)
    del(centres); del(X_grid); del(cond)
    gc.collect()
    return -mean, sigma
def estimate_centre_anderson(X,Y,Y_err,LSF_solution,scatter=None):
    
    def value_(x):
        _, cond = gp.condition(Y,jnp.array([x]))
        return cond.mean[0]
    # @partial(gp=cond,Y=Y)
    def derivative_(x):#,gp,Y,rng_key):
        # return jax.grad(partial(value_,gp=gp,Y=Y,rng_key=rng_key))(x)
        return jax.grad(value_)(x)
    # @jit
    gp = build_LSF_GP(LSF_solution,X,Y,Y_err,scatter)
    
    vn = value_(-0.5)
    vp = value_(+0.5)
    dn = derivative_(-0.5)
    dp = derivative_(+0.5)
    shift = (vp - vn)/(dp - dn)
    if not np.isfinite(shift):
        shift=np.random.normal(0,0.005)
    
    return shift, 0.
def estimate_centre_median(X,Y,Y_err,LSF_solution,scatter=None):
    from scipy.special import erfinv
    
    gp = build_LSF_GP(LSF_solution,X,Y,Y_err,scatter)
    y = jnp.linspace(-1.0+1e-7, 1.0-1e-7, num=500)
    X_grid_ = jnp.sort(erfinv(y))
    sampled_Xrange = X_grid_.max() - X_grid_.min()
    
    Xmin = X.min()
    Xmax = X.max()
    desired_Xrange=X.max()-X.min()
    
    X_grid_2 = X_grid_/sampled_Xrange*desired_Xrange
    
    offset = X_grid_2.max() - X.max()
    
    
    X_grid   = X_grid_2 - offset
    _, cond  = gp.condition(Y,X_grid)
    # cumsum   = 
    mean_lsf = cond.mean
    cumsum   = np.cumsum(mean_lsf)/np.sum(mean_lsf)-0.5
    
    x1       = np.argmin(np.abs(cumsum))
    x2       = x1+1
    m = (cumsum[x2]-cumsum[x1])/(X_grid[x2]-X_grid[x1])
    b = cumsum[x2] - m*X_grid[x2]
    shift = b/m
    # plt.figure()
    # plt.plot(X_grid,cumsum)
    # plt.scatter([X_grid[x1],X_grid[x2],shift],
    #             [cumsum[x1],cumsum[x2],m*shift+b],
    #             c='r',s=10,marker='o')
    # plt.plot([X_grid[x1],X_grid[x2]],[cumsum[x1],cumsum[x2]])
    # print('linterp',shift)
    # splr  = interpolate.splrep(X_grid,cumsum)
    # shift = brentq(cumsum_at,-0.5,0.5)
    # print('spline',shift)
    return shift, 0.

def estimate_centre_centroid(X,Y,Y_err,LSF_solution,scatter=None):
    gp = build_LSF_GP(LSF_solution,X,Y,Y_err,scatter)
    
    X_grid   = jnp.linspace(X.min(),X.max(),1000)
    _, cond  = gp.condition(Y,X_grid)
    mean_lsf = cond.mean
    shift    = -np.average(X_grid,weights=mean_lsf)
    return shift, 0.

def estimate_centre_mean(X,Y,Y_err,LSF_solution,scatter=None):
    gp = build_LSF_GP(LSF_solution,X,Y,Y_err,scatter)
    X_grid   = jnp.linspace(X.min(),X.max(),1000)
    _, cond  = gp.condition(Y,X_grid)
    mean_lsf = cond.mean
    shift    = -np.average(X_grid,weights=mean_lsf)
    return shift, 0.