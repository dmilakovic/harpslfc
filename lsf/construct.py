#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:39:20 2023

@author: dmilakov
"""
from fitsio import FITS
import numpy as np
import jax.numpy as jnp
import harps.functions as hf
import harps.peakdetect as pkd
import harps.lsf.aux as aux
import harps.lsf.gp_aux as gp_aux
import harps.lsf.plot as lsfplot
import harps.lsf.gp as lsfgp
import harps.lsf.inout as lio
# import harps.lsf.write as write
import harps.lsf.read as read
import harps.fit as hfit
import harps.inout as hio
import harps.version as hv
import harps.progress_bar as progress_bar
import harps.settings as hs
import hashlib
import matplotlib.pyplot as plt
# import scipy.interpolate as interpolate
import gc
import multiprocessing
multiprocessing.log_to_stderr()
from functools import partial
import time
import ctypes
import logging, sys, os, datetime
from logging.handlers import QueueHandler, QueueListener

import ray

def initialize_ray(address=None):
    if ray.is_initialized():
        return
    if address:
        # Connects to an existing HPC cluster or remote head node
        ray.init(address=address)
    else:
        # Starts a local cluster using all available cores on your laptop/remote machine
        ray.init()


def model_1si(i,seglims,x2d,flx2d,err2d,numiter=5,filter=None,model_scatter=False,
                    plot=False,save_plot=False,metadata=None,
                    **kwargs):
    logger = logging.getLogger(__name__)
    pixl = seglims[i]
    pixr = seglims[i+1]
    x1s  = np.ravel(x2d[pixl:pixr])
    flx1s = np.ravel(flx2d[pixl:pixr])
    err1s = np.ravel(err2d[pixl:pixr])
    checksum = aux.get_checksum(x1s, flx1s, err1s,uniqueid=i)
    
    try:
        metadata.update({'segment':i+1,'checksum':checksum})
    except:
        pass
    out  = model_1s(x1s,flx1s,err1s,numiter=numiter,
                    filter=filter,model_scatter=model_scatter,
                    plot=plot,metadata=metadata,
                    **kwargs)
    if out is not None:
        out['ledge'] = pixl
        out['redge'] = pixr
        out['segm'] = i
    else:
        out = None
    return i, out

@ray.remote
def model_1s_remote(od, pixl, pixr, x2d, flx2d, err2d, **kwargs):
    # This is exactly the existing model_1s_ logic
    # It will now run on whatever CPU Ray assigns it to
    return model_1s_(od, pixl, pixr, x2d, flx2d, err2d, **kwargs)

def model_1s_(od,pixl,pixr,x2d,flx2d,err2d,numiter=5,filter=None,model_scatter=False,
                    plot=False,save_plot=False,metadata=None,logger=None,
                    **kwargs):
    x1s  = np.ravel(x2d[od,pixl:pixr])
    flx1s = np.ravel(flx2d[od,pixl:pixr])
    err1s = np.ravel(err2d[od,pixl:pixr])
    
    valid = np.any(x1s)
    if not valid:
        parnames = gp_aux.parnames_lfc.copy() + gp_aux.parnames_sct.copy()
        out = aux._prepare_lsf1s(n_data=1,n_sct=1,pars=parnames)
        return out
    
    checksum = aux.get_checksum(x1s, flx1s, err1s,uniqueid=pixl+pixr+od)
    # print(f"segment = {i+1}/{len(seglims)-1}")
    try:
        metadata.update({'checksum':checksum})
    except:
        pass
    metadata.update({'order':od})
    segm = int(divmod((pixl+pixr)/2.,(pixr-pixl))[0])
    metadata.update({'segment':segm})
    if logger is not None:
        logger = logger.getChild('model_1s_')
    else:
        logger = logging.getLogger(__name__).getChild('model_1s_')
    logging.info(f"Order, segment : {od}, {segm}")
    # print(f"Order, segment : {od}, {segm}")
    # try:
    out  = model_1s(x1s,flx1s,err1s,
                    numiter=numiter,
                    filter=filter,
                    model_scatter=model_scatter,
                    plot=plot,
                    save_plot=save_plot,
                    metadata=metadata,
                    logger=logger,
                    **kwargs)
    # except:
        # out = None
    if out is not None:
        out['ledge'] = pixl
        out['redge'] = pixr
        out['order'] = od
        out['segm']  = segm
    else:
        # parnames = gp_aux.parnames_lfc.copy()
        # out = aux._prepare_lsf1s(N_data=1,N_sct=0,pars=parnames)
        out = (None,od,segm)
        msg = f'Failed to construct IP for order {od}, segment {segm}. ' +\
              f'Printing x1s: {repr(x1s)} ' +\
              f'Printing flx1s: {repr(flx1s)} '+\
              f'Printing err1s: {repr(err1s)}'
        # print(msg)
        logger.error(msg)
    return out

#@profile
def stack_segment(x_star,f_star,x1s,flx1s,err1s,minima_x,scale='pixel'):
    '''
    

    Parameters
    ----------
    x_star : list
        line centres.
    f_star : list
        line brightness.
    x1s : array-like
        data x-coordinates.
    flx1s : array-like
        data y-coordinates.
    err1s : array-like
        data y-coordinate errors.

    Returns
    -------
    x_stacked : TYPE
        DESCRIPTION.
    y_stacked : TYPE
        DESCRIPTION.
    err_stacked : TYPE
        DESCRIPTION.

    '''
    assert len(x1s)==len(flx1s)==len(err1s)
    assert len(x_star)==len(f_star)==(len(minima_x)-1)
    assert scale in ['pixel','velocity']
    
    N     = len(minima_x)-1
    X     = np.zeros_like(x1s,dtype=np.float32)
    Y     = np.zeros_like(flx1s)
    Y_err = np.zeros_like(err1s)
    
    for i in range(N):
        pixl,pixr = minima_x[i],minima_x[i+1]
        _         = slice(pixl,pixr)
        print(i,_,x_star[i])
        X[_]      = x1s[_] - x_star[i]
        Y[_]      = flx1s[_] / f_star[i]
        Y_err[_]  = err1s[_] / f_star[i]
        
    
    
    return X, Y, Y_err

def get_initial_guess(x1s,flx1s,err1s,minima_x):
    '''
    Returns the locations of minima between LFC lines and the initial guess
    for line positions and brightness. 

    Parameters
    ----------
    x1s : array
        data x-coordinates.
    flx1s : array
        data y-coordinates.
    err1s : array
        data y-coordinate error.
    minima_x : list
        list of points separating LFC lines.

    Returns
    -------
    minima_x : list
        A list of x-coordinates for minima in the input data.
    x_star_0 : list
        A list of LFC line centroids (centre of mass).
    f_star_0 : list
        A list of LFC line brigntess (sum of flux).

    '''
    # detect minima in the data, lines are between minima
    
    npix = len(x1s)
    nlines = len(minima_x)-1
    
    x_star_0 = np.zeros(nlines)
    f_star_0 = np.zeros(nlines)
    for i in range(nlines):
        lpix, rpix = minima_x[i], minima_x[i+1]
        if lpix==0:
            lpix = 1
        if rpix==npix-1:
            rpix = npix-2 
        
        x = x1s[lpix-1:rpix+1]
        f = flx1s[lpix-1:rpix+1]
        e = err1s[lpix-1:rpix+1]
        # bkgx = background[lpix-1:rpix+1]
        # envx = envelope[lpix-1:rpix+1]
        fit_result = hfit.gauss(x,f,e,line_model='SingleGaussian')
        success, pars,errs,chisq,chisqnu,integral = fit_result
        x_star_0[i] = pars[1]
        f_star_0[i] = integral
        
    return minima_x, x_star_0, f_star_0


def model_1s(pix1s,flx1s,err1s,numiter=5,filter_n_elements=None,
             model_scatter=False,
             remove_outliers=True,
             plot=False,save_plot=False,metadata=None,logger=None,
             debug=True,**kwargs):
    '''
    Constructs the LSF model for a single segment
    '''
    if logger is not None:
        logger = logger.getChild('model_1s')
    else:
        logger = logging.getLogger(__name__).getChild('model_1s')
    logger = logging.getLogger(__name__).getChild('model_1s')
    # c_handler = logging.StreamHandler()
    # logger.addHandler(c_handler)
    # logger.setLevel(logging.WARNING) # <-- THIS!
    # logger.setLevel(logging.INFO)
    
    verbose             = kwargs.pop('verbose',False)
    
    pix1s, flx1s, err1s = aux.clean_input(pix1s,flx1s,err1s,
                                          sort=True,
                                          verbose=verbose,
                                          filter_n_elements=filter_n_elements)
    if len(pix1s)==0:
        return None
    
    
        
    shift    = 0
    oldshift = 1
    relchange = 1
    delta     = 100
    delta_jm1 = 0
    shift_j  = 0
    keep_full = np.full_like(pix1s, True, dtype=bool)
    keep_jm1  = keep_full
    args = {}
    dictionary_j = {}
    metadata.update({'model_scatter':model_scatter})
    for j in range(numiter):
        metadata.update({'recentering':j})
        # shift the values along x-axis for improved centering
        # remove outliers from last iteration
        if np.abs(shift)>1: shift=np.sign(shift)*0.25
        
        pix1s_j = (pix1s + shift)[keep_jm1]
        flx1s_j = flx1s[keep_jm1]
        err1s_j = err1s[keep_jm1]
        dictionary_jm1 = dictionary_j
        dictionary_j=construct_tinygp(pix1s_j,flx1s_j,err1s_j, 
                                    plot=plot,
                                    metadata=metadata,
                                    filter=filter,model_scatter=model_scatter,
                                    logger=logger)
        
        # save shift from previous iteration
        shift_jm1 = shift_j
        # update this iterations shift
        shift_j  = dictionary_j['lsfcen']
        if not ( np.isfinite(shift_j) or 
             np.isfinite(dictionary_j['chisq'])) and j>0:
            dictionary_j = dictionary_jm1
        # lsf1s  = dictionary_j['lsf1s']
        # update total shift
        shift += shift_j
        # shift = shift_j
        
        cenerr = dictionary_j['lsfcen_err']
        chisq  = dictionary_j['chisq']
        rsd    = dictionary_j['rsd']
        # remove outliers in residuals before proceeding with next iteration
        if remove_outliers:
            outliers_j   = hf.is_outlier_original(rsd)
            cut          = np.where(outliers_j==True)
            keep_full[cut] = False
            keep_jm1 =  keep_full
            keep_full = np.full_like(pix1s,True,dtype='bool')
        else:
            keep_jm1 = np.full_like(pix1s,True,dtype='bool')
        
        # change in shift between this iteration and the previous one
        delta_jm2 = delta_jm1
        delta_jm1 = delta
        delta = np.abs(shift_j - shift_jm1)
        
        dictionary_j.update({'shift':shift})
        dictionary_j.update({'scale':metadata['scale'][:3]})
        logger.debug(f"iter {j:2d}   shift={shift:+5.2e}  " + \
              f"delta={delta:5.2e}   " +\
              f"N={len(rsd)}  chisq={chisq:6.2f}")
        # print(f"iter {j:2d}   shift={shift:+5.2e}  " + \
        #       f"delta={delta:5.2e}   " +\
        #       f"N={len(rsd)}  chisq={chisq:6.2f}")
        # break if
        # 1. change in LSF centre (delta) smaller than
        delta_lim = 1e-3 # pix
        # or
        # 2. total shift smaller than 
        shift_lim = 1e-3 # pix
        # or
        # 3. iteration number equal to iteration limit
        condition = (np.abs(delta)<delta_lim 
                     or np.abs(shift)<=shift_lim 
                     or j==numiter-1 
                     or delta==delta_jm2) 
        if not np.isfinite(shift_j) or not np.isfinite(chisq):
            if j>0:
                print('Shift is not finite, breaking')
                dictionary_j = dictionary_jm1
                condition = True
            else:
                print(dictionary_j['shift'])
                print('Failed at first iteration')
        lsf1s = dictionary_j['lsf1s']
        if condition:
            if plot:
                plotfunction = lsfplot.plot_solution
                LSF_solution = dictionary_j['LSF_solution']
                scatter      = dictionary_j['scatter']
                plotkwargs = dict(params_LSF=LSF_solution, 
                                  scatter=scatter, 
                                  metadata=metadata, 
                                  save=save_plot,
                                  shift=shift,
                                  **kwargs)
                if np.all(pix1s_j) and np.all(flx1s_j) and np.all(err1s_j):
                    plotfunction(pix1s_j, flx1s_j, err1s_j, **plotkwargs)
                
                
            break
        else:
            for variable in [dictionary_jm1, lsf1s, shift, cenerr, chisq, rsd]:
                del(variable)
    # print(f'total shift : {shift*1e3:12.6f} mpix ')
    logger.debug(f'total shift : {shift*1e3:12.6f} mpix '+\
                f'after {j} iterations, rmv_outliers:{remove_outliers}'+\
                f' (delta={delta:+6.2f}, chisq={chisq:6.2f})') 
   
    chisqlimit=10
    if chisq>chisqlimit:
        logger.warning(f'Chisq above limit ({chisqlimit}): {chisq}')
    
    # save the total number of points used
    lsf1s['numlines'] = len(pix1s_j)
    lsf1s['shift'] = shift
    return lsf1s


def construct_tinygp(x,y,y_err,plot=False,
                     filter=None,N_test=20,model_scatter=False,
                     logger=None,
                     **kwargs):
    '''
    Returns the LSF model for one segment using TinyGP framework

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    y_err : TYPE
        DESCRIPTION.
    numpix : TYPE
        DESCRIPTION.
    subpix : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    checksum : TYPE, optional
        DESCRIPTION. The default is None.
    filter : TYPE, optional
        DESCRIPTION. The default is 10.
    N_test : TYPE, optional
        DESCRIPTION. The default is 400.
    model_scatter : TYPE, optional
        DESCRIPTION. The default is True.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    out_dict : TYPE
        DESCRIPTION.

    '''
    X        = jnp.array(x)
    Y        = jnp.array(y)
    Y_err    = jnp.array(y_err)
    assert len(X)==len(Y)==len(Y_err)
    
    
    N_data   = len(X)
    # print(X,Y,Y_err)
    if logger is not None:
        logger = logger.getChild('construct_tinygp')
    else:
        logger = logging.getLogger(__name__).getChild('construct_tinygp')
    # if kwargs['metadata']['segment']==10:
    #     print(X,kwargs['metadata'])
    LSF_solution_nosct = lsfgp.train_LSF_tinygp(X,Y,Y_err)
    logger.info(f"Found solution without scatter")
    if model_scatter:
        scatter = lsfgp.train_scatter_tinygp(X,Y,Y_err,LSF_solution_nosct)
        LSF_solution = lsfgp.train_LSF_tinygp(X,Y,Y_err,scatter=scatter)
        logger.info(f"Found solution with scatter")
    else:
        scatter=None
        LSF_solution = LSF_solution_nosct
        
    Y_data_err = Y_err
    if scatter is not None:
        S, S_var = lsfgp.rescale_errors(scatter, X, Y_err)
        Y_data_err = S
    # print(jnp.sum(jnp.isfinite(Y_data_err))/len(Y_data_err))    
    gp = lsfgp.build_LSF_GP(LSF_solution,X,Y,Y_data_err)
    
    # --------  Save output -------- 
    
    if scatter is not None:
        parnames = gp_aux.parnames_lfc.copy() + gp_aux.parnames_sct.copy()
        assert len(scatter[1])==len(scatter[2])
        N_sct = len(scatter[1])
    else:
        parnames = gp_aux.parnames_lfc.copy()
        N_sct = 0
        
    # Initialize an LSF for this segment
    lsf1s    = aux._prepare_lsf1s(N_data,N_sct,pars=parnames)
    # if model_scatter:
        # parnames_ = gp_aux.parnames_lfc.copy()
        # lsf1s_nosct = aux._prepare_lsf1s(N_data,N_sct=0,pars=parnames_)
    
    # Save parameters
    # The parameters are saved in gp_aux
    npars = 0
    # print(gp_aux.parnames_lfc)
    for parname in gp_aux.parnames_lfc:
        lsf1s[parname] = LSF_solution[parname]
        npars = npars + 1
    if scatter is not None:
        for parname in gp_aux.parnames_sct:
            lsf1s[parname] = scatter[0][parname]
            npars = npars + 1
        # for parname in gp_aux.parnames_lfc:
        #     lsf1s_nosct[parname] = LSF_solution_nosct[parname]
            # npars = npars + 1
        
    # Save data that was used to create the GP models (needed for conditioning)
    lsf1s['data_x']    = X
    lsf1s['data_y']    = Y
    lsf1s['data_yerr']    = Y_err
    
    if model_scatter:
        lsf1s['sct_x']     = scatter[1]
        lsf1s['sct_y']     = scatter[2]
        lsf1s['sct_yerr']  = scatter[3]
        
        
    
    
        
        
    # # Now condition on the same grid as data to calculate residual
    
    logL, cond    = gp.condition(Y, X)
    lsf1s['logL'] = logL
    # Y_mod_err  = np.sqrt(cond.variance)
    # Y_tot_err  = jnp.sqrt(np.sum(np.power([Y_data_err,Y_mod_err],2.),axis=0))
    rsd        = lsfgp.get_residuals(X, Y, Y_data_err, LSF_solution)
    dof        = len(rsd) - npars
    chisq      = np.sum(rsd**2)
    chisqdof   = chisq / dof
    centre_estimator = lsfgp.estimate_centre_anderson
    # centre_estimator = lsfgp.estimate_centre_median
    # centre_estimator = lsfgp.estimate_centre_mean
    
    lsfcen, lsfcen_err = centre_estimator(X, Y, Y_err,
                                          LSF_solution,scatter=scatter)
    # lsf1s['shift']     = lsfcen
    out_dict = dict(lsf1s=lsf1s, lsfcen=lsfcen, lsfcen_err=lsfcen_err,
                    chisq=chisqdof, rsd=rsd, 
                    LSF_solution=LSF_solution,
                    LSF_solution_nosct = LSF_solution_nosct)
    out_dict.update(dict(model_scatter=model_scatter))
    out_dict.update(dict(scatter=scatter))
        # out_dict.update(dict(lsf1s_nosct=lsf1s_nosct))
    gc.collect()
    return out_dict


def copy_lsf1s_data(copy_from,copy_to):
    # print(copy_from.dtype.names)
    # print(copy_to.dtype.names)
    assert copy_from.dtype.names == copy_to.dtype.names, print("Unmatching dtype names.\n "\
                                                               f"Names 1: {copy_from.dtype.names}\n"\
                                                               f"Names 2: {copy_to.dtype.names}")
    names = copy_from.dtype.names
    for name in names:
        try:
            # Data can be directly copied
            copy_to[name] = copy_from[name]
        except:
            # Array lengths do not match so copy only where needed
            len_data  = len(copy_from[name])
            copy_to[name] = np.nan
            copy_to[name][slice(0,len_data)] = copy_from[name]
            
    return copy_to



class SequenceIterator:
    # Based in part on https://realpython.com/python-iterators-iterables/
    def __init__(self,orders,seglimits):
        self._index = 0
        self._orders = orders
        self._seglimits = seglimits
        self._index_od   = 0
        self._current_od = self._orders[0]
        self._index_seg  = 0
        self._current_seg = self._seglimits[0],self._seglimits[1]
        self._current = (self._current_od,*self._current_seg)
        
        self._max_od = len(orders)
        self._max_seg = len(seglimits)-1
        self._max = len(orders)*(len(seglimits)-1)
    def __len__(self):
        return self._max
    def __iter__(self):
        return self
    def __next__(self):
        item = (self._current_od,*self._current_seg)
        self._index += 1
        cond1 = self._index_seg < self._max_seg-1
        cond2 = self._index_od < self._max_od-1
        cond3 = self._index < self._max + 1
        if cond3:
            try:
                self._next_segment()
            except:
                try:
                    self._next_order()
                except:
                    pass
            return item
            # if cond1: # is not last segment in the order
            #     self._next_segment()
            #     return item
            # elif not cond1 and cond2: # is last segment but is not last order
            #     self._next_order()
            #     return item
            # elif cond1 and not cond2: # is not last segment but is last order
            #     self._next_segment()
            #     return item
            # elif not cond1 and not cond2: # is last segment and is last order
            #     return item
        else:
            raise StopIteration
    def _next_order(self):
        self._index_od += 1
        self._current_od = self._orders[self._index_od]
        self._index_seg = 0
        self._current_seg = (self._seglimits[self._index_seg],
                             self._seglimits[self._index_seg+1])
    def _next_segment(self):
        self._index_seg +=1 
        self._current_seg = (self._seglimits[self._index_seg],
                             self._seglimits[self._index_seg+1])


def from_spectrum_2d(spec,orders,iteration,scale='pixel',iter_center=5,
                  numseg=16,minpix=None,maxpix=None,filter=None,wavesol_version=700,
                  model_scatter=True,save_fits=True,clobber=False,
                  plot=False,save_plot=False,force_version=None,
                  interpolate=False,update_linelist=True,logger=None):
    assert scale in ['pixel','velocity']
    assert iteration>0
    
    if logger is not None:
        logger = logger.getChild('from_spectrum_2d')
    else:
        logger = logging.getLogger(__name__).getChild('from_spectrum_2d')
    logger.setLevel(logging.INFO)
    if force_version is None:
        version = hv.item_to_version(dict(iteration=iteration,
                                            model_scatter=model_scatter,
                                            interpolate=interpolate
                                            ),
                                       ftype='lsf'
                                       )
    else:
        version = force_version
    logger.info(f'{__name__}, subbkg = {hs.subbkg}, divenv = {hs.divenv} ')
    pix3d,vel3d,flx3d,err3d,orders_=aux.stack_spectrum(spec,
                                                       version=version,
                                                       wavesol_version=wavesol_version,
                                                       orders=orders,
                                                       subbkg=hs.subbkg,
                                                       divenv=hs.divenv)
    if scale=='pixel':
        x2d = pix3d[:,:,0]
    elif scale=='velocity':
        x2d = vel3d[:,:,0]
    flx2d = flx3d[:,:,0]
    err2d = err3d[:,:,0]
    
    metadata = dict(
        scale=scale,
        # order=order,
        iteration=iteration,
        model_scatter=model_scatter,
        interpolate=interpolate
        )
    
    # print(np.shape(x2d))
    npix   = np.shape(x2d)[1]
    minpix = minpix if minpix is not None else 0
    maxpix = maxpix if maxpix is not None else npix
    seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
    iterator = SequenceIterator(orders,seglims)
    
    parnames = gp_aux.parnames_lfc.copy()
    if model_scatter:
        parnames = gp_aux.parnames_all.copy()
    lsf2d = aux.get_empty_lsf(len(iterator), 
                              n_data=600, n_sct=40, pars=parnames)
    
    
    time_start = time.time()
    
    
    
    option=3
    if option!=3:
        partial_function = partial(model_1s_,
                                    x2d=x2d,
                                    flx2d=flx2d,
                                    err2d=err2d,
                                    numiter=iter_center,
                                    filter=filter,
                                    model_scatter=model_scatter,
                                    plot=plot,
                                    save_plot=save_plot,
                                    metadata=metadata,
                                    logger=None
                                    )
    if option==1:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(partial_function,
                                    iterator)
    elif option==2:
        # job_queue = multiprocessing.Queue(maxsize=8)
        # results   = multiprocessing.Queue()
        # for item in iterator:
            # job_queue.put(item)
        logger.info('Starting LSF fitting')
        manager = multiprocessing.Manager()
        inq = manager.Queue()
        outq = manager.Queue()
    
        # construct the workers
        nproc = multiprocessing.cpu_count()
        
        logger.info(f"Using {nproc} workers")
        workers = [Worker(str(name+1), partial_function,inq, outq,logger) 
                   for name in range(nproc)]
        for worker in workers:
            worker.start()
    
        # add data to the queue for processing
        work_len = len(iterator)
        for item in iterator:
            # print(f"Item before putting into queue: {item}")
            inq.put(item)
    
        while outq.qsize() < work_len:
            # waiting for workers to finish
            done = outq.qsize()
            progress = done/(work_len)
            time_elapsed = time.time() - time_start
            progress_bar.update(progress,name=f'LSF_2d {scale} {iteration}',
                               time=time_elapsed,
                               logger=None)
            
            # print("Waiting for workers. Out queue size {}".format(outq.qsize()))
            time.sleep(1)
    
        # clean up
        for worker in workers:
            worker.terminate()
    
        # print the outputs
        results = []
        while not outq.empty():
            results.append(outq.get())
    
    elif option == 3:
        logger.info('Starting distributed LSF fitting via Ray')
        
        # 1. Initialize Ray (Auto-detects laptop cores or HPC cluster)
        if not ray.is_initialized():
            ray.init()
        
        # 2. Put large spectral arrays into the Object Store [7]
        x2d_ref = ray.put(x2d)
        flx2d_ref = ray.put(flx2d)
        err2d_ref = ray.put(err2d)
    
        # 3. Launch tasks for every segment in the echelle orders [3, 8]
        futures = [
            model_1s_remote.remote(
                item[0], item[1], item[2], # Unpack (od, pixl, pixr) from iterator [4]
                x2d_ref, flx2d_ref, err2d_ref, 
                numiter=iter_center,
                filter=filter,
                model_scatter=model_scatter,
                plot=plot,
                save_plot=save_plot,
                metadata=metadata,
                logger=None
            ) 
            for item in iterator
        ]
        
        work_len = len(futures)
        time_start = time.time()
        finished_count = 0
        unready = futures
        results_ordered = [None] * work_len
        # Map the futures to their original iterator indices to preserve order
        future_to_index = {f: i for i, f in enumerate(futures)}

        while unready:
            # Wait for at least one task to finish (timeout=1s to refresh time display)
            ready, unready = ray.wait(unready, num_returns=1, timeout=1.0)
            
            # Update stats
            finished_count = work_len - len(unready)
            progress = finished_count / work_len
            time_elapsed = time.time() - time_start
            
            progress_bar.update(
                progress, 
                name=f'Ray LSF Fitting {scale} {iteration}',
                time=time_elapsed,
                logger=None
            )
        # 4. Asynchronous collection of results
        results = ray.get(futures)
        
    for i,lsf1s_out in enumerate(results):
        if lsf1s_out[0] == None:
            msg = f"LSF1s model order {lsf1s_out[1]} segm {lsf1s_out[2]} failed"
            logger.critical(msg)
        else:
            lsf2d[i]=copy_lsf1s_data(lsf1s_out[0],lsf2d[i])
    worktime = (time.time() - time_start)
    h, m, s = progress_bar.get_time(worktime)
    logger.info(f"Total time elapsed = {h:02d}h {m:02d}m {s:02d}s")
    
    if save_fits:
        
        # Save GP parameters and data
        lsf_filepath = hio.get_fits_path('lsf',spec.filepath)
        lio.write_lsf_to_fits(lsf2d, lsf_filepath, f"{scale}_gp",
                              version=version,
                              clobber=clobber)   
        # Save LSF numerical models
        nummodel_lsf = numerical_models(lsf2d,xrange=(-6,6),subpix=50)
        lio.write_lsf_to_fits(nummodel_lsf, lsf_filepath, f"{scale}_model",
                              version=version,
                              clobber=clobber)   
    gc.collect()
    
    return lsf2d

def from_outpath_2d(outpath,orders,iteration,scale='pixel',iter_center=5,
                  numseg=16,minpix=None,maxpix=None,filter=None,
                  model_scatter=True,save_fits=True,clobber=False,
                  plot=False,save_plot=False,force_version=None,
                  interpolate=False,update_linelist=True,logger=None):
    assert scale in ['pixel','velocity']
    assert iteration>0
    
    if logger is not None:
        logger = logger.getChild('from_spectrum_2d')
    else:
        logger = logging.getLogger(__name__).getChild('from_spectrum_2d')
    
    
    
    if force_version is None:
        version = hv.item_to_version(dict(iteration=iteration,
                                            model_scatter=model_scatter,
                                            interpolate=interpolate
                                            ),
                                       ftype='lsf'
                                       )
    else:
        version = force_version
    logger.info(f'{__name__}, subbkg = {hs.subbkg}, divenv = {hs.divenv} ')
    pix3d,vel3d,flx3d,err3d,orders_=aux.stack_outpath(outpath,version,
                                                       orders=orders,
                                                       subbkg=hs.subbkg,
                                                       divenv=hs.divenv)
    if scale=='pixel':
        x2d = pix3d[:,:,0]
    elif scale=='velocity':
        x2d = vel3d[:,:,0]
    flx2d = flx3d[:,:,0]
    err2d = err3d[:,:,0]
    
    

    metadata = dict(
        scale=scale,
        # order=order,
        iteration=iteration,
        model_scatter=model_scatter,
        interpolate=interpolate
        )
    
    # print(np.shape(x2d))
    npix   = np.shape(x2d)[1]
    minpix = minpix if minpix is not None else 0
    maxpix = maxpix if maxpix is not None else npix
    seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
    iterator = SequenceIterator(orders,seglims)
    
    parnames = gp_aux.parnames_lfc.copy()
    if model_scatter:
        parnames = gp_aux.parnames_all.copy()
    lsf2d = aux.get_empty_lsf(len(iterator), 
                              n_data=600, n_sct=40, pars=parnames)
    
    
    time_start = time.time()
    
    
    
    option=3
    if option !=3:
        partial_function = partial(model_1s_,
                                    x2d=x2d,
                                    flx2d=flx2d,
                                    err2d=err2d,
                                    numiter=iter_center,
                                    filter=filter,
                                    model_scatter=model_scatter,
                                    plot=plot,
                                    save_plot=save_plot,
                                    metadata=metadata,
                                    logger=None
                                    )
    if option==1:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(partial_function,
                                    iterator)
    elif option==2:
        # job_queue = multiprocessing.Queue(maxsize=8)
        # results   = multiprocessing.Queue()
        # for item in iterator:
            # job_queue.put(item)
        logger.info('Starting LSF fitting')
        manager = multiprocessing.Manager()
        inq = manager.Queue()
        outq = manager.Queue()
    
        # construct the workers
        nproc = multiprocessing.cpu_count()
        logger.info(f"Using {nproc} workers")
        workers = [Worker(str(name+1), partial_function,inq, outq,logger) 
                   for name in range(nproc)]
        for worker in workers:
            worker.start()
    
        # add data to the queue for processing
        work_len = len(iterator)
        for item in iterator:
            # print(f"Item before putting into queue: {item}")
            inq.put(item)
    
        while outq.qsize() < work_len:
            # waiting for workers to finish
            done = outq.qsize()
            progress = done/(work_len)
            time_elapsed = time.time() - time_start
            progress_bar.update(progress,name=f'LSF_2d {scale} {iteration}',
                               time=time_elapsed,
                               logger=None)
            
            # print("Waiting for workers. Out queue size {}".format(outq.qsize()))
            time.sleep(1)
    
        # clean up
        for worker in workers:
            worker.terminate()
    
        # print the outputs
        results = []
        while not outq.empty():
            results.append(outq.get())
    
    elif option == 3:
        logger.info('Starting distributed LSF fitting via Ray')
        
        # 1. Initialize Ray (Auto-detects laptop cores or HPC cluster)
        if not ray.is_initialized():
            ray.init()
    
        # 2. Put large spectral arrays into the Object Store [7]
        x2d_ref = ray.put(x2d)
        flx2d_ref = ray.put(flx2d)
        err2d_ref = ray.put(err2d)
        logger.info('Ray: spectral arrays placed into Object Store')
        # 3. Launch tasks for every segment in the echelle orders [3, 8]
        futures = [
            model_1s_remote.remote(
                item, item[9], item[10], # Unpack (od, pixl, pixr) from iterator [4]
                x2d_ref, flx2d_ref, err2d_ref, 
                numiter=iter_center,
                metadata=metadata,
                **kwargs
            ) 
            for item in iterator
        ]
    
        # 4. Asynchronous collection of results
        results = ray.get(futures)
        
    for i,lsf1s_out in enumerate(results):
        if lsf1s_out[0] == None:
            msg = f"LSF1s model order {lsf1s_out[1]} segm {lsf1s_out[2]} failed"
            logger.critical(msg)
        else:
            lsf2d[i]=copy_lsf1s_data(lsf1s_out[0],lsf2d[i])
    worktime = (time.time() - time_start)
    h, m, s = progress_bar.get_time(worktime)
    logger.info(f"Total time elapsed = {h:02d}h {m:02d}m {s:02d}s")
    
    if save_fits:
        
        # Save GP parameters and data
        lsf_filepath = hio.get_fits_path('lsf',spec.filepath)
        lio.write_lsf_to_fits(lsf2d, lsf_filepath, f"{scale}_gp",
                              version=version,
                              clobber=clobber)   
        # Save LSF numerical models
        nummodel_lsf = numerical_models(lsf2d,xrange=(-6,6),subpix=50)
        lio.write_lsf_to_fits(nummodel_lsf, lsf_filepath, f"{scale}_model",
                              version=version,
                              clobber=clobber)   
    gc.collect()
    
    return lsf2d
# def worker(input_queue,output_queue,function):
#     item = input_queue.get(timeout=10)
#     result = function(*item)
#     output_queue.put(result)
#     return None

class Worker(multiprocessing.Process):
    """
    Simple worker.
    """

    def __init__(self, name, function, in_queue, out_queue,logger=None):
        super(Worker, self).__init__()
        self.name = name
        self.function = function
        self.in_queue = in_queue
        self.out_queue = out_queue
        logger = logger if logger is not None else logging.getLogger(__name__)
        self.logger = logger.getChild("worker_"+name)

    def run(self):
        while True:
            # grab work; do something to it (+1); then put the result on the output queue
            item = self.in_queue.get()
            # print(f'item after queue.get = {item}')
            result = self.function(*item)
            self.out_queue.put(result)
            
            
def get_lsf1s_numerical_model(lsf1s_gp,x_array):
    y_array,sct_array = evaluate_lsf1s(lsf1s_gp,x_array)
    return y_array, sct_array

def numerical_models(lsf1d_gp,xrange=(-6,6),subpix=50):
    from harps.containers import lsf_spline
    x_min, x_max = xrange
    numsegs = len(lsf1d_gp)
    npts    = (x_max - x_min) * subpix + 1
    lsf1d_model = lsf_spline(numsegs, npts)
    
    x_array = np.linspace(x_min,x_max,npts)
    lsf1d_model['x']=x_array
    for i,lsf1s_gp in enumerate(lsf1d_gp):
        y_array,sct_array     = get_lsf1s_numerical_model(lsf1s_gp,x_array)
        lsf1d_model[i]['y'] = y_array
        lsf1d_model[i]['scatter'] = sct_array
        names = lsf1d_model.dtype.names
        for name in names:
            if name not in ['x','y','scatter']:
                lsf1d_model[i][name] = lsf1s_gp[name]
        progress_bar.update(i/(len(lsf1d_gp)-1),'numerical model')
    return lsf1d_model



def evaluate_GP(GP,y_data,x_test):
    _, cond = GP.condition(y_data,X_test=x_test)
    mean = cond.mean
    var  = jnp.sqrt(cond.variance)
    
    return mean, var

def build_scatter_GP_from_lsf1s(lsf1s):
    scatter    = read.scatter_from_lsf1s(lsf1s)
    scatter_gp = lsfgp.build_scatter_GP(scatter[0],
                                         X=scatter[1],
                                         Y_err=scatter[3])
    return scatter_gp

def evaluate_scatter_GP_from_lsf1s(lsf1s,x_test):
    theta_sct, sct_x, sct_y, sct_yerr  = read.scatter_from_lsf1s(lsf1s)
    sct_gp = lsfgp.build_scatter_GP(theta_sct,sct_x,sct_yerr)
   
    return evaluate_GP(sct_gp, sct_y, x_test)


    
def build_LSF_GP_from_lsf1s(lsf1s,return_scatter=False):
    theta_LSF, data_x, data_y, data_yerr = read.LSF_from_lsf1s(lsf1s)
    scatter  = read.scatter_from_lsf1s(lsf1s)
    LSF_gp = lsfgp.build_LSF_GP(theta_LSF, data_x, data_y,
                                data_yerr,scatter=scatter)
    if return_scatter:
        return LSF_gp, scatter
    else:
        return LSF_gp

def evaluate_LSF_GP_from_lsf1s(lsf1s,x_test):
    theta_LSF, data_x, data_y, data_yerr = read.LSF_from_lsf1s(lsf1s)
    scatter = read.scatter_from_lsf1s(lsf1s)
    LSF_gp = lsfgp.build_LSF_GP(theta_LSF, data_x, data_y,
                                data_yerr,scatter=scatter)
    
    return evaluate_GP(LSF_gp, data_y, x_test)




def evaluate_lsf1s(lsf1s_gp,x_test):
    return evaluate_LSF_GP_from_lsf1s(lsf1s_gp,x_test)
    

# def lsf_1d(fittype,linelist1d,x1d_stacked,flx1d_stacked,err1d_stacked,
#            iter_center=5,numseg=16,model_scatter=True,metadata=None):
    
    
#     plot=False; save_plot=False
#     # if scale=='pixel':
#     #     x1d = pix1d
#     # elif scale=='velocity':
#     #     x1d = vel1d
#     metadata_=dict(
#         # order=od,
#         # scale=scale,
#         model_scatter=model_scatter,
#         # iteration=iteration,
#         )
#     if metadata is not None:
#         metadata.update(metadata_)
#     else:
#         metadata = metadata_
#     lsf1d=models_1d(x1d_stacked,flx1d_stacked,err1d_stacked,
#                               numseg=numseg,
#                               numiter=iter_center,
#                               minpts=15,
#                               model_scatter=model_scatter,
#                               minpix=None,maxpix=None,
#                               filter=None,plot=plot,
#                               metadata=metadata,
#                               save_plot=save_plot)
    
#     return lsf1d


import itertools

def get_most_likely_lsf2d(lsfpath,scale,nbo=72,nseg=16):
    data = {}
    with FITS(lsfpath) as hdul:
        for ext in hdul:
            extname = ext.get_extname()
            extver  = ext.get_extver()
            # if extver==511: continue
            if extname==f'{scale}_gp':
                data[extver] = ext.read()
    numver = len(data)
    dtype = np.dtype([('version',int, (numver)),
                      ('order',int, ()),
                      ('segm',int, ()),
                      ('logL',np.float32, (numver)),
                      ('loc',int, (numver)),
                      ])
    
    array = np.zeros(nbo*nseg,dtype=dtype)
    comb=itertools.product(np.arange(nbo),np.arange(nseg))
    for i,(od,segm) in enumerate(comb):
        for j, (ver,lsf2d) in enumerate(data.items()):
            
            odver = np.unique(lsf2d['order'])
            segver = np.unique(lsf2d['segm'])
            if od in odver and segm in segver:
                pass
            else:
                continue
            
            array[i]['order']=od
            array[i]['segm']=segm
            cut = np.where((lsf2d['order']==od)&(lsf2d['segm']==segm))[0]
            array['version'][i,j] = ver
            if len(cut)>0:
                keep = np.argmax(lsf2d[cut]['logL'])
                cut  = cut[keep]
            array['loc'][i,j] = cut
            
            try: 
                array['logL'][i,j] = lsf2d[cut]['logL']
            except:
                array['logL'][i,j] = gp_aux.get_likelihood_from_lsf1s(lsf2d[cut])
    nonzero = np.where(array['order']!=0)
    array = array[nonzero]
    # find the location of the maximum in log likelihood
    best = np.argmax(array['logL'],axis=1)
    
    most_likely_lsf2d = []
    for i,entry in enumerate(array):
        
        veritem = entry['version'][best[i]]
        locitem = entry['loc'][best[i]]
        print(f"Most likely IP at ({entry['order']},{entry['segm']}) is version {veritem}")
        most_likely_lsf2d.append(data[veritem][locitem])
    
    return np.hstack(most_likely_lsf2d)

def save_most_likely(lsf_filepath,scale,nbo=72,nseg=16,save_filepath=None,
                     clobber=False):
    most_likely_lsf2d = get_most_likely_lsf2d(lsf_filepath,scale,nbo=nbo,nseg=sneg)
    
    save_filepath = save_filepath if save_filepath is not None else lsf_filepath
    lio.write_lsf_to_fits(most_likely_lsf2d, save_filepath, f"{scale}_gp",
                          version=1,
                          clobber=clobber)  
    nummodel_lsf = numerical_models(most_likely_lsf2d,xrange=(-6,6),subpix=50)
    lio.write_lsf_to_fits(nummodel_lsf, save_filepath, f"{scale}_model",
                          version=1,
                          clobber=clobber)  
    return most_likely_lsf2d
    