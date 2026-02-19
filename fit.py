#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:07:32 2018

@author: dmilakov
"""

from harps.core import odr, np, os, plt, curve_fit, json, leastsq

from harps.constants import c
from harps.plotter import Figure2

import harps.settings as hs
import harps.emissionline as emline
import harps.containers as container
import harps.functions as hf
import harps.functions.spectral as specfunc
import harps.version as hv
import harps.lsf.fit as lsfit
import harps.gaps as hg
import warnings
import logging

import numpy.polynomial.legendre as leg

from scipy.linalg import svd, cholesky, solve_triangular, LinAlgError
from scipy.optimize import least_squares, OptimizeWarning

import warnings


quiet = hs.quiet
version = hs.version
#==============================================================================



#==============================================================================
    
#                           G A P S    F I L E   
    
#==============================================================================
#def read_gaps(filepath=None):
#    if filepath is not None:
#        filepath = filepath  
#    else:
#        dirpath = hs.get_dirname('gaps')
#        filepath = os.path.join(dirpath,'gaps.json')
#    with open(filepath,'r') as json_file:
#        gaps_file = json.load(json_file)
#    gaps = []
#    for block in range(1,4):
#        orders  = gaps_file['orders{}'.format(block)]
#        norders = len(orders)
#        block_gaps = container.gaps(norders)
#        block_gaps['order'] = orders
#        block_gaps['gaps']  = gaps_file['block{}'.format(block)]
#        gaps.append(block_gaps)
#    gaps = np.hstack(gaps)
#    return np.sort(gaps)
#    
#    
#
#def get_gaps(order,filepath=None):
#    gapsfile  = read_gaps(filepath)
#    orders   = np.array(gapsfile[:,0],dtype='i4')
#    gaps2d   = np.array(gapsfile[:,1:],dtype='f8')
#    selection = np.where(orders==order)[0]
#    gaps1d    = gaps2d[selection]
#    return np.ravel(gaps1d)
#
#
#def introduce_gaps(centers,gaps1d,npix=4096):
#    if np.size(gaps1d)==0:
#        return centers
#    elif np.size(gaps1d)==1:
#        gap  = gaps1d
#        gaps = np.full((7,),gap)
#    else:
#        gaps = gaps1d
#    centc = np.copy(centers)
#    
#    for i,gap in enumerate(gaps):
#        ll = (i+1)*npix/(np.size(gaps)+1)
#        cut = np.where(centc>ll)[0]
#        centc[cut] = centc[cut]-gap
#    return centc
#==============================================================================
#
#                         C U R V E      F I T T I N G                  
#
#==============================================================================
def _wrap_func(func, xdata, ydata, transform, fargs={}):
    if transform is None:
        def func_wrapped(params,fargs={}):
            return func(xdata, *params,**fargs) - ydata
    elif transform.ndim == 1:
        def func_wrapped(params,fargs={}):
            return transform * (func(xdata, *params,**fargs) - ydata)
    else:
        # Chisq = (y - yd)^T C^{-1} (y-yd)
        # transform = L such that C = L L^T
        # C^{-1} = L^{-T} L^{-1}
        # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
        # Define (y-yd)' = L^{-1} (y-yd)
        # by solving
        # L (y-yd)' = (y-yd)
        # and minimize (y-yd)'^T (y-yd)'
        def func_wrapped(params,fargs={}):
            return solve_triangular(transform, func(xdata, *params,**fargs) - ydata, lower=True)
    return func_wrapped
def _wrap_jac(jac, xdata, transform):
    if transform is None:
        def jac_wrapped(params):
            return jac(xdata, *params)
    elif transform.ndim == 1:
        def jac_wrapped(params):
            return transform[:, np.newaxis] * np.asarray(jac(xdata, *params))
    else:
        def jac_wrapped(params):
            return solve_triangular(transform, np.asarray(jac(xdata, *params)), lower=True)
    return jac_wrapped


def _initialize_feasible(lb, ub):
    p0 = np.ones_like(lb)
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])

    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1

    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1

    return p0
def curve(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False,
              check_finite=True, bounds=(-np.inf, np.inf), method='lm',
              jac=None, fargs=None, **kwargs):
    if p0 is None:
        # determine number of parameters by inspecting the function
        from scipy._lib._util import getargspec_no_self as _getargspec
        args, varargs, varkw, defaults = _getargspec(f)
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1
    else:
        p0 = np.atleast_1d(p0)
        n = p0.size


    # NaNs can not be handled
    if check_finite:
        ydata = np.asarray_chkfinite(ydata)
    else:
        ydata = np.asarray(ydata)

    if isinstance(xdata, (list, tuple, np.ndarray)):
        # `xdata` is passed straight to the user-defined `f`, so allow
        # non-array_like `xdata`.
        if check_finite:
            xdata = np.asarray_chkfinite(xdata)
        else:
            xdata = np.asarray(xdata)

    # Determine type of sigma
    if sigma is not None:
        sigma = np.asarray(sigma)

        # if 1-d, sigma are errors, define transform = 1/sigma
        if sigma.shape == (ydata.size, ):
            transform = 1.0 / sigma
        # if 2-d, sigma is the covariance matrix,
        # define transform = L such that L L^T = C
        elif sigma.shape == (ydata.size, ydata.size):
            try:
                # scipy.linalg.cholesky requires lower=True to return L L^T = A
                transform = cholesky(sigma, lower=True)
            except LinAlgError:
                raise ValueError("`sigma` must be positive definite.")
        else:
            raise ValueError("`sigma` has incorrect shape.")
    else:
        transform = None

    func = _wrap_func(f, xdata, ydata, transform, fargs=fargs)
    if callable(jac):
        jac = _wrap_jac(jac, xdata, transform)
    elif jac is None and method != 'lm':
        jac = '2-point'

    if method == 'lm':
        # Remove full_output from kwargs, otherwise we're passing it in twice.
        return_full = kwargs.pop('full_output', False)
        res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
        popt, pcov, infodict, errmsg, ier = res
        cost = np.sum(infodict['fvec'] ** 2)
        if ier not in [1, 2, 3, 4]:
            raise RuntimeError("Optimal parameters not found: " + errmsg)
    else:
        # Rename maxfev (leastsq) to max_nfev (least_squares), if specified.
        if 'max_nfev' not in kwargs:
            kwargs['max_nfev'] = kwargs.pop('maxfev', None)

        res = least_squares(func, p0, jac=jac, bounds=bounds, method=method,
                            **kwargs)

        if not res.success:
            raise RuntimeError("Optimal parameters not found: " + res.message)

        cost = 2 * res.cost  # res.cost is half sum of squares!
        popt = res.x

        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        return_full = False

    warn_cov = False
    if pcov is None:
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warn_cov = True
    elif not absolute_sigma:
        if ydata.size > p0.size:
            s_sq = cost / (ydata.size - p0.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(np.inf)
            warn_cov = True

    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning)

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov

# def residuals(function,data,pars,weights):
#         ''' Returns the residuals of individual data points to the model.
#         Args:
#         ----
#             pars: tuple (amplitude, mean, sigma) of the model
#         Returns:
#         -------
#              1d array (len = len(xdata)) of residuals
#         '''
# #        model = self.model(*pars)
#         obsdata = self.ydata[1:-1]
#         resids  = ((obsdata - self.model(pars))/self.yerr[1:-1])
#         return resids

#==============================================================================
#
#                         L I N E      F I T T I N G                  
#
#==============================================================================
# default_line_class = 'SimpleGaussian'
default_line_class = 'SingleGaussian'
def gauss(x,flux,error,
          line_class=default_line_class,
          output_model=False,
          xscale='pixel',
          npars=None,
          *args,**kwargs):
    assert np.size(x)==np.size(flux)
    line_model   = getattr(emline,line_class)
    line         = line_model()    
    
    npars  = npars if npars is not None else hs.npars
    pars   = np.full(npars,np.nan)
    errors = np.full(npars,np.nan)
    chisq   = np.nan
    chisqnu = np.nan
    model  = np.full_like(flux,np.nan)
    success = False
    integral = np.nan
    
    if np.all(x==0.):
        
        model = np.zeros_like(flux)
        if output_model:
            
            return success, pars, errors, chisq, chisqnu,integral, model
        else:
            return success, pars, errors, chisq, chisqnu,integral
        
    
    # print(f"{np.min(x):>10.4f} {np.max(x):>10.4f} {len(x):>4d} before fit")
    try:
        pars, errors = line.fit(x,flux,error,bounded=False)
        chisqnu      = line.rchi2
        chisq        = line.cost
        model = line.evaluate(pars)
        success = True
        integral = pars[0]*(pars[2]*np.sqrt(2*np.pi))
    except:
        print(f"{np.min(x):>10.4f} {np.max(x):>10.4f} {len(x):>4d} exception")
        # print(np.min(x),np.max(x))
        # print(pars)
        # print(flux)
        # print(error)
        
        plt.figure()
        plt.plot(x,flux)
        # return
        pass
    
    if output_model:
        
        return success, pars, errors, chisq, chisqnu,integral, model
    else:
        return success, pars, errors, chisq, chisqnu,integral
def assign_weights(pixels):
        weights  = np.zeros_like(pixels)
        binlims  = [-5,-2.5,2.5,5]
        idx      = np.digitize(pixels,binlims)
        cut1     = np.where(idx==2)[0]
        cutl     = np.where(idx==1)[0]
        cutr     = np.where(idx==3)[0]
        # ---------------
        # weights are:
        #  = 1,           -2.5<=x<=2.5
        #  = 0,           -5.0>=x & x>=5.0
        #  = linear[0-1]  -5.0<x<2.5 & 2.5>x>5.0 
        # ---------------
        weights[cut1] = 1
        weights[cutl] = 0.4*(5+pixels[cutl])
        weights[cutr] = 0.4*(5-pixels[cutr])
        return weights
    

def lsf(x1l,flx1l,err1l,LSF1d,interpolate=True,
        output_model=False,output_rsd=False,plot=False,*args,**kwargs):
    '''
    Calls harps.lsf.gp_aux.fit_lsf2line

    Parameters
    ----------
    pix : array
        the pixel coordinates.
    flux : array
        the spectral array.
    background : array
        spectral background array.
    error : array
        spectral error array.
    lsf1d : structured numpy array. 
        Contains  the 1d LSF. Not harps.lsf.classes.LSF class.
    interpolate : bool, optional
        Controls whether interpolation for a local LSF is done. 
        The default is True.
    output_model : bool, optional
        Controls whether a line model is output. The default is False.
    plot : bool, optional
        Controls plotting of the fit. The default is False.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    return lsfit.line(x1l, flx1l, err1l, LSF1d,
                      interpolate=interpolate,
                      output_model=output_model,
                      output_rsd = output_rsd,
                      plot=plot
                      *args, **kwargs)
    # import harps.lsf.gp_aux as gp_aux
    # return gp_aux.fit_lsf2line(pix, flux, error, lsf1d,
    #                            interpolate=interpolate,
    #                            output_model=output_model,
    #                            plot=plot,
    #                            *args, **kwargs)
    

#==============================================================================
#
#        W A V E L E N G T H     D I S P E R S I O N      F I T T I N G                  
#
#==============================================================================
def ordinary(centers,wavelengths,cerror,werror,polyord):
    numcen = np.size(centers)
    assert numcen>polyord, "No. centers too low, {}".format(numcen)
    # beta0 is the initial guess
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=np.RankWarning)
        beta0 = np.flip(np.polyfit(centers,wavelengths,polyord,cov=False))
    data  = odr.RealData(centers,wavelengths,sx=cerror,sy=werror)
    model = odr.polynomial(order=polyord)
    ODR   = odr.ODR(data,model,beta0=beta0)
    out   = ODR.run()
    #pars  = out.beta
    #errs  = out.sd_beta
    #ssq   = out.sum_square
    return out #pars, errs, ssq
def legval(B,x):
    return leg.legval(x,B)
def poly(polytype,centers,wavelengths,cerror,werror,polyord):
    numcen = np.size(centers)
    assert numcen>polyord, "No. centers too low, {}".format(numcen)
    assert polytype in ['ordinary','legendre']
    if polytype=='ordinary':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=np.RankWarning)
            beta0 = np.flip(np.polyfit(centers,wavelengths,polyord,cov=False))
        model = odr.polynomial(order=polyord)
    if polytype=='legendre':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=np.RankWarning)
            beta0 = leg.legfit(centers,wavelengths,polyord,full=False)
        model = odr.Model(legval)
    data  = odr.RealData(centers,wavelengths,sx=cerror,sy=werror)
    ODR   = odr.ODR(data,model,beta0=beta0)
    out   = ODR.run()
    return out

# Assumption: Frequencies are known with 1MHz accuracy
freq_err = 2e4
def dispersion(linelist,version,fittype,npix,errorfac=1,polytype='ordinary',
               anchor_offset=None, 
               limit=None,q=None):
    """
    Fits the wavelength solution to the data provided in the linelist.
    Calls 'dispersion1d' for all orders in linelist.
    
    
    Input:
    -------
        linelist  : numpy structured array
        version   : integer
        fittype   : centers to use, 'gauss' or 'lsf'
        npix      : number of pixels, used for normalisation to domain (-1,1).
        errorfac  : multiplier for the center error array
        
    Returns:
    -------
        wavesol2d : a structured array with coefficients
        
    """
    anchor_offset = anchor_offset if anchor_offset is not None else 0.0
    orders  = np.unique(linelist['order'])
    polyord, gaps, do_segment = hv.unpack_integer(version,'wavesol')
    disperlist = []
    if gaps:
        gaps2d     = hg.read_gaps(None)
    plot=False
    if plot:
        plt.figure()
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
    linelist0 = container.Generic(linelist)
    #Linelist0 = hf.remove_bad_fits(linelist,fittype)
    
    fittype = fittype+'_pix'
    for i,order in enumerate(orders):
        linelis1d_dirty = linelist0[order].values
        if len(linelis1d_dirty)==0:
            print('No lines found')
            continue
        linelis1d = hf.remove_bad_fits(linelis1d_dirty,fittype,limit=limit,q=q)
        
        centers1d = linelis1d[fittype][:,1]
        
        if np.sum(centers1d) ==0:
            # print(f"No lines to fit in order {order}")
            continue
        
        cerrors1d = errorfac*linelis1d[f'{fittype}_err'][:,1]
        wavelen1d = hf.freq_to_lambda(linelis1d['freq']+anchor_offset)
        werrors1d = 1e10*(c/((linelis1d['freq'])**2)) * freq_err
        # if order==68:
        #     for values in np.transpose([centers1d,cerrors1d,wavelen1d,werrors1d]):
        #         print(*values)
        if gaps:
            if plot:
                centersold = centers1d
            cut       = np.where(gaps2d['order']==order)
            gaps1d    = gaps2d[cut]['gaps'][0]
            centers1d = hg.introduce_gaps(centers1d,gaps1d)
            if plot:
                plt.scatter(centersold,centers1d-centersold,s=2,c=[colors[i]])
        else:
            pass
        di1d      = dispersion1d(centers1d,wavelen1d,
                              cerrors1d,werrors1d,
                              version,polytype,npix,plot=plot)
        
        di1d['order'] = order
        disperlist.append(di1d)
    dispersion2d = np.hstack(disperlist)
    return dispersion2d


        
def dispersion1d(centers,wavelengths,cerror,werror,version,polytype,
                 npix,plot=False,verbose=False):
    """
    Uses 'segment' to fit polynomials of degree given by polyord keyword.
    
    
    If version=xx1, divides the data into 8 segments, each 512 pix wide. 
    A separate polyonomial solution is derived for each segment.
    """
    polyord, gaps, do_segment = hv.unpack_integer(version,'wavesol')
    if do_segment==True:
        numsegs = 8
    else:
        numsegs = 1
            
    # remove NaN
    #centers     = hf.removenan(centers)
    #wavelengths = hf.removenan(wavelengths)
    #cerror      = hf.removenan(cerror)
    #werror      = hf.removenan(werror)
    seglims  = np.linspace(npix//numsegs,npix,numsegs)
    binned   = np.digitize(centers,seglims,right=True)
    # new container for coefficients
    coeffs = container.coeffs(polyord,numsegs)
    for i in range(numsegs):
        sel = np.where(binned==i)[0]
        pixl   = seglims[i]-npix//numsegs
        pixr   = seglims[i]
        cen_ = centers[sel]
        wav_ = wavelengths[sel]
        cer_ = cerror[sel]
        wer_ = werror[sel]
        sorter = np.argsort(cen_)
        # if i==7: print(cen_[sorter])
        output = segment(cen_[sorter],wav_[sorter],
                               cer_[sorter],wer_[sorter],
                               polyord,polytype,
                               xmin=pixl,
                               xmax=pixr,
                               plot=plot)
        pars, errs, chisq, chisqnu, success = output
        p = len(pars)
        n = len(cen_)
        # print(n)
        
        coeffs[i]['pixl']   = pixl
        coeffs[i]['pixr']   = pixr
        if (n-p-1)<1 or not success:
            if verbose: print(f'Need {p} points to fit segment {i}, {n} provided')
            continue
        else:
            pass
        coeffs[i]['pars']   = pars
        coeffs[i]['errs']   = errs
        coeffs[i]['chisq']  = chisq
        coeffs[i]['chisqnu']= chisqnu
        coeffs[i]['npts']   = n
        coeffs[i]['aicc']   = chisq + 2*p + 2*p*(p+1)/(n-p-1)
    return coeffs

    
def segment(centers,wavelengths,cerror,werror,polyord,polytype,xmin,xmax,
            plot=True,verbose=False,
            **plot_kwargs):
    """
    Fits a polynomial to the provided data and errors.
    Uses scipy's Orthogonal distance regression package in order to take into
    account the errors in both x and y directions.
    
    Returns:
    -------
        coef : len(polyord) array
        errs : len(polyord) array
    """
    logger = logging.getLogger().getChild('fit.segment')
    numcen = np.size(centers)
    success = False
    
    if numcen>polyord:
        # print('Sufficient number of points')
        pass
    else:
        # pars    = np.zeros(polyord+1)
        pars = np.full(polyord+1,np.nan)
        if wavelengths.size>0:
            pars[-1] = np.mean(wavelengths)
        errs    = np.full(polyord+1,np.inf)
        chisq   = -1
        chisqnu = -1
        return pars, errs, chisq, chisqnu, success
    
    
    arenan = np.isnan(centers)
    centers     = hf.contract(centers[~arenan],(xmin,xmax))
    wavelengths = wavelengths[~arenan]
    cerror      = hf.contract(cerror[~arenan],(xmin,xmax))
    werror      = werror[~arenan]
#    if plot:
#        plt.figure()
#        plt.errorbar(centers,wavelengths,yerr=werror,xerr=cerror,ms=2,ls='',capsize=4)
#        [plt.axvline(512*i,ls='--',lw=0.3,c='k') for i in range(9)]
    # clip0: points kept in previous iteration
    clip0 = np.full_like(centers,False,dtype='bool')
    # clip1: points kept in this iteration
    clip1 = np.full_like(centers,True,dtype='bool')
    j = 0
    # repeat until no more points are being rejected or a max of 10 times
    
    break_cond = False
    while not break_cond and j<10:
        j+=1
        previous_success = success
        
        clip0        = np.ravel(clip1)
        centers0     = centers[clip0]
        wavelengths0 = wavelengths[clip0]
        cerror0      = cerror[clip0]
        werror0      = werror[clip0]
        
        try:
            model        = poly(polytype,centers0,wavelengths0,cerror0,werror0,
                                polyord)
            success      = True
            # print([np.shape(array) for array in [clip0]])
        except:
            success      = False
            
            
        if success:
            pars         = model.beta
            errs         = model.sd_beta
            chisqnu      = model.res_var
            nu           = len(wavelengths0) - len(pars)
            chisq        = chisqnu * nu
            if polytype=='ordinary':
                residuals = wavelengths-np.polyval(np.flip(pars),centers)
            elif polytype=='legendre':
                residuals = wavelengths-leg.legval(centers,pars) 
            previous_success = True
            save = dict(
                pars = pars,
                errs = errs,
                chisqnu = chisqnu,
                chisq = chisq,
                )
            # clip outliers from the residuals and check condition
            outliers     = hf.is_outlier(residuals)
            clip1        = np.ravel(~outliers)
            break_cond = np.sum(clip0)==np.sum(clip1)
        else:
            logger.warning(f'Failed at iteration {j}, previous success = {previous_success}')
            logger.warning([clip0,clip1,np.shape(centers0),np.shape(wavelengths0)])
            if previous_success:
                pars = save['pars']
                errs = save['errs']
                chisqnu = save['chisqnu']
                chisq = save['chisq']
            else:
                pars         = np.full(polyord+1,np.nan)
                errs         = np.full(polyord+1,np.inf)
                chisqnu      = np.inf
                chisq        = np.inf
            break_cond = True
        
        
        
        
        
        # print(j,success,clip1,[np.shape(array) for array in [residuals,clip0,clip1]])
        if plot:# and np.sum(outliers)>0:
            plotter = Figure2(2,1,**plot_kwargs)
            ax0     = plotter.add_subplot(0,1,0,1)
            ax1     = plotter.add_subplot(1,2,0,1,sharex=ax0)
#            plt.figure()
            ax0.plot(centers[clip1],wavelengths[clip1],
                     marker = 'o',ms=8,ls='',c='C1')
            ax1.scatter(centers,residuals,s=2)
            ax1.scatter(centers[outliers],residuals[outliers],
                        s=16,marker='x',c='C2')
            x = np.linspace(np.min(centers),np.max(centers),200)
            
            if polytype=='ordinary':
                y_=np.polyval(pars[::-1],centers[clip1])
                y = np.polyval(np.flip(pars),x)
            elif polytype=='legendre':
                y_=leg.legval(centers[clip1],pars)
                y =leg.legval(x,pars)
            
            
            ax0.plot(x,y)
    return pars, errs, chisq, chisqnu, success

# =============================================================================
    
#                              S  P  L  I  N  E
    
# =============================================================================

def dispersion1d_spline(centers,wavelengths,cerror,werror,npix,node_dist=64,
                        plot=False):
    import scipy.interpolate as interpolate
    sorter = np.argsort(centers)
    nodes = np.arange(node_dist,npix-node_dist,node_dist)
    tck,fp,ier,msg = interpolate.splrep(centers[sorter], wavelengths[sorter],
                                        w = 1./werror[sorter],
                                        k = 3,
                                        t = nodes,
                                        task=-1,
                                        full_output=True
                                           )
    xgrid = np.arange(npix)
    wav = interpolate.splev(xgrid,tck)
    #print(fp,ier,msg)
    if plot:
        # print(len(idx))
        plt.figure()
        # plt.plot(xarray,yarray,drawstyle='steps-mid',lw=0.3)
        plt.errorbar(centers,wavelengths,werror,marker='.',c='r',ls='',alpha=0.3)
        
        
        plt.scatter(tck[0],interpolate.splev(tck[0],tck),marker='o',c='k')
        
        
        
        plt.plot(xgrid,wav,c='r',lw=2.)
    return wav

# https://stackoverflow.com/questions/51321100/python-natural-smoothing-splines

#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.linear_model import LinearRegression
#from sklearn.pipeline import Pipeline
## 
#
#def spline(*args,**kwargs):
#    return get_natural_cubic_spline_model(*args,**kwargs)
#
#def get_natural_cubic_spline_model(x, y, minval=None, maxval=None, n_knots=None, knots=None):
#    """
#    Get a natural cubic spline model for the data.
#
#    For the knots, give (a) `knots` (as an array) or (b) minval, maxval and n_knots.
#
#    If the knots are not directly specified, the resulting knots are equally
#    space within the *interior* of (max, min).  That is, the endpoints are
#    *not* included as knots.
#
#    Parameters
#    ----------
#    x: np.array of float
#        The input data
#    y: np.array of float
#        The outpur data
#    minval: float 
#        Minimum of interval containing the knots.
#    maxval: float 
#        Maximum of the interval containing the knots.
#    n_knots: positive integer 
#        The number of knots to create.
#    knots: array or list of floats 
#        The knots.
#
#    Returns
#    --------
#    model: a model object
#        The returned model will have following method:
#        - predict(x):
#            x is a numpy array. This will return the predicted y-values.
#    """
#
#    if knots:
#        spline = NaturalCubicSpline(knots=knots)
#    else:
#        maxval = maxval if maxval is not None else np.max(x)
#        minval = minval if minval is not None else np.min(x)
#        spline = NaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)
#
#    p = Pipeline([
#        ('nat_cubic', spline),
#        ('regression', LinearRegression(fit_intercept=True))
#    ])
#
#    p.fit(x, y)
#
#    return p
#
#
#class AbstractSpline(BaseEstimator, TransformerMixin):
#    """Base class for all spline basis expansions."""
#
#    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
#        if knots is None:
#            if not n_knots:
#                n_knots = self._compute_n_knots(n_params)
#            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
#            max, min = np.max(knots), np.min(knots)
#        self.knots = np.asarray(knots)
#
#    @property
#    def n_knots(self):
#        return len(self.knots)
#
#    def fit(self, *args, **kwargs):
#        return self
#
#
#class NaturalCubicSpline(AbstractSpline):
#    """Apply a natural cubic basis expansion to an array.
#    The features created with this basis expansion can be used to fit a
#    piecewise cubic function under the constraint that the fitted curve is
#    linear *outside* the range of the knots..  The fitted curve is continuously
#    differentiable to the second order at all of the knots.
#    This transformer can be created in two ways:
#      - By specifying the maximum, minimum, and number of knots.
#      - By specifying the cutpoints directly.  
#
#    If the knots are not directly specified, the resulting knots are equally
#    space within the *interior* of (max, min).  That is, the endpoints are
#    *not* included as knots.
#    Parameters
#    ----------
#    min: float 
#        Minimum of interval containing the knots.
#    max: float 
#        Maximum of the interval containing the knots.
#    n_knots: positive integer 
#        The number of knots to create.
#    knots: array or list of floats 
#        The knots.
#    """
#
#    def _compute_n_knots(self, n_params):
#        return n_params
#
#    @property
#    def n_params(self):
#        return self.n_knots - 1
#
#    def transform(self, X, **transform_params):
#        X_spl = self._transform_array(X)
#        
#        return X_spl
#
#    def _make_names(self, X):
#        first_name = "{}_spline_linear".format(X.name)
#        rest_names = ["{}_spline_{}".format(X.name, idx)
#                      for idx in range(self.n_knots - 2)]
#        return [first_name] + rest_names
#
#    def _transform_array(self, X, **transform_params):
#        X = np.atleast_1d(X).squeeze()
#        try:
#            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
#        except IndexError: # For arrays with only one element
#            X_spl = np.zeros((1, self.n_knots - 1))
#        X_spl[:, 0] = X.squeeze()
#
#        def d(knot_idx, x):
#            def ppart(t): return np.maximum(0, t)
#
#            def cube(t): return t*t*t
#            numerator = (cube(ppart(x - self.knots[knot_idx]))
#                         - cube(ppart(x - self.knots[self.n_knots - 1])))
#            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
#            return numerator / denominator
#
#        for i in range(0, self.n_knots - 2):
#            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
#        return X_spl