#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:26:15 2018

@author: dmilakov
"""
import numpy as np
import logging
import matplotlib.pyplot as plt
import traceback

from harps.constants import c
from harps.background import getbkg

import harps.settings as hs
# import harps.functions as hf
import harps.peakdetect as pkd
import harps.functions.lfc as lfcfunc
import harps.functions.data as datafunc
import harps.functions.spectral as specfunc
import harps.containers as container
import harps.fit as hfit
import harps.emissionline as emline
import harps.lsf as hlsf
import harps.lines_aux as laux
import harps.progress_bar as progress_bar
#from . import curves as curve
import harps.noise as noise

# from numba import jit
import scipy.stats as stats

quiet = hs.quiet
# hs.setup_logging()

frequency_error = 2e4 # Hz

def _make_extname(order):
    return "ORDER{order:2d}".format(order=order)

# def arange_modes_from_coefficients(center1d,coeff1d,reprate,anchor):
#     """
#     Uses the positions of maxima to assign mode numbers to all lines in the 
#     echelle order.
    
#     Uses the ThAr coefficients (in vacuum) to calculate the mode of the central 
#     line.
#     """
#     # warn if ThAr solution does not exist for this order:
#     if np.all(coeff1d)==0:
#         raise UserWarning("ThAr WAVELENGTH SOLUTION DOES NOT EXIST")

#     # total number of lines
#     nlines = len(center1d)
#     # central line
#     ref_index = nlines//2
#     ref_pixel = center1d[ref_index]
#     # calculate frequencies of the central line from ThAr solution
#     ref_wave_thar = hf.polynomial(ref_pixel,*coeff1d)
#     ref_freq_thar = c/ref_wave_thar*1e10
#     # convert frequency into mode number
#     decimal_n = ((ref_freq_thar - (anchor))/reprate)
#     integer_n = np.rint(decimal_n).astype(np.int32)
#     ref_n     = integer_n
#     #print("{0:3d}/{1:3d} (pixel={2:8.4f})".format(ref_index,nlines,ref_pixel))
    
#     # make a decreasing array of modes, where modes[ref_index]=ref_n:
#     aranged  = np.flip(np.arange(nlines))
#     shifted  = aranged - (nlines-ref_index-1)
#     modes    = shifted+ref_n
#     return modes, ref_index

def arange_modes_from_array(center1d,wave1d,reprate,anchor):
    """
    Uses the positions of maxima to assign mode numbers to all lines in the 
    echelle order.
    
    Uses an array of wavelengths to calculate the mode of the central 
    line.
    """
    # warn if ThAr solution does not exist for this order:


    # total number of lines
    nlines = len(center1d)
    assert nlines>=2, f"The number of lines must be >2. Instead, it is {nlines}."
    # central line
    ref_index = nlines//2
    ref_pixel = center1d[ref_index]
    # calculate frequencies of the central line from ThAr solution
    ref_wave_thar = wave1d[int(ref_pixel)]
    ref_freq_thar = c/ref_wave_thar*1e10
    # convert frequency into mode number
    decimal_n = ((ref_freq_thar - (anchor))/reprate)
    integer_n = np.rint(decimal_n).astype(np.int32)
    ref_n     = integer_n
    #print("{0:3d}/{1:3d} (pixel={2:8.4f})".format(ref_index,nlines,ref_pixel))
    
    # make a decreasing array of modes, where modes[ref_index]=ref_n:
    aranged  = np.flip(np.arange(nlines))
    shifted  = aranged - (nlines-ref_index-1)
    modes    = shifted+ref_n
    return modes, ref_index


def detect1d_from_spec(spec,order,fittype=['gauss'],
                       xscale=['pix','wav'],wavereference='LFC',
                       lsf_filepath=None,
                       version=700,
                       redisperse=False,
                       logger=None,plot=False,debug=False,*args,**kwargs):
    
    
    velocity_step = kwargs.pop('velocity_step',0.82)
    
    flx_od = spec['flux'][order]
    # bkg_od = spec['background'][order]
    wav_thar = spec['wavereference'][order]
    linepos  = spec['line_positions']
    cut = np.where(linepos['order']==order)[0]
    # minima1d = spec.extrema[1][order]
    linepos1d = linepos[cut]
    minima1d = np.union1d(linepos1d['pixl'],linepos1d['pixr'])
    
    if wavereference=='ThAr':
        wav_od = wav_thar
        pass
    elif wavereference=='LFC':
        # detect lines in ThAr
        linelist_temp = detect1d_from_spec(spec,order,fittype=fittype,
                                           xscale=xscale,
                                           wavereference='ThAr',
                                           redisperse=False,
                                           logger=logger,
                                           plot=plot,
                                           debug=debug,
                                           *args,
                                           **kwargs)
        # produce a wavelength calibration
        import harps.wavesol as ws
        
        fittype_ = 'gauss'
        centers = linelist_temp[f'{fittype_}_pix'][:,1]
        cerror  = linelist_temp[f'{fittype_}_pix_err'][:,1]
        freq    = linelist_temp['freq']
        wave    = lfcfunc.freq_to_lambda(freq)
        werror  = 1e10*(c/freq**2) * frequency_error
        polytype = 'ordinary'
        npix    = spec.npix
        coeffs_lfc = hfit.dispersion1d(centers,wave,cerror,werror,version,polytype,
                         npix,plot=False,verbose=False)
        wav_lfc = ws.disperse1d(coeffs_lfc,npix,polytype)
        # wav_lfc = ws.polynomial(linelist_temp,version=701,fittype='gauss',
        #                        npix=spec.npix,nord=None,
        #                        full_output=False,*args,**kwargs)[order]
        # and use that calibration
        wav_od = wav_lfc
        # for _ in np.array_split(wav_lfc,32):
        #     print(_)
    line_positions, env1d, bkg1d = pkd.process_spectrum(flx_od,
                                                         plot_main_details=False,
                                                         verbose=debug)
    wav1d, flx1d, err1d = datafunc.prepare_data1d(wav_od, flx_od, bkg1d,
                                                  redisperse=redisperse,
                                                  subbkg=hs.subbkg,
                                                  velocity_step=velocity_step)
    
    keys = dict(
        comb_anchor=spec.lfckeys['comb_anchor'],
        comb_reprate=spec.lfckeys['comb_reprate'],
        segsize=spec.segsize
        )
    try:
        linelist = detect1d_from_array(wav1d,flx1d,err1d,
                                       line_positions,
                                   # minima1d.T,
                                   keys,
                                   fittype=fittype,
                                   xscale=xscale,
                                   logger=logger,
                                   plot=plot,
                                   debug=debug,
                                   *args,
                                   **kwargs)
        linelist['order']=order
        linelist['optord']=spec.optical_orders[order]
    except Exception as e:
        print(f"{e}")
        traceback.print_exc()
        
    
    return linelist, env1d, bkg1d,

def detect2d_from_spec(spec,order=None,fittype=['gauss'],
                       xscale=['pix','wav'],wavereference='LFC',
                       lsf_filepath=None,
                       version=700,
                       redisperse=False,velocity_step=0.82,
                       logger=None,plot=False,debug=False,
                       return_thar_linelist=False,*args,**kwargs):
    
    
    assert wavereference in ['LFC', 'ThAr'], f"Unknown wavelength reference. [LFC or ThAr] but {wavereference} provided."
    orders = spec.prepare_orders(order)
    flx = spec['flux']
    bkg2d = spec['background']
    line_positions = spec['line_positions']
    # bkg=spec('background')
    
    print(fittype, wavereference)
    
    wav_thar = spec['wavereference']
    wav = wav_thar
    if wavereference=='LFC':
        # detect lines in ThAr
        linelist_thar = detect2d_from_spec(spec,order,fittype=['gauss'],
                                           xscale=xscale,
                                           wavereference='ThAr',
                                           redisperse=False,
                                           logger=logger,
                                           plot=plot,
                                           debug=debug,
                                           *args,
                                           **kwargs)
        # produce a wavelength calibration from the linelist produced 
        import harps.wavesol as ws
        wav_lfc = ws.polynomial(linelist_thar,version=version,fittype='gauss',
                               npix=spec.npix,nord=spec.nbo,
                               full_output=False,*args,**kwargs)
        
        # and use that calibration
        wav = wav_lfc
    wav2d, flx2d, err2d = datafunc.prepare_data2d(wav, flx, bkg2d,
                                                  redisperse=redisperse,
                                                  subbkg=hs.subbkg,
                                                  velocity_step=velocity_step)
    
    
    keys = dict(
        comb_anchor=spec.lfckeys['comb_anchor'],
        comb_reprate=spec.lfckeys['comb_reprate'],
        segsize=spec.segsize
        )
    
    linelist2d = []
    for i,od in enumerate(orders):
        cut_od = np.where(line_positions['order']==od)[0]
        if len(cut_od)==0:
            continue
        # if od==52:
        #     spec.plot_spectrum(od)
        
        try:
            linelist1d = detect1d_from_array(wav2d[od],flx2d[od],err2d[od],
                                         line_positions[cut_od],
                                         keys,
                                       fittype=fittype,
                                       xscale=xscale,
                                       logger=logger,
                                       plot=plot,
                                       debug=debug,
                                       *args,
                                       **kwargs)
        except Exception as e:
            print(f"Problem encountered at order {od}, see {e}")
            traceback.print_exc()
        
        linelist1d['order']=od
        linelist1d['optord']=spec.optical_orders[od]
        if spec.instrument == 'ESPRESSO':
            if od%2 == 0:
                linelist1d['slice'] = 1
            else:
                linelist1d['slice'] = 2
        linelist2d.append(linelist1d)
        progress_bar.update(i/(len(orders)-1),f'Linelist, {wavereference}, {od}/{max(orders)}')
    linelist2d = np.hstack(linelist2d)
    
    
    cond1 = 'lsf' in fittype 
    cond2 = lsf_filepath is not None
    cond3 = spec.lsf_filepath is not None
    print('conditions 1,2,3 = ',cond1,cond2, cond3)
    if cond1 and (cond2 or cond3):
        if not cond2 and cond3:
            lsf_filepath = spec.lsf_filepath
        elif not cond1 and not cond3:
            raise ValueError
        import harps.linelist.fit as fit
        print('FITTING LSF', lsf_filepath)
        data = dict(
            flx=flx2d,
            wav=wav2d,
            err=err2d
            )
        fit.ip_bulk(linelist2d,data,lsf_filepath,version=1,
                    scale=['pixel','wave'],
                    interpolate=True,logger=None)
    
        print('FINISHED FITTING')
    else:
        print("NOT FITTING LSF")
    if return_thar_linelist:
        return linelist2d, linelist_thar
    return linelist2d

def detect(spec,*args,**kwargs):
    return detect2d_from_spec(spec,*args,**kwargs)


def detect1d_from_array(wav1d,flx1d,err1d,
                        line_positions,
                        # minima1d,
                        keys,fittype='gauss',
                        xscale=['pix','wav'],
                        logger=None,plot=False,ax=None,debug=False,
                        npars = None,
                        *args,**kwargs):
        
    """
    Returns a list of all LFC lines and fit parameters in the specified order.
    """
  
    log           = logger or logging.getLogger(__name__)
    reprate       = keys['comb_reprate']
    anchor        = keys['comb_anchor']
    offset        = kwargs.pop('anchor_offset',0)
    
    gauss_model   = kwargs.pop('gauss_model','SimpleGaussian')
    npix          = len(wav1d)
    
    # Make sure fittype and wavescale can be indexed
    fittype   = np.atleast_1d(fittype)
    xscale    = np.atleast_1d(xscale)
    
    # Data
    nlines             = len(line_positions)
    # nlines = len(maxima_x)
    
    if plot:
        if ax is not None:
            pass
        else:
            fig, axes = plt.subplots(len(xscale),sharey=True)
        # ax.plot(np.arange(npix),flx1d)
        if len(axes)>1:
            ax_pix, ax_wav = axes
        else:
            ax_pix = ax_wav = axes
        
        if 'wav' in xscale:
            ax_wav.plot(wav1d,flx1d)
            ax_wav.set_xlabel("Wavelength (A)")
        if 'pix' in xscale: 
            ax_pix.plot(np.arange(npix), flx1d)
            ax_pix.set_xlabel('Pixel')
        
        # ax.vlines(minima_x,0,np.max(flx1d),linestyles=':',
        #             linewidths=0.4,colors='C1')
        
    # New data container
    linelist          = container.linelist(nlines,npars=npars)
    for i, line_ in enumerate(line_positions):
        lpix = line_['pixl']
        rpix = line_['pixr']
        # lpix, rpix = (int(maxima_x[i]),int(maxima_x[i+1]))
        # lpix = int(maxima_x[i]-)
        # rpix = int(maxima_x[i+1])
        # barycenter
        pix1l  = np.arange(lpix,rpix,1)
        flx1l  = flx1d[lpix:rpix]
        if rpix-lpix<=4: # do not fit narrow lines
            continue
        if rpix-lpix>=50: # do not fit broad lines
            continue
        
        # bary = centroid, flux weighted mean position
        bary = np.average(pix1l,weights=flx1l)
        # bmean = flux weighted mean of two brightest pixels
        s = np.argsort(flx1l)[-2:]
        bmean = np.average(pix1l[s],weights=flx1l[s])
        # skewness
        skew = stats.skew(flx1l,bias=False)
        # CCD segment assignment (pixel space)
        local_seg = bary//keys['segsize']
        # photon noise
        # sumw = np.sum(pn_weights[lpix:rpix])
        # pn   = (c/np.sqrt(sumw))
        # signal to noise ratio
        err1l = err1d[lpix:rpix]
        snr = np.sum(flx1l)/np.sum(err1l)
        # background
        
        
        linelist[i]['pixl']   = lpix
        linelist[i]['pixr']   = rpix
        # linelist[i]['noise']  = pn
        linelist[i]['sumflx'] = np.sum(flx1l)
        linelist[i]['segm']   = local_seg
        linelist[i]['bary']   = bary
        linelist[i]['bmean']  = bmean
        linelist[i]['skew']   = skew
        linelist[i]['snr']    = snr
        linelist[i]['id']     = get_line_index(linelist[i])
    if debug:
        log.debug("Lines prepared for fitting using {}".format(fittype))
    # dictionary that contains functions for line profile fitting
    fitfunc = dict(gauss=fit_gauss1d)
    fitargs = dict(gauss=dict(line_model=gauss_model))
    # print('all fine to here')
    
    for i,ft in enumerate(['gauss']):
        for j,ws in enumerate(xscale):
            if 'pix' in ws:
                wave  = np.arange(npix)
            elif 'wav' in ws or 'vel' in ws:
                wave  = wav1d
            try:
                linepars = fitfunc[ft](linelist,wave,flx1d,err1d,
                                        xscale=ws,
                                        **fitargs[ft])
                linelist[f'{ft}_{ws}']          = linepars['pars']
                linelist[f'{ft}_{ws}_err']      = linepars['errs']
                linelist[f'{ft}_{ws}_chisq']    = linepars['chisq']
                linelist[f'{ft}_{ws}_chisqnu']  = linepars['chisqnu']
                linelist['success'][:,i*2+j*1]  = linepars['conv']
                linelist[f'{ft}_{ws}_integral'] = linepars['integral']
                if 'wav' in ws:
                    _, lbd, sigma_lbd = linepars['pars'].T
                    
                    sigma = sigma_lbd / lbd 
                    fwhm  = 2*np.sqrt(2*np.log(2)) * sigma
                    resolution = 1. / fwhm
                    linelist['resolution'] = resolution
            except:
                linelist['success'][:,i*2+j]    = False
            
    
    # print("Fitting of order {} completed ".format(order))
    # arange modes of lines in the order using ThAr coefficients in vacuum
    # wave1d = spec.wavereference[order]
    center1d = linelist['gauss_pix'][:,1]
    modes,refline = arange_modes_from_array(center1d,wav1d,
                                            reprate,anchor+offset)
    waves1d = linelist['gauss_wav'][:,1]
    for i in range(0,nlines,1):
          # mode and frequency of the line
        linelist[i]['mode'] = modes[i]
        linelist[i]['freq'] = anchor + modes[i]*reprate + offset
        if plot:
            if i==refline:
                lw = 1; ls = '-'
            else:
                lw = 0.5; ls = '--'
            if 'wav' in xscale: ax_wav.axvline(waves1d[i],c='r',ls=ls,lw=lw) 
            if 'pix' in xscale: ax_pix.axvline(center1d[i],c='r',ls=ls,lw=lw) 
    return linelist
    
def fit(spec,order=None):
    """
    Wrapper around 'detect'. Returns a dictionary.
    """
    return detect(spec,order)
def fit_gauss1d(linelist,wave,data,error,xscale='pixel',
                line_model='SimpleGaussian',
                npars=hs.npars,
                *args,**kwargs):

    nlines  = len(linelist)
    linepars = container.linepars(nlines,npars=npars)
    npix = len(data)
    for i,line in enumerate(linelist):
        # mode edges
        lpix, rpix = (line['pixl'],line['pixr'])
        # fit lines     
        # using 'SingleGaussian' class, extend by one pixel in each direction
        # make sure the do not go out of range
        if lpix==0:
            lpix = 1
        if rpix==npix-1:
            rpix = npix-2 
        if line_model=='SingleGaussian':
            pixx = wave[lpix-1:rpix+1]
            flxx = data[lpix-1:rpix+1]
            errx = error[lpix-1:rpix+1]
        else:
            pixx = wave[lpix:rpix]
            flxx = data[lpix:rpix]
            errx = error[lpix:rpix]
            
       
        # bkgx = background[lpix-1:rpix+1]
        # envx = envelope[lpix-1:rpix+1]
        
        fit_result = hfit.gauss(pixx,flxx,errx,line_model,xscale=xscale,
                                *args,**kwargs)
        success, pars,errs,chisq,chisqnu,integral = fit_result
        linepars[i]['pars'] = pars
        linepars[i]['errs'] = errs
        linepars[i]['chisq']= chisq
        linepars[i]['chisqnu']=chisqnu
        linepars[i]['conv'] = success
        linepars[i]['integral'] = integral
        # print(i,linepars[i])
    return linepars

def fit_gauss1d_minima(minima,data,wave,background,error,line_model='SingleGaussian',
                *args,**kwargs):

    nlines  = len(minima)-1
    linepars = container.linepars(nlines,3)
    npix = len(data)
    for i in range(nlines):
        # mode edges
        lpix, rpix = int(minima[i]), int(minima[i+1])
        print(lpix,rpix)
        # fit lines     
        # using 'SingleGaussian' class, extend by one pixel in each direction
        # make sure the do not go out of range
        if lpix==0:
            lpix = 1
        if rpix==npix-1:
            rpix = npix-2 
        
        pixx = np.arange(lpix-1,rpix+1,1)
        flxx = data[lpix-1:rpix+1]
        errx = error[lpix-1:rpix+1]
        bkgx = background[lpix-1:rpix+1]
        
        success, pars,errs,chisq, chisqnu = hfit.gauss(pixx,flxx,bkgx,errx,
                                     line_model,*args,**kwargs)
#        p0 = (np.max(flxx),np.mean(pixx),np.std(pixx),np.min(flxx),0e0)
#        pars, cov = curve_fit(curve.gauss5p,pixx,flxx,p0=p0,sigma=errx,
#                              absolute_sigma=True)
#        errs      = np.sqrt(np.diag(cov))
#        resid     = (flxx - curve.gauss5p(pixx,*pars))/errx
#        dof       = len(pixx) - len(pars)
#        chisq     = np.sum(resid**2) / dof
#        success   = 1
        linepars[i]['pars'] = pars
        linepars[i]['errs'] = errs
        linepars[i]['chisq']= chisq
        linepars[i]['chisqnu']= chisqnu
        linepars[i]['conv'] = success
    return linepars
def fit_lsf1d(linelist,wave,data,error,lsf,interpolation=False):
    """
    lsf must be an instance of LSF class with all orders and segments present
    (see harps.lsf)
    """
    nlines  = len(linelist)
    linepars = container.linepars(nlines)   
#    plt.figure()
    # print(lsf)
    # assert method in ['analytic','spline','gp']
    for i,line in enumerate(linelist):
        # mode edges
        lpix, rpix = (line['pixl'],line['pixr'])
        flx  = data[lpix:rpix]
        pix  = np.arange(lpix,rpix,1.) 
        err  = error[lpix:rpix]
        # line center
        cent = line['bary']#line[fittype][1]
        # segment
        order = line['order']
        if interpolation:
            lsf1s = lsf.interpolate(order,cent)
            
        else:
            segm  = line['segm']
            # lsf1s = lsf[order,segm]
            lsf1s = lsf.values[segm]
        # initial guess
        p0   = (np.max(flx),cent,1.)
        # success,pars,errs,chisq,chisqnu,integral,model = hfit.lsf(pix-cent,flx,bkg,err,
        #                                   lsf1s,output_model=True)
        try:
            fit_result = hfit.lsf(pix-cent,flx,err,lsf1s,output_model=True)
            success,pars,errs,chisq,chisqnu,integral,model = fit_result
        except:
            success = False
            pars    = np.full_like(p0,np.nan)
            errs    = np.full_like(p0,np.inf)
            chisq   = -1
            chisqnu = -1
            # print('exception',p0,method,lsf1s)
       
        flux, center, wid = pars
        #center = cent - shift
        linepars[i]['pars']   = pars
        linepars[i]['errs']   = errs
        linepars[i]['chisq']  = chisq
        linepars[i]['chisqnu']= chisqnu
        linepars[i]['conv']   = success
        linepars[i]['integral'] = integral
    return linepars
# def get_maxmin1d(yarray,xarray=None,background=None,use='minima',**kwargs):
#     """
#     Returns the positions of the minima between the LFC lines and the 
#     approximated positions of the maxima of the lines.
#     """
#     window = kwargs.pop('window',3)
    
#     assert use in ['minima','maxima']
     
#     if xarray is None:
#         xarray = np.arange(len(yarray))
#     assert np.shape(xarray)==np.shape(yarray)
    
#     # determine the positions of minima
#     yarray0 = yarray
#     if background is not None:
#         yarray0 = yarray - background
        
#     kwargs = dict(remove_false=kwargs.pop('remove_false',False),
#                   method='peakdetect_derivatives',
#                   window=window)
    # maxima, minima = hf.detect_maxmin(yarray,xarray,window=window)
#    if use=='minima':
#        extreme = 'min'
#    elif use=='maxima':
#        extreme = 'max'
    
#    priext_x,priext_y = hf.peakdet(yarray0,xarray,extreme=extreme,**kwargs)
#    priext = (priext_x).astype(np.int16)
#    secext = ((priext+np.roll(priext,1))/2).astype(np.int16)[1:]
#    if use == 'minima':
#        minima = priext
#        maxima = secext
#    elif use == 'maxima':
#        minima = secext
#        maxima = priext
    # return np.array(maxima[0],dtype=int),np.array(minima[0],dtype=int)
# def get_maxmin(spec,order,*args,**kwargs):
#     """
#     Returns the positions of the minima between the LFC lines and the 
#     approximated positions of the maxima of the lines.
#     """
#     # extract arrays
#     data = spec.data[order]
#     bkg  = spec.background[order]
#     pixels = np.arange(spec.npix)
    
#     # determine the positions of minima
#     yarray = data-bkg
#     # kwargs = dict(remove_false=remove_false,
#                   # method='peakdetect_derivatives',
#                   # window=11)#spec.lfckeys['window_size'])
    
#     maxima,minima = hf.detect_maxmin(yarray,pixels,plot=True,*args,**kwargs)
    
#     return minima,maxima

def model(spec,fittype,line_model=None,lsf=None,fibre=None,nobackground=False,
          interpolate_lsf=True):
    """
    Default behaviour is to use SingleGaussian class from EmissionLines.
    
    lsf must be an instance of LSF class (see harps.lsf)
    """
    line_model   = line_model if line_model is not None else hfit.default_line_class
    linelist     = spec['linelist']
    lineclass    = getattr(emline,line_model)
    numlines     = len(linelist)
    model2d      = np.zeros_like(spec.data)
    bkg2d        = spec['background']
    fibre        = fibre if fibre is not None else 'A'
    if fittype == 'lsf':
        if lsf is not None:
            if isinstance(lsf,str):
                lsf  = hlsf.from_file(lsf)
            elif isinstance(lsf,object):
                lsf  = lsf
        else:
            lsf      = hlsf.read_lsf(fibre,spec.datetime,'gp')
    ftype = f'{fittype}_pix'
    for i in range(numlines):
        order = linelist[i]['order']
        pixl  = linelist[i]['pixl']
        pixr  = linelist[i]['pixr']
        segm  = linelist[i]['segm']
        pars  = linelist[i][ftype]
        
        if fittype == 'gauss':
            if line_model == "SimpleGaussian":
                pix = np.arange(pixl, pixr)
            elif line_model == "SingleGaussian":
                pix   = np.arange(pixl-1,pixr+1)
            line  = lineclass()
            model2d[order,pixl:pixr] = line.evaluate(pars,pix)
        elif fittype == 'lsf':
            pix = np.arange(pixl,pixr)
            center = pars[1]
            if np.isfinite(center):
                if interpolate_lsf:
                    lsf1s = hlsf.interpolate_local_spline(lsf,order,center)
                else:
                    lsf1s = lsf[order,segm]
                model2d[order,pixl:pixr] = hfit.lsf_model_spline(lsf1s,pars,pix)
            else:
                continue
    if nobackground==False:
        model2d += bkg2d
    return model2d
def model_gauss(spec,*args,**kwargs):
    return model(spec,'gauss',*args,**kwargs)
def model_lsf(spec,*args,**kwargs):
    return model(spec,'lsf',*args,**kwargs)

def select_order(linelist,order):
    if isinstance(order,slice):
        orders = np.arange(order.start,order.stop+1,order.step)
    orders = np.atleast_1d(orders)
    cut = np.isin(linelist['order'],orders)
    return linelist[cut]
def remove_order(linelist,order):
    if isinstance(order,slice):
        orders = np.arange(order.start,order.stop+1,order.step)
    orders = np.atleast_1d(orders)
    cut = np.isin(linelist['order'],orders)
    return linelist[~cut]
def select(linelist,condict):
    condtup = tuple(linelist[key]==val for key,val in condict.items())
    condition = np.logical_and.reduce(condtup) 
    cut = np.where(condition==True)
    return linelist[cut]

def get_line_index(linelist_like):
    fac = 10000
    MOD = 1
    centers = linelist_like['bary']
    orders  = linelist_like['order']*fac
    cround  = np.round(centers/MOD)*MOD
    cint    = np.asarray(cround,dtype=int)
    index0  = orders+cint
    return index0

class Linelist(container.Generic):
    def __init__(self,narray):
        self._values = narray
    def __getitem__(self,item):
        condict, segm_sent = self._extract_item(item)
        return self.select(condict)
    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning order plus segment.
        
        To be used with partial decorator
        """
        condict = {}
        segm_sent = False
        if isinstance(item,dict):
            if len(item)==2: segm_sent=True
            
            condict.update(item)
        else:
            dict_sent=False
            if isinstance(item,tuple):
                
                nitem = len(item) 
                if nitem==2:
                    segm_sent=True
                    order,segm = item
                    
                elif nitem==1:
                    segm_sent=False
                    order = item[0]
            else:
                segm_sent=False
                order=item
            condict['order']=order
            if segm_sent:
                condict['segm']=segm
        return condict, segm_sent
    def __len__(self):
        return len(self.values)
    @property
    def values(self):
        return self._values
    def select(self,condict):
        values  = self.values 
        condtup = tuple(values[key]==val for key,val in condict.items())
        
        condition = np.logical_and.reduce(condtup)
        
        cut = np.where(condition==True)
        return Linelist(values[cut])
    
    