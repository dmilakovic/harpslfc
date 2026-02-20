#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:19:15 2023

@author: dmilakov
"""
import numpy as np
import spectres
import harps.settings as hs
import harps.functions.math as mathfunc
import harps.functions.aux as auxfunc
import harps.plotter as hplt
import harps.progress_bar as progress_bar

from scipy.signal import welch
import matplotlib.pyplot as plt
import harps.peakdetect as pkd
import spectres



c = 299792458e0
def wave_(WL0,CRPIX1,CD1,NPIX):
    wa =np.array([np.power(10,WL0+((i+1)-CRPIX1)*CD1) for i in range(NPIX)])
    return wa
def wave_from_header(header):
    try:
        wl0 = header['crval1']
    except:
        wl0 = np.log10(header['up_wlsrt'])
    crpix1 = header['crpix1']
    cd1    = header['cd1_1']
    npix   = header['naxis1']
    print(wl0,crpix1,cd1,npix)
    return wave_(wl0,crpix1,cd1,npix,)



def prepare_orders(order=None):
    '''
    Returns an array or a list containing the input orders.
    '''
    if order is None:
        orders = np.arange(hs.sOrder,hs.eOrder,1)
    else:
        orders = auxfunc.to_list(order)
    return orders


def select_orders(orders):
    use = np.zeros((hs.nOrder,),dtype=bool); use.fill(False)
    for order in range(hs.sOrder,hs.eOrder,1):
        if order in orders:
            o = order - hs.sOrder
            use[o]=True
    col = np.where(use==True)[0]
    return col

def redisperse1d(old_wav1d,flx1d,err1d,velocity_step):
    '''
    

    Parameters
    ----------
    old_wavs : 2d array of floats
        Old wavelength array.
    velocity_step : float
        velocity step in km/s.

    Returns
    -------
    new_wavs : 2d array of floats
        New wavelength array.
    new_fluxes : 2d array of floats
        New spectral flux array.
    new_errs : 2d array of floats
        New spectral error array.

    '''
    
    
    assert np.shape(old_wav1d)==np.shape(flx1d)
    
    
    step = velocity_step/299792.458
    
    w0 = old_wav1d[0]
    new_wav  = w0*np.exp(step*np.arange(len(flx1d)))
    new_flux, new_errs = spectres.spectres(new_wav, old_wav1d, 
                                             spec_fluxes=flx1d,
                                             spec_errs = err1d,
                                             fill = 0,)
    
    return new_wav,new_flux,new_errs


def redisperse2d(old_wav2d,flx2d,err2d,velocity_step):
    assert np.shape(old_wav2d)==np.shape(flx2d)
    nbo, npix = np.shape(old_wav2d)
    
    new_wavs = np.zeros_like(old_wav2d)
    new_flux = np.zeros_like(flx2d)
    new_errs = np.zeros_like(flx2d)
    for od in range(nbo):
        if np.sum(old_wav2d[od])==0:
            continue
        else:
            pass
        result = redisperse1d(old_wav2d[od],flx2d[od],err2d[od],velocity_step) 
                               
        new_wavs[od] = result[0]
        new_flux[od] = result[1]
        new_errs[od] = result[2]
    return new_wavs,new_flux,new_errs

def plot_power_spectrum(y_axis,nperseg=300):
    freq0, P0    = welch(y_axis,nperseg=nperseg)
    # cut = np.where(freq0>0.002)[0]
    freq, P = freq0, P0
    
    plt.figure()
    plt.plot(1./freq,P)
    plt.xlabel('1/freq')
    plt.ylabel('Power')

def peakdet_window(y_axis,plot=False):
    freq0, P0    = welch(y_axis,nperseg=512)
    cut = np.where(freq0>0.02)[0]
    freq, P = freq0[cut], P0[cut]
    maxind     = np.argmax(P)
    maxfreq    = freq[maxind]
    if plot:
        plt.figure()
        plt.semilogy(freq,P)
        plt.semilogy(maxfreq,P[maxind],marker='x',c='C1')
    return mathfunc.round_down_to_odd(1./maxfreq)

def peakdet_limits(y_axis,plot=False):
    freq0, P0    = welch(y_axis,nperseg=512)
    cut = np.where(freq0>0.02)[0]
    freq, P = freq0[cut], P0[cut]
    maxind     = np.argmax(P)
    maxfreq    = freq[maxind]
    
    
        
    
    # maxima and minima in the power spectrum
    maxima, minima = (np.transpose(x) for x in pkd.peakdetect(P,freq,delta=20,
                                                              lookahead=20))
    if plot:
        plt.figure()
        plt.plot(freq,P)
        plt.scatter(minima[0],minima[1],marker='v')
        plt.scatter(maxima[0],maxima[1],marker='^')
    minsorter  = np.argsort(minima[0])
    # find the most powerful peak in the power spectrum
    index      = np.searchsorted(minima[0],maxfreq,sorter=minsorter)
    # find minima surrounding the most powerful peak
    minfreq = (minima[0][index-1:index+1])
    print(maxfreq)
    try:
        maxdist, mindist = tuple(1./minfreq)
    except:
        maxdist = -1
        mindist = -1
    if plot:
        [plt.axvline(pos,c='C1',ls='--') for pos in tuple(1./minfreq)]
        
    mindist = max(mindist,7)
    maxdist = max(maxdist,20)
    return mindist,maxdist

def remove_false_maxima(x_axis,y_axis,extreme,x_xtrm,y_xtrm,limit,
                        mindist,maxdist,polyord=1,N=None,plot=True):
    '''
    Removes false minima (or maxima) by considering the distances between
    sequential extrema and the characteristics of the data.

    Parameters
    ----------
    x_axis : array-like
        Pixel number or wavelength.
    y_axis : array-like
        Flux or S/N.
    input_xmin : list
        x-coordinates of the extremes in y_axis.
    input_ymin : list
        y-coordinates of the extremes in y_axis.
    limit : float
        largest allowed distance between sequential extremes.
    mindist : float
        smallest allowed distance between sequential extremes.
    maxdist : float
        largest allowed distance between sequential extremes.
    polyord : int, optional
        the order of the polynomial fitted through the distances between
        extremes. The default is 1.
    N : int, optional
        the window (in pixels) over which RMS and mean are calculated and
        compared. The default is approximately len(yaxis)/10, rounded to 
        the closest 100.
    plot : bool, optional
        Plots the results. The default is True.

    Returns
    -------
    xmin : list
        Cleaned x-coordinates of extremes in y_axis.
    ymin : list
        Cleaned y-coordinates of extremes in y_axis.

    '''
    
    outliers   = np.full_like(x_xtrm,True)
    N = int((maxdist+mindist)/2.)
    mean_y = mathfunc.running_mean(y_axis, N=N)
    new_xmin = x_xtrm
    new_ymin = y_xtrm
    j = 0
    if plot:
        fig=hplt.Figure(4,1,height_ratios=[3,1,1,1])
        ax = [fig.ax() for i in range(4)]
        ax[0].plot(x_axis,y_axis,drawstyle='steps-mid')
        ax[0].plot(x_axis,mean_y,drawstyle='steps-mid',
                    label='Mean over {} pixels'.format(N))
        ax[0].scatter(x_xtrm,y_xtrm,marker='^',c='red',s=30)
        ax[0].set_ylabel('Data')
        ax[1].set_ylabel('Distances between\nextremes')
        ax[2].set_ylabel("Residuals")
        ax[3].set_ylabel("Flux < mean\nover {} pixels".format(N))
    while sum(outliers)>0 and j<10:
        # print('BEGIN iteration {}, outliers={}/{}'.format(j, sum(outliers), len(outliers)))
        old_xmin = new_xmin
        old_ymin = new_ymin
        # Distances to the left and to the right neighbour
        dist_r = np.diff(old_xmin, append=old_xmin[-1]+mindist)
        dist_l = np.diff(np.roll(old_xmin,-1),prepend=old_xmin[0]-mindist) 
        
        # Due to rolling, last element of dist_l is wrong, set to 0
        dist_l[-1]=0
        
        # Difference in the distances between the left and right neighbour
        dist   = dist_r - dist_l; dist[-1] = 0
        # Fit a polynomial of order polyord to the left and right distances
        # Save residuals to the best fit model into arrays
        arrays = []
        for i,values in enumerate([dist_l,dist_r]):
            keep = (values>mindist) & (values<maxdist)
            pars, cov = np.polyfit(old_xmin[keep],
                                    values[keep],polyord,
                                      cov=True)
            model = np.polyval(pars,old_xmin)
            resid = values-model
            cond_ = np.abs(resid)>limit
            arrays.append(cond_)
            if plot:
                if i == 0:
                    c='b'; marker='<'  
                else:
                    c='r'; marker='>'
                ax[2].scatter(old_xmin,resid,marker=marker,c=c,s=15)
                [ax[2].axhline(l,c='r',lw=2) for l in [-limit,limit]]
        
        # Maxima for which residuals from the left AND from the right are
        # larger than some limit  
        cond0     = np.bitwise_and(*arrays)
        # Maxima for which distances from the immediate left and from the 
        # immediate right neighbour disagree by more than some limit
        cond1     = np.abs(dist)>limit
        # Maxima for which the distance to the left OR the distance to the 
        # right neighbour is smaller than the minimum allowed distance between
        # neighbouring maxima
        cond2     = np.bitwise_or(np.abs(dist_l)<mindist,
                                  np.abs(dist_r)<mindist)
        # Maxima for y_values which are below the running mean y_axis across 
        # N pixels 
        indices   = np.asarray(old_xmin,dtype=np.int16)
        if extreme == 'max':
            cond3     = old_ymin<1.1*mean_y[indices]
        elif extreme == 'min':
            cond3     = old_ymin>1.1*mean_y[indices]
        
        outliers_ = np.bitwise_and(cond1,cond0)
        outliers_ = np.bitwise_and(cond2,outliers_)
        outliers_ = np.bitwise_or(cond3,outliers_)
        cut       = np.where(outliers_==True)[0]
        if len(cut)>0:
            for i in cut:
                if i+2 in cut and outliers_[i]==True and i+2<len(outliers_):
                    # print("i={}, i+2={}".format(dist[i],dist[i+2]))
                    if np.sign(dist[i])!=np.sign(dist[i+2]):
                        # print("Changing values in outliers")
                        outliers_[i] = False
                        outliers_[i+1]=True
                        outliers_[i+2]=False
                else:
                    pass
        outliers = outliers_
        
        
        if plot:
            ax[0].scatter(old_xmin[outliers],old_ymin[outliers],marker='x',
                          c="C{}".format(j))
            ax[1].scatter(old_xmin,dist_l,marker='<',s=15,c='b')
                          # c="C{}".format(j))
            ax[1].scatter(old_xmin,dist_r,marker='>',s=15,c='r')
                            # c="C{}".format(j))
            [ax[1].axhline(l,c='r',lw=2) for l in [mindist,maxdist]]
            ax[2].scatter(old_xmin[cond0],arrays[0][cond0],marker='x',s=15,c='b')
            ax[2].scatter(old_xmin[cond0],arrays[1][cond0],marker='x',s=15,c='r')
            # ax[3].axhline(-limit,c='r',lw=2)
            # ax[3].axhline(limit,c='r',lw=2)
            ax[3].scatter(old_xmin,cond3,marker='x')
        new_xmin = (old_xmin[~outliers])
        new_ymin = (old_ymin[~outliers])
        # print('END iteration {}, outliers={}/{}'.format(j, sum(outliers), len(input_xmin)))
        j+=1
        
    xmin, ymin = new_xmin, new_ymin
    if plot:
        maxima0 = (np.roll(xmin,1)+xmin)/2
        maxima = np.array(maxima0[1:],dtype=np.int)
        [ax[0].axvline(x,ls=':',lw=0.5,c='r') for x in maxima]
        # ax[0].legend()
    
    return xmin,ymin

def remove_false_minima(x_axis,y_axis,input_xmin,input_ymin,rsd_limit,
                        mindist,maxdist,polyord=1,N=None,plot=False):
    '''
    DO NOT USE
    
    
    Removes false minima (or maxima) by considering the distances between
    sequential extrema and the characteristics of the data.

    Parameters
    ----------
    x_axis : array-like
        Pixel number or wavelength.
    y_axis : array-like
        Flux or S/N.
    input_xmin : list
        x-coordinates of the extremes in y_axis.
    input_ymin : list
        y-coordinates of the extremes in y_axis.
    rsd_limit : float
        largest allowed distance between sequential extremes.
    mindist : float
        smallest allowed distance between sequential extremes.
    maxdist : float
        largest allowed distance between sequential extremes.
    polyord : int, optional
        the order of the polynomial fitted through the distances between
        extremes. The default is 1.
    N : int, optional
        the window (in pixels) over which RMS and mean are calculated and
        compared. The default is approximately len(yaxis)/10, rounded to 
        the closest 100.
    plot : bool, optional
        Plots the results. The default is True.

    Returns
    -------
    xmin : list
        Cleaned x-coordinates of extremes in y_axis.
    ymin : list
        Cleaned y-coordinates of extremes in y_axis.

    '''
    
    new_xmin = input_xmin
    new_ymin = input_ymin
    new_outliers   = np.full_like(input_xmin,True)
    outliers      =  np.full_like(input_xmin,False)
    N = N if N is not None else int(mathfunc.round_to_closest(len(y_axis),1000)/10)
    M = int(N/10)
    # mean_arrayN = mathfunc.running_mean(y_axis, N=N)
    # mean_arrayM = mathfunc.running_mean(y_axis, N=M)
    # rms_arrayM = mathfunc.running_rms(y_axis, N=M)
    # y_lt_rms    = y_axis < rms_arrayM
    # rms_lt_mean = rms_arrayM < mean_arrayN
    # remove 
    # cond0 = y_lt_rms[np.asarray(old_xmin[1:],dtype=np.int32)]==False
    
    
    if plot:
        fig=hplt.Figure(3,1,height_ratios=[3,1,1])
        ax = [fig.ax() for i in range(3)]
        ax[0].plot(x_axis,y_axis,drawstyle='steps-mid')
        # ax[0].plot(x_axis,rms_arrayM,drawstyle='steps-mid',
        #            label='RMS over {} pixels'.format(M))
        # ax[0].plot(x_axis,mean_arrayN,drawstyle='steps-mid',
        #            label='Mean over {} pixels'.format(N))
        ax[0].scatter(input_xmin,input_ymin,marker='^',c='red',s=8)
        # ax[3].plot(x_axis[np.asarray(input_xmin,np.int32)],
        #            rms_lt_mean[np.asarray(input_xmin,np.int32)],
        #            drawstyle='steps-mid')
        ax[0].set_ylabel('Data')
        ax[1].set_ylabel('Residuals')
        ax[2].set_ylabel("Distances between\nextremes")
        # ax[3].set_ylabel("RMS < Mean\nover {} pixels".format(N))
    j = 0
    while sum(new_outliers)>0 and j<50:
        # print('iteration ',j, sum(outliers), len(outliers))
        old_xmin = new_xmin
        old_ymin = new_ymin
        # xpos = old_xmin
        # ypos = old_ymin
        dist = np.diff(old_xmin) # returns array length len(old_xmin)-1
        
        cond1 = dist>mindist
        cond2 = dist<maxdist
        
        keep  = cond1 & cond2
        pars  = np.polyfit(old_xmin[1:][keep],dist[keep],polyord)
        model = np.polyval(pars,old_xmin[1:])
        
        resids = dist-model        
        cond3 = np.abs(resids)<rsd_limit
        # cond4 = y_lt_rms[np.asarray(old_xmin[1:],dtype=np.int32)]==False
        keep1 = np.logical_and.reduce([cond1,cond2,cond3])
        outliers1 = ~keep1
        cut = np.where(outliers1)[0]
        # print(cut)
        outliers[cut] = True
        # print('iteration ',j, 'outliers1', sum(outliers1), len(outliers1))
        # outliers1 = np.logical_or(outliers1,cond4)
        # make outliers a len(xpos) array, taking care to remove/keep the first
        # point
        insert_value = False if outliers1[0]==False else True
        # print(*[len(array) for array in [cond1,cond2,cond3,outliers1]])
        outliers2 = np.insert(outliers1,0,insert_value) # returns array of length len(diff)+1
        # print('iteration ',j, 'outliers2', sum(outliers2), len(outliers2))
        new_outliers = outliers2
        new_xmin = (old_xmin[~new_outliers])
        new_ymin = (old_ymin[~new_outliers])
        if plot:
            ax[0].scatter(old_xmin[outliers2],old_ymin[outliers2],marker='x',s=15,
                          c="C{}".format(j))
            ax[1].scatter(old_xmin[1:],resids,marker='o',s=3,c="C{}".format(j))
            ax[1].scatter(old_xmin[1:][outliers1],resids[outliers1],marker='x',s=15,
                          c="C{}".format(j))
            ax[1].axhline(rsd_limit,c='r',lw=2)
            ax[1].axhline(-rsd_limit,c='r',lw=2)
            ax[2].scatter(old_xmin[1:],dist,marker='o',s=3,c="C{}".format(j))
            ax[2].axhline(0.9*mindist,c='r',lw=2)
        # print('END iteration ',j, sum(outliers), len(outliers))
        j+=1
        
    # good_range = y_axis > mean_arrayM
    # cond4 = good_range[np.asarray(new_xmin,dtype=np.int32)]==True
    # outliers = np.logical_and(~outliers_,cond4)
    # outliers = outliers_
    # print(outliers)
    xmin, ymin = new_xmin[~outliers], new_ymin[~outliers]
    if plot:
        maxima0 = (np.roll(xmin,1)+xmin)/2
        maxima = np.array(maxima0[1:],dtype=np.int)
        [ax[0].axvline(x,ls=':',lw=0.5,c='r') for x in maxima]
        ax[0].legend()
    
    keep = ~outliers.astype(bool)
    # print(len(input_xmin),sum(outliers),len(outliers),keep,)
    return keep


        
        
def detect_maxmin(y_axis,x_axis=None,plot=False,*args,**kwargs):
    
    return get_extrema1d(y_axis,x_axis,plot=plot,*args,**kwargs)

def detect_minima(yarray,xarray=None,*args,**kwargs):
    return get_extrema1d(yarray,xarray,*args,**kwargs)[1]

def detect_maxima(yarray,xarray=None,*args,**kwargs):
    return get_extrema1d(yarray,xarray,*args,**kwargs)[0]

def peakdet(yarray,xarray=None,*args,**kwargss):
    return get_extrema1d(yarray,xarray=None,*args,**kwargs)

def get_extrema1d(y_axis, x_axis = None, y_error = None, remove_false=False,
            method='peakdetect_derivatives', plot=False, window_len=None,
            logger=None):
    '''
    A more general function to detect minima and maxima in the data
    
    Returns a list of minima or maxima 
    
    '''
    
    
    if y_error is not None:
        assert len(y_error)==len(y_axis), "y_error not same length as y_axis"
        y_axis = y_axis / y_error
    x_axis = x_axis if x_axis is not None else np.arange(len(y_axis),dtype=int)
        
    maxima, minima = [np.array(a) for a 
                     in pkd.peakdetect_derivatives(y_axis, 
                                                   x_axis, 
                                                   window_len=window_len,
                                                   plot=plot)]
    maxima = np.transpose(maxima)
    minima = np.transpose(minima)
    
    
    
    # if remove_false:
    #     cut = remove_false_minima(x_axis, y_axis, minima[0], minima[1],
    #                               rsd_limit=3, mindist=8, maxdist=20, 
    #                               polyord=1)
    #     minima = np.array([_[cut] for _ in minima])
        # maxima = np.array([_[cut] for _ in maxima])

    # if plot:
    #     plt.figure()
    #     if x_axis is not None:
    #         x_axis = x_axis
    #     else:
    #         x_axis = np.arange(len(y_axis))
    #     plt.plot(x_axis,y_axis,drawstyle='steps-mid')
    #     plt.scatter(maxima[0],maxima[1],c='g',marker='^',label='Maxima')
    #     plt.scatter(minima[0],minima[1],c='r',marker='o',label='Minima')
    #     # plt.scatter(minima[0],minima[1],c='k',marker='s',label='Not used minima')
    #     plt.legend()
    # if remove_false:
    #     limit = limit if limit is not None else 7
    #     x_axis = x_axis if x_axis is not None else np.arange(len(y_axis))
    #     mindist, maxdist = peakdet_limits(y_axis,plot=plot)
    #     # print(mindist,maxdist)
    #         # return x_axis,y_axis,data[0],data[1],limit,mindist,maxdist
        
    #     return_array = remove_false_maxima(x_axis,y_axis,extreme,
    #             return_array[0],return_array[1],limit,mindist,maxdist,plot=plot)
        # except:
            # logger = logger or logging.getLogger(__name__)
            # logger.warning("Could not remove false minima")
        
    return maxima,minima

# def get_extrema2d(y_axis, x_axis=None, y_error = None,
#                   sOrder = None, eOrder = None,
#                   method='peakdetect_derivatives', plot=False,
#                   logger=None):
    
#         pkd.process_spectrum2d(y_axis,
#                            sOrder=None,
#                            x_axis_orig_input=None,
#                            super_sample_factor=3,
#                            snr_threshold = 3, 
#                            window_len_method_options: dict = None,
#                            deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
#                            # Parameters for Sav-Gol filter
#                            savgol_params: dict = None, 
#                            # Consolidated parameters for the hybrid filter (rules + separate DBSCAN)
#                            hybrid_filter_params: dict = None,
#                            triplet_removal_params: dict = None,
#                            y_refinement_on_candidates_params: dict = None,
#                            plot_main_details=False,
#                            plot_deriv_stage_details=True, # Plot for initial smoothing/derivatives
#                            plot_filter_stage_details=True,  # Plot for rule-based filter diagnostics
#                            verbose=False
#                        ):
# def get_extrema2d(y_axis, x_axis = None, y_error = None, #remove_false=False,
#               sOrder=None, eOrder = None,
#             method='peakdetect_derivatives', plot=False,
#             logger=None):
#     '''
#     A more general function to detect minima and maxima in the data
    
#     Returns a list of minima or maxima 
    
#     '''
#     nbo,npix = np.shape(y_axis)
#     sOrder   = sOrder if sOrder is not None else 39
#     eOrder   = eOrder if eOrder is not None else nbo
#     orders  = np.arange(sOrder,eOrder)
#     # nbo, npix = np.shape(y_axis)
#     if y_error is not None:
#         assert np.shape(y_error)==np.shape(y_axis), "y_error not same length as y_axis"
#         y_axis = y_axis / y_error
#     x_axis = x_axis if x_axis is not None else np.tile(np.arange(npix),(nbo,1))

#     assert np.shape(y_axis) == np.shape(x_axis)
    
#     maxima2d = {}
#     minima2d = {}
    
#     for i,od in enumerate(orders):
#         if od<sOrder: continue
#         y_axis1d = y_axis[od]
#         x_axis1d = x_axis[od]
        
#         # plot = True if od==40 else False
#         maxima1d, minima1d = [np.array(a) for a 
#                          in pkd.process_spectrum1d(
#                                                        y_axis1d, 
#                                                        x_axis1d, 
#                                                        super_sample_factor=5,
#                                                        window_len_method_options='auto_robust',
#                                                        plot_main_details=plot)]
        
#         maxima1d = np.transpose(maxima1d)
#         minima1d = np.transpose(minima1d)
        
        
        
#         # if remove_false:
#         #     cut = remove_false_minima(x_axis1d, y_axis1d,
#         #                               minima1d[0], minima1d[1],
#         #                               rsd_limit=3, mindist=8, maxdist=20, 
#         #                               polyord=1)
#         #     minima1d = np.array([_[cut] for _ in minima1d])
#         #     # maxima = np.array([_[cut] for _ in maxima])

#         maxima2d[od] = maxima1d
#         minima2d[od] = minima1d
#         progress_bar.update(i/(len(orders)-1),f'Extrema {od}/{eOrder-1}')
    
        
#     return maxima2d,minima2d

# =============================================================================
    
#                    S P E C T R U M     H E L P E R
#                          F U N C T I O N S
    
# =============================================================================
optordsA   = [161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149,
   148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136,
   135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123,
   122, 121, 120, 119, 118, 117, 116, 114, 113, 112, 111, 110, 109,
   108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98,  97,  96,
    95,  94,  93,  92,  91,  90,  89]
optordsB   = [161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149,
   148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136,
   135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123,
   122, 121, 120, 119, 118, 117,      114, 113, 112, 111, 110, 109,
   108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98,  97,  96,
    95,  94,  93,  92,  91,  90,  89]

def prepare_orders(order,nbo,sOrder,eOrder):
    '''
    Returns an array or a list containing the input orders.
    '''
    orders = np.arange(nbo)
    select = slice(sOrder,eOrder,1)
    
    if isinstance(order,list):
        return orders[order]
    elif order is not None:
        select = prepare_slice(order,nbo,sOrder)
    return orders[select]
def prepare_slice(order,nbo,sOrder):
    import numbers
    if isinstance(order,numbers.Integral):
        start = order
        stop = order+1
        step = 1
    elif isinstance(order,tuple):
        range_sent = True
        numitems = np.shape(order)[0]
        if numitems==3:
            start, stop, step = order
        elif numitems==2:
            start, stop = order
            step = 1
        elif numitems==1:
            start = order
            stop  = order+1
            step  = 1
    else:
        start = sOrder
        stop  = nbo
        step  = 1
    return slice(start,stop,step)

def wrap_order(order,sOrder,eOrder):
    '''
    Returns an array or a list containing the input orders.
    '''
    orders = np.arange(eOrder)
    select = slice(sOrder,eOrder,1)
    if order is not None:
        select = slice_order(order)
    return orders[select]
def slice_order(order):
    #nbo = self.meta['nbo']
    start = None
    stop  = None
    step  = None
    if isinstance(order,int):
        start = order
        stop = order+1
        step = 1
    elif isinstance(order,tuple):
        numitems = np.shape(order)[0]
        if numitems==3:
            start, stop, step = order
        elif numitems==2:
            start, stop = order
            step = 1
        elif numitems==1:
            start = order
            stop  = order+1
            step  = 1
    return slice(start,stop,step)

def wave_to_velocity(wave,wave0):
    return (wave-wave0)/wave0*299792458
def velocity_to_wave(vel,wave0):
    return wave0*(1+vel/299792458)
    