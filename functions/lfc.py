#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:25:37 2023

@author: dmilakov
"""
import numpy as np
import logging
import harps.containers as container
import harps.plotter as hplt
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

c = 299792458e0
#------------------------------------------------------------------------------
# 
#                           C O M B     S P E C I F I C
#
#------------------------------------------------------------------------------  

def freq_to_lambda(freq):
    return 1e10*c/(freq) #/1e9
    
def noise_from_linelist(linelist):
    x = (np.sqrt(np.sum(np.power(linelist['noise']/c,-2))))
    return c/x
def remove_bad_fits(linelist,fittype,limit=None,q=None):
    """ 
    Removes lines which have uncertainties in position larger than a given 
    limit.
    """
    limit  = limit if limit is not None else 0.05
    q      = q     if q     is not None else 0.95
    
    field  = '{}_err'.format(fittype)
    values = linelist[field][:,1]

    n_finite = np.sum(np.isfinite(values))
    if n_finite == 0:
        raise ValueError(
            f"remove_bad_fits: every value in '{field}' is non-finite (NaN/inf). "
            f"'{field}' <= limit is False for ANY limit when "
            f"the values are NaN, so the retry loop below would never terminate "
            f"— this used to hang forever instead of raising. If '{fittype}' "
            f"centres come from a fit that doesn't produce a per-line "
            f"uncertainty (e.g. the JAX/LBFGSB line fit, which currently "
            f"leaves *_err as NaN), fix it at the source rather than here: "
            f"either supply a real uncertainty estimate, or call this with "
            f"a fittype whose _err field is actually populated."
        )

    keep   = np.where(values<=limit)[0]
    frac   = len(keep)/len(values)
    # if fraction of kept lines is smaller than some limit q (e.g. 90%)
    # increase the limit such to remove the worst (1-q) percent of the lines
    max_iter = 100_000  # second line of defense: fail loudly, don't hang,
                        # even if some other future case again makes frac
                        # unable to reach q.
    n_iter = 0
    while frac<q:
        limit  += 0.001
        keep   = np.where(values<=limit)[0]
        frac   = len(keep)/len(values)
        n_iter += 1
        if n_iter > max_iter:
            raise RuntimeError(
                f"remove_bad_fits: failed to reach the target kept-fraction "
                f"q={q} after {max_iter} iterations (limit grew to {limit:.3f}, "
                f"still only keeping {frac:.1%}). {n_finite}/{len(values)} "
                f"values in '{field}' are finite — this looks like a real "
                f"data problem, not just a slow convergence."
            )
    logger=logging.getLogger(__name__)
    N      = len(values)
    K      = len(keep)
    msg = "{0:5d}/{1:5d} ({2:5.2%}) kept ; ".format(K,N,frac) +\
          "{0:5d}/{1:5d} ({2:5.2%}) discarded".format(N-K,N,1-frac)
    # print(msg)
    logger.debug(msg)
    return linelist[keep]
def _get_index(centers):
    ''' Input: dataarray with fitted positions of the lines
        Output: 1d array with indices that uniquely identify every line'''
    fac = 10000
    MOD = 2.
    od = centers.od.values[:,np.newaxis]*fac
    centers_round = np.rint(centers.values/MOD)*MOD
    centers_nonan = np.nan_to_num(centers_round)
    ce = np.asarray(centers_nonan,dtype=np.int)
    index0=np.ravel(od+ce)
    mask = np.where(index0%fac==0)
    index0[mask]=999999999
    return index0
def _get_sorted(index1,index2):
    print('len indexes',len(index1),len(index2))
    # lines that are common for both spectra
    intersect=np.intersect1d(index1,index2)
    intersect=intersect[intersect>0]

    indsort=np.argsort(intersect)
    
    argsort1=np.argsort(index1)
    argsort2=np.argsort(index2)
    
    sort1 =np.searchsorted(index1[argsort1],intersect)
    sort2 =np.searchsorted(index2[argsort2],intersect)
    
    return argsort1[sort1],argsort2[sort2]
def average_line_flux(linelist,flux2d,bkg2d=None,orders=None):
    ''' 
    Returns the average line flux per line of an exposure.
    '''
    orders = orders if orders is not None else np.unique(linelist['order'])
    if bkg2d is not None:
        totflux = np.sum(flux2d[orders]-bkg2d[orders])
    else:
        totflux = np.sum(flux2d[orders])
    ll     = container.Generic(linelist)
    nlines = len(ll[orders])
    return totflux/nlines
def make_comb_interpolation(lines_LFC1, lines_LFC2,ftype='gauss'):
    ''' Routine to use the known frequencies and positions of a comb, 
        and its repetition and offset frequencies to build a frequency
        solution by linearly interpolating between individual lines
        
        Arguments must be for the same fibre!
        
        LFC1: known
        LFC2: to be interpolated
        
        Args:
        -----
            lines_LFC1 : lines xarray Dataset for a single exposure
            lines_LFC2 : lines xarray Dataset for a single exposure
    ''' 
    
    freq_LFC1 = lines_LFC1['freq']
    freq_LFC2 = lines_LFC2['freq']

    pos_LFC1  = lines_LFC1[ftype][:,1]
    pos_LFC2  = lines_LFC2['bary']
    #plt.figure(figsize=(12,6))
    minord  = np.max(tuple(np.min(f['order']) for f in [lines_LFC1,lines_LFC2]))
    maxord  = np.min(tuple(np.max(f['order']) for f in [lines_LFC1,lines_LFC2]))
    interpolated = {}
    for od in np.arange(minord,maxord):
        print("Order {0:=>30d}".format(od))
        # get fitted positions of LFC1 and LFC2 lines
        inord1 = np.where(lines_LFC1['order']==od)
        inord2 = np.where(lines_LFC2['order']==od)
        cen1   = pos_LFC1[inord1]
        cen2   = pos_LFC2[inord2]
        x1, x2 = (np.sort(x) for x in [cen1,cen2])
        # find the closest LFC1 line to each LFC2 line in this order 
        freq1  = freq_LFC1[inord1]
        freq2  = freq_LFC2[inord2]
        f1, f2 = (np.sort(f)[::-1] for f in [freq1,freq2])
        vals, bins = f2, f1
        right = np.digitize(vals,bins,right=False)
        print(right)
        fig = hplt.Figure(2,1,height_ratios=[3,1])
        ax0 = fig.ax()
        ax1 = fig.ax(sharex=ax0)
        ax = [ax0,ax1]
        ax[0].set_title("Order = {0:2d}".format(od))
        ax[0].scatter(f1,x1,c="C0",label='LFC1')
        ax[0].scatter(f2,x2,c="C1",label='LFC2')
        interpolated_LFC2 = []
        for x_LFC2,f_LFC2,index_LFC1 in zip(x2,f2,right):
            if index_LFC1 == 0 or index_LFC1>len(bins)-1:
                interpolated_LFC2.append(np.nan)
                continue
            else:
                pass
            
            f_left  = f1[index_LFC1-1]
            x_left  = x1[index_LFC1-1]
            f_right = f1[index_LFC1]
            x_right = x1[index_LFC1]
#            if x_LFC2 > x_right:
#                interpolated_LFC2.append(np.nan)
#                continue
#            else:
#                pass
            
            # fit linear function 
            fitpars = np.polyfit(x=[f_left,f_right],
                                 y=[x_left,x_right],deg=1)
            ax[0].scatter([f_left,f_right],[x_left,x_right],c='C0',marker='x',s=4)
            fspace = np.linspace(f_left,f_right,10)
            xspace = np.linspace(x_left,x_right,10)
            ax[0].plot(fspace,xspace,c='C0')
            x_int   = np.polyval(fitpars,f_LFC2)
            interpolated_LFC2.append(x_int)
            ax[1].scatter([f_LFC2],[(x_LFC2-x_int)*829],c='C1',marker='x',s=4)
            print("{:>3d}".format(index_LFC1),
                  (3*("{:>14.5f}")).format(x_left,x_LFC2,x_right),
                  "x_int = {:>10.5f}".format(x_int),
                  "RV = {:>10.5f}".format((x_LFC2-x_int)*829))
            #print(x_LFC2,interpolated_x)
        interpolated[od] = interpolated_LFC2
        #[plt.axvline(x1,ls='-',c='r') for x1 in f1]
        #[plt.axvline(x2,ls=':',c='g') for x2 in f2]
        break
    return interpolated
def get_comb_offset(source_anchor,source_offset,source_reprate,modefilter):
    m,k     = divmod(round((source_anchor-source_offset)/source_reprate),
                                   modefilter)
    comb_offset = (k-1)*source_reprate + source_offset #+ anchor_offset
#    f0_comb = k*source_reprate + source_offset 
    return comb_offset

# =============================================================================
#    
#                    W A V E L E N G T H    C A L I B R A T I O N
#                           H E L P E R    F U N C T I O N S
#   
# =============================================================================
# def contract(x,npix):
#     return 2*x/(npix-1) - 1.

# def expand(x,npix):
#     return (npix-1)*(x+1)/2 
def contract(x,xrange):
    assert len(xrange)==2
    x_mid  = np.average(xrange)
    x_span = (np.diff(xrange))/2
    return (x - x_mid) / x_span

def expand(x_contracted,xrange):
    assert len(xrange)==2
    x_mid  = np.average(xrange)
    x_span = (np.diff(xrange))/2
    return x_contracted * x_span + x_mid