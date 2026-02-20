#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:54:05 2023

@author: dmilakov
"""
import numpy as np
import scipy.interpolate as intp
import matplotlib.pyplot as plt
from matplotlib import ticker
import harps.plotter as hplot
import harps.lsf.gp as hlsfgp
import harps.lsf.plot as lp
import harps.containers as container
import harps.progress_bar as progress_bar
import harps.settings as hs
import harps.lsf.read as hread
import hashlib
from fitsio import FITS
import os

import jax.numpy as jnp

class LSF1d(object):
    def __init__(self,narray):
        self._values = narray
        self._cache  = {}
    def __len__(self):
        return len(self._values)
    def __getitem__(self,item):
        # condict, segm_sent = self._extract_item(item)
        values  = self.values 
        # condition = np.logical_and.reduce(tuple(values[key]==val \
        #                                for key,val in condict.items()))
        cut = np.where(values['segm']==item)
        return LSF1d(values[cut])

    @property
    def values(self):
        return self._values
    @property
    def x(self):
        return self._values['x']
    @property
    def y(self):
        return self._values['y']
    @property
    def pars(self):
        return self._values['pars']
    
    def save(self,filepath,version=None,overwrite=False):
        with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
            hdu.write(self.values,extname='LSF',extver=version)
        hdu.close()
        print("File saved to {}".format(filepath))
        return
    def plot(self,segm=None,ax=None,title=None,saveto=None,*args,**kwargs):
        
        return_fig = False
        if segm is not None:
            values = self[segm].values
        else:
            values = self.values
        if ax is not None:
            ax = ax  
        else:
            return_fig = True
            figure = hplot.Figure2(1,1,figsize=(5,4),
                                   left=0.12,bottom=0.12)
            ax = figure.add_subplot(0,1,0,1)
        print(values.shape)
        ax = lp.plot_numerical_model(ax,values,*args,**kwargs)
        # except:
        #     ax = plot_analytic_lsf(self.values,ax,title,saveto,*args,**kwargs)
        
        # ax.set_ylim(-5,100)
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel("Intensity (arb.)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5,steps=[1,2,5]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#        ax.set_yticklabels([])
        ax.grid(True,ls=':',lw=1,which='both',axis='both')

        if title:
            ax.set_title(title)
        if saveto:
            figure.savefig(saveto)
        if return_fig:
            return figure, ax
        else:
            return ax
    # @property
    # def numerical_model(self,xrange=(-8,8),subpix=11):
    #     try:
    #         nummodel = self._cache['numerical_model']
    #     except:
    #         nummodel = LSF1d(numerical_model(self.values,
    #                                        xrange=xrange,
    #                                        subpix=subpix))
    #         self._cache['numerical_model'] = nummodel
    #     return nummodel
    #     # LSF2d_numerical = LSF(lsf2d_numerical)
    def interpolate_lsf(self,center,N=2):
        lsf1d_num = self.values
        return interpolate_local_lsf(center,lsf1d_num,N=N)
    def interpolate_scatter(self,center,N=2):
        lsf1d_num = self.values
        return interpolate_local_scatter(center,lsf1d_num,N=N)
    
class LSF2d(object):
    def __init__(self,narray):
        self._values = narray
        self._cache  = {}
        
    def __len__(self):
        return len(self._values)
    def __getitem__(self,item):
        condict, segm_sent = self._extract_item(item)
        values  = self.values 
        condition = np.logical_and.reduce(tuple(values[key]==val \
                                       for key,val in condict.items()))
        cut = np.where(condition==True)[0]
        if len(cut)>0:
            return LSF1d(values[cut])
        else:
            print(f"No data for input {condict}")

    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning order plus segment.
        
        To be used with partial decorator
        """
        return _extract_item_(item)
    @property
    def values(self):
        return self._values
    @property
    def x(self):
        return self._values['x']
    @property
    def y(self):
        return self._values['y']
    @property
    def deriv(self):
        return self._values['dydx']
    @property
    def pars(self):
        return self._values['pars']
    @property
    def segcens(self):
        return get_segment_centres(self.values)
    def save(self,filepath,version=None,overwrite=False):
        with FITS(filepath,mode='rw',clobber=overwrite) as hdu:
            hdu.write(self.values,extname='LSF',extver=version)
        hdu.close()
        print("File saved to {}".format(filepath))
        return
    def plot(self,ax=None,title=None,saveto=None,*args,**kwargs):
        if ax is not None:
            ax = ax  
        else:
            figure = hplot.Figure2(1,1,left=0.08,bottom=0.12)
            ax = figure.add_subplot(0,1,0,1)
        # try:
        
        ax = lp.plot_numerical_model(ax,self._values,*args,**kwargs)
        # except:
        #     ax = plot_analytic_lsf(self.values,ax,title,saveto,*args,**kwargs)
        
        # ax.set_ylim(-5,100)
        ax.set_xlabel("Distance from center")
        ax.set_ylabel("Intensity (arb.)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5,steps=[1,2,5]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#        ax.set_yticklabels([])
        ax.grid(True,ls=':',lw=1,which='both',axis='both')

        if title:
            ax.set_title(title)
        if saveto:
            figure.savefig(saveto)
        return ax
    # @property
    # def numerical_model(self,xrange=(-8,8),subpix=11):
    #     try:
    #         nummodel = self._cache['numerical_model']
    #     except:
    #         nummodel = LSF1d(numerical_model(self.values,
    #                                        xrange=xrange,
    #                                        subpix=subpix))
    #         self._cache['numerical_model'] = nummodel
    #     return nummodel
    #     # LSF2d_numerical = LSF(lsf2d_numerical)
    def interpolate(self,order,center,N=2):
        lsf1d_num = self.numerical_model[order].values
        return interpolate_local_lsf(center,lsf1d_num,N=N)



    
def associate_waverange_to_segment(lsf2dObj,wstart,wend,wav3d,err3d=None):
    '''
    
    Given a series of 2d wavelength solutions, identifies segments of the 
    lsf2dObj that fall in the wavelength range between wstart and wend and 
    calculates their relative weights based on the number of pixels which fall
    in a specific segment. 
    
    If err3d is given, the weights are determined using inverse variances. 
    
    Returns a list containing [(order,segment),weight] for all contributing
    segments.

    Parameters
    ----------
    lsf2dObj : numpy structured array or LSF2d object
        Contains the LSF.
    wstart : scalar
        Start wavelength.
    wend : scalar
        End wavelength.
    wav3d : numpy array
        The wavelength solution for a single or multiple exposures.
    err3d : numpy array, optional
        The spectral error array for a single or multiple exposures. 
        The default is None.

    Returns
    -------
    return_list : list
        A list containing [(order,segment),weight] for all contributing
        segments.

    '''
    wav3d = np.atleast_3d(wav3d)
    var3d = np.atleast_3d(err3d)**2 if err3d is not None else np.ones_like(wav3d)

    orders, pixels,exposures = np.where((wav3d>=wstart)&(wav3d<=wend))
    lsfod, seglims = get_segment_limits(lsf2dObj)
    return_list = []
    for od in np.unique(orders):
        # condition = wstart<=w<=wend
        cut0   = np.where(orders==od)[0]
        pix    = pixels[cut0]  # pixels satisfying wstart<=w<=wend in order od
        exp    = exposures[cut0] # exposures also satysfying that condition 
        var    = var3d[od,pix,exp] # associated variance
        
        cut1   = np.where(lsfod==od)[0] # lsf array in order od
        bins   = seglims[cut1[0]] # segment edges in this order
        hist,_ = np.histogram(pix,bins) # number of contributing pixels per seg
        segm   = np.where(hist>0)[0] # segments with >0 contributing pixels 
        idx    = np.digitize(pix,bins[1:]) # digitize without the leftmost bin
        for s in segm:
            if (od==39 and s<8): continue
            cut2 = np.where(idx==s)[0] # pixels in this segment
            weight = np.sum(1./var[cut2])
            # save.append([od,s,exp,weight])
            return_list.append([(od,s),weight])
        # print(od,wave2d[od,pix])
        # print(od,hist,len(pix))
    sumw = np.sum([row[-1] for row in return_list])
    for row in return_list:
        row[-1] = row[-1]/sumw
    return return_list

def combine_ips(lsf2dObj,xrange,dv,subpix,wstart,wend,wave3d,err3d=None,
                plot=False,save=False,save_to=None):
    '''
    

    Parameters
    ----------
    lsf2dObj : TYPE
        DESCRIPTION.
    xrange : scalar
        x range covered, in pixels 
        (a range from -xrange to +xrange will be saved to file).
    dv : scalar
        step in velocity space (units km/s).
    subpix : scalar
        How many subpixels per 1 km/s.
    wstart : scalar
        Starting wavelength in Angstrom.
    wend : scalar
        End wavelength in Angstrom.
    wave3d : array-like
        A list of 2d wavelength solutions.
    err3d : array-like, optional
        A list of 2d error arrays. If provided, the program will use the 
        errors to weight each pixel by its inverse variance. Otherwise, 
        each pixel will have the same weight (1). The default is None.
    plot : boolean, optional
        Plot the final model. The default is False.
    save : boolean, optional
        Save to file. The default is False.
    save_to : string, optional
        Path to the file in which to save the model. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    assert isinstance(lsf2dObj,LSF2d)
    def kmps2pix(vel,dv):
        return vel/dv
    def pix2kmps(dv,vel):
        return vel * dv
    combination_info = associate_waverange_to_segment(lsf2dObj,wstart,wend,
                                                      wave3d,err3d=None)
    pixstep = 1/subpix
    x_pix =  np.arange(-xrange,xrange+pixstep,pixstep)
    x_vel = pix2kmps(dv,x_pix)
    y_vel = np.zeros_like(x_pix)
    if plot: plt.figure()
    for item,weight in combination_info:
        lsf1s = lsf2dObj[item]
        x1s = lsf1s.x
        y1s = lsf1s.y
        if len(x1s.shape)>1:
            x1s = x1s[0]
        if len(y1s.shape)>1:
            y1s = y1s[0]
        
        if plot: plt.scatter(x1s,y1s,s=2,label=f"{item},w={weight:3.1%}")
        splr = intp.splrep(x1s,y1s)
        y   = weight*intp.splev(x_vel,splr)
        print(item,weight)
        y_vel += y
    Y = y_vel + np.abs(np.min(y_vel)) # make sure all values are positive
    if plot:
        plt.scatter(x_vel,Y,marker='s',s=2,ls='-',lw=2)
        plt.legend()
    # x = lsf2dObj.x[0]
    # y = np.zeros_like(x)
    if save:
        
        if save_to is not None:
            save_to=save_to 
        else:
            from datetime import datetime
            save_dir   = hs.dirnames['vpfit_ip']
            _ = np.sum([wave3d,err3d])
            control = hashlib.md5(_).hexdigest()
            now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            save_fname = f"{wstart}-{wend}_{now}_{control}.dat"
            save_to = os.path.join(save_dir,save_fname)
        np.savetxt(save_to,np.transpose([x_pix,Y/np.sum(Y)]),fmt='%02.6e')
    
    return x_pix,Y

def combine_from_list_of_files(lsfpath,version,xrange,dv,subpix,wstart,wend,
                               wavefilelist,errfilelist=None,
                               save=False,save_to=None):
    '''
    

    Parameters
    ----------
    lsfpath : TYPE
        Path to the FITS file containing IP numerical models in velocity space.
    version : TYPE
        Version to use (version=1 uses the most likely IP).
    xrange : TYPE
        A new numerical model will span (-xrange,+xrange). Units pixels.
        xrange=8 seems ok
    dv : TYPE
        Step in velocity scale (units km/s). 0.82 for HARPS
    subpix : TYPE
        How many subpixels per pixel. Must be odd number. 21.
    wstart : TYPE
        Start wavlength. 
    wend : TYPE
        End wavelength.
    wavefilelist : TYPE
        List of RDGEN created files containing wavelengths.
    errfilelist : TYPE, optional
        List of RDGEN created files containing errors.
    save : TYPE, optional
        If True, saves output. The default is False.
    save_to : TYPE, optional
        If save is True, this is where the output will be written to file. 
        The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    with FITS(lsfpath,'r') as hdul:
        lsf_model=hdul['velocity_model',version].read()
    lsf2dObj = LSF2d(lsf_model)
    wavelist = []
    for file in wavefilelist:
        with FITS(file,'r') as hdul:
            wavelist.append(hdul[0].read())
    wav3d = np.dstack(wavelist)
    if errfilelist is not None:
        errlist = []
        for file in errfilelist:
            with FITS(file,'r') as hdul:
                errlist.append(hdul[0].read())
        err3d = np.dstack(errlist)
    else:
        err3d = None
    
    return combine_ips(lsf2dObj,xrange,dv,subpix,wstart,wend,wav3d,err3d,
                      save=save,save_to=save_to)
     
            
def interpolate_local_lsf(center,lsf1d_num,N=2):
    return interpolate_local(center,'LSF',lsf1d_num,N=N)
    
def interpolate_local_scatter(center,lsf1d_num,N=2):
    return interpolate_local(center,'scatter',lsf1d_num,N=N)
def interpolate_local(x,what,lsf1d_arr,N=2):
    '''
    Interpolates the local IP/scatter model at the position x. 

    Parameters
    ----------
    x : scalar
        position, x-coordinate in units detector pixels.
    what : string
        'LSF' or 'scatter'.
    lsf1d_arr : numpy structured array
        the array containing the numerical models of the LSF/scatter in a
        single echelle order.
    N : scalar, optional
        The number of closest models to use in interpolation. The default is 2.

    Returns
    -------
    loc_lsf_x : numpy array
        x-coordinates of the interpolated LSF/scatter model.
    loc_lsf_y : numpy array
        y-coordinates of the interpolated LSF/scatter model.

    '''
    assert np.isfinite(x)==True, "Center not finite, {}".format(x)
    # values  = lsf1d_num[order].values
    # assert len(values)>0, "No LSF model for order {}".format(order)
    order, (indices, weights) = get_segment_weights(lsf1d_arr, x, N=N)
    if what=='LSF':
        name = 'y'
    elif what=='scatter':
        name = 'scatter'
    lsf_array = [lsf1d_arr[i][name] for i in indices]
    loc_lsf_x    = lsf1d_arr[indices[0]]['x']
    loc_lsf_y    = helper_calculate_average(lsf_array,
                                            weights)
    return loc_lsf_x,loc_lsf_y

def helper_calculate_average(list_array,weights):
    N = len(list_array[0])
    weights_= np.vstack([np.full(N,w,dtype='float32') for w in weights])
    average = np.average(list_array,axis=0,weights=weights_) 
    return average


def _extract_item_(item):
    condict = {}
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
            

def sort_array(func):
    def wrapped_func(lsfObj,*args,**kwargs):
        if isinstance(lsfObj,LSF1d) or isinstance(lsfObj,LSF2d):
            values = lsfObj.values
        elif isinstance(lsfObj,np.ndarray):
            values = lsfObj
        else:
            print('NOT LSF1d or LSF2d or array')
        orders = np.unique(values['order'])
        if len(orders)==1:
            sorter = np.argsort(values['segm'])
        else:
            sorter_ = []
            for od in orders:
                cut = np.where(values['order']==od)[0]
                _ = np.argsort(values[cut]['segm'])
                sorter_.append(cut[_])
            sorter = np.vstack(sorter_)
        return orders,func(values[sorter],*args,**kwargs)
    return wrapped_func

@sort_array
def get_segment_ledges(lsfObj):
    return lsfObj['ledge']

@sort_array
def get_segment_redges(lsfObj):
    return lsfObj['redge']

@sort_array 
def get_segment_centres(lsfObj):
    return (lsfObj['ledge']+lsfObj['redge'])/2

@sort_array
def get_segment_limits(lsfObj):
    ledges = np.array(lsfObj['ledge'])
    redges = np.array(lsfObj['redge'])
    ndim = len(np.shape(ledges))
    a      = ledges
    if ndim>1:
        b  = np.transpose(redges[:,-1])
        return np.append(a,b[:,None],axis=-1)
    else:
        b  = redges
        return np.append(a,b,axis=-1)
    
@sort_array
def get_segment_weights(lsf1d,center,N=2):
    sorter=np.argsort(lsf1d['segm'])
    orders, segcens   = get_segment_centres(lsf1d[sorter])
    segdist   = np.diff(segcens)[0] # assumes equally spaced segment centres
    distances = np.abs(center-segcens)
    if N>1:
        used  = distances<segdist*(N-1)
    else:
        used  = distances<segdist/2.
    inv_dist  = 1./distances
    
    segments = np.where(used)[0]
    weights  = inv_dist[used]/np.sum(inv_dist[used])
    
    return segments, weights

def create_segment_indexation(nbo,npix,nseg):
    seglen = int(npix//nseg)
    array = np.zeros((nbo,nseg,seglen),dtype=np.int)
    for i,od in enumerate(range(nbo)):   
        k = 0
        for j,segm in enumerate(range(nseg)):
            array[i,j] = np.arange(k*seglen,(k+1)*seglen)
            k+=1
    return array

def get_array_segment(item,array,nseg=16):
    '''
    Assumes array is 2-dimensional. Routine divides the array into segments
    in the second dimension. 

    Parameters
    ----------
    item : integer (order) or a 2-tuple of integers (order,segment)
        DESCRIPTION.
    array : np.array
        DESCRIPTION.
    nseg : integer, optional
        Number of segments along the second axis. The default is 16.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    condict, segsent = _extract_item_(item)
    indexation = create_segment_indexation(*np.shape(array), nseg)
    if segsent:
        # cut1  = np.where(indexation[:,0,0]==condict['order'])
        order = condict['order']
        segm  = condict['segm']
        pix   = indexation[order,segm]
        return array[order,pix]
    else:
        order = condict['order']
        output = np.zeros(indexation.shape)
        for segm in range(nseg):
            pix   = indexation[order,segm]
            output[order,segm] = array[order,pix]
        return output[order]
        
        