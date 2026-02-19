#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:40:01 2018

@author: dmilakov
"""

from harps.core import np, plt
from harps.constants import c

import harps.inout as io
import harps.functions as hf
import harps.fit as fit
import harps.containers as container
import harps.plotter as plot
import harps.gaps as hg
import harps.version as hv
import numpy.polynomial.legendre as leg
#==============================================================================
#    
#                   H E L P E R     F U N C T I O N S  
#    
#==============================================================================
# def contract(x,npix):
#     return hf.contract(x,npix)

# def expand(x,npix):
#     return hf.expand(x,npix)
def contract(x,xrange):
    return hf.contract(x,xrange)

def expand(x,xrange):
    return hf.expand(x,xrange)

def evaluate(polytype,pars,x=None,xrange=None):
    if xrange:
        assert xrange[0]<xrange[1], "Starting pixel larger than ending pixel"
    x = x if x is not None else np.arange(*xrange,1)
    if not xrange:
        xrange = (np.min(x),np.max(x))
    x_contracted = contract(x,xrange)
    
    # Only proceed to evaluate polynomials if all coefficients are finite
    if np.all(np.isfinite(pars)) != True:
        return np.zeros_like(x)
    
    
    if polytype=='ordinary':
        return np.polyval(pars[::-1],x_contracted)
    elif polytype=='legendre':
        return leg.legval(x_contracted,pars)
def evaluate_centers(coefficients,centers,cerrors,polytype='ordinary',
                     errors=False):
    wave    = np.zeros(len(centers)) 
    waverr  = np.zeros(len(centers))
    for coeff in coefficients:
        order = coeff['order']
        pixl  = coeff['pixl']
        pixr  = coeff['pixr']
        cut   = np.where((centers >= pixl) & (centers <= pixr) )
        centsegm = centers[cut]
        
        pars     = coeff['pars']
        parerrs  = coeff['errs']
        wavesegm = evaluate(polytype,pars,centsegm,xrange=(pixl,pixr))
        wave[cut] = wavesegm
        if errors:
            waverr[cut] = error(centers[cut],cerrors[cut],pars,parerrs)
    if errors:
        return wave, waverr
    else:
        return wave        
def evaluate2d(coefficients,linelist,fittype='gauss',polytype='ordinary',
               errors=False):
    """
    Returns 1d array of wavelength of all lines from linelist, as calculated
    from the coefficients. 
    """
    centers = linelist[fittype][:,1]  
    cerrors = linelist['{}_err'.format(fittype)][:,1]
    return evaluate_centers(coefficients,centers,cerrors,polytype,errors)

def evaluate_on_linelist(coefficients,linelist,fittype='gauss',polytype='ordinary',
               errors=False):
    """
    Returns 1d array of wavelength of all lines from linelist, as calculated
    from the coefficients. 
    """
    fittype = f'{fittype}_pix'
    centers = linelist[fittype][:,1]  
    cerrors = linelist['{}_err'.format(fittype)][:,1]
    orders  = linelist['order']
    wave    = np.zeros(len(centers)) 
    waverr  = np.zeros(len(centers))
    
    for coeff in coefficients:
        order = coeff['order']
        pixl  = coeff['pixl']
        pixr  = coeff['pixr']
        cut   = np.where((centers >= pixl) & (centers <= pixr) & (orders==order))
        centsegm = centers[cut]
        
        pars     = coeff['pars']
        parerrs  = coeff['errs']
        if len(np.shape(pars))>1:
            pars = pars[0]
        
        wavesegm = evaluate(polytype,pars,centsegm)
        wave[cut] = wavesegm
        if errors:
            waverr[cut] = error(centers[cut],cerrors[cut],pars,parerrs)
    if errors:
        return wave, waverr
    else:
        return wave 
    

def dispersion(coeffs2d,npix,nord):
    '''
    

    Parameters
    ----------
    coeffs2d : TYPE
        DESCRIPTION.
    npix : TYPE
        DESCRIPTION.
    nord : INT
        Optional, number of echelle orders. If provided, the array will have 
        shape (nord, npix). Rows will be filled at the appropriate order. 
                    
                    This is used to ensure that the shape of the wavelength
                    solution matches the shape of the data frame (flux)

    Returns
    -------
    wavesol : TYPE
        DESCRIPTION.

    '''
    wavesol = disperse2d(coeffs2d,npix=npix,nord=nord)
    return wavesol

def disperse1d(coeffs,npix,polytype):
    wavesol1d  = np.zeros(npix)
    for segment in coeffs:
        pixl = int(segment['pixl'])
        pixr = int(segment['pixr'])
        pars = segment['pars']
        wavesol1d[pixl:pixr] = evaluate(polytype,pars,None,xrange=(pixl,pixr))
    return wavesol1d
def disperse2d(coeffs,npix,nord=None,polytype='ordinary'):
    if nord is not None:
        orders = np.arange(nord)
    else:
        orders    = np.unique(coeffs['order'])
    nbo       = np.max(orders)+1
    
    wavesol2d = np.zeros((nbo,npix))
    for order in orders:
        coeffs1d = coeffs[np.where(coeffs['order']==order)]
        wavesol2d[order] = disperse1d(coeffs1d,npix,polytype)
        
    return wavesol2d

def construct(coeffs,npix):
    """ For ThAr only"""
    #nbo,deg = np.shape(a)
    wavesol = np.array([np.polyval(c[::-1],np.arange(npix)) for c in coeffs])
    
    return wavesol


def _refrindex(wavelength,p=760.,t=15.):
    index    =  1e-6 * p * (1.0 + (1.049 - 0.0157 * t) * 1e-6 * p) \
                / 720.883 / (1.0 + 0.003661 * t) \
                *(64.328+29498.1/(146.0-np.power(1e4/wavelength,2)) \
                +255.4/(41.0-np.power(1e4/wavelength,2)))+1.0
    return index

def _to_vacuum(lambda_air,p=760.,t=15.):
    """
    Returns wavelengths in vacuum.
    
    Args:    
        lambda_air: 1D numpy array
    Returns:
        lambda_vacuum : 1D numpy array
    """
    if np.sum(lambda_air)!=0:
        index = _refrindex(lambda_air,p,t)
    else:
        index = 1
    lambda_vacuum = lambda_air*index
    return lambda_vacuum

def _to_air(lambda_vacuum,p=760.,t=15.):
    """
    Returns wavelengths in air.
    
    Args:    
        lambda_air: 1D numpy array
    Returns:
        lambda_vacuum : 1D numpy array
    """
    assert np.sum(lambda_vacuum)!=0, "Wavelength array is empty."
    index      = _refrindex(lambda_vacuum,p,t)
    lambda_air = lambda_vacuum/index
    return lambda_air

def residuals(linelist,coefficients,version,fittype,npix,anchor_offset=None,
              polytype='ordinary',**kwargs):
    print(version)
    anchor_offset  = anchor_offset if anchor_offset is not None else 0.0
    field = fittype+'_pix' if 'pix' not in fittype else fittype
    centers        = linelist[field][:,1]
    cerrors        = linelist[f'{field}_err'][:,1]
    photnoise      = linelist['noise']
    wavelengths    = hf.freq_to_lambda(linelist['freq']+anchor_offset)
    nlines         = len(linelist)
    result         = container.residuals(nlines)
    poly,gaps,segm = hv.unpack_integer(version,'wavesol')
    if gaps:
        gaps2d = hg.read_gaps(**kwargs)
    for coeff in coefficients:
        order = coeff['order']
        optord= coeff['optord']
        segm  = coeff['segm']
        pixl  = coeff['pixl']
        pixr  = coeff['pixr']
        cut   = np.where((linelist['order']==order) & 
                         (centers >= pixl) &
                         (centers <= pixr))
        centsegm = centers[cut]
        cerrsegm = cerrors[cut]
        wavereal  = wavelengths[cut]
        if gaps:
            cutgap = np.where(gaps2d['order']==order)[0]
            gaps1d = gaps2d[cutgap]['gaps'][0]
            centsegm = hg.introduce_gaps(centsegm,gaps1d)
            centers[cut] = centsegm
        wavefit = evaluate(polytype,coeff['pars'],centsegm,xrange=(pixl,pixr))
        
        waverr  = error(centsegm,cerrsegm,coeff['pars'],coeff['errs'],
                        polytype,npix)
        result['residual_A'][cut]=(wavefit-wavereal)
        result['residual_mps'][cut]=(wavefit-wavereal)/wavereal*c
        result['order'][cut]=order
        result['optord'][cut]=optord
        result['segm'][cut]=segm
        
        result['noise'][cut] = photnoise[cut]
        result['wavefit'][cut] = wavefit
        result['waverr'][cut] = waverr
    result[field]  = centers
    result['cenerr'] = cerrors
    
    return result
def distortions(linelist,coeff,order=None,fittype='gauss_pix',anchor_offset=None):
    '''
    Returns an array of distortions between the ThAr and LFC calibrations.
    Uses coefficients in air for ThAr, converts wavelengths to vacuum.
    
    Input:
    -----
        linelist:     array of lines, output from lines.detect
        coeff:        array of ThAr coefficients (in air!)
    '''
    anchor_offset = anchor_offset if anchor_offset is not None else 0.0
    
    orders    = np.unique(linelist['order'])
    wave      = hf.freq_to_lambda(linelist['freq']+anchor_offset)
        
    cens        = linelist['{}'.format(fittype)][:,1]
    distortions = container.distortions(len(linelist))
    for i,order in enumerate(orders):
        cut  = np.where(linelist['order']==order)[0]
        
        pars = coeff[order]['pars']
        if len(np.shape(pars))>1:
            pars = pars[0]
        thar_air = np.polyval(np.flip(pars),cens[cut])
        thar_vac = _to_vacuum(thar_air)
        shift    = (wave[cut]-thar_vac)/wave[cut] * c
        distortions['dist_mps'][cut] = shift
        distortions['dist_A'][cut]   = wave[cut]-thar_vac
        distortions['order'][cut]    = linelist['order'][cut]
        distortions['optord'][cut]   = linelist['optord'][cut]
        distortions['segm'][cut]     = linelist['segm'][cut]
        distortions['freq'][cut]     = linelist['freq'][cut]
        distortions['mode'][cut]     = linelist['mode'][cut]
        distortions['cent'][cut]     = cens[cut]
    return distortions
def twopoint_coeffs(linelist,fittype='gauss',exclude_gaps=True,*args,**kwargs):
    """ Uses the input array to return the coefficients of the wavelength 
    calibration by interpolating between neighbouring comb lines.
    """
    def exclude(coeffs,lim):
        cut = ((coeffs['pixl']<=lim)&(coeffs['pixr']>=lim))
        #print(lim,np.sum(cut))
        return coeffs[~cut]
    MOD    = 2
    
    
    numseg = len(linelist)-1
    coeffs = container.coeffs(1,numseg) 
    for i in range(numseg):
        if linelist['order'][i]!=linelist['order'][i+1]:
            continue
        order = linelist['order'][i]
        left  = linelist[fittype][i,1]
        right = linelist[fittype][i+1,1]
        if left>right:
            continue
        if not np.isfinite(left) or not np.isfinite(right):
            #print('left',left)
            #print('right',right)
            continue
#        pixl  = left#np.int(np.around(left/MOD)*MOD)
#        pixr  = right#np.int(np.around(right/MOD)*MOD)
        
        waveL = hf.freq_to_lambda(linelist['freq'][i])
        waveR = hf.freq_to_lambda(linelist['freq'][i+1])
        # y(x) = a0 + a1*x
        a0    = waveL - (waveR-waveL)/(right-left)*left
        a1    = (waveR-waveL)/(right-left)
        coeffs[i]['order'] = order
        coeffs[i]['segm']  = i
        coeffs[i]['pixl']  = left
        coeffs[i]['pixr']  = right
        coeffs[i]['pars']  = [a0,a1]
    if exclude_gaps:
        seglims = np.linspace(512*1,512*8,8)
        coeffs0 = np.copy(coeffs)
        for lim in seglims:
            coeffs0  = exclude(coeffs0,lim)
        coeffs = coeffs0
    #dispersion = disperse2d(coeffs,npix)
    #np.place(dispersion,dispersion==0,np.nan)
    return coeffs
def twopoint(linelist,fittype,npix,full_output=False,
             exclude_gaps=True,*args,**kwargs):
    """ Uses the input array to return the coefficients of the wavelength 
    calibration by interpolating between neighbouring comb lines.
    """
    
    coeffs = twopoint_coeffs(linelist,fittype,exclude_gaps,*args,**kwargs)
    dispersion = disperse2d(coeffs,npix)
    np.place(dispersion,dispersion==0,np.nan)
    if full_output:
        return dispersion, coeffs
    else:
        return dispersion
def polynomial(linelist,version,fittype,npix,nord=None,
               full_output=False,*args,**kwargs):

    coeffs = fit.dispersion(linelist,version,fittype,npix=npix,
                            *args,**kwargs)
    dispersion = disperse2d(coeffs,npix,nord)
    if full_output:
        return dispersion, coeffs
    else:
        return dispersion
def error(centers,cerrors,pars,parerrs,polytype,npix):
    ''' 
    Returns the errors (in A) on the wavelength calibration fit.
    
    Assumes fitting was performed using a normalized coordinate system, in
    which x in [0,1].
    
    lambda(x) = sum_i ( a_i * x**i ) = sum_i (a'_i * x'**i )
    
    If
        x'   = npix * x
    then
        a'_i = a_i / npix
    
    Calculates the error on wavelength as 
    
    sigma_lambda**2 = (dy/dx * sigma_x)**2 + sum_i ( (dy/da_i * sigma_a_i)**2 )
    
    but transforms it to the primed coordinate system.
    '''
    npars = len(pars)
    dydx  = np.arange(npars)[:,np.newaxis] * evaluate(polytype,pars,centers)
    xvar  = np.sum(dydx/npix * cerrors/npix,axis=0)

#    pvar0 = np.zeros(len(centers))
    pvar = np.sum([(centers/npix)**i*parerrs[i] for i in range(npars)],axis=0)
#    plt.plot(xvar)
#    plt.plot(pvar)
    return np.sqrt(xvar)
allowed_calibrators = ['thar','comb']
#==============================================================================
#    
#                       W A V E S O L    C L A S S  
#    
#==============================================================================
class Wavesol(object):
    __metric = {"A":1,"m":1e10,'m/s':1}
    def __init__(self,narray,unit='A'):
        self._values  = narray
        self._unit    = unit
    def __getitem__(self,item):
        order,pix, pix_sent = self._extract_item(item)
        
        values = self.values
        if pix_sent:
            return Wavesol(values[order,pix],self.unit)
        else:
            return Wavesol(values[order],self.unit)
        
    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning order plus pixel limits.
        
        To be used with partial decorator
        """
        pix_sent = False
        pix      = None
        if isinstance(item,tuple):
            if len(item)==2: 
                pix_sent=True
                order = item[0]
                pix   = item[1]
            if len(item)==1:
                order = item
        else:
            order = item
        return order, pix, pix_sent
    def __add__(self,other):
        values0 = self._into_angstrom() 
        values1 = other._into_angstrom()
        
        return Wavesol((values0+values1)/Wavesol.__metric[self.unit],self.unit)
    def __sub__(self,other):
        values0 = self._into_angstrom() 
        values1 = other._into_angstrom()
        
        return Wavesol((values0-values1)/Wavesol.__metric[self.unit],self.unit)
    def __neg__(self):
        return Wavesol(-self.values,self.unit)
    def __pos__(self):
        return Wavesol(self.values,self.unit)
    def __truediv__(self,other):
        dv      = np.zeros_like(self.values)
        values0 = self._into_angstrom() 
        values1 = other._into_angstrom()
        cut0    = values0!=0
        cut1    = values1!=0
        cut     = np.where(cut0&cut1)
        
        dv[cut] = (values1[cut]-values0[cut])/values1[cut] * c
        return Wavesol(dv,'m/s')
    def _into_angstrom(self):
        return self.values * Wavesol.__metric[self.unit]
    
    @property
    def values(self):
        return self._values
    @property
    def unit(self):
        return self._unit
    @property
    def shape(self):
        return np.shape(self.values)
    def plot(self,plot2d=False,*args,**kwargs):
        figure = plot.Figure(1)
        fig    = figure.fig
        ax     = figure.axes
        values = self.values
        sumord = np.sum(values,axis=1)
        orders = np.where(sumord!=0)[0]
        
        if plot2d:
            
            image = ax[0].imshow(values,aspect='auto')
            cbar  = fig.colorbar(image)
            cbar.set_label(self.unit)
        else:
            numord = len(orders)
            colors = plt.cm.jet(np.linspace(0,1,10))
            if numord>5:
                colors = plt.cm.jet(np.linspace(0,1,numord))
            for i,order in enumerate(orders):
                if np.sum(values[order])==0:
                    continue
                ax[0].plot(values[order],c=colors[i],lw=0.8)
        return figure
                
#==============================================================================
#    
#               T H O R I U M    A R G O N     F U N C T I O N S  
#    
#==============================================================================   
class ThAr(object):
    def __init__(self,filepath,vacuum):
        self._filepath  = filepath
        self._vacuum    = vacuum
        self._coeffs_air = None 
        self._coeffs_vac = None
        self._bad_orders = None
        # self._qc         = None
        pass
    def __call__(self,vacuum,npix):
        dispers2d, bad_orders= get_thar_solution(self._filepath,vacuum,npix)
        self._bad_orders = bad_orders
        # self._qc          = qc
        return dispers2d   
    def get_wavecoeff_vacuum(self):    
        def get1d(order):
            if order not in bad_orders:
                pars   = np.polyfit(np.arange(npix),
                                    wavesol_vacuum[order],
                                    deg)
            else:
                pars = np.zeros((1,deg+1))
            return (order,optical[order],0,0,4095,-1.,-1.,-1.,0,
                    np.flip(pars),np.zeros_like(pars))
        npix=4096
        wavesol_vacuum, bad_orders = get_thar_solution(self._filepath,True,npix)
        meta   = io.read_e2ds_meta(self._filepath)
        nbo    = meta['nbo']
        deg    = 3 # HARPS uses cubic 
        npix   = meta['npix'] 
        optical= io.read_optical_orders(self._filepath)
        coeff2d = container.coeffs(deg,nbo)
        for order in range(nbo):
            coeff2d[order] = get1d(order)
        
        return coeff2d
    
    def get_coeffs(self,vacuum):
        if vacuum==False:
            if self._coeffs_air is not None:
                pass
            else:
                coeffs,bad_orders = get_wavecoeff_air(self._filepath)
                self._coeffs_air = coeffs
            return self._coeffs_air
        if vacuum==True:
            if self._coeffs_vac is not None:
                pass
            else:
                self._coeffs_vac = self.get_wavecoeff_vacuum()
            return self._coeffs_vac

    @property
    def bad_orders(self):
        if self._bad_orders is not None:
            pass
        else:
            coeffs,bad_orders  = get_wavecoeff_air(self._filepath)
            self._bad_orders = bad_orders
        return self._bad_orders
    # @property
    # def qc(self):
    #     if self._qc is not None:
    #         pass
    #     else:
    #         coeffs,bad_orders,qc = ThAr._get_wavecoeff_air(self._filepath)
    #         self._qc = qc
    #     return self._qc
def get_wavecoeff_air(filepath):
     ''' 
     Returns coefficients of a third-order polynomial from the FITS file 
     header in a matrix. This procedure is described in the HARPS DRS  
     user manual.
     
     https://www.eso.org/sci/facilities/lasilla/
             instruments/harps/doc/DRS.pdf
             
     HARPS specific, do not use on ESPRESSO spectra.
     '''
     
     header  = io.read_e2ds_header(filepath)
     meta    = io.read_e2ds_meta(filepath)
     nbo     = meta['nbo']
     deg     = 3 # HARPS uses cubic 
     optical = io.read_optical_orders(filepath)
     coeffs  = np.vstack([_read_wavecoef1d(order,header,deg,optical) \
                          for order in range(nbo)])
     bad_orders = np.unique(np.where(np.sum(coeffs['pars'],axis=1)==0)[0])
     # qc     = meta['qc']
     return coeffs, bad_orders#, qc
def _read_wavecoef1d(order,header,deg,optical):
    """ 
    Returns ThAr wavelength calibration coefficients saved in the header.
    Returns zeroes when no coefficients are found.
    """
    coeff1d = container.coeffs(deg,1)
    coeff1d['order'] = order
    coeff1d['pixl']  = 0
    coeff1d['pixr']  = 4095
    coeff1d['optord']= optical[order]
    for i in range(deg+1):                    
        ll    = i + order*(deg+1)
        try:
            a = header["ESO DRS CAL TH COEFF LL{0}".format(ll)]
            coeff1d['pars'][0,i] = a
        except:
            continue
    return coeff1d

def get_thar_solution(filepath,vacuum,npix):
    """ 
    Return the ThAr wavelength solution, as saved in the header of the
    e2ds file. 
    """
    coeffs, bad_orders = get_wavecoeff_air(filepath)
    wavesol_air = construct(coeffs['pars'][:,0,:],npix)
    if vacuum==True:
        return _to_vacuum(wavesol_air), bad_orders
    else:
        return wavesol_air, bad_orders
    
class ThFP(object):
    def __init__(self,filepath,vacuum):
        self._filepath  = filepath
        self._vacuum    = vacuum
        self._coeffs_air = None 
        self._coeffs_vac = None
        self._bad_orders = None
        # self._qc         = None
        pass
    def __call__(self,vacuum=True):
        hdul = io.FITS(self._filepath)
        wavedispersion2d = hdul[1].read()
        if vacuum==False:
            wavedispersion2d=_to_air(wavedispersion2d)
        
        return wavedispersion2d
        
        
#==============================================================================
#    
#               L A S E R    F R E Q U E N C Y    C O M B
#
#                            F U N C T I O N S  
#    
#==============================================================================
def get_wavecoeff_comb(linelist,version,fittype,npix):
    """
    Returns a dictionary with the wavelength solution coefficients derived from
    LFC lines
    """
    if version==1:
        coeffs2d = twopoint_coeffs(linelist,fittype,npix)
    else:
        coeffs2d = fit.dispersion(linelist,version,fittype,npix)
    return coeffs2d

def comb_dispersion(linelist,version,fittype,npix,*args,**kwargs):
    if version==1:
        wavesol_comb = twopoint(linelist,fittype,npix,*args,**kwargs)
    else:
        wavesol_comb = polynomial(linelist,version,fittype,npix,*args,**kwargs)
    #coeffs2d = get_wavecoeff_comb(linelist,version,fittype)
    #wavesol_comb = dispersion(coeffs2d,npix)
    return wavesol_comb
