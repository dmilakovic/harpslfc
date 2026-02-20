#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:45:04 2018

@author: dmilakov
"""
#from harps.core import sys
from harps.core import np
from harps.core import os
# import numbers
#from harps.core import leastsq, curve_fit,  interpolate
from harps.core import FITS, FITSHDR
from harps.core import plt
#from harps.core import warnings, numbers

#from multiprocessing import Pool

import harps.functions as hf
import harps.settings as hs
import harps.inout as io
import harps.wavesol as ws
import harps.background as background
import harps.lines as lines
import harps.spec_aux as saux
import harps.version as hv
import harps.lines_aux as laux
import harps.functions.spectral as specfunc 

from harps.constants import c
import harps.containers as container
from harps.plotter import Figure2, ccd_from_linelist, ticks, scinotate

from matplotlib import ticker
import logging
version      = hs.__version__
harps_home   = hs.harps_home
harps_data   = hs.harps_data
harps_dtprod = hs.harps_dtprod
harps_plots  = hs.harps_plot
harps_prod   = hs.harps_prod


# hs.setup_logging()
analysis = 'normal'
# analysis = 'technical'
primary_head_names = ['Simple','Bitpix','Naxis','Extend','Author',
                     'npix','mjd','date-obs','fibshape','totflux']
if analysis in ['technical']:
    for name in ['temp7','temp22','temp23','temp30','temp31','temp32',
                     'temp33','temp40','temp41','temp44','pressure','exptime',
                     'det1_ctot','det2_ctot','lfc_slmlevel','lfc_status']:
        primary_head_names.append(name)
                     

class Spectrum(object):
    ''' Spectrum object contains functions and methods to read data from a 
        FITS file processed by the HARPS pipeline
    '''
    def __init__(self,filepath,f0=None,fr=None,vacuum=None,f0_offset=None,
                 model='SingleGaussian',instrument='HARPS',blazepath=None,
                 lsf_filepath=None,
                 overwrite=False,ftype=None,sOrder=None,eOrder=None,dirpath=None,
                 filename=None,logger=None,debug=False):
        '''
        Initialise the Spectrum.
        '''
        self.filepath = filepath
        self.name     = "LFC Spectrum"
        basename_str  = os.path.basename(filepath)
        filename_str  = os.path.splitext(basename_str)[0]
        filetype_str  = basename_str.split('_')[1]
        self.filetype = ftype if ftype is not None else filetype_str
        self.logger   = logger or logging.getLogger(__name__)
        self.blazepath = blazepath
        self.lsf_filepath = lsf_filepath
        
        
        
        f0_offset = f0_offset if f0_offset is not None else 0
        if f0 is not None:
            self.lfckeys['comb_anchor']  = f0 + f0_offset
        if fr is not None:
            self.lfckeys['comb_reprate'] = fr
        print(self.lfckeys)
        
        
        self.npix     = self.meta['npix']
        self.nbo      = self.meta['nbo']
        # self.d        = self.meta['d']
        self.sOrder   = sOrder if sOrder is not None else hs.sOrder
        self.eOrder   = eOrder if eOrder is not None else self.meta['nbo']
        self.model    = model
        
        self.version  = self._item_to_version()
        versiondict   = self._version_to_dict(self.version)
        self.polyord  = versiondict['polyord']
        self.gaps     = versiondict['gaps']
        self.segment  = versiondict['segment']
        
        if self.blazepath is not None:
            ext=1 if self.instrument=='ESPRESSO' else 0
            with FITS(self.blazepath) as hdul:
                blaze = hdul[ext].read()
        else:
            blaze = np.ones((self.nbo,self.npix))
            
        self.blaze = blaze
        self.flux  = self.flux/self.blaze*self.meta['gain']
        self.data  = self.flux
        
        
        
            
        self.datetime = np.datetime64(self.meta['obsdate'])
        dirpath       = dirpath if dirpath is not None else None
        exists        = io.fits_exists('fits',self.filepath)
        self._outpath = io.get_fits_path('fits',self.filepath,
                                         version,dirpath,filename)        
        
        if not exists or overwrite:
            self.write_primaryheader(overwrite=True)
            
        #self.wavesol  = Wavesol(self)
        if debug:
            self.logger.info("{} initialized".format(filename_str))
        
        return
    def __getitem__(self,item):
        '''
        Tries reading data from file, otherwise runs __call__. 
        
        Args:
        ----
            datatype (str) :  ['linelist','coeff','model_gauss',
                               'wavesol_comb','wavesol_thar']
            save     (bool):  saves to the FITS file if true
        
        Returns:
        -------
            data (array_like) : values of dataset
            
        '''
        ext, ver, versent = saux.extract_item(item)
        #print(ext,ver,versent)
        mess = f"Extension {ext:>20}"
        if versent:
            mess+= f", version {ver:<5}:"
        
        status = ' failed.'
        with FITS(self._outpath,'rw') as hdu:
            try:
                if versent:
                    data = hdu[ext,ver].read()
                else:
                    data = hdu[ext].read()
                status  = " read from file."
            except:
                data   = self.__call__(ext,ver)
                header = self.return_header(ext)
                hdu.write(data=data,header=header,extname=ext,extver=ver)
                status = " calculated and saved to file."
            finally:
                self.log('__getitem__',20,mess+status)
        return data

    def __str__(self):
        meta     = self.meta
        dirname  = os.path.dirname(self.filepath)
        basename = os.path.basename(self.filepath)
        mess =  "{0:^80s} \n".format("S P E C T R U M")+\
                "{0:-^80s} \n".format("")+\
                "{0:<20s}:{1:>60s}\n".format("Directory",dirname)+\
                "{0:<20s}:{1:>60s}\n".format("File",basename)+\
                "{0:<20s}:{1:>60.2f} GHz\n".format("LFC f0",self.lfckeys['comb_anchor']/1e9)+\
                "{0:<20s}:{1:>60.2f} GHz\n".format("LFC fr",self.lfckeys['comb_reprate']/1e9)+\
                "{0:<20s}:{1:>60s}\n".format("Obsdate",meta['obsdate'])+\
                "{0:<20s}:{1:>60s}\n".format("Model",meta['model'])
        return mess
    
    def __call__(self,dataset,version=None,write=False,debug=False,
                 update=False,*args,**kwargs):
        """ 
        
        Calculate dataset.
        
        Parameters are dataset name and version. 
        
        Args:
        ----
        dataset (str) : name of the dataset
        version (int) : version number, 3 digit (PGS)
                        P = polynomial order
                        G = gaps
                        S = segment
        
        Returns:
        -------
        data (array_like) : values of dataset
            
        """
        def funcargs(name):
            if   name == 'linelist':
                args_ = args
            elif name =='coeff_gauss':
                args_ = (self['linelist'],version,'gauss',self.npix)
            elif name =='coeff_lsf':
                args_ = (self['linelist'],version,'lsf',self.npix)
            elif name=='wavesol_gauss':
                args_ = (self['linelist'],version,'gauss',self.npix)
            elif name=='wavesol_lsf':
                args_ = (self['linelist'],version,'lsf',self.npix)
            elif name=='residuals_gauss':
                args_ = (self['linelist'],self['coeff_gauss',version],
                        version,'gauss',self.npix)
            elif name=='residuals_lsf':
                args_ = (self['linelist'],self['coeff_lsf',version],
                        version,'lsf',self.npix)
            elif name=='wavesol_2pt_gauss':
                args_ = (self['linelist'],'gauss',self.npix)
            elif name=='wavesol_2pt_lsf':
                args_ = (self['linelist'],'lsf',self.npix)
            return args_
        # def funckwargs(name):
        #     if   name == 'linelist':
        #         kwargs_ = dict(spec=self,lsf_filepath=self.lsf_filepath)
        #     else:
        #         kwargs_ = dict()
        #     return kwargs_
        assert dataset in io.allowed_hdutypes, "Allowed: {}".format(io.allowed_hdutypes)
        # version = hf.item_to_version(version)
        functions = {'linelist':lines.detect,
                     'line_positions':self.line_positions,
                     'extrema':self.get_extrema2d,
                     'error':self.get_error2d,
                     'coeff_gauss':ws.get_wavecoeff_comb,
                     'coeff_lsf':ws.get_wavecoeff_comb,
                     'wavesol_gauss':ws.comb_dispersion,
                     'wavesol_lsf':ws.comb_dispersion,
                     'model_gauss':lines.model_gauss,
                     'model_lsf':lines.model_lsf,
                     'residuals_gauss':ws.residuals,
                     'residuals_lsf':ws.residuals,
                     'wavesol_2pt_gauss':ws.twopoint,
                     'wavesol_2pt_lsf':ws.twopoint,
                     'weights':self.get_weights2d,
                     'error':self.get_error2d,
                     #'background':self.background,
                     # 'envelope':self.get_envelope,
                     'wavereference':self.wavereference,
                     'noise':self.sigmav2d,
                     'flux':getattr,
                     # 'flux_norm':self.normalised_flux,
                     # 'error_norm':self.normalised_error
                     }
        if debug:
            msg = 'Calling {}'.format(functions[dataset])
            self.log('__call__',20,msg)
            print(msg)
        if dataset in ['coeff_gauss','coeff_lsf',
                       'wavesol_gauss','wavesol_lsf',
                       'residuals_gauss','residuals_lsf',
                       'wavesol_2pt_gauss','wavesol_2pt_lsf']:
            # print(dataset,'funcargs=',funcargs(dataset))
            data = functions[dataset](*funcargs(dataset))
        elif dataset in ['weights','noise','extrema','error']:
            data = functions[dataset]()
        elif dataset in ['linelist','model_gauss','model_lsf']:
            data = functions[dataset](self,*args,**kwargs)
        elif dataset in ['flux','background','envelope','line_positions',
                         'wavereference','flux_norm','err_norm',
                         ]:
            data = getattr(self,dataset)
        if write:
            with FITS(self._outpath,'rw') as hdu:
                header = self.return_header(dataset)
                hdu.write(data=data,header=header,extname=dataset,extver=version)
        # if update:
        #     with FITS(self._outpath,'rw') as hdu:
        #         header = self.return_header(dataset)
        #         hdu[dataset,version].write(data=data,header=header)
        return data

    @staticmethod
    def _version_to_dict(ver):
        """ 
        Converts the integer representation of settings into a dictionary.
        
        Args:
        ----
        ver (int) : version number
        
        Returns:
        -------
        dict : (polyord,gaps,segment)
        """
        if isinstance(ver,int) and ver>99 and ver<1000:
            split  = [int((ver/10**x)%10) for x in range(3)][::-1]
            polyord, gaps, segment = split
        return dict(polyord=polyord,gaps=gaps,segment=segment)
    def _item_to_version(self,item=None):
        """
        Returns an integer representing the settings provided
        
        Returns the default version if no args provided.
        
        Args:
        -----
        item (dict,int,tuple) : contains information on the version
        
        Returns:
        -------
        version (int): 
        """

        return saux.item_to_version(item)
    
    def _extract_item(self,item):
        """
        Utility function to extract an "item", meaning
        a extension number,name plus version.
        """

        ext,ver,ver_sent = saux.extract_item(item)
        return ext,ver,ver_sent
    def log(self,name,level,message,*args,**kwargs):
        '''
        Logs 'message' of given level to logger 'name'. Fails silently.
        '''
        try:
            log = logging.getLogger(self.logger.name+'.'+name)
            log.log(level,message,*args,**kwargs)
        except:
            pass
        return 
    def write(self,data,extname,version=None,filepath=None):
        """
        Writes the input item (extension plus version) to the output HDU file.
        Equivalent to __call__(item,write=True).
        """
        versent = True if version is not None else False
        # data   = self.__call__(ext,ver)
        header = self.return_header(extname)
        
        filepath = filepath if filepath is not None else self._outpath
        with FITS(filepath,'rw') as hdu:
            if versent:
                hdu.write(data=data,header=header,
                          extname=extname,extver=version)
            else:
                hdu.write(data=data,header=header,extname=extname)
        return data
    def write_primaryheader(self,overwrite=False):
        ''' Writes the spectrum metadata to the HDU header'''
        header = self.return_header('primary')
        with FITS(self._outpath,'rw',clobber=overwrite) as hdu:
            # hdu[0].write_keys(header)
            hdu.write(data=np.array([0]),header=header)
        # hdul = FITS(self._outpath,'rw',clobber=overwrite)
        #print(header)
        # hdul[0].write_keys(header)
        # hdul.close()
        return 
    def return_header(self,extension):
        """
        Returns a FITSHDR object with header information for this extension.
        """
        meta = self.meta
        LFC  = self.lfckeys
        # ------- Reads metadata and LFC keywords
        
        def return_value(name):
            if name=='Simple':
                value = True
            elif name=='Bitpix':
                value = 32
            elif name=='Naxis':
                value = 0
            elif name=='Extend':
                value = True
            elif name=='Author':
                value = 'Dinko Milakovic'
            elif name=='version':
                value = version
            elif name=='npix':
                value = meta['npix']
            elif name=='mjd':
                value = meta['mjd']
            elif name=='date-obs':
                value = meta['obsdate']
            elif name=='fibshape':
                value = meta['fibshape']
            # elif name=='lfc':
                # value = LFC['name'],
            elif name=='reprate':
                value = LFC['comb_reprate']
            elif name=='anchor':
                value = LFC['comb_anchor']
            elif name=='gaps':
                value = meta['gaps']
            elif name=='segment':
                value = meta['segment']
            elif name=='polyord':
                value = meta['polyord']
            elif name=='model':
                value = meta['model']
            elif name=='totflux':
                value = np.sum(self.data)
            elif name=='totnoise':
                value = self.sigmav()
            elif name=='pressure':
                value = self.header['HIERARCH ESO INS SENS1 VAL']
            elif name=='exptime':
                value = self.header['EXPTIME']
            elif name=='det1_ctot':
                value = self.header['HIERARCH ESO INS DET1 CTTOT']
            elif name=='det2_ctot':
                value = self.header['HIERARCH ESO INS DET2 CTTOT']
            elif name=='lfc_slmlevel':
                value = self.header['HIERARCH ESO INS LFC1 SLMLEVEL']
            elif name=='lfc_status':
                value = self.header['HIERARCH ESO INS LFC1 STATUS']
            elif 'temp' in name:
                upper = name.upper()
                value = self.header['HIERARCH ESO INS {} VAL'.format(upper)]
            else:
                self.log('COULD NOT FIND VALUE FOR {}'.format(name),40,'header')
            return value
        def make_dict(name,value,comment=''):
            return dict(name=name,value=value,comment=comment)
            
        if extension == 'primary':
            names = primary_head_names.copy()
            
        elif extension == 'linelist':
            names = ['version','totflux']
        elif extension in ['wavesol_gauss','wavesol_lsf']:
            names = ['anchor','reprate','gaps','segment','polyord']
        elif extension in ['coeff_gauss','coeff_lsf']:
            names = ['gaps','segment','polyord']
        elif extension in ['model_gauss', 'model_lsf']:
            names = ['model']
        elif extension in ['residuals_gauss', 'residuals_lsf']:
            names = ['anchor','reprate','gaps','segment','polyord']
        elif extension in ['wavesol_2pt_gauss','wavesol_2pt_lsf']:
            names = ['anchor','reprate']
        elif extension == 'weights':
            names = ['version']
        elif extension in ['flux','error','background','envelope','noise',
                           'line_positions','extrema',
                           'wavereference','flux_norm','error_norm']:
            names = ['totflux']
        else:
            raise UserWarning("HDU type not recognised")

        comments_dict={'Simple':'Conforms to FITS standard',
                  'Bitpix':'Bits per data value',
                  'Naxis':'Number of data axes',
                  'Extend':'FITS dataset may contain extensions',
                  'Author':'',
                  'version':'Code version used',
                  'npix':'Number of pixels',
                  'mjd':'Modified Julian Date',
                  'date-obs':'Date of observation',
                  'fibshape':'Fibre shape',
                  'lfc':'LFC name',
                  'reprate':'LFC repetition frequency',
                  'anchor':'LFC offset frequency',
                  'gaps':'Shift lines using gap file',
                  'segment':'Fit wavelength solution in 512 pix segments',
                  'polyord':'Polynomial order of the wavelength solution',
                  'model':'EmissionLine class used to fit lines',
                  'totflux':'Total flux in the exposure',
                  'totnoise':'Photon noise of the exposure [m/s]',
                  'temp7':'VV inside Detector side',
                  'temp22':'Collimator',
                  'temp23':'Echelle grating',
                  'temp30':'Temperature Air-Coude room w',
                  'temp31':'Air HARPS enclosure',
                  'temp32':'Air HARPS isolation box',
                  'temp33':'Air through fan 4 IB',
                  'temp40':'CCD control reference',
                  'temp41':'CCD secondary',
                  'temp44':'FP temp',
                  'pressure':'Inside pressure',
                  'exptime':'Exposure time',
                  'det1_ctot':'Total counts detector 1',
                  'det2_ctot':'Total counts detector 2',
                  'lfc_slmlevel':'LFC attenuation',
                  'lfc_status':'LFC status'}
        values_dict = {name:return_value(name) for name in names}
        if extension=='primary':
            #b2e = self.background/self.envelope
            for order in range(self.nbo):
                flxord = 'flux{0:03d}'.format(order+1)
                names.append(flxord)
                values_dict[flxord] = np.nansum(self.data[order])
                comments_dict[flxord] = "Total flux in order {0:03d}".format(order+1)
            # for order in range(self.nbo):
            #     b2eord = 'b2e{0:03d}'.format(order+1)
            #     names.append(b2eord)
            #     valord = b2e[order]
            #     index  = np.isfinite(valord)
            #     nanmean = np.nanmean(valord[index])
            #     if not np.isfinite(nanmean):
            #         nanmean = 0.0
            #     values_dict[b2eord] = nanmean
            #     comments_dict[b2eord] = "Mean B2E in order {0:03d}".format(order+1)
        values   = [values_dict[name] for name in names]
        comments = [comments_dict[name] for name in names]
        header   = [make_dict(n,v,c) for n,v,c in zip(names,values,comments)]
        return FITSHDR(header)


    @property
    def error(self,*args):
        """
        Returns the 2d error on flux values for the entire exposure. Caches the
        array.
        """
        return self['error']
        # try:
        #     error2d = self._cache['error2d']
        # except:
        #     # 1. Check if it's on disk before calculating
        #     # This is faster than the 2D process
        #     try:
        #         return self.__getitem__('error')
        #     except:
        #         error2d = self.get_error2d()  
        #         self.write(error2d, 'error')
        #         self._cache['error2d']=error2d
        # return error2d
        
    def get_error2d(self):
        """
        Returns a 2d array with errors on flux values. Adds the error due to 
        background subtraction in quadrature to the photon counting error.
        """
        data2d  = np.abs(self.data)
        bkg2d   = self['background']
        error2d = np.sqrt(np.abs(data2d) + np.abs(bkg2d))
        return error2d
    
    
    def get_linepos_env_bkg(self):
        flux2d = self.flux
        sOrder = self.sOrder
        eOrder = self.eOrder
        line_positions, env2d, bkg2d = background.get_linepos_env_bkg(flux2d,sOrder,plot=False,verbose=False)
        # 3. Save ALL THREE to disk immediately to save time later
        self.write(bkg2d, 'background')
        self.write(env2d, 'envelope')
        self.write(line_positions, 'line_positions')
        self._cache['line_positions'] = line_positions
        self._cache['envelope2d']=env2d
        self._cache['background2d']=bkg2d
        return line_positions, env2d, bkg2d
    
    
    @property
    def background(self):
        """
        Returns the 2d background model for the entire exposure. Caches the
        array.
        """
        try:
            return self.__getitem__('background')
        except:
            
            line_positions, env2d, bkg2d = self.get_linepos_env_bkg()
        return bkg2d
    def get_background(self,*args):
        """
        Returns the 2d background model for the entire exposure. 
        """
        return self.background
        # return background.get2d(self,*args)
    
    
    @property
    def envelope(self):
        """
        Returns the 2d background model for the entire exposure. Caches the
        array.
        """
        try:
            return self.__getitem__('background')
        except:
            
            line_positions, env2d, bkg2d = self.get_linepos_env_bkg()
        return env2d
    def get_envelope(self,*args):
        """
        Returns the 2d background model for the entire exposure. 
        """
        return self.envelope
        # return background.getenv2d(self,*args)
    
    
    @property
    def line_positions(self):
        try:
            return self.__getitem__('line_positions')
        except:
            
            line_positions, env2d, bkg2d = self.get_linepos_env_bkg()
        return line_positions
    @property
    def weights(self):
        """
        Returns the photon noise weights for each pixel in the spectrum
        """
        try:
            weights2d = self._cache['weights2d']
        except:
            weights2d = self.get_weights2d()  
            self._cache['weights2d']=weights2d
        return weights2d
    def get_weights1d(self,order):
        """
        Calculates the photon noise of this order.
        """
        sigma_v = self.sigmav1d(order)
        return (sigma_v/c)**-2
    
    def get_weights2d(self):
        """
        Calculates the photon noise of the entire spectrum.
        """
        sigmav2d = self.sigmav2d()
        return (sigmav2d/c)**-2
    def sigmav(self,order=None,unit='mps'):
        """
        Calculates the theoretical limiting velocity precision from the photon 
        noise weights of this order(s).
        """
        orders = self.prepare_orders(order)
        precision_order = [1./np.sqrt(np.sum(self.weights[order])) \
                           for order in orders]
        precision_total = 1./np.sqrt(np.sum(np.power(precision_order,-2)))
        if unit == 'mps':
            fac = 2.99792458e8
        else:
            fac = 1.
        return precision_total * fac
    def sigmav2d(self):
        """
        Calculates the limiting velocity precison of all pixels in the spectrum
        using ThAr wavelengths.
        """
        orders  = np.arange(self.nbo)
        sigma_v = np.array([self.sigmav1d(order) for order in orders])
        return sigma_v

    def sigmav1d(self,order):
        """
        Calculates the limiting velocity precison of all pixels in the order
        using ThAr wavelengths.
        """
        data    = self['flux'][order]
        thar    = self['wavereference'][order]
        err     = self['error'][order]
        # weights for photon noise calculation
        # Equation 5 in Murphy et al. 2007, MNRAS 380 p839
        #pix2d   = np.vstack([np.arange(spec.npix) for o in range(spec.nbo)])
        df_dlbd = hf.derivative1d(data,thar)
        sigma_v = c*err/(thar*df_dlbd)
        return sigma_v
    
    @property
    def wavereference(self):
        """
        Returns the 2d ThAr wavelengths (vacuum) for this exposure. Caches
        the array.
        """
        try:
            wavereferencedisp2d = self._cache['wavereference']
        except:
            wavereferencedisp2d = self.wavereference_object(vacuum=True,
                                                            npix=self.npix)
            self._cache['wavereference'] = wavereferencedisp2d
        return wavereferencedisp2d

    @property
    def wavereference_object(self):
        return self._wavereference
    @wavereference_object.setter
    def wavereference_object(self,waveref_object):
        """ Input is a wavesol.ThAr object or wavesol.ThFP object """
        self._wavereference = waveref_object
    
    
    

    def fit_lines(self,order=None,*args,**kwargs):
        """
        Performs line fitting for the order(s) provided.
        """
        orders = self.prepare_orders(order)
        if len(orders)==1:
            linelist = lines.fit1d(self,orders[0])
            return linelist
        else:
            linedict = lines.fit(self,orders)
            return linedict
        
    def process(self,fittype=['gauss','lsf'],do_comb_specific=True):
        # if isinstance(settings,str):
        #     settings_dict = hs.Settings(settings)
        # elif isinstance(settings,dict):
        #     settings_dict = settings
        # else:
        #     raise Exception(f"{settings} not recognised. Allowed input is "
        #     "a string or a dictionary")
        settings_dict = dict(
            f0=self.lfckeys['comb_anchor'],
            fr=self.lfckeys['comb_reprate'],
            sOrder=self.sOrder,
            eOrder=self.eOrder,
            version=self.version,
            fittype=np.atleast_1d(fittype),
            remove_false_lines=True,
            do_comb_specific=do_comb_specific,
            overwrite=True,
            )
        
        return process(self,settings_dict)
    
    def model_ip(self,order=None,scale='pixel',iter_solve=2,iter_center=5,
                 numseg=16,filter=None,save=False):
        
        orders   = self.prepare_orders(order)
        
    
    def redisperse1d(self,order,velocity_step=0.82,old_wavelengths=None,
                     wavereference=None):
        import harps.functions.spectral as specfunc 
        flx1d = self.flux[order]
        err1d = np.sqrt(np.abs(flx1d))
        wavereference = wavereference if wavereference is not None else 'LFC'
        if old_wavelengths is not None:
            wav1d = old_wavelengths
        else:
            if wavereference=='ThAr':
                wav1d = self.wavereference[order]
            else:
                wav1d = self['wavesol_gauss',701][order]
        
        return specfunc.redisperse1d(wav1d, flx1d, err1d, velocity_step)
    
    def redisperse2d(self,velocity_step=0.82,old_wavelengths=None,
                     wavereference='LFC'):
        import harps.functions.spectral as specfunc 
        if old_wavelengths is not None:
            old_wav2d = old_wavelengths
        else:
            if wavereference=='ThAr':
                old_wav2d = self.wavereference
            else:
                old_wav2d = self['wavesol_gauss',701]
                
        flx2d = self.flux
        err2d = np.sqrt(np.abs(self.flux))
        return  specfunc.redisperse2d(old_wav2d,flx2d,err2d,velocity_step)
        
        # result = np.dstack(np.transpose([self.redisperse1d(od, 
        #                                                    velocity_step,
        #                                                    old_wavs,
        #                                                    wavereference=wavereference) 
        #                                  for od in np.arange(self.sOrder,self.nbo)]
        #                                 )
        #                    )
        # new_wavs,new_flux,new_errs = result
        # return new_wavs,new_flux,new_errs
        
    def order_from_grating_order(self, order_value):
        """
        Finds the index of a given 'order_value' within the mapping array
        determined by self.nbo.

        Args:
            order_value (int): The order value (e.g., an element from the array
                               returned by get_order_mapping_array).

        Returns:
            int: The 0-based index of order_value in the mapping array.

        Raises:
            ValueError: If order_value is not found in the corresponding mapping array
                        or if self.nbo is invalid.
        """
        # First, get the correct mapping array based on self.nbo
        try:
            mapping_array = self.optical_orders
        except ValueError: # Propagate the error if nbo is invalid
            raise

        # Find the index of the order_value
        # np.where returns a tuple of arrays; we need the first element of the first array
        indices = np.where(mapping_array == order_value)[0]

        if len(indices) > 0:
            return int(indices[0]) # Return the first (and should be only) index
        else:
            raise ValueError(f"Order value {order_value} not found in the mapping array for nbo={self.nbo}.")
            
    def plot_spectrum(self,*args,**kwargs):
        '''
        Plots the spectrum. 
        
        Args:
        ----
            order:          integer of list or orders to be plotted
            nobackground:   boolean, subtracts the background
            scale:          'pixel', 'combsol' or 'tharsol'
            model:          boolean, fits the lines and shows the fits
            
        Returns:
        --------
            plotter:    Plotter Class object
        '''
        
        
        if self.filetype=='s1d':
            plotter = self.plot_spectrum_s1d(*args,**kwargs)
        else :
            plotter = self.plot_spectrum_e2ds(*args,**kwargs)
        return plotter
    def plot_spectrum_s1d(self,plotter=None,*args,**kwargs):
        """
        Plots the spectrum if file type is s1d.
        """
        # ----------------------      READ ARGUMENTS     ----------------------
        
        scale   = kwargs.pop('scale','pixel')
        ai      = kwargs.pop('axnum', 0)
        legend  = kwargs.pop('legend',False)
        plotter = plotter if plotter is not None else Figure2(1,1,**kwargs)
        figure  = plotter.figure
        axes    = plotter.axes
        
        item    = kwargs.pop('version',None)
        version = self._item_to_version(item)
        if scale=='pixel':
            x1d    = np.arange(self.npix)
            xlabel = 'Pixel'
        else:
            x1d    = self.comb.dispersion(version)
            xlabel = 'Wavelength [A]'
        y1d = self.data
        axes[ai].errorbar(x1d,y1d,yerr=0,label='Data',capsize=3,capthick=0.3,
                ms=10,elinewidth=0.3,color='C0',zorder=100)  
        axes[ai].set_xlabel(xlabel)
        axes[ai].set_ylabel('Counts')
        m = hf.round_to_closest(np.max(y1d),hs.rexp)
        axes[ai].set_yticks(np.linspace(0,m,3))
        if legend:
            axes[ai].legend()
        figure.show()
    def plot_spectrum_e2ds(self,order=None,ax=None,plot_cens=False,
                           nobackground=False,scale='pixel',model=False,
                           fittype='gauss',legend=True,style='steps',
                           show_background=False,show_envelope=False,
                           color='C0',**kwargs):
        """
        Plots the 1d spectrum if file type is e2ds.
        
        Args:
        -----
            order:          integer or list/array or orders to be plotted, 
                            default None.
            ax:             matplotlib.pyplot.Axes instance. Creates an instance
                            of Figure2 if None (default).
            plot_cens:      plot vertical lines at line centers
            nobackground:   boolean (opt). Subtracts the background, 
                            default False.
            scale:          str (opt). Allowed values 'pixel', 'combsol' and
                            'tharsol', default 'pixel'.
            model:          boolean (opt). Plots the line fits, default false.
            fittype:        str, list of strings (opt). Allowed values are
                            'lsf' and 'gauss' (default is 'gauss').
            legend:         bool (opt). Shows the legend if true, default true.
            style:          plotstyle. Default value = 'steps'.
            kind:           str (opt). Sets the plot command. Allowed values 
                            are 'errorbar', 'line', 'points', default 'errorbar'
            show_background bool(opt). Plots the background if true, default
                            false.
            show_envelope   bool(opt). Plots the background if true, default
                            false.
            color:          Specifies the plotting colour.
        
        """
        # ----------------------      READ ARGUMENTS     ----------------------
        
        nobkg   = nobackground

        shwbkg  = show_background
        shwenv  = show_envelope
        return_plotter = False
        if ax is not None:
            ax  = ax
        else:
            plotter = Figure2(1,1,**kwargs)
            ax      = plotter.add_subplot(0,1,0,1)
            return_plotter = True
        
        fittypes = np.atleast_1d(fittype)
        # print(f'ORDER at input = {order}')
        orders   = self.prepare_orders(order)
        # print(f'ORDERS = {orders}')
        lstyles  = {'gauss':'-','lsf':'--'}
        colors   = {'gauss':'C2','lsf':'C1'}
        numcol   = 1
        # ----------------------        READ DATA        ----------------------
        
        if model==True:
            model2d = {}
            for ft in fittypes:
                model2d[ft] = self['model_{ft}'.format(ft=ft)]
        if plot_cens==True:
            linelist = container.Generic(self['linelist'])
        item    = kwargs.pop('version',None)
        version = self._item_to_version(item)
        assert scale in ['pixel','combsol','wavereference']
        scaleabbv = 'pix' if scale=='pixel' else 'wav'
        if scale=='pixel':
            x2d    = np.vstack([np.arange(self.npix) for i in range(self.nbo)])
            xlabel = 'Pixel'
        elif scale=='combsol':
            x2d    = self['wavesol_{}'.format(fittype),version]/10
            xlabel = r'Wavelength [nm]'
        elif scale=='wavereference':
            x2d    = self.wavereference/10
            xlabel = r'Wavelength [nm]'
        for order in orders:
            x      = x2d[order]
            y      = self.data[order]
            if nobkg:
                bkg = self['background'][order]
                y = y-bkg 
            yerr   = self.error[order]
            if style=='errorbar':
                ax.errorbar(x,y,yerr=yerr,label='Flux',capsize=3,
                    capthick=0.3,ms=10,elinewidth=0.3,zorder=100,
                    rasterized=True)
            elif style=='points':
                ax.plot(x,y,label='Flux',ls='',marker='o',
                    ms=10,zorder=100,rasterized=True)
                
            else:
                ax.plot(x,y,label='Flux',ls='-',zorder=100,
                        drawstyle='steps-mid',rasterized=True)
            if model==True:   
                for i,ft in enumerate(fittypes):
                    model1d = model2d[ft][order]
                    ax.plot(x,model1d,c=colors[ft],drawstyle='steps-mid',
                                 label='Model {}'.format(ft),ls='--')
            if shwbkg==True:
                bkg1d = self['background'][order]
                ax.plot(x,bkg1d,label='Background',#drawstyle='steps-mid',
                        ls='--',color='C1',
                        zorder=100,rasterized=True)
                numcol+=1
            if shwenv==True:
                bkg1d = self['envelope'][order]
                ax.plot(x,bkg1d,label='Envelope',#drawstyle='steps-mid',
                        ls='-.',color='C2',
                        zorder=100,rasterized=True)
                numcol+=1
            if plot_cens==True:
                linelist1d = linelist[order]
                for i,ft in enumerate(fittypes):
                    centers = linelist1d.values[f'{ft}_{scaleabbv}'][:,1]
                    if scale!='pixel':
                        centers /= 10.
                    # print('centers',centers)
                    ax.vlines(centers,-0.25*np.mean(y),-0.05*np.mean(y),
                              linestyles=lstyles[ft],
                              colors=colors[ft])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Flux [counts]')
        m = hf.round_to_closest(np.log10(np.max(y)),1)-1
#        ax.set_yticks(np.linspace(0,m,3))
        if legend:
            handles,labels = ax.get_legend_handles_labels()
            ax.legend(handles[:numcol],labels[:numcol],ncol=numcol)
        if scale=='pixel':
            ticks(ax,which='major',axis='x',tick_every=1024)
        else:
            ticks(ax,which='major',axis='x',ticknum=3)
        ticks(ax,which='major',axis='y',ticknum=5)
        scinotate(ax,'y',m,dec=0)
        if not return_plotter:
            return ax
        else:
            return ax,plotter
    def plot_2d(self,order=None,ax=None,*args,**kwargs):
        '''
        Plots the spectrum in 2d.
        
        Args:
        ----
            order   : int, list or tuple . Tuple must be in slice format.
            ax      : matplotlib.pyplot.Axes instance. New if None (default)
            kwargs  : arguments passed on to Figure 
            
        Returns:
        -------
            ax      : matplotlib.pyplot.Axes or harps.plotter.Figure 2 instance
        '''
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        ai      = kwargs.pop('axnum', 0)
        cmap    = kwargs.get('cmap','inferno')
        return_plotter = False
        if ax is not None:
            ax  = ax
        else:
            plotter = Figure2(1,1,**kwargs)
            ax      = plotter.add_subplot(0,1,0,1)
            return_plotter = True
        
        data    = self.data[orders]
        # optord  = self.optical_orders[orders]
        
        vmin,vmax = np.percentile(data,[0.05,99.5])
        
        im = ax.imshow(data,aspect='auto',origin='lower',
                 vmin=vmin,vmax=vmax,
                 extent=(0,self.npix,orders[0],orders[-1]))
        cb = plotter.figure.colorbar(im,cmap=cmap)
        plotter.ticks(ai,'x',5,0,self.npix)
        if not return_plotter:
            return ax
        else:
            return ax,plotter
      
    def plot_b2e(self,order=None,ax=None,plot2d=False,scale='tharsol',
                 fittype='gauss',version=None,vmin=0,vmax=0.1,*args,**kwargs):
        orders  = hf.wrap_order(order,0,self.nbo)
        
        return_plotter = False
        if ax is not None:
            ax  = ax
        else:
            plotter = Figure2(1,1,**kwargs)
            ax      = plotter.add_subplot(0,1,0,1)
            return_plotter = True
        
        background = self.background
        envelope   = self.envelope
        b2e        = background / envelope
            
        assert scale in ['pixel','combsol','tharsol']
        if scale=='pixel':
            x2d    = np.vstack([np.arange(self.npix) for i in range(self.nbo)])
            xlabel = 'Pixel'
        elif scale=='combsol':
            x2d    = self['wavesol_{}'.format(fittype),version]/10
            xlabel = r'Wavelength [nm]'
        elif scale=='tharsol':
            x2d    = self.wavereference/10
            xlabel = r'Wavelength [nm]'
            
            
        if plot2d:
            im = ax.imshow(b2e,aspect='auto',vmin=vmin,vmax=vmax)
            plt.colorbar(im)
        else:
            # print(kwargs.keys())
            if 'color' not in kwargs.keys():
                colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
            else:
                color = kwargs.pop('color')
                colors = [color for i in range(len(orders))]
            # print(colors)
            for i,order in enumerate(orders):
                ax.plot(x2d[order],b2e[order]*100,drawstyle='steps-mid',
                        color=colors[i],**kwargs)
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Background / Envelope [%]')
        
        if not return_plotter:
            return ax
        else:
            return ax,plotter
    def plot_ccd_from_linelist(self,desc,fittype='gauss',scale='pix',mean=False,
                               column=None,*args,**kwargs):
        
        linelist = self['linelist']
        return ccd_from_linelist(linelist,desc,fittype,scale,
                                 mean=mean,column=column,*args,**kwargs)
        
    def plot_flux_per_order(self,order=None,ax=None,optical=False,scale=None,
                            yscale='linear',
                            *args,**kwargs):
        '''
        Plots the cumulative number of counts per echelle order. 
        
        Args:
        ----
            order   : int, list or tuple . Tuple must be in slice format.
            ax      : matplotlib.pyplot.Axes instance. New if None (default)
            optical : bool (default False). Arranges order in echelle numbering
            kwargs  : arguments passed on to Figure and ax.plot
            
        Returns:
        -------
            ax      : matplotlib.pyplot.Axes or harps.plotter.Figure 2 instance
        '''
        orders  = hf.wrap_order(order,0,self.nbo)
        return_plotter = False
        if ax is not None:
            ax  = ax
        else:
            plotter = Figure2(1,1,**kwargs)
            ax      = plotter.add_subplot(0,1,0,1)
            return_plotter = True
        data   = self.data[orders].sum(axis=1)
        pltord = orders
        xlabel = 'Order'
        
        if optical==True:
            ordbreak = 115 if self.meta['fibre']=='A' else 114
            optord = self.optical_orders[orders]
            sortind= np.argsort(optord)
            limit0 = np.searchsorted(optord,ordbreak,sorter=sortind)
            limit1 = sortind[limit0]
            pltord = np.insert(optord,limit1,ordbreak)
            data   = np.insert(data,limit1,np.nan)
        if scale=='wave':
            pltord = np.mean(self.wavereference,axis=1)/10.
            xlabel = 'Wavelength [nm]'
        
        ax.plot(pltord,data,drawstyle='steps-mid',ls='-',**kwargs)
        

        ylabel = 'Total flux [counts]'
        if optical:
            xlabel = 'Echelle order'
        ax.set_yscale(yscale)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if not return_plotter:
            return ax
        else:
            return ax,plotter
        
    def plot_distortions(self,order=None,kind='lines',xscale='pixel',
                         yscale='velocity',ax=None,**kwargs):
        '''
        Plots the distortions in the CCD in two varieties:
        kind = 'lines' plots the difference between LFC theoretical wavelengths
        and the value inferred from the ThAr wavelength solution. 
        kind = 'wavesol' plots the difference between the LFC and the ThAr
        wavelength solutions.
        
        Uses ThAr coefficients in air and converts the calculated wavelengths 
        to vacuum.
        
        Args:
        ----
            order:      integer of list or orders to be plotted
            kind:       'lines' or 'wavesol'
            xscale:     'pixel' or 'wave'
            yscale:     'velocity' (default) or 'angstrom'
            plotter:    Figure class object from harps.plotter (opt), 
                        default None.
        Returns:
        --------
            plotter:    Figure class object
        '''
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        fittype = kwargs.pop('fittype','gauss')
#        marker  = kwargs.get('marker','x')
#        color   = kwargs.pop('color',None)
        anchor  = kwargs.pop('anchor_offset',0e0)
        return_plotter = False
        if ax is not None:
            ax  = ax
        else:
            plotter = Figure2(1,1,left=0.15,bottom=0.12,**kwargs)
            ax      = plotter.add_subplot(0,1,0,1)
            return_plotter = True
        # ----------------------      PLOT SETTINGS      ----------------------    
        usecmap    = kwargs.pop('cmap','jet')
        cmap       = plt.cm.get_cmap(usecmap)
        ncolors    = kwargs.pop('ncols',10)
        norders    = len(orders)
        n          = int(np.ceil(norders / ncolors)) 
        colors     = np.vstack([cmap(np.linspace(0, 1, ncolors)) \
                                     for i in range(n)])
        marker     = kwargs.pop('marker','o')
        markersize = kwargs.pop('markersize',16)
        alpha      = kwargs.pop('alpha',1.)
        color      = kwargs.pop('color',None)
        
        # plotargs = {'s':markersize,'marker':marker,'alpha':alpha,'cmap':cmap}
        # ----------------------        PLOT DATA        ----------------------
        
        
        if kind == 'lines':
            plotargs = {'s':markersize,'marker':marker,'alpha':alpha,'cmap':cmap}
            print("Anchor offset applied {0:+12.3f} MHz".format(anchor/1e6))
            data  = self['linelist']
            wave  = hf.freq_to_lambda(data['freq']+anchor)
            cens  = data[f'{fittype}_{xscale[:3]}'][:,1]
            x = cens
            if xscale == 'wave':
                x = wave
            #if not vacuum:
            tharObj  = self.wavereference_object
            coeff = tharObj.get_coeffs(vacuum=False)
            
            Dist = np.array([])
            Vel  = np.array([])
            
            print("LFC-ThAr mean shift")
            print('color = ',color)
            for i,order in enumerate(orders):
                if len(orders)>5:
                    plotargs['color']=color if color is not None else colors[i]
#                    plotargs['color']=colors[i]
                elif color is not None:
                    plotargs['color']=color
                else:
                    pass
                cut  = np.where(data['order']==order)[0]
                pars = coeff[order]['pars']
                if len(np.shape(pars))>1:
                    pars = pars[0]
                thar_air = np.polyval(np.flip(pars),cens[cut])
                thar_vac = ws._to_vacuum(thar_air)
#                print(order,thar,wave[cut])
                dist  = wave[cut]-thar_vac
                vel   = dist/wave[cut] * c
                
                Dist  = np.hstack([Dist,dist])
                Vel   = np.hstack([Vel,vel])
                
                mdist = np.mean(dist)*1e3; mvel = np.mean(vel)
                rdist = hf.rms(dist,True)*1e3; rvel = hf.rms(vel,True)
                print("OD = {}".format(order),
                      "Mean: {0:+5.3f} mA ({1:+5.1f} m/s)".format(mdist,mvel),
                      "RMS:  {0:5.3f} mA ({1:5.1f} m/s)".format(rdist,rvel))
                y = vel
                if yscale=='angstrom':
                    y = dist*1e3
                ax.scatter(x[cut],y,**plotargs)
            # for all considered orders:
            mDist = np.mean(Dist)*1e3; mVel = np.mean(Vel)
            rDist = hf.rms(Dist,True)*1e3; rVel = hf.rms(Vel,True)
            print("ALL CONSIDERED ORDERS")
            print("Mean: {0:+5.3f} mA ({1:+5.1f} m/s)".format(mDist,mVel),
                  "RMS:  {0:5.3f} mA ({1:5.1f} m/s)".format(rDist,rVel))
        elif kind == 'wavesol':
            plotargs = {'ls':'--','alpha':alpha}
            # plotargs['ls']='-'
            # plotargs['ms']=0
                
            version = kwargs.pop('version',self._item_to_version(None))
            wave = self['wavesol_{}'.format(fittype),version]
            x    = np.tile(np.arange(self.npix),self.nbo).reshape(self.nbo,-1)
            if xscale=='wave': x = wave
            
            reference = self.wavereference
            # plotargs['ls']='--'
            for i,order in enumerate(orders):
                plotargs['color']=colors[i]
                dist  = wave[order]-reference[order]
                vel   = dist/wave[order] * c
                y = vel
                if yscale=='angstrom':
                    y = dist*1e3
                print('all good to here')
                print(np.shape(x))
                print(np.shape(y))
                ax.plot(x[order],y,**plotargs)
        ax.axhline(0,ls=':',c='k')
        if xscale=='pixel':
            [ax.axvline(512*(i+1),lw=0.3,ls='--') for i in range (8)]
            ax.set_xlabel('Pixel')
        if xscale=='wave':
            ax.set_xlabel('Wavelength '+r'[${\rm \AA}$]')
        yunit = r'ms$^{-1}$'
        if yscale=='angstrom':
            yunit=r'm${\rm \AA}$'
        ax.set_ylabel('ThAr - LFC  ' + r'$\frac{\Delta \lambda}{\lambda}$'+\
                      '({0})'.format(yunit))
        
#        ax.set_title("Fittype = {0}, Anchor offset = {1:9.1f} MHz".format(fittype,anchor/1e6))
        if not return_plotter:
            return ax
        else:
            return ax,plotter
    def plot_line(self,order,lineid,fittype='gauss',center=True,residuals=False,
                  plotter=None,axnum=None,title=None,figsize=(12,12),show=True,
                  error_blowup=1, **kwargs):
        ''' Plots the selected line and the models with corresponding residuals
        (optional).'''
        naxes = 1 if residuals is False else 2
        left  = 0.1 if residuals is False else 0.15
        ratios = None if residuals is False else [4,1]
        if plotter is None:
            plotter = Figure2(nrows=naxes,ncols=1,title=title,left=left,
                              bottom=0.12,height_ratios=ratios,**kwargs)
            
        else:
            pass
        ai = axnum if axnum is not None else 0
        ax0 = plotter.add_subplot(0,1,0,1)
        if naxes>1:
            ax1 = plotter.add_subplot(1,2,0,1,sharex=ax0)
        figure, axes = plotter.figure, plotter.axes
        # handles and labels
        labels  = []
        
        # Load line data
        linelist  = self['linelist']
        line      = linelist[np.where((linelist['order']==order) & \
                                      (linelist['index']==lineid))]
        pixl      = line['pixl'][0]
        pixr      = line['pixr'][0]
        pix       = np.arange(pixl,pixr)
        
        flux      = self['flux'][order,pixl:pixr]
        error     = self['error'][order,pixl:pixr]
        
        
        # save residuals for later use in setting limits on y axis if needed
        if residuals:
            resids = []
        # Plot measured line
        axes[ai].errorbar(pix,flux,yerr=error*error_blowup,ls='',color='k',
            marker='o',markerfacecolor='None',label='Flux',zorder=0,
            elinewidth=2,markeredgewidth=2)
        
#        axes[ai].bar(pix,flux,width=1,align='center',edgecolor='k',fill=False)
        axes[ai].step(pix,flux,where='mid',color='k')
        axes[ai].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        # Plot models of the line
        if type(fittype)==list:
            pass
        elif type(fittype)==str and fittype in ['epsf','gauss']:
            fittype = [fittype]
        else:
            fittype = ['epsf','gauss']
        print((4*("{:^12}")).format("Fittype","A","mu","sigma"))
        for j,ft in enumerate(np.atleast_1d(fittype)):
            if ft == 'lsf':
                label = 'eLSF'
                c   = kwargs.pop('c','C3')
                m   = 's'
                ls  = '--'
            elif ft == 'gauss':
                label = 'Gauss'
                c   = kwargs.pop('c','C2')
                m   = '^'
                ls  = ':'
            pars = np.ravel(line[ft])
            errs = np.ravel(line['{}_err'.format(ft)])
            print("{:<12}".format(ft+' pars'),
                    (len(pars)*("{:>12.5f}")).format(*pars))
            print("{:<12}".format(ft+' errs'),
                    (len(pars)*("{:>12.5f}")).format(*errs))
            labels.append(label)
            model     = self['model_{ft}'.format(ft=ft)][order,pixl:pixr]
            
            
            axes[ai].plot(pix,model,ls=ls,color="None",marker=m,
                markeredgewidth=2,label=ft,markeredgecolor=c)
            if residuals:
                rsd        = (flux-model)/error
                resids.append(rsd)
                axes[ai+1].plot(pix,rsd,color='None',ls='',marker=m,
                    markeredgecolor=c,markeredgewidth=2)
        # Plot centers
            if center:
                
                cen = line[ft][0][1]
                cenerr = line['{}_err'.format(ft)][0][1]
                axes[ai].axvline(cen,ls='--',c=c)
#                axes[ai].axvspan(cen-cenerr,cen+cenerr,color=c,alpha=0.3)
        # Makes plot beautiful
        labels.append('Data')
        axes[ai].set_ylabel('Flux [counts]')
        rexp = hs.rexp
        m   = hf.round_to_closest(np.max(flux),rexp)
#        axes[ai].set_yticks(np.linspace(0,m,3))
        plotter.ticks(ai,'y',3,0,m)
        # Handles and labels
        handles, oldlabels = axes[ai].get_legend_handles_labels()
        axes[ai].legend(handles,labels)
        if residuals:
            axes[ai+1].axhline(0,ls='--',lw=1,c='k')
            axes[ai+1].set_ylabel('Residuals\n[$\sigma$]')
            # make ylims symmetric
            lim = 1.2*np.nanpercentile(np.abs(resids),100)
            lim = np.max([5,lim])
            axes[ai+1].set_ylim(-lim,lim)
            # makes x-axis tick labels invisible in the top panel
            [label.set_visible(False) for label in axes[ai].get_xticklabels()]
            axes[ai+1].set_xlabel('Pixel')
            # makes x-axis ticks more sparse
            nticks  = 3
            div,mod = divmod(len(pix),nticks)
            xmin    = np.min(pix)
            xmax    = np.max(pix)+mod
            #plotter.ticks(ai+1,'x',nticks,xmin,xmax)
            # mark 5sigma limits
            axes[ai+1].axhspan(-5,5,alpha=0.3,color='k')
        else:
            axes[ai].set_xlabel('Pixel')
            
        plotter.figure.align_ylabels()
        
        if show == True: figure.show()
        return plotter
    def plot_lfresiduals(self,order=None,hist=False,ax=None,xscale='pixel',
                               fittype='gauss',normed=False,
                               **kwargs):
        ''' Plots the residuals of the line fits as either a function of 
            position on the CCD or a produces a histogram of values'''
        fittypes = np.atleast_1d(fittype)
        
        if hist == False:
            figsize = (12,9)
        else: 
            figsize = (9,9)
            
        if ax is None:
            plotter=Figure2(len(fittypes),1,figsize=figsize,
                            bottom=0.12,left=0.15,**kwargs)
            axes = [plotter.add_subplot(i,i+1,0,1) \
                    for i in range(len(fittypes))]
        else:
            axes = [ax]
        
        figure,axes = plotter.figure, plotter.axes
        
        orders = self.prepare_orders(order)
        plot2d = True if len(orders)>1 else False
        data   = self.data
        if len(orders)>5:
            colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        else:
            colors = ['C0','C1','C2','C3','C4']
        markers = {'gauss':'o','lsf':'x'}
        
        for j,ft in enumerate(np.atleast_1d(fittype)):
            if len(axes)>1:
                ax = axes[j]
            else:
                ax = axes[0]
            ax.text(0.9,0.8,ft,transform=ax.transAxes)
            model  = self['model_{ft}'.format(ft=ft)]
            resids = (data - model)[orders]
            if normed: resids = resids/self['error'][orders]
            if hist == True:
                bins = kwargs.get('bins',30)
                xrange = kwargs.get('range',None)
                log  = kwargs.get('log',False)
                label = kwargs.get('label',ft)
                alpha = kwargs.get('alpha',1.)
                fitresids1d = np.ravel(resids)
                ax.hist(fitresids1d,bins=bins,range=xrange,log=log,
                    label=label,alpha=alpha,histtype='step')
                ax.set_ylabel('Number of lines')
                ax.set_xlabel('Residuals [$e^-$]')
            else:
                if plot2d:
                    from matplotlib.colors import Normalize
                    sig       = np.std(resids)
                    normalize = Normalize(-sig,sig,False)
                    
                    img = ax.imshow(resids,aspect='auto',norm=normalize,
                            extent=[0,self.npix,self.nbo,self.sOrder])
                    cbar      = plt.colorbar(img)
                    cbar.set_label('Residuals [$e^-$]')
                    ax.set_ylabel('Order')
                    ax.set_xlabel('Pixel')
                else:
                    x = np.arange(self.npix)
                    if xscale=='wave':
                        wave = self.tharsol
                    for i,order in enumerate(orders):
                        if xscale=='wave': 
                            x = wave[order]
                        ax.scatter(x,resids[i],marker=markers[ft],
                                   s=4,color=colors[i])
                    ax.set_xlabel('Pixel')
                    ax.set_ylabel('Residuals [$e^-$]')
        return plotter
    def plot_wsresiduals(self,order=None,fittype='gauss',version=None,ax=None,
                       xscale='pixel',normalised=False,colorbar=False,
                       unit='nm',**kwargs):
        '''
        Plots the residuals of LFC lines to the wavelength solution. 
        
        Args:
        ----
            order:      integer or list/array or orders to be plotted, default 
                        None
            fittype:    str or list of str (opt). Sets the fit model, default
                        'gauss'
            plotter:    Figure class object from harps.plotter (opt), 
                        default None.
        Returns:
        --------
            plotter:    Figure class object
        '''
        # ----------------------      READ ARGUMENTS     ----------------------
        
        version = hv.item_to_version(version,'wavesol')
        return_plotter = False
        if ax is not None:
            ax  = ax
        else:
            plotter = Figure2(1,1,**kwargs)
            ax      = plotter.add_subplot(0,1,0,1)
            return_plotter = True
        # ----------------------        READ DATA        ----------------------
        linelist  = self['linelist']
        if order is not None:
            orders    = np.atleast_1d(order)
        else:
            orders = np.unique(linelist['order'])
        centers2d = linelist[f'{fittype}_pix'][:,1]
        if xscale == 'wave':
            centers2d = hf.freq_to_lambda(linelist['freq'])
            if unit=='nm':
                centers2d = centers2d/10.
        
        
        noise     = linelist['noise']
        errors2d  = linelist[f'{fittype}_pix_err'][:,1]
        coeffs    = ws.get_wavecoeff_comb(linelist,
                                          version=version,
                                          fittype=fittype,
                                          npix=self.npix)
        residua2d = ws.residuals(linelist,coeffs,version=version,
                                 fittype=fittype,npix=self.npix)
        
        # ----------------------      PLOT SETTINGS      ----------------------
        usecmap    = kwargs.pop('cmap','jet')
        cmap       = plt.cm.get_cmap(usecmap)
        ncolors    = kwargs.pop('ncols',10)
        norders    = len(orders)
        n          = int(np.ceil(norders / ncolors)) 
        colors     = np.vstack([cmap(np.linspace(0, 1, ncolors)) \
                                     for i in range(n)])
        marker     = kwargs.pop('marker','o')
        markersize = kwargs.pop('markersize',16)
        alpha      = kwargs.pop('alpha',1.)
        color      = kwargs.pop('color',None)
        
        plotargs = {'s':markersize,'marker':marker,'alpha':alpha}
        # ----------------------       PLOT DATA         ----------------------
        for i,order in enumerate(orders):
            
            cutcen  = np.where(linelist['order']==order)[0]
            cent1d  = centers2d[cutcen]
            error1d  = errors2d[cutcen]
#            cutres = np.where(residua2d['order']==order)[0]
            resi1d = residua2d['residual_mps'][cutcen]

            if normalised:
                resi1d = resi1d/(error1d)
            if len(orders)>5:
                plotargs['color']=color if color is not None else colors[i]
            else:
                pass
            ax.scatter(cent1d,resi1d,**plotargs)
        if normalised:
            ax.set_ylabel(r'Residuals [$\sigma$]')
        else:
            ax.set_ylabel('Residuals'+r' [ms$^{-1}$]')    
        # 512 pix vertical lines
        if xscale=='wave':
            ax.set_xlabel('Wavelength [{}]'.format(unit))
        else:
            # [ax.axvline(512*(i),lw=0.3,ls='--') for i in range (9)]
            ax.set_xlabel('Pixel')

       
        if not return_plotter:
            return ax
        else:
            return ax,plotter
    def plot_wsresiduals_chisq(self,order=None,fittype='gauss',version=None,
                       normalised=False,colorbar=False,**kwargs):
        '''
        Plots the residuals of LFC lines to the wavelength solution. 
        
        Args:
        ----
            order:      integer or list/array or orders to be plotted, default 
                        None
            fittype:    str or list of str (opt). Sets the fit model, default
                        'gauss'
            plotter:    Figure class object from harps.plotter (opt), 
                        default None.
        Returns:
        --------
            plotter:    Figure class object
        '''
        # ----------------------      READ ARGUMENTS     ----------------------
        
        
        version = hf.item_to_version(version)
        phtnois = kwargs.pop('photon_noise',False)
#        ai      = kwargs.pop('axnum', 0)
        mean    = kwargs.pop('mean',False)
        
        naxes   = 2 if colorbar == False else 1
        plotter = Figure2(naxes,1,**kwargs)
        ax0     = plotter.add_subplot(0,1,0,1)
        if naxes >1:
            ax1 = plotter.add_subplot(1,2,0,1,sharex=ax0)
        figure  = plotter.figure
        axes    = plotter.axes
        # ----------------------        READ DATA        ----------------------
        linelist  = self['linelist']
        if order is not None:
            orders    = np.atleast_1d(order)
        else:
            orders = np.unique(linelist['order'])
        centers2d = linelist[fittype][:,1]
        
        noise     = linelist['noise']
        errors2d  = linelist['{}_err'.format(fittype)][:,1]
        coeffs    = ws.get_wavecoeff_comb(linelist,version,fittype)
        residua2d = ws.residuals(linelist,coeffs,version,fittype)
        
        # ----------------------      PLOT SETTINGS      ----------------------
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        marker     = kwargs.pop('marker','o')
        markersize = kwargs.pop('markersize',16)
        alpha      = kwargs.pop('alpha',1.)
        color      = kwargs.pop('color',None)
        cmap       = kwargs.pop('cmap','viridis')
        plotargs = {'s':markersize,'marker':marker,'alpha':alpha,'cmap':cmap}
        # ----------------------       PLOT DATA         ----------------------
        for i,order in enumerate(orders):
            
            cutcen  = np.where(linelist['order']==order)[0]
            cent1d  = centers2d[cutcen]
            error1d  = errors2d[cutcen]
            chisq1d = linelist[cutcen]['{}chisq'.format(fittype[0])]
            
#            cutres = np.where(residua2d['order']==order)[0]
            resi1d = residua2d['residual_mps'][cutcen]
            if normalised:
                resi1d = resi1d/(error1d*829)
            if len(orders)>5:
                if not colorbar:
                    plotargs['color']=color if color is not None else colors[i]
                
            if not phtnois:
                if not colorbar:
                    axes[0].scatter(cent1d,resi1d,**plotargs)
                    axes[1].scatter(cent1d,chisq1d,**plotargs)
                else:
#                    color_remove = plotargs.pop('color',None)
                    sc = axes[0].scatter(cent1d,resi1d,c=chisq1d,**plotargs)
                    figure.colorbar(sc,ax=axes[0],
                                    label=r'Line fit $\chi_\nu^2$')
#                axes[0].plot(cent1d,resi1d,**plotargs)
            else:
                pn = noise[cutcen]
                axes[0].errorbar(cent1d,y=resi1d,yerr=pn,
                                    ls='--',lw=0.3,**plotargs)
            if mean==True:
                meanplotargs={'lw':0.8}
                w  = kwargs.get('window',5)                
                rm = hf.running_mean(resi1d,w)
                if len(orders)>5:
                    meanplotargs['color']=colors[i]
                axes[0].plot(cent1d,rm,**meanplotargs)
            chisq_val = coeffs[np.where(coeffs['order']==order)]['chisq'].T
#            cellText  = np.array([1,['{0:5.3f}'.format(val) for val in chisq_val]])
            cols      = ["{0:1d}".format(i) for i in range(1,9,1)]
            rows      = [r'$\chi_\nu^2$']
            print("Chi^2 (poly):", chisq_val)
#            chisqT  = axes[0].table(cellText=chisq_val,
#                          rowLabels=rows,
#                          colLabels=cols)
            
        # 512 pix vertical lines
        for ax in axes:
            [ax.axvline(512*(i),lw=0.3,ls='--') for i in range (9)]
        if normalised:
            [axes[0].axhline(i,lw=1,ls='--',c='k') for i in [-1,1]]
        
        axes[-1].set_xlabel('Pixel')
        if not colorbar:
            axes[1].set_ylabel(r'Line fit $\chi_\nu^2$')
        if normalised:
            axes[0].set_ylabel('Residuals [$\sigma$]')
        else:
            axes[0].set_ylabel('Residuals [m/s]')
        axes[0].set_title("Version PGS = {v:3d}; "
                          "order = {o}; "
                          "fit = {f}".format(v=version,o=order,f=fittype))
        #plotter.ticks(0,'x',9,0,4096)
        return plotter
    def plot_histogram(self,kind,order=None,separate=False,fittype='epsf',
                       show=True,plotter=None,axnum=None,**kwargs):
        '''
        Plots a histogram of residuals of LFC lines to the wavelength solution 
        (kind = 'residuals') or a histogram of R2 goodness-of-fit estimators 
        (kind = 'R2').
        
        Args:
        ----
            kind:       'residuals' or 'chisq'
            order:      integer or list/array of orders to be plotted
            plotter:    Figure class object from harps.plotter (opt), 
                        default None.
            show:       boolean
        Returns:
        --------
            plotter:    Figure class object
        '''
        if kind not in ['residual_mps','gchisq']:
            raise ValueError('No histogram type specified \n \
                              Valid options: \n \
                              \t residual \n \
                              \t R2')
        else:
            pass
        
        histrange = kwargs.pop('range',None)
        normed    = kwargs.pop('normed',False)
        orders = self.prepare_orders(order)
            
        N = len(orders)
        if plotter is None:
            if separate == True:
                plotter = Figure2(1,1,**kwargs)
            elif separate == False:
                plotter = Figure2(1,1,**kwargs)
        else:
            pass
        # axis index if a plotter was passed
        ai = axnum if axnum is not None else 0
        ax = plotter.ax()
        figure, axes = plotter.figure, plotter.axes
        
        # plot residuals or chisq
        if kind == 'residual_mps':
            data = self['residuals_gauss']
        elif kind == 'gchisq':
            data = self['linelist']
        bins    = kwargs.get('bins',10)
        alpha   = kwargs.get('alpha',1.0)
        if separate == True:
            for i,order in enumerate(orders):
                cut = np.where(data['order']==order)
                sel = data[kind][cut]
                axes[i].hist(sel,bins=bins,normed=normed,range=histrange,
                             alpha=alpha)
                if kind == 'residual':
                    mean = np.mean(sel)
                    std  = np.std(sel)
                    A    = 1./np.sqrt(2*np.pi*std**2)
                    x    = np.linspace(np.min(sel),np.max(sel),100)
                    y    = A*np.exp(-0.5*((x-mean)/std)**2)
                    axes[i].plot(x,y,color='#ff7f0e')
                    axes[i].plot([mean,mean],[0,A],color='#ff7f0e',ls='--')
                    axes[i].text(0.8, 0.95,r"$\mu={0:8.3e}$".format(mean), 
                                horizontalalignment='center',
                                verticalalignment='center',transform=axes[i].transAxes)
                    axes[i].text(0.8, 0.9,r"$\sigma={0:8.3f}$".format(std), 
                                horizontalalignment='center',
                                verticalalignment='center',transform=axes[i].transAxes)
        elif separate == False:
            sel = data[kind]
            if histrange is not None:
                cut = np.where((sel>=histrange[0])&(sel<=histrange[1]))[0]
                sel = sel[cut]
            counts, bins, _ = axes[0].hist(sel,bins=bins,density=normed,
                                           range=histrange,alpha=alpha,lw=2)
            if kind == 'residual_mps':
                mean = np.mean(sel)
                std  = np.std(sel)
                A    = 1./np.sqrt(2*np.pi*std**2)
                if not normed:
                    density_norm = (sum(counts) * np.diff(bins)[0])
                    A = len(sel) / density_norm 
                x    = np.linspace(np.min(sel),np.max(sel),100)
                y    = A*np.exp(-0.5*((x-mean)/std)**2)
                axes[ai].plot(x,y,color='C1')
                axes[ai].plot([mean,mean],[0,A],color='C1',ls='--')
                axes[ai].text(0.8, 0.95,r"$\mu={0:8.3e}$".format(mean), 
                            horizontalalignment='center',
                            verticalalignment='center',transform=axes[0].transAxes)
                axes[ai].text(0.8, 0.9,r"$\sigma={0:8.3f}$".format(std), 
                            horizontalalignment='center',
                            verticalalignment='center',transform=axes[0].transAxes)
            axes[ai].set_xlabel("{}".format(kind))
            axes[ai].set_ylabel('Number of lines')
        figure.show() 
        return plotter
    
    def plot_shift(self,order=None,p1='lsf',p2='gauss',lsf=None,lsf_method=None,
                   ax=None,show=True,**kwargs):
        ''' 
        Plots the shift between the selected estimators of the line centers.
            
        Args:
        -----
            order   : int or list/array of integers, default None.
            p1      : primary estimator, default 'lsf'.
            p2      : secondary estimator, default 'gauss'.
            plotter : Figure class object from harps.plotter (opt), 
                        default None.
        '''
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = np.atleast_1d(order) if order is not None else self.prepare_orders(order)
        ai      = kwargs.pop('axnum', 0)
        return_plotter = False
        if ax is not None:
            ax  = ax
        else:
            plotter = Figure2(1,1,**kwargs)
            ax      = plotter.add_subplot(0,1,0,1)
            return_plotter = True
        
        #
        if lsf is not None:
            linelist0 = lines.detect(self,order,fittype=['gauss','lsf'],
                                     lsf=lsf,lsf_method=lsf_method)
            linelist = lines.Linelist(linelist0)
        else:
            linelist  = lines.Linelist(self['linelist'])
        
        def get_center_estimator(linelist1d,p):
            if p == 'lsf':
                cen = linelist1d.values['lsf'][:,1]
                label = 'cen_{lsf}'
            elif p == 'gauss':
                cen = linelist1d.values['gauss'][:,1]
                label = 'cen_{gauss}'
            elif p == 'bary':
                cen = linelist1d.values['bary']
                label = 'b'
            return cen, label
        colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        for i,order in enumerate(orders):
            linelist1d   = linelist[order]
            cen1,label1  = get_center_estimator(linelist1d,p1)
            cen2,label2  = get_center_estimator(linelist1d,p2)
            bary,labelb  = get_center_estimator(linelist1d,'bary')
            delta = cen1 - cen2 
            
            shift = delta * 829
            
            ax.scatter(bary,shift,marker='o',s=2,c=[colors[i]],
                    label="${0} - {1}$".format(label1,label2),rasterized=True)
        ax.set_ylabel('Velocity shift '+r'[${\rm ms^{-1}}$]')
        ax.set_xlabel('Line barycenter [pix]')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1024))
        #axes[ai].legend()
        if return_plotter:
            return ax,plotter
        else:
            return ax
    
    def plot_wavesolution(self,order=None,calibrator='comb',
                          fittype=['gauss','lsf'],version=None,ax=None,
                          **kwargs):
        '''
        Plots the wavelength solution of the spectrum for the provided orders.
        '''
        
        # ----------------------      READ ARGUMENTS     ----------------------
        orders  = self.prepare_orders(order)
        ai      = kwargs.pop('axnum', 0)
        if ax is None:
            plotter = Figure2(1,1,**kwargs)
            ax = plotter.ax()
            axes = plotter.axes
        else:
            ax = ax
        # ----------------------        READ DATA        ----------------------
        
        
        fittype = np.atleast_1d(fittype)
        # Check and retrieve the wavelength calibration
        
        linelist = self['linelist']
        frequencies = linelist['freq'] 
        wavelengths = hf.freq_to_lambda(frequencies)
        # Manage colors
        #cmap   = plt.get_cmap('viridis')
        if len(orders)>10:
            colors = plt.cm.jet(np.linspace(0, 1, len(orders)))
        else:
            colors = ["C{:1d}".format(i) for i in np.arange(10)]
        marker = kwargs.get('marker','x')
        ms     = kwargs.get('markersize',5)
        ls     = {'lsf':'--','gauss':'-'}
        lw     = kwargs.get('lw',2)
        # Plot the line through the points?
        plotline = kwargs.get('plot_line',True)
        # Select line data    
        for ft in fittype:
            print(ft)
            if plotline == True:
                
                if calibrator == 'comb':
                    wavesol = self['wavesol_{}'.format(ft),version]
                else:
                    wavesol = self.tharsol
            centers = linelist[f'{ft}_pix'][:,1]
            # Do plotting
            for i,order in enumerate(orders):
                cut = np.where(linelist['order']==order)[0]
                pix = centers[cut]
                wav = wavelengths[cut]
                print("Number of lines = {0:8d} (order {1:2d})".format(len(cut),order))
                axes[ai].plot(pix,wav,color=colors[i],
                    ls='',ms=ms,marker=marker)
                if plotline == True:
                    axes[ai].plot(wavesol[order],color=colors[i],ls=ls[ft],
                        lw=lw)
        axes[ai].set_xlabel('Pixel')
        axes[ai].set_ylabel('Wavelength [$\AA$]')
        return plotter
    def prepare_orders(self,order):
        '''
        Returns an array or a list containing the input orders.
        '''
        nbo = self.meta['nbo']
        orders = np.arange(nbo)
        select = slice(self.sOrder,self.eOrder,1)
        
        if isinstance(order,list):
            return orders[order]
        else:
            pass
        if order is not None:
            select = self._slice(order)
        return orders[select]
    def available_orders(self):
        flux2d = self.data
        fluxod = np.sum(flux2d,axis=1)
    
    def _slice(self,order):
        nbo = self.meta['nbo']
        if isinstance(order,(int, float, complex)):
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
            start = self.sOrder
            stop  = nbo
            step  = 1
        return slice(start,stop,step)
    
    
    
    
class ESPRESSO(Spectrum):
    def __init__(self,filepath,wavereference,fr=None,f0=None,vacuum=True,
                 sOrder=40,eOrder=170,dllfile=None,*args,**kwargs):
        ext = 1 
        self._cache   = {}
        self.instrument = "ESPRESSO"
        self.meta     = io.read_e2ds_meta(filepath,ext)
        self.data     = io.read_e2ds_data(filepath,ext=1)
        self.flux     = self.data # needed for process._single_file
        self._cache['error2d']  = io.read_e2ds_data(filepath,ext=2)
        self.hdrmeta  = io.read_e2ds_meta(filepath,ext=ext)
        self.header   = io.read_e2ds_header(filepath,ext=ext)
        # include anchor offset if provided (in Hz)
        self.lfckeys  = io.read_LFC_keywords(filepath,fr,f0)
        super().__init__(filepath,f0=f0,fr=fr,vacuum=vacuum,
                         sOrder=sOrder,eOrder=eOrder,
                         *args,**kwargs)
        self.segsize  = self.meta['npix']
        varmeta       = dict(sOrder=self.sOrder,polyord=self.polyord,
                             gaps=self.gaps,segment=self.segment,
                             segsize=self.segsize,model=self.model,
                             gain=1.1)
        self.meta.update(varmeta)
        
        with FITS(wavereference,'r') as hdul:
            self._cache['wavereference'] = hdul[1].read()
        # try:
        #     vacuum    = vacuum if vacuum is not None else True
        #     self.wavereference_object = ws.ThFP(wavereference,vacuum)
        # except:
        #     self.wavereference_object = None
        if isinstance(dllfile, str):
            self._dllfile = dllfile
           
    @property
    def dll(self):
        try:
            dll2d = self._cache['dll2d']
        except:
            with FITS(self._dllfile,'r') as hdul:
                dll2d = hdul[1].read()
            self._cache['dll2d']=dll2d
        return dll2d
    @property
    def optical_orders(self):
        optord = np.arange(78+self.nbo//2-1,77,-1)
        # shift=0
        # order 117 appears twice (blue and red CCD)
        cut=np.where(optord>117)
        optord[cut]=optord[cut]-1
        
        return np.repeat(optord, 2)
    
    
    
    
class HARPS(Spectrum):
    def __init__(self,filepath,fr=None,f0=None,vacuum=True,*args,**kwargs):
        ext = 0 
        self._cache   = {}
        self.instrument = "HARPS"
        self.meta     = io.read_e2ds_meta(filepath,ext=ext)
        self.data     = io.read_e2ds_data(filepath,ext=ext)
        self.flux     = self.data # needed for process._single_file
        self.hdrmeta  = io.read_e2ds_meta(filepath,ext=ext)
        self.header   = io.read_e2ds_header(filepath,ext=ext)
        # include anchor offset if provided (in Hz)
        self.lfckeys  = io.read_LFC_keywords(filepath,fr,f0)
        
        # exclude 'wavereference' from kwargs:
        wavereference = kwargs.pop('wavereference',filepath)
        super().__init__(filepath,fr=fr,f0=f0,vacuum=vacuum,*args,**kwargs)
        
        
        self.segsize  = self.npix//16 #pixel
        varmeta       = dict(sOrder=self.sOrder,polyord=self.polyord,
                             gaps=self.gaps,segment=self.segment,
                             segsize=self.segsize,model=self.model)
        self.meta.update(varmeta)
        
        try:
            vacuum    = vacuum if vacuum is not None else True
            self.wavereference_object = ws.ThAr(wavereference,vacuum)
        except:
            self.wavereference_object = None
            
    def get_distortions(self,order=None,fittype='gauss',anchor_offset=None):
        '''
        Returns an array containing the difference between the LFC and ThAr 
        wavelengths for individual LFC lines. 
        
        Uses ThAr coefficients in air (associated to this spectrum) and
        converts the calculated wavelengths to vacuum.
        
        Args:
        ----
            order:          integer of list or orders to be plotted
            fittype:        str, list of strings: allowed values 'gauss' and
                            'lsf'. Default is 'gauss'.
            anchor_offset:  float, frequency to be artificialy added to LFC 
                            line frequencies listed in the linelist
        Returns:
        --------
            plotter:    Figure class object
        '''
        anchor_offset = anchor_offset if anchor_offset is not None else 0.0
        
        orders    = self.prepare_orders(order)
        fittypes  = np.atleast_1d(fittype)
        linelist  = self['linelist']
        wave      = hf.freq_to_lambda(linelist['freq']+anchor_offset)
        
        tharObj  = self._wavereference
        coeff, bo, qc = tharObj._get_wavecoeff_air(tharObj._filepath)
        distdict = {}
        for ft in fittypes:
            cens        = linelist['{}'.format(fittype)][:,1]
            cenerrs     = linelist['{}_err'.format(fittype)][:,1]
            distortions = container.distortions(len(linelist))
            for i,order in enumerate(orders):
                cut  = np.where(linelist['order']==order)[0]
                
                pars = coeff[order]['pars']
                if len(np.shape(pars))>1:
                    pars = pars[0]
                thar_air = np.polyval(np.flip(pars),cens[cut])
                thar_vac = ws._to_vacuum(thar_air)
                shift    = (wave[cut]-thar_vac)/wave[cut] * c
                distortions['dist_mps'][cut] = shift
                distortions['dist_A'][cut]   = wave[cut]-thar_vac
                distortions['order'][cut]    = linelist['order'][cut]
                distortions['optord'][cut]   = linelist['optord'][cut]
                distortions['segm'][cut]     = linelist['segm'][cut]
                distortions['freq'][cut]     = linelist['freq'][cut]
                distortions['mode'][cut]     = linelist['mode'][cut]
                distortions['cent'][cut]     = cens[cut]
                distortions['cenerr'][cut]   = cenerrs[cut]
            distdict[ft] = distortions
        return distdict
    
    def _generate_mask(self, start=161, end=89, step=-1, exclusions=None):
        """Helper to generate a single masked and ordered array."""
        # Ensure end for arange is correct for descending order
        actual_end = end - 1 if step < 0 else end + 1
        if step == 0:
            raise ValueError("Step cannot be zero.")

        full_range = np.arange(start, actual_end, step)

        if exclusions:
            # Using boolean indexing for potentially better performance and to keep order
            mask = ~np.isin(full_range, exclusions)
            masked_array = full_range[mask]
            return masked_array
        else:
            return full_range
        
    @property
    def optical_orders(self):
        """
        Returns a specific order mapping array based on self.nbo.

        If self.nbo == 72, returns an array from 161 down to 89, skipping 115.
        If self.nbo == 71, returns an array from 161 down to 89, skipping 115 and 116.
        Otherwise, raises a ValueError.
        """
        start_val = 161
        end_val = 89 # Inclusive end for the conceptual range
        step_val = -1

        if self.nbo == 72:
            exclusions = [115]
            return self._generate_mask(start_val, end_val, step_val, exclusions)
        elif self.nbo == 71:
            exclusions = [115, 116]
            return self._generate_mask(start_val, end_val, step_val, exclusions)
        else:
            raise ValueError(f"Unsupported nbo value: {self.nbo}. Expected 71 or 72.")

    
def distortion_statistic():
    return


def process(spec,settings_dict):
    '''
    Main routine to analyse e2ds files. 
    
    Performs line identification and fitting as well as wavelength 
    calibration. Uses provided settings to set the range of echelle orders
    to analyse, line-spread function model, ThAr calibration, etc. 
    Keeps a log.
    
    Args:
    ----
        filepath (str): path to the e2ds file
    '''
    import logging
    def get_item(spec, item, version=None, **speckwargs):
        try:
            if version is None:
                itemdata = spec[item]
            else:
                itemdata = spec[item, version]
            message = 'exists (nothing happened)'
            del itemdata
    
        except Exception:
            message = 'calculating and saving'
            try:
                if version is None:
                    itemdata = spec(item, write=True, **speckwargs)
                else:
                    itemdata = spec(item, version, write=True, **speckwargs)
                del itemdata
            except Exception:
                message = 'FAILED'
    
        finally:
            msg = (
                f"SPECTRUM {spec.filepath} "
                f"item {item.upper()} "
                f"version {version} "
                f"{message}"
            )
            logger.info(msg)
        
    def comb_specific(fittype):
        comb_items = ['coeff','wavesol','residuals','model']
        return ['{}_{}'.format(item,fittype) for item in comb_items]
    logger    = logging.getLogger(__name__+'.processing')
    versions  = np.atleast_1d(settings_dict['version'])
    # print('settings_dict=',settings_dict)
    speckwargs = _spec_kwargs(settings_dict) 
    # print(speckwargs)
    basic    = ['envelope','background','flux','error','weights',
                'noise','wavereference']#,'flux_norm','error_norm'] 
    for item in basic:
        get_item(spec,item,None)
        
    get_item(spec,
             'linelist',
             version=None,
             order=(settings_dict['sOrder'], settings_dict['eOrder']),
             fittype=settings_dict['fittype'],
            )
    # linelist = spec('linelist',order=(settings_dict['sOrder'],
    #                                   settings_dict['eOrder']),write=True,
    #                 fittype=settings_dict['fittype'],
    #                 # fittype=['gauss']
    #                 )
 
    
    
        
        
    if settings_dict['do_comb_specific']:
        combitems = []
        for fittype in np.atleast_1d(settings_dict['fittype']):
        # for fittype in np.atleast_1d(['gauss']):
            combitems = combitems + comb_specific(fittype) 
        for item in combitems:
            if item in ['model_lsf','model_gauss']:
                get_item(spec,item,None)
            else:
                for version in versions:
                    get_item(spec,item,version)
            pass
    else:
        pass
        
        
    # savepath = spec._outpath + '\n'
    # with open(settings_dict['outlist'],'a+') as outfile:
    #     outfile.write(savepath)
    logger.info('Spectrum {} FINISHED'.format(spec.filepath))
    del(spec); 
    
    return None

def _spec_kwargs(settings):
    '''
    Returns a dictionary of keywords and correspodning values that are 
    provided to harps.spectrum.Spectrum class inside self._single_file. 
    The keywords are hard coded, values should be given in the settings 
    file.
    '''
    
    kwargs = {}
    
    keywords = ['f0','fr','debug','dirpath','overwrite','sOrder','eOrder',
                'wavereference']
    
    for key in keywords:
        try:
            kwargs[key] = settings[key]
        except:
            kwargs[key] = None
    return kwargs

def get_base(filename):
    basename = os.path.basename(filename)
    return basename[0:29]  


