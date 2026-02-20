#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:34:22 2018

@author: dmilakov
"""
from harps.core import FITS
from harps.core import os, np

import harps.settings as hs
import harps.functions as hf
import harps.spec_aux as saux
import warnings
#from   harps.lsf import LSF
version = hs.__version__
#==============================================================================
    
#                          T E X T    F I L E S   
    
#==============================================================================
def read_textfile(filepath,start=None,stop=None,step=None):
    if os.path.isfile(filepath):
        mode = 'r+'
    else:
        mode = 'a+'
    data = [line.strip('\n') for line in open(filepath,mode)
              if line[0]!='#']
    use = slice(start,stop,step)
    return data[use]
def write_textfile(data,filepath,header=None):
    '''
    Writes data to file.
    
    Args:
    -----
        data : str or list
    '''
    
    write_header = True if header is not None else False
    # Make directory if does not exist
    dirname = os.path.dirname(filepath)
    success = hs.make_directory(dirname)
    if success:
        pass
    else:
        raise ValueError("Could not make directory")
    # Check if file exists
    if os.path.isfile(filepath):
        mode = 'a'
    else:
        mode = 'w'
        
    # Write data
    data_out = to_string(data)
    with open(filepath,mode) as outfile:
        if write_header:
            outfile.write(header)
        else:
            pass
        outfile.write(data_out)
        
def to_string(obj,sep='\n'):
    '''
    Separator must be a string
    '''
    assert isinstance(sep,str)==True
    
    if isinstance(obj,str):
        return obj
    elif isinstance(obj,list):
        return sep.join(str(val) for val in obj)
    else:
        return None
#==============================================================================
    
#                                 E 2 D S   
    
#==============================================================================
def read_e2ds_data(filepath,ext=0):
    with FITS(filepath,memmap=False) as hdulist:
        data     = hdulist[ext].read()
    return data
def mread_e2ds_data(e2dslist,ext=0):
    ''' Reads the data of all files in the list provided'''
    data = []
    filelist = read_textfile(e2dslist)
    for i,filepath in enumerate(filelist):
        hf.update_progress(i/(len(filelist)-1),"Read fits")
        with FITS(filepath,memmap=False) as hdulist:
            data.append(hdulist[ext].read())
    return np.stack(data)

def read_e2ds_meta(filepath,ext=0):
    header   = read_e2ds_header(filepath,ext=ext)
    npix     = header["NAXIS1"]
    nbo      = header["NAXIS2"]
    header0  = read_e2ds_header(filepath,ext=0)
    exptime  = header0['EXPTIME']
    obsdate  = header0['DATE-OBS']
    mjdobs   = header0['MJD-OBS']
    try:
        gain   = header['HIERARCH ESO DRS CCD CONAD']
    except:
        gain = 1.
    # Fibre information is not saved in the header, but can be obtained 
    # from the filename 
    fibre    = filepath[-6]
    fibshape = 'octogonal'
    
    meta     = dict(npix=npix, nbo=nbo, exptime=exptime, fibre=fibre, 
               fibshape=fibshape, obsdate=obsdate,mjd=mjdobs,gain=gain)
    return meta 
def read_e2ds_header(filepath,ext=0):
    with FITS(filepath) as hdulist:
        header   = hdulist[ext].read_header()
    return header
def read_e2ds(filepath):
    try:
        data   = read_e2ds_data(filepath,ext=0)
        ext    = 0
    except:
        data   = read_e2ds_data(filepath,ext=1)
        ext    = 1
    meta   = read_e2ds_meta(filepath,ext)
    header = read_e2ds_header(filepath,ext)
    return data, meta, header
def read_LFC_keywords(filepath,fr=None,f0=None,ext=0):
    with FITS(filepath,memmap=False) as hdulist:
        header   = hdulist[ext].read_header()
    
    
    if f0 is None:
        try:
            f0 = header["HIERARCH ESO INS5 LAMP4 FREQOFFS"]
        except:
            f0 = header['ESO INS LFC1 ANCHOR']
    else:
        f0 = f0
    if fr is None:
        try:
            fr = header["HIERARCH ESO INS5 LAMP4 FREQ"]
        except:
            fr = header['ESO INS LFC1 REPRATE']
    else:
        fr = fr
    window = int(np.ceil(fr/5e9) * 2 + 1) 
    # pixPerLine = window*7

    
    LFC_keys = dict(comb_anchor=f0, window_size=window,
                    source_anchor=f0, comb_reprate=fr)
    return LFC_keys
def read_optical_orders(filepath):
    '''
    Returns a list of optical orders corresponding to the zero-indexed arrays
    saved in FITS files. 
    
    Specific to HARPS. Do not use for ESPRESSO.

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.

    Returns
    -------
    optord : list
        Optical orders that correspond to the grating equation.

    '''
    meta   = read_e2ds_meta(filepath)
    nbo    = meta['nbo']
    optord = np.arange(88+nbo,88,-1)
    # fibre A doesn't contain order 115
    if meta['fibre'] == 'A':
        shift = 1
    # fibre B doesn't contain orders 115 and 116
    elif meta['fibre'] == 'B':
        shift = 2
    cut=np.where(optord>114)
    optord[cut]=optord[cut]+shift
    
    return optord
# =============================================================================
#    
#                        O U T    F I L E S
#    
# =============================================================================
    
def _check_if_list(item):
    if isinstance(item,list):
        return item
    else:
        return [item]
    
def read_outfile(filepath,version=None):
    return read_outfile_extension(filepath,['linelist','wavesol_comb'],version)
def read_outfile_header(filepath,extension=0,version=None):
    with FITS(filepath,'r') as fits:
        header = fits[extension,version].read_header()
    fluxes = [rec['value'] for rec in header.records() \
                        if 'FLUX_ORD' in rec['name']]
    b2e    = [rec['value'] for rec in header.records() \
                        if 'B2E_ORD' in rec['name']]
    return fluxes, b2e

def check_outfile(func):
    '''
    Checks if the input file exists. Raises exception otherwise.
    '''
    def func_wrapper(filelist,*args,**kwargs):
        if isinstance(filelist,list):
            filelist = filelist
        elif isinstance(filelist,str):
            exists = os.path.exists(filelist)
            if exists:
                filelist = read_textfile(filelist)
            else:
                raise ValueError('Provided file does not exist')
        else:
            raise ValueError('Input not understood')
        filelist = np.sort(filelist)
        return func(filelist,*args,**kwargs)
    return func_wrapper

@check_outfile
def mread_outfile_primheader(filelist,records=['flux','b2e'],*args,**kwargs):
    '''
    Reads the primary headers of all 'out' FITS files in outlist.
    '''
    records = np.atleast_1d(records)
    cache = {record:[] for record in records}
    for i,file in enumerate(filelist):
        hf.update_progress(i/(len(filelist)-1),__name__+ \
                           '.mread_outfile_primheader')
        with FITS(file,'r') as fits: 
            header = fits[0].read_header()
            hrecords = header.records()
            for record in records:
                if record in ['flux','b2e']:
                    values = [hrec['value'] for hrec in  hrecords \
                       if '{}ORD'.format(record.upper()) in hrec['name']]
                else:
                    values = [hrec['value'] for hrec in  hrecords \
                       if record.upper() in hrec['name']]
                if record=='date-obs':
                    values = np.array(values,dtype='datetime64[s]')
                cache[record].append(values)
                del(values)
    for ext,lst in cache.items():
        cache[ext] = np.array(lst)
    return cache, len(filelist)
def read_outfile_extension(filepath, extension=['wavesol_comb'],version=501):
    extension = np.atleast_1d(extension)
    data = []
    with FITS(filepath,'r') as fits: 
        for ext in extension:
            if ext == 'linelist':
                    data.append(fits[ext].read())
            else:
                try:
                    data.append(fits[ext,version].read())
                except:
                    raise ValueError("Extension {0}, v{1} "
                                     "could not be found".format(ext,version))
    return tuple(data)
def read_fluxord(filepath):
    '''
    Read flux per order from the header of 'out' FITS file.
    '''
    with FITS(filepath,'r') as fits:
        header = fits[0].read_header()
    fluxes = [rec['value'] for rec in header.records() \
                        if 'FLUX_ORD' in rec['name']]
    return fluxes
def read_b2eord(filepath):
    '''
    Read flux per order from the header of 'out' FITS file.
    '''
    with FITS(filepath,'r') as fits:
        header = fits[0].read_header()
    b2e    = [rec['value'] for rec in header.records() \
                        if 'B2E_ORD' in rec['name']]
    return b2e

@check_outfile
def mread_outfile(filelist,extensions,version=None,avflux=False,
                  **kwargs):
    version    = saux.item_to_version(version)
    extensions = np.atleast_1d(extensions)
    orders     = kwargs.pop('order',None)
    
    cache = {ext:[] for ext in extensions}
        
    N     = len(filelist)
    M     = max(1,N)
    for i,file in enumerate(filelist):
        
        with FITS(file,'r') as fits:
            for ext,lst in cache.items():
                # try:
                #     ext,ver = extension.split('_')
                # except:
                #     ext = extension
                #     ver = version
                # print(ext,version)
                try:
                    if ext=='datetime':
                        data = hf.basename_to_datetime(file)
                    elif ext=='avnoise':
                        linelist = fits['linelist'].read()
                        data = hf.noise_from_linelist(linelist)
                    elif ext=='avflux':             
                        linelist = fits['linelist'].read()
                        flux2d   = fits['flux'].read()
                        bkg2d    = fits['background'].read()
                        data = hf.average_line_flux(linelist,flux2d,bkg2d,orders)
                    elif ext in ['fluxord','b2eord']:
                        header = fits[0].read_header()
                        data   = [rec['value'] for rec in header.records() \
                                   if ext.upper() in rec['name']]
                    elif ext not in ['weights',
                                   'background','flux','error']:
                        data = fits[ext,version].read() 
                    else:
                        data = fits[ext].read()
                except:
                    try:
                        data = fits[ext].read()
                    except:
                        warnings.warn(f"Could not read extension '{ext}'",
                                      f" version {version} "
                                      f"in {file}, number {i}")
                hf.update_progress(i/M,'Read')
                lst.append(data)
                
            
    for ext,lst in cache.items():
        # print(ext)
        cache[ext] = np.array(lst)

    return cache, len(filelist)

#==============================================================================
    
#               L I N E L I S T      A N D     W A V E S O L   
    
#==============================================================================
allowed_hdutypes = ['linelist','flux','background','error','weights','envelope',
                    'coeff_gauss','coeff_lsf','wavesol_gauss','wavesol_lsf',
                    'model_gauss','model_lsf','residuals_gauss','residuals_lsf',
                    'wavesol_2pt_lsf','wavesol_2pt_gauss','noise',
                    'wavereference', 'extrema', 'line_positions']
def new_fits(filepath,dirpath=None):
    # ------- Checks 
#    assert hdutype in allowed_hdutypes, 'Unrecognized HDU type'
    path = get_fits_path(filepath,dirpath)
    try:
        os.remove(path)
    except:
        pass
    newfits = FITS(path,mode='rw')
    return newfits

def open_fits(filepath,dirpath=None,mode='rw',overwrite=False):
    fits = read_fits(filepath,dirpath,mode)
    if fits is not None:
        if overwrite==True:
            return new_fits(filepath,dirpath)
        else:
            return fits
    else:
        return new_fits(filepath,dirpath)
def fits_exists(filetype,filepath,version=version,dirpath=None):
    '''
    Returns true if the appropriate file exits. 
    '''
    exists = False
    path = get_fits_path(filetype,filepath,version=version,dirpath=dirpath)
    if os.path.isfile(path):
        exists = True
    return exists
def check_exists(func):
    '''
    Checks if the appropriate file exits. Raises exception otherwise
    '''
    def func_wrapper(filetype,filepath,version=version,dirpath=None):
        path = get_fits_path(filetype,filepath,version=version,dirpath=dirpath)
        if os.path.isfile(path):
            return func
        else:
            raise ValueError("No file {} found.".format(path))
        return 

def read_fits(filepath,dirpath=None,mode='rw',overwrite=False):
    '''
    Wrapper around fitsio.FITS. Returns None if the file does not exist. 
    Doesn't raise an error.
    '''
    try:
        path = get_fits_path(filepath,dirpath)
        hdu  = FITS(path,mode=mode,clobber=overwrite)
    except:
#        raise IOError("File does not exist")
        hdu = None
    return hdu

def read_hdudata(filepath,dirpath=None):
    '''
    Returns the content of the FITS file as a dictionary. 
    '''
    with get_hdu(filepath,dirpath) as hdu:
        hdudata = {}
        for h in hdu[1:]:
            extname = h.get_extname()
            data    = h.read()
            hdudata[extname]=data
    return hdudata
    

def read_hduext(filepath,extname,dirpath=None):
    '''
    Reads the extension 'extname' from the FITS file 'filepath'
    '''
    assert extname in allowed_hdutypes
    with open_fits(filepath,dirpath) as fits:
        hduext = fits[extname].read()
        return hduext
def write_hdu(filepath,data,extname,header=None,dirpath=None):
    assert extname in allowed_hdutypes
    with open_fits(filepath,dirpath) as fits:
        fits.write(data=data,extname=extname,header=header)
        print(fits)
    return
def get_fits_path(filetype,filepath,version=version,dirpath=None,filename=None):
    '''
    Returns the path to the FITS file of certain type
    '''
    dirname  = get_dirpath(filetype,version,dirpath)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    filename  = filename if filename is not None else get_filename(basename,filetype)
    path     = os.path.join(dirname,filename)
    return path  
def get_filename(basename,filetype):
    '''
    Returns the basename of a certain filetype
    '''
    # basename = os.path.splitext(os.path.basename(basename))[0]
    if filetype=='fits':
        filename  = basename.replace('e2ds','out')+'.fits'
    elif filetype == 'objspec':
        filename  = basename+'_calib.fits'
    elif filetype=='series':
        filename = basename+'_series.fits'
    elif filetype=='dataset':
        filename  = basename+'.fits'
    elif filetype=='lsf':
        filename = basename + '_lsf.fits'
    return filename
def get_dirpath(filetype,version=version,dirpath=None):
    ''' Returns the path to the directory with files of the selected type. '''
    if dirpath is not None:
        dirpath = dirpath
    else:
        dirpath = hs.get_dirname(filetype,version)
    #print("DIRNAME = ",dirpath)
    direxists = os.path.isdir(dirpath)
    if not direxists:
        create=input("Directory {} does not exist. Create? (y/n)".format(dirpath))
        if create == 'y':
            try: 
                hs.make_directory(dirpath)
            except:
                dirpath = hs.harps_fits
        else:
            dirpath = hs.harps_fits
    return dirpath
def get_extnames(filepath,dirpath=None):
    with get_hdu(filepath,dirpath) as hdu:
        extnames = [h.get_extname() for h in hdu[1:]]
    return extnames


# =============================================================================
        
#                           L          S           F
        
# =============================================================================
def lsf_from_file(filepath):
    hdu = FITS(filepath)
    lsf = hdu[-1].read()
    return lsf