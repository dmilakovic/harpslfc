#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:32:58 2018

@author: dmilakov
"""
from harps.core import np
import harps.settings as hs
import numbers
#import harps.functions as hf

lineAxes      = ['pix','flx','bkg','err','rsd',
                 'sigma_v','wgt','mod','gauss_mod','wave']
wavPars       = ['val','err','rsd']
fitTypes      = ['epsf','gauss']
lineAttrs     = ['bary','n','freq','freq_err','seg','pn','snr']
orderPars     = ['sumflux']
    

#==============================================================================
#
#                         D A T A    C O N T A I N E R S                  
#
#==============================================================================
npars = hs.npars # number of parameters for the fit of individual LFC lines
def datashapes(name,**kwargs):
    npars = kwargs.pop('npars',hs.npars)
    datashapes_dict={
               'id':('id','u4',()),
               'order':('order','u4',()),
               'optord':('optord','u4',()),
               'slice':('slice', 'u4', ()),
               'index':('index','u4',()),
               'pixl':('pixl','u4',()),
               'pixr':('pixr','u4',()),
               'segm':('segm','u4',()),
               'bary':('bary','float32',()),
               'bmean':('bmean','float32',()),
               'skew':('skew','float32',()),
               'freq':('freq','float64',()),
               'mode':('mode','uint16',()),
               'anchor':('anchor','float64',()),
               'reprate':('reprate','float64',()),
               'noise':('noise','float64',()),
               'snr':('snr','float32',()),
               'resolution':('resolution','float32',()),
               'sumflx':('sumflx','float32',()),
               'sumbkg':('sumbkg','float32',()),
               'gauss_pix':('gauss_pix','float64',(npars,)),
               'gauss_pix_err':('gauss_pix_err','float64',(npars,)),
               'gauss_pix_chisq':('gauss_pix_chisq','float32',()),
               'gauss_pix_chisqnu':('gauss_pix_chisqnu','float32',()),
               'gauss_pix_aicc':('gauss_pix_aicc','float32',()),
               'gauss_wav':('gauss_wav','float64',(npars,)),
               'gauss_wav_err':('gauss_wav_err','float64',(npars,)),
               'gauss_wav_chisq':('gauss_wav_chisq','float32',()),
               'gauss_wav_chisqnu':('gauss_wav_chisqnu','float32',()),
               'gauss_wav_aicc':('gauss_wav_aicc','float32',()),
               'chisq':('chisq','float64',()),
               'chisqnu':('chisqnu','float64',()),
               'residual':('residual','float64',()),
               'lsf_pix':('lsf_pix','float64',(npars,)),
               'lsf_pix_err':('lsf_pix_err','float64',(npars,)),
               'lsf_pix_chisq':('lsf_pix_chisq','float32',()),
               'lsf_pix_chisqnu':('lsf_pix_chisqnu','float32',()),
               'lsf_pix_aicc':('lsf_pix_aicc','float32',()),
               'lsf_wav':('lsf_wav','float64',(npars,)),
               'lsf_wav_err':('lsf_wav_err','float64',(npars,)),
               'lsf_wav_chisq':('lsf_wav_chisq','float32',()),
               'lsf_wav_chisqnu':('lsf_wav_chisqnu','float32',()),
               'lsf_wav_aicc':('lsf_wav_aicc','float32',()),
               'shift':('shift','float64',()),
               'fibre':('fibre','U1',()),
               'pars':('pars','float64',(npars,)),
               'errs':('errs','float64',(npars,)),
               'success':('success','b',(2,)),
               'conv':('conv','b',()),
               'gauss_pix_integral':('gauss_pix_integral','float64',()),
               'gauss_wav_integral':('gauss_wav_integral','float64',()),
               'lsf_pix_integral':('lsf_pix_integral','float64',()),
               'lsf_wav_integral':('lsf_wav_integral','float64',()),
               # 'integral':('integral','float64',(2,))
               } # (gauss, lsf)
    return datashapes_dict[name]
def create_dtype(name,fmt,shape):
    return (name,fmt,shape)
def array_dtype(arraytype):
    assert arraytype in ['linelist','residuals','radial_velocity','linepars']
    if arraytype=='linelist':
        names = ['id','order','optord','slice','index','pixl','pixr',
                 'segm','bary','bmean','skew','freq','mode',
                 #'anchor','reprate',
                 'noise','snr','resolution','sumbkg','sumflx',
                 'gauss_pix','gauss_pix_err','gauss_pix_chisq','gauss_pix_chisqnu',
                 'gauss_wav','gauss_wav_err','gauss_wav_chisq','gauss_wav_chisqnu',
                 'lsf_pix','lsf_pix_err','lsf_pix_chisq','lsf_pix_chisqnu',
                 'lsf_wav','lsf_wav_err','lsf_wav_chisq','lsf_wav_chisqnu',
                 'success',
                 'gauss_pix_integral','gauss_wav_integral',
                 'lsf_pix_integral','lsf_wav_integral'
                 ]
#    elif arraytype == 'linepars':
#        names = ['index','gcen','gsig','gamp','gcenerr','gsigerr','gamperr',
#                         'lcen','lsig','lamp','lcenerr','lsigerr','lamperr']
    elif arraytype == 'wavesol':
        names = ['order',]
    elif arraytype == 'coeffs':
        names = ['order','segm','pixl','pixr','chisq','chisqnu','pars','errs']
    elif arraytype == 'residuals':
        names = ['order','index','segm','residual','bary','noise']
    elif arraytype == 'linepars':
        names = ['index','pars','errs','chisq','chisqnu','conv']
    else:
        names = []
    dtypes = [datashapes(name) for name in names]
    #formats = [datatypes[name] for name in names]
    #shapes  = [datashapes[name] for name in names]
    #return np.dtype({'names':names,'formats':formats, 'shapes':shapes})
    return np.dtype(dtypes)

_dtype = {'linelist':array_dtype('linelist'),
              'radial_velocity':array_dtype('radial_velocity')}

def narray(nlines,arraytype):
    dtype=array_dtype(arraytype)
    narray = np.zeros(nlines,dtype=dtype)
    narray['index'] = np.arange(nlines)
    return narray
        
def linelist(nlines,npars=hs.npars):
    linelist = narray(nlines,'linelist')
    return linelist
def linepars(nlines,npars=hs.npars):
    dtype=np.dtype([('index','u4',()),
                    ('pars','float64',(npars,)),
                    ('errs','float64',(npars,)),
                    ('chisq','float64',()),
                    ('chisqnu','float64',()),
                    ('conv','b',()),
                    ('integral','float64',()),
                    ])
    fitpars = np.zeros(nlines,dtype=dtype)
    fitpars['index'] = np.arange(nlines)
    return fitpars
def coeffs(polydeg,numsegs):
    dtype = np.dtype([('order','u4',()),
                      ('optord','u4',()),
                      ('segm','u4',()),
                      ('pixl','float64',()),
                      ('pixr','float64',()),
                      ('chisq','float64',()),
                      ('chisqnu','float64',()),
                      ('aicc','float64',()),
                      ('npts','uint16',()),
                      ('pars','float64',(polydeg+1,)),
                      ('errs','float64',(polydeg+1,))])
    narray = np.zeros(numsegs,dtype=dtype)
    narray['segm']=np.arange(numsegs)
    return narray

def residuals(nlines):
    dtype = np.dtype([('order','u4',()),
                      ('optord','u4',()),
                      ('index','u4',()),
                      ('segm','u4',()),
                      ('residual_mps','float64',()), # residual in units m/s
                      ('residual_A','float64',()), # residual in units Ansgtrom
                      ('wavefit','float64',()),
                      ('waverr','float64',()),
                      ('gauss_pix','float64',()), 
                      ('lsf_pix','float64',()),
                      ('gauss_wav','float64',()), 
                      ('lsf_wav','float64',()),
                      ('cenerr','float64',()),# center
                      ('noise','float64',())]) 
    narray = np.zeros(nlines,dtype=dtype)
    narray['index']=np.arange(nlines)
    return narray
def gaps(norders):
    dtype = np.dtype([('order','u4',()),
                      ('gaps','float64',(7,))])
    narray= np.zeros(norders,dtype=dtype)
    return narray
def radial_velocity(nexposures):
    dtype = np.dtype([('index','u4',()),
                      ('shift','float64',()),
                      ('noise','float64',()),
                      ('datetime','datetime64[s]',()),
                      ('fibre','U3',()),
                      ('flux','float64',()),
                      ('fittype','U5',())])
    narray = np.zeros(nexposures,dtype=dtype)
    narray['index'] = np.arange(nexposures)
    return narray
def lsf(numsegs,n_data,n_sct,pars=None):
    dtype_list = [('order','u4',()),
                      ('optord','u4',()),
                      ('segm','u4',()),
                      ('ledge','float32',()),
                      ('redge','float32',()),
                      ('data_x','float64',(n_data,)),
                      ('data_y','float64',(n_data,)),
                      ('data_yerr','float64',(n_data,)),
                      ('sct_x','float64',(n_sct,)),
                      ('sct_y','float64',(n_sct,)),
                      ('sct_yerr','float64',(n_sct,)),
                      ('numlines','u4',()),
                      ('logL','float64',()),
                      ('shift','float64',()),
                      ]
    if pars is not None:
        for parname in pars:
            dtype_list.append((parname,'float64',()))
    dtype = np.dtype(dtype_list)
    narray = np.full(numsegs,0,dtype=dtype)
    narray['segm'] = np.arange(numsegs)
    return narray
def lsf_gp(numsegs,npars):
    dtype = np.dtype([('order','u4',()),
                      ('optord','u4',()),
                      ('segm','u4',()),
                      ('ledge','float32',()),
                      ('redge','float32',()),
                      ('theta','float64',(npars,)),
                      ('numlines','u4',())])
    narray = np.full(numsegs,0,dtype=dtype)
    narray['segm'] = np.arange(numsegs)
    return narray
def lsf_spline(numsegs,npts):
    dtype = np.dtype([('order','u4',()),
                      ('optord','u4',()),
                      ('segm','u4',()),
                      ('ledge','u4',()),
                      ('redge','u4',()),
                      ('x','float64',(npts,)),
                      ('y','float64',(npts,)),
                      ('scatter','float64',(npts,)),
                      ('numlines','u4',())])
    narray = np.full(numsegs,0,dtype=dtype)
    narray['segm'] = np.arange(numsegs)
    return narray
def lsf_analytic(numseg,ngauss):
    '''
    Returns an empty numpy structured array containing parameters for the 
    Gaussian components. 
    
    Assumes that the LSF can be modelled as a sum of independent Gaussian
    profiles.

    Parameters
    ----------
    numseg : TYPE
        DESCRIPTION.
    ngauss : TYPE
        DESCRIPTION.

    Returns
    -------
    structured numpy array

    '''
    dtype = np.dtype([('order','u4',()),
                      ('optord','u4',()),
                      ('segm','u4',()),
                      ('pixl','u4',()),
                      ('pixr','u4',()),
#                      ('mu','float64',(ngauss,)),
#                      ('amp','float64',(ngauss,)),
#                      ('sig','float64',(ngauss,)),
                      ('pars','float64',(ngauss+2,)),
                      ('errs','float64',(ngauss+2,)),
                      ('numlines','u4',())])
    narray = np.full(numseg,0,dtype=dtype)
    narray['segm'] = np.arange(numseg)
    return narray
def datetime(numtim):
    dtype = np.dtype([('year','u4',()),
                      ('month','u4',()),
                      ('day','u4',()),
                      ('hour','u4',()),
                      ('min','u4',()),
                      ('sec','u4',())])
    narray = np.zeros(numtim,dtype=dtype)
    return narray
def distortions(nlines):
    dtype = np.dtype([('order','u4',()),
                      ('optord','u4',()),
                      ('index','u4',()),
                      ('segm','u4',()),
                      ('freq','float64',()),
                      ('mode','uint16',()),
                      ('dist_mps','float64',()),
                      ('dist_A','float64',()),
                      ('cent','float64',()),
                      ('cenerr','float64',())])
    narray = np.zeros(nlines,dtype=dtype)
    narray['index'] = np.arange(nlines)
    return narray
def add_field(a, descr):
    # https://stackoverflow.com/questions/1201817/
    # adding-a-field-to-a-structured-numpy-array
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("A must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b

class Generic(object):
    def __init__(self,narray,key1='order',key2='segm'):
        self._values = narray
        self._key1 = key1
        self._key2 = key2
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
            condict[self._key1]=order
            if segm_sent:
                condict[self._key2]=segm
        return condict, segm_sent
    def __len__(self):
        return len(self.values)
    @property
    def values(self):
        return self._values
    def select(self,condict):
        cut  = self.cut(condict) 
        
        return Generic(self.values[cut])
    def cut(self,condict):
        '''
        Speeds up the cutting, depending on the size of the dictionary.
        '''
        # check what's the maximum length of a val
        maxlen = 0
        for val in condict.values():
            lenval = len(np.atleast_1d(val))
            if lenval>maxlen:
                maxlen = lenval
        if maxlen>1:
            return self.cut_multi(condict)
        else:
            return self.cut_one(condict)
        
    def cut_one(self,condict):
        values  = self.values 
        condtup = tuple(values[key]==val for key,val in condict.items())
        
        condition = np.logical_and.reduce(condtup)
        
        return np.where(condition==True)
    
    def cut_multi(self,condict):
        values  = self.values 
        condtup = []
        # in this step, the routine iterates over all keys in condict
        # looks for logical 'or' for each value associated with this key
        # allows for selecting multiple orders or segments 
        for key,val in condict.items():    
            key_condtup = []
            for v in np.atleast_1d(val):
                if isinstance(v,numbers.Integral):
                    key_condtup.append(tuple(values[key]==v))
            key_condition = np.logical_or.reduce(key_condtup)
            condtup.append(key_condition)
        # in this step, the routine looks for logical 'and' across keys, after 
        # the 'or' step above
        condition = np.logical_and.reduce(condtup)

        return np.where(condition==True)
        
