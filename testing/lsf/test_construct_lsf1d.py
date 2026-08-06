#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:45:13 2023

@author: dmilakov
"""

import harps.spectrum as hc
filename = '/Users/dmilakov/projects/lfc/data/harps/e2ds/2018-12/2018-12-04/'+\
            'HARPS.2018-12-05T08:12:52.040_e2ds_A.fits'
spec=hc.HARPS(filename,f0=4.68e9,fr=18e9,overwrite=True)
#%%
spec.process(fittype='gauss')
#%%
import harps.lsf.construct as construct

lsf1d=construct.from_spectrum_1d(spec,order=50,iteration=1,scale='pixel',
                                 iter_center=5,numseg=16,model_scatter=True,
                                 save_fits=True,clobber=False)
#%%
import harps.lsf.aux as aux
import harps.io as hio
lsf_filepath=hio.get_fits_path('lsf',spec.filepath)
new_llist=aux.solve(spec._outpath,lsf_filepath,1,50,scale='pixel',interpolate=True)
#%%
lsf1d=construct.from_spectrum_1d(spec,order=50,iteration=2,scale='pixel',
                                 iter_center=5,numseg=16,model_scatter=True,
                                 save_fits=True,clobber=False)
#%%
lsf_filepath=hio.get_fits_path('lsf',spec.filepath)
new_llist=aux.solve(spec._outpath,lsf_filepath,2,50,scale='pixel',interpolate=True)
#%%
lsf1d=construct.from_spectrum_1d(spec,order=50,iteration=3,scale='pixel',
                                 iter_center=5,numseg=16,model_scatter=True,
                                 save_fits=True,clobber=False)
#%%
lsf_filepath=hio.get_fits_path('lsf',spec.filepath)
new_llist=aux.solve(spec._outpath,lsf_filepath,3,50,scale='pixel',interpolate=True)
#%%
iteration=4
lsf1d=construct.from_spectrum_1d(spec,order=50,iteration=iteration,scale='pixel',
                                 iter_center=5,numseg=16,model_scatter=True,
                                 save_fits=True,clobber=False)
new_llist=aux.solve(spec._outpath,lsf_filepath,iteration,50,scale='pixel',interpolate=True)