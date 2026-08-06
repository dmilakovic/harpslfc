#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:11:24 2023

@author: dmilakov
"""

from fitsio import FITS
import harps.lsf.container as container
from glob import glob
from velplot.fortread import read_fort13
import os


lsfpath = '/Users/dmilakov/projects/lfc/dataprod/v2.3.7/lsf/HARPS.2018-12-05T08:12:52.040_e2ds_A_lsf.fits'
version = 1
wavefilelist = glob('/Users/dmilakov/projects/Q0515-4414/data/harps/from_rfc/*_wave_A.fits')
errfilelist =  glob('/Users/dmilakov/projects/Q0515-4414/data/harps/from_rfc/*_error_A.fits')

fort13_files = [
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg1/reg1_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg2/reg2_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg3/reg3_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg4/reg4_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg5/reg5_in.13',
    ]
savedir = '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/2023-10-16/ip/'


for i,file in enumerate(fort13_files):
    f13_out = read_fort13(file)
    table2  = f13_out[2]
    for j,transition in enumerate(table2):
        wstart = transition['wave_start']
        wend   = transition['wave_end']
        save_path = os.path.join(savedir,f'IP_r{i+1}_s{j+1}.dat')
        container.combine_from_list_of_files(lsfpath,version,
                                             xrange=6, # pixels
                                             dv=0.83, # km/s
                                             subpix=25,
                                             wstart=wstart,
                                             wend=wend,
                                             wavefilelist=wavefilelist,
                                             errfilelist=errfilelist,
                                             save=True,
                                             save_to=save_path)
        print(f'Model for file {i+1} segment {j+1} saved to {save_path}')
#%%
fort13_files_old = [
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg1/reg1_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg2/reg2_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg3/reg3_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg4/reg4_in.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2021/mvpfit084/5alpha_lfc/reg5/reg5_in.13',
    ]

fort13_files_new = [
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/test_2023-08-21/5alpha_lfc/reg1/reg1_ip_vpfit11p1_out.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/test_2023-08-21/5alpha_lfc/reg2/reg2_ip_vpfit11p1_out.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/test_2023-08-21/5alpha_lfc/reg3/reg3_ip_vpfit11p1_out.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/test_2023-08-21/5alpha_lfc/reg4/reg4_ip_vpfit11p1_out.13',
    '/Users/dmilakov/projects/Q0515-4414/fort13/milakovic2023/test_2023-08-21/5alpha_lfc/reg5/reg5_ip_vpfit11p1_out.13',
    ]

import numpy as np
def minimal_z_uncertainty(fort13_files):
    params=[]
    errors=[]
    for i,file in enumerate(fort13_files):
        f13_out = read_fort13(file)
        params.append(f13_out[3])
        errors.append(f13_out[4])
    params=np.hstack(params)
    errors=np.hstack(errors)
    _ = errors['redshift']/(1+params['redshift'])*299792458
    cut=np.where(np.abs(_)>1e-8)[0]
    vel = _[cut]
    loc=np.argmin(vel)
    return loc,vel[loc],params[cut[loc]],errors[cut[loc]]

def minimal_doppler_b_value(fort13_files):
    params=[]
    errors=[]
    for i,file in enumerate(fort13_files):
        f13_out = read_fort13(file)
        params.append(f13_out[3])
        errors.append(f13_out[4])
    params=np.hstack(params)
    errors=np.hstack(errors)
    _ = params['doppler']
    cut=np.where(_>0)[0]
    b = _[cut]
    loc=np.argmin(b)
    return loc,b[loc],params[cut[loc]],errors[cut[loc]]

# print(minimal_z_uncertainty(fort13_files_old))
# print(minimal_z_uncertainty(fort13_files_new))
print(minimal_doppler_b_value(fort13_files_old))
print(minimal_doppler_b_value(fort13_files_new))
#%%


