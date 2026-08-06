#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:06:39 2023

@author: dmilakov
"""
import numpy as np
#%%
import harps.spectrum as hc
spec=hc.HARPS('/Users/dmilakov/projects/lfc/data/harps/e2ds/2018-12/2018-12-04/HARPS.2018-12-05T08:12:52.040_e2ds_A.fits',f0=4.68e9,fr=18e9,overwrite=False)

flx2d=spec.data
bkg2d=spec.background
err2d=spec.error
x2d=np.array([np.arange(spec.npix) for i in range(spec.nbo)])
#%%
import harps.lsf.aux as aux
x3d   = aux.prepare_array(x2d)
flx3d = aux.prepare_array(flx2d)
bkg3d = aux.prepare_array(bkg2d)
err3d = aux.prepare_array(err2d)
#%%

linelist=spec['linelist']
cut=np.where(linelist['order']==50)[0]

#%%
from fitsio import FITS
filename = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/HARPS.2018-12-05T08:12:52.040_e2ds_A_lsf.fits'
filename = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/HARPS.2018-12-05T08:12:52.040_e2ds_A_lsf.fits_bk'
hdul=FITS(filename)
lsf1d50=hdul[1].read()

#%%
exp=0
llist=linelist[cut]
line=llist[100]
fittype = 'gauss'

od   = line['order']
segm = line['segm']
# mode edges
lpix = line['pixl']
rpix = line['pixr']
bary = line['bary']
cent = line['{}_pix'.format(fittype)][1]
flx1l  = flx3d[exp,od,lpix:rpix]
x1l   = x3d[exp,od,lpix:rpix]
# pix  = np.arange(lpix,rpix,1.) 
bkg1l  = bkg3d[exp,od,lpix:rpix]
err1l  = err3d[exp,od,lpix:rpix]

y_data = flx1l-bkg1l
y_err  = np.sqrt(flx1l+bkg1l)
# wgt  = np.ones_like(pix)
# initial guess
p0 = (np.max(flx1l),cent,1)
#%%
import harps.lsf.gp_aux as gp_aux
optpars = gp_aux.get_parameters(lsf1d50,x1l,y_data,y_err,interpolate=True)
optpars2 = gp_aux.get_parameters(lsf1d50,x1l,y_data,y_err,interpolate=False)
#%%
gp_aux.plot_result(optpars[0],lsf1d50,x1l,flx1l,bkg1l,err1l,interpolate=True)
gp_aux.plot_result(optpars2[0],lsf1d50,x1l,flx1l,bkg1l,err1l,interpolate=False)
#%%
import harps.lsf.container as container
LSF = container.LSF(lsf1d50)
#%%
new_llist_notinterpolated=aux.solve(LSF,llist,x2d,flx2d,bkg2d,err2d,'gauss',interpolate=False)
#%%
new_llist_interpolated=aux.solve(LSF,llist,x2d,flx2d,bkg2d,err2d,'gauss',interpolate=True)
#%%
import matplotlib.pyplot as plt

plt.figure()
for ll in [new_llist_interpolated]:
    cent_gauss=ll['gauss_pix'][0,:,1]
    cent_lsf=ll['lsf_pix'][0,:,1]
    plt.scatter(bary,(cent_lsf-cent_gauss)*829,marker='.',s=4)
plt.xlabel("Barycentre [pix]")
plt.ylabel("LSF-Gauss centre *829 [m/s]")




#%%
import jax
import numpyro
import harps.lsf.fit as lsffit
nuts_kernel = numpyro.infer.NUTS(lsffit.numpyro_model, target_accept_prob=0.9)
mcmc = numpyro.infer.MCMC(
    nuts_kernel,
    num_warmup=150,
    num_samples=150,
    num_chains=1,
    progress_bar=True,
)
rng_key = jax.random.PRNGKey(55873)
mcmc.run(rng_key, x_test=x1l, y_data=y_data,y_err=y_err,lsf1d=lsf1d50)
samples = mcmc.get_samples()




