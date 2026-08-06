#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 17:40:45 2023

@author: dmilakov
"""
import matplotlib.pyplot as plt
import harps.compare as hcomp
import harps.plotter as hplt
from fitsio import FITS
import numpy as np
#%%
# outpath = '/Users/dmilakov/projects/lfc/dataprod/fits/v_2.0/HARPS.2018-12-05T08:12:52.fits_bk'
# outpath = '/Users/dmilakov/projects/lfc/dataprod/fits/v_2.0/HARPS.2018-12-05T08:12:52.fits'
outpath = '/Users/dmilakov/projects/lfc/dataprod/v_2.2/fits/HARPS.2018-12-05T08:12:52.fits'
# outpath = '/Users/dmilakov/projects/lfc/dataprod/fits/v_2.0/HARPS.2018-12-10T05:25:48.fits'
# outpath='/Users/dmilakov/projects/lfc/dataprod/from_bb/fits/v_2.0/HARPS.2018-12-05T08:12:52.fits'

#%%
# hdu=FITS(outpath,'rw',clobber=False)
#%%
# firstrow = 0 ; lastrow = 368 # od = 39
# firstrow = 369; lastrow=734 # od = 40
# firstrow = 734; lastrow=1034 # od = 41
# firstrow=3551 ; lastrow=3883 # od = 49
# firstrow=3884; lastrow=4216 # od = 50
# firstrow=4217; lastrow=4546 # od = 51
# firstrow = 3551; lastrow=4546
od = 50
with FITS(outpath,'rw',clobber=False) as hdu:
    linelist = hdu['linelist'].read()
    cut=np.where(linelist['order']==50)[0]
    firstrow = cut[0]; lastrow = cut[-1]
#%%
# versions = [111,211,311,411,511,611,711,811,911]
plt.figure()
with FITS(outpath,'rw',clobber=False) as hdu:
    gauss_chisqnu=hdu['linelist'].read(columns='gauss_pix_chisqnu')[firstrow:lastrow]
    lsf_chisqnu=hdu['linelist',111].read(columns='lsf_pix_chisqnu')[firstrow:lastrow]
#plt.hist(gauss_chisqnu,histtype='step',bins=50,label='gauss',lw=2)
for it in versions:
    with FITS(outpath,'rw',clobber=False) as hdu:
        lsf_chisqnu_it=hdu['linelist',it].read(columns='lsf_pix_chisqnu')[firstrow:lastrow]
    #diff = lsf_chisqnu_itm1-lsf_chisqnu_it
    plt.scatter(lsf_chisqnu,lsf_chisqnu_it,s=1,label=f"iteration={it}")
plt.xlabel('LSF chisqnu version=111')
plt.ylabel('LSF chisqnu')
plt.legend()
#%% gauss - lsf difference

# def plot_centre_differences(outpath,od,versions):
#     fig = hplt.Figure2(2,1,figsize=(8,7),top=0.9)
#     ax1 = fig.ax()
#     ax2 = fig.add_subplot(1,2,0,1,sharex=ax1)
#     # ax3 = fig.add_subplot(2,3,0,1)
#     # fig, (ax1,ax2,ax3) = plt.subplots(2,sharex=True)
#     with FITS(outpath,'r',clobber=False) as hdu:
#         linelist = hdu['linelist'].read()
#         cut_ = np.where(linelist['order']==od)[0]
#         colnum = 1
#         index1 = linelist[cut_]['id']
#         bary = linelist['bary'][cut_]
#         gcens = linelist['gauss_pix'][cut_,colnum]
#         ax1.scatter(bary,gcens-bary,ls='-',marker='x',
#                 label=f"Gauss",c='gray')
#         ax1.axhline(0,ls=':')
#         for it in versions:#,611,711,811,911]:
#             linelist_it = hdu['linelist',it].read()
#             cut  = np.where(linelist_it['order']==od)[0]
#             bary = linelist_it['bary'][cut]
#             skew = linelist_it['skew'][cut]
#             cens = linelist_it['lsf_pix'][cut,colnum]
#             errs = linelist_it['lsf_pix_err'][cut,colnum]
#             ax1.scatter(bary,cens-bary,marker='o',s=2,
#                     label=f"LSF v={it}")
#             # lsf_cens = linelist_it['lsf_pix'][cut,colnum]
#             index2 = linelist_it[cut]['id']
#             # gauss_cens=linelist_it['gauss_pix'][cut,colnum]
            
            
#             sorter1,sorter2 = hcomp.get_sorted(index1, index2)
#             gcen_sorted = gcens[sorter1]
#             lcen_sorted = cens[sorter2]
#             lerr_sorted = errs[sorter2]
#             # print(np.all(index1[sorter1]==index2[sorter2]))
#             diff = (lcen_sorted-gcen_sorted)
#             ax2.errorbar(bary,diff,lerr_sorted,ls='',lw=0.5,capsize=2,
#                          marker='.',ms=2,label=f"iteration={it}")
#             # ax3.scatter(skew,cens-bary,s=2,label=f"iteration={it}")
#     seglen=256
#     limits = np.arange(0,4097,seglen)
#     segcens = (limits[:-1]+limits[1:])/2.
#     for ax in [ax1,ax2]:
#         ax.set_xlabel("Line barycentre (pix)")
#         ylims = ax.get_ylim()
#         [ax.axvline(_,ls=":") for _ in limits]
#         [ax.text(x=_,y=ylims[1]-0.05*np.diff(ylims),s=i,horizontalalignment='center')
#                  for i,_ in enumerate(segcens)]
#     ax1.set_title(f'Order {od}')
#     ax1.set_ylabel(r"Centre $-$ barycentre"+r" (pix)")
#     ax2.set_ylabel(r"LSF $-$ Gaussian centre"+r" (pix)")#r" (ms$^{-1}$)")
#     ax1.legend(loc='center',bbox_to_anchor=(0.15,1.15), 
#               fontsize=9, bbox_transform=ax1.transAxes, ncol=3)
    
def plot_centre_differences(outpath,od,versions):
    fig = hplt.Figure2(3,1,figsize=(5,8),top=0.93,left=0.15,right=0.91,
                       height_ratios=[5,5,2])
    ax1 = fig.ax()
    ax2 = fig.add_subplot(1,2,0,1)#,sharex=ax1)
    ax3 = fig.add_subplot(2,3,0,1)#,sharex=ax1)
    # fig, (ax1,ax2,ax3) = plt.subplots(2,sharex=True)
    with FITS(outpath,'r',clobber=False) as hdu:
        linelist = hdu['linelist'].read()
        cut_ = np.where(linelist['order']==od)[0]
        colnum = 1
        index1 = linelist[cut_]['id']
        bary = linelist['bary'][cut_]
        gcens = linelist['gauss_pix'][cut_,colnum]
        ax1.scatter(bary,-(gcens-bary),ls='-',marker='.',s=2,
                label=f"Gaussian approx.",c='gray')
        ax1.axhline(0,ls=':')
        for it in versions:#,611,711,811,911]:
            linelist_it = hdu['linelist',it].read()
            cut  = np.where(linelist_it['order']==od)[0]
            bary = linelist_it['bary'][cut]
            skew = linelist_it['skew'][cut]
            cens = linelist_it['lsf_pix'][cut,colnum]
            errs = linelist_it['lsf_pix_err'][cut,colnum]
            chisq = linelist_it['lsf_pix_chisqnu'][cut]
            
            if it !=1:
                label = r"$\psi$ "+f"iteration {it//100}"
                s = 2
                zorder = 0
                marker = 'o'
            else:
                label = r"Most likely $\psi$"
                s = 2
                zorder = 10
                marker = 's'
            ax1.scatter(bary,-(cens-bary),marker=marker,s=s,zorder=zorder,
                    label=label)
            # lsf_cens = linelist_it['lsf_pix'][cut,colnum]
            index2 = linelist_it[cut]['id']
            # gauss_cens=linelist_it['gauss_pix'][cut,colnum]
            
            
            sorter1,sorter2 = hcomp.get_sorted(index1, index2)
            gcen_sorted = gcens[sorter1]
            lcen_sorted = cens[sorter2]
            lerr_sorted = errs[sorter2]
            chisq_sorted = chisq[sorter2]
            # print(np.all(index1[sorter1]==index2[sorter2]))
            diff = -(lcen_sorted-gcen_sorted)
            print(f'Median error in version {it} = {np.median(lerr_sorted):8.2e}')
            # print(f'Median chiqnu in version {it} = {np.median(chisq_sorted):8.2e}')
            ax2.errorbar(bary,diff,lerr_sorted,ls='',marker=marker,lw=0.5,
                         ms=2,label=f"iteration={it}",capsize=2,capthick=0.5,
                         zorder=zorder)
            ax3.scatter(bary,chisq_sorted,marker=marker,s=s,zorder=zorder,
                        label=f"iteration={it}")
    seglen=256
    limits = np.arange(0,4097,seglen)
    segcens = (limits[:-1]+limits[1:])/2.
    for ax in [ax1,ax2,ax3]:
        
        
        [ax.axvline(_,ls=":") for _ in limits]
    ylims = fig.axes[0].get_ylim()
    [fig.axes[0].text(x=_,y=ylims[1]+0.03*np.diff(ylims),
                      s=i+1,
                      horizontalalignment='center')
             for i,_ in enumerate(segcens)]
    
    ax1.set_title(f'Optical order {optord(od)}',pad=20)
    ax1.set_ylabel(r"Centroid $-$ measured centre")
    ax2.set_ylabel(r"Gaussian $-$ $\psi$ centre")
    # ax3.set_xlabel("Line barycentre (pix)")
    ax3.set_ylabel(r"$\chi^2_{\nu}$")
    ax3.set_yscale('log')
    ax1.legend(loc='upper right',
               #bbox_to_anchor=(0.6,0.15), 
              fontsize=9, 
              #bbox_transform=ax1.transAxes,
              # ncol=int((len(versions)+1)/2),
              ncol=1,
              framealpha=1.)
    for i in range(len(fig.axes)):
        fig.major_ticks(i,axis='x',tick_every=512)
        fig.minor_ticks(i,axis='x',tick_every=128)
        if i==0:
            ylims = fig.axes[i].get_ylim()
            fig.axes[i].text(x=1.06,y=0.98,
                              s='ms$^{-1}$',
                              transform=fig.axes[i].transAxes,
                              horizontalalignment='center')
            fig.axes[i].text(x=-.05,y=0.98,
                              s='pix',
                              transform=fig.axes[i].transAxes,
                              horizontalalignment='center')
        if i<(len(fig.axes)-1):
            fig.major_ticks(i,axis='y',ticknum=6)
            fig.axes[i].set_xticklabels([])
            temp_ax = fig.axes[i].secondary_yaxis('right',functions=(pix2vel,vel2pix))
            fig.axes[i].tick_params(axis="y", right=False)
        else:
            fig.axes[i].axhline(1,ls='-',c='C0',lw=1.2,zorder=-5)
            import matplotlib.ticker as ticker
            fig.axes[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    fig.axes[-1].set_xlabel("Line centroid (pix)")
    fig.figure.align_ylabels()
    save=False
    if save:
        fig.save()
    
def pix2vel(x):
    return x*829
def vel2pix(v):
    return v/829    

def optord(order):
    array = np.array([161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149,
           148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136,
           135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123,
           122, 121, 120, 119, 118, 117, 116, 114, 113, 112, 111, 110, 109,
           108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98,  97,  96,
            95,  94,  93,  92,  91,  90,  89])
    
    return array[order]
        
od=60

# versions = [111,211,311,411,511,611,711,811,911]
# versions = [111,211]
# versions = [111,211,311,411,511]
# versions = [111,511]
versions = [1,111,1011]
# versions=[111,211,311,411,511]
# versions = [101,201,301]
# plot_centre_differences(outpath,od,versions)
plot_centre_differences(outpath,od,versions)
#%% phase space
od=50
fig, ax = plt.subplots()
versions = [111]#,211,311,411,511]
# versions = [101,201,301]
with FITS(outpath,'rw',clobber=False) as hdu:
    linelist = hdu['linelist'].read()
    cut_ = np.where(linelist['order']==od)[0]
    colnum = 1
    index1 = linelist[cut_]['id']
    gauss_cens=linelist['gauss_pix'][cut_,colnum]
    for it in versions:#,611,711,811,911]:
        linelist_it = hdu['linelist',it].read()
        cut=np.where(linelist_it['order']==od)[0]
        lsf_cens = linelist_it['lsf_pix'][cut,colnum]
        index2 = linelist_it[cut]['id']
        
        sorter1,sorter2 = hcomp.get_sorted(index1, index2)
        gcen_sorted = gauss_cens[sorter1]
        lcen_sorted = lsf_cens[sorter2]
        bary_sorted = linelist_it['bary'][cut][sorter2]
        print(np.all(index1[sorter1]==index2[sorter2]))
        diff = (lcen_sorted-bary_sorted)
        phase = lcen_sorted - lcen_sorted.astype(int)-0.5
        ax.scatter(phase,diff,ls='-',s=2,label=f"iteration={it}")

ax.set_title(f'Order {od}')
ax.set_xlabel("Pixel phase ")
ax.set_ylabel(r"LSF $-$ baryentre"+r" (pix)")#r" (ms$^{-1}$)")
ax.legend(loc='center',bbox_to_anchor=(0.5,1.1), 
          fontsize=9, bbox_transform=ax.transAxes, ncol=3)
#%% 
plt.figure()
with FITS(outpath,'rw',clobber=False) as hdu:
    gauss_chisqnu=hdu['linelist'].read(columns='gauss_pix_chisqnu')[firstrow:lastrow]
    lsf_chisqnu=hdu['linelist',111].read(columns='lsf_pix_chisqnu')[firstrow:lastrow]
    for it in [211,311,411,511,611,711,811,911]:
        if it==211:
            lsf_chisqnu_itm1=lsf_chisqnu
        else:
            lsf_chisqnu_itm1=lsf_chisqnu_it
        index2 = linelist_it[cut]['id']
        sorter1,sorter2 = hcomp.get_sorted(index1, index2)
        
        lsf_chisqnu_it=hdu['linelist',it].read(columns='lsf_pix_chisqnu')[firstrow:lastrow]
        diff = lsf_chisqnu_itm1-lsf_chisqnu_it
        plt.scatter(lsf_chisqnu,diff,s=(it//100)*2,label=f"iteration={it}")
plt.legend()
#%% histogram of chisqnu

def plot_histogram(outpath,od,versions):
    plt.figure()
    with FITS(outpath,'rw',clobber=False) as hdu:
        linelist = hdu['linelist'].read()
        cut = np.where(linelist['order']==od)[0]
        gauss_chisqnu=linelist[cut]['gauss_pix_chisqnu']
        plt.hist(gauss_chisqnu,histtype='step',bins=50,label='Gauss')
        for i,it in enumerate(versions):
            linelist_it = hdu['linelist',it].read()
            cut_ = np.where(linelist_it['order']==od)[0]
            lsf_chisqnu=linelist_it[cut_]['lsf_pix_chisqnu']
            plt.hist(lsf_chisqnu,histtype='step',bins=50,label=f'iteration={it}',
                     lw=(i+1)*1.01,
                       range=(0,30)
                      )
    plt.legend()
od = 40
versions=[111,211,311,411,511,611,711,811,911]
versions=[111,211,311,411,511]
versions = [1]
# versions = [111,211,311]#,411,911]
plot_histogram(outpath,od,versions)
#%%
plt.figure()
with FITS(outpath,'rw',clobber=False) as hdu:
    linelist = hdu['linelist'].read()
    cut = np.where(linelist['order']==od)[0]
    gauss_width=linelist[cut]['gauss_pix'][:,2]
    plt.hist(gauss_width,histtype='step',bins=50,label='Gauss')
    for it in versions:#,211,311]:#,411,511,611,711,811,911]:
        linelist_it = hdu['linelist',it].read()
        cut_ = np.where(linelist_it['order']==od)[0]
        lsf_width=linelist_it[cut_]['lsf_pix'][:,2]
        plt.hist(lsf_width,histtype='step',bins=50,label=f'iteration={it}',
                 lw=(it//100)*1.05,
                 # range=(0,5)
                 )
plt.legend()
#%%
# versions = [201]
plt.figure()
with FITS(outpath,'rw',clobber=False) as hdu:
    linelist = hdu['linelist'].read()
    cut = np.where(linelist['order']==od)[0]
    gauss_cen=linelist[cut]['gauss_pix'][:,1]
    for it in versions: #,211,311]:#,411,511,611,711,811,911]:
        linelist_it = hdu['linelist',it].read()
        cut_ = np.where(linelist_it['order']==od)[0]
        lsf_cen=linelist_it[cut_]['lsf_pix'][:,1]
        plt.hist(gauss_cen-lsf_cen,histtype='step',bins=50,label=f'iteration={it}',
                 lw=(it//100)*1.05,
                 # range=(0,5)
                 )
plt.legend()
#%%
import harps.lsf.aux as aux
plt.figure()
segm = 8
segsize=4096//32
pixl=segm*segsize; pixr = (segm+1)*segsize

x2d=np.vstack([np.arange(4096) for i in range(72)])
versions = [111,211,311,411,511]

with FITS(outpath,'rw',clobber=False) as hdu:
    flx = hdu['flux'].read()
    bkg = hdu['background'].read()
    err = hdu['error'].read()
    for version in versions:
        linelist = hdu['linelist',version].read()
        cut = np.where(linelist['order']==od)
    
    
        for fittype in ['lsf']:
            pix3d,vel3d,flx3d,err3d,orders=aux.stack(fittype,linelist[cut],flx,
                                    x2d,err,bkg,orders=od)
            plt.errorbar(pix3d[od,pixl:pixr,0],flx3d[od,pixl:pixr,0],
                        err3d[od,pixl:pixr,0],marker='.',ls='',capsize=3,ms=3,
                        label=f'{version},{fittype}')
plt.legend()
#%%
import harps.lines_aux as laux
def plot_model(outpath,od,version):
    with FITS(outpath,'rw',clobber=False) as hdu:
        flx = hdu['flux'].read()
        bkg = hdu['background'].read()
        env = hdu['envelope'].read()
        err = hdu['error'].read()
        # modg = hdu['model_gauss'].read()
        mod = hdu['model_lsf',version].read()
    flx_norm, err_norm, bkg_norm = laux.prepare_data(flx, err, env, bkg,
                                                     subbkg=True,
                                                     divenv=True)
    fig, (ax1,ax2) = plt.subplots(2,sharex=True)
    ax1.set_title(f"Order: {od}, version: {version}")
    ax1.errorbar(np.arange(4096),flx_norm[od],err_norm[od],label='data',capsize=2,
                 drawstyle='steps-mid')
    ax1.plot(np.arange(4096),mod[od],label='model',drawstyle='steps-mid',marker='x')
    # ax1.plot(np.arange(4096),bkg[od],label='bkg',drawstyle='steps-mid')
    ax1.set_xlabel('Pixel'); ax1.set_ylabel("Flux"); ax1.legend()
    ax2.scatter(np.arange(4096),((flx_norm-mod)/err_norm)[od],s=4)
    ax2.set_xlabel('Pixel'); ax2.set_ylabel("Norm. rsd")
    ax2.set_ylim(-100,100)
plot_model(outpath,od=50,version=911)
