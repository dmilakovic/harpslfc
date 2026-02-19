#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:40:20 2018

@author: dmilakov
"""
#import matplotlib
#matplotlib.use('GTKAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
import harps.functions as hf
#from harps.lines import Linelist
#import harps.wavesol as ws
#from harps.classes import Manager, Spectrum
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import ColorbarBase
import matplotlib.colors as colors
# from matplotlib.colors import Normalize,LinearSegmentedColormap
from matplotlib import ticker
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
#------------------------------------------------------------------------------

#                                PLOTTER   

#------------------------------------------------------------------------------


class Figure(object):
    def __init__(self,nrows,ncols,width_ratios=None,height_ratios=None,title=None,
                 figsize=None,sharex=None,sharey=None,grid=None,subtitles=None,
                 enforce_figsize=False,left=0.1,right=0.95,top=0.95,
                 bottom=0.10,wspace=0.05,hspace=0.05, **kwargs):
        
        
        fig         = plt.figure(figsize=figsize)
        
        self._fig   = fig
        self._figsize = figsize
        
        self.nrows  = nrows
        self.ncols  = ncols
        self.col    = 0
        self.row    = 0
        
        
        self.top    = top
        self.bottom = bottom
        self.left   = left
        self.right  = right
        self.wspace = wspace
        self.hspace = hspace
        
#        self.alignment = alignment
        
        if enforce_figsize:
            fig.set_size_inches(figsize)
        
        gs  = GridSpec(nrows=nrows,ncols=ncols,
                       width_ratios=width_ratios, height_ratios=height_ratios,
                       left=left,right=right,top=top,bottom=bottom,
                       wspace=wspace,hspace=hspace)
        self._grid = gs
        self._axes = []
        self._axes_locations = []
        return
    
    def add_subplot(self,top,bottom,left,right,*args,**kwargs):
        ax = self.fig.add_subplot(self._grid[slice(top,bottom),
                                             slice(left,right)],*args,**kwargs)
        
        self._axes.append(ax)
        self._axes_locations.append([self.col,self.row])
        self.advance_by_one()
        return ax
    # def ax(self):
    #     col = self.col%self.ncols
    #     row = self.row%self.nrows
    #     print(col,row,self.ncols,self.nrows)
    #     # try adding a column
    #     if col==self.ncols and row==self.nrows:
    #         print("Impossible to add new subplots")
    #     if col==self.ncols-1:
    #         self.col=0
    #         self.row+=1
    #     if row==self.nrows-1:
    #         self.col+=1
    #         self.row=0
    #     if col<self.ncols and row<self.nrows:
    #         ax = self.add_subplot(row,row+1,col,col+1)
    #         print("Add subplot",row,row+1,col,col+1)
    #         self.col+=1
    #     return 
    def ax(self,*args,**kwargs):
        # print("Currently pointing to (row,row+1,col,col+1):",
               # self.row,self.row+1,self.col,self.col+1)
        return self.add_subplot(self.row,self.row+1,self.col,self.col+1,
                         *args,**kwargs)
    
    def advance_by_one(self):
        if self.col<self.ncols-1: # go to the next colum
            self.col+=1
        else: # go to the next row , reset column number 
            self.col = 0
            self.row += 1
    def is_at_right_edge(self,axnum):
        col,row = self._axes_locations[axnum]
        result = False
        if col%self.ncols==self.ncols-1:
            result = True
        return result
    def is_at_left_edge(self,axnum):
        col,row = self._axes_locations[axnum]
        result = False
        if col%self.ncols==0:
            result = True
        return result
    def is_at_top_edge(self,axnum):
        col,row = self._axes_locations[axnum]
        result = False
        if row%self.nrows==0:
            result = True
        return result
    def is_at_bottom_edge(self,axnum):
        col,row = self._axes_locations[axnum]
        result = False
        if row%self.nrows==self.nrows-1:
            result = True
        return result
    
        
        
    @property
    def fig(self):
        return self._fig
    @property
    def figure(self):
        return self._fig
    @property
    def axes(self):
        return self._axes
    @property
    def top_axes(self):
        is_top = [self.is_at_top_edge(i) for i in range(len(self.axes))]
        return self.axes[is_top]
    @property
    def bottom_axes(self):
        is_bottom = [self.is_at_bottom_edge(i) for i in range(len(self.axes))]
        return self.axes[is_bottom]
    @property
    def leftmost_axes(self):
        is_left = [self.is_at_left_edge(i) for i in range(len(self.axes))]
        return self.axes[is_left]
    @property
    def rightmost_axes(self):
        is_right = [self.is_at_righ_edge(i) for i in range(len(self.axes))]
        return self.axes[is_right]
    
    def save(self,path,*args,**kwargs):
        self.fig.savefig(path,*args,**kwargs)
        # logging.info(f'Figure saved to: {path}')
    def ticks(self,axnum,scale='x',ticknum=None,minval=None,maxval=None):
        ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
            ticks on a given scale (x or y)'''
        axis = self.axes[axnum]
        ticknum = ticknum if ticknum is not None else 4
        if minval is None or maxval is None:
            if scale == 'x':
                minval,maxval = axis.get_xlim()
            elif scale=='y':
                minval,maxval = axis.get_ylim()
            valrange = (maxval-minval)
            minval = minval + 0.05*valrange
            maxval = maxval - 0.05*valrange
        if scale=='x':
            axis.set_xticks(np.linspace(minval,maxval,ticknum))
        elif scale=='y':
            axis.set_yticks(np.linspace(minval,maxval,ticknum))
        return 

    def ticks_(self,which,axnum,axis='x',tick_every=None,ticknum=None):
        ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
            ticks on a given scale (x or y)'''
            
        ax   = self.axes[axnum]
        axsc = getattr(ax,'{0}axis'.format(axis))
        func = getattr(axsc,'set_{0}_locator'.format(which))
        
        ticknum = ticknum if ticknum is not None else 4
        if tick_every is not None:
            ax_ticker=ticker.MultipleLocator(tick_every)
        else:
            ax_ticker=ticker.MaxNLocator(ticknum)
        func(ax_ticker)
        return  
    def major_ticks(self,axnum,axis='x',tick_every=None,ticknum=None):
        ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
            ticks on a given scale (x or y)'''
        return self.ticks_('major',axnum,axis,tick_every,ticknum)
    def minor_ticks(self,axnum,axis='x',tick_every=None,ticknum=None):
        ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
            ticks on a given scale (x or y)'''
        return self.ticks_('minor',axnum,axis,tick_every,ticknum)
    def scinotate(self,axnum,axis,exp=None,dec=1,bracket='round'):
        ax   = self.axes[axnum]
        
        return scinotate(ax=ax, axis=axis, exp=exp, dec=dec,bracket=bracket)
        # axsc = getattr(ax,'{0}axis'.format(axis))
        
        # braleft = '[' 
        # brarigh = ']'
        # if bracket == 'round':
        #     braleft = '('
        #     brarigh = ')'
        
        # oldlbl = getattr(ax,'get_{0}label'.format(axis))()
        # loc    = oldlbl.find(brarigh)
        # axlim  = getattr(ax,'get_{0}lim'.format(axis))()
        # exp    = exp if exp is not None else np.floor(np.log10(axlim[1]))
        # axsc.set_major_formatter(ticker.FuncFormatter(lambda x,y : sciformat(x,y,exp,dec)))
        
        
        # if loc > 0:
        #     newlbl = oldlbl[:loc] + \
        #         r' $\times 10^{{{exp:0.0f}}}${br}'.format(exp=exp,br=brarigh)
        # else:
        #     newlbl = oldlbl + \
        #         r' {bl}$\times 10^{{{exp:.0f}}}${br}'.format(exp=exp,br=brarigh,bl=braleft)
        # print (newlbl)
        # set_lbl = getattr(ax,'set_{0}label'.format(axis))
        # set_lbl(newlbl)
        # return
    
Figure2 = Figure

def scinotate(ax,axis,exp=None,dec=1,bracket='round'):
    '''
    Args:
    ----
    ax (matplotlib.Axes instance)
    axis (str) : 'x' or 'y'
    exp (int) : exponential
    dec (int) : number of decimal points
    bracket (str) : 'round' or 'square'
    '''
    axsc = getattr(ax,'{0}axis'.format(axis))
    
    braleft = '[' 
    brarigh = ']'
    if bracket == 'round':
        braleft = '('
        brarigh = ')'
    
    oldlbl = getattr(ax,'get_{0}label'.format(axis))()
    loc    = oldlbl.find(brarigh)
    axlim  = getattr(ax,'get_{0}lim'.format(axis))()
    expmax = np.floor(np.log10(axlim[1]))
    # expmin = np.floor(np.log10(axlim[0]))
    exp    = exp if exp is not None else int(expmax)
    # if expmin==expmax:
    #     pass
    # else:
    #     exp = exp-1
    axsc.set_major_formatter(ticker.FuncFormatter(lambda x,y : sciformat(x,y,exp,dec)))
    
    
    if loc > 0:
        newlbl = oldlbl[:loc] + \
            r' $\times 10^{{{exp:0.0f}}}${br}'.format(exp=exp,br=brarigh)
    else:
        newlbl = oldlbl + \
            r' {bl}$\times 10^{{{exp:.0f}}}${br}'.format(exp=exp,br=brarigh,bl=braleft)
    set_lbl = getattr(ax,'set_{0}label'.format(axis))
    set_lbl(newlbl)
    return 
def ticks(ax,which,axis='x',tick_every=None,ticknum=None):
    ''' Makes ticks sparser on a given axis. Returns the axis with ticknum
        ticks on a given scale (x or y)
        
    Args:
    -----
        ax (plt.Axes instance)
        which (str): 'major' or 'minor'
        axis (str): 'x' or 'y'
        tick_every (int,float): distance between ticks
        ticknum (int) : number of ticks (incompatible with tick_every)
    '''
        
    
        
#        ax   = self.axes[axnum]
    axsc = getattr(ax,'{0}axis'.format(axis))
    func = getattr(axsc,'set_{0}_locator'.format(which))
    
    ticknum = ticknum if ticknum is not None else 4
    if tick_every is not None:
        ax_ticker=ticker.MultipleLocator(tick_every)
    else:
        ax_ticker=ticker.MaxNLocator(ticknum)
    func(ax_ticker)
    return  
#------------------------------------------------------------------------------

#                                PLOTTER   

#------------------------------------------------------------------------------
class SpectrumPlotter(object):
    def __init__(self,naxes=1,ratios=None,title=None,sep=0.05,figsize=(16,9),
                 alignment="vertical",sharex=None,sharey=None,**kwargs):
        fig, axes = hf.get_fig_axes(naxes,ratios=ratios,title=title,
                                 sep=sep,alignment=alignment,
                                 figsize=figsize,sharex=sharex,
                                 sharey=sharey,**kwargs)
        self.figure = fig
        self.axes   = axes   
    def show(self):
        self.figure.show()
        return


def sciformat(x,y,exp,dec):
    if x==0:
        return ('{num:.{width}f}'.format(num=x,width=dec))
    return ('{num:.{width}f}'.format(num=x/10**exp,width=dec))


# =============================================================================
#                         F  U  N  C  T  I  O  N  S
# =============================================================================

def ccd_from_linelist(linelist,desc,fittype='gauss',xscale='pix',mean=False,column=None,
                      label=None,yscale='wave',centre_estimate=None,
                      quantile=None, scale='linear',cmap='RdBu_r',
                      function=None,
                      *args,**kwargs):
    centre_colname = f'{fittype}_{xscale}'
    if mean:
        # NOTE: y is frequencies if yscale='wave'
        x, y, c = mean_val(linelist,
                           f'{desc}',
                           centre_colname,
                           column,
                           yscale)
        if yscale == 'wave': # convert Hz to nanometres
            y = hf.freq_to_lambda(y)/10
    else:
        if centre_estimate != 'bary':
            x = linelist[centre_colname][:,1]
        else:
            x = linelist['bary']
        if yscale == 'optord':
            y = linelist['optord']
        elif yscale == 'cenwav':
            y = get_cenwav(linelist)
        else:
            y = hf.freq_to_lambda(linelist['freq'])/10 # nanometres
            
        # "Colour" data
        if column is not None:
            c = linelist[desc][:,column]
        else:
            c = linelist[desc]
        if column==0:
            # normalise amplitudes by sigma
            # sigma = linelist[desc][:,2]
            # c = 1 / (sigma * np.sqrt(2*np.pi))
            # print('Normalised amplitude')
            pass
        if function is not None:
            assert callable(function)==True, 'Function must be callable.' 
            c = function(c)
    # if column is not None:
    #     c_hist = c#linelist[desc][:,column]
    # else:
    #     c_hist = c#linelist[desc]
    # print(mean,column,c.shape,c_hist.shape)
    label = label if label is not None else get_label(desc,column)  
    # override variable c with keyword 'value'
    values = kwargs.pop('values',c)
    values_hist = kwargs.pop('values_hist',values)
    
    
    return ccd(x,y,values,values_hist,label,yscale,centre_estimate=centre_estimate,
               quantile=quantile,scale=scale, cmap=cmap, *args,**kwargs)


# def clean_input(array):
#     # remove infinites, nans, zeros and outliers
#     arr = np.array([np.isfinite(x1s),
#                     np.isfinite(flx1s),
#                     np.isfinite(err1s),
#                     flx1s!=0
#                     ])
#     finite_ = np.logical_and.reduce(arr)
#     cut     = np.where(finite_)[0]

def ccd(x,y,c,c_hist=None, label=None, yscale='wave', bins=20, figsize=(6,6),
        centre_estimate=None, quantile=None, scale='linear', cmap=None,
        supress_colorbar_label=False, print_colorbar_label_above=True,
        cbar_label=None, plot_contours=False,
        *args,**kwargs):
    
      
    c_hist = c_hist if c_hist is not None else c
    
    cut = np.where(np.isfinite(c))[0]
    
    if len(cut)<len(c):
        print(len(c))
        x = x[cut]
        y = y[cut]
        c = c[cut]
        c_hist = c_hist[cut]
    
    
    # cmap = sc.get_cmap()
    minlim,maxlim = np.nanpercentile(c,[0.01,99.9])
    xrange = kwargs.pop('range',(minlim,maxlim))
    
    
    fig_kwargs = dict(
        figsize=figsize,
        left=0.13,top=0.92,right=0.9,bottom=0.09,
        vspace=0.2,hspace=0.05,
        height_ratios=[1,4],width_ratios=[30,1],
        )
    fig_kwargs.update(**kwargs)
    
    plotter = Figure2(nrows=2,ncols=2,**fig_kwargs)
    fig    = plotter.figure
    ax_top = plotter.add_subplot(0,1,0,1)
    ax_bot = plotter.add_subplot(1,2,0,1)
    ax_bar = plotter.add_subplot(1,2,1,2)
    
    if centre_estimate != 'bary':
        ax_bot.set_xlabel("Line centre (pix)")
    else:
        ax_bot.set_xlabel("Line centroid (pix)")
    if yscale == 'wave':
        ax_bot.set_ylabel(r"Wavelength (nm)")
    elif yscale == 'optord':
        ax_bot.set_ylabel("Optical order")
        ax_bot.invert_yaxis()
    elif yscale == 'cenwav':
        ax_bot.set_ylabel("Central order wavelength (nm)")
    
    log = False
    hist_label = label
    hist_range = xrange
    cbar_range = xrange
    if scale=='log':
        log = True
        try:
            label = r'$\log_{10}$('+label+')'
        except:
            pass
        # c_hist = np.log10(c_hist)
        
        minlim,maxlim = np.nanpercentile(c,[0.1,99])
        cbar_range = np.log10(cbar_range)
        c      = np.log10(c)
    
    cmap = cmap if cmap is not None else 'RdBu_r'
    cmap_min = kwargs.pop('cmap_min',None)
    cmap_max = kwargs.pop('cmap_max',None)
    cmap_mid = kwargs.pop('cmap_mid',None)
    
    
    vmin = cmap_min if cmap_min is not None else cbar_range[0]
    vmax = cmap_max if cmap_max is not None else cbar_range[1]
    vmid = cmap_mid if cmap_mid is not None else np.mean(cbar_range)
    # hist_range=(vmin,vmax)
    if cmap_min is not None or cmap_max is not None or cmap_mid is not None:
        def get_shift(vmid,vmin,vmax):
            return  1 - np.abs(vmax - vmid) / np.abs(vmax - vmin)
        
    if cmap_mid is not None:
        # cmap = shiftedColorMap(plt.get_cmap(cmap),
        #                         start=0.,
        #                         midpoint=get_shift(vmid,vmin,vmax),
        #                         stop=1.)
        cmap = shiftedColorMap(plt.get_cmap(cmap),
                                vmin=vmin,
                                vmid=vmid,
                                vmax=vmax)
    # cmap = cmap if cmap_norm is not None else
    print('cmap min, mid, max: ', cmap_min,cmap_mid,cmap_max)
        
    print('vmin, vmid, vmax: ', vmin, vmid, vmax)
        # cmap = 
        # cmap = shiftedColorMap(plt.get_cmap(cmap),
        #                        start=0.,
        #                        midpoint=midpoint,
        #                        stop=1.)
    if scale=='log':
        # norm = colors.LogNorm(vmin=vmin,vmax=vmax,clip=True)
        extend = 'max'
    else:
        extend = 'both'
    cbar_extend    = kwargs.pop('cbar_extend',extend)
    cmap_norm = kwargs.pop('cmap_norm',colors.Normalize(*cbar_range))
    
    sc = ax_bot.scatter(x,
                y,
                c=c,
                cmap=cmap,
                norm=cmap_norm,
                marker='s',s=16,rasterized=True)
    sc.set_clim(vmin,vmax)
    
    cb_label = label if not supress_colorbar_label else None
    cb_label = cbar_label if cbar_label is not None else cb_label
    cb1 = ColorbarBase(ax=ax_bar,norm=cmap_norm,label=cb_label,
                       cmap=sc.get_cmap(),extend=cbar_extend)
    if supress_colorbar_label and print_colorbar_label_above:
        ax_bar.text(1.5,1.15,label,rotation=90,
                    transform=ax_bar.transAxes,
                    horizontalalignment='left',
                    verticalalignment='center')
    bins = bins
    lw=2
    alpha=0.8
    ax_top.hist(c_hist,bins=bins,color='black',
        range=hist_range,
        histtype='step',density=False,
        lw=lw,
        # log=True
        )
    ax_top.set_ylabel(r"\# of lines")
    ax_top.set_xlabel(hist_label)
    ax_top.xaxis.tick_top()
    ax_top.xaxis.set_label_position('top') 
    
    if quantile is not None:
        for ax,array,label in zip([ax_top,ax_bar],
                                  [c_hist,c],
                                  ['hist','cbar']):
            val_q = np.quantile(array,quantile)
            print(f'Quantiles = {val_q}; {quantile} ({label})')
            for val in val_q:
                if label=='hist':
                    ax.axvline(val,ls='--',c='C1')
                elif label=='cbar':
                    ls='-' if val>=0 else (0,(1,1))
                    ax.axhline(val,ls=ls,c='k',lw=2.)
        plotter.major_ticks(0,'y',ticknum=3)
    if plot_contours:
        # import matplotlib.tri as tri
        # triang = tri.Triangulation(x, y)
        Xi, Yi, Ci = smooth_contour(x, y, c,
                grid_size=300,
                smooth_sigma=2.0  
            )
        levels = np.quantile(c,quantile) if quantile is not None else 5
        _ = ax_bot.contour(Xi, Yi, Ci, levels=levels, cmap='gray')
        
    fig.align_ylabels()
    
#    plotter.scinotate(0,'y',exp=3,dec=0)
#    plotter.major_ticks(0,'x',tick_every=25)#ticks(ax0,'x',5,0,4096)
#    plotter.minor_ticks(0,'x',tick_every=12.5)#ticks(ax0,'x',5,0,4096)
    
    plotter.major_ticks(1,'x',tick_every=1024)#ticks(ax0,'x',5,0,4096)
    plotter.minor_ticks(1,'x',tick_every=256)#ticks(ax0,'x',5,0,4096)
    if yscale=='cenwav':
      plotter.minor_ticks(1,'y',tick_every=5)#ticks(ax0,'x',5,0,4096)
    
    # plotter.major_ticks(0,'x',ticknum=5)
    # plotter.major_ticks(2,'y',ticknum=5)
    
    return plotter

def smooth_contour( x, y, c, grid_size=200, smooth_sigma=1.0, method='cubic'):
    # Create grid
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate
    Ci = griddata((x, y), c, (Xi, Yi), method=method)

    # Smooth (Gaussian filter)
    Ci_smooth = gaussian_filter(Ci, sigma=smooth_sigma)

    return Xi, Yi, Ci_smooth


def get_label(desc,column=None):
    label = desc
    if desc is not None:
        pass
    else:
        return ''
    if 'chisq' in desc:
        label = r'$\chi^2$'
    if 'chisqnu' in desc:
        label = r'$\chi_\nu^2$'
    else:
        if 'err' in desc:
            if column == 0:
                label = r'$\sigma_A$'
            elif column == 1:
                label = r'$\sigma_\mu$'
            elif column == 2:
                if 'lsf' in desc:
                    label = r'$\sigma_\omega$'
                elif 'gauss' in desc:
                    label = r'$\sigma_\sigma$'
        else:
            if column == 0:
                label = r'A'
            elif column == 1:
                label = r'$\mu$'
            elif column == 2:
                if 'lsf' in desc:
                    label = r'$\omega'
                elif 'gauss' in desc:
                    label = r'$\sigma$'
    return label
def mean_val(linelist,desc,fittype,column,yscale,values=None):
    xpositions   = []
    values      = []
    ypositions = []
    orders=np.unique(linelist['order'])
    for j,od in enumerate(orders):
        cutod  = np.where(linelist['order']==od)[0]
        mlod = linelist[cutod]
        #print(len(mlod.values))
        modes = np.unique(mlod['freq'])
        #print(len(modes))
        for f in modes:
            cut = np.where((linelist['order']==od)&(linelist['freq']==f))[0]
            #print(od,f,np.mean(linelist[cut][desc]))
            if values is not None:
                val = np.mean(values,axis=0)
            else:
                if column is not None:
                    val = np.mean(linelist[cut][desc][:,column],axis=0)
                else:
                    val = np.mean(linelist[cut][desc],axis=0)
            
            pos = np.mean(linelist[cut][fittype][:,1])
            xpositions.append(pos)
            if yscale=='wave':
                ypositions.append(f)
            else:
                ypositions.append(od)
            try:
                values.append(tuple(val))
            except:
                values.append(val)
        hf.update_progress((j+1)/len(orders),desc)
    return np.array(xpositions),np.array(ypositions),np.array(values)
def get_cenwav(linelist):
    orders = np.unique(linelist['order'])
    wav = hf.freq_to_lambda(linelist['freq'])/10.
    y = []
    for od in orders:
        cut = np.where(linelist['order']==od)[0]
        od_cwav = np.average(wav[cut])
        y.append(np.full_like(cut,od_cwav,dtype=np.float32))
    y = np.hstack(y)
    return y
#from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
def shiftedColorMap(cmap, vmin=0, vmid=0.5, vmax=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    def get_shift(x,vmin,vmax):
        return  1 - np.abs(vmax - x) / np.abs(vmax - vmin)
    
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    start = get_shift(vmin,vmin,vmax)
    stop  = get_shift(vmax,vmin,vmax)
    midpoint = get_shift(vmid,vmin,vmax)
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    # plt.register_cmap(cmap=newcmap)

    return newcmap
# class MidpointNormalize(colors.Normalize):
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         self.midpoint = midpoint
#         colors.Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip=None):
#         # I'm ignoring masked values and all kinds of edge cases to make a
#         # simple example...
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(value, x, y))
    
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, 
                 minpoint = 0.0,
                 midpoint = 0.5,
                 maxpoint = 1.0, clip=False):
        self.vcenter = vcenter
        self.midpoint = midpoint
        self.minpoint = minpoint
        self.maxpoint = maxpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [self.minpoint, self.midpoint, self.maxpoint]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [self.minpoint, self.midpoint, self.maxpoint]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)
def keep_center_colormap(vmin, vmax, center=0,cmap_default='RdBu_r'):
    vmin = vmin - center
    vmax = vmax - center
    dv = max(-vmin, vmax) * 2
    N = int(256 * dv / (vmax-vmin))
    RdBu_r = cm.get_cmap(cmap_default, N)
    newcolors = RdBu_r(np.linspace(0, 1, N))
    beg = int((dv / 2 + vmin)*N / dv)
    end = N - int((dv / 2 - vmax)*N / dv)
    newmap = colors.ListedColormap(newcolors[beg:end])
    return newmap