#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:56:48 2018

@author: dmilakov
"""
from harps.core import np
from harps.core import plt
import harps.functions.spectral as specfunc
import harps.settings as hs
import warnings
import harps.plotter as hplt

#from harps.functions import get_fig_axes

from scipy.optimize import leastsq, brentq, least_squares, OptimizeWarning
from scipy.optimize._lsq.least_squares import prepare_bounds
from scipy.linalg import svd
from scipy.special import erf, erfc, voigt_profile, dawsn
from scipy.signal import convolve

#import scipy.interpolate as interpolate


class EmissionLine(object):
    def __init__(self):#,xdata,ydata,yerr,absolute_sigma=False,bounds=None):
        ''' Initialize the object using measured data
        
        Args:
        ----
            xdata: 1d array of x-axis data (pixels or wavelengts)
            ydata: 1d array of y-axis data (electron counts)
            weights:  1d array of weights calculated using Bouchy method
            kind: 'emission' or 'absorption'
        '''
       
#        self.barycenter = np.sum(self.xdata*self.ydata)/np.sum(self.ydata)

    def _get_parameters(self):
        ''' Method to check whether the fit has been successfully performed.
        If the fit was successful, returns the fit values. Otherwise, the 
        method performs the fitting procedure and returns the fit values.
        Returns:
        -------
            pfit: tuple with fitted (amplitude, mean, sigma) values
        '''
        if self.success == True:
            params = self.pars
            errors = self.errs
        else:
            params = np.full(hs.npars, np.nan)
            errors = np.full(hs.npars, np.nan)
            pass
#            p0 = self._initialize_parameters()
#            pars, errors = self.fit(p0)
#            pfit = self.fit_parameters
#            errors = self.fit_errors
        return params, errors
    @property
    def params(self):
        return self._get_parameters()
    def _wrap_jac(self,jac,xdata=None,weights=None,*args):
        if xdata is None:
            xdata = self.xdata
        if weights is None:
            def jac_wrapped(xdata,params):
                return jac(xdata,*args)
        elif weights.ndim == 1:
            weights = self.weights
            def jac_wrapped(xdata,params):
                return weights[:, np.newaxis] * np.asarray(jac(xdata,*args))
        
        return jac_wrapped
    def residuals(self,pars,weights=None):
        ''' Returns the residuals of individual data points to the model.
        Args:
        ----
            pars: tuple (amplitude, mean, sigma) of the model
        Returns:
        -------
             1d array (len = len(xdata)) of residuals
        '''
#        model = self.model(*pars)
        if weights is not None:
            w = weights
        else:
            w = assign_weights(self.xdata, pars[1], self.scale)
        obsdata = self.ydata
        resids  = ((obsdata - self.model(pars))/self.yerr)*w
        return resids
    def chisq(self,pars=None,weights=None):
        ''' Returns the chi-square of data points to the model.
        Args:
        ----
            pars: tuple (amplitude, mean, sigma) of the model
        Returns:
        -------
            chisq
        '''
        if pars is None:
            pars = self._get_parameters()[0]
        return (self.residuals(pars)**2).sum() / self.dof
    def calc_R2(self,pars=None,weights=None):
        ''' Returns the R^2 estimator of goodness of fit to the model.
        Args:
        ----
            pars: tuple (A1, mu1, sigma1, fA, fm, sigma2) of the model
        Returns:
        -------
            chisq
        '''
        if pars is None:
            pars = self._get_parameters()[0]
        cdata = self.ydata
        weights = weights if weights is not None else self.weights
        SSR = 1 - np.sum(self.residuals(pars,weights)**2/np.std(cdata))
        SST = np.sum(weights*(cdata - np.mean(cdata))**2)
        rsq = 1 - SSR/SST
        return rsq
    
    def evaluate(self,pars=None,xdata=None):
        pars = pars if pars is not None else self._get_parameters()[0]
        xdata = xdata if xdata is not None else self.xdata
        return self.model(pars,xdata)

    def fit(self,xdata,ydata,error,p0=None,absolute_sigma=True, bounded=False,
            method=None, check_finite=True,full_output=False, **kwargs):
        ''' Performs the fitting of a Gaussian to the data. Acts as a wrapper 
        around the scipy.optimize `leastsq' function that minimizes the chisq 
        of the fit. 
        
        The function at each point is evaluated as an integral of the Gaussian 
        between the edges of the pixels (in case of wavelengths, boundary is 
        assumed to be in the midpoint between wavelength values). 
        
        The function calculates the fit parameters and the fit errors from the 
        covariance matrix provided by `leastsq'.
        
        Args:
        ----
            p0: tuple (amplitude, mean, sigma) with the initial guesses. 
                If None, is calculated from the data.
                
        Returns:
        -------
            pfit: tuple (amplitude, mean, sigma) of best fit parameters
            perror: tuple with errors on the best fit parameters
            
        Optional:
        --------
            return_full: Returns full output. Defaults to False
        '''
        
        def _unwrap_array_(array):
            try: 
                narray = array.values
            except:
                narray = array
            return narray
            
        scale = kwargs.pop('scale',None)
        if scale is not None:
            assert scale in ['pixel','velocity']
        else:
            diff = np.diff(xdata).astype(int)
            condition = np.all(diff==1)
            if condition:
                scale='pixel'
            else:
                scale='velocity'
            # if np.min(xdata)>4900:
            #     scale = 'velocity'
            # else:
            #     scale = 'pixel'
        self.scale = scale
        # print(self.scale)
        # if 'vel' in self.scale or 'wav' in self.scale:
        #     wave0 = np.average(xdata,weights=ydata)
        #     self.wave0 = wave0
        #     vel = specfunc.wave_to_velocity(xdata, wave0)*1e-3
        #     rescaled_xdata = _unwrap_array_(vel)
        # else:
        #     rescaled_xdata = xdata
        self.xdata = _unwrap_array_(xdata)
        
        # self.xdata       = _unwrap_array_(xdata)
        self.xbounds     = (self.xdata[:-1]+self.xdata[1:])/2
        self.ydata       = _unwrap_array_(ydata)

        self.yerr        = _unwrap_array_(error)
        
        self.success = None
        
        if p0 is None:
            p0 = self._initialize_parameters(xdata,ydata)
        p0 = np.atleast_1d(p0)
        n = p0.size  
        
        if bounded == True:
            try:
                bounds = self.bounds
            except:
                bounds = self._initialize_bounds()
                
        else:
            bounds=(-np.inf, np.inf)
        lb, ub = prepare_bounds(bounds, n)
        bounded_problem = np.any((lb > -np.inf) | (ub < np.inf))
        if method is None:
            if bounded_problem:
                method = 'trf'
            else:
                method = 'lm'
    
        if method == 'lm' and bounded_problem:
            raise ValueError("Method 'lm' only works for unconstrained problems. "
                             "Use 'trf' or 'dogbox' instead.")
        # NaNs can not be handled
        if check_finite:
            self.ydata = np.asarray_chkfinite(self.ydata)
        else:
            self.ydata = np.asarray(self.ydata)
    
        if isinstance(self.xdata, (list, tuple, np.ndarray)):
            # `xdata` is passed straight to the user-defined `f`, so allow
            # non-array_like `xdata`.
            if check_finite:
                self.xdata = np.asarray_chkfinite(self.xdata)
            else:
                self.xdata = np.asarray(self.xdata)
        if method != 'lm':
            #jac = '2-point'
            pass
        if method == 'lm':    
            wrapped_jac = self._wrap_jac(self.jacobian,self.xdata)
            #
            res = leastsq(self.residuals,p0,
                            Dfun=None,
                            # Dfun=wrapped_jac,
                          full_output=True,col_deriv=False,**kwargs)
            pfit, pcov, infodict, errmsg, ier = res
            #print(errmsg)
            cost = np.sum(infodict['fvec']**2)
            if ier not in [1, 2, 3, 4]:
                #raise RuntimeError("Optimal parameters not found: " + errmsg)
                pfit = np.full_like(p0,np.nan)
                pcov = None
                success = False
            else:
                success = True
        else:
            print(f'Bounded problem {method=:}, using jacobian')
            # print(p0, bounds)
            res = least_squares(self.residuals, p0, jac=self.jacobian,
                                bounds=bounds, method=method,
                                x_scale='jac',
                                **kwargs)
            # print(f'{res.success=:}')
            if not res.success:
                #raise RuntimeError("Optimal parameters not found: " + res.message)
                pfit = np.full_like(p0,np.nan)
                pcov = None
                success = False
                cost = np.inf
            else:
                cost = 2 * res.cost  # res.cost is half sum of squares!
                # print(f'{res.cost=:}')
                pfit = res.x
            
                success = res.success
                # Do Moore-Penrose inverse discarding zero singular values.
                _, s, VT = svd(res.jac, full_matrices=False)
                threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
                s = s[s > threshold]
                VT = VT[:s.size]
                pcov = np.dot(VT.T / s**2, VT)
            #return_full = False    
        warn_cov = False
#        absolute_sigma=False
        dof  = (len(self.ydata) - len(pfit))
                
        if pcov is None:
            # indeterminate covariance
            pcov = np.zeros((len(pfit), len(pfit)), dtype=float)
            pcov.fill(np.inf)
            warn_cov = True         
        elif not absolute_sigma:
            if len(self.ydata) > len(pfit):
                
                #s_sq = cost / (self.ydata.size - pfit.size)
                s_sq = cost / dof
                pcov = pcov * s_sq
            else:
                pcov.fill(self.inf)
                warn_cov = True
        if warn_cov:
            warnings.warn('Covariance of the parameters could not be estimated',
                      category=OptimizeWarning)    
        
        # From fit parameters, calculate the parameters of the two gaussians 
        # and corresponding errors
        if success == True:
            errors = [] 
            for i in range(len(pfit)):
                try:
                  errors.append(np.absolute(pcov[i][i])**0.5)
                except:
                  errors.append( 0.00 )
            # if the 
            # print('pfit',pfit)
            # if 'vel' in self.scale:
            #     vel_wrt_wav0 = pfit[1]
                
            #     wav_fit = specfunc.velocity_to_wave(vel_wrt_wav0, self.wave0)
            #     print(vel_wrt_wav0,wav_fit,self.wave0)
            #     wav_fit_err = self.wave0/299792458*errors[1]
            #     pfit[1] = wav_fit
            #     errors[1] = wav_fit_err
            self.pars = pfit
            self.errs = errors
            self.correlation_matrix = pcov / np.outer(errors, errors)
        else:
            errors = np.full_like(pfit,np.nan)
            self.pars = np.full_like(pfit,np.nan)
            self.errs = np.full_like(pfit,np.nan)
            self.correlation_matrix = np.full_like(pcov,np.inf)
            
        self.covar     = pcov
        self.rchi2     = cost / dof
        self.dof       = dof

        self.success = success
        self.cost = cost
        if full_output:
            if method == 'lm':
                return pfit, errors, infodict, errmsg, ier
            else: 
                return pfit, errors, res
        else:
            return pfit, errors
    
    def plot(self,fit=True,cofidence_intervals=True,ax=None,**kwargs):
        ''' Plots the line as a histogram of electron counts with corresponding
        errors. If `fit' flag is True, plots the result of the fitting 
        procedure.
        
        
        '''
        #import matplotlib.transforms as mtransforms
        if ax is None:
            # fig = plt.figure(figsize=(9,9))
            naxis = 1
            if fit==True:
                naxes = 2
            fig = hplt.Figure(naxes,1,height_ratios=[4,1],hspace=0.2)
            ax1 = fig.add_subplot(0,1,0,1)
            if fit==True:
                ax2 = fig.add_subplot(1,2,0,1,sharex=ax1)
            #     ax  = [ax1,ax2]
            # else:
            #     ax = [plt.subplot(111)]
            ax = fig.axes
            self.fig = fig.figure
        elif type(ax) == plt.Axes:
            ax = [ax]
        elif type(ax) == list:
            pass
        self.ax_list  = [ax]
        widths = np.diff(self.xdata)[:-1]
        # ax[0].bar(self.xdata,self.ydata,
        #           widths,align='center',alpha=0.3,color='C0')
        ax[0].errorbar(self.xdata,self.ydata,
                       yerr=self.yerr,fmt='o',color='C0')
        yeval = np.zeros_like(self.ydata)
        if fit is True:
            p_fit,pe_fit = self._get_parameters()
            p = kwargs.pop('params',p_fit)
            if np.any(~np.isfinite(p)):
                print("One or more parameter is not finite, cannot plot model.")
            else:
#            xeval = np.linspace(np.min(self.xdata),np.max(self.xdata),100)
                xeval = self.xdata
                yeval = self.evaluate(p)
               
                color = kwargs.pop('color','C1')
                label = kwargs.pop('label',None)
                ax[0].plot(xeval,yeval,color=color,marker='o',label=label)
                
                ax[1].plot(xeval,(yeval-self.ydata)/self.yerr,
                          ls='',marker='o')
                
                if cofidence_intervals is True:
        #            xeval = self.xdata
                    y,ylow,yhigh = self.confidence_band(confprob=0.05)
                    ax[0].fill_between(xeval,ylow,yhigh,alpha=0.5,color='C1')
                    y,ylow,yhigh = self.confidence_band(confprob=0.32)
                    ax[0].fill_between(xeval,ylow,yhigh,alpha=0.2,
                                      color='C1')
            ymax = np.max([1.2*np.percentile(yeval,95),1.2*np.max(self.ydata)])
        ax[0].set_ylabel('Flux [e-]')
        ax[1].axhline(0,ls='--',lw=0.5,c='k')
        ax[1].set_ylabel('Residuals [$\sigma$]')
        ax[1].set_xlabel('Pixel')
        ax[0].set_ylim(-np.percentile(yeval,20),ymax)
        ax[0].set_xlabel('Pixel')
        ax[0].set_ylabel('Counts')
        ax[0].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        return
    def confidence_band(self, confprob=0.05, absolute_sigma=False):
        
        # https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html \
                            #confidence-and-prediction-intervals
        from scipy.stats import t
        # Given the confidence probability confprob = 100(1-alpha)
        # we derive for alpha: alpha = 1 - confprob
        alpha = 1.0 - confprob
        prb = 1.0 - alpha/2
        tval = t.ppf(prb, self.dof) #degrees of freedom
                    
        C = self.covar
       
        p,pe = self._get_parameters()
        x = self.xdata
        n = len(p)              # Number of parameters from covariance matrix
        N = len(x)
        if absolute_sigma:
            covscale = 1.0
        else:
            covscale = self.rchi2 * self.dof
          
        y = self.evaluate(p)
        dfdp = self.jacobian(p).T
        
        df2 = np.zeros(N)
        for j in range(n):
            for k in range(n):
                df2 += dfdp[j]*dfdp[k]*C[j,k]
#        df2 = np.dot(np.dot(dfdp.T,self.covar),dfdp).sum(axis=1)
        df = np.sqrt(covscale*df2)
        delta = tval * df
        upperband = y + delta
        lowerband = y - delta
        return y, upperband, lowerband       
    def calculate_center_uncertainty(self,pfit=None,covar=None,N=200):
        ''' 
        Returns the standard deviation of centres drawn from a random sample.
        
        Draws N samples by randomly sampling the provided fit parameteres and
        corresponding errors to construct N models of the line. Line centre 
        is calculated for each of the N models.         
        '''
        pfit = pfit if pfit is not None else self.params[0] 
        C = covar if covar is not None else self.covar
        # It is not possible to calculate center uncertainty if the covariance 
        # matrix contains infinite values
        if np.isinf(C).any() == True:
            return -1
        else:
            pass
        mdgN  = np.random.multivariate_normal(mean=pfit,cov=C,size=N)
#        cut   = np.where(mdgN[:,3]>0)[0]
#        mdgN  = mdgN[cut]
        centers = np.zeros(N)
        for i,pars in enumerate(mdgN):
            pgauss_i    = pars
            centers[i]  = self.calculate_center(pgauss_i)
        return centers.std()    
class SingleGaussian(EmissionLine):
    '''Single gaussian model of an emission line, with error function.'''
    def model(self,pars,xdata=None,separate=False):
        ''' Calculates the expected electron counts by assuming:
            (1) The PSF is a Gaussian function,
            (2) The number of photons falling on each pixel is equal to the 
                integral of the PSF between the pixel edges. (In the case of 
                wavelengths, the edges are calculated as midpoints between
                the wavelength of each pixel.)
        
        The integral of a Gaussian between two points, x1 and x2, is calculated
        as:
            
            Phi(x1,x2) = A * sigma * sqrt(pi/2) * [erf(t2) - erf(t1)]
        
        Where A and sigma are the amplitude and the variance of a Gaussian, 
        and 't' is defined as:
            
            t = (x - mu)/(sqrt(2) * sigma)
        
        Here, mu is the mean of the Gaussian.
        '''
        xdata = xdata if xdata is not None else self.xdata[1:-1]
        xb  = (xdata[1:]+xdata[:-1])/2
        A, mu, sigma = pars
        e1  = erf((xb[:-1]-mu)/(np.sqrt(2)*sigma))
        e2  = erf((xb[1:] -mu)/(np.sqrt(2)*sigma))
        y   = A*sigma*np.sqrt(np.pi/2)*(e2-e1)
        
        return y
    
        
    def _initialize_parameters(self,xdata,ydata,npars=hs.npars):
        ''' Method to initialize parameters from data (pre-fit)
        Returns:
        ----
            p0: tuple with inital (amplitude, mean, sigma) values
        '''
        A0 = np.percentile(ydata,90)
        
        m0 = np.percentile(xdata,50)
        s0 = np.sqrt(np.var(xdata))/3
        
        if npars==5:
            p0 = (A0,m0,s0,0.,0.)
        elif npars==4:
            p0 = (A0,m0,s0,0.)
        elif npars==3:
            p0 = (A0,m0,s0)
        self.initial_parameters = p0
        return p0
    def _initialize_bounds(self):
        ''' Method to initialize bounds from data (pre-fit)
        Returns:
        ----
            (lb,ub): tuple with bounds on the fitting parameters
        '''

        bary = np.average(self.xdata,weights=self.ydata)
        lb = (0.5*np.max(self.ydata), bary-1., 0)
        ub = (1.5*np.max(self.ydata), bary+1., self.sigmabound)
        self.bounds = (lb,ub)
        return (lb,ub)
    
    def jacobian(self,pars):
        '''
        Returns the Jacobian matrix of the __fitting__ function. 
        In the case x0 and weights are not provided, uses inital values.
        '''
        # Be careful not to put gaussian parameters instead of fit parameters!
        A, mu, sigma = pars
        x = self.xdata[1:-1]
        y = self.evaluate(pars) 
        dfdp = np.array([y/A,
                         -y*(x-mu)/(sigma**2),
                         y*(x-mu)**2/(sigma**3)]).T
#        return weights[:,None]*dfdp
        return dfdp
    def calculate_center(self,pgauss=None):
        '''
        Returns the x coordinate of the line center.
        Calculates the line center by solving for CDF(x_center)=0.5
        '''
        pgauss = pgauss if pgauss is not None else self._get_gauss_parameters()[0]
        A,m,s = pgauss
        
        def eq(x):
            cdf =  0.5*erfc((m-x)/(s*np.sqrt(2)))
            return  cdf - 0.5
        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
        return x

        
class SimpleGaussian(EmissionLine):
    '''Single gaussian model of an emission line, without error function'''
    def model(self,pars,xdata=None,separate=False):
        ''' Calculates the expected electron counts by assuming:
            (1) The PSF is a Gaussian function,
            (2) The number of photons falling on each pixel is equal to the 
                value of the Gaussian in the center of the pixel.
        
        The value of the Gaussian is calculated as:
            
            Phi(x) = A * exp(-1/2*((x-mu)/sigma)**2)
        
        Where A, mu, and sigma are the amplitude, mean, and the variance 
        of the Gaussian.
        '''
        x  = xdata if xdata is not None else self.xdata 
        
        if len(pars)==5:
            A, mu, sigma, m, y0 = pars
        elif len(pars)==4:
            A, mu, sigma,y0 = pars
            m = 0
        elif len(pars)==3:
            A, mu, sigma = pars
            m = 0
            y0 = 0
        y   = np.abs(A)*np.exp(-0.5*(x-mu)**2/sigma**2)  + m*(x-mu) + y0
        # if 'vel' in self.scale:
            
        #     y   = np.abs(A)*np.exp(-0.5*(x-mu)**2/sigma**2) \
        #           * np.exp((-self.wave0/299792458)**2)
        # else:
        #     y   = np.abs(A)*np.exp(-0.5*(x-mu)**2/sigma**2)
        # y   = np.abs(A)*np.exp(-0.5*(x-mu)**2/sigma**2) + e0*x
        return y
        # return y
    def _fitpars_to_gausspars(self,pfit):
        '''
        Transforms fit parameteres into gaussian parameters.
        '''
        return pfit                
    def _initialize_parameters(self,xdata,ydata,npars=hs.npars):
        ''' Method to initialize parameters from data (pre-fit)
        Returns:
        ----
            p0: tuple with inital (amplitude, mean, sigma) values
        '''
        A0 = np.nanpercentile(ydata,99)
        
        m0 = np.average(xdata,weights=ydata)
        s0 = np.sqrt(np.var(xdata))/4
        if npars==5:
            p0 = (A0,m0,s0,0.,0.)
        elif npars==4:
            p0 = (A0,m0,s0,0.)
        elif npars==3:
            p0 = (A0,m0,s0)
        self.initial_parameters = p0
        return p0
    def _initialize_bounds(self):
        ''' Method to initialize bounds from data (pre-fit)
        Returns:
        ----
            (lb,ub): tuple with bounds on the fitting parameters
        '''

        bary = np.average(self.xdata,weights=self.ydata)
        lb = (0.1*np.max(self.ydata), bary-1., 0.1*np.std(self.xdata))
        ub = (2*np.max(self.ydata), bary+1., 1.5*np.std(self.xdata))
        self.bounds = (lb,ub)
        return (lb,ub)
    
    def jacobian(self,pars,xdata=None,yerr=None):
        '''
        Returns the Jacobian matrix of the __fitting__ function. 
        In the case x0 and weights are not provided, uses inital values.
        '''
        # Be careful not to put gaussian parameters instead of fit parameters!
        A, mu, sigma = pars
        if xdata is not None:
            x = xdata
        else:
            x = self.xdata
        if yerr is not None:
            err = yerr
        else:
            err = self.yerr
        x = x
        err = err
        y = A*np.exp(-0.5*(x-mu)**2/sigma**2)
        
        dfdp = np.stack([y/A,
                         y*(x-mu)/(sigma**2),
                         y*(x-mu)**2/(sigma**3)
                         ],axis=1)
        return dfdp/-err[:,np.newaxis]
    def calculate_center(self,pgauss=None):
        '''
        Returns the x coordinate of the line center.
        Calculates the line center by solving for CDF(x_center)=0.5
        '''
        pgauss = pgauss if pgauss is not None else self._get_gauss_parameters()[0]
        A,m,s = pgauss
        
        def eq(x):
            cdf =  0.5*erfc((m-x)/(s*np.sqrt(2)))
            return  cdf - 0.5
        x = brentq(eq,np.min(self.xdata),np.max(self.xdata))
        return x    
    
    
class CombLineVoigt(EmissionLine):
    def __init__(self):
        super().__init__()
        self.subbin_factor = 21  # Essential for precision [2]
        # Look-up tables for H_n and H_n,u are assumed globally available 
        # as per Webb & Lee [3, 4]
        
    def _initialize_parameters(self,xdata,ydata,npars=5):
        ''' Method to initialize parameters from data (pre-fit)
        Returns:
        ----
            p0: tuple with inital (amplitude, mean, sigma) values
        '''
        A0 = 10*np.max(self.ydata)
        
        mu_0 = np.average(xdata,weights=ydata)
        sigma_int0 = 2e-1
        gamma_int0 = 5e-1
        sigma_inst0 = np.sqrt(np.var(xdata))/4
        p0 = (A0, mu_0, sigma_int0, gamma_int0, sigma_inst0)
        self.initial_parameters = p0
        return p0

    def _initialize_bounds(self):
        """Sets strict limits on intrinsic width [5]."""
        bary = np.average(self.xdata, weights=self.ydata)
        # Parameters: (A, mu, sigma_int, gamma_int, sigma_inst)
        # Intrinsic widths limited to < 1 pixel as requested
        lb = (0.1 * np.max(self.ydata), bary - 1.0, 1e-6, 1e-6, 0.1)
        ub = (20.0 * np.max(self.ydata), bary + 1.0, 1.0, 2.0, 3.0)
        return (lb, ub)

    def model(self, pars, xdata=None):
        '''
        A Voigt profile convolved by a Gaussian instrumental profile. 
        Parameters are Amplitude (A), Center (μ), Intrinsic Gaussian width 
        (σ_int), Intrinsic Lorentzian width (γ_int), and instrumental 
        Gaussian width (σ_inst)
        
        ----------
        pars : TYPE
            DESCRIPTION.
        xdata : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        x = xdata if xdata is not None else self.xdata
        A, mu, s_int, g_int, s_inst = pars
        s_int = np.abs(s_int)
        g_int = np.abs(g_int)
        s_inst = np.abs(s_inst)
        
        # Sub-binning grid for accuracy [2]
        dx = np.diff(x).min() / self.subbin_factor
        x_fine = np.arange(x.min() - 5, x.max() + 5, dx)
        
        # 1. Intrinsic Voigt Profile (I_nu)
        H_sum = voigt_profile(x_fine - mu, s_int, g_int)
        # u = (x_fine - mu) / s_inst
        # a = g_int / s_int
        
        # H_sum   = sum([ (a**n) * get_Hn(n, u) for n in range(4) ])
        
        I_nu = A * H_sum
        
        # 2. Instrumental Gaussian Kernel (G)
        g_x = np.arange(-5 * s_inst, 5 * s_inst, dx)
        G = np.exp(-0.5 * (g_x / s_inst)**2)
        G /= G.sum() # Normalization [6]
        
        # 3. Convolution [1, 6]
        convolved = convolve(I_nu, G, mode='same')
        return np.interp(x, x_fine, convolved)

    def jacobian(self, pars, xdata=None, yerr=None):
        A, mu, s_int, g_int, s_inst = pars
        x = xdata if xdata is not None else self.xdata
        yerr = yerr if yerr is not None else self.yerr
        # 1. Setup Fine Grid for Convolution
        dx = np.diff(x).min() / self.subbin_factor
        x_fine = np.arange(x.min() - 5, x.max() + 5, dx)
        u = (x_fine - mu) / s_int
        a = g_int / s_int
    
        # 2. Compute Instrumental Kernel G and its Derivative
        g_x = np.arange(-5 * s_inst, 5 * s_inst, dx)
        G = np.exp(-0.5 * (g_x / s_inst)**2)
        G_norm = G / G.sum()
        dG_ds_inst = G_norm * (g_x**2 / s_inst**3 - 1/s_inst)
    
        # 3. Calculate Analytic Intrinsic Derivatives (from Look-up Tables/Taylor)
        # H_n(u) and H_n,u(u) are the Taylor expansion terms
        # H_sum   = sum([ (a**n) * get_Hn(n, u) for n in range(4) ])
        H_sum   = voigt_profile(x_fine - mu, s_int, g_int)
        Hu_sum  = sum([ (a**n) * get_Hnu(n, u) for n in range(4) ])
        Ha_sum  = sum([ n * (a**(n-1)) * get_Hn(n, u) for n in range(1, 4) ])
    
        I_nu = A * H_sum
    
        # Derivative w.r.t Amplitude (A)
        dI_dA = H_sum 
        
        # Derivative w.r.t Center (mu) - Eq 26 mapping
        # dI/dmu = A * (dH/du) * (du/dmu) where du/dmu = -1/s_int
        dI_dmu = - (A / s_int) * Hu_sum 
        
        # Derivative w.r.t Intrinsic Width (sigma_int) - Eq 27 mapping
        # dI_ds_int = (A / s_int) * (H_sum + u * Hu_sum + a * Ha_sum)
        dI_ds_int = (A / s_int) * ( - Ha_sum + mu / s_int * Hu_sum)
        
        # Derivative w.r.t Intrinsic Lorentzian (gamma_int) - Eq 29 mapping
        dI_dg_int = (A / s_int) * Ha_sum
        
    
        # 4. Apply Derivative Convolution Theorem [Eq 40]
        jac = [
            convolve(dI_dA,     G_norm, mode='same'),
            convolve(dI_dmu,    G_norm, mode='same'),
            convolve(dI_ds_int, G_norm, mode='same'),
            convolve(dI_dg_int, G_norm, mode='same'),
            convolve(I_nu,      dG_ds_inst, mode='same') # Kernel derivative
        ]
        
        # 5. Interpolate back to data grid and normalize by error [7]
        jac_matrix = np.array([np.interp(x, x_fine, j) for j in jac])
        # return np.transpose(jac_)
        # jac_matrix = np.stack(jac_, axis=1)
        # print(np.shape(jac_), np.shape(jac_matrix))
        # l2_norms = np.sqrt(np.sum(jac_matrix**2, axis=-1))
        # print(jac_matrix.shape)
        # print(l2_norms.shape)
        # print(np.max(jac_matrix,axis=1))
        # print(l2_norms)
        # avoid division by zero
        # l2_norms[l2_norms == 0] = 1.0
        jac_normalized = - jac_matrix.T / yerr [:, np.newaxis] 

        return jac_normalized
    
def get_Hn(n, u):
    """
    Returns the n-th Taylor series coefficient for the Voigt function H(a, u).
    Based on formulas in Equations (21) of the Webb & Lee paper.
    """
    exp_u2 = np.exp(-u**2)
    sqrt_pi = np.sqrt(np.pi)
    D = dawsn(u)
    
    if n == 0:
        return exp_u2
    elif n == 1:
        return -(2.0 / sqrt_pi) * (1.0 - 2.0 * u * D)
    elif n == 2:
        return (1.0 - 2.0 * u**2) * exp_u2
    elif n == 3:
        term = 1.0 - (u**2 / 3.0) - u * (1.0 - (2.0 * u**2 / 3.0)) * D
        return -(4.0 / sqrt_pi) * term
    else:
        # Higher order terms (n > 3) are generally negligible [1]
        return 0.0

def get_Hnu(n, u):
    """
    Returns the derivative of the n-th Taylor coefficient with respect to u.
    Based on formulas in Equations (30) of the Webb & Lee paper.
    """
    exp_u2 = np.exp(-u**2)
    sqrt_pi = np.sqrt(np.pi)
    D = dawsn(u)
    
    if n == 0:
        return -2.0 * u * exp_u2
    elif n == 1:
        return (4.0 / sqrt_pi) * (u + (1.0 - 2.0 * u**2) * D)
    elif n == 2:
        return (-6.0 * u + 4.0 * u**3) * exp_u2
    elif n == 3:
        term_poly = 5.0 * u - 2.0 * u**3
        term_dawson = (3.0 - 12.0 * u**2 + 4.0 * u**4) * D
        return (4.0 / (3.0 * sqrt_pi)) * (term_poly + term_dawson)
    else:
        return 0.0
    
def get_binlimits(xarray,center,scale):
    if scale[:3]=='pix':
        dx = np.array([-5,-2.5,2.5,5]) # units pix
        binlims = dx + center
    elif scale[:3]=='vel':
        # varray = (xarray-center)/center * 299792.458 # units km/s
        dv     = np.array([-4,-2,2,4]) # units km/s
        binlims = center * (1 + dv/299792.458) # units wavelength
    return binlims

def assign_weights(xarray,center,scale):
    def f(x,x1,x2): 
        # a linear function going through x1 and x2
        return np.abs((x-x1)/(x2-x1))
    
    weights  = np.zeros_like(xarray,dtype=np.float64)
    binlims = get_binlimits(xarray, center, scale)
        
    idx      = np.digitize(xarray,binlims)
    cut1     = np.where(idx==2)[0]
    cutl     = np.where(idx==1)[0]
    cutr     = np.where(idx==3)[0]
    # ---------------
    # weights are:
    #  = 1,           -2.5<=x<=2.5
    #  = 0,           -5.0>=x & x>=5.0
    #  = linear[0-1]  -5.0<x<-2.5 & 2.5>x>5.0 
    # ---------------
    weights[cutl] = f(xarray[cutl],binlims[0],binlims[1])
    weights[cutr] = f(xarray[cutr],binlims[3],binlims[2])
    weights[cut1] = 1
    return weights    
        