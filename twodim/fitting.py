#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fitting.py 

Created on Thu Apr 24 14:18:52 2025

@author: dmilakov
"""

import numpy as np

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta_rad, offset):
    """
    Calculates the value of a 2D Gaussian function.

    Args:
        xy (tuple): Tuple of (x, y) coordinate arrays (e.g., from np.meshgrid).
        amplitude (float): Amplitude of the Gaussian (above offset).
        xo (float): Center coordinate in X.
        yo (float): Center coordinate in Y.
        sigma_x (float): Standard deviation along the 'x' principal axis.
        sigma_y (float): Standard deviation along the 'y' principal axis.
        theta_rad (float): Rotation angle of the ellipse in radians
                           (angle of sigma_x axis wrt image x-axis).
        offset (float): Base offset value.

    Returns:
        np.ndarray: Flattened array of Gaussian values at the input coordinates.
    """
    (x, y) = xy
    xo = float(xo)
    yo = float(yo)
    # Ensure sigmas are reasonably positive to avoid math errors
    sigma_x = max(abs(float(sigma_x)), 1e-6)
    sigma_y = max(abs(float(sigma_y)), 1e-6)

    # Coefficients for the exponent quadratic form
    a = (np.cos(theta_rad)**2)/(2*sigma_x**2) + (np.sin(theta_rad)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta_rad))/(4*sigma_x**2) + (np.sin(2*theta_rad))/(4*sigma_y**2)
    c = (np.sin(theta_rad)**2)/(2*sigma_x**2) + (np.cos(theta_rad)**2)/(2*sigma_y**2)

    # Calculate Gaussian value
    exponent = - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2))
    g = offset + amplitude * np.exp(exponent)

    return g.ravel() # Return flattened array as expected by curve_fit