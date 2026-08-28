"""
The pixel-level forward model, in velocity space. A model pixel value is
the discretised convolution of the (point-like) LFC line with the LSF
phi(u), expressed as an integral over pixel offset with phi evaluated
after converting the pixel offset to velocity via the local scale v_pix:

    model(x_i) = integral  W_pix(x_i - x_line - tau) * phi(tau * v_pix) dtau

The Gaussian core is integrated exactly (Schmidt & Bouchy 2024, eq. 3, via
the error function) rather than approximated on a grid. The non-parametric
departure term still uses a fine velocity grid, projected through an
exact pixel/cell-overlap matrix (a flux-conserving rebin, not a sampled
kernel) rather than a Riemann sum.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.special import erf


def round_to_closest(a, b):
    if len(np.shape(a)) > 0:
        return np.array([round(v / b) * b for v in a])
    return round(a / b) * b


def round_down_to_odd(f, b=1):
    return round_to_closest(np.floor(f) // 2 * 2 + 1, b)


def get_half_window(y_axis, minimum=8):
    """ Half-window size (pixels) for each line's fitting window, from the
        dominant period of a Welch periodogram of the (envelope/
        background-corrected) flux -- picks up the LFC's own comb line
        spacing rather than a hand-picked constant. """
    if len(y_axis) == 0:
        return minimum
    freq0, P0 = welch(y_axis, nperseg=min(len(y_axis), 512))
    cut = np.where(freq0 > 0.02)[0]
    if len(cut) == 0:
        if len(freq0) > 1 and freq0[0] == 0:
            cut = [1]
        elif len(freq0) > 0 and freq0[0] > 0:
            cut = [0]
        else:
            return minimum
    freq, P = freq0[cut], P0[cut]
    maxind = np.argmax(P)
    maxfreq = freq[maxind]
    if maxfreq <= 1e-9:
        return minimum
    half_window = round_to_closest((1. / maxfreq) // 2, 1)
    return half_window if half_window > minimum else minimum


def gaussian_pixel_integral(pixel_indices, line_centre, sigma, v_pix):
    """ Exact analytic integral of the peak-normalised Gaussian LSF core
        over each pixel's boundaries (Schmidt & Bouchy 2024 eq. 3, adapted
        to a peak-normalised, not unit-area, profile). """
    Phi = lambda z: 0.5 * (1 + erf(z / np.sqrt(2)))
    edge_lo = (pixel_indices - 0.5 - line_centre) * v_pix
    edge_hi = (pixel_indices + 0.5 - line_centre) * v_pix
    return (sigma / v_pix) * np.sqrt(2 * np.pi) * (Phi(edge_hi / sigma) - Phi(edge_lo / sigma))


def convolution_matrix(u, du, line_centre, pixel_indices, v_pix):
    """ (len(pixel_indices) x n_grid) matrix mapping LSF-grid values to
        pixel-integrated flux, from the exact overlap between each pixel's
        boundary and each fine-grid cell [u_k-du/2, u_k+du/2] -- a flux-
        conserving rebin, not a sampled kernel. Used only for the non-
        parametric departure term; the Gaussian core needs no grid at all
        (see gaussian_pixel_integral). """
    p = u / v_pix
    dp = du / v_pix
    pixel_lo = pixel_indices[:, None] - line_centre - 0.5
    pixel_hi = pixel_indices[:, None] - line_centre + 0.5
    cell_lo = p[None, :] - dp / 2
    cell_hi = p[None, :] + dp / 2
    return np.clip(np.minimum(pixel_hi, cell_hi) - np.maximum(pixel_lo, cell_lo), 0, None)


def pixel_model_flux(u, du, line_centre, pixel_indices, v_pix, sigma, departure_grid):
    """ Full pixel-integrated model flux for one line: exact Gaussian core
        plus the departure term projected through convolution_matrix. """
    core = gaussian_pixel_integral(pixel_indices, line_centre, sigma, v_pix)
    conv = convolution_matrix(u, du, line_centre, pixel_indices, v_pix)
    return core + conv @ departure_grid


def make_fit_window(half_window: int, n_pixels: int):
    """ Returns a fit_window(line_centre) -> pixel-index array closure,
        bound to this order's half_window/n_pixels. """
    def fit_window(line_centre):
        lo = max(int(np.floor(line_centre)) - half_window, 0)
        hi = min(int(np.ceil(line_centre)) + half_window, n_pixels - 1)
        return np.arange(lo, hi + 1)
    return fit_window


def gaussian_mean(u_grid, sigma):
    """ Peak-normalised Gaussian core: value 1 at u=0 regardless of sigma.
        sigma is a velocity width (km/s). """
    return np.exp(-0.5 * (u_grid / sigma) ** 2)
