"""
Empirical line-spread function (LSF) and wavelength-solution reconstruction
from a laser frequency comb (LFC) exposure.

An LFC produces a series of narrow, evenly-spaced emission lines at known
laboratory frequencies. Because the true frequency of every line is known,
an LFC exposure lets us solve simultaneously for three things that are
normally hard to separate:

  1. The smooth throughput of the spectrograph along the order (the
     envelope and background level).
  2. The mapping between pixel position and wavelength (the dispersion, or
     wavelength calibration, relation).
  3. The instrumental line-spread function itself: the shape each
     intrinsically point-like LFC line is smeared into by the spectrograph
     optics, and how that shape changes across the order.

These three things are estimated in that order, but not independently:
the envelope and background are estimated first, directly from the raw
data. The LSF shape, its width, and the wavelength solution are then
refined together in a repeating cycle, because each of the three depends
on the current estimate of the other two.

THE LSF IS DEFINED IN VELOCITY, NOT PIXELS. A spectrograph's resolving
power is fundamentally a velocity width (R = lambda/delta_lambda =
c/delta_v), and pixel width only reflects that after being distorted by
the local dispersion scale (km/s per pixel) -- which is already known to
vary smoothly across an order. Expressing the LSF directly in velocity
removes that distortion: whatever width variation remains after the
conversion is the genuine instrumental effect, not an artefact of the
pixel scale changing. The detector is still pixel-based, so the forward
model still needs to convert between a pixel offset and the equivalent
velocity offset; that conversion uses the LOCAL velocity-per-pixel scale,
computed directly from the dispersion solution's own derivative at each
line's exactly-known wavelength (see section 3).
"""

import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial import chebyshev as Chebyshev

from lfc.fitting.gp import gaussian_process_smooth

SPEED_OF_LIGHT = 2.99792458e8       # m / s
C_LIGHT_KMS = SPEED_OF_LIGHT / 1e3  # km / s

SPECTRUM_FILE = '/Users/dmilakov/software/harps/lsf/test/example_data_ESPRESSO_od=50.txt'
LINES_FILE = '/Users/dmilakov/software/harps/lsf/test/line_positions_ESPRESSO_od=50.txt'


# =========================================================================
# 1. Load the exposure and the LFC line list
# =========================================================================
spectrum = np.loadtxt(SPECTRUM_FILE, comments='#')
flux_raw, err_raw, _bkg_col, _env_col = spectrum.T
n_pixels = len(flux_raw)
pixel = np.arange(n_pixels)

line_list = np.loadtxt(LINES_FILE, comments='#')
left_edge, peak_pixel, right_edge, frequency = line_list[:, 1], line_list[:, 2], \
                                                 line_list[:, 3], line_list[:, 4]
n_lines = len(peak_pixel)
wavelength = SPEED_OF_LIGHT / frequency * 1e9  # vacuum wavelength, nm

print(f"{n_pixels} pixels, {n_lines} LFC lines")

x_min, x_max = peak_pixel.min(), peak_pixel.max()

def rescaled_position(x):
    """ Pixel position rescaled to [-1, 1] over the range spanned by the
        LFC lines. Defined this early (rather than where it's first used
        for the Chebyshev position-dependence models further down) because
        the background design matrix below also needs it: raw pixel
        position raised to the 4th power, multiplied by envelope values of
        order 1e5-1e6, produces columns spanning up to ~21 orders of
        magnitude -- far beyond double precision's ~16 significant digits.
        np.linalg.lstsq's SVD-based solver tolerates this reasonably well,
        but solving the normal equations directly (needed to get the
        coefficients' covariance matrix, not just the coefficients
        themselves) does not; confirmed directly, it produced a fit that
        failed to converge and a background constant term many orders of
        magnitude off from the essentially-zero value every previous
        version of this script found. Rescaling first keeps every column
        bounded to order unity regardless of polynomial degree. """
    return 2 * (x - x_min) / (x_max - x_min) - 1



# =========================================================================
# 2. Envelope and background (unchanged: estimated directly from the raw
#    pixel data, before any LSF or wavelength-solution model is assumed,
#    and entirely in pixel/flux units -- this stage has no velocity
#    dependence at all)
# =========================================================================
def local_extremum(centres, kind, half_width=2):
    values = np.empty(len(centres))
    for i, c in enumerate(centres):
        lo = max(int(round(c)) - half_width, 0)
        hi = min(int(round(c)) + half_width + 1, n_pixels)
        values[i] = flux_raw[lo:hi].max() if kind == 'max' else flux_raw[lo:hi].min()
    return values

def local_peak_subpixel(centres, half_width=2):
    """ Sub-pixel corrected peak flux via 3-point parabolic interpolation
        around the discrete maximum near each centre -- see the extended
        discussion elsewhere: using the raw discrete maximum directly
        creates a circular normalisation (the envelope is fit through
        these same values, and every pixel's flux is later divided by the
        envelope evaluated at that pixel, including the pixel that defined
        it), which artificially pins each line's brightest-pixel flux
        close to 1 regardless of its true sub-pixel phase. """
    values = np.empty(len(centres))
    for i, c in enumerate(centres):
        lo = max(int(round(c)) - half_width, 0)
        hi = min(int(round(c)) + half_width + 1, n_pixels)
        window = flux_raw[lo:hi]
        i_max = np.argmax(window)
        if i_max == 0 or i_max == len(window) - 1:
            values[i] = window[i_max]
            continue
        y0, y1, y2 = window[i_max - 1], window[i_max], window[i_max + 1]
        denominator = y2 - 2 * y1 + y0
        values[i] = y1 if denominator >= 0 else y1 - (y2 - y0)**2 / (8 * denominator)
    return values

peak_flux = local_peak_subpixel(peak_pixel)
boundary_pixel = np.unique(np.concatenate([left_edge, right_edge]))
boundary_flux = local_extremum(boundary_pixel, 'min')
peak_flux_err = err_raw[np.round(peak_pixel).astype(int)]
boundary_flux_err = err_raw[np.round(boundary_pixel).astype(int)]

BACKGROUND_POLY_ORDER = 4
ENVELOPE_MIN_LENGTH_SCALE = 1000  # pixels; raise this to force a smoother
                                    # envelope than the marginal likelihood
                                    # would otherwise choose on its own
                                    # (see gp.py's gaussian_process_smooth)
background_coeffs = np.zeros(BACKGROUND_POLY_ORDER + 1)
background_coeffs[0] = np.median(boundary_flux)

def background_design_row(x, e):
    """ [1, x~*e, x~^2*e, ..., x~^P*e] -- the row of the background design
        matrix at pixel(s) x, given envelope value(s) e there, using
        RESCALED position x~ (see the note on rescaled_position above for
        why raw pixel position is not used directly here). Used both to
        fit background_coeffs and, being d(background)/d(coeffs), to
        propagate the coefficients' own fit uncertainty into flux_err
        below (see the extended discussion there). """
    x_tilde = rescaled_position(x)
    return np.column_stack([np.ones_like(x_tilde)] +
                            [x_tilde**p * e for p in range(1, BACKGROUND_POLY_ORDER + 1)])

for iteration in range(4):
    gain = sum(background_coeffs[p] * rescaled_position(boundary_pixel)**p
               for p in range(1, BACKGROUND_POLY_ORDER + 1))
    gain = np.where(np.abs(gain) < 1e-8, np.nan, gain)
    envelope_from_boundary = (boundary_flux - background_coeffs[0]) / gain
    envelope_from_boundary_err = np.abs(boundary_flux_err / gain)
    usable = np.isfinite(envelope_from_boundary) & np.isfinite(envelope_from_boundary_err)

    regression_x = np.concatenate([peak_pixel, boundary_pixel[usable]])
    regression_z = np.concatenate([peak_flux, envelope_from_boundary[usable]])
    regression_err = np.concatenate([peak_flux_err, envelope_from_boundary_err[usable]])
    order = np.argsort(regression_x)
    regression_x, regression_z, regression_err = (
        regression_x[order], regression_z[order], regression_err[order])

    # ENVELOPE_MIN_LENGTH_SCALE: raises the floor on how quickly the
    # envelope is allowed to vary, in pixels. Without this, the length
    # scale is whatever the marginal likelihood happens to prefer (e.g.
    # ~324 pixels in earlier runs) -- raise this value directly to force
    # a smoother envelope.
    gp_fit = gaussian_process_smooth(regression_x, regression_z, regression_err,
                                      pixel.astype(float), n_restarts=3,
                                      min_length_scale=ENVELOPE_MIN_LENGTH_SCALE)
    envelope_grid = gp_fit['z_mean']
    # The GP's own posterior uncertainty on the envelope, at every pixel --
    # this already correctly propagates the input peak/boundary flux
    # measurement errors through the GP fit. Discarding it (as the
    # previous version of this script did, keeping only z_mean) throws
    # away exactly the quantity needed to propagate the envelope's own
    # uncertainty into flux_err below.
    envelope_std_grid = gp_fit['z_std']

    def envelope(x, _grid=envelope_grid):
        return np.interp(x, pixel.astype(float), _grid)

    def envelope_std(x, _grid=envelope_std_grid):
        return np.interp(x, pixel.astype(float), _grid)

    envelope_at_boundary = envelope(boundary_pixel)
    design = background_design_row(boundary_pixel, envelope_at_boundary)
    # Weighted least squares, using the boundary flux measurements' own
    # known errors -- previously this was a plain (unweighted) lstsq,
    # which does not use boundary_flux_err at all and gives no way to
    # obtain a meaningful coefficient covariance. The weighted normal
    # equations directly give both the fit AND, as (X^T W X)^-1, the
    # coefficients' covariance matrix, needed below.
    xtw = design.T * (1.0 / boundary_flux_err**2)
    normal_matrix = xtw @ design
    new_coeffs = np.linalg.solve(normal_matrix, xtw @ boundary_flux)
    background_coeffs_covariance = np.linalg.inv(normal_matrix)
    max_change = np.max(np.abs(new_coeffs - background_coeffs))
    background_coeffs = new_coeffs
    print(f"  envelope/background iteration {iteration}: "
          f"GP length scale = {gp_fit['length_scale']:.1f} pix, "
          f"max coefficient change = {max_change:.3g}")

print(f"background(x) = {background_coeffs[0]:.1f} + "
      f"polynomial(degree {BACKGROUND_POLY_ORDER}) * envelope(x), "
      f"coefficients = {np.round(background_coeffs, 6)}")

# ---------------------------------------------------------------------
# Independent residual term for the background, on top of the coupled
# c_0 + poly(x)*E(x) term above.
#
# The coupled term is kept exactly as before: it is the physically
# motivated part, encoding the idea that the envelope and background are
# shaped by much of the same underlying process (here, understood as the
# LFC's photonic-crystal-fibre dispersion and electronics, which need not
# affect the peak and inter-line flux identically, but plausibly share a
# common smooth trend). That physical picture does not require B to be
# EXACTLY a smooth multiple of E, though -- only that the two are
# related. Forcing B to be exactly c_0 + poly(x)*E(x), with poly(x) a
# smooth low-degree gain, means B can only ever be as locally-structured
# as E itself is: once E was deliberately smoothed (ENVELOPE_MIN_LENGTH_
# SCALE), B lost the ability to track any genuine local structure of its
# own, since there was nothing left in E's shape for the gain factor to
# modulate. This residual term restores that ability directly: it is a
# SEPARATE GP fit to whatever the coupled term does not explain in the
# boundary measurements, with its own (much shorter, freely-learned)
# length scale, uncoupled from the envelope's smoothness entirely.
BACKGROUND_RESIDUAL_MIN_LENGTH_SCALE = 20  # pixels; modest floor mainly to
                                             # stop the residual from
                                             # chasing per-point noise --
                                             # much shorter than the
                                             # envelope's own 1000-pixel
                                             # floor, deliberately, since
                                             # this term's entire purpose
                                             # is to capture what the
                                             # smooth coupled term cannot
coupled_prediction_at_boundary = background_design_row(
    boundary_pixel, envelope(boundary_pixel)) @ background_coeffs
boundary_residual = boundary_flux - coupled_prediction_at_boundary

residual_gp_fit = gaussian_process_smooth(
    boundary_pixel, boundary_residual, boundary_flux_err, pixel.astype(float),
    n_restarts=3, min_length_scale=BACKGROUND_RESIDUAL_MIN_LENGTH_SCALE)
background_residual_grid = residual_gp_fit['z_mean']
background_residual_std_grid = residual_gp_fit['z_std']
print(f"  background residual: GP length scale = {residual_gp_fit['length_scale']:.1f} pix, "
      f"signal std = {residual_gp_fit['signal_std']:.1f}")

def background_residual(x):
    return np.interp(x, pixel.astype(float), background_residual_grid)

def background_residual_std(x):
    return np.interp(x, pixel.astype(float), background_residual_std_grid)

def background(x):
    e = envelope(x)
    return background_design_row(x, e) @ background_coeffs + background_residual(x)

envelope_grid_full = envelope(pixel.astype(float))
background_grid_full = background(pixel.astype(float))

# ---------------------------------------------------------------------
# Propagating the envelope's and background's own estimation uncertainty
# into flux_err, not just the raw per-pixel measurement error err_raw.
#
# flux = (F - B) / (E - B), with F = flux_raw (known variance err_raw^2)
# and E, B themselves uncertain -- and NOT independent, since
# B(x) = c_0 + poly(x)*E(x) is built directly from E(x). Writing
# N = F-B, D = E-B, the needed partial derivatives are:
#
#   dD/dE = 1 - poly(x),      dD/dc_p = -x^p * E(x)
#   dN/dE = -poly(x),         dN/dc_p = -x^p * E(x)     (dN/dF = 1)
#
# where poly(x) = sum_p c_p * x^p is the background's own "gain" factor
# (already computed as an intermediate quantity elsewhere in this
# script), and the c_p partial derivatives are exactly the background
# design-matrix row -- no new computation needed, since that row is what
# background_design_row already returns.
#
# Treating the envelope's own GP uncertainty and the background
# coefficients' fit uncertainty as independent sources (a standard,
# defensible simplification -- a fully joint treatment of how the ALS
# iteration couples them would be considerably more complex for what is
# very likely a second-order correction on top of a second-order
# correction), the variances and cross-covariance of N and D follow by
# the standard multivariate delta method, and Var(flux) follows from the
# standard ratio-of-correlated-variables formula.
poly_gain = sum(background_coeffs[p] * rescaled_position(pixel.astype(float))**p
                for p in range(1, BACKGROUND_POLY_ORDER + 1))
sigma_E = envelope_std(pixel.astype(float))
design_row_full = background_design_row(pixel.astype(float), envelope_grid_full)
coeff_variance_term = np.einsum('ij,jk,ik->i', design_row_full,
                                 background_coeffs_covariance, design_row_full)
# The residual term d_B(x) added above contributes its own, independent
# uncertainty to B: it enters additively (dB/d(d_B) = 1) and is fit
# separately from E and the coupled coefficients, so its variance simply
# adds on top of the two terms already accounted for below.
residual_variance_term = background_residual_std(pixel.astype(float))**2

var_D = (1 - poly_gain)**2 * sigma_E**2 + coeff_variance_term + residual_variance_term
var_N = err_raw**2 + poly_gain**2 * sigma_E**2 + coeff_variance_term + residual_variance_term
cov_ND = poly_gain * (poly_gain - 1) * sigma_E**2 + coeff_variance_term + residual_variance_term

N_full = flux_raw - background_grid_full
D_full = envelope_grid_full - background_grid_full
flux = N_full / D_full
flux_err = np.abs(flux) * np.sqrt(
    np.maximum(var_N / N_full**2 + var_D / D_full**2 - 2 * cov_ND / (N_full * D_full), 0.0))
inverse_variance = 1.0 / flux_err**2


# =========================================================================
# 3. Wavelength calibration model
# =========================================================================
# x(lambda) is represented directly by its value at the M known comb
# wavelengths, line_position, fit via gp.py's gaussian_process_smooth --
# not a polynomial, and not a custom kernel. This reuses the SAME utility
# already used for the envelope and background above, rather than
# building separate GP machinery just for this.
#
# gaussian_process_smooth needs (x, z, z_err) triples: here x=wavelength,
# z=pixel position. Position enters the pixel-level forward model
# NONLINEARLY (through the convolution), so those (position, uncertainty)
# values are not directly observed -- they come from linearising the
# model around the current position estimate at each outer iteration
# (the same Gauss-Newton idea used elsewhere in this script), reducing
# each line's own pixel window to a single scalar correction with a
# formal uncertainty, then letting the GP smooth those corrections across
# wavelength. This is standard practice in real wavelength-calibration
# pipelines (per-line centroid, then a global smooth fit through them);
# what keeps it consistent with "position is determined by wavelength,
# not independently observable" is that it is a refinement step repeated
# inside an iteration, not a one-shot measurement taken as ground truth.
#
# No custom kernel or inducing-point scheme is needed here (unlike an
# earlier version of this project's LSF-shape GP, which did need one): a
# dense fit over ~400 points is exactly what gaussian_process_smooth
# already does successfully for the envelope and background above, at a
# similar number of points, without the ill-conditioning that motivated
# inducing points elsewhere.
#
# velocity_per_pixel needs a LOCAL DERIVATIVE of x(lambda), which would
# otherwise mean re-running the whole GP fit (sigma-clipping, multi-
# restart hyperparameter search) for every tiny perturbation -- far too
# slow. Computed instead directly from the already-fitted (wavelength,
# line_position) pairs via a discrete derivative across neighbouring
# lines, avoiding any need to evaluate the GP off-grid at all.

lambda_min, lambda_max = wavelength.min(), wavelength.max()
lambda_span = lambda_max - lambda_min
_wavelength_order = np.argsort(wavelength)

def velocity_per_pixel_from_positions(position):
    """ Local km/s-per-pixel scale at every comb line's own wavelength,
        from the CURRENT position array, via a discrete derivative across
        neighbouring lines (sorted by wavelength) -- not fit, computed
        directly from whatever the current best position estimate is. """
    lam_sorted = wavelength[_wavelength_order]
    pos_sorted = position[_wavelength_order]
    dx_dlambda_sorted = np.gradient(pos_sorted, lam_sorted)
    v_pix_sorted = C_LIGHT_KMS / dx_dlambda_sorted / lam_sorted
    v_pix = np.empty_like(v_pix_sorted)
    v_pix[_wavelength_order] = v_pix_sorted
    return v_pix


# =========================================================================
# 4. Line-spread function model, in VELOCITY space
# =========================================================================
# The LSF is represented as a function phi(v) of Doppler velocity v (km/s)
# from a line's centre, spaced at a fixed resolution and extending +/- a
# fixed range -- generous enough to cover every pixel that will later be
# included in any line's fit window, for the largest velocity-per-pixel
# scale actually present in this order.
#
# A model pixel value is obtained by discretising the convolution of the
# (essentially point-like) LFC line with phi, expressed as an integral
# over PIXEL offset (since the detector's own anti-aliasing/pixel
# response is a pixel-space phenomenon) with phi evaluated after
# converting that pixel offset to velocity via the local scale v_pix:
#
#   model(x_i) = integral  W_pix(x_i - x_line - tau) * phi(tau * v_pix) dtau
#
# Discretising this directly on a PIXEL grid, as done previously, would
# tie the LSF's own resolution to the pixel grid; instead the change of
# variable v = tau * v_pix turns this into a sum over a FIXED VELOCITY
# grid u, with the Jacobian factor du/v_pix appearing in the discretised
# weights below.

# A representative, ORDER-AVERAGED velocity-per-pixel scale, used only to
# set a sensible grid range and resolution below -- the actual forward
# model always uses each line's own local value (velocity_per_pixel_from_
# positions above, evaluated from the current position estimate).
_v_per_pixel_typical = (C_LIGHT_KMS * (lambda_max - lambda_min)
                         / (x_max - x_min) / np.mean(wavelength))
print(f"typical velocity scale: {_v_per_pixel_typical:.4f} km/s per pixel")

PIXEL_SUBSAMPLE = 11    # native subpixel resolution, carried over in spirit:
                         # the velocity grid below gets the same NUMBER of
                         # points per typical pixel as the old pixel grid did
HALF_WINDOW = 6          # still in PIXELS -- how many detector pixels around
                          # each line's centre go into its fitting window
_SAFETY_MARGIN = 1.5     # grid range set wider than the typical scale alone
                          # would need, so it still covers the fitting window
                          # wherever the local scale is larger than typical

VELOCITY_HALF_RANGE = HALF_WINDOW * _v_per_pixel_typical * _SAFETY_MARGIN  # km/s
N_VELOCITY_GRID = HALF_WINDOW * PIXEL_SUBSAMPLE * 2 + 1
u = np.linspace(-VELOCITY_HALF_RANGE, VELOCITY_HALF_RANGE, N_VELOCITY_GRID)  # km/s
n_grid = len(u)
du = u[1] - u[0]  # km/s
ANTIALIAS_WIDTH = 0.15  # PIXELS -- unchanged, and deliberately not converted:
                          # this represents the detector's own pixel-response
                          # behaviour, a pixel-space phenomenon regardless of
                          # what units the LSF itself is expressed in

def gaussian(x, width):
    return np.exp(-0.5 * (x / width)**2) / (np.sqrt(2 * np.pi) * width)

def convolution_matrix(line_centre, pixel_indices, v_pix, width=ANTIALIAS_WIDTH):
    """ (len(pixel_indices) x n_grid) matrix mapping LSF grid values
        (velocity, km/s) to model flux at the given pixels, for a line
        centred at line_centre (pixels), using the LOCAL scale v_pix
        (km/s per pixel) for this specific line. """
    offset_pixels = pixel_indices[:, None] - line_centre - (u[None, :] / v_pix)
    return (du / v_pix) * gaussian(offset_pixels, width)

def fit_window(line_centre):
    lo = max(int(np.floor(line_centre)) - HALF_WINDOW, 0)
    hi = min(int(np.ceil(line_centre)) + HALF_WINDOW, n_pixels - 1)
    return np.arange(lo, hi + 1)

def gaussian_mean(u_grid, sigma):
    """ The Gaussian component of the LSF, peak-normalised: value 1 at
        u=0 regardless of sigma. sigma is a VELOCITY width (km/s).
        Departures of the true LSF from this Gaussian are modelled
        separately in section 7. """
    return np.exp(-0.5 * (u_grid / sigma)**2)


# =========================================================================
# 5. Width model: sigma(x), now a velocity width (km/s)
# =========================================================================
# Still represented as a low-order Chebyshev polynomial in (rescaled)
# PIXEL position -- position along the order is still naturally described
# by pixel/wavelength, only the WIDTH value itself changes units.

WIDTH_POLY_ORDER = 2
_initial_width_kms = 1.3 * _v_per_pixel_typical  # was 1.3 PIXELS previously;
                                                    # converted to the
                                                    # equivalent km/s using
                                                    # the typical scale, as a
                                                    # starting guess only

def width_design_matrix(x, order=WIDTH_POLY_ORDER):
    return Chebyshev.chebvander(rescaled_position(x), order)

def width(x, width_coeffs):
    return np.maximum(width_design_matrix(x) @ width_coeffs, 0.05 * _v_per_pixel_typical)

width_coeffs = np.linalg.lstsq(width_design_matrix(peak_pixel),
                                np.full(n_lines, _initial_width_kms), rcond=None)[0]


# =========================================================================
# 6. Wavelength calibration: fitting the position at each comb line
# =========================================================================
# Because pixel position enters the forward model nonlinearly, this is
# still an iterative (Gauss-Newton-like) refinement: linearise the model
# around each line's current position, solve the resulting 1-parameter
# weighted least-squares problem for a local correction with a formal
# uncertainty, then let gaussian_process_smooth fit a smooth curve through
# (wavelength, position + correction, uncertainty) -- exactly the same
# tool already used for the envelope and background, reused here rather
# than building separate GP machinery.
#
# The naive (photon-noise-only) uncertainty on each line's local
# correction is inflated by sqrt(chi2/dof) from that line's own fit before
# being handed to the GP: confirmed directly in an earlier version of this
# project that the raw uncertainty badly underestimates the true scatter
# (it only accounts for noise at fixed width/shape, not any of the
# model's own uncertainty or genuine line-to-line mismatch), and feeding
# that underestimate to a smoother makes it trust noise as if it were
# signal.

DISPERSION_MIN_LENGTH_SCALE = None  # nm; set a floor here if the fitted
                                       # dispersion relation looks too
                                       # wiggly, the same way
                                       # ENVELOPE_MIN_LENGTH_SCALE controls
                                       # the envelope's smoothness
DISPERSION_MAX_LENGTH_SCALE = None  # nm; None uses gaussian_process_smooth's
                                       # own default (roughly 2.7x the
                                       # wavelength span), generous enough
                                       # to capture the whole dispersion
                                       # trend if the data support it

def fit_dispersion(width_coeffs, shape_coeffs, line_position_init,
                    n_outer_steps=4, step_size=0.5, finite_difference_step=1e-3):
    line_position = line_position_init.copy()
    dispersion_gp_fit = None
    for _ in range(n_outer_steps):
        v_pix = velocity_per_pixel_from_positions(line_position)
        line_width = width(line_position, width_coeffs)
        shape_weight = Chebyshev.chebvander(rescaled_position(line_position),
                                             shape_coeffs.shape[0] - 1)
        departure = shape_weight @ shape_coeffs

        delta = np.zeros(n_lines)
        delta_err = np.zeros(n_lines)
        for m in range(n_lines):
            idx = fit_window(line_position[m])
            conv = convolution_matrix(line_position[m], idx, v_pix[m])
            lsf = gaussian_mean(u, line_width[m]) + departure[m]
            model_value = conv @ lsf

            conv_shifted = convolution_matrix(line_position[m] + finite_difference_step,
                                               idx, v_pix[m])
            model_derivative = (conv_shifted @ lsf - model_value) / finite_difference_step

            weight = inverse_variance[idx]
            denominator = np.sum(model_derivative**2 * weight)
            if denominator > 1e-12:
                delta[m] = np.sum(model_derivative * weight * (flux[idx] - model_value)) / denominator
                naive_err = 1.0 / np.sqrt(denominator)
                residual = (model_value + model_derivative * delta[m] - flux[idx]) / flux_err[idx]
                dof = max(len(idx) - 1, 1)
                chi2_reduced = np.sum(residual**2) / dof
                delta_err[m] = naive_err * np.sqrt(max(chi2_reduced, 1.0))
            else:
                delta[m] = 0.0
                delta_err[m] = np.inf

        target = line_position + delta
        dispersion_gp_fit = gaussian_process_smooth(
            wavelength, target, delta_err, wavelength, n_restarts=3,
            min_length_scale=DISPERSION_MIN_LENGTH_SCALE,
            max_length_scale=DISPERSION_MAX_LENGTH_SCALE)
        proposed = dispersion_gp_fit['z_mean']
        line_position = line_position + step_size * (proposed - line_position)

    return line_position, dispersion_gp_fit


# =========================================================================
# 7. LSF shape: departure from a Gaussian, on the velocity grid
# =========================================================================
#   phi(u; x) = G(u; sigma(x)) + sum_{k=0}^{K} d_k(u) * T_k(x~)
#
# u is now velocity (km/s), so d_k(u) and its smoothness prior are all
# expressed in km/s.
#
# Two directions are guarded against directly in the prior precision,
# rather than through a soft or approximate constraint: the direction that
# would let the departure mimic a pure WIDTH change (already the width
# model's job), and the direction that would let it mimic a pure SHIFT in
# line centre (already the dispersion model's job). Both are computed
# analytically from the Gaussian mean's own derivatives (d/dsigma and
# d/d(shift) respectively) and suppressed by adding a large rank-one term
# directly to the precision matrix along each exact direction -- this does
# not depend on the shape of the GP prior's own covariance, unlike a
# Lagrange-multiplier constraint would (an earlier version of this used
# such a constraint for the shift direction specifically, and it turned
# out not to remove the degeneracy exactly: LSF peaks fit at different
# positions across the order shifted visibly away from u=0, which is what
# leftover shift-mode contamination looks like -- see the discussion at
# shift_direction below for why).

SHAPE_POLY_ORDER = 2
GP_SIGNAL_STD = 0.05
GP_LENGTH_SCALE = 1.0 * _v_per_pixel_typical  # was 1.0 PIXEL previously;
                                                 # converted to km/s using the
                                                 # typical scale
IDENTIFIABILITY_WEIGHT = 1e8

grid_distance = u[:, None] - u[None, :]

def shape_prior_covariance():
    return (GP_SIGNAL_STD**2 * np.exp(-grid_distance**2 / (2 * GP_LENGTH_SCALE**2))
            + 1e-6 * GP_SIGNAL_STD**2 * np.eye(n_grid))

def width_change_direction(sigma_ref):
    """ d/dsigma of the Gaussian mean -- guarded against so the departure
        cannot mimic a pure width change (see module discussion). """
    direction = gaussian_mean(u, sigma_ref) * u**2 / sigma_ref**3
    return direction / np.linalg.norm(direction)

def shift_direction(sigma_ref):
    """ d/d(shift) of the Gaussian mean at zero shift, i.e. -dG/du =
        (u/sigma^2) * G(u): the direction a component of the departure
        would need in order to mimic a small shift in line centre.

        This is guarded the same way as width_change_direction, and for
        the same reason a similar guard did not already cover it: the
        model previously relied on a Lagrange-multiplier constraint
        ("zero first moment", sum(u * d(u)) * du = 0) to remove this
        degeneracy, but that constraint's projection is shaped by the
        prior's own covariance, not the constraint direction itself -- it
        removes the component of d along Sigma @ u, not along u * G(u).
        Those only coincide if the covariance's own kernel happens to
        align them, which there is no reason to expect. Confirmed
        directly: LSF peaks fit at different positions across the order
        shifted smoothly away from u=0, which is exactly what leftover
        shift-mode contamination in the departure looks like. Guarding
        the exact direction directly in the precision matrix, the same
        way the width-change direction already is, does not depend on the
        prior covariance's shape at all. """
    direction = gaussian_mean(u, sigma_ref) * u / sigma_ref**2
    return direction / np.linalg.norm(direction)

def shape_prior_precision(sigma_ref):
    isotropic_precision = np.linalg.inv(shape_prior_covariance())
    width_dir = width_change_direction(sigma_ref)
    shift_dir = shift_direction(sigma_ref)
    guard = IDENTIFIABILITY_WEIGHT * (np.outer(width_dir, width_dir)
                                        + np.outer(shift_dir, shift_dir))
    return [isotropic_precision + guard for _ in range(SHAPE_POLY_ORDER + 1)]

def fit_shape_departure(line_position, width_coeffs, v_pix):
    line_width = width(line_position, width_coeffs)
    shape_weight = Chebyshev.chebvander(rescaled_position(line_position), SHAPE_POLY_ORDER)
    sigma_ref = width(np.array([0.5 * (x_min + x_max)]), width_coeffs)[0]

    n_dim = (SHAPE_POLY_ORDER + 1) * n_grid
    normal_matrix = np.zeros((n_dim, n_dim))
    normal_vector = np.zeros(n_dim)
    for block, precision in enumerate(shape_prior_precision(sigma_ref)):
        normal_matrix[block * n_grid:(block + 1) * n_grid,
                       block * n_grid:(block + 1) * n_grid] += precision

    for m in range(n_lines):
        idx = fit_window(line_position[m])
        conv = convolution_matrix(line_position[m], idx, v_pix[m])
        residual_target = flux[idx] - conv @ gaussian_mean(u, line_width[m])
        design = np.hstack([shape_weight[m, k] * conv for k in range(SHAPE_POLY_ORDER + 1)])
        weight = inverse_variance[idx]
        normal_matrix += (design * weight[:, None]).T @ design
        normal_vector += (design * weight[:, None]).T @ residual_target

    posterior_covariance = np.linalg.inv(normal_matrix)
    solution = posterior_covariance @ normal_vector
    return solution.reshape(SHAPE_POLY_ORDER + 1, n_grid)


def fit_width(shape_coeffs, width_coeffs_initial, line_position, v_pix):
    shape_weight = Chebyshev.chebvander(rescaled_position(line_position), SHAPE_POLY_ORDER)
    departure = shape_weight @ shape_coeffs

    def residual(width_coeffs):
        line_width = width(line_position, width_coeffs)
        chunks = []
        for m in range(n_lines):
            idx = fit_window(line_position[m])
            conv = convolution_matrix(line_position[m], idx, v_pix[m])
            lsf = gaussian_mean(u, line_width[m]) + departure[m]
            chunks.append((conv @ lsf - flux[idx]) / flux_err[idx])
        return np.concatenate(chunks)

    fit = least_squares(residual, width_coeffs_initial)
    return fit.x


# =========================================================================
# 8. Joint iterative solution
# =========================================================================
line_position = peak_pixel.copy()  # bootstrap: start from the input LFC
                                     # line-list positions; refined below
shape_coeffs = np.zeros((SHAPE_POLY_ORDER + 1, n_grid))
dispersion_gp_fit = None

N_OUTER_ITERATIONS = 6

for iteration in range(N_OUTER_ITERATIONS):
    v_pix = velocity_per_pixel_from_positions(line_position)

    shape_coeffs = fit_shape_departure(line_position, width_coeffs, v_pix)
    width_coeffs = fit_width(shape_coeffs, width_coeffs, line_position, v_pix)
    line_position, dispersion_gp_fit = fit_dispersion(width_coeffs, shape_coeffs, line_position)

    line_width = width(line_position, width_coeffs)
    v_pix = velocity_per_pixel_from_positions(line_position)
    line_width_pix = line_width / v_pix   # for cross-checking against pixel-space intuition
    position_change = np.sqrt(np.mean((line_position - peak_pixel)**2))
    print(f"iteration {iteration}: "
          f"|position - input| (rms) = {position_change:.4f} pix, "
          f"sigma(v) in [{line_width.min():.4f}, {line_width.max():.4f}] km/s, "
          f"FWHM in [{2.355 * line_width.min():.3f}, {2.355 * line_width.max():.3f}] km/s "
          f"(~[{2.355 * line_width_pix.min():.2f}, {2.355 * line_width_pix.max():.2f}] pix), "
          f"dispersion GP length scale = {dispersion_gp_fit['length_scale']:.3f} nm")


# =========================================================================
# 9. Diagnostics
# =========================================================================
v_pix = velocity_per_pixel_from_positions(line_position)
# The GP's own posterior uncertainty on the fitted position, at each comb
# line's wavelength -- from the LAST outer iteration's dispersion fit, for
# plotting an honest uncertainty band rather than just the position itself.
dispersion_position_std = dispersion_gp_fit['z_std']
line_width = width(line_position, width_coeffs)
shape_weight = Chebyshev.chebvander(rescaled_position(line_position), SHAPE_POLY_ORDER)
departure = shape_weight @ shape_coeffs

model = np.zeros(n_pixels)
fitted_mask = np.zeros(n_pixels, dtype=bool)
lsf_per_line = np.zeros((n_lines, n_grid))
for m in range(n_lines):
    idx = fit_window(line_position[m])
    conv = convolution_matrix(line_position[m], idx, v_pix[m])
    lsf = gaussian_mean(u, line_width[m]) + departure[m]
    lsf_per_line[m] = lsf
    model[idx] += conv @ lsf
    fitted_mask[idx] = True

residual = (flux - model) / flux_err
chi2_per_dof = np.sum(residual[fitted_mask]**2) / fitted_mask.sum()
print(f"chi2 / dof = {chi2_per_dof:.3f}")
print(f"min(phi) = {lsf_per_line.min():.3e}")
print(f"integral(phi) [km/s units] in [{du * lsf_per_line.sum(axis=1).min():.5f}, "
      f"{du * lsf_per_line.sum(axis=1).max():.5f}]")
print(f"resolving power R = c / FWHM(v): "
      f"[{C_LIGHT_KMS / (2.355 * line_width.max()):.0f}, "
      f"{C_LIGHT_KMS / (2.355 * line_width.min()):.0f}]")

np.savez(
    'lsf_reconstruction_results.npz',
    u=u, shape_coeffs=shape_coeffs, width_coeffs=width_coeffs,
    line_position=line_position, dispersion_position_std=dispersion_position_std,
    line_width=line_width, v_pix=v_pix, shape_poly_order=SHAPE_POLY_ORDER,
    x_min=x_min, x_max=x_max, wavelength=wavelength,
    flux=flux, flux_err=flux_err, model=model, fitted_mask=fitted_mask, pixel=pixel,
    envelope_grid_full=envelope_grid_full, background_grid_full=background_grid_full,
    peak_flux=peak_flux, boundary_flux=boundary_flux,
    peak_pixel=peak_pixel, boundary_pixel=boundary_pixel,
    background_coeffs=background_coeffs, background_poly_order=BACKGROUND_POLY_ORDER,
    flux_raw=flux_raw,
)
print("Results written to lsf_reconstruction_results.npz")