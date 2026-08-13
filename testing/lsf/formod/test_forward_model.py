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
on the current estimate of the other two. See the accompanying discussion
for the full mathematical description of each stage.
"""

import numpy as np
from scipy.optimize import least_squares
from numpy.polynomial import chebyshev as Chebyshev

from lfc.fitting.gp import gaussian_process_smooth

SPEED_OF_LIGHT = 2.99792458e8  # m / s

SPECTRUM_FILE = '/Users/dmilakov/software/harps/lsf/test/example_data_ESPRESSO_od=50.txt'
LINES_FILE = '/Users/dmilakov/software/harps/lsf/test/line_positions_ESPRESSO_od=50.txt'


# =========================================================================
# 1. Load the exposure and the LFC line list
# =========================================================================
# The spectrum file holds, per detector pixel: the raw flux, its
# photon-noise uncertainty, and two columns that are not used here (an
# independently-derived background and envelope estimate, superseded by
# the fit performed below). The line list holds, per LFC line: the left
# and right pixel boundaries of its extraction window, the pixel of its
# observed peak, and its true frequency.

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


# =========================================================================
# 2. Envelope and background
# =========================================================================
# Two smooth curves describe the continuum-level structure of the order:
#
#   E(x)  the envelope: the height each LFC line's peak would have in the
#         absence of noise, as a smooth function of pixel x. Line-to-line,
#         this traces the spectrograph's blaze/throughput function and the
#         comb's own mode-power spectrum.
#
#   B(x)  the background: the flux level *between* lines, i.e. the local
#         continuum/dark/scattered-light floor.
#
# Rather than fitting E(x) and B(x) as two independent curves, B(x) is tied
# to E(x) through a polynomial in pixel position:
#
#   B(x) = c_0 + sum_{p=1}^{P} c_p * x^p * E(x)
#
# The motivation is that both curves are shaped by much of the same
# instrumental throughput, so tying them together lets both the line-peak
# measurements and the inter-line measurements jointly constrain a single
# underlying shape, with only P+1 extra numbers (the c_p) for the
# background's own scale and slope.
#
# E(x) is estimated by Gaussian Process regression (a Matern-3/2 kernel,
# with its length scale learned from the data): the peak flux e_m observed
# at each line and, via the tie relation above, the value of E(x) implied
# by each inter-line minimum b_k are combined into one regression problem.
# Because the tie relation is bilinear -- it mixes E(x) with the unknown
# coefficients c_p -- E(x) and the c_p are solved for by alternating
# least squares: fix c_p and regress E(x), then fix E(x) and solve for c_p
# by ordinary least squares, repeating until both stop changing.

def local_extremum(centres, kind, half_width=2):
    """ Maximum or minimum raw flux within +/- half_width pixels of each
        given centre. """
    values = np.empty(len(centres))
    for i, c in enumerate(centres):
        lo = max(int(round(c)) - half_width, 0)
        hi = min(int(round(c)) + half_width + 1, n_pixels)
        values[i] = flux_raw[lo:hi].max() if kind == 'max' else flux_raw[lo:hi].min()
    return values

def local_peak_subpixel(centres, half_width=2):
    """ Sub-pixel corrected peak flux via 3-point parabolic interpolation
        around the discrete maximum near each centre.

        Using the raw discrete maximum directly (as local_extremum does)
        creates a circular normalisation: the envelope is fit through
        these same values, and later every pixel's flux is divided by the
        envelope evaluated at that pixel -- including the very pixel that
        defined the envelope there. The normalised flux at each line's
        brightest pixel is then forced close to 1 almost by construction,
        for every line, regardless of how far that pixel's true sub-pixel
        phase (its offset from the line's actual continuous centre) is
        from zero. Confirmed directly: the model's own prediction at the
        brightest pixel correlates strongly with |phase| (-0.92), exactly
        as expected physically (a pixel further from the true centre
        samples a slightly lower value) -- but the DATA showed almost no
        such dependence (correlation 0.11), because the raw discrete
        maximum IS the phase-dependent quantity that should vary, and using
        it to define the normalisation erases that variation before the
        LSF fit ever sees it.

        The parabolic interpolation below estimates the peak of the
        underlying continuous profile from the three points around the
        discrete maximum, which is a genuinely phase-independent quantity
        (up to noise) -- the standard way to recover a continuous peak
        height from discrete samples. """
    values = np.empty(len(centres))
    for i, c in enumerate(centres):
        lo = max(int(round(c)) - half_width, 0)
        hi = min(int(round(c)) + half_width + 1, n_pixels)
        window = flux_raw[lo:hi]
        i_max = np.argmax(window)
        if i_max == 0 or i_max == len(window) - 1:
            values[i] = window[i_max]   # discrete max at the search edge; no interpolation possible
            continue
        y0, y1, y2 = window[i_max - 1], window[i_max], window[i_max + 1]
        denominator = y2 - 2 * y1 + y0
        if denominator >= 0:            # not a proper local maximum shape; fall back
            values[i] = y1
        else:
            values[i] = y1 - (y2 - y0)**2 / (8 * denominator)
    return values

peak_flux = local_peak_subpixel(peak_pixel)
boundary_pixel = np.unique(np.concatenate([left_edge, right_edge]))
boundary_flux = local_extremum(boundary_pixel, 'min')
peak_flux_err = err_raw[np.round(peak_pixel).astype(int)]
boundary_flux_err = err_raw[np.round(boundary_pixel).astype(int)]

BACKGROUND_POLY_ORDER = 4

background_coeffs = np.zeros(BACKGROUND_POLY_ORDER + 1)
background_coeffs[0] = np.median(boundary_flux)

for iteration in range(4):
    # value of the polynomial-in-x factor at each boundary pixel, excluding
    # the constant term, so that dividing it out gives the envelope value
    # implied by that boundary's background measurement
    gain = sum(background_coeffs[p] * boundary_pixel**p
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

    gp_fit = gaussian_process_smooth(regression_x, regression_z, regression_err,
                                      pixel.astype(float), n_restarts=3)
    envelope_grid = gp_fit['z_mean']

    def envelope(x, _grid=envelope_grid):
        return np.interp(x, pixel.astype(float), _grid)

    envelope_at_boundary = envelope(boundary_pixel)
    design = np.column_stack(
        [np.ones_like(boundary_pixel)] +
        [boundary_pixel**p * envelope_at_boundary for p in range(1, BACKGROUND_POLY_ORDER + 1)])
    new_coeffs = np.linalg.lstsq(design, boundary_flux, rcond=None)[0]
    max_change = np.max(np.abs(new_coeffs - background_coeffs))
    background_coeffs = new_coeffs
    print(f"  envelope/background iteration {iteration}: "
          f"GP length scale = {gp_fit['length_scale']:.1f} pix, "
          f"max coefficient change = {max_change:.3g}")

print(f"background(x) = {background_coeffs[0]:.1f} + "
      f"polynomial(degree {BACKGROUND_POLY_ORDER}) * envelope(x), "
      f"coefficients = {np.round(background_coeffs, 6)}")

def background(x):
    x = np.asarray(x, dtype=float)
    e = envelope(x)
    design = np.column_stack(
        [np.ones_like(x)] + [x**p * e for p in range(1, BACKGROUND_POLY_ORDER + 1)])
    return design @ background_coeffs

envelope_grid_full = envelope(pixel.astype(float))
background_grid_full = background(pixel.astype(float))


# =========================================================================
# 3. Line-spread function model
# =========================================================================
# The LSF is represented as a function phi(u) on a subpixel grid u, spaced
# at 1/PIXEL_SUBSAMPLE of a pixel and extending +/- HALF_WINDOW pixels
# either side of a line's centre -- wide enough to cover every pixel that
# will later be included in that line's fit window.
#
# A model pixel value is obtained by discretising the convolution of the
# (essentially point-like) LFC line with phi:
#
#   model(x_i) = sum_k  W_pix(x_i - x_line - u_k) * phi(u_k) * du
#
# where W_pix is a narrow Gaussian kernel (width ANTIALIAS_WIDTH, much
# smaller than one pixel) that turns the discrete subpixel grid into a
# continuous, band-limited function before it is sampled -- a standard
# anti-aliasing device, not a physical broadening.

PIXEL_SUBSAMPLE = 11
HALF_WINDOW = 6
u = np.arange(-HALF_WINDOW * PIXEL_SUBSAMPLE, HALF_WINDOW * PIXEL_SUBSAMPLE + 1) / PIXEL_SUBSAMPLE
n_grid = len(u)
du = 1.0 / PIXEL_SUBSAMPLE
ANTIALIAS_WIDTH = 0.15

def gaussian(x, width):
    return np.exp(-0.5 * (x / width)**2) / (np.sqrt(2 * np.pi) * width)

def convolution_matrix(line_centre, pixel_indices, width=ANTIALIAS_WIDTH):
    """ (len(pixel_indices) x n_grid) matrix mapping LSF grid values to
        model flux at the given pixels, for a line centred at line_centre. """
    offset = pixel_indices[:, None] - line_centre - u[None, :]
    return du * gaussian(offset, width)

def fit_window(line_centre):
    lo = max(int(np.floor(line_centre)) - HALF_WINDOW, 0)
    hi = min(int(np.ceil(line_centre)) + HALF_WINDOW, n_pixels - 1)
    return np.arange(lo, hi + 1)

def gaussian_mean(u_grid, sigma):
    """ The Gaussian component of the LSF, peak-normalised: value 1 at
        u=0 regardless of sigma. Departures of the true LSF from this
        Gaussian are modelled separately in section 6. """
    return np.exp(-0.5 * (u_grid / sigma)**2)

x_min, x_max = peak_pixel.min(), peak_pixel.max()

def rescaled_position(x):
    """ Pixel position rescaled to [-1, 1] over the range spanned by the
        LFC lines, for use with Chebyshev polynomials. """
    return 2 * (x - x_min) / (x_max - x_min) - 1


# =========================================================================
# 4. Width model: sigma(x)
# =========================================================================
# The Gaussian width is allowed to vary smoothly across the order --
# physically expected from anamorphic magnification and focus changes
# across the detector -- represented as a low-order Chebyshev polynomial
# in (rescaled) pixel position, fit by nonlinear least squares (sigma
# enters the pixel model nonlinearly, through the Gaussian above).

WIDTH_POLY_ORDER = 2

def width_design_matrix(x, order=WIDTH_POLY_ORDER):
    return Chebyshev.chebvander(rescaled_position(x), order)

def width(x, width_coeffs):
    return np.maximum(width_design_matrix(x) @ width_coeffs, 0.05)

width_coeffs = np.linalg.lstsq(width_design_matrix(peak_pixel), np.full(n_lines, 1.3),
                                rcond=None)[0]


# =========================================================================
# Flux normalisation
# =========================================================================
# Every pixel's flux is background-subtracted and divided by the envelope,
# so that a line's peak in the normalised units is close to 1 regardless
# of its absolute brightness. This is what lets one shared LSF model be
# fit across lines of very different intensity.

flux = (flux_raw - background_grid_full) / envelope_grid_full
flux_err = err_raw / envelope_grid_full
inverse_variance = 1.0 / flux_err**2


# =========================================================================
# 5. Wavelength calibration model
# =========================================================================
# The dispersion relation -- pixel position as a function of wavelength --
# is represented as a single Chebyshev expansion in (rescaled) wavelength,
# but with its coefficients split into two tiers:
#
#   x(lambda) = sum_{k=0}^{P} xi_k * T_k(lambda~)
#
#   - degrees 0 .. DISPERSION_FREE_ORDER: an unconstrained smooth trend,
#     representing the dispersion relation's overall shape over the whole
#     order.
#   - degrees DISPERSION_FREE_ORDER+1 .. DISPERSION_TOTAL_ORDER: a "local
#     perturbation" band, ridge-penalised so that a typical realisation
#     has RMS amplitude DISPERSION_LOCAL_SCALE (in pixels). A degree-k
#     Chebyshev term oscillates roughly k times across the full order, so
#     this band represents structure on a scale of tens of line spacings
#     -- shorter than the global trend, but smooth rather than a free
#     value at every line.
#
# Because pixel position enters the forward model nonlinearly, the
# coefficients are fit by Gauss-Newton iteration: linearise the model
# around the current dispersion solution, solve the resulting (linear,
# ridge-regularised) weighted least-squares problem for an updated
# coefficient vector, and repeat a few times as the solution shifts.

DISPERSION_TOTAL_ORDER = 25
DISPERSION_FREE_ORDER = 4
DISPERSION_LOCAL_SCALE = 0.15  # pixels, RMS

lambda_min, lambda_max = wavelength.min(), wavelength.max()

def rescaled_wavelength(lam):
    return 2 * (lam - lambda_min) / (lambda_max - lambda_min) - 1

def dispersion_design_matrix(lam, order=DISPERSION_TOTAL_ORDER):
    return Chebyshev.chebvander(rescaled_wavelength(lam), order)

dispersion_basis = dispersion_design_matrix(wavelength)

n_local_band = DISPERSION_TOTAL_ORDER - DISPERSION_FREE_ORDER
dispersion_ridge = np.zeros(DISPERSION_TOTAL_ORDER + 1)
dispersion_ridge[DISPERSION_FREE_ORDER + 1:] = n_local_band / DISPERSION_LOCAL_SCALE**2
dispersion_ridge_precision = np.diag(dispersion_ridge)

print(f"dispersion relation: degrees 0-{DISPERSION_FREE_ORDER} unconstrained, "
      f"degrees {DISPERSION_FREE_ORDER + 1}-{DISPERSION_TOTAL_ORDER} constrained to "
      f"~{DISPERSION_LOCAL_SCALE} pix RMS")

def fit_dispersion(width_coeffs, shape_coeffs, dispersion_coeffs,
                    n_gauss_newton_steps=4, step_size=0.5,
                    finite_difference_step=1e-3):
    coeffs = dispersion_coeffs.copy()
    for _ in range(n_gauss_newton_steps):
        line_position = dispersion_basis @ coeffs
        line_width = width(line_position, width_coeffs)
        shape_weight = Chebyshev.chebvander(rescaled_position(line_position),
                                             shape_coeffs.shape[0] - 1)
        departure = shape_weight @ shape_coeffs

        n_dim = DISPERSION_TOTAL_ORDER + 1
        normal_matrix = dispersion_ridge_precision.copy()
        normal_vector = np.zeros(n_dim)

        for m in range(n_lines):
            idx = fit_window(line_position[m])
            conv = convolution_matrix(line_position[m], idx)
            lsf = gaussian_mean(u, line_width[m]) + departure[m]
            model_value = conv @ lsf

            conv_shifted = convolution_matrix(line_position[m] + finite_difference_step, idx)
            model_derivative = (conv_shifted @ lsf - model_value) / finite_difference_step

            weight = inverse_variance[idx]
            row = dispersion_basis[m][None, :] * model_derivative[:, None]
            target = flux[idx] - model_value + model_derivative * line_position[m]

            normal_matrix += (row * weight[:, None]).T @ row
            normal_vector += (row * weight[:, None]).T @ target

        proposed = np.linalg.solve(normal_matrix, normal_vector)
        coeffs = coeffs + step_size * (proposed - coeffs)

    return dispersion_basis @ coeffs, coeffs


# =========================================================================
# 6. LSF shape: departure from a Gaussian
# =========================================================================
# The full LSF is a Gaussian mean plus a smooth, position-dependent
# departure represented as a separable expansion:
#
#   phi(u; x) = G(u; sigma(x)) + sum_{k=0}^{K} d_k(u) * T_k(x~)
#
# d_0(u) is the departure of the order-averaged LSF shape from a Gaussian;
# d_1(u), d_2(u) describe how that departure itself changes across the
# order. Each d_k(u) is given a Gaussian Process prior with a squared-
# exponential kernel, and the whole set is estimated by linear GP
# regression -- the model is linear in d_k given the current width and
# line positions, so unlike the wavelength solution this has a closed-form
# posterior (no iteration needed within this step).
#
# Two constraints are imposed on every d_k, for reasons of identifiability
# rather than physics:
#
#   * First moment zero (sum_u u * d_k(u) = 0): a nonzero first moment
#     would be indistinguishable from a shift in line centre, which is
#     already accounted for by the dispersion model.
#
#   * No component along the direction  g(u) = G(u;sigma) * u^2/sigma^3,
#     which is exactly the derivative of the Gaussian mean with respect to
#     its own width. Without this, the departure term can reproduce a pure
#     change in width just as well as the width model itself can, making
#     the two components of the fit degenerate; the constraint forces any
#     genuine width change to be captured by sigma(x) alone, leaving the
#     departure term to describe only the LSF's deviation from a Gaussian
#     shape.

SHAPE_POLY_ORDER = 2
GP_SIGNAL_STD = 0.05        # prior standard deviation of the departure, in
                             # the same (peak-normalised) units as phi
GP_LENGTH_SCALE = 1.0        # pixels
IDENTIFIABILITY_WEIGHT = 1e8  # strength of the width/shape separation constraint

grid_distance = u[:, None] - u[None, :]

def shape_prior_covariance():
    return (GP_SIGNAL_STD**2 * np.exp(-grid_distance**2 / (2 * GP_LENGTH_SCALE**2))
            + 1e-6 * GP_SIGNAL_STD**2 * np.eye(n_grid))

def width_change_direction(sigma_ref):
    direction = gaussian_mean(u, sigma_ref) * u**2 / sigma_ref**3
    return direction / np.linalg.norm(direction)

def shape_prior_precision(sigma_ref):
    isotropic_precision = np.linalg.inv(shape_prior_covariance())
    direction = width_change_direction(sigma_ref)
    guard = IDENTIFIABILITY_WEIGHT * np.outer(direction, direction)
    return [isotropic_precision + guard for _ in range(SHAPE_POLY_ORDER + 1)]

def fit_shape_departure(line_position, width_coeffs):
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
        conv = convolution_matrix(line_position[m], idx)
        residual_target = flux[idx] - conv @ gaussian_mean(u, line_width[m])
        design = np.hstack([shape_weight[m, k] * conv for k in range(SHAPE_POLY_ORDER + 1)])
        weight = inverse_variance[idx]
        normal_matrix += (design * weight[:, None]).T @ design
        normal_vector += (design * weight[:, None]).T @ residual_target

    posterior_covariance = np.linalg.inv(normal_matrix)
    unconstrained_solution = posterior_covariance @ normal_vector

    constraint_matrix = np.zeros((SHAPE_POLY_ORDER + 1, n_dim))
    for k in range(SHAPE_POLY_ORDER + 1):
        constraint_matrix[k, k * n_grid:(k + 1) * n_grid] = u * du
    constraint_target = np.zeros(SHAPE_POLY_ORDER + 1)

    constraint_gram = constraint_matrix @ posterior_covariance @ constraint_matrix.T
    lagrange_multiplier = np.linalg.solve(
        constraint_gram, constraint_target - constraint_matrix @ unconstrained_solution)
    solution = unconstrained_solution + posterior_covariance @ constraint_matrix.T @ lagrange_multiplier
    return solution.reshape(SHAPE_POLY_ORDER + 1, n_grid)


def fit_width(shape_coeffs, width_coeffs_initial, line_position):
    shape_weight = Chebyshev.chebvander(rescaled_position(line_position), SHAPE_POLY_ORDER)
    departure = shape_weight @ shape_coeffs

    def residual(width_coeffs):
        line_width = width(line_position, width_coeffs)
        chunks = []
        for m in range(n_lines):
            idx = fit_window(line_position[m])
            conv = convolution_matrix(line_position[m], idx)
            lsf = gaussian_mean(u, line_width[m]) + departure[m]
            chunks.append((conv @ lsf - flux[idx]) / flux_err[idx])
        return np.concatenate(chunks)

    fit = least_squares(residual, width_coeffs_initial)
    return fit.x


# =========================================================================
# 7. Joint iterative solution
# =========================================================================
# The LSF shape, the width, and the wavelength solution are refined
# together in a fixed-point iteration: each is fit in turn while holding
# the other two fixed at their current estimate, and the cycle is repeated
# until the estimates stop changing appreciably.

initial_dispersion_coeffs = np.linalg.lstsq(dispersion_basis, peak_pixel, rcond=None)[0]
line_position = dispersion_basis @ initial_dispersion_coeffs
dispersion_coeffs = initial_dispersion_coeffs
shape_coeffs = np.zeros((SHAPE_POLY_ORDER + 1, n_grid))

N_OUTER_ITERATIONS = 6

for iteration in range(N_OUTER_ITERATIONS):
    shape_coeffs = fit_shape_departure(line_position, width_coeffs)
    width_coeffs = fit_width(shape_coeffs, width_coeffs, line_position)
    line_position, dispersion_coeffs = fit_dispersion(width_coeffs, shape_coeffs,
                                                        dispersion_coeffs)

    line_width = width(line_position, width_coeffs)
    position_change = np.sqrt(np.mean((line_position - peak_pixel)**2))
    print(f"iteration {iteration}: "
          f"|position - input| (rms) = {position_change:.4f} pix, "
          f"sigma(x) in [{line_width.min():.3f}, {line_width.max():.3f}] pix, "
          f"FWHM in [{2.355 * line_width.min():.2f}, {2.355 * line_width.max():.2f}] pix")


# =========================================================================
# 8. Diagnostics
# =========================================================================
line_width = width(line_position, width_coeffs)
shape_weight = Chebyshev.chebvander(rescaled_position(line_position), SHAPE_POLY_ORDER)
departure = shape_weight @ shape_coeffs

model = np.zeros(n_pixels)
fitted_mask = np.zeros(n_pixels, dtype=bool)
lsf_per_line = np.zeros((n_lines, n_grid))
for m in range(n_lines):
    idx = fit_window(line_position[m])
    conv = convolution_matrix(line_position[m], idx)
    lsf = gaussian_mean(u, line_width[m]) + departure[m]
    lsf_per_line[m] = lsf
    model[idx] += conv @ lsf
    fitted_mask[idx] = True

residual = (flux - model) / flux_err
chi2_per_dof = np.sum(residual[fitted_mask]**2) / fitted_mask.sum()
print(f"chi2 / dof = {chi2_per_dof:.3f}")
print(f"min(phi) = {lsf_per_line.min():.3e}")
print(f"integral(phi) in [{du * lsf_per_line.sum(axis=1).min():.5f}, "
      f"{du * lsf_per_line.sum(axis=1).max():.5f}]")

np.savez(
    'lsf_reconstruction_results.npz',
    u=u, shape_coeffs=shape_coeffs, width_coeffs=width_coeffs,
    dispersion_coeffs=dispersion_coeffs, line_position=line_position,
    line_width=line_width, shape_poly_order=SHAPE_POLY_ORDER,
    x_min=x_min, x_max=x_max, wavelength=wavelength,
    flux=flux, flux_err=flux_err, model=model, fitted_mask=fitted_mask, pixel=pixel,
    envelope_grid_full=envelope_grid_full, background_grid_full=background_grid_full,
    peak_flux=peak_flux, boundary_flux=boundary_flux,
    peak_pixel=peak_pixel, boundary_pixel=boundary_pixel,
    background_coeffs=background_coeffs, background_poly_order=BACKGROUND_POLY_ORDER,
    flux_raw=flux_raw,
)
print("Results written to lsf_reconstruction_results.npz")