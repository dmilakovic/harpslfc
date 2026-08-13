"""
Empirical LSF and wavelength-solution reconstruction from a laser frequency
comb (LFC) exposure, using Gaussian Processes in place of fixed-degree
polynomials for every position-dependent quantity.

Three Gaussian Processes are used:

  1. The Gaussian width of the line-spread function, log(tau(x)), varying
     smoothly across the order.
  2. The line-spread function's departure from that Gaussian, d(u, x), a
     function of both the subpixel offset u and the position x.
  3. The wavelength calibration's departure from a low-order polynomial
     trend, eta(lambda).

INDUCING POINTS, NOT ONE PARAMETER PER LINE OR PER PIXEL. A Gaussian
Process evaluated with one degree of freedom at every line position (405
of them) or every subpixel grid point is not just expensive -- it is
numerically pathological whenever the correlation length spans many of
those points at once: the covariance matrix becomes so ill-conditioned
that its inverse no longer behaves like a sensible precision matrix, and
fitting the kernel's own length scale and amplitude by maximum likelihood
drives them to a degenerate boundary rather than a genuine optimum. Every
GP below is instead represented on a modest set of INDUCING POINTS (knots)
spread across the relevant domain, with values elsewhere obtained from the
GP's own predictive-mean formula. This is a standard sparse/inducing-point
GP construction. It is still a fully nonparametric-flavoured fit -- the
knot values are not free coefficients of a chosen basis, they are governed
by a learned covariance kernel -- and adding more knots simply increases
the resolution available, rather than changing the character of the model
the way raising a polynomial's degree does.

HYPERPARAMETER LEARNING. Each kernel's amplitude and length scale are fit
by maximising the Gaussian Process marginal likelihood. Because the
marginal likelihood can be driven to a degenerate optimum whenever the
underlying regression is close to rank-deficient (confirmed for the
LSF-shape problem: the per-pixel design matrix, built from many
overlapping line windows all constraining one shared function, has many
directions the data barely constrain at all), the search is restricted to
a physically reasonable range for each hyperparameter rather than left
unconstrained.

The processing order is unchanged: the envelope and background are
estimated first, directly from the raw pixel data; the LSF shape, its
width, and the wavelength solution are then refined together in a
repeating cycle, because each is defined only in terms of the current
estimate of the other two.
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from numpy.polynomial import chebyshev as Chebyshev

from lfc.fitting.gp import gaussian_process_smooth

SPEED_OF_LIGHT = 2.99792458e8  # m / s

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


# =========================================================================
# 2. Envelope and background (unchanged: estimated directly from the raw
#    data, before any LSF or wavelength-solution model is assumed)
# =========================================================================
def local_extremum(centres, kind, half_width=2):
    values = np.empty(len(centres))
    for i, c in enumerate(centres):
        lo = max(int(round(c)) - half_width, 0)
        hi = min(int(round(c)) + half_width + 1, n_pixels)
        values[i] = flux_raw[lo:hi].max() if kind == 'max' else flux_raw[lo:hi].min()
    return values

peak_flux = local_extremum(peak_pixel, 'max')
boundary_pixel = np.unique(np.concatenate([left_edge, right_edge]))
boundary_flux = local_extremum(boundary_pixel, 'min')
peak_flux_err = err_raw[np.round(peak_pixel).astype(int)]
boundary_flux_err = err_raw[np.round(boundary_pixel).astype(int)]

BACKGROUND_POLY_ORDER = 4
background_coeffs = np.zeros(BACKGROUND_POLY_ORDER + 1)
background_coeffs[0] = np.median(boundary_flux)

for iteration in range(4):
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
    background_coeffs = np.linalg.lstsq(design, boundary_flux, rcond=None)[0]

print(f"background(x) = {background_coeffs[0]:.1f} + "
      f"polynomial(degree {BACKGROUND_POLY_ORDER}) * envelope(x)")

def background(x):
    x = np.asarray(x, dtype=float)
    e = envelope(x)
    design = np.column_stack(
        [np.ones_like(x)] + [x**p * e for p in range(1, BACKGROUND_POLY_ORDER + 1)])
    return design @ background_coeffs

envelope_grid_full = envelope(pixel.astype(float))
background_grid_full = background(pixel.astype(float))

# NORMALISATION: (raw - background) / (envelope - background), not
# (raw - background) / envelope. envelope_grid_full is built from RAW flux
# maxima, so it includes the background pedestal: envelope = background +
# line_amplitude. At a line's peak pixel, raw flux = background +
# line_amplitude too, so (raw-background)/(envelope-background) = 1
# exactly at the peak, matching gaussian_mean(u=0, sigma) = 1 for any
# sigma. The old formula, (raw-background)/envelope, gives
# line_amplitude/(background+line_amplitude) < 1 at the peak always --
# and because tau has NO effect on the Gaussian component's peak height
# (only on how fast it falls off), tau cannot correct a peak-height
# mismatch; only the departure term can, by faking a sharp feature at
# u=0 that has nothing to do with the true LSF shape. This is a very
# plausible driver of the high-amplitude, high-frequency departure
# reported: the term was being asked to do a job it is not suited for.
# flux_err's denominator is changed to match, for consistent error
# propagation (it was left as envelope_grid_full alone previously,
# inconsistent with the corrected numerator/denominator above).
flux = (flux_raw - background_grid_full) / (envelope_grid_full - background_grid_full)
flux_err = err_raw / (envelope_grid_full - background_grid_full)
inverse_variance = 1.0 / flux_err**2


# =========================================================================
# 3. Line-spread function pixel model (unchanged)
# =========================================================================
PIXEL_SUBSAMPLE = 11
HALF_WINDOW = 6
u = np.arange(-HALF_WINDOW * PIXEL_SUBSAMPLE, HALF_WINDOW * PIXEL_SUBSAMPLE + 1) / PIXEL_SUBSAMPLE
n_grid = len(u)
du = 1.0 / PIXEL_SUBSAMPLE
ANTIALIAS_WIDTH = 0.15

def gaussian(x, width):
    return np.exp(-0.5 * (x / width)**2) / (np.sqrt(2 * np.pi) * width)

def convolution_matrix(line_centre, pixel_indices, width=ANTIALIAS_WIDTH):
    offset = pixel_indices[:, None] - line_centre - u[None, :]
    return du * gaussian(offset, width)

def fit_window(line_centre):
    lo = max(int(np.floor(line_centre)) - HALF_WINDOW, 0)
    hi = min(int(np.ceil(line_centre)) + HALF_WINDOW, n_pixels - 1)
    return np.arange(lo, hi + 1)

def gaussian_mean(u_grid, sigma):
    return np.exp(-0.5 * (u_grid / sigma)**2)

x_min, x_max = peak_pixel.min(), peak_pixel.max()
x_span = x_max - x_min


# =========================================================================
# 4. Reusable inducing-point Gaussian Process machinery
# =========================================================================
def squared_exponential(z_a, z_b, amplitude, length_scale, relative_jitter=1e-6):
    distance = z_a[:, None] - z_b[None, :]
    covariance = amplitude**2 * np.exp(-distance**2 / (2 * length_scale**2))
    if z_a is z_b or np.array_equal(z_a, z_b):
        covariance = covariance + relative_jitter * amplitude**2 * np.eye(len(z_a))
    return covariance

def gp_evidence_terms(design_matrix, target, weight, prior_precision):
    weighted_design = design_matrix * weight[:, None]
    return weighted_design.T @ design_matrix, weighted_design.T @ target

def gp_log_marginal_likelihood(data_term, data_vector, prior_precision, prior_logdet):
    posterior_precision = prior_precision + data_term
    sign, posterior_logdet = np.linalg.slogdet(posterior_precision)
    if sign <= 0:
        return -np.inf
    posterior_mean = np.linalg.solve(posterior_precision, data_vector)
    return (0.5 * data_vector @ posterior_mean
            - 0.5 * posterior_logdet
            + 0.5 * prior_logdet)

def fit_hyperparameters(log_bounds, evidence_fn, n_restarts=4, seed=0):
    rng = np.random.default_rng(seed)
    lower = np.array([b[0] for b in log_bounds])
    upper = np.array([b[1] for b in log_bounds])
    best_value, best_theta = -np.inf, 0.5 * (lower + upper)
    starts = [0.5 * (lower + upper)] + [
        rng.uniform(lower, upper) for _ in range(n_restarts - 1)]
    for theta0 in starts:
        result = minimize(lambda t: -evidence_fn(t), theta0,
                           method='L-BFGS-B', bounds=log_bounds,
                           options={'maxiter': 100})
        if result.success and -result.fun > best_value:
            best_value, best_theta = -result.fun, result.x
    return best_theta

def fit_length_scale_and_amplitude(length_scale_bounds, amplitude_bounds,
                                    build_design_for_length_scale,
                                    kernel_builder, n_dim, prior_slice,
                                    n_length_scale_candidates=6):
    """ Jointly choose a kernel's length scale and amplitude by maximising
        the marginal likelihood, in a way that stays consistent about which
        quantities the length scale actually affects: the design matrix
        used to fit a GP's knot values depends on the length scale (through
        the interpolation weights that map knot values onto the model),
        but not on the amplitude, which enters only through the prior. """
    lower, upper = np.exp(length_scale_bounds[0]), np.exp(length_scale_bounds[1])
    candidates = np.geomspace(lower, upper, n_length_scale_candidates)

    best_evidence, best_result = -np.inf, None
    for length_scale in candidates:
        data_term, data_vector, extra = build_design_for_length_scale(length_scale)

        def evidence(log_amplitude):
            kernel = kernel_builder(np.exp(log_amplitude[0]), length_scale)
            sign, logdet = np.linalg.slogdet(kernel)
            if sign <= 0:
                return -np.inf
            full_precision = np.zeros((n_dim, n_dim))
            full_precision[prior_slice, prior_slice] = np.linalg.inv(kernel)
            return gp_log_marginal_likelihood(data_term, data_vector, full_precision, -logdet)

        log_amplitude = fit_hyperparameters([amplitude_bounds], evidence, n_restarts=3)
        value = evidence(log_amplitude)
        if value > best_evidence:
            best_evidence = value
            best_result = (log_amplitude[0], np.log(length_scale), data_term, data_vector, extra)

    return best_result


# =========================================================================
# 5. Width: log(tau(x)) as a Gaussian Process on position knots
# =========================================================================
N_WIDTH_KNOTS = 12
width_knot_x = np.linspace(x_min, x_max, N_WIDTH_KNOTS)

WIDTH_LOG_BOUNDS = [(np.log(1e-3), np.log(0.2)), (np.log(x_span), np.log(x_span * 5))]

# ----------------------------------------------------------------------
# FWHM/tau bounds derived from resolving power, not hardcoded. R and its
# fractional tolerance are the only numbers stated directly; everything
# else (the velocity-per-pixel scale needed to convert a spectral
# resolution into a number of pixels) comes from the loaded line list's
# own wavelength range and the pixel range it spans, not an assumed
# constant.
# ----------------------------------------------------------------------
R_NOMINAL = 145000
R_FRACTIONAL_TOLERANCE = 0.20
SPEED_OF_LIGHT_KMS = SPEED_OF_LIGHT / 1e3

_mean_wavelength = 0.5 * (wavelength.min() + wavelength.max())
_velocity_per_pixel = SPEED_OF_LIGHT_KMS * (wavelength.max() - wavelength.min()) / x_span \
                       / _mean_wavelength
R_lower, R_upper = R_NOMINAL * (1 - R_FRACTIONAL_TOLERANCE), R_NOMINAL * (1 + R_FRACTIONAL_TOLERANCE)
FWHM_MIN = (SPEED_OF_LIGHT_KMS / R_upper) / _velocity_per_pixel
FWHM_MAX = (SPEED_OF_LIGHT_KMS / R_lower) / _velocity_per_pixel
TAU_MIN, TAU_MAX = FWHM_MIN / 2.3548, FWHM_MAX / 2.3548
print(f"velocity/pixel = {_velocity_per_pixel:.4f} km/s -> "
      f"FWHM bounds [{FWHM_MIN:.3f}, {FWHM_MAX:.3f}] pix from R={R_NOMINAL}"
      f"+/-{int(R_FRACTIONAL_TOLERANCE*100)}%")

# TAU_REFERENCE / TAU_RIDGE_WEIGHT: a soft penalty pulling tau(x) toward the
# centre of the physically expected range, in ADDITION to the hard clip
# above. The hard clip alone does not remove the INCENTIVE to grow tau: the
# Gaussian mean's peak is fixed at 1 regardless of tau, but its integral
# grows with tau, so any excess flux the model cannot otherwise explain
# (imperfect background/envelope, a neighbouring line's wing inside the
# fitting window, anything) can always be absorbed by growing tau without
# ever disturbing the peak match -- a "free lunch" with nothing in the cost
# function to resist it, which is a very plausible reason tau tends to
# sit at whatever boundary is given rather than settling inside it. The
# ridge below makes moving tau away from a sensible default cost something,
# so the fit only does it when the data genuinely demand it.
TAU_REFERENCE = 0.5 * (TAU_MIN + TAU_MAX)
TAU_RIDGE_WEIGHT = 0.1

def width_kernel(amplitude, length_scale):
    return squared_exponential(width_knot_x, width_knot_x, amplitude, length_scale)

def width_from_knots(x, log_tau0, knot_values, length_scale):
    weights = squared_exponential(np.asarray(x, dtype=float), width_knot_x, 1.0, length_scale)
    unit_kernel = squared_exponential(width_knot_x, width_knot_x, 1.0, length_scale)
    interpolation = weights @ np.linalg.solve(unit_kernel, knot_values)
    tau = np.exp(log_tau0 + interpolation)
    return tau
    # return np.clip(tau, TAU_MIN, TAU_MAX)

def fit_width(line_position, shape_knot_values, shape_hyperparameters,
              log_tau0_init, knot_values_init, log_amplitude_init, log_length_scale_init,
              n_gauss_newton_steps=3):
    log_tau0 = log_tau0_init
    knot_values = knot_values_init.copy()
    log_amplitude, log_length_scale = log_amplitude_init, log_length_scale_init
    n_dim = 1 + N_WIDTH_KNOTS
    finite_difference_step = 1e-3

    for _ in range(n_gauss_newton_steps):
        length_scale_for_linearisation = np.exp(log_length_scale)
        tau_current = width_from_knots(line_position, log_tau0, knot_values,
                                        length_scale_for_linearisation)
        departure = evaluate_shape_departure(line_position, shape_knot_values, shape_hyperparameters)

        def build_design_for_length_scale(length_scale):
            interp_weights = squared_exponential(line_position, width_knot_x, 1.0, length_scale)
            design_rows, targets, weights = [], [], []
            for m in range(n_lines):
                idx = fit_window(line_position[m])
                conv = convolution_matrix(line_position[m], idx)
                lsf = gaussian_mean(u, tau_current[m]) + departure[m]
                model_value = conv @ lsf

                perturbed_tau = tau_current[m] * (1 + finite_difference_step)
                lsf_perturbed = gaussian_mean(u, perturbed_tau) + departure[m]
                model_perturbed = conv @ lsf_perturbed
                d_model_d_log_tau = (model_perturbed - model_value) / finite_difference_step

                row = np.zeros((len(idx), n_dim))
                row[:, 0] = d_model_d_log_tau
                row[:, 1:] = d_model_d_log_tau[:, None] * interp_weights[m][None, :]
                target = flux[idx] - model_value + d_model_d_log_tau * np.log(tau_current[m])

                design_rows.append(row)
                targets.append(target)
                weights.append(inverse_variance[idx])

            # Soft ridge pulling log_tau0 toward TAU_REFERENCE (see note
            # above); implemented as an extra row/target pair in the same
            # weighted least-squares system, weight = TAU_RIDGE_WEIGHT.
            ridge_row = np.zeros((1, n_dim))
            ridge_row[0, 0] = 1.0
            design_rows.append(ridge_row)
            targets.append(np.array([np.log(TAU_REFERENCE)]))
            weights.append(np.array([TAU_RIDGE_WEIGHT]))

            design_matrix = np.vstack(design_rows)
            target_vector = np.concatenate(targets)
            weight_vector = np.concatenate(weights)
            prior_precision = np.zeros((n_dim, n_dim))
            data_term, data_vector = gp_evidence_terms(design_matrix, target_vector,
                                                         weight_vector, prior_precision)
            return data_term, data_vector, interp_weights

        log_amplitude, log_length_scale, data_term, data_vector, interp_weights = \
            fit_length_scale_and_amplitude(
                WIDTH_LOG_BOUNDS[1], WIDTH_LOG_BOUNDS[0],
                build_design_for_length_scale, width_kernel, n_dim, slice(1, None))

        prior_precision = np.zeros((n_dim, n_dim))
        prior_precision[1:, 1:] = np.linalg.inv(width_kernel(np.exp(log_amplitude),
                                                               np.exp(log_length_scale)))
        proposed = np.linalg.solve(prior_precision + data_term, data_vector)
        current = np.concatenate([[log_tau0], knot_values])
        step_size = 0.5
        updated = current + step_size * (proposed - current)
        log_tau0, knot_values = updated[0], updated[1:]

    return log_tau0, knot_values, log_amplitude, log_length_scale


# =========================================================================
# 6. LSF shape: a Gaussian Process departure from the Gaussian mean, over
#    both the subpixel offset u and the position x
# =========================================================================
N_SHAPE_KNOTS = 8
shape_knot_x = np.linspace(x_min, x_max, N_SHAPE_KNOTS)

# Length scale in u bounded well above the model's own two resolution
# floors -- the subpixel grid spacing (1/PIXEL_SUBSAMPLE = 0.091 pixel) and
# the anti-aliasing width (0.15 pixel). A length scale shorter than either
# makes the kernel behave close to uncorrelated noise from one grid point
# to the next, which is indistinguishable from fitting noise and shows up
# as exactly the high-frequency, ringing appearance reported -- confirmed
# by comparing the bound directly against these two floors (0.05 pixel,
# the value tried, sits BELOW both). Kept comfortably above them here.
# Upper bound stays below the line's own width, since coarser structure
# than that starts describing the whole line's width, which is tau(x)'s
# job, not this term's.
SHAPE_LOG_BOUNDS = [
    (np.log(0.01), np.log(0.3)),
    (np.log(0.5), np.log(3.0)),
    (np.log(x_span / 2), np.log(x_span * 2)),
]
SHAPE_DEPARTURE_MAX = 0.3

def shape_kernel_u(amplitude, length_scale_u):
    return squared_exponential(u, u, amplitude, length_scale_u)

def shape_kernel_x(length_scale_x):
    return squared_exponential(shape_knot_x, shape_knot_x, 1.0, length_scale_x)

def width_change_direction(tau_value):
    direction = gaussian_mean(u, tau_value) * u**2 / tau_value**3
    return direction / np.linalg.norm(direction)

def evaluate_shape_departure(line_position, knot_values, hyperparameters):
    _, _, log_length_scale_x = hyperparameters
    length_scale_x = np.exp(log_length_scale_x)
    kernel_x = shape_kernel_x(length_scale_x)
    weights = squared_exponential(line_position, shape_knot_x, 1.0, length_scale_x)
    interpolation_weights = weights @ np.linalg.inv(kernel_x)
    departure = interpolation_weights @ knot_values
    return departure
    # return np.clip(departure, -SHAPE_DEPARTURE_MAX, SHAPE_DEPARTURE_MAX)

def fit_shape_departure(line_position, log_tau0, width_knot_values, width_length_scale,
                          hyperparameters_init):
    tau_at_lines = width_from_knots(line_position, log_tau0, width_knot_values, width_length_scale)
    tau_at_knots = width_from_knots(shape_knot_x, log_tau0, width_knot_values, width_length_scale)

    n_dim = N_SHAPE_KNOTS * n_grid

    def build_constraint_matrix():
        rows = []
        for j in range(N_SHAPE_KNOTS):
            first_moment_row = np.zeros(n_dim)
            first_moment_row[j * n_grid:(j + 1) * n_grid] = u * du
            rows.append(first_moment_row)
            guard_row = np.zeros(n_dim)
            guard_row[j * n_grid:(j + 1) * n_grid] = width_change_direction(tau_at_knots[j])
            rows.append(guard_row)
        return np.array(rows)

    constraint_matrix = build_constraint_matrix()
    constraint_target = np.zeros(constraint_matrix.shape[0])

    design_rows, targets, weights = [], [], []
    for m in range(n_lines):
        idx = fit_window(line_position[m])
        conv = convolution_matrix(line_position[m], idx)
        residual_target = flux[idx] - conv @ gaussian_mean(u, tau_at_lines[m])
        design_rows.append(conv)
        targets.append(residual_target)
        weights.append(inverse_variance[idx])

    def evidence_given_data(log_amp, log_ell_u, data_term, data_vector, logdet_x, ell_x):
        amplitude, ell_u = np.exp(log_amp), np.exp(log_ell_u)
        kernel_u = shape_kernel_u(amplitude, ell_u)
        sign_u, logdet_u = np.linalg.slogdet(kernel_u)
        if sign_u <= 0:
            return -np.inf
        kernel_u_inv = np.linalg.inv(kernel_u)
        kernel_x_inv = np.linalg.inv(shape_kernel_x(ell_x))
        prior_precision = np.kron(kernel_x_inv, kernel_u_inv)
        prior_logdet = -(n_grid * logdet_x + N_SHAPE_KNOTS * logdet_u)
        return gp_log_marginal_likelihood(data_term, data_vector, prior_precision, prior_logdet)

    def data_term_for(ell_x):
        kernel_x_inv = np.linalg.inv(shape_kernel_x(ell_x))
        weight_interp = squared_exponential(line_position, shape_knot_x, 1.0, ell_x) @ kernel_x_inv
        data_term = np.zeros((n_dim, n_dim))
        data_vector = np.zeros(n_dim)
        for m in range(n_lines):
            design = np.hstack([weight_interp[m, j] * design_rows[m] for j in range(N_SHAPE_KNOTS)])
            w = weights[m]
            data_term += (design * w[:, None]).T @ design
            data_vector += (design * w[:, None]).T @ targets[m]
        return data_term, data_vector, weight_interp

    ell_x_lower, ell_x_upper = np.exp(SHAPE_LOG_BOUNDS[2][0]), np.exp(SHAPE_LOG_BOUNDS[2][1])
    ell_x_candidates = np.geomspace(ell_x_lower, ell_x_upper, 60)

    best_evidence, best_theta, best_cache = -np.inf, None, None
    for ell_x in ell_x_candidates:
        data_term, data_vector, weight_interp = data_term_for(ell_x)
        _, logdet_x = np.linalg.slogdet(shape_kernel_x(ell_x))

        inner_bounds = SHAPE_LOG_BOUNDS[:2]
        inner_result = fit_hyperparameters(
            inner_bounds,
            lambda theta: evidence_given_data(theta[0], theta[1], data_term, data_vector,
                                               logdet_x, ell_x),
            n_restarts=3)
        value = evidence_given_data(inner_result[0], inner_result[1], data_term, data_vector,
                                     logdet_x, ell_x)
        if value > best_evidence:
            best_evidence = value
            best_theta = (inner_result[0], inner_result[1], np.log(ell_x))
            best_cache = (data_term, data_vector, weight_interp)

    log_amplitude, log_ell_u, log_ell_x = best_theta
    data_term, data_vector, weight_interp = best_cache
    amplitude, ell_u, ell_x = np.exp(log_amplitude), np.exp(log_ell_u), np.exp(log_ell_x)

    kernel_u_inv = np.linalg.inv(shape_kernel_u(amplitude, ell_u))
    kernel_x_inv = np.linalg.inv(shape_kernel_x(ell_x))
    prior_precision = np.kron(kernel_x_inv, kernel_u_inv)

    normal_matrix = prior_precision + data_term
    normal_vector = data_vector
    posterior_covariance = np.linalg.inv(normal_matrix)
    unconstrained = posterior_covariance @ normal_vector
    constraint_gram = constraint_matrix @ posterior_covariance @ constraint_matrix.T
    lagrange = np.linalg.solve(constraint_gram,
                                constraint_target - constraint_matrix @ unconstrained)
    solution = unconstrained + posterior_covariance @ constraint_matrix.T @ lagrange

    return solution.reshape(N_SHAPE_KNOTS, n_grid), (log_amplitude, log_ell_u, log_ell_x)


# =========================================================================
# 7. Wavelength calibration: a Gaussian Process departure from a low-order
#    polynomial trend
# =========================================================================
DISPERSION_TREND_ORDER = 5
N_DISPERSION_KNOTS = 20
lambda_min, lambda_max = wavelength.min(), wavelength.max()
lambda_span = lambda_max - lambda_min
dispersion_knot_lambda = np.linspace(lambda_min, lambda_max, N_DISPERSION_KNOTS)

def rescaled_wavelength(lam):
    return 2 * (lam - lambda_min) / (lambda_max - lambda_min) - 1

def trend_design_matrix(lam):
    return Chebyshev.chebvander(rescaled_wavelength(lam), DISPERSION_TREND_ORDER)

trend_basis = trend_design_matrix(wavelength)

DISPERSION_LOG_BOUNDS = [
    (np.log(1e-3), np.log(1.0)),
    (np.log(lambda_span / 30), np.log(lambda_span * 2)),
]

def dispersion_kernel(amplitude, length_scale):
    return squared_exponential(dispersion_knot_lambda, dispersion_knot_lambda,
                                amplitude, length_scale)

def fit_dispersion(width_state, shape_knot_values, shape_hyperparameters,
                    trend_coeffs_init, knot_values_init,
                    log_amplitude_init, log_length_scale_init,
                    n_gauss_newton_steps=4, step_size=0.5,
                    finite_difference_step=1e-3):
    log_tau0, width_knot_values, width_length_scale = width_state
    trend_coeffs = trend_coeffs_init.copy()
    knot_values = knot_values_init.copy()
    log_amplitude, log_length_scale = log_amplitude_init, log_length_scale_init
    n_dim = (DISPERSION_TREND_ORDER + 1) + N_DISPERSION_KNOTS

    for _ in range(n_gauss_newton_steps):
        length_scale_for_linearisation = np.exp(log_length_scale)
        interp_weights_for_linearisation = squared_exponential(
            wavelength, dispersion_knot_lambda, 1.0, length_scale_for_linearisation)
        line_position = trend_basis @ trend_coeffs + \
            interp_weights_for_linearisation @ knot_values
        line_width = width_from_knots(line_position, log_tau0, width_knot_values, width_length_scale)
        departure = evaluate_shape_departure(line_position, shape_knot_values,
                                              shape_hyperparameters)

        model_values, model_derivatives = [], []
        for m in range(n_lines):
            idx = fit_window(line_position[m])
            conv = convolution_matrix(line_position[m], idx)
            lsf = gaussian_mean(u, line_width[m]) + departure[m]
            model_value = conv @ lsf
            conv_shifted = convolution_matrix(line_position[m] + finite_difference_step, idx)
            model_derivative = (conv_shifted @ lsf - model_value) / finite_difference_step
            model_values.append(model_value)
            model_derivatives.append(model_derivative)

        def build_design_for_length_scale(length_scale):
            interp_weights = squared_exponential(wavelength, dispersion_knot_lambda,
                                                  1.0, length_scale)
            design_rows, targets, weights = [], [], []
            for m in range(n_lines):
                idx = fit_window(line_position[m])
                model_derivative = model_derivatives[m]
                row = np.zeros((len(idx), n_dim))
                row[:, :DISPERSION_TREND_ORDER + 1] = \
                    trend_basis[m][None, :] * model_derivative[:, None]
                row[:, DISPERSION_TREND_ORDER + 1:] = \
                    model_derivative[:, None] * interp_weights[m][None, :]
                target = flux[idx] - model_values[m] + model_derivative * line_position[m]
                design_rows.append(row)
                targets.append(target)
                weights.append(inverse_variance[idx])

            design_matrix = np.vstack(design_rows)
            target_vector = np.concatenate(targets)
            weight_vector = np.concatenate(weights)
            prior_precision = np.zeros((n_dim, n_dim))
            data_term, data_vector = gp_evidence_terms(design_matrix, target_vector,
                                                         weight_vector, prior_precision)
            return data_term, data_vector, interp_weights

        log_amplitude, log_length_scale, data_term, data_vector, interp_weights = \
            fit_length_scale_and_amplitude(
                DISPERSION_LOG_BOUNDS[1], DISPERSION_LOG_BOUNDS[0],
                build_design_for_length_scale, dispersion_kernel, n_dim,
                slice(DISPERSION_TREND_ORDER + 1, None))

        prior_precision = np.zeros((n_dim, n_dim))
        prior_precision[DISPERSION_TREND_ORDER + 1:, DISPERSION_TREND_ORDER + 1:] = \
            np.linalg.inv(dispersion_kernel(np.exp(log_amplitude), np.exp(log_length_scale)))
        proposed = np.linalg.solve(prior_precision + data_term, data_vector)
        current = np.concatenate([trend_coeffs, knot_values])
        updated = current + step_size * (proposed - current)
        trend_coeffs, knot_values = (updated[:DISPERSION_TREND_ORDER + 1],
                                       updated[DISPERSION_TREND_ORDER + 1:])

    final_position = trend_basis @ trend_coeffs + \
        squared_exponential(wavelength, dispersion_knot_lambda, 1.0,
                             np.exp(log_length_scale)) @ knot_values
    return final_position, trend_coeffs, knot_values, log_amplitude, log_length_scale


# =========================================================================
# 8. Joint iterative solution
# =========================================================================
trend_coeffs = np.linalg.lstsq(trend_basis, peak_pixel, rcond=None)[0]
_trend_residual = peak_pixel - trend_basis @ trend_coeffs
_initial_length_scale = lambda_span / 5
_initial_interp = squared_exponential(wavelength, dispersion_knot_lambda, 1.0,
                                       _initial_length_scale)
_initial_kernel_inv = np.linalg.inv(dispersion_kernel(0.1, _initial_length_scale))
dispersion_knot_values = np.linalg.solve(
    _initial_interp.T @ _initial_interp + _initial_kernel_inv,
    _initial_interp.T @ _trend_residual)
line_position = trend_basis @ trend_coeffs + _initial_interp @ dispersion_knot_values
print(f"initial dispersion guess: rms(position - input peak) = "
      f"{np.sqrt(np.mean((line_position - peak_pixel)**2)):.4f} pix")
dispersion_log_amplitude, dispersion_log_length_scale = np.log(0.1), np.log(lambda_span / 2)

log_tau0 = np.log(TAU_REFERENCE)
width_knot_values = np.zeros(N_WIDTH_KNOTS)
width_log_amplitude, width_log_length_scale = np.log(0.1), np.log(x_span * 2)

shape_knot_values = np.zeros((N_SHAPE_KNOTS, n_grid))
shape_hyperparameters = (np.log(0.05), np.log(1.0), np.log(x_span / 5))

N_OUTER_ITERATIONS = 3

for iteration in range(N_OUTER_ITERATIONS):
    width_length_scale = np.exp(width_log_length_scale)
    shape_knot_values, shape_hyperparameters = fit_shape_departure(
        line_position, log_tau0, width_knot_values, width_length_scale, shape_hyperparameters)

    log_tau0, width_knot_values, width_log_amplitude, width_log_length_scale = fit_width(
        line_position, shape_knot_values, shape_hyperparameters,
        log_tau0, width_knot_values, width_log_amplitude, width_log_length_scale)
    width_length_scale = np.exp(width_log_length_scale)

    line_position, trend_coeffs, dispersion_knot_values, \
        dispersion_log_amplitude, dispersion_log_length_scale = fit_dispersion(
            (log_tau0, width_knot_values, width_length_scale), shape_knot_values, shape_hyperparameters,
            trend_coeffs, dispersion_knot_values,
            dispersion_log_amplitude, dispersion_log_length_scale)

    line_width = width_from_knots(line_position, log_tau0, width_knot_values, width_length_scale)
    position_change = np.sqrt(np.mean((line_position - peak_pixel)**2))
    print(f"iteration {iteration}: "
          f"|position - input| (rms) = {position_change:.4f} pix, "
          f"tau(x) in [{line_width.min():.3f}, {line_width.max():.3f}] pix, "
          f"FWHM in [{2.355 * line_width.min():.2f}, {2.355 * line_width.max():.2f}] pix, "
          f"shape amplitude = {np.exp(shape_hyperparameters[0]):.4f}, "
          f"dispersion GP amplitude = {np.exp(dispersion_log_amplitude):.4f} pix")


# =========================================================================
# 9. Diagnostics
# =========================================================================
width_length_scale = np.exp(width_log_length_scale)
line_width = width_from_knots(line_position, log_tau0, width_knot_values, width_length_scale)
departure = evaluate_shape_departure(line_position, shape_knot_values, shape_hyperparameters)

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

np.savez(
    'lsf_gp_results.npz',
    u=u, shape_knot_values=shape_knot_values, shape_hyperparameters=np.array(shape_hyperparameters),
    shape_knot_x=shape_knot_x,
    log_tau0=log_tau0, width_knot_values=width_knot_values, width_knot_x=width_knot_x,
    width_log_length_scale=width_log_length_scale,
    trend_coeffs=trend_coeffs, dispersion_knot_values=dispersion_knot_values,
    dispersion_knot_lambda=dispersion_knot_lambda, dispersion_log_length_scale=dispersion_log_length_scale,
    line_position=line_position, line_width=line_width,
    x_min=x_min, x_max=x_max, wavelength=wavelength,
    flux=flux, flux_err=flux_err, model=model, fitted_mask=fitted_mask, pixel=pixel,
    envelope_grid_full=envelope_grid_full, background_grid_full=background_grid_full,
    peak_flux=peak_flux, boundary_flux=boundary_flux,
    peak_pixel=peak_pixel, boundary_pixel=boundary_pixel,
    flux_raw=flux_raw,
)
print("Results written to lsf_gp_results.npz")