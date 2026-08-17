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
from numpy.polynomial import chebyshev as Chebyshev

from lfc.fitting.gp import gaussian_process_smooth  # still used by envelope,
                                                       # background, and
                                                       # width (both
                                                       # components) --
                                                       # migrated one
                                                       # component at a
                                                       # time, dispersion
                                                       # first
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels
from scipy.optimize import minimize  # dispersion only, for now

SPEED_OF_LIGHT = 2.99792458e8       # m / s
C_LIGHT_KMS = SPEED_OF_LIGHT / 1e3  # km / s

SPECTRUM_FILE = '/Users/dmilakov/software/harps/lsf/test/example_data_ESPRESSO_od=169.txt'
LINES_FILE = '/Users/dmilakov/software/harps/lsf/test/line_positions_ESPRESSO_od=169.txt'

#
def rational_quadratic_smooth(
    x: np.ndarray,
    z: np.ndarray,
    z_err: np.ndarray,
    x_grid: np.ndarray,
    n_restarts: int = 3,
    clip_sigma: float = 4.0,
    min_length_scale: float = None,
    max_length_scale: float = None,
    fixed_length_scale: float = None,
    fixed_alpha: float = None,
) -> dict:
    """
    tinygp/JAX equivalent of gp.py's gaussian_process_smooth, using a
    Rational Quadratic kernel instead of Matern-3/2.
 
    Chosen for dispersion specifically because RQ has a natural
    interpretation as a continuous MIXTURE of RBF kernels across a range
    of length scales (controlled by its alpha parameter): as alpha -> inf
    it reduces to a single-scale RBF, and smaller alpha blends in
    progressively shorter scales. This matches "smooth, almost
    polynomial, but might have sharper features" without needing to
    commit to a SPECIFIC second length scale the way the width model's
    explicit sum of two kernels does -- the data determines how much
    short-scale mixture is needed via alpha, rather than that being
    fixed in advance.
 
    Interface matches gp.py's gaussian_process_smooth exactly (same
    parameters, same returned dict keys) so it can be swapped in directly
    at call sites without touching anything downstream.
 
    Uses exact JAX gradients for the hyperparameter optimisation (via
    jax.grad through tinygp's log_probability), rather than gp.py's
    gradient-free L-BFGS-B-on-a-finite-difference-approximated-surface --
    this should be both faster and less prone to landing on a poor local
    optimum purely from noisy gradient estimates.
    """
    finite = np.isfinite(z) & np.isfinite(z_err) & (z_err > 0)
    x_f, z_f, e_f = x[finite], z[finite], z_err[finite]
    if len(x_f) < 4:
        return {
            'x_grid': x_grid, 'z_mean': np.full_like(x_grid, np.nan),
            'z_std': np.full_like(x_grid, np.nan),
            'length_scale': np.nan, 'signal_std': np.nan, 'n_used': 0,
            'alpha': np.nan,
        }
 
    # Sigma-clipping: identical scheme to gp.py, for consistency between
    # the two while both are in use during the migration.
    sort_idx = np.argsort(x_f)
    x_s, z_s, e_s = x_f[sort_idx], z_f[sort_idx], e_f[sort_idx]
    win = max(5, len(x_s) // 10)
    z_baseline = np.array([
        np.median(z_s[max(0, i - win):i + win + 1]) for i in range(len(z_s))
    ])
    dev = np.abs(z_s - z_baseline)
    mad = np.median(dev)
    sigma_scale = 1.4826 * mad if mad > 0 else np.std(z_s - z_baseline)
    good = dev <= clip_sigma * max(sigma_scale, np.median(e_s))
    x_c, z_c, e_c = x_s[good], z_s[good], e_s[good]
    if len(x_c) < 4:
        x_c, z_c, e_c = x_s, z_s, e_s
    n_used = len(x_c)
 
    # Normalise for numerical stability, same convention as gp.py.
    x_range = x_c.max() - x_c.min() if x_c.max() - x_c.min() > 0 else 1.0
    z_range = z_c.max() - z_c.min() if z_c.max() - z_c.min() > 0 else 1.0
    x_mu, x_n = x_c.mean(), (x_c - x_c.mean()) / x_range
    z_mu, z_n = z_c.mean(), (z_c - z_c.mean()) / z_range
    e_n = e_c / z_range
    x_grid_n = (x_grid - x_mu) / x_range
 
    x_n_j, z_n_j, e_n_j = jnp.asarray(x_n), jnp.asarray(z_n), jnp.asarray(e_n)
 
    default_log_l_bounds = (-3.0, 1.0)
    log_l_lower = (np.log(min_length_scale / x_range) if min_length_scale is not None
                   else default_log_l_bounds[0])
    log_l_upper = (np.log(max_length_scale / x_range) if max_length_scale is not None
                   else default_log_l_bounds[1])
    log_a_bounds = (-4.0, 2.0)
    log_alpha_bounds = (-2.0, 3.0)  # alpha from ~0.14 (strongly multi-scale)
                                      # to ~20 (close to a single-scale RBF)
 
    # If both are frozen, no optimisation is needed at all: skip straight
    # to the posterior mean/std at the fixed hyperparameters. This is the
    # RQ equivalent of gp.py's width freeze-after-warmup mechanism --
    # necessary here because RQ's extra alpha parameter, on top of the
    # length scale, made hyperparameter fitting considerably less stable
    # between repeated calls (confirmed directly on real dispersion data:
    # length scale ranging from 0.5 to 25 nm across 6 outer iterations,
    # where the single-parameter Matern-3/2 kernel settled on the same
    # value every time).
    both_frozen = fixed_length_scale is not None and fixed_alpha is not None
    if both_frozen:
        log_length_norm = np.log(fixed_length_scale / x_range)
    param_bounds = [log_a_bounds, (log_l_lower, log_l_upper), log_alpha_bounds]
 
    def build_gp(params):
        log_amp, log_length, log_alpha = params
        kernel = jnp.exp(2 * log_amp) * kernels.RationalQuadratic(
            alpha=jnp.exp(log_alpha), scale=jnp.exp(log_length))
        return GaussianProcess(kernel, x_n_j, diag=e_n_j**2)
 
    @jax.jit
    def neg_log_likelihood(params):
        return -build_gp(params).log_probability(z_n_j)
 
    grad_fn = jax.jit(jax.grad(neg_log_likelihood))
 
    rng = np.random.default_rng(0)
 
    if not both_frozen:
        # Neither frozen: full 3-parameter search, as before.
        log_l_mid = 0.5 * (log_l_lower + log_l_upper)
        init_guesses = [[0.0, log_l_mid, 0.0]]
        for _ in range(n_restarts - 1):
            init_guesses.append([
                rng.uniform(-2, 1),
                rng.uniform(log_l_lower, log_l_upper),
                rng.uniform(log_alpha_bounds[0], log_alpha_bounds[1]),
            ])
        best_nll, best_params = np.inf, np.array(init_guesses[0])
        for p0 in init_guesses:
            res = minimize(
                lambda p: float(neg_log_likelihood(jnp.asarray(p))), p0,
                jac=lambda p: np.asarray(grad_fn(jnp.asarray(p))),
                method='L-BFGS-B', bounds=param_bounds,
                options={'maxiter': 200, 'ftol': 1e-10},
            )
            if res.success and res.fun < best_nll:
                best_nll, best_params = res.fun, res.x
    else:
        # Both frozen: only the amplitude still needs fitting, a much
        # simpler and better-behaved 1D search.
        log_alpha_fixed = np.log(fixed_alpha)
 
        def neg_log_likelihood_amp_only(log_amp):
            return neg_log_likelihood(jnp.array([log_amp[0], log_length_norm, log_alpha_fixed]))
 
        grad_amp_only = jax.jit(jax.grad(lambda p: neg_log_likelihood_amp_only(p)))
        best_nll, best_log_amp = np.inf, 0.0
        for p0 in ([0.0] if n_restarts <= 1 else
                   [0.0] + list(rng.uniform(-2, 1, n_restarts - 1))):
            res = minimize(
                lambda p: float(neg_log_likelihood_amp_only(jnp.asarray(p))), [p0],
                jac=lambda p: np.asarray(grad_amp_only(jnp.asarray(p))),
                method='L-BFGS-B', bounds=[log_a_bounds],
                options={'maxiter': 200, 'ftol': 1e-10},
            )
            if res.success and res.fun < best_nll:
                best_nll, best_log_amp = res.fun, res.x[0]
        best_params = np.array([best_log_amp, log_length_norm, log_alpha_fixed])
 
    log_amp_opt, log_length_opt, log_alpha_opt = best_params
    gp_train = build_gp(best_params)
    cond_gp = gp_train.condition(z_n_j, jnp.asarray(x_grid_n)).gp
    z_mean_n = np.asarray(cond_gp.mean)
    z_var_n = np.asarray(cond_gp.variance)
 
    z_mean = z_mean_n * z_range + z_mu
    z_std = np.sqrt(np.maximum(z_var_n, 0.0)) * z_range
 
    return {
        'x_grid': x_grid,
        'z_mean': z_mean,
        'z_std': z_std,
        'length_scale': float(np.exp(log_length_opt) * x_range),
        'signal_std': float(np.exp(log_amp_opt) * z_range),
        'alpha': float(np.exp(log_alpha_opt)),
        'n_used': n_used,
    }

def rbf_smooth(
    x: np.ndarray,
    z: np.ndarray,
    z_err: np.ndarray,
    x_grid: np.ndarray,
    n_restarts: int = 3,
    clip_sigma: float = 4.0,
    min_length_scale: float = None,
    max_length_scale: float = None,
    fixed_length_scale: float = None,
) -> dict:
    """
    tinygp/JAX equivalent of gp.py's gaussian_process_smooth, using an RBF
    (squared-exponential) kernel -- tinygp's ExpSquared -- in place of
    Rational Quadratic for dispersion.

    Only two hyperparameters (amplitude, length scale), not three: RQ's
    extra alpha parameter was the specific source of the length-scale
    instability found when RQ was first tried here (length scale ranging
    0.5-25 nm across repeated calls on real data, confirmed directly,
    versus Matern-3/2 settling on the same value every time). Removing
    alpha removes that specific degree of freedom. A freeze mechanism is
    still supported (fixed_length_scale) and used the same way as
    before, but it is worth checking directly whether it is even still
    needed with only two hyperparameters rather than assuming so.

    Interface matches gp.py's gaussian_process_smooth (same parameters,
    same core returned dict keys; no 'alpha' key, since this kernel
    doesn't have one) so it can be swapped in at call sites directly.
    """
    finite = np.isfinite(z) & np.isfinite(z_err) & (z_err > 0)
    x_f, z_f, e_f = x[finite], z[finite], z_err[finite]
    if len(x_f) < 4:
        return {
            'x_grid': x_grid, 'z_mean': np.full_like(x_grid, np.nan),
            'z_std': np.full_like(x_grid, np.nan),
            'length_scale': np.nan, 'signal_std': np.nan, 'n_used': 0,
        }

    # Sigma-clipping: identical scheme to gp.py and rational_quadratic_smooth.
    sort_idx = np.argsort(x_f)
    x_s, z_s, e_s = x_f[sort_idx], z_f[sort_idx], e_f[sort_idx]
    win = max(5, len(x_s) // 10)
    z_baseline = np.array([
        np.median(z_s[max(0, i - win):i + win + 1]) for i in range(len(z_s))
    ])
    dev = np.abs(z_s - z_baseline)
    mad = np.median(dev)
    sigma_scale = 1.4826 * mad if mad > 0 else np.std(z_s - z_baseline)
    good = dev <= clip_sigma * max(sigma_scale, np.median(e_s))
    x_c, z_c, e_c = x_s[good], z_s[good], e_s[good]
    if len(x_c) < 4:
        x_c, z_c, e_c = x_s, z_s, e_s
    n_used = len(x_c)

    # Normalise for numerical stability, same convention as gp.py.
    x_range = x_c.max() - x_c.min() if x_c.max() - x_c.min() > 0 else 1.0
    z_range = z_c.max() - z_c.min() if z_c.max() - z_c.min() > 0 else 1.0
    x_mu, x_n = x_c.mean(), (x_c - x_c.mean()) / x_range
    z_mu, z_n = z_c.mean(), (z_c - z_c.mean()) / z_range
    e_n = e_c / z_range
    x_grid_n = (x_grid - x_mu) / x_range

    x_n_j, z_n_j, e_n_j = jnp.asarray(x_n), jnp.asarray(z_n), jnp.asarray(e_n)

    default_log_l_bounds = (-3.0, 1.0)
    log_l_lower = (np.log(min_length_scale / x_range) if min_length_scale is not None
                   else default_log_l_bounds[0])
    log_l_upper = (np.log(max_length_scale / x_range) if max_length_scale is not None
                   else default_log_l_bounds[1])
    log_a_bounds = (-4.0, 2.0)

    frozen = fixed_length_scale is not None
    if frozen:
        log_length_norm = np.log(fixed_length_scale / x_range)
    param_bounds = [log_a_bounds, (log_l_lower, log_l_upper)]

    def build_gp(params):
        log_amp, log_length = params
        kernel = jnp.exp(2 * log_amp) * kernels.ExpSquared(scale=jnp.exp(log_length))
        return GaussianProcess(kernel, x_n_j, diag=e_n_j**2)

    @jax.jit
    def neg_log_likelihood(params):
        return -build_gp(params).log_probability(z_n_j)

    grad_fn = jax.jit(jax.grad(neg_log_likelihood))

    rng = np.random.default_rng(0)

    if not frozen:
        log_l_mid = 0.5 * (log_l_lower + log_l_upper)
        init_guesses = [[0.0, log_l_mid]]
        for _ in range(n_restarts - 1):
            init_guesses.append([
                rng.uniform(-2, 1),
                rng.uniform(log_l_lower, log_l_upper),
            ])
        best_nll, best_params = np.inf, np.array(init_guesses[0])
        for p0 in init_guesses:
            res = minimize(
                lambda p: float(neg_log_likelihood(jnp.asarray(p))), p0,
                jac=lambda p: np.asarray(grad_fn(jnp.asarray(p))),
                method='L-BFGS-B', bounds=param_bounds,
                options={'maxiter': 200, 'ftol': 1e-10},
            )
            if res.success and res.fun < best_nll:
                best_nll, best_params = res.fun, res.x
    else:
        # Length scale frozen: only the amplitude still needs fitting.
        def neg_log_likelihood_amp_only(log_amp):
            return neg_log_likelihood(jnp.array([log_amp[0], log_length_norm]))

        grad_amp_only = jax.jit(jax.grad(lambda p: neg_log_likelihood_amp_only(p)))
        best_nll, best_log_amp = np.inf, 0.0
        for p0 in ([0.0] if n_restarts <= 1 else
                   [0.0] + list(rng.uniform(-2, 1, n_restarts - 1))):
            res = minimize(
                lambda p: float(neg_log_likelihood_amp_only(jnp.asarray(p))), [p0],
                jac=lambda p: np.asarray(grad_amp_only(jnp.asarray(p))),
                method='L-BFGS-B', bounds=[log_a_bounds],
                options={'maxiter': 200, 'ftol': 1e-10},
            )
            if res.success and res.fun < best_nll:
                best_nll, best_log_amp = res.fun, res.x[0]
        best_params = np.array([best_log_amp, log_length_norm])

    log_amp_opt, log_length_opt = best_params
    gp_train = build_gp(best_params)
    cond_gp = gp_train.condition(z_n_j, jnp.asarray(x_grid_n)).gp
    z_mean_n = np.asarray(cond_gp.mean)
    z_var_n = np.asarray(cond_gp.variance)

    z_mean = z_mean_n * z_range + z_mu
    z_std = np.sqrt(np.maximum(z_var_n, 0.0)) * z_range

    return {
        'x_grid': x_grid,
        'z_mean': z_mean,
        'z_std': z_std,
        'length_scale': float(np.exp(log_length_opt) * x_range),
        'signal_std': float(np.exp(log_amp_opt) * z_range),
        'n_used': n_used,
    }

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
ENVELOPE_MIN_LENGTH_SCALE = 100  # pixels; raise this to force a smoother
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
    gp_fit = rbf_smooth(regression_x, regression_z, regression_err,
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
# =========================================================================
# 5. Width model: sigma(x), now a velocity width (km/s)
# =========================================================================
# sigma(x) is represented directly on a dense pixel grid, fit via
# gp.py's gaussian_process_smooth -- the same approach just used for
# dispersion, and for the same reason: each line's pixel window gives a
# cheap, closed-form LOCAL correction (here, to log(sigma), so sigma
# stays positive after exponentiating) via linearising around the
# current estimate, and gaussian_process_smooth fits a smooth curve
# through (line_position, log_sigma + correction, uncertainty) directly.
# Off-grid evaluation (needed at positions other than the M comb lines,
# e.g. the order-midpoint reference used by the identifiability guard)
# uses np.interp against the fitted grid, the same pattern already used
# for the envelope and background above.

WIDTH_MIN_LENGTH_SCALE = 2000  # pixels; a floor alone was not sufficient to
                                  # fix the jagged FWHM(x) reported -- see
                                  # WIDTH_WARMUP_CALLS below and the docstring
                                  # of fit_width for why
WIDTH_MAX_LENGTH_SCALE = 1e4    # pixels

_initial_width_kms = 1.3 * _v_per_pixel_typical  # was 1.3 PIXELS previously;
                                                    # converted to the
                                                    # equivalent km/s using
                                                    # the typical scale, as a
                                                    # starting guess only

def width(x, log_sigma_grid):
    """ sigma(x), from a log(sigma) grid already fit at the pixel values
        in `pixel` (see fit_width below) -- interpolated the same way
        envelope(x)/background(x) interpolate their own GP-fitted grids. """
    return np.maximum(np.exp(np.interp(x, pixel.astype(float), log_sigma_grid)),
                       0.05 * _v_per_pixel_typical)

width_coeffs = np.full(n_pixels, np.log(_initial_width_kms))  # initial grid,
                                                                 # refined by
                                                                 # fit_width below


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

MAX_POSITION_DRIFT = 1.5  # pixels; caps cumulative drift from the
                             # catalogued (peak_pixel) position, across ALL
                             # outer iterations combined. Confirmed directly
                             # (order 149): the runaway drift seen at the
                             # order edge builds up gradually ACROSS outer
                             # iterations rather than in one large jump --
                             # each individual step stays under a pixel --
                             # so capping only a single step's delta did not
                             # help; the drift has to be capped relative to
                             # the original catalogue value instead. This
                             # substantially improved but did not fully
                             # eliminate the edge anomaly (chi2/dof dropped
                             # 693 -> 131 on that dataset, and the position
                             # residual went from an unbounded ~4 pixels to
                             # exactly this cap for the last few lines) --
                             # the underlying pull for those lines to drift
                             # further is still present, just no longer
                             # able to run away unboundedly.
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
DISPERSION_WARMUP_CALLS = 4  # Kept from the RQ version, though worth
                               # testing directly whether RBF (only 2
                               # hyperparameters, not 3) still needs this
                               # at all -- RQ's extra alpha parameter was
                               # the specific source of the instability
                               # that motivated freezing in the first
                               # place (length scale ranging 0.5-25 nm
                               # across 6 outer iterations on real data).
_dispersion_gp_state = {'calls': 0, 'frozen_length_scale': None}

# EDGE UNCERTAINTY INFLATION. Every line except the two at the very ends of
# the order has a neighbour pulling on the GP fit from both sides; the
# outermost lines only ever get pulled from one side, so nothing pushes
# back if their own local correction happens to be noisy or biased --
# confirmed directly on real data (order 149): the last line's per-line
# correction grew steadily larger across outer iterations (+0.13, +0.83,
# +1.73, +1.98 pixels) instead of settling, ending in a ~4 pixel position
# error where every other line in the order stayed within ~1 pixel, and
# that mis-registration is what produced an apparent second peak in the
# LSF there (confirmed separately: strengthening the LSF shape's own
# regularisation by 100x left the position error completely unchanged,
# so the problem was never in the shape fit -- it was already present at
# the position-fitting stage). This inflates delta_err for lines within
# EDGE_INFLATION_RANGE (in units of the local line spacing) of either end
# of the wavelength range, making the GP trust an isolated edge point's
# own noisy correction less and rely more on extrapolating the trend from
# its one-sided neighbours -- the direct fix for one-sided support, rather
# than a general damping change that would also slow convergence
# everywhere else in the order.
EDGE_INFLATION_RANGE = 3.0  # line spacings; how close to either edge this applies
EDGE_INFLATION_MAX_FACTOR = 5.0  # uncertainty multiplier right at the edge itself

def edge_uncertainty_inflation():
    """ Per-line multiplicative factor for delta_err, 1.0 in the interior
        and growing up to EDGE_INFLATION_MAX_FACTOR for lines within
        EDGE_INFLATION_RANGE line-spacings of either end of the
        wavelength range. """
    sorted_idx = np.argsort(wavelength)
    sorted_wavelength = wavelength[sorted_idx]
    typical_spacing = np.median(np.diff(sorted_wavelength))
    distance_to_min = sorted_wavelength - sorted_wavelength.min()
    distance_to_max = sorted_wavelength.max() - sorted_wavelength
    distance_to_nearest_edge = np.minimum(distance_to_min, distance_to_max)
    threshold = EDGE_INFLATION_RANGE * typical_spacing
    closeness = np.clip(1.0 - distance_to_nearest_edge / threshold, 0.0, 1.0)
    factor_sorted = 1.0 + (EDGE_INFLATION_MAX_FACTOR - 1.0) * closeness
    factor = np.empty_like(factor_sorted)
    factor[sorted_idx] = factor_sorted
    return factor

_edge_inflation_factor = None  # computed once, lazily, the first time it's needed

def fit_dispersion(width_coeffs, shape_coeffs, line_position_init,
                    n_outer_steps=4, step_size=0.5, finite_difference_step=1e-3):
    global _edge_inflation_factor
    if _edge_inflation_factor is None:
        _edge_inflation_factor = edge_uncertainty_inflation()

    line_position = line_position_init.copy()
    dispersion_gp_fit = None
    for _ in range(n_outer_steps):
        v_pix = velocity_per_pixel_from_positions(line_position)
        line_width = width(line_position, width_coeffs)
        departure = evaluate_departure(line_position, line_width, shape_coeffs)

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

        delta_err = delta_err * _edge_inflation_factor
        target = line_position + delta
        # Rational Quadratic (tinygp), not Matern-3/2 (gp.py) -- chosen for
        # its natural interpretation as a continuous MIXTURE of length
        # scales (via alpha), matching "smooth, almost polynomial, but
        # might have sharper features" without committing to a specific
        # second length scale up front. Checked directly against the old
        # kernel on real data before switching: the two are virtually
        # identical everywhere except the known, already-documented edge
        # region, where RQ's greater local flexibility makes it trust the
        # (already known to be unreliable) edge data more than Matern-3/2
        # does. That is exactly why MAX_POSITION_DRIFT and the edge
        # uncertainty inflation above still matter here -- they operate
        # on the fitted position regardless of which kernel produced it,
        # and are what keeps this extra flexibility from reintroducing
        # the edge instability those safeguards were built to contain.
        # RBF (ExpSquared), not Rational Quadratic: RQ was tried first,
        # and while it improved order 50 substantially (chi2/dof 13.6 ->
        # 3.6), it made order 160 worse (min(phi) -0.02 -> -0.08) even
        # after its length-scale instability was frozen out -- traced to
        # RQ's extra flexibility trusting the already-known-unreliable
        # edge-region data more than a single-length-scale kernel does.
        # RBF removes that specific source of extra flexibility. The
        # existing MAX_POSITION_DRIFT and edge uncertainty inflation
        # safeguards still apply regardless of kernel choice.
        dispersion_frozen = _dispersion_gp_state['frozen_length_scale']
        dispersion_gp_fit = rbf_smooth(
            wavelength, target, delta_err, wavelength, n_restarts=3,
            min_length_scale=DISPERSION_MIN_LENGTH_SCALE,
            max_length_scale=DISPERSION_MAX_LENGTH_SCALE,
            fixed_length_scale=dispersion_frozen)
        proposed = dispersion_gp_fit['z_mean']
        line_position = line_position + step_size * (proposed - line_position)
        # Cap CUMULATIVE drift from the catalogued position -- see the note
        # at MAX_POSITION_DRIFT above for why this, rather than capping
        # delta itself, is what actually works.
        line_position = peak_pixel + np.clip(line_position - peak_pixel,
                                               -MAX_POSITION_DRIFT, MAX_POSITION_DRIFT)

    _dispersion_gp_state['calls'] += 1
    if dispersion_frozen is None and _dispersion_gp_state['calls'] >= DISPERSION_WARMUP_CALLS:
        _dispersion_gp_state['frozen_length_scale'] = dispersion_gp_fit['length_scale']
        print(f"  dispersion RBF length scale now frozen at {dispersion_gp_fit['length_scale']:.3f} nm")

    return line_position, dispersion_gp_fit


# =========================================================================
# =========================================================================
# =========================================================================
# 7. LSF shape: departure from a Gaussian, via a genuine 2D Gaussian Process
# =========================================================================
#   phi(u,x) = G(u;sigma(x)) + f(u,x),   f ~ GP(0, k)
#   k((u,x),(u',x')) = a^2 * exp(-(u-u')^2/(2*l_u^2)) * exp(-(x-x')^2/(2*l_x^2))
#
# Replaces the earlier Gauss-Hermite version. That model put its
# flexibility entirely in x (h_i(x) fit freely, but multiplying FIXED,
# low-order Hermite functions of u) -- which structurally cannot
# represent a genuinely asymmetric u-profile no matter how many h_i are
# added, confirmed directly: fitting a Gaussian-plus-one-sided-exponential
# test profile plateaued at ~12% residual regardless of using 4 or 30
# Hermite orders. A real 2D GP does not have that ceiling: its
# flexibility comes from the kernel length scales, not from a fixed
# functional family, so a short enough l_u can represent shapes no
# Hermite truncation can reach.
#
# INDUCING POINTS IN BOTH DIRECTIONS, NOT A DENSE PER-LINE KERNEL. f is
# represented on a modest (N_U_INDUCING x N_X_INDUCING) grid D, not one
# value per line or per native pixel -- the same discipline that made
# every other GP in this script tractable, and specifically what an
# earlier, abandoned attempt at a similar 2D model got wrong (a dense
# kernel with a correlation length spanning many lines was numerically
# pathological). Cov(vec(D)) = kron(K_u, K_x) for row-major flattening
# (verified numerically before writing this). Values at any (u,x),
# including the native u grid and each line's own position, come from
# the standard GP conditional-mean interpolation, which -- thanks to the
# kernel's separability -- factors into an independent u-side weight
# matrix and x-side weight matrix (see evaluate_departure).
#
# HYPERPARAMETER STABILITY. Checked directly: as a GP's signal amplitude
# shrinks toward its own noise floor -- exactly the regime our departure
# amplitudes sit in (previously fitted h_3, h_4 ~ 0.01-0.04) -- the
# marginal likelihood's sensitivity to length scale collapses (a
# controlled test showed the likelihood's spread across a wide range of
# candidate length scales shrinking by two orders of magnitude as
# amplitude dropped from 0.5 to 0.02). Learning l_u AND l_x freely would
# be at least as prone to the wandering width's single length scale
# showed before it was frozen, likely worse with two coupled parameters.
# So NEITHER is learned by marginal likelihood here: l_u is fixed
# analytically, tied to the CURRENT width estimate
# (l_u = SHAPE_U_LENGTH_SCALE_FACTOR * sigma_ref) so it tracks width
# without being independently fit; l_x and the amplitude are fixed
# constants below. This is a deliberate simplification, not a hidden
# limitation -- evidence-based selection of l_x (analogous to width's
# freeze-after-warmup) is a natural extension once this simpler version
# is validated, not something skipped by oversight.
#
# The two directions that would let the departure mimic a pure WIDTH
# change or SHIFT in line centre are guarded exactly as in every earlier
# version of this section, evaluated at u_inducing rather than the full
# u grid, and applied identically to every x-inducing column (a single
# representative sigma_ref, not a per-column one -- width varies only
# ~10% across the order, judged not worth the added complexity of a
# position-dependent guard for this first version).

N_U_INDUCING = 21
N_X_INDUCING = 7
u_inducing = np.linspace(u.min(), u.max(), N_U_INDUCING)
x_inducing = np.linspace(x_min, x_max, N_X_INDUCING)

SHAPE_U_LENGTH_SCALE_FACTOR = 0.5  # l_u = this * current sigma_ref (km/s);
                                     # tied to width rather than fit
SHAPE_X_LENGTH_SCALE = 3000         # pixels; fixed for now (see module note
                                     # above on why l_x is not learned yet)
SHAPE_GP_AMPLITUDE = 0.0075         # fixed prior std of the departure.
                                     # NOT simply carried over from the old
                                     # 1D models' GP_SIGNAL_STD=0.05 --
                                     # checked directly and that value left
                                     # the prior ~45x weaker than the data
                                     # precision in this 2D joint system,
                                     # providing essentially no effective
                                     # regularisation (D's fitted std came
                                     # out at 0.34-0.43 against an intended
                                     # 0.05, and min(phi) as negative as
                                     # -1.5). This value brings prior and
                                     # data precision to a comparable scale
                                     # for THIS dataset; it is a fit-time
                                     # constant, not a universal one -- a
                                     # different dataset's S/N would shift
                                     # the right value
SHAPE_IDENTIFIABILITY_WEIGHT = 1e8
_JITTER = 1e-3  # relative to the unit-diagonal correlation matrices;
                  # confirmed necessary directly, not just conservative
                  # caution: with N_X_INDUCING points spread across the
                  # full order but a length scale several times their
                  # spacing, R_x is severely ill-conditioned (checked:
                  # raw condition number ~1e10, i.e. numerically singular
                  # in double precision) -- there are fewer effective
                  # degrees of freedom than inducing points once the
                  # length scale is this much longer than their spacing,
                  # and 1e-8 left the fit free to blow up to ~7x its
                  # intended amplitude scale (checked directly: D's std
                  # came out at 0.34 against a prior amplitude of 0.05).

def unit_rbf(a, b, length_scale):
    """ Amplitude-free RBF correlation matrix (diagonal 1 when a is b). """
    return np.exp(-(a[:, None] - b[None, :])**2 / (2 * length_scale**2))

def width_change_direction_inducing(sigma_ref):
    """ Same role as in every earlier version of this section, evaluated
        on u_inducing instead of the full u grid. """
    direction = gaussian_mean(u_inducing, sigma_ref) * u_inducing**2 / sigma_ref**3
    return direction / np.linalg.norm(direction)

def shift_direction_inducing(sigma_ref):
    direction = gaussian_mean(u_inducing, sigma_ref) * u_inducing / sigma_ref**2
    return direction / np.linalg.norm(direction)

def evaluate_departure(line_position, sigma_at_lines, shape_state):
    """ The (n_lines, n_grid) departure array, using the GP's own
        conditional-mean interpolation from the inducing grid D
        (shape_state) -- separable thanks to the kernel's structure, so
        this is an independent u-side and x-side weighting rather than a
        full joint interpolation. """
    D = shape_state
    l_u = SHAPE_U_LENGTH_SCALE_FACTOR * np.median(sigma_at_lines)
    R_u = unit_rbf(u_inducing, u_inducing, l_u) + _JITTER * np.eye(N_U_INDUCING)
    W_u_full = unit_rbf(u, u_inducing, l_u) @ np.linalg.inv(R_u)         # (n_grid, N_U_INDUCING)

    R_x = unit_rbf(x_inducing, x_inducing, SHAPE_X_LENGTH_SCALE) + _JITTER * np.eye(N_X_INDUCING)
    W_x_lines = unit_rbf(line_position, x_inducing, SHAPE_X_LENGTH_SCALE) @ np.linalg.inv(R_x)  # (n_lines, N_X_INDUCING)

    return (W_u_full @ D @ W_x_lines.T).T   # (n_lines, n_grid)

def fit_shape_departure(line_position, width_coeffs, v_pix):
    sigma_current = width(line_position, width_coeffs)
    sigma_ref = np.median(sigma_current)
    l_u = SHAPE_U_LENGTH_SCALE_FACTOR * sigma_ref

    R_u = unit_rbf(u_inducing, u_inducing, l_u) + _JITTER * np.eye(N_U_INDUCING)
    R_u_inv = np.linalg.inv(R_u)
    W_u_full = unit_rbf(u, u_inducing, l_u) @ R_u_inv                    # (n_grid, N_U_INDUCING)

    R_x = unit_rbf(x_inducing, x_inducing, SHAPE_X_LENGTH_SCALE) + _JITTER * np.eye(N_X_INDUCING)
    R_x_inv = np.linalg.inv(R_x)
    W_x_lines = unit_rbf(line_position, x_inducing, SHAPE_X_LENGTH_SCALE) @ R_x_inv  # (n_lines, N_X_INDUCING)

    width_dir = width_change_direction_inducing(sigma_ref)
    shift_dir = shift_direction_inducing(sigma_ref)
    guard_u = SHAPE_IDENTIFIABILITY_WEIGHT * (np.outer(width_dir, width_dir)
                                                 + np.outer(shift_dir, shift_dir))
    K_u_prior_inv = R_u_inv / SHAPE_GP_AMPLITUDE**2

    n_dim = N_U_INDUCING * N_X_INDUCING
    # GP prior precision (kron(K_u,K_x) inverse = kron(K_u_inv,K_x_inv)) plus
    # the identifiability guard, which is BLOCK-DIAGONAL in x (kron(guard,I),
    # not kron(guard,K_x_inv)): it penalises each x-column's own shift/width
    # component independently, which is a different kind of constraint from
    # "smooth in x" and should not be entangled with the x-kernel's own
    # correlation structure.
    normal_matrix = np.kron(K_u_prior_inv, R_x_inv) + np.kron(guard_u, np.eye(N_X_INDUCING))
    normal_vector = np.zeros(n_dim)

    for m in range(n_lines):
        idx = fit_window(line_position[m])
        conv = convolution_matrix(line_position[m], idx, v_pix[m])
        sigma_m = sigma_current[m]
        gaussian_component = gaussian_mean(u, sigma_m)
        residual_target = flux[idx] - conv @ gaussian_component

        CW_m = conv @ W_u_full   # (len(idx), N_U_INDUCING)
        design_m = (CW_m[:, :, None] * W_x_lines[m, None, None, :]).reshape(len(idx), n_dim)

        weight = inverse_variance[idx]
        normal_matrix += design_m.T @ (weight[:, None] * design_m)
        normal_vector += design_m.T @ (weight * residual_target)

    solution = np.linalg.solve(normal_matrix, normal_vector)
    return solution.reshape(N_U_INDUCING, N_X_INDUCING)


WIDTH_WARMUP_CALLS = 4  # number of OUTER JOINT iterations' worth of fit_width
                          # calls to let the length scale fit freely before
                          # freezing it. Was 2 -- too early: at that point
                          # shape and dispersion have barely started
                          # converging (dispersion's own length scale takes
                          # a couple of iterations to lock in; see the
                          # printed diagnostics), so the width GP was being
                          # frozen against a still-shifting, immature model
                          # state. Confirmed directly: the SAME script,
                          # re-run, can freeze onto either a smooth (~3400
                          # pixel) or a much shorter, visibly wiggly length
                          # scale depending on exactly what that early
                          # state looked like -- a genuine gamble, not
                          # reliable behaviour. Waiting for more of the
                          # outer iteration to elapse first gives the
                          # freeze a much more representative, settled
                          # target to lock onto.
_width_gp_state = {'calls': 0, 'frozen_length_scale': None}

WIDTH_FINE_MIN_LENGTH_SCALE = 10    # pixels; deliberately short -- this
                                       # component exists specifically to
                                       # capture genuine fine-scale
                                       # structure the smooth component
                                       # above cannot represent by
                                       # construction. Believed to be a
                                       # real optical effect (present in
                                       # both LFC and Fabry-Perot data, so
                                       # not a comb-sampling artifact),
                                       # not something to smooth away or
                                       # treat as pure noise.
WIDTH_FINE_MAX_LENGTH_SCALE = 200   # pixels; capped well below the smooth
                                       # component's own scale (which
                                       # settles in the thousands of
                                       # pixels) so the two stay clearly
                                       # separated rather than redundant
WIDTH_FINE_WARMUP_CALLS = 4  # same freeze-after-warmup discipline as the
                               # smooth component, applied independently
_width_fine_gp_state = {'calls': 0, 'frozen_length_scale': None}

def fit_width(shape_coeffs, log_sigma_grid_init, line_position, v_pix,
              n_outer_steps=3, step_size=0.5, finite_difference_step=1e-3):
    """ Refines the width grid (log(sigma) at every pixel) the same way
        fit_dispersion refines position: each line's pixel window gives a
        cheap local correction (here to log(sigma), via linearising the
        model's response to a small multiplicative change in sigma), and
        gaussian_process_smooth fits the smooth curve through
        (line_position, log_sigma + correction, uncertainty) directly.

        The length scale is fit freely only for the first
        WIDTH_WARMUP_CALLS calls, then FROZEN at whatever value it found
        and reused for every subsequent call, rather than being re-fit
        from scratch every time. This is different from the equivalent
        dispersion function, which re-fits its length scale every call
        with no trouble -- and the reason for the difference matters: the
        log(sigma) correction signal is small (order 0.02-0.1) and
        comparable to its own noise, unlike dispersion's strong, stable
        trend, so each fresh re-fit of the width GP's length scale can
        land on a noticeably different value (confirmed directly: 10000,
        10000, then 6243 pixels across just 3 calls, despite a stated
        floor of 2000). Since fit_width is called repeatedly across many
        outer iterations with a DAMPED update each time, a length scale
        that wanders between calls means the final grid is a superposition
        of several different "few-broad-bump" patterns, each from a
        different call, rather than one consistent smooth curve -- which
        is what actually produced the jagged-looking FWHM(x), not a
        length-scale bound being violated (checked directly: the bound
        was respected every time; the instability was between calls, not
        within any single one). Freezing removes that source of
        inconsistency directly.

        AFTER this smooth component converges, a SEPARATE, independent
        fine-scale residual GP is fit on top of it, mirroring the
        coupled-trend-plus-independent-residual pattern already used for
        the background: one more per-line local-correction pass, this
        time measuring departure from the smooth prediction specifically,
        fed to its own gaussian_process_smooth call bounded to a short
        length scale. Confirmed directly (via a sign-change test on the
        per-line residuals: 254/326 alternations vs ~163 expected for
        pure noise) that this fine-scale structure is not just
        measurement noise -- and per the user, the same pattern is seen
        in Fabry-Perot data too, which rules out a comb-sampling
        artifact and points to a genuine, repeatable optical effect,
        worth modelling rather than smoothing away or inflating past. """
    log_sigma_grid = log_sigma_grid_init.copy()
    frozen = _width_gp_state['frozen_length_scale']

    for _ in range(n_outer_steps):
        sigma_current = width(line_position, log_sigma_grid)
        log_sigma_current = np.log(sigma_current)
        # Departure recomputed EVERY step (not once, outside the loop): the
        # Gauss-Hermite functions' argument is u/sigma(x), so the departure
        # itself depends on the CURRENT width estimate, which changes each
        # step here.
        departure = evaluate_departure(line_position, sigma_current, shape_coeffs)

        delta = np.zeros(n_lines)
        delta_err = np.zeros(n_lines)
        for m in range(n_lines):
            idx = fit_window(line_position[m])
            conv = convolution_matrix(line_position[m], idx, v_pix[m])
            lsf = gaussian_mean(u, sigma_current[m]) + departure[m]
            model_value = conv @ lsf

            sigma_perturbed = sigma_current[m] * np.exp(finite_difference_step)
            lsf_perturbed = gaussian_mean(u, sigma_perturbed) + departure[m]
            model_perturbed = conv @ lsf_perturbed
            model_derivative = (model_perturbed - model_value) / finite_difference_step

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

        target = log_sigma_current + delta
        if frozen is not None:
            min_ls, max_ls = frozen, frozen
        else:
            min_ls, max_ls = WIDTH_MIN_LENGTH_SCALE, WIDTH_MAX_LENGTH_SCALE
        width_gp_fit = gaussian_process_smooth(
            line_position, target, delta_err, pixel.astype(float), n_restarts=3,
            min_length_scale=min_ls, max_length_scale=max_ls)
        proposed_grid = width_gp_fit['z_mean']
        log_sigma_grid = log_sigma_grid + step_size * (proposed_grid - log_sigma_grid)

    _width_gp_state['calls'] += 1
    if frozen is None and _width_gp_state['calls'] >= WIDTH_WARMUP_CALLS:
        _width_gp_state['frozen_length_scale'] = width_gp_fit['length_scale']
        print(f"  width GP length scale now frozen at {width_gp_fit['length_scale']:.1f} pix")

    # Fine-scale residual component (see docstring above). Computed ONCE
    # per call, using the now-converged smooth grid, and blended in with
    # the SAME step_size damping the smooth component uses -- since
    # sigma_current below already reflects any fine correction from a
    # PREVIOUS call, a well-converged fine component naturally produces a
    # shrinking delta each time, the same self-limiting behaviour that
    # already keeps the smooth component from diverging.
    sigma_current = width(line_position, log_sigma_grid)
    departure = evaluate_departure(line_position, sigma_current, shape_coeffs)

    fine_delta = np.zeros(n_lines)
    fine_delta_err = np.zeros(n_lines)
    for m in range(n_lines):
        idx = fit_window(line_position[m])
        conv = convolution_matrix(line_position[m], idx, v_pix[m])
        lsf = gaussian_mean(u, sigma_current[m]) + departure[m]
        model_value = conv @ lsf

        sigma_perturbed = sigma_current[m] * np.exp(finite_difference_step)
        lsf_perturbed = gaussian_mean(u, sigma_perturbed) + departure[m]
        model_perturbed = conv @ lsf_perturbed
        model_derivative = (model_perturbed - model_value) / finite_difference_step

        weight = inverse_variance[idx]
        denominator = np.sum(model_derivative**2 * weight)
        if denominator > 1e-12:
            fine_delta[m] = np.sum(model_derivative * weight * (flux[idx] - model_value)) / denominator
            naive_err = 1.0 / np.sqrt(denominator)
            residual = (model_value + model_derivative * fine_delta[m] - flux[idx]) / flux_err[idx]
            dof = max(len(idx) - 1, 1)
            chi2_reduced = np.sum(residual**2) / dof
            fine_delta_err[m] = naive_err * np.sqrt(max(chi2_reduced, 1.0))
        else:
            fine_delta[m] = 0.0
            fine_delta_err[m] = np.inf

    fine_frozen = _width_fine_gp_state['frozen_length_scale']
    if fine_frozen is not None:
        fine_min_ls, fine_max_ls = fine_frozen, fine_frozen
    else:
        fine_min_ls, fine_max_ls = WIDTH_FINE_MIN_LENGTH_SCALE, WIDTH_FINE_MAX_LENGTH_SCALE
    fine_gp_fit = gaussian_process_smooth(
        line_position, fine_delta, fine_delta_err, pixel.astype(float), n_restarts=3,
        min_length_scale=fine_min_ls, max_length_scale=fine_max_ls)
    log_sigma_grid = log_sigma_grid + step_size * fine_gp_fit['z_mean']

    _width_fine_gp_state['calls'] += 1
    if fine_frozen is None and _width_fine_gp_state['calls'] >= WIDTH_FINE_WARMUP_CALLS:
        _width_fine_gp_state['frozen_length_scale'] = fine_gp_fit['length_scale']
        print(f"  width FINE-SCALE GP length scale now frozen at {fine_gp_fit['length_scale']:.1f} pix")

    return log_sigma_grid, width_gp_fit, target, delta_err, fine_gp_fit


# =========================================================================
# 8. Joint iterative solution
# =========================================================================
line_position = peak_pixel.copy()  # bootstrap: start from the input LFC
                                     # line-list positions; refined below
shape_coeffs = np.zeros((N_U_INDUCING, N_X_INDUCING))
dispersion_gp_fit = None

N_OUTER_ITERATIONS = 6

for iteration in range(N_OUTER_ITERATIONS):
    v_pix = velocity_per_pixel_from_positions(line_position)

    shape_coeffs = fit_shape_departure(line_position, width_coeffs, v_pix)
    width_coeffs, width_gp_fit, width_raw_target, width_raw_target_err, width_fine_gp_fit = fit_width(
        shape_coeffs, width_coeffs, line_position, v_pix)
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
# Same idea for width: the GP's own posterior uncertainty on log(sigma),
# on the same pixel grid width(x, ...) interpolates against. Combined in
# quadrature from BOTH the smooth and fine-scale components, since they
# are independent GP fits.
width_log_sigma_std = np.sqrt(width_gp_fit['z_std']**2 + width_fine_gp_fit['z_std']**2)
line_width = width(line_position, width_coeffs)
departure = evaluate_departure(line_position, line_width, shape_coeffs)

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
    u=u, shape_coeffs=shape_coeffs, u_inducing=u_inducing, x_inducing=x_inducing,
    width_coeffs=width_coeffs,
    width_log_sigma_std=width_log_sigma_std,
    width_raw_target=width_raw_target, width_raw_target_err=width_raw_target_err,
    line_position=line_position, dispersion_position_std=dispersion_position_std,
    line_width=line_width, v_pix=v_pix,
    x_min=x_min, x_max=x_max, wavelength=wavelength,
    flux=flux, flux_err=flux_err, model=model, fitted_mask=fitted_mask, pixel=pixel,
    envelope_grid_full=envelope_grid_full, background_grid_full=background_grid_full,
    peak_flux=peak_flux, boundary_flux=boundary_flux,
    peak_pixel=peak_pixel, boundary_pixel=boundary_pixel,
    background_coeffs=background_coeffs, background_poly_order=BACKGROUND_POLY_ORDER,
    flux_raw=flux_raw,
)
print("Results written to lsf_reconstruction_results.npz")