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

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erf
import matplotlib.pyplot as plt
import matplotlib

plt.ion()  # non-blocking: windows opened during the fit stay open and
             # update without halting execution, so window 1 can be
             # inspected while width/dispersion/shape are still fitting

SPEED_OF_LIGHT = 2.99792458e8       # m / s
C_LIGHT_KMS = SPEED_OF_LIGHT / 1e3  # km / s
SPEED_OF_LIGHT_KMS = C_LIGHT_KMS    # alias used by the plotting sections below
EXPECTED_R = 145000  # resolving power, for the reference line on the FWHM panel
EXPECTED_FWHM_KMS = SPEED_OF_LIGHT_KMS / EXPECTED_R


DATA_SOURCE = 'spectrum'   # 'file' or 'harps
ORDER = 164             # echelle order to use; only read when DATA_SOURCE == 'spectrum'

FILENAME = ('/Users/dmilakov/projects/j1333/lfc/LFC_reduced_files/'
           '2023-02-22_ESPRESSO_S2D_LFC_FP_A.fits')
WAVEREFERENCE = ('/Users/dmilakov/projects/j1333/lfc/LFC_reduced_files/'
                 '2023-02-22_ESPRESSO_LFC_WAVE_MATRIX_A.fits')
# =========================================================================
# GP hyperparameter fitting: MAP with priors, for every component
# =========================================================================
# Every GP-based component in this script now goes through this ONE
# function, deliberately, rather than mixing hard-bounded optimisation
# (with ad hoc freeze-after-warmup patches for components where that
# turned out unstable) with whatever else. That mixture was flagged
# directly as a problem, not just a style preference: hard bounds let an
# optimiser sit exactly at a boundary for free, and we found a concrete
# case (the envelope) where that let amplitude get pushed to its bound
# trying to compensate for a length scale forced too long for the data's
# genuine short-scale structure, producing wild, unphysical extrapolation
# far from the training data. A proper prior does not have that failure
# mode: moving away from the expected region always costs something, so
# there is no free lunch at a boundary.
#
# Priors are LogNormal on every hyperparameter (equivalently, Normal on
# its log), since every hyperparameter here (an amplitude or a length
# scale) is strictly positive. Length-scale priors are set from what this
# project has already learned empirically about each component (see the
# call sites below for the specific values and the reasoning behind
# each); amplitude priors are set FROM THE DATA ITSELF, each time this is
# called: the normalised data's own standard deviation is used as the
# prior's centre, which is what "inspecting the data" concretely means
# here, rather than a hard-coded absolute number that would not transfer
# between datasets of different scales.
#
# Because every component now goes through the same function with a
# consistent scheme, none of them need the freeze-after-warmup patches
# built up over earlier turns for width and dispersion specifically --
# those existed only to work around hard-bounded optimisation's own
# instability, which a proper prior addresses directly. Confirmed this
# is not merely a hope but tested behaviour before removing that
# machinery (see the per-component notes at each call site).
#
# The objective and its gradient are defined at MODULE level, not as
# closures inside map_gp_fit, and JIT-compiled once per component count
# (1 or 2 -- the only cases this script needs) rather than fresh inside
# every call. This matters in practice, not just in principle: a closure
# redefined inside the function body gives JAX a brand-new function
# object every single call, so its JIT cache can never be reused --
# confirmed directly, this was making a full 6-outer-iteration run
# (dozens of map_gp_fit calls between width, dispersion, and their
# repeated inner steps) take long enough to hit this environment's
# execution timeout, entirely from repeated recompilation rather than
# actual computation.

_KERNEL_CLASSES = {'expsquared': kernels.ExpSquared, 'matern32': kernels.Matern32,
                    'matern52': kernels.Matern52}

def _make_kernel(params, kernel_types):
    kernel = None
    for k, kernel_type in enumerate(kernel_types):
        log_amp, log_length = params[2 * k], params[2 * k + 1]
        kernel_cls = _KERNEL_CLASSES[kernel_type]
        component = jnp.exp(2 * log_amp) * kernel_cls(scale=jnp.exp(log_length))
        kernel = component if kernel is None else kernel + component
    return kernel

# Cache of JIT-compiled (neg_log_posterior, grad, predict) triples, keyed
# by kernel_types (a tuple, e.g. ('matern32', 'matern32')) -- lazily
# populated the first time a given combination is used, then reused for
# every subsequent call with that same combination. This generalises
# what used to be two hand-written cases (n_components 1 and 2, always
# ExpSquared) to any mix of kernel families without needing a new
# hand-written function for every combination -- and keeps the same
# property that made this fast in the first place: a closure redefined
# inside map_gp_fit every call would give JAX a fresh function object
# each time and defeat its own compilation cache, which is exactly what
# was making full pipeline runs hit this environment's execution
# timeout before it was fixed (see the note on .predict() below for the
# other, larger part of that same fix).
_compiled_cache = {}

def _get_compiled_functions(kernel_types):
    key = tuple(kernel_types)
    if key not in _compiled_cache:
        def neg_log_posterior(params, x_n, z_n, e_n, prior_means, prior_stds):
            gp = GaussianProcess(_make_kernel(params, key), x_n, diag=e_n**2)
            nll = -gp.log_probability(z_n)
            log_prior = -0.5 * jnp.sum(((params - prior_means) / prior_stds)**2)
            return nll - log_prior

        def predict(params, x_n, z_n, e_n, x_grid_n):
            gp = GaussianProcess(_make_kernel(params, key), x_n, diag=e_n**2)
            return gp.predict(z_n, x_grid_n, return_var=True)

        neg_log_posterior_jit = jax.jit(neg_log_posterior)
        _compiled_cache[key] = (neg_log_posterior_jit, jax.jit(jax.grad(neg_log_posterior_jit)),
                                  jax.jit(predict))
    return _compiled_cache[key]

def map_gp_fit(
    x: np.ndarray,
    z: np.ndarray,
    z_err: np.ndarray,
    x_grid: np.ndarray,
    length_scale_priors: list,   # list of (mean_in_x_units, log_std), one
                                   # per additive kernel component -- always
                                   # a list, even for a single component,
                                   # for consistency across every call site
    amplitude_priors: list = None,  # list of (mean_normalised, log_std),
                                       # one per component; None -> every
                                       # component's prior is set from the
                                       # data's own normalised std
    kernel_types: list = None,   # list of 'expsquared' / 'matern32' /
                                    # 'matern52', one per component; None
                                    # -> every component uses 'expsquared'
                                    # (RBF), the original default
    n_restarts: int = 3,
    clip_sigma: float = 4.0,
) -> dict:
    """
    MAP (maximum a posteriori) fit of a GP whose kernel is a SUM of
    components, each with its own kernel family, length-scale prior, and
    amplitude prior, replacing hard bounds with proper LogNormal priors
    throughout -- see the module note above for why that matters, not
    just as a style choice.

    Kernel family matters, not just its length-scale prior: RBF
    (ExpSquared)'s spectral density falls off like a Gaussian in
    frequency, an extremely sharp cutoff, so it retains almost no
    high-frequency content once its length scale is set -- forcing that
    length scale long leaves it no way to explain any genuine local
    structure except by collapsing the length scale back down. Matern
    kernels' spectral densities fall off as a POWER LAW instead, a much
    heavier tail, so a Matern process can hold a long nominal length
    scale while still absorbing some local structure through that tail,
    rather than being forced to choose between the two. Confirmed
    directly on the envelope: a single unconstrained RBF kernel's true
    optimum was near 5.6 pixels, ~242,000 in log-posterior worse at 1000
    pixels even with the best possible amplitude there -- no reasonable
    prior could bridge that. The equivalent Matern32 comparison was
    NEGATIVE (long genuinely preferred over short with no prior fight at
    all), and a full MAP fit with only a mildly-tightened prior
    (log-std=0.15) landed within 10% of the intended 1000-pixel target.

    A sum of components (rather than one component alone) still helps on
    top of the kernel-family fix, not instead of it: even with Matern32,
    a single component's true optimum (43 pixels) undershot the intended
    long-scale target, because the likelihood still gets a vote. Splitting
    into a long and a short component removes the conflict directly: each
    is free to explain the scale of structure it's actually suited to.
    A second, independently-scaled component removes the need for either
    one to be distorted: each is free to explain the scale of structure
    it is actually suited to.

    Interface matches the project's earlier gaussian_process_smooth /
    rbf_smooth (same x, z, z_err, x_grid, n_restarts, clip_sigma), but
    takes a LIST of priors (one per component) instead of hard bounds,
    and returns 'length_scale' / 'signal_std' as lists (length 1 for a
    single-component fit) for the same reason -- one consistent
    interface regardless of how many components a given call uses.
    """
    finite = np.isfinite(z) & np.isfinite(z_err) & (z_err > 0)
    x_f, z_f, e_f = x[finite], z[finite], z_err[finite]
    n_components = len(length_scale_priors)
    if len(x_f) < 4:
        return {
            'x_grid': x_grid, 'z_mean': np.full_like(x_grid, np.nan),
            'z_std': np.full_like(x_grid, np.nan),
            'length_scale': [np.nan] * n_components,
            'signal_std': [np.nan] * n_components, 'n_used': 0,
        }

    # Sigma-clipping: same scheme used throughout this project.
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

    x_range = x_c.max() - x_c.min() if x_c.max() - x_c.min() > 0 else 1.0
    z_range = z_c.max() - z_c.min() if z_c.max() - z_c.min() > 0 else 1.0
    x_mu, x_n = x_c.mean(), (x_c - x_c.mean()) / x_range
    z_mu, z_n = z_c.mean(), (z_c - z_c.mean()) / z_range
    e_n = e_c / z_range
    x_grid_n = (x_grid - x_mu) / x_range

    x_n_j, z_n_j, e_n_j = jnp.asarray(x_n), jnp.asarray(z_n), jnp.asarray(e_n)

    if kernel_types is None:
        kernel_types = ['expsquared'] * n_components

    log_length_prior_means = [np.log(mean_px / x_range) for mean_px, _ in length_scale_priors]
    length_log_stds = [log_std for _, log_std in length_scale_priors]

    if amplitude_priors is None:
        emp_std = max(float(np.std(z_n)), 1e-3)
        log_amp_prior_means = [np.log(emp_std)] * n_components
        log_amp_prior_stds = [0.5] * n_components
    else:
        log_amp_prior_means = [np.log(mean_norm) for mean_norm, _ in amplitude_priors]
        log_amp_prior_stds = [log_std for _, log_std in amplitude_priors]

    # Interleaved as [log_amp_1, log_length_1, log_amp_2, log_length_2, ...],
    # matching the parameter ordering used everywhere below.
    prior_means = jnp.array(
        [v for k in range(n_components) for v in (log_amp_prior_means[k], log_length_prior_means[k])])
    prior_stds = jnp.array(
        [v for k in range(n_components) for v in (log_amp_prior_stds[k], length_log_stds[k])])

    neg_log_posterior_fn, grad_fn_raw, predict_fn_raw = _get_compiled_functions(kernel_types)

    def neg_log_posterior(params):
        return neg_log_posterior_fn(params, x_n_j, z_n_j, e_n_j, prior_means, prior_stds)

    def grad_fn(params):
        return grad_fn_raw(params, x_n_j, z_n_j, e_n_j, prior_means, prior_stds)

    rng = np.random.default_rng(0)
    base_init = []
    for k in range(n_components):
        base_init += [log_amp_prior_means[k], log_length_prior_means[k]]
    init_guesses = [base_init]
    for _ in range(n_restarts - 1):
        guess = []
        for k in range(n_components):
            guess += [rng.normal(log_amp_prior_means[k], log_amp_prior_stds[k]),
                      rng.normal(log_length_prior_means[k], length_log_stds[k])]
        init_guesses.append(guess)

    # Unlike the earlier hard-bounded versions, this is genuinely
    # unconstrained: the prior itself is what keeps the optimum away from
    # pathological values, rather than a box constraint the optimiser can
    # sit against for free.
    best_nlp, best_params = np.inf, np.array(init_guesses[0])
    for p0 in init_guesses:
        res = minimize(
            lambda p: float(neg_log_posterior(jnp.asarray(p))), p0,
            jac=lambda p: np.asarray(grad_fn(jnp.asarray(p))),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10},
        )
        if res.success and res.fun < best_nlp:
            best_nlp, best_params = res.fun, res.x

    z_mean_n, z_var_n = predict_fn_raw(jnp.asarray(best_params), x_n_j, z_n_j, e_n_j, jnp.asarray(x_grid_n))
    z_mean_n, z_var_n = np.asarray(z_mean_n), np.asarray(z_var_n)

    z_mean = z_mean_n * z_range + z_mu
    z_std = np.sqrt(np.maximum(z_var_n, 0.0)) * z_range

    length_scales_opt = [float(np.exp(best_params[2 * k + 1]) * x_range) for k in range(n_components)]
    signal_stds_opt = [float(np.exp(best_params[2 * k]) * z_range) for k in range(n_components)]

    return {
        'x_grid': x_grid,
        'z_mean': z_mean,
        'z_std': z_std,
        'length_scale': length_scales_opt,
        'signal_std': signal_stds_opt,
        'n_used': n_used,
    }


def poly_plus_gp_fit(
    x: np.ndarray,
    z: np.ndarray,
    z_err: np.ndarray,
    x_grid: np.ndarray,
    poly_degree: int,
    length_scale_prior: tuple,
    kernel_type: str = 'expsquared',
    n_restarts: int = 5,
) -> dict:
    """
    Fits z(x) as a FIXED-DEGREE CHEBYSHEV POLYNOMIAL TREND (via weighted
    least squares, giving an exact, closed-form coefficient covariance)
    PLUS an independent, small-amplitude GP on the residual -- the same
    coupled-trend-plus-independent-residual pattern already used for the
    background elsewhere in this script, applied here to fix a real,
    confirmed problem with fitting a smooth, large-dynamic-range trend
    (like the dispersion relation) as a single zero-mean GP.

    That single-GP approach was tried first and found to badly inflate
    the reported uncertainty: with no explicit trend to absorb the
    dispersion relation's own ~9000-pixel range, the GP's amplitude
    hyperparameter had to explain that whole range through its
    covariance alone (confirmed directly: fitted amplitude ~9086 pixels,
    essentially the data's own full range), and the resulting posterior
    std came out around 1.08 pixels -- ~37x the actual per-line
    measurement precision (~0.03 pixels). Trying a joint MAP fit of a
    degree-7 polynomial MEAN FUNCTION together with the GP hyperparameters
    made things numerically worse, not better: once the polynomial
    absorbed most of the trend, the GP's own residual amplitude was
    pushed to an extremely small, poorly-conditioned value, and the
    posterior variance computation suffered catastrophic cancellation --
    confirmed directly, the computed normalised variance came out LARGER
    than the normalised amplitude squared, which is mathematically
    impossible for a correctly-computed GP posterior.

    Splitting the polynomial fit out entirely -- ordinary weighted least
    squares, not part of the GP's own optimisation -- avoids both
    problems at once: the polynomial coefficients get a well-defined,
    exact covariance from standard linear regression, the GP only ever
    has to explain a small residual (so its own hyperparameters stay in
    a numerically comfortable range), and the two uncertainty sources
    combine in quadrature at every point. Also using a CHEBYSHEV basis
    (via numpy's chebvander on x rescaled to [-1,1]), not a raw power
    series: confirmed directly this matters for conditioning at degree 7
    (condition number 42 vs 14,120 for the equivalent power series).
    """
    x_lo, x_hi = x.min(), x.max()
    x_resc = 2 * (x - x_lo) / (x_hi - x_lo) - 1
    design = np.polynomial.chebyshev.chebvander(x_resc, poly_degree)
    weight = 1.0 / z_err**2
    normal_matrix = (design * weight[:, None]).T @ design
    normal_vector = (design * weight[:, None]).T @ z
    poly_coeffs = np.linalg.solve(normal_matrix, normal_vector)
    poly_coeffs_cov = np.linalg.inv(normal_matrix)

    poly_pred = design @ poly_coeffs
    poly_pred_var = np.einsum('ij,jk,ik->i', design, poly_coeffs_cov, design)

    # Clamped to [-1,1] -- NOT left free to extrapolate. This matters
    # concretely, not just in principle: unlike a GP, which reverts
    # toward its own prior mean away from training data, a polynomial has
    # no such safeguard and can diverge arbitrarily fast outside the
    # range it was fit on. Confirmed directly on the envelope (which,
    # unlike dispersion, needs values across the FULL pixel range, well
    # beyond where any peak/boundary measurement actually constrains it):
    # left unclamped, the fitted envelope reached 907,185 counts at pixel
    # 0 and -166,852 (negative -- physically impossible for a flux
    # envelope) at pixel 9000, against a typical scale of ~300,000.
    # Clamping holds the polynomial at its boundary value beyond the
    # training range, which combined with the GP residual's own already-
    # verified reversion to zero there, keeps the whole fit well-behaved
    # everywhere it gets evaluated, not just where there is data.
    x_grid_resc = np.clip(2 * (x_grid - x_lo) / (x_hi - x_lo) - 1, -1, 1)
    design_grid = np.polynomial.chebyshev.chebvander(x_grid_resc, poly_degree)
    poly_pred_grid = design_grid @ poly_coeffs
    poly_pred_var_grid = np.einsum('ij,jk,ik->i', design_grid, poly_coeffs_cov, design_grid)

    residual = z - poly_pred
    gp_fit = map_gp_fit(x, residual, z_err, x_grid, n_restarts=n_restarts,
                         length_scale_priors=[length_scale_prior],
                         kernel_types=[kernel_type])

    z_mean = poly_pred_grid + gp_fit['z_mean']
    z_std = np.sqrt(poly_pred_var_grid + gp_fit['z_std']**2)

    return {
        'x_grid': x_grid,
        'z_mean': z_mean,
        'z_std': z_std,
        'poly_coeffs': poly_coeffs,
        'poly_pred_residual_std': float(np.std(residual)),
        'gp_length_scale': gp_fit['length_scale'][0],
        'gp_signal_std': gp_fit['signal_std'][0],
    }

# 1. Load the exposure and the LFC line list
# =========================================================================
# Two interchangeable sources, selected by DATA_SOURCE. 'file' is the
# original round trip through the plain-text files this script always
# used; 'harps' reads directly from an in-memory harps.spectrum object
# for a single order, with no intermediate np.savetxt/np.loadtxt step at
# all. Only one of the two blocks below actually executes.


def _load_spectrum(filename, wavereference, overwrite=False):
    """Load the ESPRESSO spectrum object once, shared by both entry points."""
    import harps.spectrum as hc
    spec = hc.ESPRESSO(
        FILENAME,
        f0=7.40e9,
        fr=18e9,
        overwrite=overwrite,
        sOrder=40,
        wavereference=WAVEREFERENCE,
    )
    spec.process(fittype='gauss', do_comb_specific=True)
    return spec



if DATA_SOURCE == 'spectrum':

    spec = _load_spectrum(FILENAME, WAVEREFERENCE)
    
    linelist = spec['linelist']
    flux_raw = np.asarray(spec.flux[ORDER], dtype=float)
    err_raw = np.asarray(spec.error[ORDER], dtype=float)

    cut = np.where(linelist['order'] == ORDER)[0]
    cut2 = np.where(spec.line_positions['order'] == ORDER)[0]
    # Each row of spec.line_positions[cut2] is itself a (order, left,
    # centre, right) record; appending the matching freq column gives the
    # same (order, left edge, centre, right edge, freq_Hz) layout the
    # file format uses, so everything below this point is identical
    # regardless of which branch ran.
    line_list = np.array(
        [list(row[0]) + [row[1]]
         for row in np.transpose([spec.line_positions[cut2], linelist[cut]['freq']])],
        dtype=float,
    )
else:
    SPECTRUM_FILE = '/Users/dmilakov/software/harps/lsf/test/example_data_ESPRESSO_od=160.txt'
    LINES_FILE = '/Users/dmilakov/software/harps/lsf/test/line_positions_ESPRESSO_od=160.txt'
    spectrum = np.loadtxt(SPECTRUM_FILE, comments='#')
    flux_raw, err_raw, _bkg_col, _env_col = spectrum.T

    line_list = np.loadtxt(LINES_FILE, comments='#')

n_pixels = len(flux_raw)
pixel = np.arange(n_pixels)

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
    """ Sub-pixel corrected (x, y) peak via 3-point parabolic interpolation
        around the discrete maximum near each centre, fit in LOG space:
        ln(y) is exactly quadratic in position for a Gaussian peak, y
        itself is not, so fitting the parabola directly to y systematically
        under-corrects the true offset. Checked directly before this
        change: sampling a true Gaussian (sigma 0.8-2.0 pixels) at three
        consecutive pixels with a known injected sub-pixel offset
        (0.05-0.45 pixels) and reconstructing that offset, the raw-y
        parabola recovered it with a bias of -0.003 to -0.073 pixels
        (worse for narrower lines and larger offsets), while the log-y
        version recovered the injected offset to numerical precision in
        every case tested -- confirmed again just now on a synthetic
        Gaussian with an arbitrary offset and amplitude, recovering both
        exactly. Both coordinates still come from the same parabola, so
        the pair stays self-consistent.

        dx is clipped to +/-0.5 pixels: the correction is only physically
        meaningful as a refinement WITHIN the sampling interval that
        produced abs_pixel as the discrete argmax in the first place, but
        the denominator can be an arbitrarily small negative number (e.g.
        from integer-count quantisation), which without a clip produces
        an unbounded correction -- checked directly, holding (y0-y2)
        fixed, dx grows from -0.5 to -10000 as the denominator shrinks
        from -2.0 to -0.0001, with no protection against this from the
        >=0 branch, which only guards the opposite (positive-denominator)
        case. """
    x_peak = np.empty(len(centres))
    y_peak = np.empty(len(centres))
    for i, c in enumerate(centres):
        lo = max(int(round(c)) - half_width, 0)
        hi = min(int(round(c)) + half_width + 1, n_pixels)
        window = flux_raw[lo:hi]
        i_max = np.argmax(window)
        abs_pixel = lo + i_max
        if i_max == 0 or i_max == len(window) - 1:
            x_peak[i] = abs_pixel
            y_peak[i] = window[i_max]
            continue
        y0, y1, y2 = window[i_max - 1], window[i_max], window[i_max + 1]
        if y0 <= 0 or y1 <= 0 or y2 <= 0:
            # log undefined (e.g. a background-subtracted negative dip);
            # fall back to the uncorrected discrete maximum rather than
            # silently producing a NaN or an unphysical correction
            x_peak[i] = abs_pixel
            y_peak[i] = y1
            continue
        ln_y0, ln_y1, ln_y2 = np.log(y0), np.log(y1), np.log(y2)
        denominator = ln_y2 - 2 * ln_y1 + ln_y0
        if denominator >= 0:
            x_peak[i] = abs_pixel
            y_peak[i] = y1
        else:
            dx = np.clip(0.5 * (ln_y0 - ln_y2) / denominator, -0.5, 0.5)
            x_peak[i] = abs_pixel + dx
            y_peak[i] = np.exp(ln_y1 - (ln_y2 - ln_y0)**2 / (8 * denominator))
    return x_peak, y_peak

peak_pixel, peak_flux = local_peak_subpixel(peak_pixel)
boundary_pixel = np.unique(np.concatenate([left_edge, right_edge]))
boundary_flux = local_extremum(boundary_pixel, 'min')
peak_flux_err = err_raw[np.round(peak_pixel).astype(int)]
boundary_flux_err = err_raw[np.round(boundary_pixel).astype(int)]

BACKGROUND_POLY_ORDER = 4
# Envelope is fit from PEAK flux data only now -- not peak plus a
# boundary-derived estimate. That combination was the actual source of
# the excess waviness, not kernel choice: envelope_from_boundary = (
# boundary_flux - c0) / gain requires dividing by a 4th-order polynomial
# in position that has NO constant term, so it necessarily crosses zero
# somewhere across the order -- confirmed directly, its range came out
# roughly -35,000,000 to +2,300,000 (peak_flux's own range, for
# comparison, is a clean 205,000-403,000), with standard deviation ~48x
# larger and median relative error ~23x larger than peak_flux's. Every
# kernel choice tried still picked some of this noise up: Matern32,
# whose heavier spectral tail is exactly what let it hold a long nominal
# length scale, is for the same
# reason MORE able to track this noise than RBF's sharp cutoff was --
# and simply raising the near-zero-gain exclusion threshold did not
# behave predictably (checked directly: interior curvature went
# 4281 -> 3510 -> 3351 -> 2938 -> 29 -> 2128 as the threshold moved
# smoothly from 1e-8 to 0.03, landing on qualitatively different ALS
# solutions rather than improving smoothly). Excluding boundary-derived
# points from the ENVELOPE regression entirely, rather than tuning a
# threshold, gave the same low-curvature, stable result on every test
# (curvature ~149, matching -- slightly beating -- the smoothness of an
# earlier working version, identical across all 4 ALS iterations since
# the envelope fit no longer depends on the evolving background
# coefficients at all). background_coeffs is still fit from boundary_flux
# directly below, just not via this back-solved intermediate.
# Envelope's broad trend is now absorbed by a Chebyshev polynomial (fit
# by weighted least squares, exact coefficient covariance), the same
# poly_plus_gp_fit pattern used for dispersion and width above -- not a
# bare Matern32 GP explaining the whole trend through its own covariance.
# That bare-GP approach worked (checked directly: sane extrapolation,
# low curvature) once restricted to peak-flux-only data with a tightened
# prior, but it shared the same underlying mechanism as dispersion's
# confirmed problem: the GP's amplitude has to explain peak_flux's own
# ~200,000-count range through zero-mean covariance alone. A polynomial
# mean function removes that requirement directly, the same way it did
# for dispersion, rather than relying on a carefully-tuned prior to keep
# it in check.
ENVELOPE_POLY_DEGREE = 5
ENVELOPE_RESIDUAL_LENGTH_SCALE_PRIOR = (100, 1.0)  # (mean pixels, log-std);
                                                      # short, since this
                                                      # now only has to
                                                      # explain whatever
                                                      # local structure
                                                      # the polynomial
                                                      # trend leaves behind
ENVELOPE_KERNEL_TYPE = 'matern32'
# background_coeffs is no longer initialised here: fit_envelope_background
# computes it fresh from a closed-form solve every call (it never used a
# starting guess), and the module-level name below is set directly from
# that call's return value instead, so it always reflects the MOST
# RECENT fit rather than a placeholder that a later save/print could
# accidentally read instead of the true, current coefficients.

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

BACKGROUND_RESIDUAL_LENGTH_SCALE_PRIOR = (30, 0.6)  # (mean pixels, log-std);
                                             # much shorter than the
                                             # envelope's own prior,
                                             # deliberately, since this
                                             # term's entire purpose is to
                                             # capture what the smooth
                                             # coupled term cannot -- 30
                                             # pixels matches what this
                                             # settled on empirically
                                             # across earlier runs (order
                                             # 149: 27.9 pixels) before any
                                             # prior was in place

def fit_envelope_background(centre_positions):
    """ Full envelope+background fit, as a reusable function rather than
        inline top-level code: re-locate each line's peak flux near
        `centre_positions` (via local_peak_subpixel), fit envelope(x) and
        background(x) from those, and propagate their combined
        uncertainty into flux/flux_err.

        The previous version of this ran as a 4-iteration alternating
        (ALS-style) loop, refitting the envelope and background in turn.
        That loop is NOT reproduced here: the envelope's own inputs
        (peak flux, its position, its error) never depended on
        background_coeffs at all, so re-running the envelope fit after
        fitting background changed nothing -- confirmed directly on real
        data, "max coefficient change" was exactly 0 on every iteration
        after the first. Both halves of the old loop are a single
        closed-form (background) or single MAP (envelope) solve given
        their own inputs; there was nothing to alternate.

        `centre_positions` is a separate argument, deliberately NOT tied
        to the module-level `peak_pixel` (the fixed anchor
        MAX_POSITION_DRIFT clips line_position against elsewhere) -- this
        function is called once before the outer loop with the initial
        catalogue positions, and again after every outer iteration with
        the CONVERGED line_position from that iteration, so that a
        genuinely improved position estimate (from the LSF/dispersion
        fit) can feed back into a better peak-flux reading and hence a
        better envelope, rather than envelope having no path by which
        anything learned downstream could ever reach it. """
    env_peak_pixel, env_peak_flux = local_peak_subpixel(centre_positions)
    env_peak_flux_err = err_raw[np.clip(np.round(env_peak_pixel).astype(int), 0, n_pixels - 1)]

    gp_fit = poly_plus_gp_fit(env_peak_pixel, env_peak_flux, env_peak_flux_err,
                                pixel.astype(float), n_restarts=3,
                                poly_degree=ENVELOPE_POLY_DEGREE,
                                length_scale_prior=ENVELOPE_RESIDUAL_LENGTH_SCALE_PRIOR,
                                kernel_type=ENVELOPE_KERNEL_TYPE)
    envelope_grid = gp_fit['z_mean']
    envelope_std_grid = gp_fit['z_std']

    def envelope(x, _grid=envelope_grid):
        return np.interp(x, pixel.astype(float), _grid)

    def envelope_std(x, _grid=envelope_std_grid):
        return np.interp(x, pixel.astype(float), _grid)

    envelope_at_boundary = envelope(boundary_pixel)
    design = background_design_row(boundary_pixel, envelope_at_boundary)
    xtw = design.T * (1.0 / boundary_flux_err**2)
    normal_matrix = xtw @ design
    background_coeffs = np.linalg.solve(normal_matrix, xtw @ boundary_flux)
    background_coeffs_covariance = np.linalg.inv(normal_matrix)

    coupled_prediction_at_boundary = background_design_row(
        boundary_pixel, envelope(boundary_pixel)) @ background_coeffs
    boundary_residual = boundary_flux - coupled_prediction_at_boundary

    residual_gp_fit = map_gp_fit(
        boundary_pixel, boundary_residual, boundary_flux_err, pixel.astype(float),
        n_restarts=3, length_scale_priors=[BACKGROUND_RESIDUAL_LENGTH_SCALE_PRIOR])
    background_residual_grid = residual_gp_fit['z_mean']
    background_residual_std_grid = residual_gp_fit['z_std']

    def background_residual(x):
        return np.interp(x, pixel.astype(float), background_residual_grid)

    def background_residual_std(x):
        return np.interp(x, pixel.astype(float), background_residual_std_grid)

    def background(x):
        e = envelope(x)
        return background_design_row(x, e) @ background_coeffs + background_residual(x)

    envelope_grid_full = envelope(pixel.astype(float))
    background_grid_full = background(pixel.astype(float))

    # Propagating the envelope's and background's own estimation
    # uncertainty into flux_err, not just the raw per-pixel measurement
    # error err_raw -- see the module's derivation: flux = (F-B)/(E-B),
    # with F=flux_raw (known variance err_raw^2) and E, B themselves
    # uncertain and NOT independent, since B(x) = c_0 + poly(x)*E(x) is
    # built directly from E(x). N=F-B, D=E-B; dD/dE=1-poly(x),
    # dD/dc_p=-x^p*E(x); dN/dE=-poly(x), dN/dc_p=-x^p*E(x) (dN/dF=1),
    # combined via the standard multivariate delta method and the
    # standard ratio-of-correlated-variables formula for Var(flux).
    poly_gain = sum(background_coeffs[p] * rescaled_position(pixel.astype(float))**p
                    for p in range(1, BACKGROUND_POLY_ORDER + 1))
    sigma_E = envelope_std(pixel.astype(float))
    design_row_full = background_design_row(pixel.astype(float), envelope_grid_full)
    coeff_variance_term = np.einsum('ij,jk,ik->i', design_row_full,
                                     background_coeffs_covariance, design_row_full)
    residual_variance_term = background_residual_std(pixel.astype(float))**2

    var_D = (1 - poly_gain)**2 * sigma_E**2 + coeff_variance_term + residual_variance_term
    var_N = err_raw**2 + poly_gain**2 * sigma_E**2 + coeff_variance_term + residual_variance_term
    cov_ND = poly_gain * (poly_gain - 1) * sigma_E**2 + coeff_variance_term + residual_variance_term

    N_full = flux_raw - background_grid_full
    D_full = envelope_grid_full - background_grid_full
    flux_new = N_full / D_full
    flux_err_new = np.abs(flux_new) * np.sqrt(
        np.maximum(var_N / N_full**2 + var_D / D_full**2 - 2 * cov_ND / (N_full * D_full), 0.0))

    print(f"  envelope/background: envelope residual GP length scale = "
          f"{gp_fit['gp_length_scale']:.1f} pix, background residual GP length scale = "
          f"{residual_gp_fit['length_scale'][0]:.1f} pix, signal std = "
          f"{residual_gp_fit['signal_std'][0]:.1f}, "
          f"background(x) coefficients = {np.round(background_coeffs, 6)}")

    return {
        'envelope_grid_full': envelope_grid_full,
        'background_grid_full': background_grid_full,
        'flux': flux_new,
        'flux_err': flux_err_new,
        'env_peak_pixel': env_peak_pixel,
        'env_peak_flux': env_peak_flux,
        'background_coeffs': background_coeffs,
    }


envelope_background_fit = fit_envelope_background(peak_pixel)
envelope_grid_full = envelope_background_fit['envelope_grid_full']
background_grid_full = envelope_background_fit['background_grid_full']
flux = envelope_background_fit['flux']
flux_err = envelope_background_fit['flux_err']
inverse_variance = 1.0 / flux_err**2
background_coeffs = envelope_background_fit['background_coeffs']

# =========================================================================
# WINDOW 1: raw data + envelope + background, and background/envelope
# ratio, shown NOW (before width/dispersion/shape have even started)
# rather than waiting for the whole fit to finish -- non-blocking, via
# plt.ion() above, so this window stays open and the script keeps
# running underneath it.
# =========================================================================
fig1, axes1 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
ax = axes1[0]
ax.plot(pixel, flux_raw, lw=0.5, color='0.6', label='raw flux')
ax.plot(pixel, envelope_grid_full, 'r-', lw=1, label='envelope E(x)')
ax.plot(pixel, background_grid_full, 'b-', lw=1, label='background B(x)')
ax.plot(envelope_background_fit['env_peak_pixel'], envelope_background_fit['env_peak_flux'], 'r.', ms=3)
ax.plot(boundary_pixel, boundary_flux, 'b.', ms=3)
ax.set_ylim(0, envelope_background_fit['env_peak_flux'].max() * 1.15)
ax.legend(fontsize=8)
ax.set_title('Envelope and background (initial fit; refit each outer iteration)')

ax = axes1[1]
ax.plot(pixel, background_grid_full / envelope_grid_full, color='purple', lw=1)
ax.axhline(0, color='gray', lw=0.5)
ax.set_title('Background / envelope ratio')
ax.set_xlabel('pixel')
axes1[0].set_xlim(peak_pixel.min() - 50, peak_pixel.max() + 50)
fig1.tight_layout()
fig1.canvas.draw()
plt.pause(0.1)
print("Window 1 (envelope/background) shown -- fitting continues underneath it.")


# =========================================================================
# 3. Wavelength calibration model
# =========================================================================
# x(lambda) is represented directly by its value at the M known comb
# wavelengths, line_position, fit via map_gp_fit -- not a polynomial, and
# not a bespoke kernel. This reuses the SAME utility already used for
# the envelope and background above, rather than building separate GP
# machinery just for this.
#
# map_gp_fit needs (x, z, z_err) triples: here x=wavelength,
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
# dense fit over ~400 points is exactly what map_gp_fit already does
# successfully for the envelope and background above, at a similar
# number of points, without the ill-conditioning that motivated inducing
# points elsewhere.
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

PIXEL_SUBSAMPLE = 31    # native subpixel resolution, carried over in spirit:
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
# ANTIALIAS_WIDTH is gone (see the module note that used to live here, and
# the discussion that led to this change): pixel integration is now
# handled EXACTLY rather than fit, following Schmidt & Bouchy (2024) --
# see gaussian_pixel_integral and the rebuilt convolution_matrix below.


def gaussian_pixel_integral(pixel_indices, line_centre, sigma, v_pix):
    """ EXACT analytic integral of the peak-normalised Gaussian LSF core
        (same function as gaussian_mean) over each pixel's boundaries, via
        the Gauss error function -- Schmidt & Bouchy (2024) eq. 3, adapted
        to a peak-normalised (not unit-area) profile. Replaces
        conv @ gaussian_mean(u, sigma): no grid, no antialiasing
        parameter, no approximation. Phi is the standard normal CDF.

        The (sigma/v_pix) prefactor (rather than sigma alone) is needed
        because the OLD conv @ lsf this replaces integrated in PIXEL
        units (its (du/v_pix) factor), not velocity units -- confirmed
        directly: a flux-conservation smoke test against pixel_model_flux
        summed over a wide pixel range caught this as a factor-of-1/v_pix
        discrepancy before this fix, and matches the analytic total to
        high precision after it. """
    Phi = lambda z: 0.5 * (1 + erf(z / np.sqrt(2)))
    edge_lo = (pixel_indices - 0.5 - line_centre) * v_pix
    edge_hi = (pixel_indices + 0.5 - line_centre) * v_pix
    return (sigma / v_pix) * np.sqrt(2 * np.pi) * (Phi(edge_hi / sigma) - Phi(edge_lo / sigma))

def convolution_matrix(line_centre, pixel_indices, v_pix):
    """ (len(pixel_indices) x n_grid) matrix mapping LSF-grid values to
        pixel-integrated flux, built from the EXACT overlap between each
        pixel's boundary and each fine-grid cell [u_k-du/2, u_k+du/2] --
        a flux-conserving rebin, not a kernel convolution. This replaces
        the old Gaussian-antialiasing-kernel Riemann sum: a boxcar is the
        physically correct pixel-response shape (a pixel collects light
        uniformly across its width, nothing outside it), but point-
        sampling a boxcar kernel the way the old Gaussian kernel was
        sampled converges only at first order across the box's hard edge
        (checked directly: ~1-3% error against an exact erf ground truth
        at this grid's resolution, WORSE than the old, physically-wrong-
        but-smooth Gaussian kernel). Exact cell overlap has no such
        discontinuity to approximate -- checked directly against the same
        ground truth: ~3-11e-5 relative error, roughly an order of
        magnitude BETTER than the old method, at no extra cost.

        Used only for the non-parametric departure term now -- the
        parametric Gaussian core is integrated exactly via
        gaussian_pixel_integral instead, which needs no grid at all. """
    p = u / v_pix
    dp = du / v_pix
    pixel_lo = pixel_indices[:, None] - line_centre - 0.5
    pixel_hi = pixel_indices[:, None] - line_centre + 0.5
    cell_lo = p[None, :] - dp / 2
    cell_hi = p[None, :] + dp / 2
    return np.clip(np.minimum(pixel_hi, cell_hi) - np.maximum(pixel_lo, cell_lo), 0, None)

def pixel_model_flux(line_centre, pixel_indices, v_pix, sigma, departure_grid):
    """ Full pixel-integrated model flux for one line: the exact Gaussian
        core plus the departure term projected through the exact-overlap
        convolution_matrix. Consolidates the conv/lsf/model pattern that
        used to appear at every call site separately -- the split between
        "exact, no grid" and "grid-based, exact overlap" is made once
        here, rather than being re-implemented (and risking inconsistency)
        at each of fit_dispersion, fit_shape_departure, fit_width, and the
        diagnostics sections that all need this same quantity. """
    core = gaussian_pixel_integral(pixel_indices, line_centre, sigma, v_pix)
    conv = convolution_matrix(line_centre, pixel_indices, v_pix)
    return core + conv @ departure_grid

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
# map_gp_fit -- the same approach just used for
# dispersion, and for the same reason: each line's pixel window gives a
# cheap, closed-form LOCAL correction (here, to log(sigma), so sigma
# stays positive after exponentiating) via linearising around the
# current estimate, and map_gp_fit fits a smooth curve
# through (line_position, log_sigma + correction, uncertainty) directly.
# Off-grid evaluation (needed at positions other than the M comb lines,
# e.g. the order-midpoint reference used by the identifiability guard)
# uses np.interp against the fitted grid, the same pattern already used
# for the envelope and background above.

WIDTH_LENGTH_SCALE_PRIOR = (20, 2)  # (mean pixels, log-std). Replaces
                                          # the earlier two-GP structure
                                          # (a ~2000px "smooth" residual
                                          # plus a SEPARATE ~40px "fine"
                                          # residual) with a single
                                          # Matern32 residual on top of
                                          # the polynomial, per direct
                                          # request: this one GP now has
                                          # to do double duty (mild
                                          # residual smoothing AND
                                          # tracking the high-frequency
                                          # width oscillation), so its
                                          # length scale is chosen for
                                          # THAT, not for the old
                                          # residual's own separate role.
                                          # 20px is informed directly by
                                          # a Lomb-Scargle periodogram of
                                          # the raw per-line width
                                          # residual (order 160): a
                                          # clear, dominant period at
                                          # 64.4px (power 0.51, a strong
                                          # detection), not the near-
                                          # Nyquist alternation an
                                          # earlier, cruder sign-change
                                          # test wrongly suggested.
                                          # Checked directly with a fixed-
                                          # length-scale scan: the
                                          # earlier ~40-100px prior
                                          # clearly undercaptured this
                                          # (var_explained ~0.964),
                                          # while shrinking toward 20px
                                          # (~period/3, a standard rule
                                          # of thumb for a stationary
                                          # kernel tracking a known
                                          # oscillation) recovers most of
                                          # the improvement (~0.985)
                                          # without going so short that
                                          # var_explained keeps climbing
                                          # with no clear elbow (checked
                                          # down to 3px, still climbing --
                                          # a sign of fitting individual
                                          # point noise, not the genuine
                                          # periodic signal, since a plain
                                          # Matern32 has no way to lock
                                          # onto 64px specifically without
                                          # also picking up shorter-scale
                                          # noise; an ExpSineSquared
                                          # periodic term would target
                                          # this more directly and is
                                          # worth considering if this
                                          # matters further).
WIDTH_POLY_DEGREE = 5  # Chebyshev polynomial degree for the smooth width
                          # trend, mirroring DISPERSION_POLY_DEGREE's role.

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
# uncertainty, then let map_gp_fit fit a smooth curve through
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

MAX_POSITION_DRIFT = 1.0  # pixels; caps cumulative drift from the
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
DISPERSION_LENGTH_SCALE_PRIOR = (3, 0.5)  # (mean nm, log-std). RBF (not
                                              # Matern32): checked directly
                                              # -- Matern32's heavier
                                              # spectral tail, the exact
                                              # property that let it hold
                                              # a long, useful length
                                              # scale for the envelope,
                                              # works AGAINST us here.
                                              # With a tight prior at 3nm
                                              # (log-std=0.5), Matern32
                                              # still converged to
                                              # 488nm -- barely moved,
                                              # the likelihood
                                              # overwhelmed the prior --
                                              # while RBF converged to
                                              # 7.9nm regardless of
                                              # whether the prior was
                                              # centred at 3nm or 500nm.
                                              # RBF's sharp spectral
                                              # cutoff structurally can't
                                              # "cheat" with a
                                              # pathologically long
                                              # scale the way Matern32
                                              # can, so it's forced to
                                              # track the dispersion
                                              # relation's genuine, mild
                                              # curvature instead of
                                              # collapsing to a global
                                              # near-linear fit. This
                                              # matters concretely: a
                                              # length scale forced to
                                              # ~87x the order's own
                                              # ~11nm span (which is what
                                              # a loose (500,1.0) prior
                                              # on Matern32 produced)
                                              # left the GP unable to
                                              # explain real structure,
                                              # and that model mismatch
                                              # showed up as inflated
                                              # posterior uncertainty
                                              # (~1 pixel, ~37x the
                                              # actual per-line
                                              # measurement precision of
                                              # ~0.03 pixels) rather than
                                              # genuine uncertainty.
                                              # 3nm, a modest fraction of
                                              # the order's own span,
                                              # gives room to track
                                              # real within-order
                                              # curvature without being
                                              # so short it chases noise.
DISPERSION_KERNEL_TYPES = ['expsquared']
DISPERSION_POLY_DEGREE = 9  # Chebyshev polynomial degree for the smooth
                              # dispersion trend (see poly_plus_gp_fit) --
                              # a starting point, not derived from any
                              # specific requirement; worth checking
                              # whether the fit is sensitive to this
                              # choice if it matters for your use case

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
            model_value = pixel_model_flux(line_position[m], idx, v_pix[m],
                                            line_width[m], departure[m])
            model_shifted = pixel_model_flux(line_position[m] + finite_difference_step, idx,
                                              v_pix[m], line_width[m], departure[m])
            model_derivative = (model_shifted - model_value) / finite_difference_step

            weight = inverse_variance[idx]
            denominator = np.sum(model_derivative**2 * weight)
            if denominator > 1e-12:
                delta[m] = np.sum(model_derivative * weight * (flux[idx] - model_value)) / denominator
                naive_err = 1.0 / np.sqrt(denominator)
                residual = (model_value + model_derivative * delta[m] - flux[idx]) / flux_err[idx]
                # KNOWN, UNADDRESSED LIMITATION: dof only ever subtracts
                # THIS stage's own one parameter (delta). model_value
                # already incorporates width_coeffs and shape_coeffs, both
                # independently fit to this SAME pixel window by the other
                # two stages in the outer loop -- so chi2_reduced here is
                # computed against a model that has already had structure
                # from this exact data removed by fits elsewhere, without
                # reducing the nominal degrees of freedom to reflect that.
                # This is a standard way for iterative, coupled refitting
                # of the same data to systematically UNDERESTIMATE the
                # true residual uncertainty. A correct fix needs the joint
                # covariance across all three fits (or a cross-validation-
                # style held-out scheme), not a one-line patch here --
                # deliberately left undone rather than patched with
                # something unverified; the same limitation applies
                # identically to fit_width's copy of this block below.
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
        # RBF (ExpSquared) with a short, physically-motivated prior -- not
        # Matern32, and not a long prior. See DISPERSION_LENGTH_SCALE_PRIOR
        # above for the full reasoning: Matern32's heavier spectral tail
        # let it hold a pathologically long length scale (~87x the
        # order's own span) that collapsed the fit toward a global
        # near-linear trend and inflated posterior uncertainty roughly
        # 37x beyond the actual per-line measurement precision. RBF's
        # sharp cutoff can't do that -- checked directly, it converges to
        # a short, sensible length scale regardless of whether the prior
        # itself is centred short or long.
        # RBF (ExpSquared) residual GP, on top of a Chebyshev polynomial
        # trend fit by weighted least squares -- not a bare GP (single-
        # or mixed-kernel), and not a joint polynomial-mean-function MAP
        # fit. See poly_plus_gp_fit's docstring for the full story: a
        # bare zero-mean GP on the full dispersion relation inflated
        # posterior uncertainty ~37x beyond the actual per-line
        # measurement precision (had to explain the whole ~9000-pixel
        # range through its covariance alone), and a joint MAP fit of a
        # polynomial mean function fixed the point estimate but broke
        # the variance calculation numerically (posterior variance
        # exceeding the prior variance, confirmed directly -- impossible
        # for a correctly-computed GP). Splitting the polynomial out as
        # its own weighted-least-squares fit, with its own closed-form
        # coefficient covariance, avoids both problems.
        dispersion_fit = poly_plus_gp_fit(
            wavelength, target, delta_err, wavelength, n_restarts=5,
            poly_degree=DISPERSION_POLY_DEGREE,
            kernel_type=DISPERSION_KERNEL_TYPES[0],
            length_scale_prior=DISPERSION_LENGTH_SCALE_PRIOR)
        dispersion_gp_fit = {'z_mean': dispersion_fit['z_mean'], 'z_std': dispersion_fit['z_std'],
                               'length_scale': [dispersion_fit['gp_length_scale']]}
        proposed = dispersion_gp_fit['z_mean']
        line_position = line_position + step_size * (proposed - line_position)
        # Cap CUMULATIVE drift from the catalogued position -- see the note
        # at MAX_POSITION_DRIFT above for why this, rather than capping
        # delta itself, is what actually works.
        line_position = peak_pixel + np.clip(line_position - peak_pixel,
                                               -MAX_POSITION_DRIFT, MAX_POSITION_DRIFT)

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

N_U_INDUCING = 31
N_X_INDUCING = 32
u_inducing = np.linspace(u.min(), u.max(), N_U_INDUCING)
x_inducing = np.linspace(x_min, x_max, N_X_INDUCING)

SHAPE_U_LENGTH_SCALE_FACTOR = 0.8  # l_u = this * current sigma_ref (km/s); previously 0.5
                                     # tied to width rather than fit
SHAPE_X_LENGTH_SCALE = 3000         # pixels; fixed for now (see module note
                                     # above on why l_x is not learned yet)

# NON-STATIONARY PRIOR VARIANCE, kappa(u) -- replaces the old flat
# SHAPE_GP_AMPLITUDE with the Schmidt & Bouchy (2024, eq. 13-15) design:
# a squared-exponential CORRELATION (still SHAPE_U_LENGTH_SCALE_FACTOR *
# sigma_ref, unchanged) combined with a separately-shaped prior STD that
# is loose near the line core (where the LSF actually has structure to
# explain) and tight in the far wings (where it should decay to zero,
# same physical reasoning as their choice). Unlike their version, which
# fixes Lkappa=3.5 km/s as an absolute constant (tied to their assumed
# Lmu=2.3 km/s nominal resolution), the envelope width here is tied to
# the CURRENT sigma_ref, consistent with how l_u is already handled --
# this is a deliberate departure from a literal port, not an oversight,
# since a fixed km/s constant would silently stop making sense for a
# dataset with meaningfully different resolution.
SHAPE_KAPPA_SIGMA0 = 0.002        # floor prior std, far from the line core
                                     # (their sigma0=0.002, kept as-is: both
                                     # this and SHAPE_KAPPA_SIGMAF are in
                                     # departure-amplitude units, which for
                                     # us (unlike their flux-normalised LSF)
                                     # were already O(0.01-0.05), so this
                                     # floor is likely too tight -- treat
                                     # as a starting point, not carried
                                     # over as validated for this pipeline
SHAPE_KAPPA_SIGMAF = 0.05         # additional prior std at u=0, on top of
                                     # the floor (replaces the old flat
                                     # SHAPE_GP_AMPLITUDE=0.05 at the core)
SHAPE_KAPPA_WIDTH_FACTOR = 1.5    # width of the variance envelope, as a
                                     # multiple of sigma_ref (their own
                                     # Lkappa/Lmu ratio is 3.5/2.3 = 1.52,
                                     # so this reproduces their RELATIVE
                                     # shape while tying the absolute scale
                                     # to width, per the note above)
SHAPE_IDENTIFIABILITY_WEIGHT_SHIFT = 1e5  # unchanged -- pooling many lines'
                                     # individual position residuals through
                                     # an unguarded shift direction is what
                                     # produced the double-humped, unphysical
                                     # LSF when the guard was loosened wholesale
                                     # earlier; nothing since then argues this
                                     # direction is anything but genuinely
                                     # degenerate with line_position, so it
                                     # stays at full strength.
SHAPE_IDENTIFIABILITY_WEIGHT_WIDTH = 1e8  # loosened from the old shared 1e8,
                                     # as a direct, narrow test: the observed
                                     # residual bump sits almost exactly at
                                     # width_change_direction's own analytic
                                     # peak (sqrt(2)*sigma_ref), AND per-line
                                     # free width (fully unconstrained, no
                                     # smoothness at all) already failed to
                                     # explain the same bump -- meaning
                                     # whatever this direction would need to
                                     # express here is not simply "sigma(x)
                                     # is slightly wrong", which is the
                                     # specific degeneracy this guard exists
                                     # to prevent. Loosened by 1e5x (not to
                                     # zero) as a controlled first test, not
                                     # a decision that this direction is safe
                                     # to leave essentially unconstrained.
#
# SCOPE CAVEAT on both weights above: width_change_direction and
# shift_direction are derived as first-derivative (infinitesimal) tangent
# directions of the Gaussian with respect to sigma and position -- an
# exactly-degenerate statement only in that infinitesimal limit, applied
# here as a hard penalty on FINITE fitted coefficients (up to
# SHAPE_KAPPA_SIGMAF, non-negligible next to the core's own peak height of
# 1). Checked directly whether this matters in practice: projecting a
# departure of amplitude 0.01/0.03/0.05 purely along width_change_direction
# and asking what sigma a plain Gaussian fit would recover gives effective
# width changes of +0.47%/+1.42%/+2.37% -- close to linear in the
# coefficient, not blowing up, so the infinitesimal approximation does not
# appear to be breaking down at the amplitudes this model actually
# operates at. A real but currently non-biting theoretical scope mismatch,
# not something fixed here.
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

def _check_conditioning(matrix, name, max_condition=1e10):
    """ _JITTER (1e-3, defined above) is one fixed constant applied
        identically to every correlation matrix built in this section,
        regardless of matrix size or the length-scale-to-spacing ratio
        actually in play for a given fit -- both of which affect how
        small the smallest eigenvalue is even though jitter is added to
        a UNIT diagonal either way. There was previously no runtime check
        that a fixed 1e-3 was actually sufficient for whatever regime the
        outer loop's evolving sigma_ref/SHAPE_X_LENGTH_SCALE currently
        put the fit in, as opposed to the one specific case it was
        originally set against. This does not adapt jitter automatically
        -- it only reports the resulting condition number, so a
        genuinely insufficient case is visible rather than silent. """
    eigenvalues = np.linalg.eigvalsh(matrix)
    condition_number = eigenvalues.max() / max(eigenvalues.min(), 1e-300)
    if condition_number > max_condition:
        print(f"  WARNING: {name} condition number = {condition_number:.2e} "
              f"(exceeds {max_condition:.0e}) -- _JITTER may be insufficient "
              f"for the current length scale / grid spacing")
    return condition_number

def shape_kappa(u_grid, sigma_ref):
    """ Non-stationary prior std envelope kappa(u), Schmidt & Bouchy (2024)
        eq. 15 in form (their front normalisation factor is dropped here --
        it exists in their version only to express sigma0/sigma_f relative
        to their flux-normalised prior-mean peak, which has no equivalent
        in our zero-mean, dimensionless-departure GP): a constant floor
        plus a Gaussian bump giving extra freedom near the line core. """
    L_kappa = SHAPE_KAPPA_WIDTH_FACTOR * sigma_ref
    return SHAPE_KAPPA_SIGMA0 + SHAPE_KAPPA_SIGMAF * np.exp(-4 * np.log(2) * u_grid**2 / L_kappa**2)

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
        full joint interpolation.

        With a non-stationary u-side kernel K(u,u') = kappa(u) rho(u,u')
        kappa(u'), the conditional mean at a new point u* is
        kappa(u*) * [rho(u*,U) Sigma_inv] * (D / kappa(U)) -- i.e. the
        stationary-correlation weight matrix (rho @ Sigma_inv, identical
        in form to the old W_u_full) sandwiched between kappa(u*) and
        1/kappa(U). Folding both kappa factors directly into W_u_full
        keeps D itself, and every other line in this function, unchanged. """
    D = shape_state
    sigma_ref = np.median(sigma_at_lines)
    l_u = SHAPE_U_LENGTH_SCALE_FACTOR * sigma_ref
    R_u = unit_rbf(u_inducing, u_inducing, l_u) + _JITTER * np.eye(N_U_INDUCING)
    base_W_u = unit_rbf(u, u_inducing, l_u) @ np.linalg.inv(R_u)     # (n_grid, N_U_INDUCING)
    kappa_grid = shape_kappa(u, sigma_ref)                            # (n_grid,)
    kappa_inducing = shape_kappa(u_inducing, sigma_ref)                # (N_U_INDUCING,)
    W_u_full = (kappa_grid[:, None] * base_W_u) / kappa_inducing[None, :]

    R_x = unit_rbf(x_inducing, x_inducing, SHAPE_X_LENGTH_SCALE) + _JITTER * np.eye(N_X_INDUCING)
    W_x_lines = unit_rbf(line_position, x_inducing, SHAPE_X_LENGTH_SCALE) @ np.linalg.inv(R_x)  # (n_lines, N_X_INDUCING)

    return (W_u_full @ D @ W_x_lines.T).T   # (n_lines, n_grid)

# A genuinely deliberate duplication of fit_shape_departure's matrix
# construction, rather than a shared helper: the MAP solve path is
# already validated and in active use, and this diagnostic's only job is
# to answer one question (is l_x identifiable) without touching that path.
def shape_departure_log_evidence(candidate_l_x, line_position, width_coeffs, v_pix):
    sigma_current = width(line_position, width_coeffs)
    sigma_ref = np.median(sigma_current)
    l_u = SHAPE_U_LENGTH_SCALE_FACTOR * sigma_ref

    R_u = unit_rbf(u_inducing, u_inducing, l_u) + _JITTER * np.eye(N_U_INDUCING)
    R_u_inv = np.linalg.inv(R_u)
    kappa_grid = shape_kappa(u, sigma_ref)
    kappa_inducing = shape_kappa(u_inducing, sigma_ref)
    W_u_full = (kappa_grid[:, None] * (unit_rbf(u, u_inducing, l_u) @ R_u_inv)) / kappa_inducing[None, :]

    R_x = unit_rbf(x_inducing, x_inducing, candidate_l_x) + _JITTER * np.eye(N_X_INDUCING)
    R_x_inv = np.linalg.inv(R_x)
    W_x_lines = unit_rbf(line_position, x_inducing, candidate_l_x) @ R_x_inv

    width_dir = width_change_direction_inducing(sigma_ref)
    shift_dir = shift_direction_inducing(sigma_ref)
    guard_u = (SHAPE_IDENTIFIABILITY_WEIGHT_WIDTH * np.outer(width_dir, width_dir)
                + SHAPE_IDENTIFIABILITY_WEIGHT_SHIFT * np.outer(shift_dir, shift_dir))
    K_u_prior_inv = (R_u_inv / kappa_inducing[:, None]) / kappa_inducing[None, :]

    n_dim = N_U_INDUCING * N_X_INDUCING
    prior_precision = np.kron(K_u_prior_inv, R_x_inv) + np.kron(guard_u, np.eye(N_X_INDUCING))
    normal_matrix = prior_precision.copy()
    normal_vector = np.zeros(n_dim)

    n_data = 0
    log_weight_sum = 0.0
    y_weighted_sq_sum = 0.0

    for m in range(n_lines):
        idx = fit_window(line_position[m])
        sigma_m = sigma_current[m]
        residual_target = flux[idx] - gaussian_pixel_integral(idx, line_position[m], sigma_m, v_pix[m])

        conv = convolution_matrix(line_position[m], idx, v_pix[m])
        CW_m = conv @ W_u_full
        design_m = (CW_m[:, :, None] * W_x_lines[m, None, None, :]).reshape(len(idx), n_dim)

        weight = inverse_variance[idx]
        normal_matrix += design_m.T @ (weight[:, None] * design_m)
        normal_vector += design_m.T @ (weight * residual_target)

        n_data += len(idx)
        log_weight_sum += np.sum(np.log(weight))
        y_weighted_sq_sum += np.sum(weight * residual_target**2)

    _, logdet_prior = np.linalg.slogdet(prior_precision)
    _, logdet_post = np.linalg.slogdet(normal_matrix)
    solution = np.linalg.solve(normal_matrix, normal_vector)
    quad_term = y_weighted_sq_sum - normal_vector @ solution

    return (-0.5 * n_data * np.log(2 * np.pi) + 0.5 * log_weight_sum
             + 0.5 * logdet_prior - 0.5 * logdet_post - 0.5 * quad_term)

SHAPE_X_LENGTH_SCALE_PRIOR_MEAN = SHAPE_X_LENGTH_SCALE  # pixels -- the long-standing
                                     # 3000 px default is now the PRIOR CENTRE, not a
                                     # fixed value: confirmed directly (log-evidence
                                     # profiles spanning THOUSANDS of nats across three
                                     # real orders -- 164, 150, 50 -- none of them flat)
                                     # that l_x is strongly identifiable, not in the
                                     # "wandering, low-amplitude GP" regime the module's
                                     # earlier caution was about. Given that, this prior
                                     # is a mild numerical safety net, not a real
                                     # regulariser -- the evidence's own curvature
                                     # dominates the prior width below by orders of
                                     # magnitude for every order checked so far.
SHAPE_X_LENGTH_SCALE_PRIOR_LOG_STD = 1.0  # generous: ~2.2x up or down at 1 prior std
SHAPE_X_LENGTH_SCALE_BOUNDS = (200, 10000)  # pixels; matches the identifiability scan's
                                     # own range, comfortably inside where all three
                                     # checked profiles were still smooth and well-
                                     # behaved at both edges
SHAPE_X_LENGTH_SCALE_STEP_SIZE = 0.5  # same damping discipline used for every other
                                     # coupled parameter in this iterative scheme

def fit_shape_x_length_scale(line_position, width_coeffs, v_pix):
    """ MAP estimate of l_x: maximise the closed-form evidence
        (shape_departure_log_evidence) plus a LogNormal prior centred on
        SHAPE_X_LENGTH_SCALE_PRIOR_MEAN, following the same MAP-not-MLE
        discipline used everywhere else in this script. Confirmed directly
        (the l_x identifiability check, run after this loop, and the
        stand-alone scan across three orders before this was wired in)
        that this is safe to free: the evidence profile is sharply peaked,
        not flat, for every order checked so far -- and the peak location
        genuinely differs by order (roughly 1000-2000 px, versus the old
        fixed 3000 px, which sat past the peak for two of the three), so a
        single hand-picked constant was never going to be right for all
        of them simultaneously. """
    def negative_log_posterior(log_l_x):
        l_x = np.exp(log_l_x)
        log_ev = shape_departure_log_evidence(l_x, line_position, width_coeffs, v_pix)
        log_prior = -0.5 * ((log_l_x - np.log(SHAPE_X_LENGTH_SCALE_PRIOR_MEAN))
                              / SHAPE_X_LENGTH_SCALE_PRIOR_LOG_STD) ** 2
        return -(log_ev + log_prior)

    result = minimize_scalar(negative_log_posterior,
                               bounds=(np.log(SHAPE_X_LENGTH_SCALE_BOUNDS[0]),
                                       np.log(SHAPE_X_LENGTH_SCALE_BOUNDS[1])),
                               method='bounded', options={'xatol': 1e-3})
    return np.exp(result.x)


def fit_shape_departure(line_position, width_coeffs, v_pix):
    sigma_current = width(line_position, width_coeffs)
    sigma_ref = np.median(sigma_current)
    l_u = SHAPE_U_LENGTH_SCALE_FACTOR * sigma_ref

    R_u = unit_rbf(u_inducing, u_inducing, l_u) + _JITTER * np.eye(N_U_INDUCING)
    _check_conditioning(R_u, "R_u")
    R_u_inv = np.linalg.inv(R_u)
    kappa_grid = shape_kappa(u, sigma_ref)                              # (n_grid,)
    kappa_inducing = shape_kappa(u_inducing, sigma_ref)                  # (N_U_INDUCING,)
    W_u_full = (kappa_grid[:, None] * (unit_rbf(u, u_inducing, l_u) @ R_u_inv)) / kappa_inducing[None, :]

    R_x = unit_rbf(x_inducing, x_inducing, SHAPE_X_LENGTH_SCALE) + _JITTER * np.eye(N_X_INDUCING)
    _check_conditioning(R_x, "R_x")
    R_x_inv = np.linalg.inv(R_x)
    W_x_lines = unit_rbf(line_position, x_inducing, SHAPE_X_LENGTH_SCALE) @ R_x_inv  # (n_lines, N_X_INDUCING)

    width_dir = width_change_direction_inducing(sigma_ref)
    shift_dir = shift_direction_inducing(sigma_ref)
    guard_u = (SHAPE_IDENTIFIABILITY_WEIGHT_WIDTH * np.outer(width_dir, width_dir)
                + SHAPE_IDENTIFIABILITY_WEIGHT_SHIFT * np.outer(shift_dir, shift_dir))
    # K_u_prior = diag(kappa_inducing) @ R_u @ diag(kappa_inducing), so its
    # inverse is EXACTLY diag(1/kappa_inducing) @ R_u_inv @ diag(1/kappa_inducing)
    # -- no new matrix inversion needed, just row/column rescaling of R_u_inv.
    K_u_prior_inv = (R_u_inv / kappa_inducing[:, None]) / kappa_inducing[None, :]

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
        sigma_m = sigma_current[m]
        residual_target = flux[idx] - gaussian_pixel_integral(idx, line_position[m], sigma_m, v_pix[m])

        conv = convolution_matrix(line_position[m], idx, v_pix[m])
        CW_m = conv @ W_u_full   # (len(idx), N_U_INDUCING)
        design_m = (CW_m[:, :, None] * W_x_lines[m, None, None, :]).reshape(len(idx), n_dim)

        weight = inverse_variance[idx]
        normal_matrix += design_m.T @ (weight[:, None] * design_m)
        normal_vector += design_m.T @ (weight * residual_target)

    solution = np.linalg.solve(normal_matrix, normal_vector)
    return solution.reshape(N_U_INDUCING, N_X_INDUCING)


def fit_width(shape_coeffs, log_sigma_grid_init, line_position, v_pix,
              n_outer_steps=6, step_size=0.5, finite_difference_step=1e-3):
    """ Refines the width grid (log(sigma) at every pixel) the same way
        fit_dispersion refines position: each line's pixel window gives a
        cheap local correction (here to log(sigma), via linearising the
        model's response to a small multiplicative change in sigma), and
        map_gp_fit fits the smooth curve through (line_position,
        log_sigma + correction, uncertainty) directly, using a length-
        scale PRIOR rather than a hard bound (see the module note on
        map_gp_fit for why that distinction matters -- it removes the
        need for the freeze-after-warmup patch this function used to
        need, since the earlier instability came specifically from a
        hard-bounded optimiser being free to sit exactly at its boundary
        with no penalty, which a prior does not allow).

        AFTER this smooth component converges, a SEPARATE, independent
        fine-scale residual GP is fit on top of it, mirroring the
        coupled-trend-plus-independent-residual pattern already used for
        the background: one more per-line local-correction pass, this
        time measuring departure from the smooth prediction specifically,
        fed to its own map_gp_fit call with a much shorter length-scale
        prior. Confirmed directly (via a sign-change test on the per-line
        residuals: 254/326 alternations vs ~163 expected for pure noise)
        that this fine-scale structure is not just measurement noise --
        and per the user, the same pattern is seen in Fabry-Perot data
        too, which rules out a comb-sampling artifact and points to a
        genuine, repeatable optical effect, worth modelling rather than
        smoothing away or inflating past. """
    log_sigma_grid = log_sigma_grid_init.copy()

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
            model_value = pixel_model_flux(line_position[m], idx, v_pix[m],
                                            sigma_current[m], departure[m])

            sigma_perturbed = sigma_current[m] * np.exp(finite_difference_step)
            model_perturbed = pixel_model_flux(line_position[m], idx, v_pix[m],
                                                sigma_perturbed, departure[m])
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
        # See the identical block in fit_dispersion above for why this
        # chi2_reduced-based inflation is a known, unaddressed
        # underestimate of the true uncertainty (reused data across
        # coupled fits, without an effective-dof correction).

        target = log_sigma_current + delta
        width_fit = poly_plus_gp_fit(
            line_position, target, delta_err, pixel.astype(float), n_restarts=3,
            poly_degree=WIDTH_POLY_DEGREE,
            length_scale_prior=WIDTH_LENGTH_SCALE_PRIOR,
            kernel_type='matern32')
        width_gp_fit = {'z_mean': width_fit['z_mean'], 'z_std': width_fit['z_std'],
                          'length_scale': [width_fit['gp_length_scale']]}
        proposed_grid = width_gp_fit['z_mean']
        log_sigma_grid = log_sigma_grid + step_size * (proposed_grid - log_sigma_grid)

    return log_sigma_grid, width_gp_fit, target, delta_err


# =========================================================================
# 8. Joint iterative solution
# =========================================================================
line_position = peak_pixel.copy()  # bootstrap: start from the input LFC
                                     # line-list positions; refined below
shape_coeffs = np.zeros((N_U_INDUCING, N_X_INDUCING))
dispersion_gp_fit = None

N_OUTER_ITERATIONS = 10

# Convergence tolerances for the outer loop. Previously this loop had NO
# stopping criterion at all -- it always ran exactly N_OUTER_ITERATIONS
# times and reported whatever state existed afterward, with nothing
# checking whether that state had actually settled versus simply run out
# of budget. These track the step-to-step change in the three quantities
# the loop is jointly solving for; when all three fall below tolerance
# for CONVERGENCE_MIN_ITERATIONS consecutive iterations in a row, the
# loop stops early and says so explicitly. If the tolerance is never met,
# the loop says that explicitly too, rather than silently reporting a
# possibly-unconverged final state as if it were a settled answer.
CONVERGENCE_TOL_POSITION = 1e-4  # pixels, rms change in line_position
CONVERGENCE_TOL_WIDTH = 1e-4     # rms change in log(sigma) grid (width_coeffs)
CONVERGENCE_TOL_LX = 1.0         # pixels, change in SHAPE_X_LENGTH_SCALE
CONVERGENCE_MIN_ITERATIONS = 3   # never declare convergence before this many
                                    # iterations have run, so a spuriously
                                    # small early step (e.g. iteration 0,
                                    # before the joint fit has done
                                    # anything yet) cannot trigger a false
                                    # "converged" on its own

prev_line_position = line_position.copy()
prev_width_coeffs = width_coeffs.copy()
prev_l_x = SHAPE_X_LENGTH_SCALE
converged = False

for iteration in range(N_OUTER_ITERATIONS):
    v_pix = velocity_per_pixel_from_positions(line_position)

    proposed_l_x = fit_shape_x_length_scale(line_position, width_coeffs, v_pix)
    SHAPE_X_LENGTH_SCALE = SHAPE_X_LENGTH_SCALE + SHAPE_X_LENGTH_SCALE_STEP_SIZE * (
        proposed_l_x - SHAPE_X_LENGTH_SCALE)

    shape_coeffs = fit_shape_departure(line_position, width_coeffs, v_pix)
    width_coeffs, width_gp_fit, width_raw_target, width_raw_target_err = fit_width(
        shape_coeffs, width_coeffs, line_position, v_pix)
    line_position, dispersion_gp_fit = fit_dispersion(width_coeffs, shape_coeffs, line_position)

    # Envelope/background refit, using the position estimates just
    # produced by this iteration's dispersion fit -- not looped for its
    # own sake (see fit_envelope_background's docstring: alone, it has
    # nothing to alternate), but genuinely coupled to the rest of the
    # joint fit now: a converged line_position is, in general, a better
    # estimate of where each line's peak actually is than the initial
    # catalogue position, so re-reading peak flux there and refitting
    # envelope/background can improve on the initial fit, and the
    # improved flux/flux_err then feeds the NEXT iteration's shape/width/
    # dispersion fits in turn.
    envelope_background_fit = fit_envelope_background(line_position)
    envelope_grid_full = envelope_background_fit['envelope_grid_full']
    background_grid_full = envelope_background_fit['background_grid_full']
    flux = envelope_background_fit['flux']
    flux_err = envelope_background_fit['flux_err']
    inverse_variance = 1.0 / flux_err**2
    background_coeffs = envelope_background_fit['background_coeffs']

    line_width = width(line_position, width_coeffs)
    v_pix = velocity_per_pixel_from_positions(line_position)
    line_width_pix = line_width / v_pix   # for cross-checking against pixel-space intuition
    position_change = np.sqrt(np.mean((line_position - peak_pixel)**2))

    position_step = np.sqrt(np.mean((line_position - prev_line_position)**2))
    width_step = np.sqrt(np.mean((width_coeffs - prev_width_coeffs)**2))
    l_x_step = abs(SHAPE_X_LENGTH_SCALE - prev_l_x)
    step_converged = (position_step < CONVERGENCE_TOL_POSITION
                        and width_step < CONVERGENCE_TOL_WIDTH
                        and l_x_step < CONVERGENCE_TOL_LX)

    print(f"iteration {iteration}: "
          f"|position - input| (rms) = {position_change:.4f} pix, "
          f"sigma(v) in [{line_width.min():.4f}, {line_width.max():.4f}] km/s, "
          f"FWHM in [{2.355 * line_width.min():.3f}, {2.355 * line_width.max():.3f}] km/s "
          f"(~[{2.355 * line_width_pix.min():.2f}, {2.355 * line_width_pix.max():.2f}] pix), "
          f"dispersion GP length scale = {dispersion_gp_fit['length_scale'][0]:.3f} nm, "
          f"SHAPE_X_LENGTH_SCALE = {SHAPE_X_LENGTH_SCALE:.0f} pix\n"
          f"    step-to-step change: position={position_step:.2e} pix, "
          f"width={width_step:.2e}, l_x={l_x_step:.1f} pix"
          f"{'  [meets convergence tolerance]' if step_converged else ''}")

    prev_line_position = line_position.copy()
    prev_width_coeffs = width_coeffs.copy()
    prev_l_x = SHAPE_X_LENGTH_SCALE

    if step_converged and iteration >= CONVERGENCE_MIN_ITERATIONS:
        converged = True
        print(f"Outer loop converged after {iteration + 1} iterations (stopping early).")
        break

if not converged:
    print(f"Outer loop did NOT meet the convergence tolerance within "
          f"{N_OUTER_ITERATIONS} iterations -- reporting the final state anyway, "
          f"but treat it as a possibly-unconverged snapshot rather than a settled fit.")


# =========================================================================
# 8b. l_x FIT VALIDATION -- closed-form marginal likelihood profile
# =========================================================================
# l_x is now fit every outer iteration (fit_shape_x_length_scale, above),
# rather than fixed. This block re-checks the SAME profile post-convergence,
# now purely as a sanity plot confirming the converged SHAPE_X_LENGTH_SCALE
# actually sits at (or very near) the evidence peak, not as the identify-
# ability decision itself -- that question was already settled directly,
# across three real orders (164, 150, 50), each showing a sharply peaked
# profile spanning thousands of nats, well before this was wired into the
# main loop. See fit_shape_x_length_scale's docstring for the closed-form
# evidence formula and why a MAP (not MLE) discipline is used regardless.
RUN_LX_VALIDATION_PLOT = True  # set False to skip (each candidate costs
                                 # one full pass over all lines plus an
                                 # (n_dim x n_dim) slogdet+solve -- a few
                                 # seconds per point, so the whole scan
                                 # below is a couple of minutes)
if RUN_LX_VALIDATION_PLOT:
    v_pix = velocity_per_pixel_from_positions(line_position)
    candidate_l_x = np.logspace(np.log10(200), np.log10(10000), 20)  # pixels
    log_evidence = np.array([
        shape_departure_log_evidence(lx, line_position, width_coeffs, v_pix)
        for lx in candidate_l_x
    ])

    fig_lx, ax_lx = plt.subplots(figsize=(9, 6))
    ax_lx.plot(candidate_l_x, log_evidence - log_evidence.max(), 'o-')
    ax_lx.axvline(SHAPE_X_LENGTH_SCALE, color='tab:red', ls='--', lw=1.5,
                   label=f'converged fitted value ({SHAPE_X_LENGTH_SCALE:.0f} pix)')
    ax_lx.axvline(SHAPE_X_LENGTH_SCALE_PRIOR_MEAN, color='k', ls=':', lw=1,
                   label=f'prior centre ({SHAPE_X_LENGTH_SCALE_PRIOR_MEAN:.0f} pix)')
    ax_lx.set_xscale('log')
    ax_lx.set_xlabel('candidate SHAPE_X_LENGTH_SCALE (l_x) [pixels]')
    ax_lx.set_ylabel('log evidence, relative to max')
    ax_lx.set_title('l_x fit validation: converged value vs. the evidence profile')
    ax_lx.legend(fontsize=8)
    fig_lx.tight_layout()
    fig_lx.canvas.draw()
    plt.pause(0.1)

    peak_idx = np.argmax(log_evidence)
    spread = log_evidence.max() - log_evidence.min()
    print(f"\nl_x fit validation: evidence-scan peak at l_x={candidate_l_x[peak_idx]:.0f} pix, "

          f"total range across scan = {spread:.3f} nats")
    print("(as a rule of thumb: a range >> a few nats across this scan means l_x is "
          "meaningfully identifiable; a range of order 1 or less means the profile is "
          "close to flat, and a free optimiser would likely wander the way the module's "
          "own notes warn about for a low-amplitude GP)")



v_pix = velocity_per_pixel_from_positions(line_position)
# The GP's own posterior uncertainty on the fitted position, at each comb
# line's wavelength -- from the LAST outer iteration's dispersion fit, for
# plotting an honest uncertainty band rather than just the position itself.
dispersion_position_std = dispersion_gp_fit['z_std']
# Same idea for width: the GP's own posterior uncertainty on log(sigma),
# on the same pixel grid width(x, ...) interpolates against. Single
# component now (poly_plus_gp_fit's residual GP), not combined from a
# separate smooth+fine pair.
width_log_sigma_std = width_gp_fit['z_std']
line_width = width(line_position, width_coeffs)
departure = evaluate_departure(line_position, line_width, shape_coeffs)

model = np.zeros(n_pixels)
fitted_mask = np.zeros(n_pixels, dtype=bool)
lsf_per_line = np.zeros((n_lines, n_grid))
for m in range(n_lines):
    idx = fit_window(line_position[m])
    # INCLUDES departure now -- this was previously Gaussian-core only
    # (gaussian_pixel_integral with no + convolution_matrix @ departure term,
    # and lsf itself built with departure[m] commented out below). That
    # silently excluded the entire shape-departure fit from every diagnostic
    # that reads `model` (window 5's raw-flux plot, chi2_per_dof, and any
    # forward_model.txt-style export) -- confirmed directly: the resulting
    # per-line peak flux/model ratio was a tight, systematic ~1.4% undershoot
    # (std 0.5% across 318 lines), matching the scale of the departure
    # amplitude at line centre (SHAPE_KAPPA_SIGMA0 + SHAPE_KAPPA_SIGMAF), and
    # residuals many tens of sigma appeared almost exclusively in low-flux
    # wing pixels (88% of |residual|>50sigma points sit below 10% of that
    # line's own peak flux) -- exactly where a fixed few-percent-of-peak
    # absolute mismatch, divided by a photon-noise error that shrinks with
    # sqrt(flux), is mechanically amplified into a huge normalised residual,
    # not evidence the shape fit itself is bad. The properly departure-
    # inclusive, pixel-integrated comparison (pixel_model_flux, used
    # elsewhere in the fit and in window 2's model_pixel_segm export) shows
    # a near-zero core residual once this term is restored.
    lsf = gaussian_mean(u, line_width[m]) + departure[m]
    lsf_per_line[m] = lsf
    model[idx] += pixel_model_flux(line_position[m], idx, v_pix[m],
                                     line_width[m], departure[m])
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
_diag_sigma_ref = np.median(line_width)
_kappa_range = (shape_kappa(np.array([np.max(np.abs(u_inducing))]), _diag_sigma_ref)[0],
                shape_kappa(np.array([0.0]), _diag_sigma_ref)[0])
print(f"std(shape_coeffs) = {np.std(shape_coeffs):.4f}, "
      f"kappa(u) range at sigma_ref={_diag_sigma_ref:.3f}: "
      f"[{_kappa_range[0]:.4f} (wings), {_kappa_range[1]:.4f} (core)]")

# One more pass of the same per-line local correction fit_dispersion uses
# internally, at the now-fully-converged state -- needed for window 3's
# wavelength-calibration residual, which is NOT the same thing as the
# pixel-space "fitted position minus input peak pixel" plotted before:
# it is each line's own, individually-fit ("raw", pre-GP-smoothing)
# position compared against the final smooth dispersion solution,
# converted from pixels into a wavelength offset via the local
# dispersion dlambda/dx = wavelength * v_pix / c.
dispersion_delta = np.zeros(n_lines)
dispersion_delta_err = np.zeros(n_lines)
for m in range(n_lines):
    idx = fit_window(line_position[m])
    model_value = pixel_model_flux(line_position[m], idx, v_pix[m], line_width[m], departure[m])
    model_shifted = pixel_model_flux(line_position[m] + 1e-3, idx, v_pix[m], line_width[m], departure[m])
    model_derivative = (model_shifted - model_value) / 1e-3
    weight = inverse_variance[idx]
    denom = np.sum(model_derivative**2 * weight)
    if denom > 1e-12:
        dispersion_delta[m] = np.sum(model_derivative * weight * (flux[idx] - model_value)) / denom
        naive_err = 1.0 / np.sqrt(denom)
        resid = (model_value + model_derivative * dispersion_delta[m] - flux[idx]) / flux_err[idx]
        dof = max(len(idx) - 1, 1)
        chi2r = np.sum(resid**2) / dof
        dispersion_delta_err[m] = naive_err * np.sqrt(max(chi2r, 1.0))
    else:
        dispersion_delta[m] = 0.0
        dispersion_delta_err[m] = np.inf

dlambda_dx = wavelength * v_pix / C_LIGHT_KMS  # nm per pixel, local dispersion
wavelength_residual = dispersion_delta * dlambda_dx        # nm
wavelength_residual_err = dispersion_delta_err * np.abs(dlambda_dx)  # nm
dispersion_position_std_nm = dispersion_position_std * np.abs(dlambda_dx)

# Velocity units (m/s), via the standard dv/c = dlambda/lambda relation --
# the natural unit here given this is ultimately headed toward an RV
# precision budget, not a wavelength-precision one. SPEED_OF_LIGHT is in
# m/s, and wavelength_residual/wavelength is a dimensionless ratio (both
# in nm), so this division directly gives m/s without any unit juggling.
velocity_residual = SPEED_OF_LIGHT * (wavelength_residual / wavelength)              # m/s
velocity_residual_err = SPEED_OF_LIGHT * (wavelength_residual_err / wavelength)      # m/s
dispersion_position_std_ms = SPEED_OF_LIGHT * (dispersion_position_std_nm / wavelength)  # m/s

_indices = np.linspace(0, len(peak_pixel) - 1, 16).astype(int)
_x_pos_array = peak_pixel[_indices]
_cmap = matplotlib.colormaps.get_cmap('nipy_spectral')
_norm = matplotlib.colors.Normalize(vmin=peak_pixel.min(), vmax=peak_pixel.max())
_colours = _cmap(_norm(_x_pos_array))

def _data_constrained_mask(x_pos, u_values):
    """ True where u_values lies within the pixel window that actually has
        data at this position -- outside this range the fitted curve is
        extrapolation from the GP prior, not from data. """
    local_v_pix = np.interp(x_pos, line_position, v_pix)
    half_range = HALF_WINDOW * local_v_pix
    return np.abs(u_values) <= half_range

# =========================================================================
# PER-LINE INDEPENDENT WIDTH -- fully free sigma_m, not tied to the smooth
# width(x) GP at all. Direct analogue of Schmidt & Bouchy (2024) section
# 4.3's per-line re-fit (their eq. 16), which is what first revealed their
# own FWHM bimodality (their Fig. 10-11) -- and this mirrors the very
# question this whole investigation started from: is there real,
# systematic line-to-line width structure that a SMOOTH function of
# position is forced to average over and therefore miss, and could
# averaging over it be contributing to the shoulder-excess residual?
# =========================================================================
def fit_per_line_width(line_position, shape_coeffs, v_pix, sigma_init):
    """ For each line independently, find the sigma that minimises THAT
        line's own chi2, holding line_position and the departure term
        (evaluated ONCE, from the current converged shape_coeffs and
        sigma_init -- the same sigma_ref convention the main fit itself
        uses, so this isolates the effect of sigma alone) fixed. Returns
        an array of N_LINES values with no smoothing or regularisation
        across lines at all -- genuinely independent, unlike
        width(x, width_coeffs). Validated directly against known
        synthetic ground truth (including an injected period-2 wobble)
        before being wired in here: recovers true sigma to within the
        noise floor, correlation 0.9995. """
    departure_all = evaluate_departure(line_position, sigma_init, shape_coeffs)
    sigma_per_line = np.empty(n_lines)
    for m in range(n_lines):
        idx = fit_window(line_position[m])
        weight = inverse_variance[idx]
        departure_m = departure_all[m]
        target = flux[idx]

        def neg_log_likelihood(log_sigma, idx=idx, weight=weight,
                                 departure_m=departure_m, target=target, m=m):
            sigma_m = np.exp(log_sigma)
            model = pixel_model_flux(line_position[m], idx, v_pix[m], sigma_m, departure_m)
            resid = target - model
            return 0.5 * np.sum(weight * resid**2)

        result = minimize_scalar(neg_log_likelihood,
                                   bounds=(np.log(0.3 * sigma_init[m]), np.log(3 * sigma_init[m])),
                                   method='bounded', options={'xatol': 1e-5})
        sigma_per_line[m] = np.exp(result.x)
    return sigma_per_line

sigma_per_line = fit_per_line_width(line_position, shape_coeffs, v_pix, line_width)
departure_all = evaluate_departure(line_position, line_width, shape_coeffs)

# --- period-2 (odd/even line) structure check, the same question this
# whole investigation began with, now on the fully converged, exact-
# pixel-integration, MAP-l_x, non-stationary-kernel model ---
order_by_position = np.argsort(line_position)
fwhm_sorted = 2.355 * sigma_per_line[order_by_position]
position_sorted = line_position[order_by_position]
odd_mean = np.mean(fwhm_sorted[1::2])
even_mean = np.mean(fwhm_sorted[0::2])
print(f"\nPer-line width, odd/even split (position-sorted): "
      f"even-index mean FWHM = {even_mean:.4f} km/s, odd-index mean FWHM = {odd_mean:.4f} km/s, "
      f"difference = {abs(even_mean - odd_mean) * 1000:.2f} m/s")

fig_plw, (ax_plw1, ax_plw2) = plt.subplots(1, 2, figsize=(15, 6))

# --- panel 1: per-line free width vs. the smooth width(x) curve --------
smooth_fwhm_curve = 2.355 * width(pixel.astype(float), width_coeffs)
ax_plw1.plot(pixel, smooth_fwhm_curve, color='tab:orange', lw=1.5,
              label='smooth width(x) (current model)', zorder=3)
ax_plw1.scatter(position_sorted[0::2], fwhm_sorted[0::2], s=14, marker='^',
                  color='tab:green', label='even index', zorder=4)
ax_plw1.scatter(position_sorted[1::2], fwhm_sorted[1::2], s=14, marker='v',
                  color='tab:red', label='odd index', zorder=4)
ax_plw1.set_xlim(peak_pixel.min() - 50, peak_pixel.max() + 50)
ax_plw1.set_ylim(1.5, 3.0)
ax_plw1.set_xlabel('pixel')
ax_plw1.set_ylabel('FWHM [km/s]')
ax_plw1.set_title('Per-line free width vs. the smooth width(x) GP')
ax_plw1.legend(fontsize=7)

# --- panel 2: does per-line free width change the shoulder-excess -------
# residual pattern? Same bins used throughout this investigation
# (synthetic position-error, width-error, and background-error tests),
# but now on REAL data. departure_all is held fixed across both cases so
# this isolates the effect of sigma alone, matching the same
# hold-everything-else-fixed discipline used in every earlier test here.
def stacked_residual_bins(sigma_array):
    u_all, resid_all, weight_all = [], [], []
    for m in range(n_lines):
        idx = fit_window(line_position[m])
        model = pixel_model_flux(line_position[m], idx, v_pix[m], sigma_array[m], departure_all[m])
        u_all.append((idx - line_position[m]) * v_pix[m])
        resid_all.append(flux[idx] - model)
        weight_all.append(inverse_variance[idx])
    u_all = np.concatenate(u_all)
    resid_all = np.concatenate(resid_all)
    weight_all = np.concatenate(weight_all)

    def wmean(mask):
        return np.average(resid_all[mask], weights=weight_all[mask]) if mask.sum() > 0 else np.nan

    peak_mask = np.abs(u_all) < 0.5
    shoulder_mask = (u_all >= 0.5) & (u_all < 1.5)
    far_mask = (u_all >= 2.0) & (u_all < 3.0)
    return wmean(peak_mask), wmean(shoulder_mask), wmean(far_mask)

peak_smooth, shoulder_smooth, far_smooth = stacked_residual_bins(line_width)
peak_free, shoulder_free, far_free = stacked_residual_bins(sigma_per_line)

print(f"\nShoulder-excess bins, smooth width(x) model:  "
      f"peak={peak_smooth:+.5f}, shoulder={shoulder_smooth:+.5f}, far_wing={far_smooth:+.5f}")
print(f"Shoulder-excess bins, per-line FREE width:     "
      f"peak={peak_free:+.5f}, shoulder={shoulder_free:+.5f}, far_wing={far_free:+.5f}")

x_pos = np.arange(3)
bar_width = 0.35
ax_plw2.bar(x_pos - bar_width / 2, [peak_smooth, shoulder_smooth, far_smooth], bar_width,
             label='smooth width(x)', color='tab:orange', alpha=0.8)
ax_plw2.bar(x_pos + bar_width / 2, [peak_free, shoulder_free, far_free], bar_width,
             label='per-line free width', color='tab:blue', alpha=0.8)
ax_plw2.axhline(0, color='k', lw=0.8)
ax_plw2.set_xticks(x_pos)
ax_plw2.set_xticklabels(['peak', 'shoulder', 'far_wing'])
ax_plw2.set_ylabel('weighted mean residual (flux - model)')
ax_plw2.set_title('Shoulder-excess bins: smooth vs. per-line free width')
ax_plw2.legend(fontsize=8)

fig_plw.suptitle('Releasing the Gaussian core width per line, not tied to width(x)')
fig_plw.tight_layout()
fig_plw.canvas.draw()
plt.pause(0.1)

# =========================================================================
# l_x SHORT-SCALE COMPARISON -- does forcing SHAPE_X_LENGTH_SCALE shorter
# (comparable to window 2's ~569 px panel spacing and N_X_INDUCING's own
# ~285 px knot spacing) reduce the shoulder-excess residual, even though
# the MAP fit above found the pooled evidence prefers something 2-5x
# longer? A genuinely different question from "what maximises the
# marginal likelihood" -- this tests directly whether a shorter length
# scale helps THIS specific, localised residual pattern, using the same
# stacked-bin machinery and hold-everything-else-fixed discipline as
# every earlier test in this investigation. sigma(x) (line_width) is held
# at its current smooth-model value in BOTH cases, so only the shape
# term's x-length-scale differs between them.
# =========================================================================
SHAPE_X_LENGTH_SCALE_CONVERGED = SHAPE_X_LENGTH_SCALE  # save the MAP-fit value
SHAPE_X_LENGTH_SCALE_SHORT_TEST = 400  # pixels; comparable to the ~569 px
                                          # window-2 panel spacing and the
                                          # ~285 px N_X_INDUCING knot
                                          # spacing, both far shorter than
                                          # the ~1000-3000 px the MAP fit
                                          # itself converged to

SHAPE_X_LENGTH_SCALE = SHAPE_X_LENGTH_SCALE_SHORT_TEST
shape_coeffs_short = fit_shape_departure(line_position, width_coeffs, v_pix)
SHAPE_X_LENGTH_SCALE = SHAPE_X_LENGTH_SCALE_CONVERGED  # restore immediately

departure_all_short = evaluate_departure(line_position, line_width, shape_coeffs_short)

def stacked_residual_bins_with_departure(sigma_array, departure_array):
    """ Same peak/shoulder/far_wing binning as stacked_residual_bins
        above, generalised to take an explicit departure array rather
        than closing over the module-level departure_all -- needed here
        because this test compares TWO DIFFERENT departure fits (long vs
        short l_x), not two different sigma arrays. Also returns the
        total chi2 and point count in one pass, since a shorter length
        scale has more effective degrees of freedom and will almost
        always fit the TRAINING data at least as well regardless of
        whether it is capturing genuine structure -- the log-evidence
        comparison already accounts for that trade-off, a bare chi2
        comparison does not, so both are reported for an honest read. """
    u_all, resid_all, weight_all = [], [], []
    for m in range(n_lines):
        idx = fit_window(line_position[m])
        model = pixel_model_flux(line_position[m], idx, v_pix[m], sigma_array[m], departure_array[m])
        u_all.append((idx - line_position[m]) * v_pix[m])
        resid_all.append(flux[idx] - model)
        weight_all.append(inverse_variance[idx])
    u_all = np.concatenate(u_all)
    resid_all = np.concatenate(resid_all)
    weight_all = np.concatenate(weight_all)

    def wmean(mask):
        return np.average(resid_all[mask], weights=weight_all[mask]) if mask.sum() > 0 else np.nan

    peak_mask = np.abs(u_all) < 0.5
    shoulder_mask = (u_all >= 0.5) & (u_all < 1.5)
    far_mask = (u_all >= 2.0) & (u_all < 3.0)
    chi2 = np.sum(weight_all * resid_all**2)
    return wmean(peak_mask), wmean(shoulder_mask), wmean(far_mask), chi2, len(resid_all)

peak_long, shoulder_long, far_long, chi2_long, n_pts_long = \
    stacked_residual_bins_with_departure(line_width, departure_all)
peak_short, shoulder_short, far_short, chi2_short, n_pts_short = \
    stacked_residual_bins_with_departure(line_width, departure_all_short)

print(f"\nShoulder-excess bins, MAP-fit l_x={SHAPE_X_LENGTH_SCALE_CONVERGED:.0f} px (long):    "
      f"peak={peak_long:+.5f}, shoulder={shoulder_long:+.5f}, far_wing={far_long:+.5f}, "
      f"chi2/n={chi2_long / n_pts_long:.4f}")
print(f"Shoulder-excess bins, forced l_x={SHAPE_X_LENGTH_SCALE_SHORT_TEST} px (short):       "
      f"peak={peak_short:+.5f}, shoulder={shoulder_short:+.5f}, far_wing={far_short:+.5f}, "
      f"chi2/n={chi2_short / n_pts_short:.4f}")

fig_lxshort, ax_lxshort = plt.subplots(figsize=(8, 6))
x_pos = np.arange(3)
bar_width = 0.35
ax_lxshort.bar(x_pos - bar_width / 2, [peak_long, shoulder_long, far_long], bar_width,
                 label=f'MAP-fit l_x ({SHAPE_X_LENGTH_SCALE_CONVERGED:.0f} px)',
                 color='tab:orange', alpha=0.8)
ax_lxshort.bar(x_pos + bar_width / 2, [peak_short, shoulder_short, far_short], bar_width,
                 label=f'forced short l_x ({SHAPE_X_LENGTH_SCALE_SHORT_TEST} px)',
                 color='tab:purple', alpha=0.8)
ax_lxshort.axhline(0, color='k', lw=0.8)
ax_lxshort.set_xticks(x_pos)
ax_lxshort.set_xticklabels(['peak', 'shoulder', 'far_wing'])
ax_lxshort.set_ylabel('weighted mean residual (flux - model)')
ax_lxshort.set_title('Shoulder-excess bins: MAP-fit vs. forced-short shape l_x')
ax_lxshort.legend(fontsize=8)
fig_lxshort.tight_layout()
fig_lxshort.canvas.draw()
plt.pause(0.1)

# =========================================================================
# CONTINUOUS RESIDUAL PROFILE vs. u -- direct replacement for the coarse
# 3-bin summary above, built specifically to pin down "model sits above
# data at the peak, below at the wings" precisely: WHERE (in km/s) does
# the sign flip actually happen, and how large is it at its largest,
# rather than three averages that may straddle the real structure. Also
# reports the residual's own point-to-point (lag-1) autocorrelation
# within this stack, as a cross-check against the separately-reported
# per-line autocorrelation -- if a similar lag-1 correlation shows up
# here, that supports the noise-covariance explanation for chi2/dof
# rather than genuine model mismatch; if it does not, that argues for a
# real, structural shape mismatch instead.
# =========================================================================
u_stack, resid_stack, weight_stack = [], [], []
for m in range(n_lines):
    idx = fit_window(line_position[m])
    model_window = pixel_model_flux(line_position[m], idx, v_pix[m], line_width[m], departure_all[m])
    u_stack.append((idx - line_position[m]) * v_pix[m])
    resid_stack.append(flux[idx] - model_window)
    weight_stack.append(inverse_variance[idx])
u_stack = np.concatenate(u_stack)
resid_stack = np.concatenate(resid_stack)
weight_stack = np.concatenate(weight_stack)

order_idx = np.argsort(u_stack)
u_stack, resid_stack, weight_stack = u_stack[order_idx], resid_stack[order_idx], weight_stack[order_idx]

N_RESIDUAL_BINS = 30
bin_edges = np.linspace(u_stack.min(), u_stack.max(), N_RESIDUAL_BINS + 1)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_mean = np.full(N_RESIDUAL_BINS, np.nan)
bin_err = np.full(N_RESIDUAL_BINS, np.nan)
bin_n = np.zeros(N_RESIDUAL_BINS, dtype=int)
for i in range(N_RESIDUAL_BINS):
    mask = (u_stack >= bin_edges[i]) & (u_stack < bin_edges[i + 1])
    if mask.sum() > 5:
        bin_mean[i] = np.average(resid_stack[mask], weights=weight_stack[mask])
        # standard error of a weighted mean, NOT assuming independence --
        # see the printed lag-1 correlation below for whether that
        # assumption would even be reasonable here
        bin_err[i] = 1 / np.sqrt(np.sum(weight_stack[mask]))
        bin_n[i] = mask.sum()

fig_resprof, ax_resprof = plt.subplots(figsize=(11, 6))
ax_resprof.errorbar(bin_centres, bin_mean, yerr=bin_err, fmt='o-', ms=4, capsize=2, color='tab:blue')
ax_resprof.axhline(0, color='k', lw=0.8)
ax_resprof.set_xlabel('u [km/s]')
ax_resprof.set_ylabel('weighted mean residual (flux - model)')
ax_resprof.set_title(f'Continuous residual profile vs. u (order {ORDER}, '
                       f'{N_RESIDUAL_BINS} bins, current MAP-fit model)')
fig_resprof.tight_layout()
fig_resprof.canvas.draw()
plt.pause(0.1)

print("\nContinuous residual profile (u_centre, weighted mean, n_points):")
for uc, bm, be, bn in zip(bin_centres, bin_mean, bin_err, bin_n):
    if bn > 5:
        print(f"  u={uc:+.3f} km/s: residual={bm:+.5f} +/- {be:.5f}  (n={bn})")

# lag-1 autocorrelation of this SPECIFIC (u-sorted) residual stack, for
# direct comparison against the separately-reported per-line (pixel-
# index-sorted) autocorrelation -- a different ordering, so a different
# question: this one asks whether NEARBY-IN-VELOCITY residuals move
# together (consistent with genuine shape mismatch, which is smooth in
# u), while the earlier one asks whether NEARBY-IN-PIXEL residuals do
# (consistent with extraction-correlated noise).
if len(resid_stack) > 1:
    lag1_corr = np.corrcoef(resid_stack[:-1], resid_stack[1:])[0, 1]
    print(f"\nStacked-residual (u-sorted) lag-1 autocorrelation: {lag1_corr:+.4f}")

# =========================================================================
# WINDOW 2: LSF models overplotted on data -- one subplot per POSITION
# BIN, not per single line: each panel shows every line's own data within
# that bin, stacked in u-space (which normalises out each line's own
# position and width, so many noisy individual lines' data can overlay
# meaningfully), against ONE representative model curve for that bin.
# A single line's own data is noisy enough that it is hard to judge
# whether the model is a good match; many lines stacked together make
# that comparison far more direct, at the cost of the (small, since
# width/shape vary smoothly) approximation that one representative curve
# stands in for the whole bin.
# =========================================================================
N_LSF_PANELS = 16
_bin_edges = np.linspace(0, n_lines, N_LSF_PANELS + 1).astype(int)
_bin_indices = [np.arange(_bin_edges[i], _bin_edges[i + 1]) for i in range(N_LSF_PANELS)]
_bin_rep_idx = [bin_idx[len(bin_idx) // 2] for bin_idx in _bin_indices]  # bin's middle line
_bin_x_pos = peak_pixel[_bin_rep_idx]
_bin_colours = _cmap(_norm(_bin_x_pos))

_LSF_DATA_DIR = '/Users/dmilakov/software/harps/testing/lsf/formod/data'

# PIXEL-INTEGRATED MODEL AT THE DATA'S OWN SAMPLE POINTS -- added alongside
# the existing continuous curve, NOT instead of it, because the two answer
# different questions and conflating them previously produced a misleading
# diagnostic. flux[idx] is a PIXEL-BOXCAR AVERAGE of the true continuous
# profile (that is what a detector pixel physically measures), while
# gaussian_mean(u, sigma) + departure[rep_idx] is the continuous curve
# POINT-SAMPLED at u -- exactly what a real pixel does not measure.
# Comparing a point-sampled curve against boxcar-averaged data produces a
# spurious mismatch that is largest exactly where curvature is largest (the
# line core), independent of whether the underlying fit is actually good --
# confirmed directly on this order's own saved output: the previous
# points/model files showed a residual that was negative (model above data)
# in 90-100% of individual lines at the core specifically, the signature of
# this effect rather than of a genuine fit failure. pixel_model_flux is
# already the quantity the optimiser itself compares against flux[idx]
# throughout the fit (fit_dispersion, fit_width, fit_shape_departure all use
# it via gaussian_pixel_integral + convolution_matrix @ departure);
# evaluating it here, at each line's own native pixel indices, is the
# correct apples-to-apples comparison, using each line's OWN sigma and
# departure (not just the bin's representative line, unlike the continuous
# curve, which stands in for the whole bin only for visual shape reference).
fig2, axes2 = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
print("\nWindow 2 per-bin residual diagnostics (data minus PIXEL-INTEGRATED "
      "model, the correct like-for-like comparison): "
      "core = |u|<0.5 km/s, shoulder = 0.5<=u<1.5 km/s")
for i, (ax, bin_idx, rep_idx, x_pos, colour) in enumerate(zip(axes2.flat, _bin_indices, _bin_rep_idx, _bin_x_pos, _bin_colours)):
    sigma = line_width[rep_idx]
    lsf = gaussian_mean(u, sigma) + departure[rep_idx]   # continuous curve; shape reference only, not pixel-integrated
    mask = _data_constrained_mask(x_pos, u)

    U_DATA, F_DATA, F_MODEL_PIX = [], [], []
    for m in bin_idx:
        idx = fit_window(line_position[m])
        u_data = (idx - line_position[m]) * v_pix[m]
        model_pix = pixel_model_flux(line_position[m], idx, v_pix[m],
                                       line_width[m], departure[m])
        U_DATA.append(u_data)
        F_DATA.append(flux[idx])
        F_MODEL_PIX.append(model_pix)
        ax.plot(u_data, flux[idx], '.', ms=4, alpha=0.8, color=colour)

    U_DATA_flat = np.concatenate(U_DATA)
    F_DATA_flat = np.concatenate(F_DATA)
    F_MODEL_PIX_flat = np.concatenate(F_MODEL_PIX)
    resid_flat = F_DATA_flat - F_MODEL_PIX_flat

    # Sort by u purely for a clean connecting line across the (interleaved,
    # per-line) pixel-integrated model points -- does not affect the data
    # used for the saved files or the residual statistics below.
    _order = np.argsort(U_DATA_flat)
    ax.plot(U_DATA_flat[_order], F_MODEL_PIX_flat[_order], 'x', ms=4, mew=1.0,
            color='k', alpha=0.6, label='pixel-integrated model (at data points)')
    ax.plot(u[mask], lsf[mask], color=colour, lw=1.0, ls='--', alpha=0.7,
            label='continuous curve (reference only)')
    ax.plot(u[mask], gaussian_mean(u, sigma)[mask], 'k', ls=':', alpha=0.5, lw=1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f'pixel {x_pos:.0f}, n={len(bin_idx)} lines (FWHM={2.355*sigma:.3f} km/s)', fontsize=8)
    if i == 0:
        ax.legend(fontsize=5)

    core_m = np.abs(U_DATA_flat) < 0.5
    shoulder_m = (U_DATA_flat >= 0.5) & (U_DATA_flat < 1.5)
    core_str = (f"core resid={resid_flat[core_m].mean():+.4f}+/-{resid_flat[core_m].std():.4f} (n={core_m.sum()})"
                if core_m.sum() > 0 else "core: n/a")
    shoulder_str = (f"shoulder resid={resid_flat[shoulder_m].mean():+.4f}+/-{resid_flat[shoulder_m].std():.4f} (n={shoulder_m.sum()})"
                     if shoulder_m.sum() > 0 else "shoulder: n/a")
    print(f"  bin {i+1:2d} (pixel {x_pos:.0f}): {core_str}, {shoulder_str}")

    np.savetxt(f'{_LSF_DATA_DIR}/order={ORDER}_lsf_model_segm={i+1}.txt',
               np.transpose([u[mask], lsf[mask]]),
               header='velocity, value  (CONTINUOUS curve, point-sampled -- '
                      'NOT pixel-integrated; shape reference only, uses the '
                      "bin's representative line)")
    np.savetxt(f'{_LSF_DATA_DIR}/order={ORDER}_points_segm={i+1}.txt',
               np.transpose([U_DATA_flat, F_DATA_flat]),
               header='velocity, value  (raw per-pixel data, flux[idx], all '
                      'lines in this bin stacked in u)')
    np.savetxt(f'{_LSF_DATA_DIR}/order={ORDER}_model_pixel_segm={i+1}.txt',
               np.transpose([U_DATA_flat, F_MODEL_PIX_flat]),
               header='velocity, value  (PIXEL-INTEGRATED model at the SAME '
                      'points as the points file -- gaussian_pixel_integral '
                      '+ convolution_matrix @ departure, per line\'s own '
                      'sigma/departure -- exactly the quantity the fit '
                      'itself compares against flux[idx]; subtract this '
                      'from the points file for a correct residual, not '
                      'the lsf_model_segm file)')
fig2.suptitle('LSF model vs. data, one panel per position bin (all lines in the bin overlaid)')
fig2.supxlabel('u [km/s]')
fig2.tight_layout()
fig2.canvas.draw()

plt.pause(0.1)

# =========================================================================
# WINDOW 3: wavelength calibration residuals -- inferred wavelength (each
# line's own, individually-fit position, converted through the local
# dispersion) minus its theoretical wavelength (from the LFC comb
# frequency) -- NOT the pixel-space residual shown elsewhere. Vertical
# lines at 1/8-detector boundaries: if a pattern lines up with these,
# that points to a per-amplifier readout effect.
# =========================================================================
fig3, ax3 = plt.subplots(figsize=(11, 6))
ax3.errorbar(wavelength, velocity_residual, yerr=velocity_residual_err,
             fmt='.', ms=4, elinewidth=0.5, capsize=0, label='per-line residual')
ax3.fill_between(wavelength[np.argsort(wavelength)],
                  -dispersion_position_std_ms[np.argsort(wavelength)],
                  dispersion_position_std_ms[np.argsort(wavelength)],
                  color='tab:green', alpha=0.25, label='dispersion GP posterior std')
ax3.axhline(0, color='gray', lw=0.5)
ax3.legend(fontsize=8)
ax3.set_title('Wavelength calibration residuals: inferred - theoretical wavelength')
ax3.set_xlabel('wavelength [nm]')
ax3.set_ylabel('residual [m/s]')
fig3.tight_layout()
fig3.canvas.draw()
plt.pause(0.1)

# =========================================================================
# WINDOW 4: FWHM(x) (left, unchanged from the earlier combined panel), plus
# a new right panel testing a specific hypothesis: does FWHM correlate
# with the local background-to-envelope ratio B(x)/E(x)? B/E is a proxy
# for how much of the pixel's flux is scattered-light/background versus
# genuine comb signal -- if the per-line width estimate is being pulled
# around by an imperfectly-subtracted background (rather than reflecting
# a genuine instrumental effect), it should show up here as a trend
# rather than a scatter cloud.
# =========================================================================
fig4, (ax4, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
sigma_grid = np.maximum(np.exp(width_coeffs), 0.05 * np.median(v_pix))
fwhm_grid = 2.355 * sigma_grid
fwhm_upper = 2.355 * np.exp(width_coeffs + width_log_sigma_std)
fwhm_lower = 2.355 * np.exp(width_coeffs - width_log_sigma_std)
ax4.plot(pixel, fwhm_grid, color='tab:orange', lw=1.3, label='FWHM(x)')
ax4.errorbar(line_position, 2.355 * np.exp(width_raw_target),
             yerr=2.355 * np.exp(width_raw_target) * width_raw_target_err,
             fmt='.', ms=4, elinewidth=0.5, capsize=0, alpha=0.5,
             label='per-line target (pre-GP)')
ax4.fill_between(pixel, fwhm_lower, fwhm_upper, color='tab:orange', alpha=0.3,
                  label='GP posterior std')
ax4.axhline(EXPECTED_FWHM_KMS, color='k', ls='--', lw=1,
            label=f'R={EXPECTED_R} ({EXPECTED_FWHM_KMS:.3f} km/s)')
ax4.legend(fontsize=8)
ax4.set_xlim(peak_pixel.min() - 50, peak_pixel.max() + 50)
ax4.set_title('Width: FWHM(x) [km/s], from the fitted GP grid')
ax4.set_xlabel('pixel')
ax4.set_ylabel('FWHM [km/s]')
ax4.set_ylim(1.5, 3.0)
ax4_r = ax4.secondary_yaxis('right', functions=(lambda x: SPEED_OF_LIGHT_KMS / x, lambda x: SPEED_OF_LIGHT_KMS / x))
ax4_r.set_ylabel('R = c / FWHM')

# --- FWHM vs. local background/envelope ratio, per line -----------------
# Both B(x) and E(x) are interpolated at each line's own position (not
# read off the file's raw columns -- envelope_grid_full/background_grid_full
# are this script's own GP-fitted estimates, the same ones the forward
# model itself uses), so this tests the ratio actually seen by the fit,
# not a separately-measured proxy for it.
_be_ratio_lines = (np.interp(line_position, pixel, background_grid_full)
                    / np.interp(line_position, pixel, envelope_grid_full))
_fwhm_lines = 2.355 * np.exp(width_raw_target)
_line_colours = _cmap(_norm(line_position))
ax4b.scatter(_be_ratio_lines, _fwhm_lines, c=_line_colours, s=10, alpha=0.6)
_finite = np.isfinite(_be_ratio_lines) & np.isfinite(_fwhm_lines)
if np.sum(_finite) > 2:
    _r = np.corrcoef(_be_ratio_lines[_finite], _fwhm_lines[_finite])[0, 1]
    _slope, _intercept = np.polyfit(_be_ratio_lines[_finite], _fwhm_lines[_finite], 1)
    _be_trend_x = np.linspace(_be_ratio_lines[_finite].min(), _be_ratio_lines[_finite].max(), 50)
    ax4b.plot(_be_trend_x, _slope * _be_trend_x + _intercept, 'k--', lw=1,
              label=f'linear fit, r={_r:.3f}')
    ax4b.legend(fontsize=8)
ax4b.set_title('Per-line FWHM vs. local background/envelope ratio')
ax4b.set_xlabel('B(x) / E(x)')
ax4b.set_ylabel('FWHM [km/s]')

fig4.tight_layout()
fig4.canvas.draw()
plt.pause(0.1)

# =========================================================================
# WINDOW 5: raw data + forward model for ALL lines in sequence (top),
# normalised residuals (bottom), sharing the same x-axis. Top panel shown
# in RAW flux units, not the normalised (E,B-divided-out) space the fit
# actually works in -- the inverse of flux = (flux_raw - B) / (E - B) is
# flux_raw = flux*(E - B) + B, applied here to both the data (flux_raw is
# already available directly, being exactly this by construction) and
# the model (computed explicitly, since no "raw-space model" exists yet).
# =========================================================================
model_raw = model * (envelope_grid_full - background_grid_full) + background_grid_full

fig5, axes5 = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
ax = axes5[0]
ax.errorbar(pixel[fitted_mask], flux_raw[fitted_mask], err_raw[fitted_mask],
            marker='.', ms=8, alpha=0.5, label='data (raw)')
ax.plot(pixel[fitted_mask], model_raw[fitted_mask], 'r-', lw=0.8, label='model (raw)')
ax.plot(pixel, envelope_grid_full, color='tab:orange', lw=1, alpha=0.8, label='envelope E(x)')
ax.plot(pixel, background_grid_full, color='tab:blue', lw=1, alpha=0.8, label='background B(x)')
ax.legend(fontsize=8)
ax.set_title('All lines, data + forward model, in sequence (raw flux units)')
ax.set_ylabel('flux [counts]')
# Restricted to the data-covered range, matching window 1's convention --
# envelope/background are plotted over the FULL pixel array, and outside
# where peak/boundary measurements actually constrain them they revert
# toward the GP prior rather than tracking anything real (visible
# directly here as a large spike beyond ~600,000 counts near the left
# edge if left unrestricted, well above the ~300,000-count data scale).
axes5[0].set_xlim(peak_pixel.min() - 50, peak_pixel.max() + 50)

ax = axes5[1]
ax.plot(pixel[fitted_mask], residual[fitted_mask], '.', ms=2, alpha=0.4)
ax.axhline(0, color='r', lw=0.8)
ax.set_ylim(-30, 30)
ax.set_title('Normalised residuals, (flux - model) / error')
ax.set_xlabel('pixel')
fig5.tight_layout()
fig5.canvas.draw()

np.savetxt(f'/Users/dmilakov/software/harps/testing/lsf/formod/data/order={ORDER}_forward_model.txt',
           np.transpose([pixel[fitted_mask],
                         flux_raw[fitted_mask], 
                         err_raw[fitted_mask],
                         model_raw[fitted_mask],
                         residual[fitted_mask]]),
           header = 'pixel, raw flux, raw err, model, residual',
           )

plt.pause(0.1)

# =========================================================================
# WINDOW 6: LSF departure from a Gaussian -- (1) how the departure curve
# itself varies across the order, at the same 16 reference positions used
# in window 2/5's colour scheme, and (2) the fitted 2D GP grid D(u,x) that
# generates every one of those curves. The two panels show the same
# underlying quantity two ways: the left panel is a set of 1D slices
# through phi(u;x) - G(u;sigma(x)) at fixed x, the right panel is the
# genuinely 2D inducing-point surface those slices are interpolated from.
# =========================================================================
fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6))

# --- departure curves at 16 positions across the order ------------------
ax = axes6[0]
_sigma_x_pos_array = width(_x_pos_array, width_coeffs)
_departure_x_pos_array = evaluate_departure(_x_pos_array, _sigma_x_pos_array, shape_coeffs)
for x_pos, colour, dep in zip(_x_pos_array, _colours, _departure_x_pos_array):
    mask = _data_constrained_mask(x_pos, u)
    ax.plot(u[mask], dep[mask], color=colour, label=f'{x_pos:.0f}')
ax.axhline(0, color='gray', lw=0.5)
ax.legend(fontsize=6, ncol=2, title='pixel')
ax.set_title('Departure from Gaussian: phi(u;x) - G(u;sigma(x))')
ax.set_xlabel('u [km/s]')
ax.set_ylabel('departure')

# --- 2D GP departure grid D(u,x) -----------------------------------------
# The inducing grid itself: each row is a position knot, each column a
# velocity knot, colour is the fitted departure value there. There's no
# natural "one curve per order" decomposition for a genuinely 2D quantity
# like this one, so the left panel's slices and this panel's surface are
# complementary, not redundant -- the left panel shows what the model
# looks like at 16 chosen positions, this one shows everything it learned.
ax = axes6[1]
im = ax.imshow(shape_coeffs.T, aspect='auto', origin='lower', cmap='RdBu_r',
               extent=[u_inducing.min(), u_inducing.max(), x_inducing.min(), x_inducing.max()],
               vmin=-np.max(np.abs(shape_coeffs)), vmax=np.max(np.abs(shape_coeffs)))
fig6.colorbar(im, ax=ax, label='departure')
ax.set_title('2D GP departure grid D(u,x)')
ax.set_xlabel('u [km/s]')
ax.set_ylabel('pixel')

fig6.suptitle('LSF shape: departure from a Gaussian, across the order')
fig6.tight_layout()
fig6.canvas.draw()
plt.pause(0.1)

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

# --- residual autocorrelation check, within each line's own window -------
max_lag = 4
lag_products = {lag: [] for lag in range(1, max_lag + 1)}
var_terms = []
for m in range(len(line_position)):
    idx = fit_window(line_position[m])
    r = residual[idx]
    var_terms.append(np.mean(r**2))
    for lag in range(1, max_lag + 1):
        if len(r) > lag:
            lag_products[lag].append(np.mean(r[:-lag] * r[lag:]))
mean_var = np.mean(var_terms)
print("\nResidual autocorrelation within each line's own window (1.0 would be perfectly correlated):")
for lag in range(1, max_lag + 1):
    ac = np.mean(lag_products[lag]) / mean_var
    print(f"  lag {lag}: {ac:+.4f}")

print("\nAll windows shown. Close them (or Ctrl-C) to exit.")
plt.ioff()
plt.show()