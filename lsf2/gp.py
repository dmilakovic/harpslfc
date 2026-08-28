"""
Generic MAP (maximum a posteriori) Gaussian-Process fitting, shared by
every GP-based component in the pipeline (envelope, background residual,
width, dispersion). Ported behaviour-for-behaviour from
reconstruct_lsf_master.py's map_gp_fit / poly_plus_gp_fit.

Every hyperparameter here (amplitude or length scale) gets a LogNormal
prior instead of a hard bound: a hard bound lets an optimiser sit exactly
at the boundary for free, which was a real, confirmed failure mode in the
original script (see git history / project memory for the envelope
length-scale case). A proper prior has no such free lunch.

Kernel family matters as much as the prior: RBF (ExpSquared) has a sharp
Gaussian spectral cutoff and structurally cannot "cheat" with a
pathologically long length scale the way Matern32's heavier spectral tail
can -- which is why RBF is used for the dispersion residual (a short,
physically-motivated scale) while Matern32 is used for envelope/width
residuals (where a longer nominal scale with some local flexibility is
wanted). See dispersion.py and envelope.py for the specific choice at
each call site.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels
from scipy.optimize import minimize

_KERNEL_CLASSES = {
    'expsquared': kernels.ExpSquared,
    'matern32': kernels.Matern32,
    'matern52': kernels.Matern52,
}

# Per-kernel-type hyperparameter spec: one entry per non-amplitude
# parameter, (name, rescale_by_x_range). 'locallyperiodic' is
# ExpSquared(decay_length) * ExpSineSquared(period, gamma) -- the "locally
# periodic" construction from Duvenaud's kernel cookbook: exactly periodic
# at lag=period, with that periodicity allowed to decay/drift over
# decay_length rather than being forced exactly periodic over the whole
# input range.
_KERNEL_PARAM_SPECS = {
    'expsquared':      [('length', True)],
    'matern32':        [('length', True)],
    'matern52':        [('length', True)],
    'locallyperiodic': [('length', True), ('period', True), ('gamma', False)],
}

_compiled_cache: dict = {}


def _component_kernel(kernel_type, log_subparams):
    if kernel_type == 'locallyperiodic':
        log_length, log_period, log_gamma = log_subparams
        return (kernels.ExpSquared(scale=jnp.exp(log_length))
                * kernels.ExpSineSquared(scale=jnp.exp(log_period), gamma=jnp.exp(log_gamma)))
    (log_length,) = log_subparams
    return _KERNEL_CLASSES[kernel_type](scale=jnp.exp(log_length))


def _component_offsets(kernel_types):
    offsets, offset = [], 0
    for kernel_type in kernel_types:
        offsets.append(offset)
        offset += 1 + len(_KERNEL_PARAM_SPECS[kernel_type])
    return offsets


def _make_kernel(params, kernel_types):
    kernel = None
    for offset, kernel_type in zip(_component_offsets(kernel_types), kernel_types):
        n_sub = len(_KERNEL_PARAM_SPECS[kernel_type])
        log_amp = params[offset]
        log_subparams = params[offset + 1: offset + 1 + n_sub]
        component = jnp.exp(2 * log_amp) * _component_kernel(kernel_type, log_subparams)
        kernel = component if kernel is None else kernel + component
    return kernel


def _get_compiled_functions(kernel_types):
    """ JIT-compiled (neg_log_posterior, grad, predict) triple, cached by
        kernel_types tuple. Defined once per combination rather than as a
        closure rebuilt inside map_gp_fit: a closure rebuilt every call
        gives JAX a fresh function object each time and defeats its own
        compilation cache -- confirmed to matter in practice, not just in
        principle (a full multi-iteration pipeline run was slow enough
        from repeated recompilation alone to be a real problem). """
    key = tuple(kernel_types)
    if key not in _compiled_cache:
        def neg_log_posterior(params, x_n, z_n, e_n, prior_means, prior_stds):
            gp = GaussianProcess(_make_kernel(params, key), x_n, diag=e_n ** 2)
            nll = -gp.log_probability(z_n)
            log_prior = -0.5 * jnp.sum(((params - prior_means) / prior_stds) ** 2)
            return nll - log_prior

        def predict(params, x_n, z_n, e_n, x_grid_n):
            gp = GaussianProcess(_make_kernel(params, key), x_n, diag=e_n ** 2)
            return gp.predict(z_n, x_grid_n, return_var=True)

        neg_log_posterior_jit = jax.jit(neg_log_posterior)
        _compiled_cache[key] = (
            neg_log_posterior_jit,
            jax.jit(jax.grad(neg_log_posterior_jit)),
            jax.jit(predict),
        )
    return _compiled_cache[key]


def map_gp_fit(
    x: np.ndarray,
    z: np.ndarray,
    z_err: np.ndarray,
    x_grid: np.ndarray,
    length_scale_priors: list,
    amplitude_priors: list = None,
    kernel_types: list = None,
    n_restarts: int = 3,
    clip_sigma: float = 4.0,
) -> dict:
    """
    MAP fit of a GP whose kernel is a sum of components, each with its
    own kernel family, length-scale prior, and amplitude prior.

    length_scale_priors: list, one entry per additive component. For a
        kernel with a single non-amplitude hyperparameter, the entry is
        (mean_in_x_units, log_std). For a multi-parameter kernel (only
        'locallyperiodic' currently: length, period, gamma) it is a list
        of that many (mean, log_std) tuples, in the order given by
        _KERNEL_PARAM_SPECS[kernel_type]; length-like sub-parameters get
        the same /x_range normalisation as an ordinary length scale,
        gamma (dimensionless) is used as given.
    amplitude_priors: list of (mean_normalised, log_std), one per
        component; None -> each component's prior is centred on the
        data's own normalised std.
    kernel_types: list of 'expsquared'/'matern32'/'matern52'/
        'locallyperiodic', one per component; None -> all 'expsquared'.

    Returns a dict with 'x_grid', 'z_mean', 'z_std', 'length_scale' (list,
    one per component -- a float, or a {'length','period','gamma'} dict
    for 'locallyperiodic'), 'signal_std' (list), 'n_used'.
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

    # Sigma-clipping against a running local median.
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
    x_mu = x_c.mean()
    x_n = (x_c - x_mu) / x_range
    z_mu = z_c.mean()
    z_n = (z_c - z_mu) / z_range
    e_n = e_c / z_range
    x_grid_n = (x_grid - x_mu) / x_range

    x_n_j, z_n_j, e_n_j = jnp.asarray(x_n), jnp.asarray(z_n), jnp.asarray(e_n)

    if kernel_types is None:
        kernel_types = ['expsquared'] * n_components

    component_sub_priors = []
    for k in range(n_components):
        spec = _KERNEL_PARAM_SPECS[kernel_types[k]]
        entry = length_scale_priors[k]
        if len(spec) == 1:
            mean_val, log_std = entry
            _, rescale = spec[0]
            log_mean = np.log(mean_val / x_range) if rescale else np.log(mean_val)
            component_sub_priors.append([(log_mean, log_std)])
        else:
            sub = []
            for (mean_val, log_std), (_, rescale) in zip(entry, spec):
                log_mean = np.log(mean_val / x_range) if rescale else np.log(mean_val)
                sub.append((log_mean, log_std))
            component_sub_priors.append(sub)

    if amplitude_priors is None:
        emp_std = max(float(np.std(z_n)), 1e-3)
        log_amp_prior_means = [np.log(emp_std)] * n_components
        log_amp_prior_stds = [0.5] * n_components
    else:
        log_amp_prior_means = [np.log(mean_norm) for mean_norm, _ in amplitude_priors]
        log_amp_prior_stds = [log_std for _, log_std in amplitude_priors]

    prior_means_list, prior_stds_list = [], []
    for k in range(n_components):
        prior_means_list.append(log_amp_prior_means[k])
        prior_stds_list.append(log_amp_prior_stds[k])
        for log_mean, log_std in component_sub_priors[k]:
            prior_means_list.append(log_mean)
            prior_stds_list.append(log_std)
    prior_means = jnp.array(prior_means_list)
    prior_stds = jnp.array(prior_stds_list)

    neg_log_posterior_fn, grad_fn_raw, predict_fn_raw = _get_compiled_functions(kernel_types)

    def neg_log_posterior(params):
        return neg_log_posterior_fn(params, x_n_j, z_n_j, e_n_j, prior_means, prior_stds)

    def grad_fn(params):
        return grad_fn_raw(params, x_n_j, z_n_j, e_n_j, prior_means, prior_stds)

    rng = np.random.default_rng(0)
    base_init = []
    for k in range(n_components):
        base_init.append(log_amp_prior_means[k])
        base_init.extend(log_mean for log_mean, _ in component_sub_priors[k])
    init_guesses = [base_init]
    for _ in range(n_restarts - 1):
        guess = []
        for k in range(n_components):
            guess.append(rng.normal(log_amp_prior_means[k], log_amp_prior_stds[k]))
            guess.extend(rng.normal(log_mean, log_std) for log_mean, log_std in component_sub_priors[k])
        init_guesses.append(guess)

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

    offsets = _component_offsets(kernel_types)
    length_scales_opt, signal_stds_opt = [], []
    for k, kernel_type in enumerate(kernel_types):
        off = offsets[k]
        spec = _KERNEL_PARAM_SPECS[kernel_type]
        signal_stds_opt.append(float(np.exp(best_params[off]) * z_range))
        if len(spec) == 1:
            _, rescale = spec[0]
            val = float(np.exp(best_params[off + 1]))
            length_scales_opt.append(val * x_range if rescale else val)
        else:
            sub_vals = {}
            for j, (name, rescale) in enumerate(spec):
                val = float(np.exp(best_params[off + 1 + j]))
                sub_vals[name] = val * x_range if rescale else val
            length_scales_opt.append(sub_vals)

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
    length_scale_prior,
    kernel_type: str = 'expsquared',
    n_restarts: int = 5,
) -> dict:
    """
    z(x) as a fixed-degree Chebyshev polynomial trend (weighted least
    squares, exact coefficient covariance) plus an independent, small-
    amplitude GP on the residual. Used for the dispersion relation, the
    width trend, and the envelope, rather than a single zero-mean GP:
    a bare GP forced to explain a large-dynamic-range trend through its
    own covariance alone badly inflates the reported posterior
    uncertainty (confirmed directly during development, see project
    history), and a joint MAP fit of trend+GP hyperparameters together
    was numerically worse (catastrophic cancellation in the posterior
    variance). Splitting the polynomial out as its own closed-form WLS
    fit avoids both.

    The polynomial is evaluated on a Chebyshev basis (not a raw power
    series) for conditioning, and clamped to x in [-1, 1] outside the
    training range rather than left to extrapolate freely.
    """
    x_lo, x_hi = x.min(), x.max()
    x_resc = 2 * (x - x_lo) / (x_hi - x_lo) - 1
    design = np.polynomial.chebyshev.chebvander(x_resc, poly_degree)
    weight = 1.0 / z_err ** 2
    normal_matrix = (design * weight[:, None]).T @ design
    normal_vector = (design * weight[:, None]).T @ z
    poly_coeffs = np.linalg.solve(normal_matrix, normal_vector)
    poly_coeffs_cov = np.linalg.inv(normal_matrix)

    poly_pred = design @ poly_coeffs

    x_grid_resc = np.clip(2 * (x_grid - x_lo) / (x_hi - x_lo) - 1, -1, 1)
    design_grid = np.polynomial.chebyshev.chebvander(x_grid_resc, poly_degree)
    poly_pred_grid = design_grid @ poly_coeffs
    poly_pred_var_grid = np.einsum('ij,jk,ik->i', design_grid, poly_coeffs_cov, design_grid)

    residual = z - poly_pred
    gp_fit = map_gp_fit(x, residual, z_err, x_grid, n_restarts=n_restarts,
                         length_scale_priors=[length_scale_prior],
                         kernel_types=[kernel_type])

    z_mean = poly_pred_grid + gp_fit['z_mean']
    z_std = np.sqrt(poly_pred_var_grid + gp_fit['z_std'] ** 2)

    return {
        'x_grid': x_grid,
        'z_mean': z_mean,
        'z_std': z_std,
        'poly_coeffs': poly_coeffs,
        'poly_coeffs_cov': poly_coeffs_cov,
        'poly_x_lo': x_lo,
        'poly_x_hi': x_hi,
        'poly_degree': poly_degree,
        'poly_pred_residual_std': float(np.std(residual)),
        'gp_length_scale': gp_fit['length_scale'][0],
        'gp_signal_std': gp_fit['signal_std'][0],
    }


def gp_predict_fixed_hyperparams(
    x: np.ndarray, z: np.ndarray, z_err: np.ndarray, x_grid: np.ndarray,
    kernel_type: str, length_scale, signal_std: float, clip_sigma: float = 4.0,
) -> dict:
    """ Same sigma-clipping/normalisation as map_gp_fit, but skips the
        optimiser entirely: the (length_scale, signal_std) hyperparameters
        are given directly (already in real x/z units, exactly as
        map_gp_fit itself returns them), so this reproduces the SAME
        posterior mean/std map_gp_fit would have at whatever iteration
        produced those hyperparameters -- used to re-evaluate an
        already-converged residual GP on a new (denser) x_grid without
        re-running the L-BFGS-B search, e.g. when building the dense
        dispersion lookup table in pipeline.py. length_scale is a plain
        float for a single-sub-param kernel, or a
        {'length','period','gamma'} dict for 'locallyperiodic'. """
    finite = np.isfinite(z) & np.isfinite(z_err) & (z_err > 0)
    x_f, z_f, e_f = x[finite], z[finite], z_err[finite]
    if len(x_f) < 4:
        return {'x_grid': x_grid, 'z_mean': np.full_like(x_grid, np.nan),
                'z_std': np.full_like(x_grid, np.nan)}

    sort_idx = np.argsort(x_f)
    x_s, z_s, e_s = x_f[sort_idx], z_f[sort_idx], e_f[sort_idx]
    win = max(5, len(x_s) // 10)
    z_baseline = np.array([np.median(z_s[max(0, i - win):i + win + 1]) for i in range(len(z_s))])
    dev = np.abs(z_s - z_baseline)
    mad = np.median(dev)
    sigma_scale = 1.4826 * mad if mad > 0 else np.std(z_s - z_baseline)
    good = dev <= clip_sigma * max(sigma_scale, np.median(e_s))
    x_c, z_c, e_c = x_s[good], z_s[good], e_s[good]
    if len(x_c) < 4:
        x_c, z_c, e_c = x_s, z_s, e_s

    x_range = x_c.max() - x_c.min() if x_c.max() - x_c.min() > 0 else 1.0
    z_range = z_c.max() - z_c.min() if z_c.max() - z_c.min() > 0 else 1.0
    x_mu, z_mu = x_c.mean(), z_c.mean()
    x_n = (x_c - x_mu) / x_range
    z_n = (z_c - z_mu) / z_range
    e_n = e_c / z_range
    x_grid_n = (x_grid - x_mu) / x_range

    kernel_types = [kernel_type]
    spec = _KERNEL_PARAM_SPECS[kernel_type]
    log_amp = np.log(max(signal_std, 1e-300) / z_range)
    if len(spec) == 1:
        _, rescale = spec[0]
        val = length_scale / x_range if rescale else length_scale
        params = jnp.array([log_amp, np.log(max(val, 1e-300))])
    else:
        params_list = [log_amp]
        for name, rescale in spec:
            val = length_scale[name] / x_range if rescale else length_scale[name]
            params_list.append(np.log(max(val, 1e-300)))
        params = jnp.array(params_list)

    _, _, predict_fn_raw = _get_compiled_functions(kernel_types)
    z_mean_n, z_var_n = predict_fn_raw(params, jnp.asarray(x_n), jnp.asarray(z_n),
                                        jnp.asarray(e_n), jnp.asarray(x_grid_n))
    z_mean_n, z_var_n = np.asarray(z_mean_n), np.asarray(z_var_n)
    return {
        'x_grid': x_grid,
        'z_mean': z_mean_n * z_range + z_mu,
        'z_std': np.sqrt(np.maximum(z_var_n, 0.0)) * z_range,
    }


def eval_poly_plus_gp(fit: dict, x_eval: np.ndarray) -> np.ndarray:
    """ Re-evaluate a poly_plus_gp_fit result's polynomial term only, at
        new points -- used by the FITS writer to build the dense
        dispersion lookup table without re-running the GP fit. The GP
        residual term is small by construction (see poly_plus_gp_fit's
        docstring) and is already baked into the dense grid saved at fit
        time; this helper is for extrapolation diagnostics / consistency
        checks only. """
    x_resc = np.clip(2 * (x_eval - fit['poly_x_lo']) / (fit['poly_x_hi'] - fit['poly_x_lo']) - 1, -1, 1)
    design = np.polynomial.chebyshev.chebvander(x_resc, fit['poly_degree'])
    return design @ fit['poly_coeffs']
