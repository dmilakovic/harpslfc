"""
LSF shape: departure from a Gaussian core, via a genuine 2D Gaussian
Process on an inducing grid:

    phi(u,x) = G(u; sigma(x)) + f(u,x),   f ~ GP(0, k)
    k((u,x),(u',x')) = kappa(u) rho_u(u,u') kappa(u') * rho_x(x,x')

rho_u, rho_x are unit-amplitude RBF correlations; kappa(u) is a
non-stationary prior-std envelope (Schmidt & Bouchy 2024 eq. 13-15
inspired): loose near the line core, tight in the far wings.

l_u is tied analytically to the current width estimate
(l_u = shape_u_length_scale_factor * sigma_ref) rather than fit by
marginal likelihood: at the low departure amplitudes this model operates
at, the likelihood's sensitivity to length scale collapses, so fitting it
freely would be prone to wandering (confirmed directly during
development). l_x, by contrast, IS fit every outer iteration
(fit_shape_x_length_scale) via a closed-form evidence profile that was
checked to be sharply peaked (not flat) on real data.

Two "identifiability guard" directions (width_change_direction,
shift_direction_inducing) are penalised heavily in the fit, since the
departure GP would otherwise be degenerate with a pure width change or a
pure shift in line centre.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .config import LSFConfig
from .data import OrderData
from .forward_model import gaussian_pixel_integral, convolution_matrix, gaussian_mean
from .sigma_model import width as width_fn


def unit_rbf(a, b, length_scale):
    """ Amplitude-free RBF correlation matrix (diagonal 1 when a is b). """
    return np.exp(-(a[:, None] - b[None, :]) ** 2 / (2 * length_scale ** 2))


def shape_kappa(u_grid, sigma_ref, cfg: LSFConfig):
    L_kappa = cfg.shape_kappa_width_factor * sigma_ref
    return cfg.shape_kappa_sigma0 + cfg.shape_kappa_sigmaf * np.exp(
        -4 * np.log(2) * u_grid ** 2 / L_kappa ** 2)


def width_change_direction_inducing(u_inducing, sigma_ref):
    direction = gaussian_mean(u_inducing, sigma_ref) * u_inducing ** 2 / sigma_ref ** 3
    return direction / np.linalg.norm(direction)


def shift_direction_inducing(u_inducing, sigma_ref):
    direction = gaussian_mean(u_inducing, sigma_ref) * u_inducing / sigma_ref ** 2
    return direction / np.linalg.norm(direction)


def evaluate_departure(state_like, sigma_at_lines, shape_state, positions):
    """ (n_positions, n_grid) departure array via the GP's own conditional-
        mean interpolation from the inducing grid D (shape_state) --
        separable into an independent u-side and x-side weighting. This
        is the SAME function used both inside the fit (evaluated at
        line_position) and by the standalone reconstruction reader
        (evaluated at an arbitrary pixel) -- it only needs u, u_inducing,
        x_inducing, cfg and the fitted shape grid, nothing else from the
        fit's internal state. See reconstruct.py.

        state_like must provide: u, u_inducing, x_inducing, cfg,
        shape_x_length_scale (attributes) -- an OrderState during fitting,
        or a lightweight namespace built by the reader at reconstruction
        time. """
    D = shape_state
    u = state_like.u
    u_inducing = state_like.u_inducing
    x_inducing = state_like.x_inducing
    cfg = state_like.cfg
    l_x = state_like.shape_x_length_scale

    sigma_ref = np.median(sigma_at_lines)
    l_u = cfg.shape_u_length_scale_factor * sigma_ref
    n_u, n_x = len(u_inducing), len(x_inducing)

    R_u = unit_rbf(u_inducing, u_inducing, l_u) + cfg.shape_jitter * np.eye(n_u)
    base_W_u = unit_rbf(u, u_inducing, l_u) @ np.linalg.inv(R_u)
    kappa_grid = shape_kappa(u, sigma_ref, cfg)
    kappa_inducing = shape_kappa(u_inducing, sigma_ref, cfg)
    W_u_full = (kappa_grid[:, None] * base_W_u) / kappa_inducing[None, :]

    R_x = unit_rbf(x_inducing, x_inducing, l_x) + cfg.shape_jitter * np.eye(n_x)
    W_x_lines = unit_rbf(np.asarray(positions), x_inducing, l_x) @ np.linalg.inv(R_x)

    return (W_u_full @ D @ W_x_lines.T).T   # (n_positions, n_grid)


def _check_conditioning(matrix, name, max_condition=1e10):
    eigenvalues = np.linalg.eigvalsh(matrix)
    condition_number = eigenvalues.max() / max(eigenvalues.min(), 1e-300)
    if condition_number > max_condition:
        print(f"  WARNING: {name} condition number = {condition_number:.2e} "
              f"(exceeds {max_condition:.0e}) -- shape_jitter may be insufficient")
    return condition_number


def _build_shape_matrices(state, l_x, line_position, sigma_current):
    cfg = state.cfg
    u, u_inducing, x_inducing = state.u, state.u_inducing, state.x_inducing
    n_u, n_x = len(u_inducing), len(x_inducing)
    sigma_ref = np.median(sigma_current)
    l_u = cfg.shape_u_length_scale_factor * sigma_ref

    R_u = unit_rbf(u_inducing, u_inducing, l_u) + cfg.shape_jitter * np.eye(n_u)
    R_u_inv = np.linalg.inv(R_u)
    kappa_grid = shape_kappa(u, sigma_ref, cfg)
    kappa_inducing = shape_kappa(u_inducing, sigma_ref, cfg)
    W_u_full = (kappa_grid[:, None] * (unit_rbf(u, u_inducing, l_u) @ R_u_inv)) / kappa_inducing[None, :]

    R_x = unit_rbf(x_inducing, x_inducing, l_x) + cfg.shape_jitter * np.eye(n_x)
    R_x_inv = np.linalg.inv(R_x)
    W_x_lines = unit_rbf(line_position, x_inducing, l_x) @ R_x_inv

    width_dir = width_change_direction_inducing(u_inducing, sigma_ref)
    shift_dir = shift_direction_inducing(u_inducing, sigma_ref)
    guard_u = (cfg.shape_identifiability_weight_width * np.outer(width_dir, width_dir)
               + cfg.shape_identifiability_weight_shift * np.outer(shift_dir, shift_dir))
    K_u_prior_inv = (R_u_inv / kappa_inducing[:, None]) / kappa_inducing[None, :]

    return W_u_full, W_x_lines, R_x_inv, K_u_prior_inv, guard_u, sigma_ref


def shape_departure_log_evidence(candidate_l_x, state, line_position, width_coeffs, v_pix):
    """ Closed-form log marginal likelihood (evidence) for a candidate
        l_x, used both by fit_shape_x_length_scale (MAP) and as a
        standalone identifiability diagnostic. """
    data, cfg = state.data, state.cfg
    sigma_current = width_fn(line_position, width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)

    W_u_full, W_x_lines, R_x_inv, K_u_prior_inv, guard_u, sigma_ref = _build_shape_matrices(
        state, candidate_l_x, line_position, sigma_current)

    n_u, n_x = len(state.u_inducing), len(state.x_inducing)
    n_dim = n_u * n_x
    prior_precision = np.kron(K_u_prior_inv, R_x_inv) + np.kron(guard_u, np.eye(n_x))
    normal_matrix = prior_precision.copy()
    normal_vector = np.zeros(n_dim)

    n_data = 0
    log_weight_sum = 0.0
    y_weighted_sq_sum = 0.0

    for m in range(data.n_lines):
        idx = state.fit_window(line_position[m])
        sigma_m = sigma_current[m]
        residual_target = state.flux[idx] - gaussian_pixel_integral(idx, line_position[m], sigma_m, v_pix[m])

        conv = convolution_matrix(state.u, state.du, line_position[m], idx, v_pix[m])
        CW_m = conv @ W_u_full
        design_m = (CW_m[:, :, None] * W_x_lines[m, None, None, :]).reshape(len(idx), n_dim)

        weight = state.inverse_variance[idx]
        normal_matrix += design_m.T @ (weight[:, None] * design_m)
        normal_vector += design_m.T @ (weight * residual_target)

        n_data += len(idx)
        log_weight_sum += np.sum(np.log(weight))
        y_weighted_sq_sum += np.sum(weight * residual_target ** 2)

    _, logdet_prior = np.linalg.slogdet(prior_precision)
    _, logdet_post = np.linalg.slogdet(normal_matrix)
    solution = np.linalg.solve(normal_matrix, normal_vector)
    quad_term = y_weighted_sq_sum - normal_vector @ solution

    return (-0.5 * n_data * np.log(2 * np.pi) + 0.5 * log_weight_sum
            + 0.5 * logdet_prior - 0.5 * logdet_post - 0.5 * quad_term)


def fit_shape_x_length_scale(state, line_position, width_coeffs, v_pix):
    """ MAP estimate of l_x: maximise the closed-form evidence plus a
        LogNormal prior centred on cfg.shape_x_length_scale_init. """
    cfg = state.cfg
    prior_mean = cfg.shape_x_length_scale_init
    prior_log_std = cfg.shape_x_length_scale_prior_log_std

    def negative_log_posterior(log_l_x):
        l_x = np.exp(log_l_x)
        log_ev = shape_departure_log_evidence(l_x, state, line_position, width_coeffs, v_pix)
        log_prior = -0.5 * ((log_l_x - np.log(prior_mean)) / prior_log_std) ** 2
        return -(log_ev + log_prior)

    result = minimize_scalar(
        negative_log_posterior,
        bounds=(np.log(cfg.shape_x_length_scale_bounds[0]), np.log(cfg.shape_x_length_scale_bounds[1])),
        method='bounded', options={'xatol': 1e-3},
    )
    return float(np.exp(result.x))


def fit_shape_departure(state, line_position, width_coeffs, v_pix):
    data, cfg = state.data, state.cfg
    sigma_current = width_fn(line_position, width_coeffs, data.pixel, cfg, state.v_per_pixel_typical)

    W_u_full, W_x_lines, R_x_inv, K_u_prior_inv, guard_u, sigma_ref = _build_shape_matrices(
        state, state.shape_x_length_scale, line_position, sigma_current)
    _check_conditioning(unit_rbf(state.u_inducing, state.u_inducing,
                                  cfg.shape_u_length_scale_factor * sigma_ref)
                         + cfg.shape_jitter * np.eye(len(state.u_inducing)), "R_u")
    _check_conditioning(unit_rbf(state.x_inducing, state.x_inducing, state.shape_x_length_scale)
                         + cfg.shape_jitter * np.eye(len(state.x_inducing)), "R_x")

    n_u, n_x = len(state.u_inducing), len(state.x_inducing)
    n_dim = n_u * n_x
    normal_matrix = np.kron(K_u_prior_inv, R_x_inv) + np.kron(guard_u, np.eye(n_x))
    normal_vector = np.zeros(n_dim)

    for m in range(data.n_lines):
        idx = state.fit_window(line_position[m])
        sigma_m = sigma_current[m]
        residual_target = state.flux[idx] - gaussian_pixel_integral(idx, line_position[m], sigma_m, v_pix[m])

        conv = convolution_matrix(state.u, state.du, line_position[m], idx, v_pix[m])
        CW_m = conv @ W_u_full
        design_m = (CW_m[:, :, None] * W_x_lines[m, None, None, :]).reshape(len(idx), n_dim)

        weight = state.inverse_variance[idx]
        normal_matrix += design_m.T @ (weight[:, None] * design_m)
        normal_vector += design_m.T @ (weight * residual_target)

    solution = np.linalg.solve(normal_matrix, normal_vector)
    return solution.reshape(n_u, n_x)
