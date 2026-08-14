"""
Diagnostic plots for the LSF/wavelength-solution reconstruction produced by
lsf_reconstruction_velocity.py. Run this after that script has written
lsf_reconstruction_results.npz in the same directory.

The LSF's own axis (u) is velocity (km/s), not pixels -- see
lsf_reconstruction_velocity.py for the conversion. Position along the
order (x) is still naturally described in pixels/wavelength, so panels
about WHERE ALONG THE ORDER something is (envelope/background, width,
dispersion) stay in pixel units; panels about the LSF's OWN shape (the
LSF itself, its departure from Gaussian, the stacked per-line data) are
in km/s.

DATA-CONSTRAINED MASKING. The native u grid is built wider than any single
line's fitting window actually reaches (deliberately -- it has to cover
every line, including where the local km/s-per-pixel scale v_pix is
larger than typical). This means the full grid extends into a region
where few or no lines' data constrain the fit at all, and the fitted
curve there is governed by the GP prior, not by the data. Every u-space
curve below is masked to only the region within +/- HALF_WINDOW pixels
(converted to velocity via the LOCAL v_pix relevant to that curve) of
wherever it is being evaluated.

PANELS ARE GROUPED BY TYPE, in this order: envelope/background,
LSF shape (u-space), width (x-space), dispersion (wavelength-space),
overall fit diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.polynomial import chebyshev as Chebyshev

RESULTS_FILE = 'lsf_reconstruction_results.npz'
SPEED_OF_LIGHT_KMS = 2.99792458e5
EXPECTED_R = 145000  # resolving power, for the reference line on the FWHM panel
EXPECTED_FWHM_KMS = SPEED_OF_LIGHT_KMS / EXPECTED_R

results = np.load(RESULTS_FILE)
u = results['u']                        # km/s
shape_coeffs = results['shape_coeffs']
width_coeffs = results['width_coeffs']
width_log_sigma_std = results['width_log_sigma_std']
width_raw_target = results['width_raw_target']          # log(sigma), pre-GP
width_raw_target_err = results['width_raw_target_err']  # its uncertainty
line_position = results['line_position']  # pixels
dispersion_position_std = results['dispersion_position_std']  # pixels
wavelength = results['wavelength']  # nm, needed for the dispersion relation panel
line_width = results['line_width']        # km/s
v_pix = results['v_pix']                  # km/s per pixel, per line
shape_poly_order = int(results['shape_poly_order'])
x_min, x_max = results['x_min'], results['x_max']
flux, flux_err = results['flux'], results['flux_err']
model, fitted_mask, pixel = results['model'], results['fitted_mask'], results['pixel']
envelope_grid_full, background_grid_full = results['envelope_grid_full'], results['background_grid_full']
peak_flux, boundary_flux = results['peak_flux'], results['boundary_flux']
peak_pixel, boundary_pixel = results['peak_pixel'], results['boundary_pixel']
flux_raw = results['flux_raw']

PIXEL_SUBSAMPLE = 31
HALF_WINDOW = 7
n_pixels = len(pixel)


def rescaled_position(x):
    return 2 * (x - x_min) / (x_max - x_min) - 1


def gaussian_mean(u_grid, sigma):
    return np.exp(-0.5 * (u_grid / sigma)**2)


def fit_window(line_centre):
    lo = max(int(np.floor(line_centre)) - HALF_WINDOW, 0)
    hi = min(int(np.ceil(line_centre)) + HALF_WINDOW, n_pixels - 1)
    return np.arange(lo, hi + 1)


def v_pix_at(x_pos):
    """ Local km/s-per-pixel scale at an arbitrary pixel position,
        interpolated from the per-line values actually computed during
        fitting (the same way sigma(x) is already interpolated below). """
    return np.interp(x_pos, line_position, v_pix)


def data_constrained_mask(x_pos, u_values):
    """ True where u_values lies within the pixel window that actually has
        data at this position, i.e. +/- HALF_WINDOW pixels converted to
        velocity via the LOCAL v_pix there. Outside this range the fitted
        curve is extrapolation from the GP prior, not from data. """
    half_range = HALF_WINDOW * v_pix_at(x_pos)
    return np.abs(u_values) <= half_range


_indices = np.linspace(0, len(peak_pixel) - 1, 16).astype(int)
_x_pos_array = peak_pixel[_indices]
_cmap = matplotlib.colormaps.get_cmap('Spectral')
_norm = matplotlib.colors.Normalize(vmin=peak_pixel.min(), vmax=peak_pixel.max())
_colours = _cmap(_norm(_x_pos_array))


fig, axes = plt.subplots(3, 4, figsize=(20, 12))

# =========================================================================
# Envelope / background
# =========================================================================
ax = axes[0, 0]
ax.plot(pixel, flux_raw, lw=0.5, color='0.6', label='raw flux')
ax.plot(pixel, envelope_grid_full, 'r-', lw=1, label='envelope E(x)')
ax.plot(pixel, background_grid_full, 'b-', lw=1, label='background B(x)')
ax.plot(peak_pixel, peak_flux, 'r.', ms=3)
ax.plot(boundary_pixel, boundary_flux, 'b.', ms=3)
ax.set_xlim(peak_pixel.min() - 50, peak_pixel.max() + 50)
ax.set_ylim(0, peak_flux.max() * 1.15)
ax.legend(fontsize=8)
ax.set_title('Envelope and background')
ax.set_xlabel('pixel')

# =========================================================================
# LSF shape (u-space): LSF vs Gaussian, departure, all-lines, basis funcs
# =========================================================================
ax = axes[0, 1]
for x_pos, colour in zip(_x_pos_array, _colours):
    weight = Chebyshev.chebvander(rescaled_position(np.array([x_pos])), shape_poly_order)[0]
    sigma = np.interp(x_pos, line_position, line_width)
    lsf = gaussian_mean(u, sigma) + weight @ shape_coeffs
    mask = data_constrained_mask(x_pos, u)
    ax.plot(u[mask], lsf[mask], color=colour, label=f'{x_pos:.0f} (FWHM={2.355 * sigma:.3f} km/s)')
    ax.plot(u[mask], gaussian_mean(u, sigma)[mask], color=colour, ls=':', alpha=0.4)
ax.axhline(0, color='gray', lw=0.5)
ax.legend(fontsize=6, ncol=1)
ax.set_title('LSF (solid) vs. Gaussian component (dotted)\nmasked to data-constrained u')
ax.set_xlabel('u [km/s]')

ax = axes[0, 2]
for x_pos, colour in zip(_x_pos_array, _colours):
    weight = Chebyshev.chebvander(rescaled_position(np.array([x_pos])), shape_poly_order)[0]
    departure_curve = weight @ shape_coeffs
    mask = data_constrained_mask(x_pos, u)
    ax.plot(u[mask], departure_curve[mask], color=colour)
ax.axhline(0, color='gray', lw=0.5)
ax.set_title('Departure from Gaussian: phi(u) - Gaussian(u)\nmasked to data-constrained u')
ax.set_xlabel('u [km/s]')

ax = axes[0, 3]
for m in range(len(line_position)):
    idx = fit_window(line_position[m])
    u_data = (idx - line_position[m]) * v_pix[m]   # pixel offset -> km/s
    colour = _cmap(_norm(line_position[m]))
    ax.plot(u_data, flux[idx], '.', ms=2, alpha=0.15, color=colour)
u_fine = np.linspace(u.min(), u.max(), 2000)
for x_pos, colour in zip(_x_pos_array, _colours):
    weight = Chebyshev.chebvander(rescaled_position(np.array([x_pos])), shape_poly_order)[0]
    sigma = np.interp(x_pos, line_position, line_width)
    lsf_native = gaussian_mean(u, sigma) + weight @ shape_coeffs
    lsf_fine = np.interp(u_fine, u, lsf_native)
    mask_fine = data_constrained_mask(x_pos, u_fine)
    ax.plot(u_fine[mask_fine], lsf_fine[mask_fine], color=colour, lw=1.2)
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylim(-0.3, 1.3)
ax.set_title('All lines (points) + oversampled LSF model (lines)\nmodel masked to data-constrained u')
ax.set_xlabel('u = (pixel - fitted line centre) x v_pix [km/s]')

# --- shape departure basis functions d_k(u), one per Chebyshev order -----
# These ARE shape_coeffs -- the departure at any position x is
# sum_k T_k(rescaled_position(x)) * d_k(u), so d_0(u) is the order-averaged
# departure shape, d_1(u) is how much of a LINEAR-in-x correction gets
# added, d_2(u) a QUADRATIC-in-x correction, and so on. Masked to the
# region most lines' own windows actually reach (using the median v_pix
# across the order as a representative scale, since these basis functions
# are shared across the whole order rather than evaluated at one position).
ax = axes[1, 0]
median_half_range = HALF_WINDOW * np.median(v_pix)
mask_shared = np.abs(u) <= median_half_range
for k in range(shape_coeffs.shape[0]):
    ax.plot(u[mask_shared], shape_coeffs[k][mask_shared], label=f'd_{k}(u)')
ax.axhline(0, color='gray', lw=0.5)
ax.legend(fontsize=8)
ax.set_title('Shape departure basis functions d_k(u)\n(masked to median data-constrained u)')
ax.set_xlabel('u [km/s]')

# =========================================================================
# Width (x-space): FWHM(x) with uncertainty and resolving power, and the
# raw per-line data that GP fit was built from
# =========================================================================
# The two previous versions of this plot each showed a separate FWHM(x)
# panel (one from line_width interpolated at the comb lines, one from the
# raw fitted grid) -- these were redundant, since line_width is itself
# just an interpolation of the same grid. Merged into one panel here,
# keeping BOTH pieces of information that were unique to each (the GP
# uncertainty band, and the resolving-power twin axis).
ax = axes[1, 1]
sigma_grid = np.maximum(np.exp(width_coeffs), 0.05 * np.median(v_pix))
fwhm_grid = 2.355 * sigma_grid
fwhm_upper = 2.355 * np.exp(width_coeffs + width_log_sigma_std)
fwhm_lower = 2.355 * np.exp(width_coeffs - width_log_sigma_std)
ax.plot(pixel, fwhm_grid, color='tab:blue', lw=1.3, label='FWHM(x)')
ax.fill_between(pixel, fwhm_lower, fwhm_upper, color='tab:blue', alpha=0.3,
                label='GP posterior std')
ax.axhline(EXPECTED_FWHM_KMS, color='k', ls='--', lw=1,
           label=f'R={EXPECTED_R} ({EXPECTED_FWHM_KMS:.3f} km/s)')
ax.legend(fontsize=8)
ax.set_xlim(peak_pixel.min() - 50, peak_pixel.max() + 50)
ax.set_title('Width: FWHM(x) [km/s], from the fitted GP grid')
ax.set_xlabel('pixel')
ax.set_ylabel('FWHM [km/s]')
ax_r = ax.twinx()
ax_r.set_ylim(SPEED_OF_LIGHT_KMS / ax.get_ylim()[1], SPEED_OF_LIGHT_KMS / max(ax.get_ylim()[0], 1e-6))
ax_r.set_ylabel('R = c / FWHM')

# --- raw per-line width data (pre-GP) vs the fitted GP curve --------------
# Directly answers "does the GP curve look like a reasonable smoothing of
# the actual per-line data, or is it doing something strange": each
# line's own 1-parameter local correction (in log(sigma), before any
# smoothing), with its formal uncertainty, plotted against the final
# fitted grid. If the GP curve looks like a sensible smooth trend through
# this scatter, the fit is behaving as intended; if the scatter looks
# structured in a way the curve does not track, or the curve looks
# unreasonably wiggly relative to the scatter, that is a sign of trouble
# worth investigating directly here rather than only in the final FWHM(x).
ax = axes[1, 2]
ax.errorbar(line_position, width_raw_target, yerr=width_raw_target_err,
            fmt='.', ms=4, elinewidth=0.5, capsize=0, alpha=0.5,
            label='per-line target (pre-GP)')
ax.plot(pixel, width_coeffs, '-', color='tab:orange', lw=1.5, label='GP-fitted curve')
ax.legend(fontsize=8)
ax.set_xlim(peak_pixel.min() - 50, peak_pixel.max() + 50)
ax.set_title('Width: raw per-line data vs. fitted GP curve')
ax.set_xlabel('pixel')
ax.set_ylabel('log(sigma [km/s])')

# =========================================================================
# Dispersion (wavelength-space): the relation itself, and residuals
# =========================================================================
ax = axes[1, 3]
wavelength_order = np.argsort(wavelength)
lam_sorted = wavelength[wavelength_order]
pos_sorted = line_position[wavelength_order]
std_sorted = dispersion_position_std[wavelength_order]
ax.plot(lam_sorted, pos_sorted, '-', color='tab:green', lw=1)
ax.fill_between(lam_sorted, pos_sorted - std_sorted, pos_sorted + std_sorted,
                color='tab:green', alpha=0.3, label='GP posterior std')
ax.legend(fontsize=8)
ax.set_title('Dispersion relation: x(lambda)')
ax.set_xlabel('wavelength [nm]')
ax.set_ylabel('pixel')

# --- dispersion residuals, with the GP's own uncertainty band ------------
# Plotted against detector position (not line index), with vertical lines
# at 1/8-detector boundaries: if a pattern lines up with these, that
# points to a per-amplifier readout effect (many CCDs used in echelle
# spectrographs are read out through several amplifiers, each covering an
# equal share of the columns, and small gain/offset mismatches between
# them show up exactly at these boundaries).
ax = axes[2, 0]
ax.errorbar(line_position, line_position - peak_pixel,
            yerr=dispersion_position_std, fmt='.', ms=4, elinewidth=0.5, capsize=0)
ax.axhline(0, color='gray', lw=0.5)
[ax.axvline(n_pixels / 8 * i, color='red', lw=0.8) for i in range(9)]
ax.set_title('Fitted line position minus input peak pixel\n(error bars: GP posterior std)')
ax.set_xlabel('pixel')
ax.set_ylabel('pixel')

# =========================================================================
# Overall fit diagnostics
# =========================================================================
ax = axes[2, 1]
m = len(line_position) // 2
i0, i1 = int(line_position[m]) - 8, int(line_position[m]) + 9
idx = np.arange(i0, i1)
ax.errorbar(idx, flux[idx], yerr=flux_err[idx], fmt='o', ms=3, label='data (normalised)')
ax.plot(idx, model[idx], 'r-', label='model')
ax.legend(fontsize=8)
ax.set_title(f'Example line, FWHM = {2.355 * np.interp(line_position[m], line_position, line_width):.3f} km/s')
ax.set_xlabel('pixel')

ax = axes[2, 2]
residual = (flux - model) / flux_err
ax.plot(pixel[fitted_mask], residual[fitted_mask], '.', ms=2, alpha=0.4)
ax.axhline(0, color='r', lw=0.8)
ax.set_ylim(-30, 30)
ax.set_title('Normalised residuals, (flux - model) / error')
ax.set_xlabel('pixel')

axes[2, 3].axis('off')

plt.tight_layout()
plt.savefig('lsf_reconstruction_diagnostics.png', dpi=130)
print("Saved lsf_reconstruction_diagnostics.png")

chi2_per_dof = np.sum(residual[fitted_mask]**2) / fitted_mask.sum()
print(f"chi2 / dof = {chi2_per_dof:.2f}")
print(f"FWHM(x) in [{fwhm_grid.min():.4f}, {fwhm_grid.max():.4f}] km/s "
      f"(expected ~{EXPECTED_FWHM_KMS:.4f} km/s for R={EXPECTED_R})")
print(f"resolving power R in [{SPEED_OF_LIGHT_KMS/fwhm_grid.max():.0f}, "
      f"{SPEED_OF_LIGHT_KMS/fwhm_grid.min():.0f}]")
print(f"rms(fitted position - input peak) = {np.sqrt(np.mean((line_position - peak_pixel)**2)):.4f} pix")

# --- residual autocorrelation check, within each line's own window -------
resid_full = (flux - model) / flux_err
max_lag = 4
lag_products = {lag: [] for lag in range(1, max_lag + 1)}
var_terms = []
for m in range(len(line_position)):
    idx = fit_window(line_position[m])
    r = resid_full[idx]
    var_terms.append(np.mean(r**2))
    for lag in range(1, max_lag + 1):
        if len(r) > lag:
            lag_products[lag].append(np.mean(r[:-lag] * r[lag:]))
mean_var = np.mean(var_terms)
print("\nResidual autocorrelation within each line's own window (1.0 would be perfectly correlated):")
for lag in range(1, max_lag + 1):
    ac = np.mean(lag_products[lag]) / mean_var
    print(f"  lag {lag}: {ac:+.4f}")