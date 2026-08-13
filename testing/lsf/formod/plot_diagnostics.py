"""
Diagnostic plots for the LSF/wavelength-solution reconstruction produced by
lsf_reconstruction.py (a.k.a. test_forward_model.py). Run this after that
script has written lsf_reconstruction_results.npz in the same directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy.polynomial import chebyshev as Chebyshev

RESULTS_FILE = 'lsf_reconstruction_results.npz'
EXPECTED_FWHM = 4.92  # pixels, from c / R with R ~ 145000 at 0.420 km/s per pixel

results = np.load(RESULTS_FILE)
u = results['u']
shape_coeffs = results['shape_coeffs']
line_position = results['line_position']
line_width = results['line_width']
shape_poly_order = int(results['shape_poly_order'])
x_min, x_max = results['x_min'], results['x_max']
flux, flux_err = results['flux'], results['flux_err']
model, fitted_mask, pixel = results['model'], results['fitted_mask'], results['pixel']
envelope_grid_full, background_grid_full = results['envelope_grid_full'], results['background_grid_full']
peak_flux, boundary_flux = results['peak_flux'], results['boundary_flux']
peak_pixel, boundary_pixel = results['peak_pixel'], results['boundary_pixel']
flux_raw = results['flux_raw']

# geometry needed to rebuild each line's own fitting window, matching
# lsf_reconstruction.py's PIXEL_SUBSAMPLE / HALF_WINDOW exactly
PIXEL_SUBSAMPLE = 11
HALF_WINDOW = 6
n_pixels = len(pixel)


def rescaled_position(x):
    return 2 * (x - x_min) / (x_max - x_min) - 1


def gaussian_mean(u_grid, sigma):
    return np.exp(-0.5 * (u_grid / sigma)**2)


def fit_window(line_centre):
    lo = max(int(np.floor(line_centre)) - HALF_WINDOW, 0)
    hi = min(int(np.ceil(line_centre)) + HALF_WINDOW, n_pixels - 1)
    return np.arange(lo, hi + 1)


# a fixed set of 16 representative positions across the order, coloured by
# position with the same colormap used throughout, reused by both new panels
_indices = np.linspace(0, len(peak_pixel) - 1, 16).astype(int)
_x_pos_array = peak_pixel[_indices]
_cmap = matplotlib.cm.get_cmap('Spectral')
_norm = matplotlib.colors.Normalize(vmin=peak_pixel.min(), vmax=peak_pixel.max())
_colours = _cmap(_norm(_x_pos_array))


fig, axes = plt.subplots(2, 4, figsize=(20, 8))

# --- envelope / background -------------------------------------------------
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

# --- LSF shape at 16 positions across the order -----------------------
ax = axes[0, 1]
for x_pos, colour in zip(_x_pos_array, _colours):
    weight = Chebyshev.chebvander(rescaled_position(np.array([x_pos])), shape_poly_order)[0]
    sigma = np.interp(x_pos, line_position, line_width)
    lsf = gaussian_mean(u, sigma) + weight @ shape_coeffs
    ax.plot(u, lsf, color=colour, label=f'{x_pos:.0f} (FWHM={2.355 * sigma:.2f})')
    ax.plot(u, gaussian_mean(u, sigma), color=colour, ls=':', alpha=0.4)
ax.axhline(0, color='gray', lw=0.5)
ax.legend(fontsize=6, ncol=1)
ax.set_title('LSF (solid) vs. Gaussian component (dotted)')
ax.set_xlabel('u [pixel]')

# --- NEW: departure from Gaussianity, phi(u) - Gaussian(u), same 16 positions
ax = axes[0, 2]
for x_pos, colour in zip(_x_pos_array, _colours):
    weight = Chebyshev.chebvander(rescaled_position(np.array([x_pos])), shape_poly_order)[0]
    departure_curve = weight @ shape_coeffs   # = lsf - gaussian_mean(u, sigma), by construction
    ax.plot(u, departure_curve, color=colour)
ax.axhline(0, color='gray', lw=0.5)
ax.set_title('Departure from Gaussian: phi(u) - Gaussian(u)')
ax.set_xlabel('u [pixel]')

# --- example single-line fit -----------------------------------------------
ax = axes[0, 3]
m = len(line_position) // 2
i0, i1 = int(line_position[m]) - 8, int(line_position[m]) + 9
idx = np.arange(i0, i1)
ax.errorbar(idx, flux[idx], yerr=flux_err[idx], fmt='o', ms=3, label='data (normalised)')
ax.plot(idx, model[idx], 'r-', label='model')
ax.legend(fontsize=8)
ax.set_title(f'Example line, FWHM = {2.355 * np.interp(line_position[m], line_position, line_width):.2f} pix')
ax.set_xlabel('pixel')

# --- NEW: every line's data, in its own local (u = pixel - centre) frame,
#     overplotted with the oversampled model at the same 16 positions.
#     This is the master-profile check: if the shared LSF genuinely
#     describes every line, the scatter should hug the curves everywhere,
#     with no systematic colour-dependent (i.e. position-dependent) offset
#     and no systematic pattern as a function of u.
ax = axes[1, 0]
for m in range(len(line_position)):
    idx = fit_window(line_position[m])
    u_data = idx - line_position[m]
    colour = _cmap(_norm(line_position[m]))
    ax.plot(u_data, flux[idx], '.', ms=2, alpha=0.15, color=colour)
u_fine = np.linspace(u.min(), u.max(), 2000)
for x_pos, colour in zip(_x_pos_array, _colours):
    weight = Chebyshev.chebvander(rescaled_position(np.array([x_pos])), shape_poly_order)[0]
    sigma = np.interp(x_pos, line_position, line_width)
    lsf_native = gaussian_mean(u, sigma) + weight @ shape_coeffs
    lsf_fine = np.interp(u_fine, u, lsf_native)   # oversampled beyond the native subpixel grid
    ax.plot(u_fine, lsf_fine, color=colour, lw=1.2)
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylim(-0.3, 1.3)
ax.set_title('All lines (points) + oversampled LSF model (lines)')
ax.set_xlabel('u = pixel - fitted line centre')

# --- residuals ---------------------------------------------------------
ax = axes[1, 1]
residual = (flux - model) / flux_err
ax.plot(pixel[fitted_mask], residual[fitted_mask], '.', ms=2, alpha=0.4)
ax.axhline(0, color='r', lw=0.8)
ax.set_ylim(-30, 30)
ax.set_title('Normalised residuals, (flux - model) / error')
ax.set_xlabel('pixel')

# --- dispersion residuals ------------------------------------------------
ax = axes[1, 2]
ax.plot(np.arange(len(line_position)), line_position - peak_pixel, '.')
ax.set_title('Fitted line position minus input peak pixel')
ax.set_xlabel('line index')
ax.set_ylabel('pixel')

# --- FWHM(x) --------------------------------------------------------------
ax = axes[1, 3]
ax.plot(peak_pixel, 2.355 * line_width, lw=2, label='FWHM(x)')
if EXPECTED_FWHM is not None:
    ax.axhline(EXPECTED_FWHM, color='k', ls='--', lw=1,
               label=f'expected ({EXPECTED_FWHM} pix)')
ax.legend(fontsize=8)
ax.set_title('FWHM across the order')
ax.set_xlabel('pixel')

plt.tight_layout()
plt.savefig('lsf_reconstruction_diagnostics.png', dpi=130)
print("Saved lsf_reconstruction_diagnostics.png")

chi2_per_dof = np.sum(residual[fitted_mask]**2) / fitted_mask.sum()
print(f"chi2 / dof = {chi2_per_dof:.2f}")
print(f"FWHM(x) in [{2.355 * line_width.min():.2f}, {2.355 * line_width.max():.2f}] pix "
      f"(expected ~{EXPECTED_FWHM} pix)")
print(f"rms(fitted position - input peak) = {np.sqrt(np.mean((line_position - peak_pixel)**2)):.4f} pix")

# --- a residual-vs-u check, printed rather than plotted: bin the same
#     residuals used above by u (subpixel offset from each line's own
#     centre) instead of by absolute pixel, and report whether the mean
#     residual in any bin is significantly non-zero. A significant,
#     repeatable pattern here (as opposed to in pixel space) points at the
#     shared LSF shape itself, not the dispersion or envelope/background.
u_of_residual = np.full(n_pixels, np.nan)
for m in range(len(line_position)):
    idx = fit_window(line_position[m])
    u_of_residual[idx] = idx - line_position[m]

bins = np.linspace(u.min(), u.max(), 25)
bin_index = np.digitize(u_of_residual[fitted_mask], bins)
print("\nMean normalised residual binned by u (subpixel offset from line centre):")
for b in range(1, len(bins)):
    sel = bin_index == b
    if sel.sum() > 5:
        mean_r = residual[fitted_mask][sel].mean()
        sem_r = residual[fitted_mask][sel].std() / np.sqrt(sel.sum())
        flag = "  <-- >3 sigma from zero" if abs(mean_r) > 3 * sem_r else ""
        print(f"  u in [{bins[b-1]:+.2f}, {bins[b]:+.2f}]: "
              f"mean = {mean_r:+.3f} +/- {sem_r:.3f} (n={sel.sum()}){flag}")