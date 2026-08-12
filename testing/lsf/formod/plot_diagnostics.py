"""
Diagnostic plots for the LSF/wavelength-solution reconstruction produced by
lsf_reconstruction.py. Run this after that script has written
lsf_reconstruction_results.npz in the same directory.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def rescaled_position(x):
    return 2 * (x - x_min) / (x_max - x_min) - 1


def gaussian_mean(u_grid, sigma):
    return np.exp(-0.5 * (u_grid / sigma)**2)


fig, axes = plt.subplots(2, 3, figsize=(15, 8))

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

# --- LSF shape at three positions across the order -------------------------
ax = axes[0, 1]
for x_pos, label, colour in [(peak_pixel.min(), 'blue end', 'b'),
                              (0.5 * (peak_pixel.min() + peak_pixel.max()), 'middle', 'g'),
                              (peak_pixel.max(), 'red end', 'r')]:
    weight = Chebyshev.chebvander(rescaled_position(np.array([x_pos])), shape_poly_order)[0]
    sigma = np.interp(x_pos, line_position, line_width)
    lsf = gaussian_mean(u, sigma) + weight @ shape_coeffs
    ax.plot(u, lsf, color=colour, label=f'{label} (FWHM={2.355 * sigma:.2f})')
    ax.plot(u, gaussian_mean(u, sigma), color=colour, ls=':', alpha=0.4)
ax.axhline(0, color='gray', lw=0.5)
ax.legend(fontsize=7)
ax.set_title('LSF (solid) vs. Gaussian component (dotted)')
ax.set_xlabel('u [pixel]')

# --- example single-line fit -----------------------------------------------
ax = axes[0, 2]
m = len(line_position) // 2
i0, i1 = int(line_position[m]) - 8, int(line_position[m]) + 9
idx = np.arange(i0, i1)
ax.errorbar(idx, flux[idx], yerr=flux_err[idx], fmt='o', ms=3, label='data (normalised)')
ax.plot(idx, model[idx], 'r-', label='model')
ax.legend(fontsize=8)
ax.set_title(f'Example line, FWHM = {2.355 * np.interp(line_position[m], line_position, line_width):.2f} pix')
ax.set_xlabel('pixel')

# --- residuals ---------------------------------------------------------
ax = axes[1, 0]
residual = (flux - model) / flux_err
ax.plot(pixel[fitted_mask], residual[fitted_mask], '.', ms=2, alpha=0.4)
ax.axhline(0, color='r', lw=0.8)
ax.set_ylim(-30, 30)
ax.set_title('Normalised residuals, (flux - model) / error')
ax.set_xlabel('pixel')

# --- dispersion residuals ------------------------------------------------
ax = axes[1, 1]
ax.plot(np.arange(len(line_position)), line_position - peak_pixel, '.')
ax.set_title('Fitted line position minus input peak pixel')
ax.set_xlabel('line index')
ax.set_ylabel('pixel')

# --- FWHM(x) --------------------------------------------------------------
ax = axes[1, 2]
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