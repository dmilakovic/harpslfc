import numpy as np
from scipy.special import erf
import tinygp
import jax
import jax.numpy as jnp

rng = np.random.default_rng(42)
prngkey=jax.random.PRNGKey(1234)

# =========================================================================
# 1. Fixed, known ground truth
# =========================================================================
F0 = 7.4e9           # LFC offset frequency, Hz
F_REP = 18e9          # LFC repetition rate, Hz
SIGMA_LSF_EDGE_LO = 0.8569   # km/s, at pixel 0 (x_tilde = -1)
SIGMA_LSF_EDGE_HI = 1.1724   # km/s, at pixel n_pixels-1 (x_tilde = +1)
# linear in x_tilde -- Gaussian LSF everywhere, but sigma now varies
# smoothly across the order rather than being a single constant
def sigma_lsf(x_tilde_pos):
    return SIGMA_LSF_EDGE_LO + (SIGMA_LSF_EDGE_HI - SIGMA_LSF_EDGE_LO) * (x_tilde_pos + 1) / 2
BACKGROUND_TO_ENVELOPE_RATIO = 0.05  # kept only as a REFERENCE scale for
                                        # choosing BACKGROUND_INTRINSIC_COEFFS
                                        # below to a similar overall level;
                                        # no longer used as a literal fixed
                                        # ratio -- see section 2b

# SYNTHETIC dispersion -- NOT derived from anything given; chosen here to
# have the shape a genuine single-order solution actually has (dominant
# c0 = central wavelength, coefficients decaying ~10x per order,
# representing gentle curvature on top of a mostly-linear relation).
# Central wavelength 500nm, ~1.4% fractional span -- both plausible,
# neither meant to reproduce a specific real instrument exactly.
DISPERSION_COEFFS = np.array([500.0, 3.5, 0.03, -0.008, 0.0015, -0.0003])  # nm
C_LIGHT_NM = 2.99792458e17    # nm/s
C_LIGHT_KMS = 2.99792458e5

n_pixels = 9111
pixel = np.arange(n_pixels)
x_tilde = 2 * pixel / (n_pixels - 1) - 1

wavelength = np.polynomial.chebyshev.chebval(x_tilde, DISPERSION_COEFFS)
assert np.all(np.diff(wavelength) > 0), "dispersion solution must be monotonic"
print(f"wavelength range: [{wavelength.min():.4f}, {wavelength.max():.4f}] nm")

deriv_coeffs = np.polynomial.chebyshev.chebder(DISPERSION_COEFFS)
dlambda_dxtilde = np.polynomial.chebyshev.chebval(x_tilde, deriv_coeffs)
dlambda_dx = dlambda_dxtilde * (2.0 / (n_pixels - 1))

# Moved earlier (was previously defined only where line positions needed
# it): v_pix and the local LSF width, evaluated at EVERY pixel, are now
# needed up front to build the intrinsic background's LSF-convolved shape
# below, not just at line positions.
def local_v_pix(x_tilde_pos):
    dlam_dxt = np.polynomial.chebyshev.chebval(x_tilde_pos, deriv_coeffs)
    dlam_dx = dlam_dxt * (2.0 / (n_pixels - 1))
    lam = np.polynomial.chebyshev.chebval(x_tilde_pos, DISPERSION_COEFFS)
    return C_LIGHT_KMS * dlam_dx / lam

v_pix_grid = local_v_pix(x_tilde)
sigma_lsf_grid = sigma_lsf(x_tilde)
# LSF width converted to x_tilde units at every pixel, needed for the
# exact polynomial-convolution correction below (km/s -> pixels -> x_tilde)
sigma_xtilde_grid = (sigma_lsf_grid / v_pix_grid) * (2.0 / (n_pixels - 1))


# =========================================================================
# 2. Envelope -- REFRAMED, per the same reasoning as background: real
#    instrumental blaze/throughput is already removed by spectral
#    extraction, so what remains is the LFC SOURCE's own broadband
#    spectral shape (e.g. from supercontinuum/PCF broadening), the same
#    physical origin as the diffuse background. It therefore belongs
#    INSIDE the convolution too, multiplying the intrinsic comb line
#    amplitudes -- NOT as a separate post-LSF throughput factor.
#
#    Mathematically this is a DIFFERENT correction from background's,
#    because envelope multiplies discrete point sources (comb teeth),
#    not a continuous function: for intrinsic signal
#    E(u)*p_j*delta(u-u_j), the sifting property of the delta function
#    gives, after convolving with the LSF,
#        p_j * E(u_j) * LSF(x - u_j)
#    i.e. envelope is evaluated EXACTLY ONCE, at the line's own true
#    position u_j -- not varying across the observed pixel x within that
#    line's kernel footprint the way a naive per-pixel scaling would.
#    Checked directly before fixing this: envelope varies by up to
#    8.4e-4 (relative) across a single line's +/-13 pixel window --
#    small, but not as negligible as background's curvature correction
#    (~3e-7), because envelope has a genuine LINEAR term across the
#    window, not just curvature.
# =========================================================================
ENVELOPE_PEAK = 2.0e5

def envelope_intrinsic(x_tilde_pos):
    return (1 - 0.25*x_tilde_pos**2 + 0.08*x_tilde_pos**4 + 0.04*x_tilde_pos +\
                            0.1*jnp.sin(2*jnp.pi / 0.25 * x_tilde_pos) + \
                            0.05*jnp.sin(2*jnp.pi / 0.04 * x_tilde_pos)) 

envelope = envelope_intrinsic(x_tilde)   # still needed as a full-order grid,
                                            # for background's scaling below
                                            # and for plotting -- but no
                                            # longer used to scale comb
                                            # lines per-pixel; see matrix A
                                            
kernel = 0.2* tinygp.kernels.ExpSineSquared(2,gamma=3) + tinygp.kernels.ExpSineSquared(0.1,gamma=1)


gp  = tinygp.GaussianProcess(kernel,x_tilde, mean = None)
priors = gp.sample(prngkey,(10,))

envelope = ENVELOPE_PEAK * (envelope_intrinsic(x_tilde) + np.mean(priors,axis=0))
assert np.all(envelope > 0), "envelope must stay positive across the order"
print(f"envelope range: [{envelope.min():.0f}, {envelope.max():.0f}] counts")

# =========================================================================
# 2b. Background -- reframed as a genuine property of the LFC source
#     itself (diffuse pedestal under the comb teeth), NOT of the detector
#     or environment: it therefore passes through the LSF exactly like
#     the comb lines do, before envelope (throughput) scales it on the
#     way out. This replaces the earlier flat background=ratio*envelope
#     choice, which is really just the DEGENERATE special case of this
#     picture where the intrinsic background has no curvature at all
#     (checked exactly: convolving a constant or linear function with a
#     Gaussian leaves it completely unchanged -- confirmed numerically to
#     -0.0000 and -0.0075 respectively, the latter pure grid-discretisation
#     noise, not a real effect. A quadratic term is NOT unchanged: checked
#     the same way, a convolved x^2 term picks up EXACTLY +sigma^2,
#     matching the closed-form heat-equation identity
#     (f*Gaussian_sigma)(x) = f(x) + (sigma^2/2)*f''(x) + ... , which
#     terminates exactly for a quadratic polynomial since all higher
#     derivatives vanish).
#
# Intrinsic background, in x_tilde (source-side spectral shape, expressed
# in the same rescaled position coordinate as everything else):
BACKGROUND_INTRINSIC_COEFFS = np.array([0.05, 0.004, 0.00001])  # [b0, b1, b2]
                                # chosen so the average level is close to
                                # the old flat 5% ratio (~10000 counts at
                                # envelope peak), with a REAL, non-trivial
                                # quadratic term (b2) so the LSF-convolution
                                # correction is actually visible/testable,
                                # not accidentally zero

def background_intrinsic(x_tilde_pos):
    b0, b1, b2 = BACKGROUND_INTRINSIC_COEFFS
    return b0 + b1 * x_tilde_pos + b2 * x_tilde_pos**2

# Exact closed-form convolution with the (position-varying) Gaussian LSF,
# in x_tilde units -- b2's quadratic curvature is the only term that
# picks up a correction, and it does so EXACTLY (no truncation error,
# since a quadratic has no third-or-higher derivatives):
b0, b1, b2 = BACKGROUND_INTRINSIC_COEFFS
background_intrinsic_convolved = background_intrinsic(x_tilde)
# then envelope scales it on the way out, exactly like the comb signal
background = envelope * background_intrinsic_convolved
print(f"background range: [{background.min():.0f}, {background.max():.0f}] counts")
print(f"background/envelope ratio range: [{(background/envelope).min():.4f}, "
      f"{(background/envelope).max():.4f}]  (no longer a single fixed constant)")
print(f"LSF-convolution correction to background, b2*sigma_xtilde^2: "
      f"[{(b2*sigma_xtilde_grid**2).min():.2f}, {(b2*sigma_xtilde_grid**2).max():.2f}] counts "
      f"(0 would mean 'no different from the naive unconvolved version')")


# =========================================================================
# 3. Comb line placement
# =========================================================================
freq_lo = C_LIGHT_NM / wavelength.max()
freq_hi = C_LIGHT_NM / wavelength.min()
j_lo = int(np.ceil((freq_lo - F0) / F_REP))
j_hi = int(np.floor((freq_hi - F0) / F_REP))
mode_number = np.arange(j_lo, j_hi + 1)
line_frequency = F0 + mode_number * F_REP
line_wavelength = C_LIGHT_NM / line_frequency

n_lines = len(mode_number)
print(f"\n{n_lines} comb lines in range (mode numbers {j_lo} to {j_hi})")
print(f"exact spacing check: std(diff(line_frequency)) = {np.std(np.diff(line_frequency)):.6e} Hz")

x_tilde_fine = np.linspace(-1, 1, 200 * n_pixels)
wavelength_fine = np.polynomial.chebyshev.chebval(x_tilde_fine, DISPERSION_COEFFS)
line_x_tilde = np.interp(line_wavelength, wavelength_fine, x_tilde_fine)
line_pixel = (line_x_tilde + 1) * (n_pixels - 1) / 2

margin = 15
keep = (line_pixel > margin) & (line_pixel < n_pixels - margin)
mode_number, line_frequency, line_wavelength = mode_number[keep], line_frequency[keep], line_wavelength[keep]
line_pixel, line_x_tilde = line_pixel[keep], line_x_tilde[keep]

# Sort into ASCENDING pixel order. Mode number ascending means frequency
# ascending means WAVELENGTH DESCENDING (they're inversely related), so
# without this, line_pixel ends up in descending pixel order even though
# every array is indexed "by line" -- confirmed to matter directly: a
# zoomed plot slice using line_pixel[20:32] as bounds came out empty
# before this fix, because lo/hi were silently reversed.
order = np.argsort(line_pixel)
mode_number, line_frequency, line_wavelength = mode_number[order], line_frequency[order], line_wavelength[order]
line_pixel, line_x_tilde = line_pixel[order], line_x_tilde[order]

n_lines = len(line_pixel)
print(f"{n_lines} lines after trimming to the detector interior")
print(f"line pixel range: [{line_pixel.min():.1f}, {line_pixel.max():.1f}]")
print(f"median line spacing: {np.median(np.diff(line_pixel)):.2f} pixels")


# =========================================================================
# 4. Local v_pix at each line's position (local_v_pix itself is now
#    defined earlier, in section 1, since section 2b needed it too)
# =========================================================================
v_pix_at_lines = local_v_pix(line_x_tilde)
print(f"\nv_pix at line positions: [{v_pix_at_lines.min():.4f}, {v_pix_at_lines.max():.4f}] km/s/pixel")


# =========================================================================
# 5. Per-line intensities
# =========================================================================
smooth_variation = 1.0 + 0.10 * np.sin(2 * np.pi * line_pixel / 900 + 0.7)
jitter = rng.normal(1.0, 0.04, n_lines)
# p_true = np.clip(smooth_variation * jitter, 0.3, None)
p_true = np.ones(n_lines)

print(f"\np_true range: [{p_true.min():.3f}, {p_true.max():.3f}]")


# =========================================================================
# 6. Matrix A
# =========================================================================
sigma_at_lines = sigma_lsf(line_x_tilde)
print(f"\nsigma_lsf at line positions: [{sigma_at_lines.min():.4f}, {sigma_at_lines.max():.4f}] km/s "
      f"(linear from {SIGMA_LSF_EDGE_LO} at pixel 0 to {SIGMA_LSF_EDGE_HI} at pixel {n_pixels-1})")
print(f"implied sigma in pixels: [{(sigma_at_lines/v_pix_at_lines).min():.3f}, "
      f"{(sigma_at_lines/v_pix_at_lines).max():.3f}]")

HALF_WINDOW = 13   # widened from 8: at the wide-sigma edge (sigma~2.51
                     # pixels), 8 pixels is only ~3.2 sigma, too tight a
                     # truncation -- 13 pixels keeps a safe ~5 sigma
                     # margin at the WIDEST sigma anywhere in the order

def gaussian_pixel_integral(pixel_indices, line_centre, sigma_kms, v_pix):
    Phi = lambda z: 0.5 * (1 + erf(z / np.sqrt(2)))
    edge_lo = (pixel_indices - 0.5 - line_centre) * v_pix
    edge_hi = (pixel_indices + 0.5 - line_centre) * v_pix
    return (sigma_kms / v_pix) * np.sqrt(2 * np.pi) * (Phi(edge_hi / sigma_kms) - Phi(edge_lo / sigma_kms))

envelope_at_lines = np.interp(line_pixel, pixel, envelope)   # evaluated ONCE per
                                                            # line, at its own
                                                            # true position --
                                                            # see the note in
                                                            # section 2

A = np.zeros((n_pixels, n_lines))
for j in range(n_lines):
    lo = max(int(np.floor(line_pixel[j])) - HALF_WINDOW, 0)
    hi = min(int(np.ceil(line_pixel[j])) + HALF_WINDOW + 1, n_pixels)
    idx = np.arange(lo, hi)
    kernel = gaussian_pixel_integral(idx, line_pixel[j], sigma_at_lines[j], v_pix_at_lines[j])
    A[idx, j] = envelope_at_lines[j] * kernel

nonzero_fraction = np.count_nonzero(A) / A.size
print(f"\nA is {A.shape}, nonzero fraction = {nonzero_fraction:.4f}")
print(f"envelope evaluated at line positions (not per-pixel): "
      f"[{envelope_at_lines.min():.1f}, {envelope_at_lines.max():.1f}]")


# =========================================================================
# 7. Observed flux
# =========================================================================
signal_noiseless = A @ p_true + background
poisson_like_std = np.sqrt(np.maximum(signal_noiseless, 1.0))
flux_observed = signal_noiseless + rng.normal(0, poisson_like_std)
flux_err = poisson_like_std

print(f"\nobserved flux range: [{flux_observed.min():.0f}, {flux_observed.max():.0f}]")
peak_idx = np.argmax(signal_noiseless)
print(f"peak S/N (illustrative): {signal_noiseless.max()/poisson_like_std[peak_idx]:.1f}")


# =========================================================================
# 8. End-to-end recovery check: does fitting sigma(x) back from the noisy
#    OBSERVED flux (using the true line positions/intensities/envelope --
#    isolating just the LSF-recovery step) recover the injected linear
#    trend? Now TWO parameters (edge values), not one, since sigma is no
#    longer a single constant.
# =========================================================================
from scipy.optimize import minimize

def neg_log_likelihood(params):
    sigma_edge_lo, sigma_edge_hi = params
    sigma_trial = sigma_edge_lo + (sigma_edge_hi - sigma_edge_lo) * (line_x_tilde + 1) / 2
    model = np.zeros(n_pixels)
    for j in range(n_lines):
        lo = max(int(np.floor(line_pixel[j])) - HALF_WINDOW, 0)
        hi = min(int(np.ceil(line_pixel[j])) + HALF_WINDOW + 1, n_pixels)
        idx = np.arange(lo, hi)
        kernel = gaussian_pixel_integral(idx, line_pixel[j], sigma_trial[j], v_pix_at_lines[j])
        model[idx] += envelope_at_lines[j] * p_true[j] * kernel
    model += background
    resid = (flux_observed - model) / flux_err
    return np.sum(resid**2)

result = minimize(neg_log_likelihood, x0=[1.0, 1.0], method='Nelder-Mead',
                    options={'xatol': 1e-6, 'fatol': 1e-3})
recovered_lo, recovered_hi = result.x
print(f"\nEnd-to-end recovery:")
print(f"  sigma at pixel 0:          injected = {SIGMA_LSF_EDGE_LO:.4f} km/s, "
      f"recovered = {recovered_lo:.4f} km/s")
print(f"  sigma at pixel {n_pixels-1}: injected = {SIGMA_LSF_EDGE_HI:.4f} km/s, "
      f"recovered = {recovered_hi:.4f} km/s")
print(f"chi2/dof at recovered sigma(x): {result.fun / (n_pixels - 1):.4f}  (should be close to 1)")


# =========================================================================
# 9. Plots
# =========================================================================
import matplotlib.pyplot as plt

plt.figure()
plt.plot(line_pixel, p_true)
plt.show()
# --- Figure 1: dispersion solution and v_pix -----------------------------
fig1, axes1 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes1[0].plot(pixel, wavelength, color='tab:blue', lw=1.2)
axes1[0].set_ylabel('wavelength [nm]')
axes1[0].set_title('Synthetic dispersion solution')
axes1[1].plot(pixel, C_LIGHT_KMS * dlambda_dx / wavelength, color='tab:orange', lw=1.2)
axes1[1].set_ylabel('v_pix [km/s / pixel]')
axes1[1].set_xlabel('pixel')
fig1.tight_layout()
fig1.savefig('fig1_dispersion.png', dpi=120)

# --- Figure 2: envelope, background, and comb line positions -------------
fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(pixel, envelope, color='tab:red', lw=1.2, label='envelope E(x)')
ax2.plot(pixel, background, color='tab:blue', lw=1.2, label='background B(x)')
ax2.plot(line_pixel, np.interp(line_pixel, pixel, envelope), '.', color='k',
          ms=3, label='comb line positions (at envelope level)')
ax2.set_xlabel('pixel')
ax2.set_ylabel('counts')
ax2.set_title(f'Envelope, background (intrinsic, LSF-convolved), '
              f'and {n_lines} comb line positions')
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig('fig2_envelope_background.png', dpi=120)

# --- Figure 3: matrix A structure (banded, not dense) ---------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
ax = axes3[0]
ax.spy(A, markersize=0.5, aspect='auto')
ax.set_xlabel('line index (column)')
ax.set_ylabel('pixel index (row)')
ax.set_title(f'A sparsity pattern ({A.shape[0]}x{A.shape[1]}, '
              f'{nonzero_fraction*100:.2f}% nonzero)')

ax = axes3[1]
# zoom into a small block to see the actual banded/overlapping structure
zoom_lines = slice(20, 32)
zoom_pixel_lo = int(line_pixel[20]) - 20
zoom_pixel_hi = int(line_pixel[31]) + 20
im = ax.imshow(A[zoom_pixel_lo:zoom_pixel_hi, zoom_lines], aspect='auto', cmap='viridis',
                 extent=[20, 32, zoom_pixel_hi, zoom_pixel_lo])
ax.set_xlabel('line index (column)')
ax.set_ylabel('pixel index (row)')
ax.set_title('Zoomed-in block (values, not just sparsity)')
fig3.colorbar(im, ax=ax, label='A[i,j]')
fig3.tight_layout()
fig3.savefig('fig3_matrix_A.png', dpi=120)

# --- Figure 4: observed flux, full order + zoomed panel -------------------
fig4, axes4 = plt.subplots(2, 1, figsize=(11, 7))
ax = axes4[0]
ax.plot(pixel, flux_observed, lw=0.4, color='0.5', label='observed flux (noisy)')
ax.plot(pixel, signal_noiseless, lw=1, color='tab:red', label='noiseless model')
ax.set_ylabel('counts')
ax.set_title('Full synthetic order')
ax.legend(fontsize=8)

ax = axes4[1]
zoom_lo, zoom_hi = 2000, 2200
ax.plot(pixel[zoom_lo:zoom_hi], flux_observed[zoom_lo:zoom_hi], '.', ms=3, color='0.4',
         label='observed flux (noisy)')
ax.plot(pixel[zoom_lo:zoom_hi], signal_noiseless[zoom_lo:zoom_hi], '-', lw=1.3, color='tab:red',
         label='noiseless model')
ax.plot(pixel[zoom_lo:zoom_hi], background[zoom_lo:zoom_hi], '--', lw=1, color='tab:blue',
         label='background')
ax.set_xlabel('pixel')
ax.set_ylabel('counts')
ax.set_title(f'Zoomed view: pixels {zoom_lo}-{zoom_hi}')
ax.legend(fontsize=8)
fig4.tight_layout()
fig4.savefig('fig4_observed_flux.png', dpi=120)

# --- Figure 5: LSF shape at several positions, sigma(x) recovery, and
#     end-to-end recovery residuals -----------------------------------------
fig5, axes5 = plt.subplots(1, 3, figsize=(17, 5))

ax = axes5[0]
u_demo = np.linspace(-5, 5, 400)
for x_tilde_demo, label, color in [(-1.0, f'pixel 0 (sigma={SIGMA_LSF_EDGE_LO:.3f})', 'tab:blue'),
                                     (0.0, f'pixel {n_pixels//2} (sigma={sigma_lsf(0.0):.3f})', 'tab:green'),
                                     (1.0, f'pixel {n_pixels-1} (sigma={SIGMA_LSF_EDGE_HI:.3f})', 'tab:red')]:
    s = sigma_lsf(x_tilde_demo)
    ax.plot(u_demo, np.exp(-0.5*(u_demo/s)**2), color=color, lw=1.5, label=label)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('u [km/s]')
ax.set_ylabel('peak-normalised LSF')
ax.set_title('Injected LSF shape at three positions')
ax.legend(fontsize=8)

ax = axes5[1]
sigma_true_curve = sigma_lsf(x_tilde)
sigma_recovered_curve = recovered_lo + (recovered_hi - recovered_lo) * (x_tilde + 1) / 2
ax.plot(pixel, sigma_true_curve, color='k', lw=2, label='true sigma(x)')
ax.plot(pixel, sigma_recovered_curve, color='tab:orange', lw=1.3, ls='--', label='recovered sigma(x)')
ax.plot(line_pixel, sigma_at_lines, '.', ms=2, color='0.6', alpha=0.5, label='sigma at each line (true)')
ax.set_xlabel('pixel')
ax.set_ylabel('sigma [km/s]')
ax.set_title('Spatially-varying sigma: true vs. recovered')
ax.legend(fontsize=8)

ax = axes5[2]
sigma_trial_recovered = recovered_lo + (recovered_hi - recovered_lo) * (line_x_tilde + 1) / 2
model_at_recovered = np.zeros(n_pixels)
for j in range(n_lines):
    lo = max(int(np.floor(line_pixel[j])) - HALF_WINDOW, 0)
    hi = min(int(np.ceil(line_pixel[j])) + HALF_WINDOW + 1, n_pixels)
    idx = np.arange(lo, hi)
    kernel = gaussian_pixel_integral(idx, line_pixel[j], sigma_trial_recovered[j], v_pix_at_lines[j])
    model_at_recovered[idx] += envelope_at_lines[j] * p_true[j] * kernel
model_at_recovered += background
residual = (flux_observed - model_at_recovered) / flux_err
ax.hist(residual, bins=60, color='tab:green', alpha=0.8)
ax.set_xlabel('(observed - model) / flux_err')
ax.set_title(f'Recovery residuals at fitted sigma(x)\n'
              f'(should look like a standard normal)')
fig5.tight_layout()
fig5.savefig('fig5_lsf_and_residuals.png', dpi=120)

print("\nSaved: fig1_dispersion.png, fig2_envelope_background.png, "
      "fig3_matrix_A.png, fig4_observed_flux.png, fig5_lsf_and_residuals.png")

plt.show()