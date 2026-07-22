#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:00:15 2026

@author: dmilakov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_single_segment.py

End-to-end test of the two-phase iterative GP fit on a single segment.
Tests each component individually before running the full pipeline,
making it easy to pinpoint exactly where failures occur.

Phases tested:
  1. Data loading and batching
  2. Starts and bounds generation (shape checks)
  3. Phase 1: use_scatter=False — jit+vmap fit
  4. Scatter training (CPU, between phases)
  5. Phase 2: use_scatter=True  — jit+vmap fit with rescaled errors
  6. Full pipeline timing (compile vs cached)
  7. Sanity checks on results

Usage:
    python test_single_segment.py [/path/to/spectrum.fits]
    
    If no filepath is given, falls back to the hardcoded ESPRESSO path.
"""

import sys
import time
import logging
import traceback

import numpy as np
import jax
import jax.numpy as jnp

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s  %(levelname)-8s  %(message)s',
)
logger = logging.getLogger(__name__)

# ── Imports ───────────────────────────────────────────────────────────────────
import harps.spectrum   as hc
import harps.lsf.gp     as lsfgp
import harps.lsf.gp_aux as gp_aux
import harps.lsf.aux    as aux
import harps.lsf.batch  as batch_utils
import harps.version    as hv
import harps.settings   as hs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _ok(label: str, value: str = '') -> None:
    print(f"  ✓  {label}  {value}")


def _fail(label: str, detail: str = '') -> None:
    print(f"  ✗  {label}  {detail}")


def _check(condition: bool, label: str, detail: str = '') -> bool:
    if condition:
        _ok(label, detail)
    else:
        _fail(label, detail)
    return condition


def _check_shapes(d: dict, expected_ndim: int, context: str) -> bool:
    all_ok = True
    for k, v in d.items():
        v = jnp.array(v)
        ok = v.ndim == expected_ndim
        _check(ok,
               f"{context}['{k}'] ndim={v.ndim}",
               f"shape={v.shape}")
        all_ok = all_ok and ok
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_spectrum(filepath: str | None,
                  order   : int,
                  scale   : str,
                  ) -> tuple:
    """
    Load spectrum and return (x2d, flx2d, err2d, seglims) for the
    requested order.
    """
    version = hv.item_to_version(
        dict(iteration=1, model_scatter=False, interpolate=False),
        ftype='lsf',
    )

    if filepath is not None:
        spec = hc.Spectrum(filepath)
    else:
        filename      = ('/Users/dmilakov/projects/j1333/lfc/'
                         'LFC_reduced_files/'
                         '2023-02-22_ESPRESSO_S2D_LFC_FP_A.fits')
        wavereference = ('/Users/dmilakov/projects/j1333/lfc/'
                         'LFC_reduced_files/'
                         '2023-02-22_ESPRESSO_LFC_WAVE_MATRIX_A.fits')
        spec = hc.ESPRESSO(
            filename,
            f0=7.40e9, fr=18e9,
            overwrite=False, sOrder=40,
            wavereference=wavereference,
        )

    pix3d, vel3d, flx3d, err3d, _ = aux.stack_spectrum(
        spec,
        version       = version,
        wavesol_version = 700,
        orders        = order,
        subbkg        = hs.subbkg,
        divenv        = hs.divenv,
    )

    x2d   = pix3d[:, :, 0] if scale == 'pixel' else vel3d[:, :, 0]
    flx2d = flx3d[:, :, 0]
    err2d = err3d[:, :, 0]

    return x2d, flx2d, err2d


# ─────────────────────────────────────────────────────────────────────────────
# Main test function
# ─────────────────────────────────────────────────────────────────────────────

def test_single_segment(filepath     : str | None = None,
                        order        : int   = 50,
                        seg_index    : int   = 8,
                        numseg       : int   = 16,
                        num_starts   : int   = 4,
                        numiter      : int   = 5,
                        maxiter      : int   = 300,
                        scale        : str   = 'pixel',
                        model_scatter: bool  = True,
                        ) -> dict:
    """
    Full two-phase test on one segment.

    Returns a dict with all intermediate and final results so the caller
    can inspect anything interactively.
    """
    print(f"\n{'='*60}")
    print(f"  Two-phase GP fit test")
    print(f"  order={order}  seg={seg_index}  model_scatter={model_scatter}")
    print(f"{'='*60}")

    results = {}
    all_passed = True

    # ── Step 1: Load data ────────────────────────────────────────────────────
    _section("Step 1: Load spectrum and extract segment")
    try:
        x2d, flx2d, err2d = load_spectrum(filepath, order, scale)
        _ok("Spectrum loaded", f"x2d shape={x2d.shape}")
    except Exception as e:
        _fail("Spectrum loading FAILED", str(e))
        traceback.print_exc()
        return {}

    npix    = x2d.shape[1]
    seglims = np.linspace(0, npix, numseg + 1, dtype=int)
    pixl    = seglims[seg_index]
    pixr    = seglims[seg_index + 1]

    x1s   = np.ravel(x2d  [order, pixl:pixr])
    flx1s = np.ravel(flx2d[order, pixl:pixr])
    err1s = np.ravel(err2d[order, pixl:pixr])

    _ok(f"Segment pixels {pixl}→{pixr}", f"{len(x1s)} data points")
    _ok(f"Flux range",
        f"{float(flx1s.min()):.2f} → {float(flx1s.max()):.2f}")
    _check(np.all(np.isfinite(flx1s)),
           "All flux values finite")
    _check(np.all(err1s > 0),
           "All errors positive")

    # ── Step 2: Build batch ──────────────────────────────────────────────────
    _section("Step 2: Build padded SegmentBatch (1 segment)")
    try:
        full_batch = batch_utils.make_batch(x2d, flx2d, err2d,
                                            seglims, [order])
        single = batch_utils.SegmentBatch(
            x    = full_batch.x   [seg_index:seg_index + 1],
            flx  = full_batch.flx [seg_index:seg_index + 1],
            err  = full_batch.err [seg_index:seg_index + 1],
            mask = full_batch.mask[seg_index:seg_index + 1],
            meta = full_batch.meta[seg_index:seg_index + 1],
        )
        max_len = single.x.shape[1]
        n_real  = int(jnp.sum(single.mask[0]))
        _ok("SegmentBatch created",
            f"shape=(1, {max_len})  real points={n_real}")
        _check(n_real > 10,
               f"Enough real points ({n_real} > 10)")
        results['single'] = single
    except Exception as e:
        _fail("SegmentBatch creation FAILED", str(e))
        traceback.print_exc()
        return results

    # ── Step 3: Generate starts and bounds ───────────────────────────────────
    _section("Step 3: Generate starts and bounds")
    try:
        starts = lsfgp.generate_starts_batch(
            single.x, single.flx, single.err, num_starts
        )
        bounds = lsfgp.generate_bounds_batch(
            single.x, single.flx, single.err
        )
        lower_b, upper_b = bounds

        print("\n  Starts leaf shapes (expected: (1, num_starts) = "
              f"(1, {num_starts})):")
        starts_ok = _check_shapes(starts, expected_ndim=2, context='starts')

        print("\n  Bounds lower leaf shapes (expected: (1,)):")
        bounds_ok = _check_shapes(lower_b, expected_ndim=1, context='lower_bounds')

        # Check bounds are consistent: lower < upper for every param
        print("\n  Bounds consistency (lower < upper):")
        for k in lower_b:
            lo = float(jnp.array(lower_b[k])[0])
            hi = float(jnp.array(upper_b[k])[0])
            _check(lo < hi, f"  {k:20s}", f"lo={lo:.4f} < hi={hi:.4f}")

        # Check starts are within bounds
        print("\n  Starts within bounds:")
        for k in starts:
            s   = np.array(starts[k][0])   # (num_starts,)
            lo  = float(jnp.array(lower_b[k])[0])
            hi  = float(jnp.array(upper_b[k])[0])
            ok  = np.all((s >= lo) & (s <= hi))
            _check(ok, f"  {k:20s}",
                   f"range=[{s.min():.3f}, {s.max():.3f}]")

        all_passed = all_passed and starts_ok and bounds_ok
        results['starts'] = starts
        results['bounds'] = bounds

    except Exception as e:
        _fail("Starts/bounds generation FAILED", str(e))
        traceback.print_exc()
        return results

    # ── Step 4: Phase 1 (use_scatter=False) ──────────────────────────────────
    _section("Step 4: Phase 1 — use_scatter=False (compile + run)")
    try:
        phase1_fitter = lsfgp.make_phase_fitter(
            use_scatter = False,
            numiter     = numiter,
            maxiter     = maxiter,
        )

        # First call: compilation
        t0 = time.time()
        phase1_state = phase1_fitter(
            single.x, single.flx, single.err, single.mask,
            starts, bounds,
            single.err,      # scatter_y_err = original err in Phase 1
        )
        jax.block_until_ready(phase1_state.params)
        t_compile = time.time() - t0
        _ok(f"Phase 1 compile+run", f"{t_compile:.2f}s")

        # Second call: cached
        t0 = time.time()
        phase1_state = phase1_fitter(
            single.x, single.flx, single.err, single.mask,
            starts, bounds,
            single.err,
        )
        jax.block_until_ready(phase1_state.params)
        t_cached = time.time() - t0
        _ok(f"Phase 1 cached run", f"{t_cached:.4f}s")

        # Extract results for seg 0
        p1 = jax.tree_util.tree_map(lambda a: float(a[0]), phase1_state.params)
        shift1  = float(phase1_state.shift[0])
        n_good1 = int(jnp.sum(phase1_state.mask[0]))
        converged1 = bool(phase1_state.converged[0])

        print(f"\n  Phase 1 results:")
        print(f"    Final shift  : {shift1:+.6f} pix")
        print(f"    Good points  : {n_good1} / {n_real}")
        print(f"    Converged    : {converged1}")
        print(f"    Params:")
        for k, v in p1.items():
            print(f"      {k:22s} = {v:+.4f}")

        _check(abs(shift1) < 1.0,    "Shift < 1 pix",         f"{shift1:+.4f}")
        _check(n_good1 > 10,         "Enough good points",     str(n_good1))
        _check(p1['mf_amp'] > 0,     "Amplitude positive",     f"{p1['mf_amp']:.4f}")
        _check(np.isfinite(p1['mf_amp']),  "Amplitude finite")
        _check(np.isfinite(p1['gp_log_scale']), "GP scale finite")

        results['phase1_state']  = phase1_state
        results['phase1_params'] = p1
        results['t_phase1_compile'] = t_compile
        results['t_phase1_cached']  = t_cached

    except Exception as e:
        _fail("Phase 1 FAILED", str(e))
        traceback.print_exc()
        all_passed = False
        return results

    # ── Step 5: Scatter training (CPU, between phases) ────────────────────────
    if model_scatter:
        _section("Step 5: Scatter training (CPU)")
        try:
            t0 = time.time()
            scatter_list, scatter_y_err = lsfgp.train_scatter_batch(
                single.x, single.flx, single.err,
                phase1_state.mask,
                phase1_state.params,
            )
            t_sct = time.time() - t0
            _ok(f"Scatter training", f"{t_sct:.2f}s")

            scatter_0 = scatter_list[0]
            _check(scatter_0 is not None,
                   "Scatter trained successfully for seg 0")

            if scatter_0 is not None:
                theta_sct, logvar_x, logvar_y, logvar_err = scatter_0
                _ok("Scatter GP params",
                    "  ".join(f"{k}={float(v):.3f}"
                               for k, v in theta_sct.items()))
                _ok(f"logvar_x length", str(len(logvar_x)))

            # Check rescaled errors are finite and positive
            sct_err_seg = np.array(scatter_y_err[0])
            good_mask   = np.array(single.mask[0]) > 0.5
            _check(np.all(np.isfinite(sct_err_seg[good_mask])),
                   "Rescaled errors finite for real points")
            _check(np.all(sct_err_seg[good_mask] > 0),
                   "Rescaled errors positive for real points")
            _ok("Rescaled err range (real pts)",
                f"{sct_err_seg[good_mask].min():.4f} → "
                f"{sct_err_seg[good_mask].max():.4f}")

            results['scatter_list']   = scatter_list
            results['scatter_y_err']  = scatter_y_err
            results['t_scatter']      = t_sct

        except Exception as e:
            _fail("Scatter training FAILED", str(e))
            traceback.print_exc()
            all_passed = False
            # Fall back: use original errors for Phase 2
            scatter_list  = [None]
            scatter_y_err = single.err
    else:
        _section("Step 5: Scatter training — SKIPPED (model_scatter=False)")
        scatter_list  = [None]
        scatter_y_err = single.err

    # ── Step 6: Phase 2 (use_scatter=True) ───────────────────────────────────
    if model_scatter:
        _section("Step 6: Phase 2 — use_scatter=True (compile + run)")
        try:
            phase2_fitter = lsfgp.make_phase_fitter(
                use_scatter = True,
                numiter     = numiter,
                maxiter     = maxiter,
            )

            # First call: compilation
            t0 = time.time()
            phase2_state = phase2_fitter(
                single.x, single.flx, single.err,
                phase1_state.mask,   # warm-start mask from Phase 1
                starts, bounds,
                scatter_y_err,       # pre-rescaled errors
            )
            jax.block_until_ready(phase2_state.params)
            t_compile2 = time.time() - t0
            _ok(f"Phase 2 compile+run", f"{t_compile2:.2f}s")

            # Second call: cached
            t0 = time.time()
            phase2_state = phase2_fitter(
                single.x, single.flx, single.err,
                phase1_state.mask,
                starts, bounds,
                scatter_y_err,
            )
            jax.block_until_ready(phase2_state.params)
            t_cached2 = time.time() - t0
            _ok(f"Phase 2 cached run", f"{t_cached2:.4f}s")

            p2 = jax.tree_util.tree_map(
                lambda a: float(a[0]), phase2_state.params
            )
            shift2     = float(phase2_state.shift[0])
            n_good2    = int(jnp.sum(phase2_state.mask[0]))
            converged2 = bool(phase2_state.converged[0])

            print(f"\n  Phase 2 results:")
            print(f"    Final shift  : {shift2:+.6f} pix")
            print(f"    Good points  : {n_good2} / {n_real}")
            print(f"    Converged    : {converged2}")
            print(f"    Params:")
            for k, v in p2.items():
                print(f"      {k:22s} = {v:+.4f}")

            # Compare Phase 1 vs Phase 2 parameters
            print(f"\n  Phase 1 vs Phase 2 parameter differences:")
            for k in p1:
                diff = p2[k] - p1[k]
                print(f"    {k:22s}  Δ = {diff:+.4f}")

            _check(abs(shift2) < 1.0,   "Shift < 1 pix",        f"{shift2:+.4f}")
            _check(n_good2 > 10,        "Enough good points",    str(n_good2))
            _check(p2['mf_amp'] > 0,    "Amplitude positive",    f"{p2['mf_amp']:.4f}")
            _check(np.isfinite(p2['mf_amp']), "Amplitude finite")

            results['phase2_state']     = phase2_state
            results['phase2_params']    = p2
            results['t_phase2_compile'] = t_compile2
            results['t_phase2_cached']  = t_cached2

        except Exception as e:
            _fail("Phase 2 FAILED", str(e))
            traceback.print_exc()
            all_passed = False
    else:
        _section("Step 6: Phase 2 — SKIPPED (model_scatter=False)")

    # ── Step 7: Summary ───────────────────────────────────────────────────────
    _section("Summary")
    final_params = results.get('phase2_params', results.get('phase1_params', {}))
    final_shift  = (float(results['phase2_state'].shift[0])
                    if 'phase2_state' in results
                    else float(results['phase1_state'].shift[0]))

    amp_ok   = 0 < final_params.get('mf_amp', -1) < 2 * float(
        jnp.nanmax(single.flx[0]))
    shift_ok = abs(final_shift) < 1.0
    good_ok  = (int(jnp.sum(results.get(
        'phase2_state', results['phase1_state']).mask[0])) > 10)

    all_passed = all_passed and amp_ok and shift_ok and good_ok

    print(f"  Amplitude reasonable : {'✓' if amp_ok   else '✗'}  "
          f"({final_params.get('mf_amp', float('nan')):.4f})")
    print(f"  Shift < 1 pix        : {'✓' if shift_ok else '✗'}  "
          f"({final_shift:+.4f})")
    print(f"  Enough good points   : {'✓' if good_ok  else '✗'}")

    print(f"\n  Timing:")
    if 't_phase1_compile' in results:
        print(f"    Phase 1 compile : {results['t_phase1_compile']:.2f}s")
        print(f"    Phase 1 cached  : {results['t_phase1_cached']:.4f}s")
    if 't_scatter' in results:
        print(f"    Scatter (CPU)   : {results['t_scatter']:.2f}s")
    if 't_phase2_compile' in results:
        print(f"    Phase 2 compile : {results['t_phase2_compile']:.2f}s")
        print(f"    Phase 2 cached  : {results['t_phase2_cached']:.4f}s")

    print(f"\n{'='*60}")
    print(f"  {'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}")
    print(f"{'='*60}\n")

    results['all_passed'] = all_passed
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick component tests — run these first to isolate issues
# ─────────────────────────────────────────────────────────────────────────────

def test_starts_shapes(num_starts: int = 4, max_len: int = 600) -> bool:
    """
    Verify starts and bounds have the correct shapes before any fitting.
    Catches shape mismatches that would cause cryptic vmap errors.
    """
    _section("Component test: starts and bounds shapes")
    N_seg = 3   # test with a small batch

    # Synthetic data
    key  = jax.random.PRNGKey(0)
    x    = jnp.tile(jnp.linspace(-3, 3, max_len), (N_seg, 1))
    flx  = jax.random.normal(key, (N_seg, max_len)) + 5.0
    err  = jnp.ones((N_seg, max_len)) * 0.1
    mask = jnp.ones((N_seg, max_len))

    starts      = lsfgp.generate_starts_batch(x, flx, err, num_starts)
    lower, upper = lsfgp.generate_bounds_batch(x, flx, err)

    print(f"  N_seg={N_seg}, num_starts={num_starts}, max_len={max_len}")
    ok = True

    print(f"\n  Starts — expected shape (N_seg={N_seg}, num_starts={num_starts}):")
    for k, v in starts.items():
        v = jnp.array(v)
        expected = (N_seg, num_starts)
        match = v.shape == expected
        _check(match, f"    {k:22s}", f"shape={v.shape}")
        ok = ok and match

    print(f"\n  Bounds — expected shape (N_seg={N_seg},):")
    for k, v in lower.items():
        v = jnp.array(v)
        match = v.shape == (N_seg,)
        _check(match, f"    {k:22s}", f"shape={v.shape}")
        ok = ok and match

    print(f"\n  Result: {'✓ shapes correct' if ok else '✗ shape mismatch'}")
    return ok


def test_single_start(max_len: int = 600) -> bool:
    """
    Test run_lsf_optimization_local on a single synthetic segment.
    Isolates the optimizer from vmap/batch logic.
    """
    _section("Component test: single L-BFGS-B run")

    key = jax.random.PRNGKey(42)
    x   = jnp.linspace(-3, 3, max_len)
    # Synthetic Gaussian LSF
    y   = 2.0 * jnp.exp(-0.5 * (x / 0.8)**2) + 0.01 * jax.random.normal(key, (max_len,))
    ye  = jnp.full(max_len, 0.05)

    starts      = lsfgp.generate_starts_batch(
        x[None], y[None], ye[None], num_starts=1
    )
    lower, upper = lsfgp.generate_bounds_batch(x[None], y[None], ye[None])

    # Extract single start — shape (7_params,) per leaf
    start0 = jax.tree_util.tree_map(lambda a: a[0, 0], starts)   # scalar per param
    bounds0 = (
        jax.tree_util.tree_map(lambda a: a[0], lower),
        jax.tree_util.tree_map(lambda a: a[0], upper),
    )

    try:
        t0 = time.time()
        params, loss = lsfgp.run_lsf_optimization_local(
            start0, x, y, ye, use_scatter=False, bounds=bounds0
        )
        dt = time.time() - t0
        _ok("Single run completed", f"{dt:.3f}s  loss={float(loss):.4f}")
        _check(np.isfinite(float(loss)), "Loss is finite")
        _check(float(params['mf_amp']) > 0, "Amplitude positive",
               f"{float(params['mf_amp']):.4f}")
        return True
    except Exception as e:
        _fail("Single run FAILED", str(e))
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else None

    # Run component tests first — fast, no spectrum needed
    print("\n" + "="*60)
    print("  COMPONENT TESTS (synthetic data, no spectrum file needed)")
    print("="*60)
    shapes_ok = test_starts_shapes(num_starts=4)
    single_ok = test_single_start()

    if not (shapes_ok and single_ok):
        print("\n✗ Component tests failed — fix these before running full test.")
        sys.exit(1)

    # Full pipeline test on real data
    print("\n" + "="*60)
    print("  FULL PIPELINE TEST")
    print("="*60)

    # Test Phase 1 only first (faster, no scatter)
    print("\n--- Phase 1 only (model_scatter=False) ---")
    r1 = test_single_segment(
        filepath      = filepath,
        order         = 50,
        seg_index     = 8,
        model_scatter = False,
        numiter       = 5,
        maxiter       = 300,
    )

    if not r1.get('all_passed', False):
        print("\n✗ Phase 1 test failed — fix before testing with scatter.")
        sys.exit(1)

    # Full two-phase test
    print("\n--- Full two-phase test (model_scatter=True) ---")
    r2 = test_single_segment(
        filepath      = filepath,
        order         = 50,
        seg_index     = 8,
        model_scatter = True,
        numiter       = 5,
        maxiter       = 300,
    )

    sys.exit(0 if r2.get('all_passed', False) else 1)