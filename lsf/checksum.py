#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harps/lsf/checksum.py

Numerical regression-testing tool: compares two FITS files (typically a
saved "reference" output from a known-good run, and a freshly-produced
"candidate" output after a code change) extension by extension, field by
field, within a chosen relative/absolute tolerance.

This exists to catch exactly the kind of regression this project has hit
more than once — a refactor that looks correct on inspection but silently
changes numerical output (an optimizer swap, a signature-mismatch bug that
changes which code path actually runs, an accidental double subtraction in
a normalization formula). Run this after any change to harps.lsf and
compare against a reference output captured before the change.

Usage (CLI)
-----------
    python -m harps.lsf.checksum reference.fits candidate.fits
    python -m harps.lsf.checksum reference.fits candidate.fits --rtol 1e-8
    python -m harps.lsf.checksum ref.fits cand.fits --skip model_gauss

Usage (library)
---------------
    from harps.lsf import checksum
    report = checksum.compare_fits_files(ref_path, cand_path, rtol=1e-6)
    if not report.ok:
        print(report.summary())
        raise SystemExit(1)

Design notes
------------
- Integer/bool/string fields are compared for EXACT equality — a tolerance
  isn't meaningful for e.g. 'order' or 'segm', and a mismatch there usually
  means something structural changed (wrong row, wrong indexing), which is
  worth flagging regardless of magnitude.
- Float fields are compared only on elements finite in BOTH files, using
  rtol/atol (numpy's np.isclose convention). The NaN/non-NaN pattern is
  checked separately and reported if it differs — a field that used to be
  NaN and now has a value (or vice versa) is exactly the kind of change
  a pure "compare the finite values" check would silently miss.
- Extensions present in only one file are reported, not silently ignored.
"""
import argparse
import dataclasses
from typing import Optional, Sequence

import numpy as np
from fitsio import FITS


@dataclasses.dataclass
class FieldMismatch:
    extname: str
    extver: object
    field: str
    n_compared: int
    n_mismatched: int
    max_rel_diff: float
    max_abs_diff: float
    nan_pattern_matches: bool
    detail: str = ""


@dataclasses.dataclass
class ComparisonReport:
    ref_path: str
    cand_path: str
    rtol: float
    atol: float
    mismatches: list
    extensions_only_in_ref: list
    extensions_only_in_cand: list
    n_fields_compared: int

    @property
    def ok(self) -> bool:
        return (not self.mismatches
                and not self.extensions_only_in_ref
                and not self.extensions_only_in_cand)

    def summary(self) -> str:
        lines = [
            f"Checksum comparison: {self.ref_path}  vs  {self.cand_path}",
            f"  tolerance: rtol={self.rtol}, atol={self.atol}",
            f"  fields compared: {self.n_fields_compared}",
        ]
        if self.extensions_only_in_ref:
            lines.append(f"  extensions only in reference: {self.extensions_only_in_ref}")
        if self.extensions_only_in_cand:
            lines.append(f"  extensions only in candidate: {self.extensions_only_in_cand}")
        if self.ok:
            lines.append("  RESULT: PASS -- no field exceeded tolerance.")
        else:
            if not self.mismatches:
                lines.append("  RESULT: FAIL -- extension set differs (see above); "
                             "no field-level mismatches otherwise.")
            else:
                lines.append(f"  RESULT: FAIL -- {len(self.mismatches)} field(s) exceeded tolerance:")
            for m in self.mismatches:
                nan_note = "" if m.nan_pattern_matches else "  [NaN pattern differs!]"
                lines.append(
                    f"    [{m.extname},{m.extver}] {m.field}: "
                    f"{m.n_mismatched}/{m.n_compared} elements beyond tolerance "
                    f"(max_rel_diff={m.max_rel_diff:.3g}, max_abs_diff={m.max_abs_diff:.3g})"
                    f"{nan_note}"
                    + (f"  {m.detail}" if m.detail else "")
                )
        return "\n".join(lines)


def _list_extensions(fits_obj: FITS) -> dict:
    """Returns {(extname, extver): hdu_index} for every non-primary HDU."""
    out = {}
    for i in range(1, len(fits_obj)):
        hdr = fits_obj[i].read_header()
        extname = hdr.get('EXTNAME', f'HDU{i}')
        extver = hdr.get('EXTVER', None)
        out[(extname, extver)] = i
    return out


def _compare_array(ref_arr, cand_arr, rtol: float, atol: float):
    """
    Elementwise comparison of two arrays.

    Returns (n_compared, n_mismatched, max_rel_diff, max_abs_diff,
    nan_pattern_matches).
    """
    ref_arr = np.asarray(ref_arr)
    cand_arr = np.asarray(cand_arr)

    if ref_arr.shape != cand_arr.shape:
        return (int(ref_arr.size), int(ref_arr.size), np.inf, np.inf, False)

    if not np.issubdtype(ref_arr.dtype, np.floating):
        # Exact comparison for int / bool / string / other non-float dtypes —
        # a tolerance isn't meaningful for these.
        try:
            eq = (ref_arr == cand_arr)
            n_mismatched = int(np.size(eq) - np.count_nonzero(eq))
        except (TypeError, ValueError):
            n_mismatched = int(ref_arr.tolist() != cand_arr.tolist())
        worst = 0.0 if n_mismatched == 0 else np.inf
        return (int(ref_arr.size), n_mismatched, worst, worst, True)

    ref_nan = np.isnan(ref_arr)
    cand_nan = np.isnan(cand_arr)
    nan_pattern_matches = bool(np.array_equal(ref_nan, cand_nan))

    finite = ~ref_nan & ~cand_nan
    n_compared = int(np.sum(finite))
    if n_compared == 0:
        return (0, 0, 0.0, 0.0, nan_pattern_matches)

    r = ref_arr[finite]
    c = cand_arr[finite]
    close = np.isclose(r, c, rtol=rtol, atol=atol)
    n_mismatched = int(np.sum(~close))
    abs_diff = np.abs(r.astype(float) - c.astype(float))
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(r != 0, abs_diff / np.abs(r), abs_diff)
    max_rel_diff = float(np.max(rel_diff))
    max_abs_diff = float(np.max(abs_diff))
    return (n_compared, n_mismatched, max_rel_diff, max_abs_diff, nan_pattern_matches)


def compare_fits_files(ref_path: str, cand_path: str, rtol: float = 1e-6,
                       atol: float = 1e-12,
                       skip_extensions: Optional[Sequence[str]] = None
                       ) -> ComparisonReport:
    """
    Compares every extension/field common to both files.

    Parameters
    ----------
    skip_extensions : extnames to ignore entirely (e.g. large diagnostic
        images not part of the numerical result you care about).
    """
    skip_extensions = set(skip_extensions or [])
    mismatches = []
    n_fields_compared = 0

    with FITS(ref_path, 'r') as fref, FITS(cand_path, 'r') as fcand:
        ref_exts = _list_extensions(fref)
        cand_exts = _list_extensions(fcand)
        common = sorted(set(ref_exts) & set(cand_exts), key=lambda t: (str(t[0]), str(t[1])))
        only_ref = sorted(set(ref_exts) - set(cand_exts), key=lambda t: (str(t[0]), str(t[1])))
        only_cand = sorted(set(cand_exts) - set(ref_exts), key=lambda t: (str(t[0]), str(t[1])))

        for (extname, extver) in common:
            if extname in skip_extensions:
                continue
            ref_hdu = fref[ref_exts[(extname, extver)]]
            cand_hdu = fcand[cand_exts[(extname, extver)]]

            if ref_hdu.get_exttype() == 'BINARY_TBL':
                ref_data = ref_hdu.read()
                cand_data = cand_hdu.read()
                names = ref_data.dtype.names or ()
                cand_names = set(cand_data.dtype.names or ())
                for name in names:
                    if name not in cand_names:
                        continue
                    n_fields_compared += 1
                    n_cmp, n_mis, max_rel, max_abs, nan_ok = _compare_array(
                        ref_data[name], cand_data[name], rtol, atol
                    )
                    if n_mis > 0 or not nan_ok:
                        mismatches.append(FieldMismatch(
                            extname, extver, name, n_cmp, n_mis, max_rel, max_abs, nan_ok
                        ))
            else:
                # IMAGE_HDU: one field, named after the extension itself.
                ref_data = ref_hdu.read()
                cand_data = cand_hdu.read()
                n_fields_compared += 1
                n_cmp, n_mis, max_rel, max_abs, nan_ok = _compare_array(
                    ref_data, cand_data, rtol, atol
                )
                if n_mis > 0 or not nan_ok:
                    mismatches.append(FieldMismatch(
                        extname, extver, '<image>', n_cmp, n_mis, max_rel, max_abs, nan_ok
                    ))

    return ComparisonReport(
        ref_path=ref_path, cand_path=cand_path, rtol=rtol, atol=atol,
        mismatches=mismatches,
        extensions_only_in_ref=only_ref, extensions_only_in_cand=only_cand,
        n_fields_compared=n_fields_compared,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare two FITS files field-by-field within a tolerance, "
                    "to catch numerical regressions across code changes."
    )
    parser.add_argument('reference', help='Path to the reference (known-good) FITS file')
    parser.add_argument('candidate', help='Path to the candidate (newly-produced) FITS file')
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--atol', type=float, default=1e-12)
    parser.add_argument('--skip', nargs='*', default=None,
                        help='Extension names to skip entirely')
    args = parser.parse_args()

    report = compare_fits_files(args.reference, args.candidate,
                                rtol=args.rtol, atol=args.atol,
                                skip_extensions=args.skip)
    print(report.summary())
    raise SystemExit(0 if report.ok else 1)


if __name__ == '__main__':
    main()
