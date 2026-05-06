#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:40:31 2026

@author: dmilakov

harps/lsf/batch.py

Converts between the native per-segment representation (variable-length
numpy arrays) and padded JAX arrays suitable for vmap'd GPU computation.

Padding strategy: all segments are padded to MAX_SEG_LEN with:
  - x, flx, err  → 0.0 fill
  - err           → 1e9 for padded positions (effectively infinite noise
                    so padded points contribute nothing to the GP likelihood)
  - mask          → 0 for padded, 1 for real data points
"""

import numpy as np
import jax.numpy as jnp
from typing import NamedTuple

MAX_SEG_LEN = 600   # maximum pixels per segment — adjust to your data


class SegmentBatch(NamedTuple):
    """
    Padded, stacked arrays ready for jax.vmap.
    All arrays have shape (N_segments, MAX_SEG_LEN).
    meta has length N_segments.
    """
    x    : jnp.ndarray   # pixel/wavelength coordinates
    flx  : jnp.ndarray   # flux
    err  : jnp.ndarray   # flux uncertainty (1e9 for padded)
    mask : jnp.ndarray   # 1.0 = real data, 0.0 = padded
    meta : list          # [(order, pixl, pixr), ...] — one tuple per segment


def _pad_one(x: np.ndarray,
             flx: np.ndarray,
             err: np.ndarray,
             max_len: int = MAX_SEG_LEN
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad a single segment to max_len."""
    n   = len(x)
    pad = max_len - n
    if pad < 0:
        raise ValueError(
            f"Segment length {n} exceeds MAX_SEG_LEN={max_len}. "
            f"Increase MAX_SEG_LEN in batch.py."
        )
    x_p   = np.pad(x,   (0, pad), constant_values=0.0)
    flx_p = np.pad(flx, (0, pad), constant_values=0.0)
    err_p = np.pad(err, (0, pad), constant_values=1e9)
    mask  = np.array([1.0] * n + [0.0] * pad, dtype=np.float32)
    return x_p, flx_p, err_p, mask


def make_batch(x2d      : np.ndarray,
               flx2d    : np.ndarray,
               err2d    : np.ndarray,
               seglims  : dict,
               orders   : list[int],
               max_len  : int = MAX_SEG_LEN
               ) -> SegmentBatch:
    """
    Collect all valid segments from a 2D spectrum into a SegmentBatch.

    Parameters
    ----------
    x2d, flx2d, err2d : np.ndarray, shape (n_orders, n_pixels)
    seglims : list of (pixl, pixr) tuples
    orders  : list of order indices to process
    max_len : padding target

    Returns
    -------
    SegmentBatch with N_valid_segments entries
    """
    X_list, F_list, E_list, M_list, meta = [], [], [], [], []

    for od in orders:
        for (pixl, pixr) in zip(seglims[:-1], seglims[1:]):
            x1s   = np.ravel(x2d  [od, pixl:pixr])
            flx1s = np.ravel(flx2d[od, pixl:pixr])
            err1s = np.ravel(err2d [od, pixl:pixr])

            # Skip empty or all-NaN segments
            valid = np.isfinite(flx1s) & (flx1s != 0.0) & np.isfinite(err1s)
            if valid.sum() < 10:
                continue

            x_p, f_p, e_p, mask = _pad_one(x1s, flx1s, err1s, max_len)
            X_list.append(x_p)
            F_list.append(f_p)
            E_list.append(e_p)
            M_list.append(mask)
            meta.append((od, pixl, pixr))

    return SegmentBatch(
        x    = jnp.array(np.stack(X_list)),
        flx  = jnp.array(np.stack(F_list)),
        err  = jnp.array(np.stack(E_list)),
        mask = jnp.array(np.stack(M_list)),
        meta = meta,
    )


def split_batch(batch: SegmentBatch, n_chunks: int) -> list[SegmentBatch]:
    """
    Split a SegmentBatch into n_chunks roughly equal sub-batches.
    Used to distribute segments across Ray workers (one per GPU).
    """
    N    = len(batch.meta)
    size = (N + n_chunks - 1) // n_chunks   # ceiling division
    chunks = []
    for i in range(n_chunks):
        sl = slice(i * size, min((i + 1) * size, N))
        chunks.append(SegmentBatch(
            x    = batch.x   [sl],
            flx  = batch.flx [sl],
            err  = batch.err [sl],
            mask = batch.mask[sl],
            meta = batch.meta[sl.start:sl.stop],
        ))
    return [c for c in chunks if len(c.meta) > 0]


def unbatch_results(results_list: list[list[dict | None]],
                    meta_list   : list[list[tuple]]
                    ) -> dict[tuple, dict]:
    """
    Reassemble per-segment results from Ray workers into a single dict
    keyed by (order, pixl, pixr).
    """
    combined = {}
    for results, meta in zip(results_list, meta_list):
        for result, (od, pixl, pixr) in zip(results, meta):
            combined[(od, pixl, pixr)] = result
    return combined