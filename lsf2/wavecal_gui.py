#!/usr/bin/env python
"""
GUI comparing an lsf2 dispersion solution (an lsf2.cli_run output FITS
file) against an independent ESO WAVE_MATRIX solution.

Usage:
    python -m harps.lsf2.wavecal_gui [LSF2_FITS] [WAVE_MATRIX_FITS]

Requires PyQt5, same as harps.lsf2.gui.

Five tabs, one per requirement this tool was built for:
  1. "Per-order accuracy" -- lsf2 vs wave_matrix, and each vs the LFC's
     own theoretical line wavelengths, for one order at a time. Per-line
     points carry position-uncertainty error bars, are split into
     even/odd (by wavelength order) with a sliding-median trend line
     each, and a side histogram shows the median + central-68% range.
     An optional second order can be overlaid directly (e.g. an
     image-slicer slice pair or an overlapping adjacent order).
  2. "Full range accuracy" -- the same plot, pooling every common order
     across the file's entire wavelength coverage.
  3. "Overlapping spectra" -- for a chosen wavelength, every order/slice
     that covers it, with the observed LFC spectrum overplotted under
     each order's own calibration (needs the original S2D flux; optional).
  4. "Summary table" -- RMS/median residual per order, sortable, to spot
     which orders/slices are worst at a glance.
  5. "2D difference" -- lsf2 minus wave_matrix, order x pixel, as an
     image, for detector-wide systematics.
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "harps.lsf2.wavecal_gui needs PyQt5 ('pip install PyQt5'); it is "
        "not a dependency of the rest of harps.lsf2."
    ) from exc

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

from .reconstruct import LSFLibrary
from .wavematrix import WaveMatrix
from .flux_source import FluxSource
from . import wavecal_compare as wc


class MplCanvas(FigureCanvas):
    def __init__(self, nrows=1, ncols=1, figsize=(8, 6), **subplot_kw):
        self.fig = Figure(figsize=figsize)
        self.axes = self.fig.subplots(nrows, ncols, **subplot_kw)
        super().__init__(self.fig)


class AccuracyCanvas(FigureCanvas):
    """ The shared 2x2 layout for every accuracy plot in this GUI: a main
        panel plus a horizontal-orientation histogram to its right, for
        each of the two rows (top: continuous lsf2-wave_matrix difference;
        bottom: each solution's own per-line accuracy vs. theoretical). """

    def __init__(self, figsize=(10, 8)):
        self.fig = Figure(figsize=figsize, layout='constrained')
        gs = self.fig.add_gridspec(2, 2, width_ratios=[5, 1.3], height_ratios=[2, 1.4])
        self.ax_top = self.fig.add_subplot(gs[0, 0])
        self.ax_top_hist = self.fig.add_subplot(gs[0, 1], sharey=self.ax_top)
        self.ax_bot = self.fig.add_subplot(gs[1, 0], sharex=self.ax_top)
        self.ax_bot_hist = self.fig.add_subplot(gs[1, 1], sharey=self.ax_bot)
        super().__init__(self.fig)


def _convert_residual(unit: str, ms_values, wavelength_nm, v_pix_kms=None):
    """ ms_values -> the chosen display unit, using each POINT's own local
        scale (v_pix_kms), not a single global conversion factor -- matters
        once points from several orders (with different local dispersion)
        are pooled on one plot. """
    if unit == 'm/s':
        return ms_values
    if unit == 'nm':
        return (np.asarray(ms_values) / wc.C_MS) * np.asarray(wavelength_nm)
    return (np.asarray(ms_values) / 1e3) / np.asarray(v_pix_kms)   # pixels


def _draw_side_histogram(ax_hist, series: list, orientation_axis: str = 'y'):
    """ series: [(values, color, label, show_band), ...]. One horizontal
        histogram per entry (semi-transparent so they can overlap), each
        with its own median line; show_band additionally shades its
        central-68% interval. """
    ax_hist.clear()
    for values, color, label, show_band in series:
        median, p16, p84 = wc.pooled_percentiles(values)
        if not np.isfinite(median):
            continue
        finite = np.asarray(values)[np.isfinite(values)]
        ax_hist.hist(finite, bins=25, orientation='horizontal', color=color, alpha=0.45,
                     label=f"{label}\nmed={median:.3g}, 68%=[{p16:.3g},{p84:.3g}]")
        ax_hist.axhline(median, color=color, lw=1.6)
        if show_band:
            ax_hist.axhspan(p16, p84, color=color, alpha=0.12)
        else:
            ax_hist.axhline(p16, color=color, lw=0.8, ls=':')
            ax_hist.axhline(p84, color=color, lw=0.8, ls=':')
    ax_hist.tick_params(axis='y', labelleft=False)
    ax_hist.set_xlabel('count')
    ax_hist.legend(fontsize=6, loc='upper right')


# Colour scheme: consistent across single-order, two-order-compare, and
# full-range views. 'primary'/'secondary' distinguish orders when
# req. 3's "compare with order" is used; both collapse to the same primary
# colours when there's only one order (or the full-range pool, where every
# order plays the same role).
_LSF2_COLOR_PRIMARY, _WM_COLOR_PRIMARY = 'tab:blue', 'tab:red'
_LSF2_COLOR_SECONDARY, _WM_COLOR_SECONDARY = 'tab:cyan', 'tab:orange'
_CONTINUOUS_COLOR_PRIMARY, _CONTINUOUS_COLOR_SECONDARY = 'tab:purple', 'tab:brown'


def draw_accuracy_plot(canvas: AccuracyCanvas, comparisons: list, unit: str = 'm/s',
                        labels: list = None, show_even_odd: bool = True, show_trend: bool = True,
                        show_errorbars: bool = True, point_alpha: float = 0.8, point_size: float = 14):
    """ Draws both rows (continuous difference + per-line-vs-theoretical)
        for one or more wc.OrderComparison objects on a shared wavelength
        x-axis, with position-uncertainty error bars and side histograms
        (median + central 68%, req. 1). Pass 2 comparisons to overlay two
        orders directly (req. 3, colours distinguish them); pass many to
        pool an entire wavelength range (req. 2, all treated identically).
        show_even_odd (req. 4) splits every per-line series into even/odd
        (by wavelength order) via marker shape, with an optional
        sliding-median trend line per subset. """
    ax_top, ax_top_hist, ax_bot, ax_bot_hist = canvas.ax_top, canvas.ax_top_hist, canvas.ax_bot, canvas.ax_bot_hist
    ax_top.clear()
    ax_bot.clear()

    is_compare = len(comparisons) == 2 and labels is not None
    pooled_top_values, pooled_lsf2_values, pooled_wm_values = [], [], []

    for i, cmp in enumerate(comparisons):
        # In "compare" mode both orders get their own legend entry (the
        # label text differs, via label_suffix). In "pooled" mode every
        # comparison would produce the SAME label text, so only the first
        # is labelled -- otherwise the legend repeats once per order.
        use_label = is_compare or (i == 0)

        if is_compare:
            continuous_color = _CONTINUOUS_COLOR_PRIMARY if i == 0 else _CONTINUOUS_COLOR_SECONDARY
            lsf2_color = _LSF2_COLOR_PRIMARY if i == 0 else _LSF2_COLOR_SECONDARY
            wm_color = _WM_COLOR_PRIMARY if i == 0 else _WM_COLOR_SECONDARY
            label_suffix = f" ({labels[i]})"
        else:
            continuous_color, lsf2_color, wm_color = _CONTINUOUS_COLOR_PRIMARY, _LSF2_COLOR_PRIMARY, _WM_COLOR_PRIMARY
            label_suffix = ""

        # --- top row: continuous difference + per-line overlay with error bars ---
        dense_v_pix = cmp.dense_v_pix_kms
        continuous = _convert_residual(unit, cmp.residual_ms, cmp.wavelength_wavemat_nm, dense_v_pix)
        line_v_pix = cmp.line_v_pix_kms
        line_continuous = _convert_residual(unit, cmp.line_continuous_residual_ms, cmp.line_wavelength_nm, line_v_pix)
        line_continuous_err = _convert_residual(unit, cmp.line_residual_err_ms, cmp.line_wavelength_nm, line_v_pix)

        ax_top.plot(cmp.wavelength_wavemat_nm, continuous, color=continuous_color, lw=1.1,
                    alpha=0.9 if is_compare else 1.0,
                    label=f"lsf2 - wave_matrix{label_suffix}" if use_label else None)
        ax_top.errorbar(cmp.line_wavelength_nm, line_continuous,
                         yerr=np.abs(line_continuous_err) if show_errorbars else None,
                         fmt='.', ms=point_size ** 0.5, color=continuous_color, alpha=point_alpha,
                         elinewidth=0.6, capsize=0, zorder=4)
        pooled_top_values.extend(line_continuous)

        # --- bottom row: per-line vs theoretical, split even/odd ---
        line_lsf2 = _convert_residual(unit, cmp.line_residual_lsf2_ms, cmp.line_wavelength_nm, line_v_pix)
        line_wm = _convert_residual(unit, cmp.line_residual_wavemat_ms, cmp.line_wavelength_nm, line_v_pix)
        line_err = _convert_residual(unit, cmp.line_residual_err_ms, cmp.line_wavelength_nm, line_v_pix)
        pooled_lsf2_values.extend(line_lsf2)
        pooled_wm_values.extend(line_wm)

        even = cmp.line_even_mask if show_even_odd else np.ones(len(cmp.line_wavelength_nm), dtype=bool)
        odd = ~even if show_even_odd else np.zeros(len(cmp.line_wavelength_nm), dtype=bool)

        for series_y, color, series_label in [(line_lsf2, lsf2_color, f"lsf2 vs theoretical{label_suffix}"),
                                                (line_wm, wm_color, f"wave_matrix vs theoretical{label_suffix}")]:
            even_label = f"{series_label}, even" if show_even_odd else series_label
            ax_bot.errorbar(cmp.line_wavelength_nm[even], series_y[even],
                             yerr=np.abs(line_err[even]) if show_errorbars else None,
                             fmt='o', ms=point_size ** 0.5, color=color, alpha=point_alpha,
                             elinewidth=0.6, capsize=0, label=even_label if use_label else None,
                             zorder=4)
            if show_even_odd and odd.any():
                ax_bot.errorbar(cmp.line_wavelength_nm[odd], series_y[odd],
                                 yerr=np.abs(line_err[odd]) if show_errorbars else None,
                                 fmt='^', ms=point_size ** 0.5, color=color, alpha=point_alpha,
                                 elinewidth=0.6, capsize=0, label=f"{series_label}, odd" if use_label else None,
                                 zorder=4)
            if show_trend:
                if show_even_odd:
                    if even.any():
                        tx, ty = wc.moving_median_trend(cmp.line_wavelength_nm[even], series_y[even])
                        ax_bot.plot(tx, ty, '-', color=color, lw=1.3, alpha=0.9, zorder=5)
                    if odd.any():
                        tx, ty = wc.moving_median_trend(cmp.line_wavelength_nm[odd], series_y[odd])
                        ax_bot.plot(tx, ty, '--', color=color, lw=1.3, alpha=0.9, zorder=5)
                else:
                    tx, ty = wc.moving_median_trend(cmp.line_wavelength_nm, series_y)
                    ax_bot.plot(tx, ty, '-', color=color, lw=1.3, alpha=0.9, zorder=5)

    unit_label = {'m/s': 'residual [m/s]', 'nm': 'residual [nm]', 'pixels': 'residual [pixels]'}[unit]
    ax_top.axhline(0, color='gray', lw=0.5)
    finite = np.asarray(pooled_top_values)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo, hi = np.nanpercentile(finite, [1, 99])
        pad = 0.25 * max(hi - lo, 1e-9)
        ax_top.set_ylim(lo - pad, hi + pad)
    ax_top.legend(fontsize=7)
    ax_top.set_ylabel(unit_label)
    ax_top.set_title("lsf2 vs wave_matrix" + (f": orders {labels}" if is_compare else ""))

    ax_bot.axhline(0, color='gray', lw=0.5)
    ax_bot.legend(fontsize=6, ncol=2)
    ax_bot.set_xlabel('wavelength [nm]')
    ax_bot.set_ylabel(unit_label)
    ax_bot.set_title('Per-line residuals vs. theoretical LFC wavelength'
                      + ('  (o = even, ^ = odd line index)' if show_even_odd else ''))

    _draw_side_histogram(ax_top_hist, [(pooled_top_values, _CONTINUOUS_COLOR_PRIMARY, 'lsf2-wave_matrix', True)])
    _draw_side_histogram(ax_bot_hist, [
        (pooled_lsf2_values, _LSF2_COLOR_PRIMARY, 'lsf2', True),
        (pooled_wm_values, _WM_COLOR_PRIMARY, 'wave_matrix', False),
    ])

    canvas.draw_idle()


# =============================================================================
# Tab 1: per-order accuracy
# =============================================================================
class PerOrderTab(QtWidgets.QWidget):
    def __init__(self, lib: LSFLibrary, wavemat: WaveMatrix, parent=None):
        super().__init__(parent)
        self.lib, self.wavemat = lib, wavemat
        self.orders = wc.common_orders(lib, wavemat)

        self.order_combo = QtWidgets.QComboBox()
        for order in self.orders:
            lo, hi = wavemat.wavelength_range_nm(order)
            qc = wavemat.qc_flag.get(order)
            qc_str = '' if qc is None else (' [QC ok]' if qc else ' [QC FAIL]')
            self.order_combo.addItem(f"{order}  ({lo:.2f}-{hi:.2f} nm){qc_str}", userData=order)

        self.compare_combo = QtWidgets.QComboBox()
        self._refresh_compare_options()

        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["m/s", "pixels", "nm"])

        self.even_odd_check = QtWidgets.QCheckBox("Split even/odd lines")
        self.even_odd_check.setChecked(True)
        self.trend_check = QtWidgets.QCheckBox("Show trend lines")
        self.trend_check.setChecked(True)
        self.errorbar_check = QtWidgets.QCheckBox("Show error bars")
        self.errorbar_check.setChecked(True)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Order:"))
        top.addWidget(self.order_combo, 1)
        top.addWidget(QtWidgets.QLabel("Compare with:"))
        top.addWidget(self.compare_combo, 1)
        top.addWidget(QtWidgets.QLabel("Units:"))
        top.addWidget(self.unit_combo)
        top.addWidget(self.even_odd_check)
        top.addWidget(self.trend_check)
        top.addWidget(self.errorbar_check)

        self.canvas = AccuracyCanvas(figsize=(11, 9))
        toolbar = NavigationToolbar(self.canvas, self)

        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setStyleSheet("font-family: monospace;")
        self.stats_label.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.stats_label)

        self.order_combo.currentIndexChanged.connect(self._on_order_changed)
        self.compare_combo.currentIndexChanged.connect(self.redraw)
        self.unit_combo.currentIndexChanged.connect(self.redraw)
        self.even_odd_check.stateChanged.connect(self.redraw)
        self.trend_check.stateChanged.connect(self.redraw)
        self.errorbar_check.stateChanged.connect(self.redraw)
        if self.orders:
            self._on_order_changed()

    def _refresh_compare_options(self):
        self.compare_combo.blockSignals(True)
        self.compare_combo.clear()
        self.compare_combo.addItem("(none)", userData=None)
        if self.orders:
            current_order = self.order_combo.currentData() if self.order_combo.count() else self.orders[0]
            suggested = wc.suggest_partner_order(self.lib, self.wavemat, current_order)
            for order in self.orders:
                if order == current_order:
                    continue
                label = str(order) + ("  [suggested slice/overlap partner]" if order == suggested else "")
                self.compare_combo.addItem(label, userData=order)
            if suggested is not None:
                idx = self.compare_combo.findData(suggested)
                if idx >= 0:
                    self.compare_combo.setCurrentIndex(idx)
        self.compare_combo.blockSignals(False)

    def _on_order_changed(self):
        self._refresh_compare_options()
        self.redraw()

    def redraw(self):
        if not self.orders:
            return
        order = self.order_combo.currentData()
        compare_order = self.compare_combo.currentData()
        unit = self.unit_combo.currentText()
        show_even_odd = self.even_odd_check.isChecked()
        show_trend = self.trend_check.isChecked()

        cmp = wc.compare_order(self.lib, self.wavemat, order)
        comparisons, labels = [cmp], None
        if compare_order is not None:
            cmp2 = wc.compare_order(self.lib, self.wavemat, compare_order)
            comparisons = [cmp, cmp2]
            labels = [str(order), str(compare_order)]

        draw_accuracy_plot(self.canvas, comparisons, unit=unit, labels=labels,
                            show_even_odd=show_even_odd, show_trend=show_trend,
                            show_errorbars=self.errorbar_check.isChecked())
        self.canvas.ax_top.set_title(f"Order {order}" + (f" vs order {compare_order}" if labels else "")
                                      + ": lsf2 vs wave_matrix")

        def line_stats(c):
            return (np.sqrt(np.nanmean(c.line_residual_lsf2_ms ** 2)),
                    np.sqrt(np.nanmean(c.line_residual_wavemat_ms ** 2)))

        lsf2_rms, wm_rms = line_stats(cmp)
        qc = self.wavemat.qc_flag.get(order)
        text = (f"Order {order}: RMS(lsf2-wave_matrix)={cmp.rms_ms:.3f} m/s, median={cmp.median_ms:+.3f} m/s, "
                f"lsf2 line RMS={lsf2_rms:.3f} m/s, wave_matrix line RMS={wm_rms:.3f} m/s, "
                f"n_lines={len(cmp.line_wavelength_nm)}, ESO QC={'--' if qc is None else ('OK' if qc else 'FAIL')}")
        if compare_order is not None:
            lsf2_rms2, wm_rms2 = line_stats(cmp2)
            qc2 = self.wavemat.qc_flag.get(compare_order)
            text += (f"\nOrder {compare_order}: RMS(lsf2-wave_matrix)={cmp2.rms_ms:.3f} m/s, "
                     f"median={cmp2.median_ms:+.3f} m/s, lsf2 line RMS={lsf2_rms2:.3f} m/s, "
                     f"wave_matrix line RMS={wm_rms2:.3f} m/s, n_lines={len(cmp2.line_wavelength_nm)}, "
                     f"ESO QC={'--' if qc2 is None else ('OK' if qc2 else 'FAIL')}")
        self.stats_label.setText(text)


# =============================================================================
# Tab 1b: accuracy across the entire wavelength range covered by the file
# =============================================================================
class FullRangeTab(QtWidgets.QWidget):
    def __init__(self, lib: LSFLibrary, wavemat: WaveMatrix, parent=None):
        super().__init__(parent)
        self.lib, self.wavemat = lib, wavemat
        self.orders = wc.common_orders(lib, wavemat)

        self.qc_only_check = QtWidgets.QCheckBox("ESO-QC-passing orders only")
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["m/s", "pixels", "nm"])
        self.even_odd_check = QtWidgets.QCheckBox("Split even/odd lines")
        self.even_odd_check.setChecked(True)
        self.trend_check = QtWidgets.QCheckBox("Show trend lines")
        self.trend_check.setChecked(True)
        self.errorbar_check = QtWidgets.QCheckBox("Show error bars")
        self.errorbar_check.setChecked(True)
        self.recompute_button = QtWidgets.QPushButton("Recompute")

        lo_all = min(wavemat.wavelength_range_nm(o)[0] for o in self.orders) if self.orders else 0
        hi_all = max(wavemat.wavelength_range_nm(o)[1] for o in self.orders) if self.orders else 0

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel(f"{len(self.orders)} common order(s), {lo_all:.1f}-{hi_all:.1f} nm"))
        top.addWidget(self.qc_only_check)
        top.addWidget(QtWidgets.QLabel("Units:"))
        top.addWidget(self.unit_combo)
        top.addWidget(self.even_odd_check)
        top.addWidget(self.trend_check)
        top.addWidget(self.errorbar_check)
        top.addWidget(self.recompute_button)
        top.addStretch(1)

        self.canvas = AccuracyCanvas(figsize=(12, 9))
        toolbar = NavigationToolbar(self.canvas, self)
        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setStyleSheet("font-family: monospace;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.stats_label)

        self.recompute_button.clicked.connect(self.redraw)
        self.unit_combo.currentIndexChanged.connect(self.redraw)
        self.even_odd_check.stateChanged.connect(self.redraw)
        self.trend_check.stateChanged.connect(self.redraw)
        self.errorbar_check.stateChanged.connect(self.redraw)
        self.qc_only_check.stateChanged.connect(self.redraw)
        if self.orders:
            self.redraw()

    def redraw(self):
        if not self.orders:
            return
        orders = self.orders
        if self.qc_only_check.isChecked():
            orders = [o for o in orders if self.wavemat.qc_flag.get(o, True)]
        if not orders:
            self.canvas.ax_top.clear()
            self.canvas.ax_top.set_title("No orders left after filtering")
            self.canvas.draw_idle()
            return

        # Fewer dense samples per order here than the single-order view --
        # this can be up to ~170 orders at once, and per-order curve detail
        # matters much less when looking at the whole detector together.
        comparisons = [wc.compare_order(self.lib, self.wavemat, o, n_samples=300) for o in orders]

        draw_accuracy_plot(self.canvas, comparisons, unit=self.unit_combo.currentText(),
                            labels=None, show_even_odd=self.even_odd_check.isChecked(),
                            show_trend=self.trend_check.isChecked(), show_errorbars=self.errorbar_check.isChecked(),
                            point_alpha=0.5, point_size=6)
        self.canvas.ax_top.set_title(f"lsf2 vs wave_matrix, {len(orders)} order(s), entire common wavelength range")

        all_rms = np.array([c.rms_ms for c in comparisons])
        pooled_lsf2 = np.concatenate([c.line_residual_lsf2_ms for c in comparisons])
        pooled_wm = np.concatenate([c.line_residual_wavemat_ms for c in comparisons])
        med_l, p16_l, p84_l = wc.pooled_percentiles(pooled_lsf2)
        med_w, p16_w, p84_w = wc.pooled_percentiles(pooled_wm)
        self.stats_label.setText(
            f"{len(orders)} orders pooled  |  per-order RMS(lsf2-wave_matrix): "
            f"mean={all_rms.mean():.3f} m/s, median={np.median(all_rms):.3f} m/s, max={all_rms.max():.3f} m/s  |  "
            f"lsf2 line residual: median={med_l:+.3f}, 68%=[{p16_l:+.3f},{p84_l:+.3f}] m/s  |  "
            f"wave_matrix line residual: median={med_w:+.3f}, 68%=[{p16_w:+.3f},{p84_w:+.3f}] m/s"
        )


# =============================================================================
# Tab 2: overlapping orders / slices -- overplot the LFC spectrum
# =============================================================================
class OverlapTab(QtWidgets.QWidget):
    def __init__(self, lib: LSFLibrary, wavemat: WaveMatrix, flux: FluxSource, parent=None):
        super().__init__(parent)
        self.lib, self.wavemat, self.flux = lib, wavemat, flux
        self.orders = wc.common_orders(lib, wavemat)
        lo_all = min(wavemat.wavelength_range_nm(o)[0] for o in self.orders)
        hi_all = max(wavemat.wavelength_range_nm(o)[1] for o in self.orders)

        self.wavelength_spin = QtWidgets.QDoubleSpinBox()
        self.wavelength_spin.setDecimals(4)
        self.wavelength_spin.setRange(lo_all, hi_all)
        self.wavelength_spin.setValue(0.5 * (lo_all + hi_all))

        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setDecimals(3)
        self.window_spin.setRange(0.001, 5.0)
        self.window_spin.setValue(0.05)
        self.window_spin.setSuffix(" nm")

        self.calibration_combo = QtWidgets.QComboBox()
        self.calibration_combo.addItems(["lsf2 calibration", "wave_matrix calibration"])

        self.detect_button = QtWidgets.QPushButton("Find overlapping orders/slices")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Wavelength [nm]:"))
        top.addWidget(self.wavelength_spin)
        top.addWidget(QtWidgets.QLabel("+/- window:"))
        top.addWidget(self.window_spin)
        top.addWidget(QtWidgets.QLabel("Plot x-axis from:"))
        top.addWidget(self.calibration_combo)
        top.addWidget(self.detect_button)
        top.addStretch(1)

        self.info_label = QtWidgets.QLabel()
        self.canvas = MplCanvas(nrows=1, ncols=1, figsize=(9, 6))
        toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.info_label)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)

        self.detect_button.clicked.connect(self.redraw)
        self.calibration_combo.currentIndexChanged.connect(self.redraw)
        self.redraw()

    def redraw(self):
        wavelength = self.wavelength_spin.value()
        window = self.window_spin.value()
        covering = wc.overlapping_orders_at_wavelength(self.lib, self.wavemat, wavelength, self.orders)

        ax = self.canvas.axes
        ax.clear()

        if not covering:
            self.info_label.setText(f"No order/slice covers {wavelength:.4f} nm.")
            self.canvas.draw_idle()
            return

        self.info_label.setText(
            f"{len(covering)} order(s)/slice(s) cover {wavelength:.4f} nm: {covering}"
            + ("" if self.flux is not None else
               "   (load an S2D exposure via File > Load spectrum to see the actual LFC flux)")
        )

        if self.flux is None:
            ax.set_title("No spectrum loaded -- File > Load spectrum (S2D exposure)")
            self.canvas.fig.tight_layout()
            self.canvas.draw_idle()
            return

        use_lsf2 = self.calibration_combo.currentIndex() == 0
        cmap = matplotlib.colormaps.get_cmap('tab10')
        for i, order in enumerate(covering):
            x_min, x_max, n_pixels = self.lib.pixel_range(order)
            pixel = np.arange(min(n_pixels, self.flux.n_pixels))
            if use_lsf2:
                wavelength_axis = self.lib.pixel_to_wavelength(order, pixel)
            else:
                wavelength_axis = self.wavemat.wavelength_nm(order, pixel)
            mask = np.abs(wavelength_axis - wavelength) <= window
            if not mask.any():
                continue
            flux_row = self.flux.order_flux(order)[:len(pixel)]
            ax.plot(wavelength_axis[mask], flux_row[mask], '.-', ms=3, lw=0.8,
                    color=cmap(i % 10), label=f'order {order}')

        ax.axvline(wavelength, color='k', ls=':', lw=1)
        ax.legend(fontsize=8)
        ax.set_xlabel('wavelength [nm]' + f'  ({self.calibration_combo.currentText()})')
        ax.set_ylabel('flux [counts]')
        ax.set_title(f'LFC spectrum, overlapping orders near {wavelength:.4f} nm')

        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()


# =============================================================================
# Tab 3: summary table across all orders
# =============================================================================
class SummaryTab(QtWidgets.QWidget):
    COLUMNS = [
        ('order', 'Order'), ('physical_order', 'Phys. order'), ('slice', 'Slice'),
        ('wave_lo_nm', 'Wave lo [nm]'), ('wave_hi_nm', 'Wave hi [nm]'),
        ('rms_ms', 'RMS(lsf2-WM) [m/s]'), ('median_ms', 'Median(lsf2-WM) [m/s]'),
        ('lsf2_line_rms_ms', 'lsf2 line RMS [m/s]'), ('wavemat_line_rms_ms', 'WM line RMS [m/s]'),
        ('n_lines', 'N lines'), ('qc_flag', 'ESO QC'),
    ]

    def __init__(self, lib: LSFLibrary, wavemat: WaveMatrix, parent=None):
        super().__init__(parent)
        self.lib, self.wavemat = lib, wavemat

        self.refresh_button = QtWidgets.QPushButton("Recompute")
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in self.COLUMNS])
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.table)

        self.refresh_button.clicked.connect(self.refresh)
        self.refresh()

    def refresh(self):
        rows = wc.summary_table(self.lib, self.wavemat)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, (key, _) in enumerate(self.COLUMNS):
                value = row[key]
                if key == 'qc_flag':
                    text = '--' if value is None else ('OK' if value else 'FAIL')
                elif isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                item = QtWidgets.QTableWidgetItem()
                item.setData(QtCore.Qt.DisplayRole, text if isinstance(value, (str, type(None))) or key == 'qc_flag'
                             else (value if not isinstance(value, float) else round(value, 6)))
                if key == 'qc_flag' and value is False:
                    item.setBackground(QtCore.Qt.red)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()


# =============================================================================
# Tab 4: 2D difference across the whole detector
# =============================================================================
class TwoDDiffTab(QtWidgets.QWidget):
    def __init__(self, lib: LSFLibrary, wavemat: WaveMatrix, parent=None):
        super().__init__(parent)
        self.lib, self.wavemat = lib, wavemat

        self.recompute_button = QtWidgets.QPushButton("Recompute")
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        self.vmax_spin.setRange(0, 1e6)
        self.vmax_spin.setValue(50.0)
        self.vmax_spin.setSuffix(" m/s (colour limit, 0 = auto)")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.recompute_button)
        top.addWidget(self.vmax_spin)
        top.addStretch(1)

        self.canvas = MplCanvas(nrows=1, ncols=1, figsize=(10, 7))
        toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)

        self.recompute_button.clicked.connect(self.redraw)
        self.vmax_spin.valueChanged.connect(self.redraw)
        self.redraw()

    def redraw(self):
        pixel_grid, orders, diff = wc.two_d_difference(self.lib, self.wavemat)

        self.canvas.fig.clear()
        ax = self.canvas.fig.subplots(1, 1)

        vmax = self.vmax_spin.value()
        if vmax <= 0:
            finite = diff[np.isfinite(diff)]
            vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0

        im = ax.imshow(diff, aspect='auto', origin='lower', cmap='RdBu_r',
                        extent=[pixel_grid.min(), pixel_grid.max(), -0.5, len(orders) - 0.5],
                        vmin=-vmax, vmax=vmax)
        self.canvas.fig.colorbar(im, ax=ax, label='lsf2 - wave_matrix [m/s]')

        step = max(1, len(orders) // 40)
        ax.set_yticks(range(0, len(orders), step))
        ax.set_yticklabels([str(orders[i]) for i in range(0, len(orders), step)], fontsize=7)
        ax.set_xlabel('pixel')
        ax.set_ylabel('order')
        ax.set_title('lsf2 - wave_matrix, across the whole detector')

        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()


# =============================================================================
# Main window
# =============================================================================
class WavecalDiagnosticsWindow(QtWidgets.QMainWindow):
    def __init__(self, lsf2_path: str = None, wavematrix_path: str = None):
        super().__init__()
        self.setWindowTitle("harps.lsf2 -- wavelength calibration diagnostics")
        self.resize(1250, 850)

        self.lib: LSFLibrary = None
        self.wavemat: WaveMatrix = None
        self.flux: FluxSource = None

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.status = self.statusBar()

        file_menu = self.menuBar().addMenu("&File")
        open_lsf2 = QtWidgets.QAction("Open lsf2 FITS...", self)
        open_lsf2.triggered.connect(self.open_lsf2_dialog)
        open_wavemat = QtWidgets.QAction("Open WAVE_MATRIX FITS...", self)
        open_wavemat.triggered.connect(self.open_wavematrix_dialog)
        open_flux = QtWidgets.QAction("Load spectrum (S2D exposure)...", self)
        open_flux.triggered.connect(self.open_flux_dialog)
        file_menu.addAction(open_lsf2)
        file_menu.addAction(open_wavemat)
        file_menu.addAction(open_flux)

        if lsf2_path:
            self._load_lsf2(lsf2_path)
        if wavematrix_path:
            self._load_wavematrix(wavematrix_path)

    def open_lsf2_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open lsf2 output FITS", "", "FITS files (*.fits *.fit)")
        if path:
            self._load_lsf2(path)

    def open_wavematrix_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open WAVE_MATRIX FITS", "", "FITS files (*.fits *.fit)")
        if path:
            self._load_wavematrix(path)

    def open_flux_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open S2D exposure FITS", "", "FITS files (*.fits *.fit)")
        if path:
            try:
                self.flux = FluxSource.load(path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Failed to load spectrum", str(exc))
                return
            self._rebuild_tabs()

    def _load_lsf2(self, path):
        try:
            self.lib = LSFLibrary(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to open lsf2 FITS", str(exc))
            return
        self._rebuild_tabs()

    def _load_wavematrix(self, path):
        try:
            self.wavemat = WaveMatrix.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to open WAVE_MATRIX FITS", str(exc))
            return
        self._rebuild_tabs()

    def _rebuild_tabs(self):
        self.tabs.clear()
        if self.lib is None or self.wavemat is None:
            self.status.showMessage("Load both an lsf2 FITS file and a WAVE_MATRIX FITS file to begin.")
            return

        n_common = len(wc.common_orders(self.lib, self.wavemat))
        if n_common == 0:
            QtWidgets.QMessageBox.warning(
                self, "No common orders",
                "The lsf2 file and the WAVE_MATRIX file share no order index in common -- "
                "check that both refer to the same instrument/order numbering.")

        self.tabs.addTab(PerOrderTab(self.lib, self.wavemat), "Per-order accuracy")
        self.tabs.addTab(FullRangeTab(self.lib, self.wavemat), "Full range accuracy")
        self.tabs.addTab(OverlapTab(self.lib, self.wavemat, self.flux), "Overlapping spectra")
        self.tabs.addTab(SummaryTab(self.lib, self.wavemat), "Summary table")
        self.tabs.addTab(TwoDDiffTab(self.lib, self.wavemat), "2D difference")
        self.status.showMessage(f"{n_common} common order(s) between lsf2 and wave_matrix.")


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication(argv)
    lsf2_path = argv[1] if len(argv) > 1 else None
    wavematrix_path = argv[2] if len(argv) > 2 else None
    window = WavecalDiagnosticsWindow(lsf2_path, wavematrix_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
