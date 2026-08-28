#!/usr/bin/env python
"""
Interactive GUI for exploring an lsf2 output FITS file (the one written by
`harps.lsf2.cli_run`, read by `harps.lsf2.cli_reconstruct` / LSFLibrary).

Three tabs:
  1. "LSF across order"     -- slide through pixel/wavelength position
                                within one order and watch phi(u) change;
                                an optional overlay shows several
                                positions at once, colour-coded.
  2. "Departure from Gaussian" -- same idea, but for phi(u) - G(u;sigma(x))
                                specifically, alongside the raw 2D
                                inducing-point grid D(u,x) the departure
                                is interpolated from.
  3. "Composite LSF"         -- pick a wavelength, see every order that
                                covers it (auto-detected or hand-picked),
                                set a per-order weight, and see both the
                                composite and each order's own weighted
                                contribution.

Usage:
    python -m harps.lsf2.gui [FITS_FILE]

Requires PyQt5 (`pip install PyQt5`) in addition to lsf2's normal
dependencies -- deliberately not a hard dependency of the rest of the
package, so `import harps.lsf2` for production runs never needs it.
"""
from __future__ import annotations

import sys

import numpy as np

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "harps.lsf2.gui needs PyQt5 ('pip install PyQt5'); it is not a "
        "dependency of the rest of harps.lsf2."
    ) from exc

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

from .reconstruct import LSFLibrary
from . import weighted_lsf as wlsf

FWHM_FACTOR = 2.354820045


# =============================================================================
# small reusable widgets
# =============================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, nrows=1, ncols=1, figsize=(7, 5), **subplot_kw):
        self.fig = Figure(figsize=figsize)
        self.axes = self.fig.subplots(nrows, ncols, **subplot_kw)
        super().__init__(self.fig)


def _order_combo(lib: LSFLibrary) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    for order in lib.orders():
        lo, hi = lib.wavelength_range(order)
        combo.addItem(f"{order}   ({lo:.2f}-{hi:.2f} nm)", userData=order)
    return combo


def _combo_order(combo: QtWidgets.QComboBox) -> int:
    return combo.currentData()


class _SaveLSFDialog(QtWidgets.QDialog):
    """ Small dialog for the HDU/extension name + overwrite choice used
        whenever a computed LSF is saved to FITS. """

    def __init__(self, default_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save LSF to FITS")
        self.name_edit = QtWidgets.QLineEdit(default_name)
        self.overwrite_check = QtWidgets.QCheckBox(
            "Overwrite if an extension with this name already exists\n"
            "(unchecked: a unique name like NAME_2 is used instead)")

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("HDU / extension name:"))
        layout.addWidget(self.name_edit)
        layout.addWidget(self.overwrite_check)
        layout.addWidget(buttons)

    def name(self) -> str:
        return self.name_edit.text().strip() or "WEIGHTED_LSF"

    def overwrite(self) -> bool:
        return self.overwrite_check.isChecked()


def save_weighted_lsf_interactive(parent: QtWidgets.QWidget, result: "wlsf.WeightedLSFResult",
                                   default_name: str = "WEIGHTED_LSF"):
    """ File-save dialog + HDU-name dialog + wlsf.save_weighted_lsf_fits,
        with success/error message boxes -- shared by every tab that can
        produce a WeightedLSFResult (both sub-tabs of CompositeTab). """
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, "Save LSF to FITS (new file, or an existing one to append to)",
        "", "FITS files (*.fits *.fit);;All files (*)")
    if not path:
        return
    if not path.lower().endswith(('.fits', '.fit')):
        path += '.fits'

    dialog = _SaveLSFDialog(default_name, parent)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return

    try:
        saved_path, hdu_name = wlsf.save_weighted_lsf_fits(
            result, path, hdu_name=dialog.name(), overwrite_hdu=dialog.overwrite())
    except Exception as exc:
        QtWidgets.QMessageBox.critical(parent, "Failed to save", str(exc))
        return
    QtWidgets.QMessageBox.information(
        parent, "Saved",
        f"Saved to {saved_path}\nextension '{hdu_name}' (+ '{hdu_name}_SEGMENTS' for provenance).")


# =============================================================================
# Tab 1: LSF across the order
# =============================================================================
class LsfAcrossOrderTab(QtWidgets.QWidget):
    def __init__(self, lib: LSFLibrary, parent=None):
        super().__init__(parent)
        self.lib = lib

        self.order_combo = _order_combo(lib)
        self.pixel_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pixel_slider.setMinimum(0)
        self.pixel_slider.setMaximum(999)
        self.pixel_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_spin.setDecimals(2)
        self.overlay_check = QtWidgets.QCheckBox("Overlay N positions across order")
        self.overlay_n_spin = QtWidgets.QSpinBox()
        self.overlay_n_spin.setRange(2, 40)
        self.overlay_n_spin.setValue(12)
        self.gaussian_check = QtWidgets.QCheckBox("Show Gaussian-core reference")
        self.gaussian_check.setChecked(True)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Order:"))
        top.addWidget(self.order_combo, 1)
        top.addWidget(QtWidgets.QLabel("Pixel:"))
        top.addWidget(self.pixel_spin)
        top.addWidget(self.gaussian_check)

        overlay_row = QtWidgets.QHBoxLayout()
        overlay_row.addWidget(self.overlay_check)
        overlay_row.addWidget(self.overlay_n_spin)
        overlay_row.addStretch(1)

        self.canvas = MplCanvas(nrows=2, ncols=1, figsize=(8, 7),
                                 gridspec_kw={'height_ratios': [3, 1]})
        toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.pixel_slider)
        layout.addLayout(overlay_row)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)

        self.order_combo.currentIndexChanged.connect(self._on_order_changed)
        self.pixel_slider.valueChanged.connect(self._on_slider_changed)
        self.pixel_spin.valueChanged.connect(self._on_spin_changed)
        self.overlay_check.stateChanged.connect(self.redraw)
        self.overlay_n_spin.valueChanged.connect(self.redraw)
        self.gaussian_check.stateChanged.connect(self.redraw)

        self._syncing = False
        self._on_order_changed()

    def _on_order_changed(self):
        order = _combo_order(self.order_combo)
        x_min, x_max, _ = self.lib.pixel_range(order)
        self.pixel_spin.setRange(x_min, x_max)
        self.pixel_spin.setValue(0.5 * (x_min + x_max))
        self.redraw()

    def _on_slider_changed(self, value):
        if self._syncing:
            return
        order = _combo_order(self.order_combo)
        x_min, x_max, _ = self.lib.pixel_range(order)
        frac = value / self.pixel_slider.maximum()
        self._syncing = True
        self.pixel_spin.setValue(x_min + frac * (x_max - x_min))
        self._syncing = False
        self.redraw()

    def _on_spin_changed(self, value):
        if self._syncing:
            return
        order = _combo_order(self.order_combo)
        x_min, x_max, _ = self.lib.pixel_range(order)
        frac = 0.0 if x_max == x_min else (value - x_min) / (x_max - x_min)
        self._syncing = True
        self.pixel_slider.setValue(int(round(frac * self.pixel_slider.maximum())))
        self._syncing = False
        self.redraw()

    def redraw(self):
        order = _combo_order(self.order_combo)
        x_min, x_max, _ = self.lib.pixel_range(order)
        x = self.pixel_spin.value()

        ax_lsf, ax_fwhm = self.canvas.axes
        ax_lsf.clear()
        ax_fwhm.clear()

        if self.overlay_check.isChecked():
            n = self.overlay_n_spin.value()
            positions = np.linspace(x_min, x_max, n)
            cmap = matplotlib.colormaps.get_cmap('viridis')
            for i, xp in enumerate(positions):
                u, phi = self.lib.lsf_at_pixel(order, xp)
                ax_lsf.plot(u, phi, color=cmap(i / max(n - 1, 1)), lw=1, alpha=0.85,
                            label=f'{xp:.0f}' if n <= 16 else None)
            if n <= 16:
                ax_lsf.legend(fontsize=6, ncol=2, title='pixel')
            # still highlight the current slider position
            u, phi = self.lib.lsf_at_pixel(order, x)
            ax_lsf.plot(u, phi, color='red', lw=2, label='current', zorder=5)
        else:
            u, phi = self.lib.lsf_at_pixel(order, x)
            ax_lsf.plot(u, phi, color='tab:blue', lw=1.8, label=f'pixel {x:.1f}')
            if self.gaussian_check.isChecked():
                sigma = self.lib.sigma_at_pixel(order, x)
                gaussian = np.exp(-0.5 * (u / sigma) ** 2)
                ax_lsf.plot(u, gaussian, 'k--', lw=1, alpha=0.6, label='Gaussian core')
            ax_lsf.legend(fontsize=8)

        ax_lsf.axhline(0, color='gray', lw=0.5)
        ax_lsf.set_xlabel('u [km/s]')
        ax_lsf.set_ylabel('phi(u)')
        ax_lsf.set_title(f'Order {order}: LSF at pixel {x:.1f}')

        pixels = np.linspace(x_min, x_max, 200)
        fwhm = FWHM_FACTOR * self.lib.sigma_at_pixel(order, pixels)
        ax_fwhm.plot(pixels, fwhm, color='tab:orange', lw=1.3)
        ax_fwhm.axvline(x, color='red', lw=1.2, ls='--')
        ax_fwhm.set_xlabel('pixel')
        ax_fwhm.set_ylabel('FWHM [km/s]')
        ax_fwhm.set_title('FWHM(x), current position marked')

        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()


# =============================================================================
# Tab 2: departure from Gaussian
# =============================================================================
class DepartureTab(QtWidgets.QWidget):
    def __init__(self, lib: LSFLibrary, parent=None):
        super().__init__(parent)
        self.lib = lib

        self.order_combo = _order_combo(lib)
        self.pixel_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pixel_slider.setMinimum(0)
        self.pixel_slider.setMaximum(999)
        self.pixel_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_spin.setDecimals(2)
        self.overlay_check = QtWidgets.QCheckBox("Overlay N positions across order")
        self.overlay_n_spin = QtWidgets.QSpinBox()
        self.overlay_n_spin.setRange(2, 40)
        self.overlay_n_spin.setValue(12)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Order:"))
        top.addWidget(self.order_combo, 1)
        top.addWidget(QtWidgets.QLabel("Pixel:"))
        top.addWidget(self.pixel_spin)

        overlay_row = QtWidgets.QHBoxLayout()
        overlay_row.addWidget(self.overlay_check)
        overlay_row.addWidget(self.overlay_n_spin)
        overlay_row.addStretch(1)

        self.canvas = MplCanvas(nrows=1, ncols=2, figsize=(11, 5))
        toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.pixel_slider)
        layout.addLayout(overlay_row)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)

        self.order_combo.currentIndexChanged.connect(self._on_order_changed)
        self.pixel_slider.valueChanged.connect(self._on_slider_changed)
        self.pixel_spin.valueChanged.connect(self._on_spin_changed)
        self.overlay_check.stateChanged.connect(self.redraw)
        self.overlay_n_spin.valueChanged.connect(self.redraw)

        self._syncing = False
        self._on_order_changed()

    def _on_order_changed(self):
        order = _combo_order(self.order_combo)
        x_min, x_max, _ = self.lib.pixel_range(order)
        self.pixel_spin.setRange(x_min, x_max)
        self.pixel_spin.setValue(0.5 * (x_min + x_max))
        self.redraw()

    def _on_slider_changed(self, value):
        if self._syncing:
            return
        order = _combo_order(self.order_combo)
        x_min, x_max, _ = self.lib.pixel_range(order)
        frac = value / self.pixel_slider.maximum()
        self._syncing = True
        self.pixel_spin.setValue(x_min + frac * (x_max - x_min))
        self._syncing = False
        self.redraw()

    def _on_spin_changed(self, value):
        if self._syncing:
            return
        order = _combo_order(self.order_combo)
        x_min, x_max, _ = self.lib.pixel_range(order)
        frac = 0.0 if x_max == x_min else (value - x_min) / (x_max - x_min)
        self._syncing = True
        self.pixel_slider.setValue(int(round(frac * self.pixel_slider.maximum())))
        self._syncing = False
        self.redraw()

    def redraw(self):
        order = _combo_order(self.order_combo)
        x_min, x_max, _ = self.lib.pixel_range(order)
        x = self.pixel_spin.value()

        # Full clear + rebuild (not ax.clear()): a colorbar lives as its
        # own axes attached to the figure, not as a child artist of the
        # axes it annotates, so ax.clear() alone leaves stale colorbars
        # behind (and colorbar.remove()-then-recreate is fragile once the
        # axes it shrank has since been cleared). Cheap at this plot size.
        self.canvas.fig.clear()
        ax_dep, ax_grid = self.canvas.fig.subplots(1, 2)
        self.canvas.axes = (ax_dep, ax_grid)

        if self.overlay_check.isChecked():
            n = self.overlay_n_spin.value()
            positions = np.linspace(x_min, x_max, n)
            cmap = matplotlib.colormaps.get_cmap('viridis')
            for i, xp in enumerate(positions):
                u, dep = self.lib.departure_at_pixel(order, xp)
                ax_dep.plot(u, dep, color=cmap(i / max(n - 1, 1)), lw=1, alpha=0.85,
                            label=f'{xp:.0f}' if n <= 16 else None)
            if n <= 16:
                ax_dep.legend(fontsize=6, ncol=2, title='pixel')
            u, dep = self.lib.departure_at_pixel(order, x)
            ax_dep.plot(u, dep, color='red', lw=2, label='current', zorder=5)
        else:
            u, dep = self.lib.departure_at_pixel(order, x)
            ax_dep.plot(u, dep, color='tab:blue', lw=1.8)

        ax_dep.axhline(0, color='gray', lw=0.5)
        ax_dep.set_xlabel('u [km/s]')
        ax_dep.set_ylabel('phi(u) - Gaussian(u; sigma(x))')
        ax_dep.set_title(f'Order {order}: departure from Gaussian at pixel {x:.1f}')

        u_inducing, x_inducing, shape_coeffs = self.lib.shape_grid(order)
        vmax = np.max(np.abs(shape_coeffs)) or 1.0
        im = ax_grid.imshow(
            shape_coeffs.T, aspect='auto', origin='lower', cmap='RdBu_r',
            extent=[u_inducing.min(), u_inducing.max(), x_inducing.min(), x_inducing.max()],
            vmin=-vmax, vmax=vmax,
        )
        ax_grid.axhline(x, color='k', lw=1.2, ls='--')
        self.canvas.fig.colorbar(im, ax=ax_grid, label='departure')
        ax_grid.set_xlabel('u [km/s]')
        ax_grid.set_ylabel('pixel')
        ax_grid.set_title('2D departure grid D(u,x)')

        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()


# =============================================================================
# Tab 3a: composite LSF at a single wavelength (existing behaviour)
# =============================================================================
class SingleWavelengthWidget(QtWidgets.QWidget):
    COL_ORDER, COL_INCLUDE, COL_WEIGHT, COL_RANGE = range(4)

    def __init__(self, lib: LSFLibrary, parent=None):
        super().__init__(parent)
        self.lib = lib
        self._last_result = None

        self.wavelength_spin = QtWidgets.QDoubleSpinBox()
        self.wavelength_spin.setDecimals(4)
        self.wavelength_spin.setRange(0, 10000)
        lo_all = min(self.lib.wavelength_range(o)[0] for o in lib.orders())
        hi_all = max(self.lib.wavelength_range(o)[1] for o in lib.orders())
        self.wavelength_spin.setRange(lo_all, hi_all)
        self.wavelength_spin.setValue(0.5 * (lo_all + hi_all))

        self.detect_button = QtWidgets.QPushButton("Detect covering orders")
        self.compute_button = QtWidgets.QPushButton("Compute composite")
        self.equal_weights_button = QtWidgets.QPushButton("Reset to equal weights")
        self.save_button = QtWidgets.QPushButton("Save this LSF to FITS...")
        self.save_button.setEnabled(False)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Wavelength [nm]:"))
        top.addWidget(self.wavelength_spin)
        top.addWidget(self.detect_button)
        top.addStretch(1)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Order", "Include", "Weight", "Wavelength range [nm]"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        table_buttons = QtWidgets.QHBoxLayout()
        table_buttons.addWidget(self.compute_button)
        table_buttons.addWidget(self.equal_weights_button)
        table_buttons.addStretch(1)

        left = QtWidgets.QVBoxLayout()
        left.addLayout(top)
        left.addWidget(self.table)
        left.addLayout(table_buttons)
        left.addWidget(self.save_button)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)

        self.canvas = MplCanvas(nrows=1, ncols=1, figsize=(7, 6))
        toolbar_holder = QtWidgets.QVBoxLayout()
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar_holder.addWidget(toolbar)
        toolbar_holder.addWidget(self.canvas, 1)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(toolbar_holder)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)

        self.detect_button.clicked.connect(self._populate_table)
        self.compute_button.clicked.connect(self.redraw)
        self.equal_weights_button.clicked.connect(self._reset_weights)
        self.save_button.clicked.connect(self._save)

        self._populate_table()

    def _populate_table(self):
        wavelength = self.wavelength_spin.value()
        covering = self.lib.orders_covering_wavelength(wavelength)
        if not covering:
            covering = self.lib.orders()  # fall back to everything, user picks manually
        self.table.setRowCount(len(covering))
        for row, order in enumerate(covering):
            lo, hi = self.lib.wavelength_range(order)

            order_item = QtWidgets.QTableWidgetItem(str(order))
            order_item.setFlags(order_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, self.COL_ORDER, order_item)

            include_check = QtWidgets.QTableWidgetItem()
            include_check.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            include_check.setCheckState(QtCore.Qt.Checked if lo <= wavelength <= hi else QtCore.Qt.Unchecked)
            self.table.setItem(row, self.COL_INCLUDE, include_check)

            weight_spin = QtWidgets.QDoubleSpinBox()
            weight_spin.setRange(0.0, 100.0)
            weight_spin.setSingleStep(0.1)
            weight_spin.setValue(1.0)
            self.table.setCellWidget(row, self.COL_WEIGHT, weight_spin)

            range_item = QtWidgets.QTableWidgetItem(f"{lo:.3f} - {hi:.3f}")
            range_item.setFlags(range_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, self.COL_RANGE, range_item)

        self.redraw()

    def _reset_weights(self):
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, self.COL_WEIGHT)
            if w is not None:
                w.setValue(1.0)
        self.redraw()

    def _selected_orders_weights(self):
        orders, weights = [], {}
        for row in range(self.table.rowCount()):
            include_item = self.table.item(row, self.COL_INCLUDE)
            if include_item is None or include_item.checkState() != QtCore.Qt.Checked:
                continue
            order = int(self.table.item(row, self.COL_ORDER).text())
            weight = self.table.cellWidget(row, self.COL_WEIGHT).value()
            orders.append(order)
            weights[order] = weight
        return orders, weights

    def redraw(self):
        wavelength = self.wavelength_spin.value()
        orders, weights = self._selected_orders_weights()

        ax = self.canvas.axes
        ax.clear()
        self._last_result = None
        self.save_button.setEnabled(False)

        if not orders or sum(weights.values()) <= 0:
            ax.set_title("No orders selected / all weights zero")
            self.canvas.draw_idle()
            return

        u_grid, composite, orders_used = self.lib.composite_lsf_at_wavelength(
            wavelength, orders=orders, weights=weights)

        w_total = sum(weights[o] for o in orders_used)
        cmap = matplotlib.colormaps.get_cmap('tab10')
        for i, order in enumerate(orders_used):
            u_o, phi_o = self.lib.lsf_at_wavelength(order, wavelength)
            contribution = np.interp(u_grid, u_o, phi_o, left=0.0, right=0.0) * (weights[order] / w_total)
            ax.plot(u_grid, contribution, color=cmap(i % 10), lw=1.2, alpha=0.85,
                    label=f'order {order} (w={weights[order]:.2f})')

        ax.plot(u_grid, composite, color='k', lw=2.2, label='composite', zorder=5)
        ax.axhline(0, color='gray', lw=0.5)
        ax.legend(fontsize=8)
        ax.set_xlabel('u [km/s]')
        ax.set_ylabel('phi(u)')
        ax.set_title(f'Composite LSF at {wavelength:.4f} nm ({len(orders_used)} order(s))')

        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()

        segments = [{'order': o, 'wavelength': wavelength} for o in orders_used]
        self._last_result = wlsf.WeightedLSFResult(
            u=u_grid, phi=composite, center_wavelength_nm=wavelength, velocity_range_kms=0.0,
            wave_lo_nm=wavelength, wave_hi_nm=wavelength,
            segments=[{**s, 'pixel': float(self.lib.wavelength_to_pixel(s['order'], wavelength)),
                       'order_weight': weights[s['order']], 'order_weight_norm': weights[s['order']] / w_total,
                       'weight_norm': weights[s['order']] / w_total}
                      for s in segments],
        )
        self.save_button.setEnabled(True)

    def _save(self):
        if self._last_result is None:
            return
        save_weighted_lsf_interactive(
            self, self._last_result, default_name=f"LSF_{self.wavelength_spin.value():.2f}NM".replace('.', 'p'))


# =============================================================================
# Tab 3b: weighted-average LSF over a wavelength RANGE
# =============================================================================
class WeightedRangeWidget(QtWidgets.QWidget):
    COL_ORDER, COL_PIXELS, COL_WAVE_IN_ORDER, COL_INCLUDE, COL_WEIGHT, COL_NORM = range(6)

    def __init__(self, lib: LSFLibrary, parent=None):
        super().__init__(parent)
        self.lib = lib
        self._last_result = None
        self._order_segments = {}   # {order: [{'pixel','wavelength'}, ...]}, from generate_order_segments
        lo_all = min(self.lib.wavelength_range(o)[0] for o in lib.orders())
        hi_all = max(self.lib.wavelength_range(o)[1] for o in lib.orders())

        # --- range input: two mutually exclusive methods ---------------------
        self.center_radio = QtWidgets.QRadioButton("Central wavelength + velocity range")
        self.center_radio.setChecked(True)
        self.direct_radio = QtWidgets.QRadioButton("Direct wavelength range")

        self.center_spin = QtWidgets.QDoubleSpinBox()
        self.center_spin.setDecimals(4)
        self.center_spin.setRange(lo_all, hi_all)
        self.center_spin.setValue(0.5 * (lo_all + hi_all))
        self.velocity_spin = QtWidgets.QDoubleSpinBox()
        self.velocity_spin.setDecimals(3)
        self.velocity_spin.setRange(0.001, 1e6)
        self.velocity_spin.setValue(50.0)
        self.velocity_spin.setSuffix(" km/s (full width)")

        self.lo_spin = QtWidgets.QDoubleSpinBox()
        self.lo_spin.setDecimals(4)
        self.lo_spin.setRange(lo_all, hi_all)
        self.lo_spin.setValue(lo_all)
        self.lo_spin.setEnabled(False)
        self.hi_spin = QtWidgets.QDoubleSpinBox()
        self.hi_spin.setDecimals(4)
        self.hi_spin.setRange(lo_all, hi_all)
        self.hi_spin.setValue(hi_all)
        self.hi_spin.setEnabled(False)

        self.range_label = QtWidgets.QLabel()
        self.range_label.setStyleSheet("font-family: monospace;")

        range_group = QtWidgets.QGroupBox("Wavelength range")
        range_layout = QtWidgets.QGridLayout(range_group)
        range_layout.addWidget(self.center_radio, 0, 0, 1, 4)
        range_layout.addWidget(QtWidgets.QLabel("Central wavelength [nm]:"), 1, 0)
        range_layout.addWidget(self.center_spin, 1, 1)
        range_layout.addWidget(QtWidgets.QLabel("Velocity range:"), 1, 2)
        range_layout.addWidget(self.velocity_spin, 1, 3)
        range_layout.addWidget(self.direct_radio, 2, 0, 1, 4)
        range_layout.addWidget(QtWidgets.QLabel("Lower [nm]:"), 3, 0)
        range_layout.addWidget(self.lo_spin, 3, 1)
        range_layout.addWidget(QtWidgets.QLabel("Upper [nm]:"), 3, 2)
        range_layout.addWidget(self.hi_spin, 3, 3)
        range_layout.addWidget(self.range_label, 4, 0, 1, 4)

        # --- quadrature resolution (NOT a per-segment weight -- see docstring
        # of weighted_lsf.generate_order_segments) ------------------------------
        self.n_segments_spin = QtWidgets.QSpinBox()
        self.n_segments_spin.setRange(1, 50)
        self.n_segments_spin.setValue(5)
        self.populate_button = QtWidgets.QPushButton("Populate orders")
        self.show_segments_check = QtWidgets.QCheckBox("Show individual segment curves (faint)")

        seg_row = QtWidgets.QHBoxLayout()
        seg_row.addWidget(QtWidgets.QLabel("Segments per order (resolves variation WITHIN an order;"
                                            " does not change that order's overall weight):"))
        seg_row.addWidget(self.n_segments_spin)
        seg_row.addWidget(self.populate_button)
        seg_row.addStretch(1)

        # --- one row per ORDER: one weight (e.g. S/N) per order, period -----------
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Order", "Pixel range used", "Wavelength range in order [nm]",
             "Include", "Weight (e.g. S/N)", "Normalised weight"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        self.compute_button = QtWidgets.QPushButton("Compute weighted-average LSF")
        self.equal_weights_button = QtWidgets.QPushButton("Reset to equal weights")
        self.save_button = QtWidgets.QPushButton("Save this LSF to FITS...")
        self.save_button.setEnabled(False)

        table_buttons = QtWidgets.QHBoxLayout()
        table_buttons.addWidget(self.compute_button)
        table_buttons.addWidget(self.equal_weights_button)
        table_buttons.addWidget(self.show_segments_check)
        table_buttons.addStretch(1)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(range_group)
        left.addLayout(seg_row)
        left.addWidget(self.table, 1)
        left.addLayout(table_buttons)
        left.addWidget(self.save_button)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)

        self.canvas = MplCanvas(nrows=2, ncols=1, figsize=(7, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})
        toolbar_holder = QtWidgets.QVBoxLayout()
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar_holder.addWidget(toolbar)
        toolbar_holder.addWidget(self.canvas, 1)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(toolbar_holder)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)

        self.center_radio.toggled.connect(self._on_method_toggled)
        self.center_spin.valueChanged.connect(self._update_range_label)
        self.velocity_spin.valueChanged.connect(self._update_range_label)
        self.lo_spin.valueChanged.connect(self._update_range_label)
        self.hi_spin.valueChanged.connect(self._update_range_label)
        self.populate_button.clicked.connect(self._populate_table)
        self.compute_button.clicked.connect(self.redraw)
        self.equal_weights_button.clicked.connect(self._reset_weights)
        self.show_segments_check.stateChanged.connect(self.redraw)
        self.save_button.clicked.connect(self._save)

        self._update_range_label()

    def _on_method_toggled(self):
        using_center = self.center_radio.isChecked()
        self.center_spin.setEnabled(using_center)
        self.velocity_spin.setEnabled(using_center)
        self.lo_spin.setEnabled(not using_center)
        self.hi_spin.setEnabled(not using_center)
        self._update_range_label()

    def current_range(self) -> tuple[float, float]:
        if self.center_radio.isChecked():
            return wlsf.velocity_range_to_wavelength(self.center_spin.value(), self.velocity_spin.value())
        return self.lo_spin.value(), self.hi_spin.value()

    def _update_range_label(self):
        lo, hi = self.current_range()
        if lo > hi:
            lo, hi = hi, lo
        self.range_label.setText(f"Selected range: {lo:.4f} - {hi:.4f} nm  (width = {hi - lo:.4f} nm)")

    def _populate_table(self):
        lo, hi = self.current_range()
        self._order_segments = wlsf.generate_order_segments(
            self.lib, lo, hi, n_segments_per_order=self.n_segments_spin.value())
        orders = sorted(self._order_segments)
        self.table.setRowCount(len(orders))
        for row, order in enumerate(orders):
            segs = self._order_segments[order]
            pixels = [s['pixel'] for s in segs]
            waves = [s['wavelength'] for s in segs]

            order_item = QtWidgets.QTableWidgetItem(str(order))
            order_item.setFlags(order_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, self.COL_ORDER, order_item)

            pixel_item = QtWidgets.QTableWidgetItem(f"{min(pixels):.1f} - {max(pixels):.1f}  ({len(segs)} seg.)")
            pixel_item.setFlags(pixel_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, self.COL_PIXELS, pixel_item)

            wave_item = QtWidgets.QTableWidgetItem(f"{min(waves):.4f} - {max(waves):.4f}")
            wave_item.setFlags(wave_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, self.COL_WAVE_IN_ORDER, wave_item)

            include_check = QtWidgets.QTableWidgetItem()
            include_check.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            include_check.setCheckState(QtCore.Qt.Checked)
            self.table.setItem(row, self.COL_INCLUDE, include_check)

            weight_spin = QtWidgets.QDoubleSpinBox()
            weight_spin.setRange(0.0, 1e6)
            weight_spin.setSingleStep(1.0)
            weight_spin.setValue(1.0)
            self.table.setCellWidget(row, self.COL_WEIGHT, weight_spin)

            norm_item = QtWidgets.QTableWidgetItem("--")
            norm_item.setFlags(norm_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, self.COL_NORM, norm_item)

        if not orders:
            QtWidgets.QMessageBox.information(
                self, "No orders",
                "No order in this file covers any part of the selected wavelength range.")

    def _reset_weights(self):
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, self.COL_WEIGHT)
            if w is not None:
                w.setValue(1.0)

    def _selected_order_weights(self):
        order_segments, order_weights = {}, {}
        for row in range(self.table.rowCount()):
            include_item = self.table.item(row, self.COL_INCLUDE)
            if include_item is None or include_item.checkState() != QtCore.Qt.Checked:
                continue
            order = int(self.table.item(row, self.COL_ORDER).text())
            weight = self.table.cellWidget(row, self.COL_WEIGHT).value()
            if order in self._order_segments:
                order_segments[order] = self._order_segments[order]
                order_weights[order] = weight
        return order_segments, order_weights

    def redraw(self):
        ax_top, ax_bot = self.canvas.axes
        ax_top.clear()
        ax_bot.clear()
        self._last_result = None
        self.save_button.setEnabled(False)

        order_segments, order_weights = self._selected_order_weights()
        if not order_segments or sum(order_weights.values()) <= 0:
            ax_top.set_title("No orders selected / all weights zero -- click 'Populate orders' first")
            self.canvas.draw_idle()
            return

        lo, hi = self.current_range()
        velocity_range = self.velocity_spin.value() if self.center_radio.isChecked() else np.nan
        center = self.center_spin.value() if self.center_radio.isChecked() else 0.5 * (lo + hi)

        result = wlsf.compute_weighted_lsf(
            self.lib, order_segments, order_weights, center_wavelength_nm=center,
            velocity_range_kms=velocity_range, wave_lo_nm=lo, wave_hi_nm=hi)

        # update the normalised-weight column (one value per order)
        norm_by_order = {}
        for seg in result.segments:
            norm_by_order.setdefault(seg['order'], seg['order_weight_norm'])
        for row in range(self.table.rowCount()):
            order = int(self.table.item(row, self.COL_ORDER).text())
            include_item = self.table.item(row, self.COL_INCLUDE)
            if include_item is None or include_item.checkState() != QtCore.Qt.Checked or order not in norm_by_order:
                self.table.item(row, self.COL_NORM).setText("--")
                continue
            self.table.item(row, self.COL_NORM).setText(f"{norm_by_order[order]:.4f}")

        cmap = matplotlib.colormaps.get_cmap('tab10')
        orders_present = sorted(order_segments)
        colour_by_order = {o: cmap(i % 10) for i, o in enumerate(orders_present)}

        if self.show_segments_check.isChecked():
            for seg, contribution in zip(result.segments, result.per_segment_phi):
                ax_top.plot(result.u, contribution * seg['weight_norm'], color=colour_by_order[seg['order']],
                            lw=0.5, alpha=0.35)

        for order in orders_present:
            u_o, phi_o = result.per_order_phi[order]
            ax_top.plot(u_o, phi_o, color=colour_by_order[order], lw=1.4, alpha=0.9,
                        label=f"order {order} (S/N-weight={norm_by_order[order]:.2f})")

        ax_top.plot(result.u, result.phi, color='k', lw=2.2, label='weighted average', zorder=5)

        # Best-fit Gaussian, for visual comparison only -- not part of the
        # saved model, not part of `result`. Also drives the departure
        # panel below, which is the direct way to see whether a
        # 'pedestal' (see chat) is present in this particular average.
        gaussian_fit = wlsf.fit_gaussian_to_lsf(result.u, result.phi)
        ax_top.plot(result.u, gaussian_fit['curve'], color='tab:red', lw=1.3, ls='--',
                    label=f"best-fit Gaussian (FWHM={gaussian_fit['fwhm']:.4f} km/s)", zorder=6)

        ax_top.axhline(0, color='gray', lw=0.5)
        ax_top.legend(fontsize=8)
        ax_top.set_ylabel('phi(u)')
        n_seg_total = sum(len(order_segments[o]) for o in orders_present)
        ax_top.set_title(f'Weighted-average LSF, {lo:.4f}-{hi:.4f} nm '
                          f'({len(orders_present)} order(s), {n_seg_total} segment(s) total)')

        ax_bot.plot(result.u, gaussian_fit['departure'], color='tab:purple', lw=1.2)
        ax_bot.axhline(0, color='gray', lw=0.5)
        ax_bot.set_xlabel('u [km/s]')
        ax_bot.set_ylabel('departure')
        ax_bot.set_title('weighted average - best-fit Gaussian', fontsize=9)

        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()

        self._last_result = result
        self.save_button.setEnabled(True)

    def _save(self):
        if self._last_result is None:
            return
        lo, hi = self._last_result.wave_lo_nm, self._last_result.wave_hi_nm
        default_name = f"WLSF_{lo:.2f}_{hi:.2f}NM".replace('.', 'p').replace('-', 'm')
        save_weighted_lsf_interactive(self, self._last_result, default_name=default_name)


# =============================================================================
# Tab 3: composite LSF -- hosts both sub-tabs above
# =============================================================================
class CompositeTab(QtWidgets.QWidget):
    def __init__(self, lib: LSFLibrary, parent=None):
        super().__init__(parent)
        inner_tabs = QtWidgets.QTabWidget()
        inner_tabs.addTab(SingleWavelengthWidget(lib), "Single wavelength")
        inner_tabs.addTab(WeightedRangeWidget(lib), "Wavelength range (weighted average)")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(inner_tabs)


# =============================================================================
# Main window
# =============================================================================
class LSFViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, fits_path: str = None):
        super().__init__()
        self.setWindowTitle("harps.lsf2 -- LSF viewer")
        self.resize(1150, 800)
        self.lib: LSFLibrary = None

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        open_action = QtWidgets.QAction("&Open FITS...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_action)

        self.status = self.statusBar()

        if fits_path:
            self.load_file(fits_path)

    def open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open lsf2 output FITS file", "", "FITS files (*.fits *.fit);;All files (*)")
        if path:
            self.load_file(path)

    def load_file(self, path: str):
        try:
            self.lib = LSFLibrary(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to open file", str(exc))
            return

        self.tabs.clear()
        self.tabs.addTab(LsfAcrossOrderTab(self.lib), "LSF across order")
        self.tabs.addTab(DepartureTab(self.lib), "Departure from Gaussian")
        self.tabs.addTab(CompositeTab(self.lib), "Composite LSF")
        self.status.showMessage(f"Loaded {path}  --  {len(self.lib.orders())} order(s)")


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication(argv)
    fits_path = argv[1] if len(argv) > 1 else None
    window = LSFViewerWindow(fits_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
