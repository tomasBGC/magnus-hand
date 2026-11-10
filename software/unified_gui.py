#!/usr/bin/env python3
"""
eFlesh Unified Control Centre
==============================
Three-tab GUI:
  1. Monitor & Record — live force (g + N), theta/phi naming
  2. Load Cell Calibration — tare, set factor, compute from known mass
  3. Data Analysis — single-file interactive plots

Usage:
    uv run unified_gui.py
    uv run unified_gui.py -o ./data --mag-port COM3 --load-port COM5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime as _dt
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtWidgets, QtCore, QtGui

from unified_controller import (
    NUM_GROUPS,
    NUM_SENSORS,
    DataRecord,
    FrameData,
    UnifiedController,
)

# ============================================================================
# Constants
# ============================================================================

GROUP_COLORS = [
    (220, 60,  60),
    (60,  180, 60),
    (60,  120, 220),
    (220, 180, 40),
]
GROUP_NAMES = [f"Group {i + 1}" for i in range(NUM_GROUPS)]

WINDOW_TITLE  = "eFlesh Unified Control Centre"
MIN_WINDOW_W  = 1060
MIN_WINDOW_H  = 720
FORCE_HISTORY = 200
DEFAULT_OUT   = "./recordings"
GF_TO_N       = 0.00981   # gram-force → Newtons

# HX711 load cell readings are noise/glitches above this cap and must be
# discarded (not clipped) before plotting.
MAX_VALID_FORCE_N = 20.0
MAX_VALID_FORCE_G = MAX_VALID_FORCE_N / GF_TO_N   # ~2038.7 g

# Per-sensor and XYZ axis colours used in plots
_S_COLORS   = ['#ff6464', '#64ff96', '#6496ff', '#ffff64', '#ff80ff']
_XYZ_COLORS = ['#ff6464', '#64ff96', '#6496ff']


# ============================================================================
# Qt Adapter — bridges UnifiedController callbacks → Qt signals
# ============================================================================

class ControllerAdapter(QtCore.QObject):

    status_update    = QtCore.pyqtSignal(str)
    error_occurred   = QtCore.pyqtSignal(str)
    record_ready     = QtCore.pyqtSignal(DataRecord)
    mag_frame        = QtCore.pyqtSignal(FrameData)
    load_serial_line = QtCore.pyqtSignal(str)
    cal_op_done      = QtCore.pyqtSignal(bool, str)
    quality_check    = QtCore.pyqtSignal(str)

    def __init__(self, data_dir: str | None = None, parent=None):
        super().__init__(parent)
        self._ctrl = UnifiedController(data_dir=data_dir or DEFAULT_OUT)
        self._cal_thread: threading.Thread | None = None

    @property
    def data_dir(self) -> Path:
        return self._ctrl.data_dir

    @property
    def is_recording(self) -> bool:
        return self._ctrl.is_recording

    @property
    def mag_connected(self) -> bool:
        return self._ctrl.mag_connected

    @property
    def load_connected(self) -> bool:
        return self._ctrl.load_connected

    @property
    def record_count(self) -> int:
        return self._ctrl.record_count

    @property
    def latest_force(self) -> float | None:
        return self._ctrl.latest_force

    @property
    def latest_frame(self) -> FrameData | None:
        return self._ctrl.latest_frame

    @property
    def loadscale_cal_factor(self) -> float | None:
        return self._ctrl.loadscale_cal_factor

    def _wire(self):
        self._ctrl.on_status           = lambda m: self.status_update.emit(m)
        self._ctrl.on_error            = lambda m: self.error_occurred.emit(m)
        self._ctrl.on_mag_frame        = lambda f: self.mag_frame.emit(f)
        self._ctrl.on_record           = lambda r: self.record_ready.emit(r)
        self._ctrl.on_load_serial_line = lambda l: self.load_serial_line.emit(l)
        self._ctrl.on_quality_check    = lambda t: self.quality_check.emit(t)

    def connect_all(self, mag_port=None, load_port=None):
        self._wire()
        try:
            self._ctrl.connect_all(mag_port=mag_port, load_port=load_port)
        except RuntimeError as e:
            self.error_occurred.emit(str(e))
            self._ctrl.disconnect_all()

    def connect_magnetic(self, port=None):
        self._wire()
        self._ctrl.connect_magnetic(port=port)

    def connect_loadscale(self, port=None):
        self._wire()
        try:
            self._ctrl.connect_loadscale(port=port)
        except RuntimeError as e:
            self.error_occurred.emit(str(e))

    def disconnect_all(self):
        self._ctrl.disconnect_all()

    def set_metadata(self, **kwargs):
        self._ctrl.set_metadata(**kwargs)

    def set_data_dir(self, path):
        self._ctrl.set_data_dir(path)

    def start_recording(self):
        try:
            self._ctrl.start_recording()
        except RuntimeError as e:
            self.error_occurred.emit(str(e))

    def stop_recording(self, filename_stem: str | None = None) -> str | None:
        return self._ctrl.stop_recording(filename_stem=filename_stem)

    def update_summary_files(self, unit: str | None = None) -> str:
        return self._ctrl.update_summary_files(unit=unit)

    def poll_force(self) -> float | None:
        return self._ctrl.poll_force()

    def tare(self):
        try:
            self._ctrl.tare_loadscale()
        except RuntimeError as e:
            self.error_occurred.emit(str(e))

    def set_cal_factor(self, factor: float):
        self._run_cal_op(lambda: self._ctrl.set_loadscale_cal_factor(factor))

    def compute_and_apply_cal(self, known_mass_g: float):
        self._run_cal_op(lambda: self._ctrl.compute_and_apply_cal(known_mass_g))

    def _run_cal_op(self, fn):
        def _worker():
            try:
                ok = fn()
                msg = ("Calibration saved to EEPROM."
                       if ok else "Timed out waiting for EEPROM confirmation.")
                self.cal_op_done.emit(ok, msg)
            except Exception as exc:
                self.cal_op_done.emit(False, str(exc))
        self._cal_thread = threading.Thread(target=_worker, daemon=True)
        self._cal_thread.start()


# ============================================================================
# Shared sensor display widgets
# ============================================================================

class ForceGaugeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._force: float | None = None
        self._history: deque[float] = deque(maxlen=FORCE_HISTORY)
        self.setMinimumHeight(64)
        self.setMaximumHeight(90)

    def update_force(self, force: float | None):
        self._force = force
        if force is not None:
            self._history.append(force)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor(30, 30, 30))

        # Header row: "Force (g)" left, Newton value right
        p.setPen(QtGui.QColor(180, 180, 180))
        f = p.font(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        p.drawText(8, 16, "Force (g)")
        if self._force is not None:
            p.setPen(QtGui.QColor(100, 220, 255))
            p.drawText(
                QtCore.QRect(0, 2, w - 8, 16),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                f"{self._force * GF_TO_N:.5f} N",
            )

        bx, by, bw, bh = 8, 22, w - 16, 22
        p.fillRect(bx, by, bw, bh, QtGui.QColor(50, 50, 50))
        if self._force is not None:
            mx = max(max(self._history, default=1), abs(self._force), 1.0)
            fill = int(min(abs(self._force) / mx, 1.0) * bw)
            color = (QtGui.QColor(80, 200, 255) if self._force >= 0
                     else QtGui.QColor(255, 100, 80))
            p.fillRect(bx, by, fill, bh, color)
            p.setPen(QtGui.QColor(255, 255, 255))
            f2 = p.font(); f2.setPointSize(13); f2.setBold(True); p.setFont(f2)
            p.drawText(bx + 6, by + 17, f"{self._force:.1f} g")
        else:
            p.setPen(QtGui.QColor(120, 120, 120))
            p.drawText(bx + 6, by + 17, "—  no data")

        if len(self._history) > 1:
            hist = list(self._history)
            sy = by + bh + 8
            sh = max(h - sy - 4, 1)
            mx2 = max(abs(v) for v in hist) or 1.0
            p.setPen(QtGui.QPen(QtGui.QColor(80, 200, 255, 100), 1))
            path = QtGui.QPainterPath()
            for i, v in enumerate(hist):
                px = bx + int(i / max(len(hist) - 1, 1) * bw)
                py = sy + sh - int((v + mx2) / (2 * mx2) * sh)
                if i == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            p.drawPath(path)
        p.end()


class MagMiniBar(QtWidgets.QWidget):
    AXIS_COLORS = [
        QtGui.QColor(255, 100, 100),
        QtGui.QColor(100, 255, 100),
        QtGui.QColor(100, 150, 255),
    ]
    LABELS = ["X", "Y", "Z"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = self._y = self._z = 0.0
        self._mag = 0.0
        self.setMinimumSize(55, 72)
        self.setMaximumWidth(70)

    def update_values(self, x, y, z):
        self._x, self._y, self._z = x, y, z
        self._mag = np.sqrt(x*x + y*y + z*z)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor(25, 25, 25))
        vals = [self._x, self._y, self._z]
        mx = max(abs(v) for v in vals) or 1.0
        area_h = h - 18
        bar_w = max(4, (w - 8) // 3)
        margin, base = 3, 3 + area_h
        for i, val in enumerate(vals):
            bh = int(abs(val) / mx * (area_h - 4))
            x0 = margin + i * bar_w
            if bh > 0:
                c = self.AXIS_COLORS[i]
                p.fillRect(x0 + 1, base - bh, bar_w - 2, bh,
                           c if val >= 0 else c.darker(140))
            p.setPen(QtGui.QColor(160, 160, 160))
            f = p.font(); f.setPointSize(6); p.setFont(f)
            p.drawText(QtCore.QRectF(x0, base + 1, bar_w, 14),
                       QtCore.Qt.AlignmentFlag.AlignCenter,
                       f"{self.LABELS[i]}:{val:+.0f}")
        p.setPen(QtGui.QColor(220, 220, 220))
        f2 = p.font(); f2.setPointSize(7); f2.setBold(True); p.setFont(f2)
        p.drawText(QtCore.QRectF(0, 0, w, 12),
                   QtCore.Qt.AlignmentFlag.AlignCenter, f"{self._mag:.0f}")
        p.end()


class GroupPanel(QtWidgets.QWidget):
    def __init__(self, group_idx: int, parent=None):
        super().__init__(parent)
        self._bars: list[MagMiniBar] = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(1)

        r, g, b = GROUP_COLORS[group_idx]
        title = QtWidgets.QLabel(GROUP_NAMES[group_idx])
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-weight:bold;font-size:10px;color:rgb({r},{g},{b});")
        layout.addWidget(title)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(2)
        for _ in range(NUM_SENSORS):
            bar = MagMiniBar()
            row.addWidget(bar)
            self._bars.append(bar)
        layout.addLayout(row)

        r4, g4, b4 = r // 4, g // 4, b // 4
        self.setStyleSheet(
            f"QWidget{{background-color:rgb({r4},{g4},{b4});"
            f"border:1px solid rgb({r},{g},{b});border-radius:3px;}}"
            f"QLabel{{background:transparent;}}"
        )

    def update_data(self, readings):
        for i, bar in enumerate(self._bars):
            if i < len(readings):
                bar.update_values(readings[i].x, readings[i].y, readings[i].z)


# ============================================================================
# Tab 1 — Monitor & Record
# ============================================================================

class MonitorTab(QtWidgets.QWidget):
    """Live sensor display + recording controls with theta/phi naming."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Force gauge ────────────────────────────────────────────────
        self._force_gauge = ForceGaugeWidget()
        layout.addWidget(self._force_gauge)

        # ── Magnetic sensor grid (2×2) ─────────────────────────────────
        grid_w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_w)
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)
        self._group_panels: list[GroupPanel] = []
        for g in range(NUM_GROUPS):
            panel = GroupPanel(g)
            self._group_panels.append(panel)
            grid.addWidget(panel, g // 2, g % 2)
        layout.addWidget(grid_w, stretch=1)

        # ── Recording controls ─────────────────────────────────────────
        rec_box = QtWidgets.QGroupBox("Recording")
        rbl = QtWidgets.QVBoxLayout(rec_box)
        rbl.setSpacing(6)

        # Row 1: Unit / Theta / Phi / optional note
        meta_row = QtWidgets.QHBoxLayout()
        meta_row.addWidget(QtWidgets.QLabel("Unit:"))
        self._unit_edit = QtWidgets.QLineEdit()
        self._unit_edit.setPlaceholderText("unit1 (default)")
        self._unit_edit.setToolTip(
            "Fingertip unit identifier. Leave blank (or 'unit1') for the "
            "original fingertip — its recordings and summary CSVs are "
            "unaffected. Any other value (e.g. 'unit2') keeps that unit's "
            "recordings and characterisation_summary_*/master_characterisation_*"
            " CSVs separate.")
        self._unit_edit.setMaximumWidth(80)
        meta_row.addWidget(self._unit_edit)
        meta_row.addSpacing(10)

        meta_row.addWidget(QtWidgets.QLabel("Theta:"))
        self._theta_edit = QtWidgets.QLineEdit()
        self._theta_edit.setPlaceholderText("e.g. 45")
        self._theta_edit.setMaximumWidth(65)
        meta_row.addWidget(self._theta_edit)
        meta_row.addSpacing(10)

        meta_row.addWidget(QtWidgets.QLabel("Phi:"))
        self._phi_edit = QtWidgets.QLineEdit()
        self._phi_edit.setPlaceholderText("e.g. 0")
        self._phi_edit.setMaximumWidth(65)
        meta_row.addWidget(self._phi_edit)
        meta_row.addSpacing(14)

        meta_row.addWidget(QtWidgets.QLabel("Note:"))
        self._notes_edit = QtWidgets.QLineEdit()
        self._notes_edit.setPlaceholderText("optional")
        self._notes_edit.setMaximumWidth(120)
        meta_row.addWidget(self._notes_edit)
        meta_row.addStretch()
        rbl.addLayout(meta_row)

        # Row 2: Record button + snapshot count + elapsed
        btn_row = QtWidgets.QHBoxLayout()
        self._rec_btn = QtWidgets.QPushButton("⏺  Start Recording")
        self._rec_btn.setEnabled(False)
        self._rec_btn.setStyleSheet(
            "QPushButton{font-weight:bold;font-size:14px;padding:10px 24px;"
            "background:#28a428;color:white;border-radius:5px;}"
            "QPushButton:hover{background:#32c432;}"
            "QPushButton:disabled{background:#444;color:#888;}"
        )
        btn_row.addWidget(self._rec_btn)
        btn_row.addSpacing(16)
        self._count_lbl = QtWidgets.QLabel("Snapshots: 0")
        self._count_lbl.setStyleSheet("font-weight:bold;color:#0f0;font-size:11px;")
        btn_row.addWidget(self._count_lbl)
        self._elapsed_lbl = QtWidgets.QLabel("")
        self._elapsed_lbl.setStyleSheet("color:#ccc;font-size:10px;")
        btn_row.addWidget(self._elapsed_lbl)
        btn_row.addStretch()
        rbl.addLayout(btn_row)

        # Row 3: Output dir
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(QtWidgets.QLabel("Output:"))
        self._out_lbl = QtWidgets.QLabel(DEFAULT_OUT)
        self._out_lbl.setStyleSheet("color:#6cf;font-size:9px;")
        self._out_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        out_row.addWidget(self._out_lbl, stretch=1)
        self._browse_btn = QtWidgets.QPushButton("Browse…")
        self._browse_btn.setMaximumWidth(80)
        out_row.addWidget(self._browse_btn)
        rbl.addLayout(out_row)

        self._saved_lbl = QtWidgets.QLabel("")
        self._saved_lbl.setStyleSheet("color:#6cf;font-size:9px;")
        self._saved_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        rbl.addWidget(self._saved_lbl)

        # ── Quality-check summary (auto-filled after each recording) ────
        self._quality_lbl = QtWidgets.QLabel("")
        self._quality_lbl.setStyleSheet(
            "color:#ccc;font-size:9px;font-family:Consolas,monospace;"
            "padding:4px 6px;background:#242424;border-radius:3px;")
        self._quality_lbl.setWordWrap(True)
        self._quality_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._quality_lbl.setVisible(False)
        rbl.addWidget(self._quality_lbl)

        layout.addWidget(rec_box)
        self._rec_start_time: float | None = None
        self._output_dir = DEFAULT_OUT

    # ── public API used by UnifiedWindow ────────────────────────────────

    def update_force(self, force: float | None):
        self._force_gauge.update_force(force)

    def update_mag(self, frame: FrameData):
        for g in range(NUM_GROUPS):
            self._group_panels[g].update_data(frame.groups[g])

    def set_recording_enabled(self, enabled: bool):
        self._rec_btn.setEnabled(enabled)

    def set_recording(self, recording: bool):
        if recording:
            self._rec_btn.setText("⏹  Stop Recording")
            self._rec_btn.setStyleSheet(
                "QPushButton{font-weight:bold;font-size:14px;padding:10px 24px;"
                "background:#c42828;color:white;border-radius:5px;}"
                "QPushButton:hover{background:#e43030;}"
            )
            self._rec_start_time = time.time()
        else:
            self._rec_btn.setText("⏺  Start Recording")
            self._rec_btn.setStyleSheet(
                "QPushButton{font-weight:bold;font-size:14px;padding:10px 24px;"
                "background:#28a428;color:white;border-radius:5px;}"
                "QPushButton:hover{background:#32c432;}"
                "QPushButton:disabled{background:#444;color:#888;}"
            )
            self._rec_start_time = None
            self._elapsed_lbl.setText("")

    def update_elapsed(self):
        if self._rec_start_time is not None:
            elapsed = time.time() - self._rec_start_time
            self._elapsed_lbl.setText(f"  {elapsed:.1f} s")

    def update_count(self, n: int):
        self._count_lbl.setText(f"Snapshots: {n}")

    def set_saved_path(self, path: str):
        self._saved_lbl.setText(f"✓  Saved: {path}")

    def set_pending_filename(self, stem: str):
        self._saved_lbl.setText(f"●  Recording → {stem}.json")
        self._quality_lbl.setVisible(False)

    def set_quality_check(self, text: str):
        self._quality_lbl.setText(text)
        self._quality_lbl.setVisible(True)

    def set_output_label(self, path: str):
        self._out_lbl.setText(path)
        self._output_dir = path

    def get_unit(self) -> str:
        return self._unit_edit.text().strip()

    def get_metadata(self) -> dict:
        return {
            "unit":  self.get_unit(),
            "theta": int(self._theta_edit.text().strip()),
            "phi":   int(self._phi_edit.text().strip()),
        }

    def build_filename_stem(self) -> str:
        """Auto-generate the recording filename stem from unit/theta/phi/date.

        Format: eflesh_[{unit}_]theta{TT}_phi{PP}_{DDMM}_v{N}[_{note}]
        The unit segment is omitted entirely when the Unit field is left
        blank (or set to "unit1"), so the original fingertip's filenames are
        unchanged. A version suffix (_v1, _v2, ...) is appended, incrementing
        past any recording subfolder that already exists for the same
        unit/theta/phi/day, so repeat presses never silently overwrite an
        earlier recording.
        """
        theta = int(self._theta_edit.text().strip())
        phi   = int(self._phi_edit.text().strip())
        today = _dt.now().strftime("%d%m")
        unit  = re.sub(r'[^\w\-]', '_', self.get_unit()).strip('_')
        unit_seg = f"{unit}_" if unit and unit.lower() != "unit1" else ""
        base  = f"eflesh_{unit_seg}theta{theta:02d}_phi{phi:02d}_{today}"
        notes = re.sub(r'[^\w\-]', '_', self._notes_edit.text().strip())

        out_dir = Path(self._output_dir)
        version = 1
        while True:
            stem = f"{base}_v{version}"
            if notes:
                stem = f"{stem}_{notes}"
            if not (out_dir / stem).exists():
                return stem
            version += 1


# ============================================================================
# Tab 2 — Load Cell Calibration
# ============================================================================

class CalibrationTab(QtWidgets.QWidget):

    tare_requested        = QtCore.pyqtSignal()
    set_factor_requested  = QtCore.pyqtSignal(float)
    compute_cal_requested = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        reading_box = QtWidgets.QGroupBox("Live Reading")
        rbl = QtWidgets.QVBoxLayout(reading_box)
        self._reading_lbl = QtWidgets.QLabel("—")
        self._reading_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._reading_lbl.setStyleSheet(
            "font-size:36px;font-weight:bold;color:#5cf;padding:6px;")
        rbl.addWidget(self._reading_lbl)
        stats_row = QtWidgets.QHBoxLayout()
        self._min_lbl   = QtWidgets.QLabel("Min: —")
        self._max_lbl   = QtWidgets.QLabel("Max: —")
        self._cal_lbl   = QtWidgets.QLabel("Cal factor: unknown")
        self._count_lbl = QtWidgets.QLabel("Samples: 0")
        for lbl in (self._min_lbl, self._max_lbl, self._cal_lbl, self._count_lbl):
            lbl.setStyleSheet("color:#aaa;font-size:10px;")
            stats_row.addWidget(lbl)
        rbl.addLayout(stats_row)
        layout.addWidget(reading_box)

        self._min_val: float | None = None
        self._max_val: float | None = None
        self._sample_count = 0

        tare_box = QtWidgets.QGroupBox("Tare")
        tbl = QtWidgets.QHBoxLayout(tare_box)
        self._tare_btn = QtWidgets.QPushButton("Zero Scale (Tare)")
        self._tare_btn.setStyleSheet(
            "QPushButton{padding:6px 20px;background:#3a6a3a;color:white;border-radius:4px;}"
            "QPushButton:hover{background:#4a8a4a;}"
            "QPushButton:disabled{background:#444;color:#888;}")
        self._tare_btn.clicked.connect(self.tare_requested)
        tbl.addWidget(self._tare_btn)
        tbl.addStretch()
        tbl.addWidget(QtWidgets.QLabel("Remove all load before taring."))
        layout.addWidget(tare_box)

        man_box = QtWidgets.QGroupBox("Set Calibration Factor Manually")
        mbl = QtWidgets.QHBoxLayout(man_box)
        mbl.addWidget(QtWidgets.QLabel("Factor:"))
        self._factor_edit = QtWidgets.QLineEdit()
        self._factor_edit.setPlaceholderText("e.g. 195.234")
        self._factor_edit.setMaximumWidth(140)
        mbl.addWidget(self._factor_edit)
        self._apply_btn = QtWidgets.QPushButton("Apply + Save to EEPROM")
        self._apply_btn.setStyleSheet(
            "QPushButton{padding:5px 14px;background:#3a4a7a;color:white;border-radius:4px;}"
            "QPushButton:hover{background:#4a5a9a;}"
            "QPushButton:disabled{background:#444;color:#888;}")
        self._apply_btn.clicked.connect(self._on_apply_factor)
        mbl.addWidget(self._apply_btn)
        mbl.addStretch()
        layout.addWidget(man_box)

        cmp_box = QtWidgets.QGroupBox("Compute & Apply Calibration from Known Mass")
        cbl = QtWidgets.QVBoxLayout(cmp_box)
        note = QtWidgets.QLabel(
            "Place the known mass on the sensor and wait for the reading to stabilise, "
            "then click Compute + Apply.  Requires a calibration factor to be known "
            "(set manually above if 'Cal factor: unknown' is shown).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#bbb;font-size:10px;")
        cbl.addWidget(note)
        cmp_row = QtWidgets.QHBoxLayout()
        cmp_row.addWidget(QtWidgets.QLabel("Known mass (g):"))
        self._mass_edit = QtWidgets.QLineEdit()
        self._mass_edit.setPlaceholderText("e.g. 100")
        self._mass_edit.setMaximumWidth(100)
        cmp_row.addWidget(self._mass_edit)
        self._compute_btn = QtWidgets.QPushButton("Compute + Apply")
        self._compute_btn.setStyleSheet(
            "QPushButton{padding:5px 14px;background:#6a3a6a;color:white;border-radius:4px;}"
            "QPushButton:hover{background:#8a4a8a;}"
            "QPushButton:disabled{background:#444;color:#888;}")
        self._compute_btn.clicked.connect(self._on_compute)
        cmp_row.addWidget(self._compute_btn)
        cmp_row.addStretch()
        cbl.addLayout(cmp_row)
        layout.addWidget(cmp_box)

        self._result_lbl = QtWidgets.QLabel("")
        self._result_lbl.setStyleSheet("font-size:11px;padding:2px;")
        layout.addWidget(self._result_lbl)

        log_box = QtWidgets.QGroupBox("Serial Log")
        lbl_layout = QtWidgets.QVBoxLayout(log_box)
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setMaximumHeight(160)
        self._log.setStyleSheet(
            "QPlainTextEdit{background:#1a1a1a;color:#a0d0a0;"
            "font-family:Consolas,monospace;font-size:9px;border:none;}")
        lbl_layout.addWidget(self._log)
        layout.addWidget(log_box)
        layout.addStretch()

        self._set_cal_controls_enabled(False)

    def update_force(self, force: float | None):
        if force is None:
            return
        self._sample_count += 1
        if self._min_val is None or force < self._min_val:
            self._min_val = force
        if self._max_val is None or force > self._max_val:
            self._max_val = force
        self._reading_lbl.setText(f"{force:.2f} g")
        self._min_lbl.setText(f"Min: {self._min_val:.2f} g")
        self._max_lbl.setText(f"Max: {self._max_val:.2f} g")
        self._count_lbl.setText(f"Samples: {self._sample_count}")

    def update_cal_factor(self, factor: float | None):
        if factor is None:
            self._cal_lbl.setText("Cal factor: unknown")
            self._compute_btn.setToolTip(
                "Cal factor unknown — set it manually above first.")
        else:
            self._cal_lbl.setText(f"Cal factor: {factor:.4f}")
            self._compute_btn.setToolTip("")
            if not self._factor_edit.text():
                self._factor_edit.setText(f"{factor:.6f}")

    def append_log(self, line: str):
        self._log.appendPlainText(line)

    def on_load_connected(self, connected: bool):
        self._set_cal_controls_enabled(connected)
        if connected:
            self._reading_lbl.setText("waiting for data…")
            self._min_val = self._max_val = None
            self._sample_count = 0

    def on_cal_op_done(self, success: bool, message: str):
        self._set_cal_controls_enabled(True)
        color = "#5f5" if success else "#f55"
        prefix = "✓" if success else "✗"
        self._result_lbl.setText(f"{prefix}  {message}")
        self._result_lbl.setStyleSheet(f"color:{color};font-size:11px;padding:2px;")
        QtCore.QTimer.singleShot(6000, lambda: self._result_lbl.setText(""))

    def _set_cal_controls_enabled(self, enabled: bool):
        for w in (self._tare_btn, self._apply_btn, self._compute_btn):
            w.setEnabled(enabled)

    def _on_apply_factor(self):
        try:
            factor = float(self._factor_edit.text().strip())
        except ValueError:
            self._show_result(False, "Enter a valid number for the calibration factor.")
            return
        self._set_cal_controls_enabled(False)
        self._show_result(None, "Saving calibration factor…")
        self.set_factor_requested.emit(factor)

    def _on_compute(self):
        try:
            mass = float(self._mass_edit.text().strip())
            if mass <= 0:
                raise ValueError
        except ValueError:
            self._show_result(False, "Enter a positive known mass in grams.")
            return
        self._set_cal_controls_enabled(False)
        self._show_result(None, "Computing and saving calibration factor…")
        self.compute_cal_requested.emit(mass)

    def _show_result(self, success: bool | None, message: str):
        color, prefix = ("#5f5", "✓") if success is True else \
                        ("#f55", "✗") if success is False else ("#fc0", "⋯")
        self._result_lbl.setText(f"{prefix}  {message}")
        self._result_lbl.setStyleSheet(f"color:{color};font-size:11px;padding:2px;")


# ============================================================================
# Tab 3 — Data Analysis: shared helpers
# ============================================================================

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "",
              hide_xticklabels: bool = False):
    ax.set_facecolor('#222222')
    ax.tick_params(colors='#aaa', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#555')
    ax.xaxis.label.set_color('#aaa')
    ax.yaxis.label.set_color('#aaa')
    ax.grid(True, alpha=0.15, color='#555')
    if title:
        ax.set_title(title, color='#ddd', fontsize=10, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if hide_xticklabels:
        ax.tick_params(labelbottom=False)


# ============================================================================
# Tab 3 — Single-file analysis widget (existing functionality)
# ============================================================================

class _SingleFileWidget(QtWidgets.QWidget):
    """Interactive plots for one recording JSON — unchanged from prior version."""

    def __init__(self, default_dir: str, parent=None):
        super().__init__(parent)
        self._default_dir = default_dir
        self._json_path: Path | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._time_axes: list = []
        self._ax_fvmag = None
        self._xhair_lines: dict = {}
        self._cids: list[int] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        top_row = QtWidgets.QHBoxLayout()
        self._open_btn = QtWidgets.QPushButton("📂  Open Recording…")
        self._open_btn.setStyleSheet(
            "QPushButton{font-weight:bold;padding:8px 18px;"
            "background:#3a4a7a;color:white;border-radius:5px;}"
            "QPushButton:hover{background:#4a5a9a;}")
        self._open_btn.clicked.connect(self._on_open)
        top_row.addWidget(self._open_btn)

        self._path_lbl = QtWidgets.QLabel("No recording loaded.")
        self._path_lbl.setStyleSheet("color:#aaa;font-size:10px;")
        self._path_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        top_row.addWidget(self._path_lbl, stretch=1)
        layout.addLayout(top_row)

        self._hover_lbl = QtWidgets.QLabel("")
        self._hover_lbl.setStyleSheet(
            "color:#bbb;font-size:9px;font-family:Consolas,monospace;padding:1px 4px;")
        layout.addWidget(self._hover_lbl)

        self._status_lbl = QtWidgets.QLabel("")
        self._status_lbl.setStyleSheet("color:#6cf;font-size:9px;padding:1px 4px;")
        self._status_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._status_lbl)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        ph = QtWidgets.QLabel(
            "Open a recording JSON to display:\n"
            "  (a) Force (N) vs time\n"
            "  (b) Magnetic magnitude vs time — Group 1\n"
            "  (c) Force (N) vs magnetic magnitude — Group 1\n"
            "  (d) X / Y / Z components per sensor — Group 1"
        )
        ph.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("color:#777;font-size:12px;line-height:1.6;")
        self._scroll.setWidget(ph)
        layout.addWidget(self._scroll, stretch=1)

    def set_default_dir(self, path: str):
        self._default_dir = path

    def _on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Recording", self._default_dir, "Recording JSON (*.json)")
        if path:
            self._load(path)

    def _load(self, path: str):
        self._json_path = Path(path)
        try:
            with open(self._json_path, encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to load recording", str(exc))
            return

        if self._canvas is not None:
            for cid in self._cids:
                self._canvas.mpl_disconnect(cid)
            self._cids.clear()

        try:
            fig, time_axes, ax_fvmag = self._build_figure(data)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Failed to build plots", str(exc))
            return

        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(int(fig.get_size_inches()[1] * fig.dpi))
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Fixed)

        self._scroll.setWidget(canvas)
        self._canvas    = canvas
        self._time_axes = time_axes
        self._ax_fvmag  = ax_fvmag
        self._xhair_lines = {
            ax: ax.axvline(0, color='#ffffff', lw=0.8, alpha=0.65, visible=False)
            for ax in time_axes
        }
        self._cids = [
            canvas.mpl_connect('motion_notify_event', self._on_motion),
            canvas.mpl_connect('axes_leave_event',    self._on_axes_leave),
            canvas.mpl_connect('figure_leave_event',  self._on_fig_leave),
        ]
        self._path_lbl.setText(str(self._json_path))
        self._status_lbl.setText("")

    def _build_figure(self, data: dict):
        snaps = data.get('snapshots', [])
        if not snaps:
            raise ValueError("Recording contains no snapshots.")
        t0      = snaps[0]['timestamp']
        times   = np.array([s['timestamp'] - t0 for s in snaps])

        def _valid_force_g(raw):
            val = raw if raw is not None else 0.0
            return val if abs(val) <= MAX_VALID_FORCE_G else float('nan')

        force_n = np.array([_valid_force_g(s.get('force_g')) * GF_TO_N for s in snaps])

        n_s = 0
        for s in snaps:
            for g in s.get('groups', []):
                if g['group_index'] == 0:
                    n_s = max(n_s, len(g.get('sensors', [])))

        mags = np.zeros((len(snaps), n_s))
        xyz  = np.zeros((len(snaps), n_s, 3))
        for i, s in enumerate(snaps):
            for g in s.get('groups', []):
                if g['group_index'] == 0:
                    for j, sensor in enumerate(g.get('sensors', [])[:n_s]):
                        mags[i, j]   = sensor.get('magnitude', 0.0)
                        xyz[i, j, 0] = sensor.get('x', 0.0)
                        xyz[i, j, 1] = sensor.get('y', 0.0)
                        xyz[i, j, 2] = sensor.get('z', 0.0)

        n_rows  = 3 + max(n_s, 1)
        h_ratio = [3, 3, 3] + [2] * max(n_s, 1)
        fig = Figure(figsize=(12, sum(h_ratio) * 0.55), facecolor='#1a1a1a')
        gs  = fig.add_gridspec(n_rows, 1, height_ratios=h_ratio, hspace=0.55,
                               left=0.09, right=0.97, top=0.98, bottom=0.02)

        ax_f  = fig.add_subplot(gs[0])
        ax_m  = fig.add_subplot(gs[1], sharex=ax_f)
        ax_sc = fig.add_subplot(gs[2])
        ax_xyz = [fig.add_subplot(gs[3 + k], sharex=ax_f) for k in range(max(n_s, 1))]

        _style_ax(ax_f, title="(a) Force vs Time", ylabel="Force (N)",
                  hide_xticklabels=True)
        ax_f.plot(times, force_n, color='#50c8ff', lw=1.2)

        _style_ax(ax_m, title="(b) Magnetic Magnitude vs Time  —  Group 1",
                  ylabel="|B|", hide_xticklabels=True)
        for j in range(n_s):
            ax_m.plot(times, mags[:, j],
                      color=_S_COLORS[j % len(_S_COLORS)], lw=1.0, label=f"S{j}")
        if n_s:
            ax_m.legend(fontsize=8, loc='upper right', facecolor='#2a2a2a',
                        edgecolor='#555', labelcolor='#ccc', ncol=n_s)

        _style_ax(ax_sc, title="(c) Force (N) vs Magnetic Magnitude  —  Group 1",
                  xlabel="|B|", ylabel="Force (N)")
        for j in range(n_s):
            ax_sc.scatter(mags[:, j], force_n,
                          color=_S_COLORS[j % len(_S_COLORS)],
                          s=5, alpha=0.6, label=f"S{j}")
        if n_s:
            ax_sc.legend(fontsize=8, loc='upper right', facecolor='#2a2a2a',
                         edgecolor='#555', labelcolor='#ccc')

        for k in range(max(n_s, 1)):
            is_last = k == max(n_s, 1) - 1
            _style_ax(ax_xyz[k],
                      title="(d) X / Y / Z Components vs Time  —  Group 1" if k == 0 else "",
                      ylabel=f"S{k}", xlabel="Time (s)" if is_last else "",
                      hide_xticklabels=not is_last)
            if k < n_s:
                for c, lbl in enumerate(['X', 'Y', 'Z']):
                    ax_xyz[k].plot(times, xyz[:, k, c],
                                   color=_XYZ_COLORS[c], lw=0.9, label=lbl)
                ax_xyz[k].legend(fontsize=7, loc='upper right', facecolor='#2a2a2a',
                                 edgecolor='#555', labelcolor='#ccc', ncol=3)

        return fig, [ax_f, ax_m] + ax_xyz, ax_sc

    def _on_motion(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax, x, y = event.inaxes, event.xdata, event.ydata
        if ax in self._time_axes:
            self._hover_lbl.setText(f"t = {x:.4f} s    y = {y:.4f}")
            for a, ln in self._xhair_lines.items():
                ln.set_xdata([x, x])
                ln.set_visible(True)
        elif ax is self._ax_fvmag:
            self._hover_lbl.setText(f"|B| = {x:.3f}    Force = {y:.6f} N")
            self._hide_crosshairs()
        else:
            return
        if self._canvas:
            self._canvas.draw_idle()

    def _on_axes_leave(self, _event):
        self._hide_crosshairs()
        self._hover_lbl.setText("")
        if self._canvas:
            self._canvas.draw_idle()

    def _on_fig_leave(self, _event):
        self._hide_crosshairs()
        self._hover_lbl.setText("")
        if self._canvas:
            self._canvas.draw_idle()

    def _hide_crosshairs(self):
        for ln in self._xhair_lines.values():
            ln.set_visible(False)


# ============================================================================
# Tab 3 — DataAnalysisTab: outer container
# ============================================================================

class DataAnalysisTab(QtWidgets.QWidget):
    """Single-file analysis, wrapped for consistency with the other tabs."""

    update_summary_requested = QtCore.pyqtSignal()

    def __init__(self, default_dir: str, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        summary_row = QtWidgets.QHBoxLayout()
        self._update_summary_btn = QtWidgets.QPushButton("🔄  Update Summary Files")
        self._update_summary_btn.setToolTip(
            "Scan the recordings folder and append any new recordings for the\n"
            "current Unit (Monitor & Record tab) to that unit's\n"
            "master_characterisation*.csv and characterisation_summary*.csv.")
        self._update_summary_btn.setStyleSheet(
            "QPushButton{font-weight:bold;padding:6px 14px;"
            "background:#3a6a4a;color:white;border-radius:5px;}"
            "QPushButton:hover{background:#4a8a5a;}")
        self._update_summary_btn.clicked.connect(self.update_summary_requested.emit)
        summary_row.addWidget(self._update_summary_btn)

        self._update_summary_lbl = QtWidgets.QLabel("")
        self._update_summary_lbl.setStyleSheet("color:#6cf;font-size:10px;padding:2px 8px;")
        summary_row.addWidget(self._update_summary_lbl, stretch=1)
        layout.addLayout(summary_row)

        self._single = _SingleFileWidget(default_dir)
        layout.addWidget(self._single)

    def set_update_summary_result(self, text: str):
        self._update_summary_lbl.setText(text)

    def set_default_dir(self, path: str):
        self._single.set_default_dir(path)


# ============================================================================
# Main Window
# ============================================================================

class UnifiedWindow(QtWidgets.QMainWindow):

    def __init__(self, mag_port=None, load_port=None, data_dir=None):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(MIN_WINDOW_W, MIN_WINDOW_H)

        self._mag_port  = mag_port
        self._load_port = load_port
        self._adapter   = ControllerAdapter(data_dir=data_dir)

        self._setup_ui()
        self._wire_signals()
        self._status("Ready — click Connect to begin.")

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        conn_row = QtWidgets.QHBoxLayout()
        self._conn_all_btn = QtWidgets.QPushButton("🔌  Connect All")
        self._conn_all_btn.setToolTip("Auto-discover both sensors")
        self._conn_all_btn.clicked.connect(self._on_connect_all)
        conn_row.addWidget(self._conn_all_btn)

        self._conn_mag_btn = QtWidgets.QPushButton("Connect Mag")
        self._conn_mag_btn.clicked.connect(self._on_connect_mag)
        conn_row.addWidget(self._conn_mag_btn)

        self._conn_load_btn = QtWidgets.QPushButton("Connect Load Cell")
        self._conn_load_btn.clicked.connect(self._on_connect_load)
        conn_row.addWidget(self._conn_load_btn)

        self._disc_btn = QtWidgets.QPushButton("⏻  Disconnect All")
        self._disc_btn.setEnabled(False)
        self._disc_btn.clicked.connect(self._on_disconnect)
        conn_row.addWidget(self._disc_btn)

        conn_row.addSpacing(16)
        self._mag_ind  = QtWidgets.QLabel("● Mag")
        self._load_ind = QtWidgets.QLabel("● Load")
        for lbl in (self._mag_ind, self._load_ind):
            lbl.setStyleSheet("color:#888;font-weight:bold;")
            conn_row.addWidget(lbl)
        conn_row.addStretch()
        root.addLayout(conn_row)

        self._status_lbl = QtWidgets.QLabel("Initialising…")
        self._status_lbl.setStyleSheet("color:#aaa;font-size:10px;padding:2px;")
        root.addWidget(self._status_lbl)

        self._tabs = QtWidgets.QTabWidget()
        self._monitor_tab = MonitorTab()
        self._tabs.addTab(self._monitor_tab, "Monitor & Record")

        self._cal_tab = CalibrationTab()
        self._tabs.addTab(self._cal_tab, "Load Cell Calibration")

        self._analysis_tab = DataAnalysisTab(default_dir=str(self._adapter.data_dir))
        self._tabs.addTab(self._analysis_tab, "Data Analysis")

        root.addWidget(self._tabs, stretch=1)
        self._monitor_tab.set_output_label(str(self._adapter.data_dir))

    def _wire_signals(self):
        a = self._adapter
        a.status_update.connect(self._status)
        a.error_occurred.connect(self._on_error)
        a.mag_frame.connect(self._monitor_tab.update_mag)
        a.record_ready.connect(self._on_record)
        a.load_serial_line.connect(self._cal_tab.append_log)
        a.cal_op_done.connect(self._on_cal_done)
        a.quality_check.connect(self._monitor_tab.set_quality_check)

        self._cal_tab.tare_requested.connect(lambda: a.tare())
        self._cal_tab.set_factor_requested.connect(lambda f: a.set_cal_factor(f))
        self._cal_tab.compute_cal_requested.connect(lambda m: a.compute_and_apply_cal(m))

        self._monitor_tab._rec_btn.clicked.connect(self._on_toggle_recording)
        self._monitor_tab._browse_btn.clicked.connect(self._on_browse_output)
        self._analysis_tab.update_summary_requested.connect(self._on_update_summary_files)

        self._prev_load_ok = False
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(100)

        self._rec_timer = QtCore.QTimer(self)
        self._rec_timer.timeout.connect(self._monitor_tab.update_elapsed)

    def _poll(self):
        mag_ok  = self._adapter.mag_connected
        load_ok = self._adapter.load_connected

        self._mag_ind.setText("● Mag ✓" if mag_ok else "● Mag ✗")
        self._mag_ind.setStyleSheet(
            f"color:{'#0f0' if mag_ok else '#888'};font-weight:bold;")
        self._load_ind.setText("● Load ✓" if load_ok else "● Load ✗")
        self._load_ind.setStyleSheet(
            f"color:{'#0f0' if load_ok else '#888'};font-weight:bold;")

        if load_ok != self._prev_load_ok:
            self._cal_tab.on_load_connected(load_ok)
            self._prev_load_ok = load_ok

        if load_ok:
            self._adapter.poll_force()

        force = self._adapter.latest_force
        self._monitor_tab.update_force(force)
        self._cal_tab.update_force(force)
        self._cal_tab.update_cal_factor(self._adapter.loadscale_cal_factor)
        self._monitor_tab.set_recording_enabled(mag_ok)

    def _on_connect_all(self):
        self._set_conn_buttons(False)
        self._status("Connecting to both sensors…")
        threading.Thread(
            target=lambda: self._adapter.connect_all(
                mag_port=self._mag_port, load_port=self._load_port),
            daemon=True).start()

    def _on_connect_mag(self):
        self._set_conn_buttons(False)
        self._status("Connecting to magnetic sensor…")
        threading.Thread(
            target=lambda: self._adapter.connect_magnetic(port=self._mag_port),
            daemon=True).start()

    def _on_connect_load(self):
        self._set_conn_buttons(False)
        self._status("Connecting to load cell…")
        threading.Thread(
            target=lambda: self._adapter.connect_loadscale(port=self._load_port),
            daemon=True).start()

    def _on_disconnect(self):
        self._adapter.disconnect_all()
        self._set_conn_buttons(True)
        self._cal_tab.on_load_connected(False)
        self._status("Disconnected.")

    def _set_conn_buttons(self, idle: bool):
        for btn in (self._conn_all_btn, self._conn_mag_btn, self._conn_load_btn):
            btn.setEnabled(idle)
        self._disc_btn.setEnabled(not idle)

    def _on_toggle_recording(self):
        if not self._adapter.is_recording:
            meta = self._monitor_tab.get_metadata()
            self._adapter.set_metadata(**meta)
            self._adapter.start_recording()
            if not self._adapter.is_recording:
                return
            # Filename is fixed at Start time so later edits to the theta/phi/
            # note fields (or a stop that races a filesystem check) can't
            # change where this recording ends up being saved.
            self._pending_stem = self._monitor_tab.build_filename_stem()
            self._monitor_tab.set_recording(True)
            self._monitor_tab.set_pending_filename(self._pending_stem)
            self._monitor_tab.update_count(0)
            self._rec_timer.start(500)
        else:
            self._rec_timer.stop()
            stem = getattr(self, '_pending_stem', None) or self._monitor_tab.build_filename_stem()
            path = self._adapter.stop_recording(filename_stem=stem)
            self._monitor_tab.set_recording(False)
            self._monitor_tab.update_count(self._adapter.record_count)
            if path:
                self._monitor_tab.set_saved_path(path)

    def _on_browse_output(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(self._adapter.data_dir))
        if d:
            self._adapter.set_data_dir(d)
            self._monitor_tab.set_output_label(d)
            self._analysis_tab.set_default_dir(d)

    def _on_update_summary_files(self):
        unit = self._monitor_tab.get_unit()
        try:
            result = self._adapter.update_summary_files(unit)
        except Exception as exc:
            result = f"Update failed: {exc}"
        self._analysis_tab.set_update_summary_result(result)

    def _on_cal_done(self, success: bool, message: str):
        self._cal_tab.on_cal_op_done(success, message)

    def _status(self, msg: str):
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet("color:#aaa;font-size:10px;padding:2px;")

    def _on_error(self, msg: str):
        self._status_lbl.setText(f"⚠  {msg}")
        self._status_lbl.setStyleSheet("color:#f66;font-size:10px;padding:2px;")
        QtCore.QTimer.singleShot(5000, lambda: self._status_lbl.setStyleSheet(
            "color:#aaa;font-size:10px;padding:2px;"))
        self._set_conn_buttons(True)

    def _on_record(self, _: DataRecord):
        self._monitor_tab.update_count(self._adapter.record_count)

    def closeEvent(self, event):
        if self._adapter.is_recording:
            stem = getattr(self, '_pending_stem', None) or self._monitor_tab.build_filename_stem()
            self._adapter.stop_recording(filename_stem=stem)
        self._adapter.disconnect_all()
        event.accept()


# ============================================================================
# Entry point
# ============================================================================

def _dark_palette() -> QtGui.QPalette:
    p = QtGui.QPalette()
    for role, color in {
        QtGui.QPalette.ColorRole.Window:          QtGui.QColor(40,  40,  40),
        QtGui.QPalette.ColorRole.WindowText:      QtGui.QColor(220, 220, 220),
        QtGui.QPalette.ColorRole.Base:            QtGui.QColor(30,  30,  30),
        QtGui.QPalette.ColorRole.AlternateBase:   QtGui.QColor(50,  50,  50),
        QtGui.QPalette.ColorRole.Text:            QtGui.QColor(220, 220, 220),
        QtGui.QPalette.ColorRole.Button:          QtGui.QColor(55,  55,  55),
        QtGui.QPalette.ColorRole.ButtonText:      QtGui.QColor(220, 220, 220),
        QtGui.QPalette.ColorRole.Highlight:       QtGui.QColor(100, 150, 255),
        QtGui.QPalette.ColorRole.HighlightedText: QtGui.QColor(30,  30,  30),
        QtGui.QPalette.ColorRole.ToolTipBase:     QtGui.QColor(220, 220, 220),
        QtGui.QPalette.ColorRole.ToolTipText:     QtGui.QColor(30,  30,  30),
    }.items():
        p.setColor(role, color)
    return p


def main():
    parser = argparse.ArgumentParser(description="eFlesh Unified Control Centre")
    parser.add_argument("--mag-port",   "-m", default=None)
    parser.add_argument("--load-port",  "-l", default=None)
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUT)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())

    window = UnifiedWindow(
        mag_port=args.mag_port,
        load_port=args.load_port,
        data_dir=args.output_dir,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
