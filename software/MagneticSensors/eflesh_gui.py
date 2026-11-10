#!/usr/bin/env python3
"""
eFlesh 4×5 Multiplexed Sensor Visualizer
========================================
Real-time GUI for visualizing 4 groups of 5 MLX90393 magnetometers
streamed from a Teensy 4.0 through a TCA9546A multiplexer.

Helps identify which physical sensor group corresponds to which
mux channel by showing all 20 sensor readings simultaneously.

The GUI delegates all hardware control to eflesh_controller.py,
which handles auto-discovery, handshake, and frame streaming.

Usage:
    uv run eflesh_gui.py              # auto-discover
    uv run eflesh_gui.py --port COM3  # connect to specific port
    uv run eflesh_gui.py --baud 2000000

Dependencies:
    uv sync  (see pyproject.toml)
"""

import sys
import time
import argparse
from collections import deque

import numpy as np

# --- PyQt / pyqtgraph ---
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# --- eFlesh hardware controller (same directory) ---
from eflesh_controller import (
    EfleshController, FrameData, SensorReading,
    NUM_GROUPS, NUM_SENSORS
)

# Re-export for convenience (used by widget code)
GROUP_COLORS = [
    (220, 60, 60),    # Group 0: Red
    (60, 180, 60),    # Group 1: Green
    (60, 120, 220),   # Group 2: Blue
    (220, 180, 40),   # Group 3: Gold/Yellow
]

GROUP_NAMES = [f"Group {i + 1} (Ch{i + 1})" for i in range(NUM_GROUPS)]


# ============================================================================
# Qt Adapter — bridges EfleshController callbacks → Qt signals
# ============================================================================

class ControllerAdapter(QtCore.QObject):
    """Wraps an EfleshController, emitting Qt signals from its callbacks."""

    frame_ready    = QtCore.pyqtSignal(FrameData)
    status_update  = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    header_line    = QtCore.pyqtSignal(str)
    connected      = QtCore.pyqtSignal(str)
    disconnected   = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctrl = EfleshController()

    # -- delegate read-only properties --
    @property
    def is_running(self) -> bool:
        return self._ctrl.is_running

    @property
    def connected_port(self) -> str | None:
        return self._ctrl.connected_port

    @property
    def frame_count(self) -> int:
        return self._ctrl.frame_count

    @property
    def dropped(self) -> int:
        return self._ctrl.dropped

    @property
    def last_seq(self) -> int:
        return self._ctrl.last_seq

    @property
    def bytes_received(self) -> int:
        return self._ctrl.bytes_received

    @property
    def sync_losses(self) -> int:
        return self._ctrl.sync_losses

    @property
    def parse_errors(self) -> int:
        return self._ctrl.parse_errors

    # -- start / stop --
    def start(self, port: str | None = None, baud: int = 2000000):
        ctrl = self._ctrl
        ctrl.on_frame       = lambda f: self.frame_ready.emit(f)
        ctrl.on_status      = lambda m: self.status_update.emit(m)
        ctrl.on_error       = lambda m: self.error_occurred.emit(m)
        ctrl.on_header_line = lambda l: self.header_line.emit(l)
        ctrl.on_connected   = lambda p: self.connected.emit(p)
        ctrl.on_disconnected = lambda: self.disconnected.emit()
        ctrl.start(port=port, baud=baud)

    def stop(self):
        self._ctrl.stop()


# ============================================================================
# Sensor Bar Widget — single sensor visualization
# ============================================================================

class SensorBarWidget(QtWidgets.QWidget):
    """Shows X, Y, Z as colored bars + numeric values for one sensor."""

    AXIS_COLORS = [
        QtGui.QColor(255, 100, 100),   # X = red
        QtGui.QColor(100, 255, 100),   # Y = green
        QtGui.QColor(100, 150, 255),   # Z = blue
    ]
    AXIS_LABELS = ['X', 'Y', 'Z']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = [0.0, 0.0, 0.0]  # x, y, z
        self.magnitude = 0.0
        self._history = deque(maxlen=50)
        self.setMinimumSize(80, 100)
        self.setMaximumWidth(120)

    def update_values(self, x: float, y: float, z: float):
        self.values = [x, y, z]
        self.magnitude = np.sqrt(x*x + y*y + z*z)
        self._history.append(self.magnitude)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin = 4
        bar_area_h = h - 30  # leave room for labels at bottom
        bar_w = (w - 2 * margin) // 3

        # Background
        painter.fillRect(0, 0, w, h, QtGui.QColor(30, 30, 30))

        # Find max for scaling (use historical peak or current max)
        hist_max = max(self._history) if self._history else 1.0
        scale_max = max(hist_max, abs(max(self.values, key=abs)), 1.0) * 1.2

        baseline_y = margin + bar_area_h

        # Draw bars
        for i in range(3):
            val = self.values[i]
            bar_h = int(abs(val) / scale_max * (bar_area_h - 10))
            x0 = margin + i * bar_w

            # Bar rectangle
            if bar_h > 0:
                color = self.AXIS_COLORS[i]
                if val < 0:
                    color = color.darker(140)
                painter.fillRect(x0 + 2, baseline_y - bar_h, bar_w - 4, bar_h, color)

            # Axis label
            painter.setPen(QtGui.QColor(180, 180, 180))
            font = painter.font()
            font.setPointSize(7)
            painter.setFont(font)
            painter.drawText(
                QtCore.QRectF(x0, baseline_y + 2, bar_w, 12),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"{self.AXIS_LABELS[i]}:{val:+.0f}"
            )

        # Magnitude at top
        painter.setPen(QtGui.QColor(220, 220, 220))
        font = painter.font()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRectF(0, 2, w, 14),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            f"{self.magnitude:.0f}"
        )

        painter.end()


# ============================================================================
# Group Panel — one mux channel (5 sensors)
# ============================================================================

class GroupPanel(QtWidgets.QWidget):
    """Panel showing 5 sensor bar widgets for one mux channel."""

    def __init__(self, group_idx: int, parent=None):
        super().__init__(parent)
        self.group_idx = group_idx
        self.sensor_widgets: list[SensorBarWidget] = []

        self._setup_ui()
        self._apply_group_color()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Title
        title = QtWidgets.QLabel(GROUP_NAMES[self.group_idx])
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 11px; padding: 2px;")
        layout.addWidget(title)

        # Sensor bars in a row
        bar_row = QtWidgets.QHBoxLayout()
        bar_row.setSpacing(2)
        for s in range(NUM_SENSORS):
            sw = SensorBarWidget()
            bar_row.addWidget(sw)
            self.sensor_widgets.append(sw)

        layout.addLayout(bar_row)

        # Magnitude sparkline area (simple label for now)
        self.info_label = QtWidgets.QLabel("--")
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("font-size: 8px; color: #888;")
        layout.addWidget(self.info_label)

    def _apply_group_color(self):
        r, g, b = GROUP_COLORS[self.group_idx]
        self.setStyleSheet(
            f"QWidget {{ background-color: rgb({r//4},{g//4},{b//4}); "
            f"border: 2px solid rgb({r},{g},{b}); border-radius: 4px; }}"
            f"QLabel {{ background: transparent; color: rgb({r},{g},{b}); }}"
        )

    def update_data(self, readings: list[SensorReading]):
        """Update all 5 sensor widgets with new readings."""
        for i, sw in enumerate(self.sensor_widgets):
            if i < len(readings):
                sw.update_values(readings[i].x, readings[i].y, readings[i].z)

        # Compute group summary
        mags = [r.magnitude for r in readings[:len(self.sensor_widgets)]]
        avg_mag = np.mean(mags) if mags else 0
        max_mag = np.max(mags) if mags else 0
        self.info_label.setText(f"avg: {avg_mag:.0f} | max: {max_mag:.0f} µT")


# ============================================================================
# Main Window
# ============================================================================

class EFlashVisualizer(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self, port: str = None, baud: int = 2000000):
        super().__init__()
        self.setWindowTitle("eFlesh 4×5 Sensor Visualizer")
        self.setMinimumSize(900, 600)

        self._port = port
        self._baud = baud
        self._adapter: ControllerAdapter | None = None
        self._fps_timer = time.time()
        self._fps = 0.0
        self._fps_frame_count = 0
        self._diag_timer = time.time()
        self._diag_last_bytes = 0
        self._prev_axes: np.ndarray | None = None
        self._active_groups: int | None = None

        self._setup_ui()

        # Auto-connect if port specified, else show ready state
        if self._port:
            self._connect(port=self._port)
        else:
            self._status_label.setText(
                "Ready. Click 'Auto-Connect' to discover device, "
                "or select a port and click Connect."
            )

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # --- Top bar ---
        top_bar = QtWidgets.QHBoxLayout()

        self._auto_btn = QtWidgets.QPushButton("🔍 Auto-Connect")
        self._auto_btn.setToolTip("Scan all COM ports for an eFlesh device")
        self._auto_btn.clicked.connect(self._auto_connect)
        top_bar.addWidget(self._auto_btn)

        top_bar.addWidget(QtWidgets.QLabel("  or Port:"))

        self._port_combo = QtWidgets.QComboBox()
        self._port_combo.setMinimumWidth(110)
        self._port_combo.setEditable(True)
        self._port_combo.setToolTip("Select a specific COM port")
        # Populate with common port names as helpful defaults
        for p in ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8',
                   '/dev/ttyACM0', '/dev/tty.usbmodem*']:
            self._port_combo.addItem(p)
        top_bar.addWidget(self._port_combo)

        self._connect_btn = QtWidgets.QPushButton("Connect")
        self._connect_btn.clicked.connect(self._toggle_connection)
        top_bar.addWidget(self._connect_btn)

        self._disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self._disconnect_btn.clicked.connect(self._disconnect)
        self._disconnect_btn.setEnabled(False)
        top_bar.addWidget(self._disconnect_btn)

        top_bar.addStretch()

        self._fps_label = QtWidgets.QLabel("FPS: --")
        self._fps_label.setStyleSheet("font-weight: bold; color: #0f0;")
        top_bar.addWidget(self._fps_label)

        self._drop_label = QtWidgets.QLabel("")
        self._drop_label.setStyleSheet("color: #f80;")
        top_bar.addWidget(self._drop_label)

        self._parse_label = QtWidgets.QLabel("")
        self._parse_label.setStyleSheet("color: #6cf;")
        top_bar.addWidget(self._parse_label)

        self._payload_label = QtWidgets.QLabel("")
        self._payload_label.setStyleSheet("color: #ccc;")
        top_bar.addWidget(self._payload_label)

        main_layout.addLayout(top_bar)

        # --- Status bar ---
        self._status_label = QtWidgets.QLabel("Ready. Select port and click Connect.")
        self._status_label.setStyleSheet("color: #888; font-size: 10px;")
        main_layout.addWidget(self._status_label)

        # --- Group panels (2×2 grid) ---
        grid_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)
        self._group_panels: list[GroupPanel] = []
        for g in range(NUM_GROUPS):
            panel = GroupPanel(g)
            self._group_panels.append(panel)
            row = g // 2
            col = g % 2
            grid.addWidget(panel, row, col)

        # --- Console log ---
        self._console = QtWidgets.QPlainTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumBlockCount(2000)   # keep memory bounded
        self._console.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #1a1a1a;"
            "  color: #a0d0a0;"
            "  font-family: 'Consolas', 'Courier New', monospace;"
            "  font-size: 10px;"
            "  border: 1px solid #444;"
            "  border-radius: 3px;"
            "}"
        )
        self._console.setPlaceholderText(
            "MCU console output will appear here after connecting...\n"
            "Shows I²C scan, sensor init, and handshake diagnostics."
        )

        # Splitter: grid on top, console on bottom (resizable)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(grid_widget)
        splitter.addWidget(self._console)
        splitter.setStretchFactor(0, 2)   # grid gets more space
        splitter.setStretchFactor(1, 1)   # console gets less

        main_layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Serial Connection
    # ------------------------------------------------------------------

    def _auto_connect(self):
        """Auto-discover and connect to an eFlesh device."""
        self._set_controls_enabled(False)
        self._status_label.setText("Scanning for eFlesh device...")
        self._connect(port=None)   # None = auto-discover

    def _toggle_connection(self):
        port = self._port_combo.currentText().strip().split(' ')[0]
        if not port:
            self._status_label.setText("No port specified!")
            return
        self._set_controls_enabled(False)
        self._status_label.setText(f"Connecting to {port}...")
        self._connect(port=port)

    def _connect(self, port: str | None = None):
        self._adapter = ControllerAdapter()
        self._adapter.frame_ready.connect(self._on_frame)
        self._adapter.status_update.connect(self._on_status)
        self._adapter.error_occurred.connect(self._on_error)
        self._adapter.header_line.connect(self._on_header_line)
        self._adapter.connected.connect(self._on_connected)
        self._adapter.disconnected.connect(self._on_disconnected)
        self._adapter.start(port=port, baud=self._baud)

    def _disconnect(self):
        if self._adapter:
            self._adapter.stop()
            self._adapter = None
        self._set_controls_enabled(True)
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._console.clear()
        self._console.setPlaceholderText(
            "MCU console output will appear here after connecting...\n"
            "Shows I²C scan, sensor init, and handshake diagnostics."
        )
        self._status_label.setText("Disconnected.")

    def _set_controls_enabled(self, enabled: bool):
        self._auto_btn.setEnabled(enabled)
        self._port_combo.setEnabled(enabled)
        self._connect_btn.setEnabled(enabled)
        self._disconnect_btn.setEnabled(not enabled)
        self._fps_label.setText("FPS: --")
        self._drop_label.setText("")
        self._parse_label.setText("")
        self._payload_label.setText("")
        self._diag_last_bytes = 0
        self._diag_timer = time.time()
        self._prev_axes = None
        self._active_groups = None
        if not enabled:
            self._console.clear()
            self._console.setPlaceholderText(
                "MCU console output will appear here after connecting...\n"
                "Shows I²C scan, sensor init, and handshake diagnostics."
            )

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _on_connected(self, port: str):
        """Called when handshake succeeds."""
        self._status_label.setText(f"Connected to {port} — reading header...")

    def _on_disconnected(self):
        self._set_controls_enabled(True)
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)

    def _on_frame(self, frame: FrameData):
        """Called when a complete frame is received."""
        # FPS calculation using a local counter (avoids touching read-only property)
        self._fps_frame_count += 1
        now = time.time()
        fps_elapsed = now - self._fps_timer
        if fps_elapsed >= 1.0:
            self._fps = self._fps_frame_count / fps_elapsed
            self._fps_frame_count = 0
            self._fps_timer = now
            self._fps_label.setText(f"FPS: {self._fps:.1f}")
            self._fps_label.setStyleSheet(
                f"font-weight: bold; color: {'#0f0' if self._fps > 10 else '#f80'};"
            )

        # Lightweight payload-health probe to distinguish "valid frames" vs "all zeros/static"
        axes = np.array(
            [[r.x, r.y, r.z] for grp in frame.groups for r in grp],
            dtype=float
        )
        mags = np.linalg.norm(axes, axis=1)
        nonzero = int(np.sum(mags > 1e-3))
        mean_mag = float(np.mean(mags)) if mags.size else 0.0
        # Per-group non‑zero breakdown (5 sensors each)
        nz_per_group = [
            int(np.sum(np.linalg.norm(
                np.array([[r.x, r.y, r.z] for r in frame.groups[g]], dtype=float),
                axis=1
            ) > 1e-3))
            for g in range(NUM_GROUPS)
        ]
        movement = 0.0
        if self._prev_axes is not None and self._prev_axes.shape == axes.shape:
            movement = float(np.mean(np.abs(axes - self._prev_axes)))
        self._prev_axes = axes

        diag_elapsed = now - self._diag_timer
        if diag_elapsed >= 1.0 and self._adapter:
            rx_delta = self._adapter.bytes_received - self._diag_last_bytes
            rx_kbps = (rx_delta / max(diag_elapsed, 1e-6)) / 1024.0
            self._diag_last_bytes = self._adapter.bytes_received
            self._diag_timer = now

            self._parse_label.setText(
                f"RX: {rx_kbps:.1f} KiB/s | sync-loss: {self._adapter.sync_losses} | parse-err: {self._adapter.parse_errors}"
            )
            nz_str = " ".join(
                f"G{g}:{n}" for g, n in enumerate(nz_per_group)
            )
            self._payload_label.setText(
                f"nz: {nonzero}/20 [{nz_str}] | mean|B|: {mean_mag:.1f} | ΔB: {movement:.3f}"
            )

        # Drops from controller
        if self._adapter and self._adapter.dropped > 0:
            self._drop_label.setText(f"Drops: {self._adapter.dropped}")

        # Update all group panels
        for g in range(NUM_GROUPS):
            self._group_panels[g].update_data(frame.groups[g])

    def _on_error(self, msg: str):
        self._status_label.setText(f"Error: {msg}")
        self._set_controls_enabled(True)
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._adapter = None

    def _on_status(self, msg: str):
        self._status_label.setText(msg)

    def _on_header_line(self, line: str):
        """Append MCU header lines to the console log."""
        is_summary = "active group(s)" in line and line.startswith("Total:")

        if line:
            self._console.appendPlainText(line)

        if is_summary:
            try:
                count = int(line.split(":", 1)[1].strip().split(" ")[0])
            except Exception:
                return

            self._active_groups = count
            if count < NUM_GROUPS:
                self._status_label.setText(
                    f"Warning: firmware reports {count}/{NUM_GROUPS} active groups. "
                    f"Expect zeros on missing channels."
                )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._adapter and self._adapter.is_running:
            self._adapter.stop()
        event.accept()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="eFlesh 4×5 Sensor Visualizer"
    )
    parser.add_argument('--port', '-p', type=str, default=None,
                        help='Serial port (e.g., COM3, /dev/ttyACM0)')
    parser.add_argument('--baud', '-b', type=int, default=2000000,
                        help='Baud rate (default: 2000000)')
    args = parser.parse_args()

    # pyqtgraph defaults
    pg.setConfigOptions(antialias=True)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark theme
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(40, 40, 40))
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 220, 220))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(50, 50, 50))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(220, 220, 220))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 220, 220))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(55, 55, 55))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(220, 220, 220))
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 60, 60))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(100, 150, 255))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(100, 150, 255))
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(30, 30, 30))
    app.setPalette(dark_palette)

    window = EFlashVisualizer(port=args.port, baud=args.baud)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
