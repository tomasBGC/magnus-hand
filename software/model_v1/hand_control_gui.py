"""Unified live sensing + servo control demo for the eFlesh hand.

Combines two already-working, independently-validated pieces into one
program:
  - sensing_gui.py's HandSensingEngine/MainWindow: the exact 4-panel
    live sensing display (force bars, contact/slip colour-coding, rolling
    history graphs, grip total, calibration, glitch guard, ...) -- reused
    HERE by importing that module and subclassing its MainWindow, NOT by
    reimplementing any of it, so that sensing behaviour is guaranteed
    byte-for-byte identical to the standalone tool (see UnifiedMainWindow
    below).
  - hand_controller.py's per-finger enable/close/open functions, relayed
    over the SAME serial connection HandSensingEngine already owns (its
    EfleshController) via hand_controller.connect(), never a second
    connection to the COM port.

This module adds a servo control panel on top of sensing_gui.py
(a shared Speed dropdown; per-finger Enable/Close/Open + Travel% spinbox +
current-position% readout; and Close All/Open All + a shared "All Fingers
%" field that overrides every finger's own Travel spinbox for those two
buttons), enlarges the force-history charts, and hides the base class's
Event Log panel to make room for that (see _hide_event_log). It also hides
the base class's combined "Total Grip Force" label (see _hide_total_label)
and suppresses the per-finger slip/no-slip visual indicator (see
_without_slip_indicator) -- both display-only changes; the underlying
computations (grip total, slip detection) still happen in
sensing_gui.py exactly as before, this module just doesn't draw them.
It does not alter calibration, contact/slip detection, the glitch guard, or
the Force floor in any way.

It also adds two further top-level windows, opened via buttons in the
Servo Control panel (see UnifiedMainWindow._setup_ui's "Additional Views"
row) rather than replacing/tabbing the main window:
  - GripOverviewWindow ("Grip Overview"): a simplified grasp/slip demo view
    -- one combined Grip Status line, 4 plain contact dots, a combined
    force-trend line, and a compact grip/slip event log.
  - DirectionOverviewWindow ("Direction / Deformation Overview"): a
    simplified per-finger contact-direction schematic, arrow orientation
    limited to the classifier's actual 3 Phi classes only.

Both are fed the exact same per-tick FingerResult list UnifiedMainWindow's
own panels already get from self._engine.poll() (see _on_tick) -- neither
owns a HandSensingEngine or connection of its own, so there is still only
ever one live serial connection/data feed for the whole app.

Usage:
    python hand_control_gui.py [--port COM7] [--baud 2000000]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from PyQt6 import QtCore, QtGui, QtWidgets

import sensing_gui as hsg
from AmazingHand import hand_controller as hc

# How long after a servo command is sent to wait before requesting that
# finger's baseline be recalibrated (see UnifiedMainWindow.
# _schedule_post_servo_recalibration) -- long enough for the servo's own
# motion (a couple hundred ms, see hand_controller.DEFAULT_SPEED) and any
# resulting mechanical settling/vibration to finish, so the recalibration's
# no-contact sample window (RECALIBRATION_MIN_STABLE_NO_CONTACT_S in
# sensing_gui.py) starts collecting from genuinely-at-rest readings
# rather than mid-movement ones.
#
# Was ALSO used (see UnifiedMainWindow._settle_until/_on_tick) for how long
# that finger's panel HOLDS its last displayed reading instead of showing
# fresh frames -- baseline recalibration alone only corrects DC drift, it
# doesn't hide the transient mechanical vibration ("ringing") the finger's
# own move produces while this window is still running, which was still
# showing up as Force noise/glitches on that finger's panel right after a
# move. Now DECOUPLED -- see PANEL_SETTLE_DELAY_MS below -- since the two
# purposes have different requirements (recalibration needs genuine DC/
# thermal settling; the display hold only needs to outlast mechanical
# ringing, which clears faster) and sharing one value made every commanded
# move (including Grip-to-Target's own repeated small steps) re-arm a
# 400ms panel freeze, which for a fast-cadence grip attempt could keep the
# panel looking unresponsive for most of the attempt even though the
# underlying Contact/Force was already being computed correctly.
SERVO_SETTLE_DELAY_MS = 400

# 2026-07-20: split from SERVO_SETTLE_DELAY_MS specifically for the panel
# display hold (see UnifiedMainWindow._settle_until/_on_tick) -- baseline
# recalibration timing above is untouched, still 400ms. Matches
# sensing_gui.py's own CONTACT_TRANSIT_SUPPRESS_S=0.15s precedent from the
# same evening (same underlying judgement: how long to hide real post-move
# mechanical settling noise without needlessly blocking readings during a
# grip attempt). Revert to the old shared-400ms behaviour by setting this
# equal to SERVO_SETTLE_DELAY_MS.
PANEL_SETTLE_DELAY_MS = 150

WINDOW_TITLE = "eFlesh Hand Sensing + Control"

# sensing_gui.MainWindow's own MIN_WINDOW_W/H sized its window for the
# 4 sensing panels alone; this module adds a Servo Control panel below them
# too, so it can't just reuse those constants as-is. The window's actual
# minimum height is no longer a second hand-picked constant here -- it used
# to be (hsg.MIN_WINDOW_H + 430), a number tuned once and never rechecked
# against the real layout, which is exactly how it silently drifted to
# demand a taller window than any real screen can give it (see
# FINGER_CHART_MAX_HEIGHT below). It's now computed from the fully-built
# layout at the end of _setup_ui() instead, so it can never again disagree
# with what the widgets actually require. WINDOW_MIN_W is kept as a floor
# on width only, since hsg.MainWindow's connection row/diagnostics text
# sizing that constant was based on is unrelated to anything changed here.
WINDOW_MIN_W = hsg.MIN_WINDOW_W

# Per-finger force-history chart/gauge minimum HEIGHT -- up from
# sensing_gui.py's ForceHistoryChart/VerticalForceGauge own default
# (130px), which hardware testing found too short to clearly read a force
# trend at a glance. This used to be a single hardcoded number (340, then
# 200) picked by assuming a specific screen resolution -- both times that
# assumption turned out wrong for a real monitor: 340 was taller than
# essentially any real screen's usable height; 200 was still taller than
# the height actually available on at least one real screen this was
# tested on, clipping the charts' bottom edge in exactly the same way
# (a widget with a hard setMinimumSize() can't shrink to fit a too-small
# parent, so when the window manager forces the window into the screen's
# actual size on maximize/full-screen, the chart/gauge's geometry extends
# past their squeezed panel's clipped paint bounds). A fixed constant can
# never be right for every monitor -- see UnifiedMainWindow.
# _compute_safe_chart_height, which replaces this with a value computed
# from the ACTUAL screen's available height at startup, so this can't
# recur on a different/smaller display than whatever was assumed here.
# This is now only the UPPER BOUND that computation is allowed to pick --
# kept modest since the stretch factors already in FingerPanel's
# content_row/MainWindow's grid_w grow the charts further to fill a bigger
# window after startup regardless of this starting size.
FINGER_CHART_MAX_HEIGHT = 260

# Width components are unaffected by the vertical-clipping problem above
# (a 2-row grid only multiplies HEIGHT overruns) so these stay fixed.
FINGER_CHART_MIN_WIDTH = 200
FINGER_GAUGE_MIN_WIDTH = 46

# sensing_gui.py's own ForceHistoryChart/VerticalForceGauge default
# minimum height -- used as the size the layout is measured at BEFORE
# enlarging (see _setup_ui), so that measurement reflects a known,
# fixed reference point the safe-height math can scale from. NOT the
# floor _compute_safe_chart_height is allowed to shrink to -- testing
# across a range of real screen sizes found that this window's OTHER
# chrome (Servo Control panel, connection row, diagnostics box, etc.)
# already totals enough that even charts held at this exact default can
# still exceed a smaller real laptop's available height (a 1366x768
# screen has only ~728px available after its taskbar, and this window's
# chrome-only total came to ~782px in testing) -- so a fixed floor here
# would reproduce the identical clipping bug on a small-but-common
# screen. See MIN_CHART_HEIGHT for the actual enforced floor.
BASELINE_CHART_HEIGHT = 130

# The ACTUAL floor _compute_safe_chart_height will shrink to on a
# constrained screen, once BASELINE_CHART_HEIGHT itself doesn't fit --
# small enough to still guarantee the window fits (see BASELINE_CHART_
# HEIGHT above) rather than clip, while still showing a recognisable
# (if compact) trend line rather than a sliver.
MIN_CHART_HEIGHT = 70

# Safety margin (px) subtracted from the screen's available height (already
# excludes the taskbar) before computing the safe chart height -- covers
# window title bar + borders, which availableGeometry() doesn't account
# for, plus a little slack rather than sizing to the exact last pixel.
SCREEN_CHROME_MARGIN_PX = 60

# Default Close/Open travel percentage per finger (see the Travel spinbox
# next to each finger's Close/Open buttons, and Finger.close/open's
# `percent` parameter in hand_controller.py). Index and Thumb are the two
# fingers that physically face each other on this hand and were observed
# crashing hard into each other on a full-travel (100%) Close All -- 70%
# gives a safe-by-default starting point for those two specifically so a
# fresh session doesn't crash out of the box; the user can raise/lower any
# finger's travel independently via its spinbox. Middle/Ring default to
# full travel, unchanged from before this control existed.
DEFAULT_TRAVEL_PERCENT = {"Index": 70, "Thumb": 70}

# (display label, enable fn, Finger object, sensing finger index) -- one row
# per finger, in the order requested: Index, Middle, Ring, Thumb. The Finger
# object (hc.index/middle/ring/thumb) is used directly, rather than the
# hc.close_*/open_* module-level wrappers, so the Speed dropdown's value can
# be passed through to Finger.close/open's `speed` kwarg -- the wrappers
# don't expose it. The sensing finger index is this servo finger's
# corresponding HandSensingEngine.finger_states index (0-based) -- confirmed
# mapping: Index = Finger 1 (index 0), Middle = Finger 2 (index 1), Ring =
# Finger 3 (index 2), Thumb = Finger 4 (index 3). Used by
# _schedule_post_servo_recalibration to recalibrate only the ONE sensing
# finger a given servo move could have disturbed, not all 4.
FINGER_CONTROLS = [
    ("Index", hc.enable_index, hc.index, 0),
    ("Middle", hc.enable_middle, hc.middle, 1),
    ("Ring", hc.enable_ring, hc.ring, 2),
    ("Thumb", hc.enable_thumb, hc.thumb, 3),
]

# label -> (Finger object, sensing index), derived from FINGER_CONTROLS --
# used by the Force-Target Grip step loop, which looks a finger up by label
# far more often than it iterates the full list.
_FINGER_CONTROLS_BY_LABEL = {label: (finger, sensing_index) for label, _, finger, sensing_index in FINGER_CONTROLS}

# Speed dropdown options: (value passed to Finger.close/open's `speed` kwarg,
# display text). Range matches hand_controller.py's own tuning notes (0-6,
# 6 == max speed per AmazingHand_FingerTest.py); DEFAULT_SPEED (3) is
# pre-selected. Shared by every servo command in this window -- individual
# finger buttons and Close All/Open All all read
# UnifiedMainWindow._current_speed() at click time.
SPEED_OPTIONS = [
    (1, "1 (Slowest)"),
    (2, "2"),
    (3, "3 (Default)"),
    (4, "4"),
    (5, "5"),
    (6, "6 (Fastest)"),
]

# The per-finger "current position %" label's 0%=open/100%=closed scale
# (see UnifiedMainWindow._update_position_label) doesn't come from a single
# close()/open() call -- those are two independent one-sided moves from each
# finger's calibrated middle_pos (see hand_controller.Finger), not points on
# one continuous 0-100 axis. This is the pivot where both directions agree
# (percent=0 on either call leaves the finger at middle_pos): a Close at
# percent P reads as NEUTRAL+P/2 (100% at P=100, fully closed); an Open at
# percent P reads as NEUTRAL-P/2 (0% at P=100, fully open).
POSITION_LABEL_NEUTRAL = 50


# ============================================================================
# Force-Target Grip -- closed-loop grip driven by live Force, not a fixed
# percentage. See UnifiedMainWindow._force_grip_round/_force_grip_step_
# fingers for the state machine these constants tune.
# ============================================================================

# Close-percent increment per step -- small enough that a finger doesn't
# overshoot its target Force by much between two consecutive Force checks.
GRIP_STEP_PERCENT = 3

# Safety Max spinbox default/floor (Close-percent scale, same as each
# finger's own Travel spinbox) -- a finger that never reaches Target Force
# stops here regardless, rather than stalling the sequence or crushing
# whatever it's pressed against. GRIP_MIN_MAX_PERCENT keeps the spinbox from
# being set so low the grip can't meaningfully close at all.
GRIP_DEFAULT_MAX_PERCENT = 100
GRIP_MIN_MAX_PERCENT = 10

# Target Force spinbox default (N).
GRIP_DEFAULT_TARGET_N = 2.0

# Live LIGHT/MEDIUM/FIRM touch classification (ForceTargetGripWindow.
# grip_level_lbl, updated every tick in _update_grip_level_label) -- the
# two boundaries sit at the midpoints between the light/medium/firm target
# forces already established by the grasping_dashboard load-cell protocol
# (DEFAULT_FORCE_LEVELS_N: light=0.5N, medium=1.5N, firm=2.0N), so a grip
# actually run at one of those targets reads as the matching label.
GRIP_LEVEL_LIGHT_MEDIUM_N = 1.0
GRIP_LEVEL_MEDIUM_FIRM_N = 1.75


def _grip_level_for_force(force_n: float, contact: bool) -> tuple[str, str]:
    """(label, colour) for the live touch-intensity badge. Uses the same
    smoothed, confirmed-contact-gated fr.force/fr.contact everything else
    reads -- not a second independent raw-force threshold."""
    if not contact:
        return "NO CONTACT", "#888888"
    if force_n < GRIP_LEVEL_LIGHT_MEDIUM_N:
        return "LIGHT", "#2a8f2a"
    if force_n < GRIP_LEVEL_MEDIUM_FIRM_N:
        return "MEDIUM", "#c98a00"
    return "FIRM", "#c0392b"

# Wait after a round's LAST finger command before the NEXT round reads Force
# again -- must be at least SERVO_SETTLE_DELAY_MS so each check reflects a
# settled reading rather than this step's own transit noise (same concern
# SERVO_SETTLE_DELAY_MS already addresses for the single Close/Open buttons).
GRIP_STEP_SETTLE_MS = SERVO_SETTLE_DELAY_MS + 100

# Index+Thumb collision safety cap, specific to this pair -- unlike every
# other aspect of Grip-to-Target, Safety Max (GRIP_DEFAULT_MAX_PERCENT/the
# spinbox above) is a single value shared by ALL FOUR fingers, with no
# per-pair collision guard: if Force never reaches Target (no object
# present, or an object removed mid-grip -- confirmed happening live,
# 2026-07-21), Index and Thumb each independently keep stepping toward that
# SAME shared max with nothing stopping them meeting each other, since nothing
# else in this algorithm accounts for one finger's position affecting
# another's safe range (true_zero_dataset's own data-collection pipeline
# hit the identical risk and caps this pair at 15% for that entirely
# different, deliberately unobstructed-closing scenario -- too conservative
# here, since a real demo object is expected to stop each finger via Force
# feedback well before this point). 70% was live-validated as safe AND
# sufficient on 2026-07-21 (a real grip reached its 2.0N target at ~69-70%
# closure comfortably under this cap) -- see true-zero-threshold-
# investigation memory. A judgment call informed by that one session, not a
# measured physical collision point -- revisit if it ever proves too
# restrictive (a larger/firmer object needing more than 70% closure to
# register Target Force) or not conservative enough (still meeting for a
# thin/soft object) for a specific demo setup.
INDEX_THUMB_GRIP_SAFETY_CAP_PCT = 70


def _grip_max_percent_for(label: str, requested_max_percent: int) -> int:
    """Effective Grip-to-Target Safety Max for `label`: the GUI-specified
    value, further capped for Index/Thumb specifically so this pair can
    never close past INDEX_THUMB_GRIP_SAFETY_CAP_PCT regardless of what the
    Safety Max spinbox says -- see that constant's own comment. Middle/Ring
    are unaffected (no collision risk between either of them and anything
    else in the current 4-finger layout)."""
    if label in ("Index", "Thumb"):
        return min(requested_max_percent, INDEX_THUMB_GRIP_SAFETY_CAP_PCT)
    return requested_max_percent


# ============================================================================
# Window 2: Grip Overview -- simplified grasp/slip demo view
# ============================================================================

# Rolling window for the combined-force trend line -- separate from (and
# shorter than) sensing_gui.py's own HISTORY_WINDOW_S (8s, used by the
# main window's 4 per-finger charts), per this window's own 10-15s request.
GRIP_CHART_WINDOW_S = 12.0

# This chart plots the SUM of all 4 fingers' Force (see GripOverviewWindow.
# apply_results), so it needs more headroom than one finger's own
# hsg.FORCE_GAUGE_MAX_N (15N) scale -- not a full 4x that (60N), since a
# realistic grasp rarely has every finger simultaneously at its own
# individual max; generous enough not to clip a firm combined grip without
# being so wide that an ordinary grip barely registers on the line.
GRIP_CHART_MAX_N = 40.0

_GRIP_DOT_ACTIVE = "#2fbf2f"    # green -- holding (matches FingerPanel's own "contact" state colour)
_GRIP_DOT_INACTIVE = "#555555"  # grey -- no contact

# (background, text) per overall grip status -- same red/green/neutral
# convention as sensing_gui.FingerPanel's own per-finger state colours,
# applied here to the single combined status line instead.
_GRIP_STATUS_STYLE = {
    "No Contact": ("#3a3a3a", "#bbbbbb"),
    "Gripping": ("#123a12", "#7dff7d"),
    "Slip Detected": ("#3a1212", "#ff8080"),
}


class GripOverviewWindow(QtWidgets.QWidget):
    """Simplified grasp/slip demo view: one combined Grip Status line, 4
    plain per-finger contact dots, one combined force-trend line, and a
    compact event log of grip/slip transitions.

    A separate TOP-LEVEL window (not a tab folded into UnifiedMainWindow),
    fed by the exact same poll() results UnifiedMainWindow's own panels
    already get each tick (see UnifiedMainWindow._on_tick -> apply_results
    below) -- this class owns no HandSensingEngine/serial connection of its
    own, so there is still exactly one live data feed/connection for the
    whole app, never a second one opened for this window.

    Slip detection DISPLAY is deliberately reinstated here, for this window
    only -- see the module docstring for how that differs from the main
    sensing window's own per-finger slip indicator. This window's "Slip
    Detected" status and event-log lines just read the exact same
    FingerResult.slip_active/slip_event fields sensing_gui.py already
    computes for every finger regardless (no new sensing/slip logic here,
    display only, same as everything else this window does).
    """

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("eFlesh Hand -- Grip Overview")
        self.resize(520, 560)

        # (timestamp, combined_force) pairs, trimmed to GRIP_CHART_WINDOW_S
        # -- this window's OWN rolling buffer, entirely separate from each
        # FingerState.force_history buffer sensing_gui.py already
        # keeps per finger (those feed the main window's 4 individual
        # charts; this is one combined series derived from the same
        # per-tick results, not a repurposing of an existing buffer).
        self._history: list[tuple[float, float]] = []
        # Previous tick's overall status, so a transition (not just the
        # current level) can be detected -- see apply_results.
        self._last_status: str | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self._status_lbl = QtWidgets.QLabel("No Contact")
        self._status_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._apply_status_style("No Contact")
        root.addWidget(self._status_lbl)

        dots_row = QtWidgets.QHBoxLayout()
        dots_row.addStretch()
        self._dots: list[QtWidgets.QLabel] = []
        for i in range(hsg.NUM_FINGERS):
            col = QtWidgets.QVBoxLayout()
            dot = QtWidgets.QLabel()
            dot.setFixedSize(28, 28)
            dot.setStyleSheet(f"background-color:{_GRIP_DOT_INACTIVE};border-radius:14px;")
            col.addWidget(dot, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
            name_lbl = QtWidgets.QLabel(f"Finger {i + 1}")
            name_lbl.setStyleSheet("color:#aaa;font-size:10px;")
            col.addWidget(name_lbl, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
            dots_row.addLayout(col)
            dots_row.addSpacing(16)
            self._dots.append(dot)
        dots_row.addStretch()
        root.addLayout(dots_row)

        chart_box = QtWidgets.QGroupBox("Combined Grip Force")
        chart_layout = QtWidgets.QVBoxLayout(chart_box)
        # Reuses sensing_gui.ForceHistoryChart directly (same rolling-
        # line-chart widget the main window's 4 per-finger charts use)
        # rather than a second, near-identical chart implementation --
        # window_s/max_n are this window's own constants, everything else
        # about how it renders/labels itself is inherited unchanged.
        self._chart = hsg.ForceHistoryChart(window_s=GRIP_CHART_WINDOW_S, max_n=GRIP_CHART_MAX_N)
        self._chart.setMinimumHeight(180)
        chart_layout.addWidget(self._chart)
        root.addWidget(chart_box, stretch=1)

        log_box = QtWidgets.QGroupBox("Grip Events")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(200)
        self._log.setMaximumHeight(120)
        self._log.setStyleSheet(
            "QPlainTextEdit{background:#1a1a1a;color:#cfd8e3;"
            "font-family:Consolas,monospace;font-size:10px;border:none;}")
        log_layout.addWidget(self._log)
        root.addWidget(log_box)
        # Hidden per-request -- not used. self._log still exists and
        # _log_line still writes into it (see apply_results' "Grip
        # established"/"Grip released"/"Slip detected" calls below), same
        # "hide, don't remove" pattern UnifiedMainWindow uses for the
        # Event Log/Total Force/Diagnostics panels -- just not drawn.
        log_box.hide()

    def _apply_status_style(self, status: str) -> None:
        bg, text = _GRIP_STATUS_STYLE[status]
        self._status_lbl.setStyleSheet(
            f"font-size:26px;font-weight:bold;padding:14px;border-radius:8px;"
            f"background-color:{bg};color:{text};")

    def _log_line(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{ts}] {msg}")

    def apply_results(self, results: list["hsg.FingerResult"]) -> None:
        """Called once per tick with the SAME results list UnifiedMainWindow
        just handed to its own per-finger panels -- see UnifiedMainWindow.
        _on_tick. Purely reactive: reads fields _process_one_finger already
        computed (contact, slip_active, slip_event, force), never touches
        FingerState or re-derives anything sensing-side.
        """
        now = time.time()
        any_slip = any(fr.slip_active for fr in results)
        any_contact = any(fr.contact for fr in results)
        status = "Slip Detected" if any_slip else ("Gripping" if any_contact else "No Contact")

        if status != self._last_status:
            self._status_lbl.setText(status)
            self._apply_status_style(status)
            if status == "Gripping" and self._last_status in (None, "No Contact"):
                self._log_line("Grip established")
            elif status == "No Contact" and self._last_status in ("Gripping", "Slip Detected"):
                self._log_line("Grip released")
            self._last_status = status

        for fr in results:
            self._dots[fr.index].setStyleSheet(
                f"background-color:{_GRIP_DOT_ACTIVE if fr.contact else _GRIP_DOT_INACTIVE};"
                f"border-radius:14px;")
            if fr.slip_event:
                self._log_line(f"Slip detected (Finger {fr.index + 1})")

        total_force = sum(fr.force for fr in results)
        self._history.append((now, total_force))
        while self._history and now - self._history[0][0] > GRIP_CHART_WINDOW_S:
            self._history.pop(0)
        self._chart.update_data(self._history, now)

    def reset(self) -> None:
        """Called on disconnect (see UnifiedMainWindow._on_disconnected/
        _on_error/_on_disconnect_clicked) -- returns the status line/dots
        to a neutral state without clearing the event log's history (same
        convention as the main window's own Event Log, which also isn't
        cleared on disconnect)."""
        self._status_lbl.setText("No Contact")
        self._apply_status_style("No Contact")
        self._last_status = None
        for dot in self._dots:
            dot.setStyleSheet(f"background-color:{_GRIP_DOT_INACTIVE};border-radius:14px;")


# ============================================================================
# Window 3: Direction / Deformation Overview
# ============================================================================

class FingerDirectionWidget(QtWidgets.QWidget):
    """One finger's contact-direction schematic: a neutral grey rounded-
    rect shape, with a directional arrow overlaid ONLY while in contact,
    rotated to one of exactly 3 fixed orientations matching Phi's actual
    3-class classifier output (-90/0/+90 -> left/up/right; see
    live_phiclass_predictor.py's module docstring for Phi's classifier
    definition) -- deliberately never anything in between, so this widget
    can't visually imply a directional precision the model doesn't have.

    Arrow length AND colour intensity both scale with the current Force
    (0..hsg.FORCE_GAUGE_MAX_N) -- Force is a genuine continuous regression
    output (unlike Phi), so scaling those two visual properties smoothly is
    honest about what's actually being measured, while the arrow's
    ORIENTATION stays confined to the 3 real classes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._contact = False
        self._phi = 0
        self._force = 0.0
        self.setMinimumSize(110, 110)

    def update_result(self, fr: "hsg.FingerResult | None") -> None:
        if fr is None:
            self._contact = False
            self._phi = 0
            self._force = 0.0
        else:
            self._contact = fr.contact
            self._phi = fr.phi
            self._force = fr.force
        self.update()

    def paintEvent(self, event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        side = max(min(w, h) - 16, 10)
        rect = QtCore.QRectF((w - side) / 2, (h - side) / 2, side, side)

        border = QtGui.QColor(47, 191, 47) if self._contact else QtGui.QColor(85, 85, 85)
        p.setPen(QtGui.QPen(border, 2))
        p.setBrush(QtGui.QColor(50, 50, 50))
        p.drawRoundedRect(rect, 14, 14)

        if self._contact:
            frac = 0.0 if hsg.FORCE_GAUGE_MAX_N <= 0 else max(0.0, min(1.0, self._force / hsg.FORCE_GAUGE_MAX_N))
            arrow_len = side * 0.18 + side * 0.30 * frac
            blue = int(90 + 165 * frac)
            arrow_color = QtGui.QColor(80, blue, 255)

            p.save()
            p.translate(rect.center())
            # QPainter.rotate() is clockwise-positive on screen; a shaft/
            # head drawn pointing toward -Y ("up") before rotation means
            # rotate(phi) with phi in {-90,0,+90} directly lands on
            # left/up/right -- no separate angle-mapping table needed,
            # this identity IS the -90/0/+90 -> left/up/right mapping.
            p.rotate(self._phi)
            p.setPen(QtGui.QPen(arrow_color, 3 + 2 * frac))
            p.setBrush(arrow_color)
            p.drawLine(QtCore.QPointF(0, arrow_len * 0.4), QtCore.QPointF(0, -arrow_len * 0.4))
            head = QtGui.QPolygonF([
                QtCore.QPointF(0, -arrow_len * 0.55),
                QtCore.QPointF(-arrow_len * 0.16, -arrow_len * 0.28),
                QtCore.QPointF(arrow_len * 0.16, -arrow_len * 0.28),
            ])
            p.drawPolygon(head)
            p.restore()
        p.end()


class DirectionOverviewWindow(QtWidgets.QWidget):
    """Simplified 4-finger contact-direction schematic -- a separate
    top-level window, fed the SAME poll() results as everything else (see
    GripOverviewWindow's docstring for why there's no second engine/
    connection here either).
    """

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("eFlesh Hand -- Direction / Deformation Overview")
        self.resize(560, 260)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        note = QtWidgets.QLabel(
            "Arrow direction: 3 fixed classes only (← –90° / ↑ 0° / "
            "→ +90°). Arrow length/brightness: current Force. "
            "No contact = plain grey, no arrow.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;font-size:10px;")
        root.addWidget(note)

        row = QtWidgets.QHBoxLayout()
        self._widgets: list[FingerDirectionWidget] = []
        for i in range(hsg.NUM_FINGERS):
            col = QtWidgets.QVBoxLayout()
            name_lbl = QtWidgets.QLabel(f"Finger {i + 1}")
            name_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            name_lbl.setStyleSheet("font-weight:bold;font-size:11px;")
            col.addWidget(name_lbl)
            fw = FingerDirectionWidget()
            col.addWidget(fw)
            row.addLayout(col)
            self._widgets.append(fw)
        root.addLayout(row, stretch=1)

    def apply_results(self, results: list["hsg.FingerResult"]) -> None:
        for fr in results:
            self._widgets[fr.index].update_result(fr)

    def reset(self) -> None:
        for w in self._widgets:
            w.update_result(None)


# ============================================================================
# Window 4: Force-Target Grip -- closed-loop grip controls
# ============================================================================

class ForceTargetGripWindow(QtWidgets.QWidget):
    """Closed-loop Force-Target Grip controls (Target Force, Safety Max,
    Grip to Target/Stop, per-finger live status) in their own top-level
    window, separate from the main sensing display -- so the main window's
    4 force-history charts get their full vertical space back (this used to
    be a group box stacked below the Servo Control panel, above "Additional
    Views", competing with those charts for room).

    Unlike GripOverviewWindow/DirectionOverviewWindow above, this is not a
    pure display: it's a thin UI shell over `main_window`'s existing
    control-panel widgets/logic. It owns no HandSensingEngine/servo
    connection of its own (same "exactly one live feed" invariant as those
    two) and no state machine of its own either -- both buttons call
    straight through to main_window._on_force_grip_clicked/_on_force_grip_
    stop_clicked, and that class's _force_grip_round/_force_grip_step_
    fingers state machine addresses this window's widgets directly
    (main_window._force_grip_window.target_spin, .status_lbls, etc.), the
    same way it addressed self.-owned widgets before this window existed.
    """

    def __init__(self, main_window: "UnifiedMainWindow", parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("eFlesh Hand -- Force-Target Grip")
        self.resize(420, 320)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Live LIGHT/MEDIUM/FIRM touch badge -- see _grip_level_for_force
        # and UnifiedMainWindow._update_grip_level_label, which updates
        # this every tick from whichever finger(s) are currently in
        # contact, independent of whether Grip to Target is running.
        self.grip_level_lbl = QtWidgets.QLabel("NO CONTACT")
        self.grip_level_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        level_font = self.grip_level_lbl.font()
        level_font.setPointSize(16)
        level_font.setBold(True)
        self.grip_level_lbl.setFont(level_font)
        self.grip_level_lbl.setStyleSheet("padding:6px; color:#888888;")
        root.addWidget(self.grip_level_lbl)

        params_row = QtWidgets.QHBoxLayout()
        params_row.addWidget(QtWidgets.QLabel("Target Force:"))
        self.target_spin = QtWidgets.QDoubleSpinBox()
        self.target_spin.setRange(0.1, hsg.FORCE_GAUGE_MAX_N)
        self.target_spin.setDecimals(1)
        self.target_spin.setSingleStep(0.1)
        self.target_spin.setValue(GRIP_DEFAULT_TARGET_N)
        self.target_spin.setSuffix(" N")
        self.target_spin.setToolTip(
            "Each finger stops closing as soon as ITS OWN live Force reading "
            "(the same smoothed, confirmed-contact-gated value shown on its "
            "panel -- fr.force, not the raw sensor reading) reaches this value.")
        params_row.addWidget(self.target_spin)

        params_row.addWidget(QtWidgets.QLabel("Safety Max:"))
        self.max_spin = QtWidgets.QSpinBox()
        self.max_spin.setRange(GRIP_MIN_MAX_PERCENT, 100)
        self.max_spin.setSuffix("%")
        self.max_spin.setValue(GRIP_DEFAULT_MAX_PERCENT)
        self.max_spin.setToolTip(
            "If a finger never reaches Target Force, it stops here regardless "
            "(Close-percent scale, same as each finger's own Travel spinbox) "
            "-- prevents stalling/crushing against an object the model never "
            f"registers contact with. Index/Thumb are further hard-capped at "
            f"{INDEX_THUMB_GRIP_SAFETY_CAP_PCT}% regardless of this setting, "
            f"to prevent them physically colliding with each other if force "
            f"feedback never stops them -- see INDEX_THUMB_GRIP_SAFETY_CAP_PCT.")
        params_row.addWidget(self.max_spin)
        params_row.addStretch()
        root.addLayout(params_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Grip to Target")
        self.start_btn.setToolTip(
            "Closes all 4 fingers in small steps, checking each finger's own "
            "live Force after every step -- each stops independently once it "
            "reaches Target Force (or Safety Max, whichever comes first).")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(main_window._on_force_grip_clicked)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip(
            "Stops the grip sequence immediately, leaving every finger at "
            "its current position.")
        self.stop_btn.clicked.connect(main_window._on_force_grip_stop_clicked)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self.status_lbls: dict[str, QtWidgets.QLabel] = {}
        for label, _, _, _ in FINGER_CONTROLS:
            status_lbl = QtWidgets.QLabel(f"{label}: —")
            status_lbl.setStyleSheet("color:#aaa;font-size:12px;")
            self.status_lbls[label] = status_lbl
            root.addWidget(status_lbl)

        self.overall_lbl = QtWidgets.QLabel("")
        self.overall_lbl.setStyleSheet("font-weight:bold;font-size:12px;")
        root.addWidget(self.overall_lbl)

        root.addStretch()


# ============================================================================
# Screen recorder -- captures the whole screen for the duration of a Test
# Data Recording session, so the exported evidence has a matching video, not
# just numeric CSV rows (useful for a dissertation/demo where "the hand
# actually did this on camera" matters as much as the logged Force values).
# ============================================================================

# Frame rate for the screen capture itself. Deliberately modest -- this
# grabs the whole screen and encodes a frame on the SAME GUI thread the
# sensing poll timer and Grip-to-Target step loop run on (PyQt is
# single-threaded by default), so a high rate risks stealing enough time
# per tick to introduce visible jitter in the live sensing display or the
# Grip-to-Target step cadence. 8fps is plenty to document a slow-moving
# grip-and-hold demo on camera; it is NOT meant to produce a smooth,
# high-motion recording.
SCREEN_RECORD_FPS = 8.0


class ScreenRecorder(QtCore.QObject):
    """Whole-screen capture via QScreen.grabWindow (no extra OS-level
    dependency like ffmpeg -- everything here is already a hard dependency
    of this app, PyQt6 + OpenCV) into a temporary .avi (MJPG -- chosen over
    an mp4/H264 container specifically because OpenCV's H264 write support
    depends on an optional system codec DLL that may not be present; MJPG
    is bundled with OpenCV itself and always available). The temp file is
    only given its real, human-chosen name later, when the caller moves it
    next to the exported CSV -- see UnifiedMainWindow._on_export_test_data_
    clicked.

    start()/stop() are deliberately NOT tied to their own button -- wired
    directly into the SAME Start/Stop Recording clicks that already gate
    the CSV tick buffer (see _on_record_toggle_clicked), per the explicit
    request that pressing that one button is what starts the video too.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._capture_tick)
        self._writer: cv2.VideoWriter | None = None
        self._temp_path: Path | None = None
        self._frame_count = 0
        # Tracked separately from `_writer is not None` -- the writer is
        # opened lazily on the FIRST captured frame (its real size isn't
        # known until then), so relying on _writer alone would report
        # active=False for the brief window between start() and that first
        # tick, letting a rapid double-click on Start Recording re-enter
        # start() and pointlessly discard the just-created temp file.
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if self.active:
            return
        self._active = True
        # Discard any previous temp file that was never claimed by an
        # export (e.g. the user started a new take without exporting the
        # last one) -- same "starting again clears unexported data"
        # convention _on_record_toggle_clicked already uses for the CSV
        # rows.
        self.discard_pending()
        fd, path_str = tempfile.mkstemp(suffix=".avi", prefix="eflesh_screen_rec_")
        os.close(fd)
        self._temp_path = Path(path_str)
        self._frame_count = 0
        self._writer = None  # opened lazily on the first captured frame, once its real size is known
        self._timer.start(int(1000 / SCREEN_RECORD_FPS))

    def _capture_tick(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        pixmap = screen.grabWindow(0)
        if pixmap.isNull():
            return
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGB888)
        width, height = image.width(), image.height()
        stride = image.bytesPerLine()
        ptr = image.bits()
        ptr.setsize(stride * height)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, stride)[:, :width * 3].reshape(height, width, 3)
        frame_bgr = arr[:, :, ::-1]  # RGB -> BGR for cv2

        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._writer = cv2.VideoWriter(str(self._temp_path), fourcc, SCREEN_RECORD_FPS, (width, height))
        self._writer.write(np.ascontiguousarray(frame_bgr))
        self._frame_count += 1

    def stop(self) -> Path | None:
        """Stops capture and returns the temp video path (None if nothing
        was ever captured, e.g. start() was never called or every frame
        grab failed) -- the caller (export) is responsible for moving it to
        its final name, or discard_pending() if the take is abandoned."""
        if not self.active and self._temp_path is None:
            return None
        self._active = False
        self._timer.stop()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        path = self._temp_path
        if self._frame_count == 0:
            # Nothing was actually captured (e.g. grabWindow failed every
            # tick) -- clean up the empty temp file rather than hand back a
            # useless path.
            if path is not None and path.exists():
                path.unlink()
            path = None
        return path

    def discard_pending(self) -> None:
        """Deletes any temp video not yet claimed by an export. Safe to
        call even if nothing is pending."""
        self._active = False
        self._timer.stop()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._temp_path is not None and self._temp_path.exists():
            self._temp_path.unlink()
        self._temp_path = None
        self._frame_count = 0


class TestDataRecordingWindow(QtWidgets.QWidget):
    """Test Data Recording controls (Start/Stop Recording, Export Test
    Data) in their own top-level window -- moved out of the main window's
    stacked group boxes (2026-07-21) since every group box added there
    directly shrinks the force-history charts' available vertical space
    (see UnifiedMainWindow._compute_safe_chart_height's own reasoning --
    the exact same problem ForceTargetGripWindow was already moved out for
    once before). Same thin-UI-shell pattern as that window: owns no
    recording state itself, just widgets main_window._on_record_toggle_
    clicked/_on_export_test_data_clicked/_on_tick address directly
    (main_window._record_window.record_btn, .status_lbl, .export_btn).
    """

    def __init__(self, main_window: "UnifiedMainWindow", parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle("eFlesh Hand -- Test Data Recording")
        self.resize(420, 140)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        btn_row = QtWidgets.QHBoxLayout()
        self.record_btn = QtWidgets.QPushButton("● Start Recording")
        self.record_btn.setToolTip(
            "Buffers every finger's Force/raw Force/Contact/Slip status each "
            "tick, independent of which controls you use (individual "
            "buttons, Close All, Grip to Target, ...) -- for exporting "
            "evidence of a specific test afterward (e.g. a grip-then-slip "
            "demo for a report). Starting again clears any previously "
            "buffered, unexported data.")
        self.record_btn.clicked.connect(main_window._on_record_toggle_clicked)
        btn_row.addWidget(self.record_btn)

        self.export_btn = QtWidgets.QPushButton("Export Test Data…")
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip(
            "Saves the currently buffered recording as a CSV (one row per "
            "tick, per-finger position/Force/raw Force/Contact/Slip "
            "columns) plus a companion _meta.json (session metadata, the "
            "live thresholds in effect, and a table of every detected slip "
            "event) -- ready to drop into a report.")
        self.export_btn.clicked.connect(main_window._on_export_test_data_clicked)
        btn_row.addWidget(self.export_btn)
        root.addLayout(btn_row)

        self.status_lbl = QtWidgets.QLabel("Not recording.")
        self.status_lbl.setStyleSheet("color:#999;")
        root.addWidget(self.status_lbl)

        root.addStretch()


class UnifiedMainWindow(hsg.MainWindow):
    """sensing_gui.MainWindow plus a servo control panel.

    Overrides only the connection-lifecycle hooks (to establish/tear down
    the shared servo link alongside the existing sensing connection), UI
    setup (to append the control panel widgets), and the display tick (to
    hold a just-moved finger's panel steady through its settle window --
    see _settle_until/_on_tick) -- calibration, polling, and contact/slip/
    recalibration PROCESSING (the engine.poll() call itself) are inherited
    unchanged from hsg.MainWindow; only what gets drawn from its results is
    touched.
    """

    def __init__(self, port: str | None = None, baud: int = 2_000_000):
        self._servo_connected = False
        self._finger_buttons: list[QtWidgets.QPushButton] = []
        self._all_buttons: list[QtWidgets.QPushButton] = []
        # sensing_index -> time.time() deadline until which that finger's
        # panel holds its last displayed reading instead of showing fresh
        # (still-settling) frames -- set in _schedule_post_servo_
        # recalibration, consumed in _on_tick.
        self._settle_until: dict[int, float] = {}
        # label -> QSpinBox (0-100%) -- one per finger, read at click time by
        # both that finger's own Close/Open buttons and by Close All/Open
        # All (see _make_handler/_on_close_all_clicked/_on_open_all_clicked),
        # so there's exactly one travel setting per finger no matter which
        # button drives it.
        self._finger_percent: dict[str, QtWidgets.QSpinBox] = {}
        # label -> QLabel showing that finger's last-commanded position, on
        # a 0%=open/100%=closed scale (see _update_position_label) -- kept
        # accurate whether the finger was last moved by its own Close/Open
        # buttons or by Close All/Open All, so no one control's state can go
        # stale relative to another's.
        self._finger_position_lbl: dict[str, QtWidgets.QLabel] = {}
        # Crosstalk correction (see sensing_gui.py's CROSSTALK_CORRECTION_
        # ENABLED / HandSensingEngine.set_finger_position) needs each
        # finger's CURRENT commanded position in real time. Rather than a
        # second, independent position tracker, _update_position_label below
        # (this class's existing, only source of "current position %") is
        # extended to also push into self._engine.set_finger_position() --
        # this dict itself is not read by the crosstalk correction directly
        # (the engine keeps its own copy); it exists here only as this
        # class's already-established plain-Python mirror of what the
        # QLabel text displays, for anything else in this class that might
        # want the numeric value instead of parsing label text. Initialized
        # here (before super().__init__(), same as the dicts above) since
        # _update_position_label can in principle be reached before this
        # method returns.
        self._finger_current_position: dict[str, float] = {}
        # sensing_index -> most recent FingerResult, updated every tick in
        # _on_tick regardless of that finger's settle-window display skip --
        # this is what the Force-Target Grip loop reads (fr.force) to decide
        # when to stop each finger, since it needs the engine's true current
        # reading, not just whatever the (possibly held) panel is showing.
        self._latest_results: dict[int, "hsg.FingerResult"] = {}
        # Force-Target Grip state -- see _on_force_grip_clicked/
        # _force_grip_round. label -> {"percent", "done", "sensing_index"}
        # while a grip sequence is running; empty otherwise.
        self._force_grip_active = False
        self._force_grip_state: dict[str, dict] = {}
        # Test Data Recording -- see _on_record_toggle_clicked/_on_tick/
        # _on_export_test_data_clicked. Independent of any specific control
        # feature (works whether the tester uses individual buttons, Close
        # All, or Grip to Target) -- just buffers every finger's Force/raw
        # Force/Contact/Slip status each tick while active, for exporting as
        # evidence of a specific test afterward (e.g. a grip-then-slip demo
        # for a report).
        self._recording_active = False
        self._recording_rows: list[dict] = []
        self._recording_start_time: float | None = None
        self._recording_start_iso: str | None = None
        self._pending_video_path: Path | None = None
        super().__init__(port=port, baud=baud)
        self.setWindowTitle(WINDOW_TITLE)
        # Screen recorder -- started/stopped by the SAME Start/Stop
        # Recording clicks as the CSV tick buffer above (see
        # _on_record_toggle_clicked); the resulting temp video is moved
        # next to the exported CSV in _on_export_test_data_clicked. See
        # ScreenRecorder's own docstring. Constructed AFTER super().__init__
        # (not alongside the plain-Python fields above) because it's a
        # QObject that takes `self` as its Qt parent -- Qt requires the
        # base widget's own __init__ to have already run before it can be
        # used as another QObject's parent.
        self._screen_recorder = ScreenRecorder(self)

    # ------------------------------------------------------------------
    # UI setup -- appends the control panel after the base class builds
    # everything else, without touching how that base UI is constructed.
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        super()._setup_ui()
        self._hide_event_log()
        self._hide_total_label()
        self._hide_diagnostics()
        root = self.centralWidget().layout()

        control_box = QtWidgets.QGroupBox("Servo Control")
        control_layout = QtWidgets.QVBoxLayout(control_box)

        speed_row = QtWidgets.QHBoxLayout()
        speed_row.addWidget(QtWidgets.QLabel("Speed:"))
        self._speed_combo = QtWidgets.QComboBox()
        for value, text in SPEED_OPTIONS:
            self._speed_combo.addItem(text, value)
        self._speed_combo.setCurrentIndex(
            next(i for i, (v, _) in enumerate(SPEED_OPTIONS) if v == hc.DEFAULT_SPEED))
        self._speed_combo.setToolTip(
            "Servo speed for Close/Open moves. Shared by every finger's own "
            "Close/Open buttons and Close All/Open All -- all read this "
            "same value at click time.")
        self._speed_combo.setMaximumWidth(140)
        speed_row.addWidget(self._speed_combo)
        speed_row.addStretch()
        control_layout.addLayout(speed_row)

        for label, enable_fn, finger, sensing_index in FINGER_CONTROLS:
            row = QtWidgets.QHBoxLayout()
            name_lbl = QtWidgets.QLabel(label)
            name_lbl.setMinimumWidth(60)
            name_lbl.setStyleSheet("font-weight:bold;")
            row.addWidget(name_lbl)

            enable_btn = QtWidgets.QPushButton("Enable")
            enable_btn.clicked.connect(
                self._make_handler(label, "Enable", enable_fn, sensing_index))
            row.addWidget(enable_btn)

            # Physical Close/Open functions for this finger, swapped for
            # Index only -- see _physical_close_open for why.
            btn_close_fn, btn_open_fn = self._physical_close_open(label, finger)

            # Travel spinbox: scales how far THIS finger's Close/Open moves
            # (see hand_controller.Finger.close/open's `percent` param), read
            # fresh at click time (not baked in here) so changing it takes
            # effect on the very next click. Used only by this finger's own
            # Close/Open buttons below -- Close All/Open All use the shared
            # "All Fingers %" field instead (see _on_close_all_clicked/
            # _on_open_all_clicked), which overrides every finger's Travel
            # setting for those two buttons.
            percent_spin = QtWidgets.QSpinBox()
            percent_spin.setRange(0, 100)
            percent_spin.setSuffix("%")
            percent_spin.setValue(DEFAULT_TRAVEL_PERCENT.get(label, 100))
            percent_spin.setToolTip(
                f"{label} Close/Open travel, as a percentage of full range. "
                f"Lower this to reduce how far {label} moves -- e.g. to stop "
                f"Index/Thumb colliding on Close All.")
            percent_spin.setMaximumWidth(70)
            self._finger_percent[label] = percent_spin

            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(self._make_handler(
                label, "Close", btn_close_fn, sensing_index,
                percent_getter=lambda lbl=label: self._finger_percent[lbl].value()))
            row.addWidget(close_btn)

            open_btn = QtWidgets.QPushButton("Open")
            open_btn.clicked.connect(self._make_handler(
                label, "Open", btn_open_fn, sensing_index,
                percent_getter=lambda lbl=label: self._finger_percent[lbl].value()))
            row.addWidget(open_btn)

            row.addWidget(QtWidgets.QLabel("Travel:"))
            row.addWidget(percent_spin)

            # Current position readout -- 0%=open/100%=closed (see
            # _update_position_label). Unknown until this finger's first
            # commanded move, since servos may not be at a known position
            # on GUI startup.
            row.addWidget(QtWidgets.QLabel("Pos:"))
            position_lbl = QtWidgets.QLabel("—")
            position_lbl.setMinimumWidth(40)
            self._finger_position_lbl[label] = position_lbl
            row.addWidget(position_lbl)

            row.addStretch()
            control_layout.addLayout(row)
            self._finger_buttons.extend([enable_btn, close_btn, open_btn])

        all_row = QtWidgets.QHBoxLayout()
        close_all_btn = QtWidgets.QPushButton("Close All")
        close_all_btn.clicked.connect(self._on_close_all_clicked)
        all_row.addWidget(close_all_btn)

        open_all_btn = QtWidgets.QPushButton("Open All")
        open_all_btn.clicked.connect(self._on_open_all_clicked)
        all_row.addWidget(open_all_btn)

        all_row.addWidget(QtWidgets.QLabel("All Fingers %:"))
        self._all_percent_spin = QtWidgets.QSpinBox()
        self._all_percent_spin.setRange(0, 100)
        self._all_percent_spin.setSuffix("%")
        self._all_percent_spin.setValue(100)
        self._all_percent_spin.setToolTip(
            "Overrides every finger's own Travel spinbox above -- Close All "
            "/ Open All move all 4 fingers to THIS percentage together. "
            "Lower this if Index/Thumb collide when closing.")
        self._all_percent_spin.setMaximumWidth(70)
        all_row.addWidget(self._all_percent_spin)

        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setToolTip(
            "Opens all 4 fingers fully (100% open), one finger at a time -- "
            "always 100% regardless of the All Fingers % field above, and "
            "resets every finger's own Travel spinbox back to 100% too.")
        reset_btn.clicked.connect(self._on_reset_clicked)
        all_row.addWidget(reset_btn)

        go_to_positions_btn = QtWidgets.QPushButton("Go to Custom Positions")
        go_to_positions_btn.setToolTip(
            "Closes all 4 fingers simultaneously, each to ITS OWN Travel% "
            "spinbox above (not the shared All Fingers % field) -- same as "
            "clicking every finger's own Close button at once.")
        go_to_positions_btn.clicked.connect(self._on_go_to_custom_positions_clicked)
        all_row.addWidget(go_to_positions_btn)

        all_row.addStretch()
        control_layout.addLayout(all_row)
        self._all_buttons.extend([close_all_btn, open_all_btn, reset_btn, go_to_positions_btn])

        root.addWidget(control_box)
        self._set_control_buttons_enabled(False)

        # Window 2/3/4 (see GripOverviewWindow/DirectionOverviewWindow/
        # ForceTargetGripWindow above) -- separate top-level windows, not
        # tabs, so the existing main window's own layout/functionality
        # (including the 4 force-history charts) is untouched; just buttons
        # to open/raise them. Created here (parent=self) rather than in
        # __init__, since _setup_ui only runs once self is already a fully
        # constructed QMainWindow (see hsg.MainWindow.__init__'s own
        # super().__init__() -> ... -> self._setup_ui() ordering) -- passing
        # self as their Qt parent this way gives them normal Qt-managed
        # lifetime (auto-deleted with the main window) while
        # QtCore.Qt.WindowType.Window (set in their own __init__) still
        # makes each render as an independent top-level window, not an
        # embedded child widget.
        self._grip_window = GripOverviewWindow(parent=self)
        self._direction_window = DirectionOverviewWindow(parent=self)
        # ForceTargetGripWindow additionally takes `self` as main_window (see
        # its own docstring) -- unlike the two views above, it's not a pure
        # display: its buttons call straight through to THIS class's
        # _on_force_grip_clicked/_on_force_grip_stop_clicked, and this
        # class's _force_grip_round/_force_grip_step_fingers state machine
        # addresses its widgets directly (self._force_grip_window.target_
        # spin, .status_lbls, etc.) the same way it addressed self.-owned
        # widgets before this window existed.
        self._force_grip_window = ForceTargetGripWindow(self, parent=self)
        self._all_buttons.append(self._force_grip_window.start_btn)
        # TestDataRecordingWindow: a separate top-level window rather than a
        # stacked group box, for the same reason as ForceTargetGripWindow
        # above -- see that class's own docstring.
        self._record_window = TestDataRecordingWindow(self, parent=self)

        views_box = QtWidgets.QGroupBox("Additional Views")
        views_layout = QtWidgets.QHBoxLayout(views_box)
        grip_view_btn = QtWidgets.QPushButton("Grip Overview")
        grip_view_btn.setToolTip(
            "Opens the simplified grasp/slip demo view (Grip Status, "
            "per-finger contact dots, combined force trend, grip/slip "
            "event log) in a separate window.")
        grip_view_btn.clicked.connect(self._open_grip_window)
        views_layout.addWidget(grip_view_btn)

        direction_view_btn = QtWidgets.QPushButton("Direction Overview")
        direction_view_btn.setToolTip(
            "Opens the simplified per-finger contact-direction schematic "
            "(3 fixed arrow orientations matching the Phi classifier's "
            "actual output) in a separate window.")
        direction_view_btn.clicked.connect(self._open_direction_window)
        views_layout.addWidget(direction_view_btn)

        force_grip_view_btn = QtWidgets.QPushButton("Force-Target Grip")
        force_grip_view_btn.setToolTip(
            "Opens the closed-loop Force-Target Grip controls (Target Force, "
            "Safety Max, Grip to Target/Stop, per-finger status) in a "
            "separate window.")
        force_grip_view_btn.clicked.connect(self._open_force_grip_window)
        views_layout.addWidget(force_grip_view_btn)

        record_view_btn = QtWidgets.QPushButton("Test Data Recording")
        record_view_btn.setToolTip(
            "Opens Start/Stop Recording + Export Test Data controls in a "
            "separate window.")
        record_view_btn.clicked.connect(self._open_record_window)
        views_layout.addWidget(record_view_btn)

        views_layout.addStretch()

        # Reset Baseline -- placed after the stretch above (bottom-right
        # corner of the window, the free space noted for it) rather than a
        # new row, since the two view buttons don't fill this row's width.
        # See _on_reset_baseline_clicked/HandSensingEngine.
        # start_manual_recalibration for what this actually does.
        self._reset_status_lbl = QtWidgets.QLabel("")
        self._reset_status_lbl.setStyleSheet("color:#999;font-size:10px;")
        views_layout.addWidget(self._reset_status_lbl)

        # Default (unchecked) behaviour is still the safe, exclusion-based
        # one described in _on_reset_baseline_clicked's own docstring --
        # this only OVERRIDES that when deliberately checked, for cases
        # where the operator knows no finger is genuinely being pressed
        # despite a nonzero reading (e.g. drift/noise, not a real touch)
        # and the exclusion would otherwise block or partially skip a
        # reset that's actually fine to run.
        self._force_reset_checkbox = QtWidgets.QCheckBox("Force")
        self._force_reset_checkbox.setToolTip(
            "When checked, Reset Baseline recalibrates ALL 4 fingers "
            "unconditionally, even ones currently showing contact/force -- "
            "bypasses the normal in-contact exclusion entirely. Only use "
            "this if you're sure no finger is genuinely being pressed "
            "right now; forcing a reset against a real press bakes that "
            "press into the finger's new \"resting\" baseline until it's "
            "reset again.")
        views_layout.addWidget(self._force_reset_checkbox)

        self._reset_baseline_btn = QtWidgets.QPushButton("Reset Baseline")
        self._reset_baseline_btn.setToolTip(
            "Runs a fresh ~2s baseline capture for every finger NOT "
            "currently in contact (same capture used once at connect "
            "time), instead of waiting for the periodic automatic "
            "recalibration. Fingers currently in contact are skipped, to "
            "avoid recalibrating against a mid-press reading -- check "
            "\"Force\" to recalibrate every finger regardless.")
        self._reset_baseline_btn.setEnabled(False)  # enabled once calibrated -- see _finish_calibration
        self._reset_baseline_btn.clicked.connect(self._on_reset_baseline_clicked)
        views_layout.addWidget(self._reset_baseline_btn)

        root.addWidget(views_box)

        # Measured BEFORE enlarging the charts, at sensing_gui.py's own
        # default (proven-to-fit) chart/gauge size -- this is the "chrome"
        # height (everything except the 2 grid rows' chart contribution)
        # the safe-height computation below needs, and it must be measured
        # here (after the Servo Control panel above is already added) since
        # that panel is itself part of the chrome being accounted for.
        baseline_min = root.minimumSize()
        chart_h = self._compute_safe_chart_height(baseline_min.height())
        self._enlarge_sensing_panels(chart_h)

        # Deliberately NOT re-reading root.minimumSize() here to get the
        # enlarged total -- testing found PyQt6 caches a QLayout's
        # minimumSize() from its first read and does not reliably
        # invalidate that cache on a later child setMinimumSize() in this
        # nested-layout structure (content_row -> FingerPanel's own layout
        # -> grid -> grid_w -> root), even with explicit invalidate()/
        # updateGeometry() calls -- re-reading it here silently produced
        # the SAME (stale, pre-enlarge) height every time, which is exactly
        # what caused charts to be clipped on any screen where the safe
        # height had to shrink below BASELINE_CHART_HEIGHT. Instead the
        # final height is derived the same way _compute_safe_chart_height
        # itself reasons about it: the 2 grid rows move together by
        # 2 * (chart_h - BASELINE_CHART_HEIGHT) from the already-measured
        # (and correct, since it's the very first read) baseline. Width is
        # unaffected by a height-only change, so baseline_min's width is
        # still accurate for it.
        min_h = baseline_min.height() + 2 * (chart_h - BASELINE_CHART_HEIGHT)
        min_w = max(WINDOW_MIN_W, baseline_min.width())
        self.setMinimumSize(min_w, min_h)
        self.resize(min_w, min_h)

    def _compute_safe_chart_height(self, baseline_min_height: int) -> int:
        """Returns the tallest chart/gauge height (capped at
        FINGER_CHART_MAX_HEIGHT, floored at MIN_CHART_HEIGHT) that still
        lets the window's total minimum height fit the ACTUAL screen's
        available height, computed fresh at startup instead of assumed --
        see FINGER_CHART_MAX_HEIGHT for why a static guess kept being
        wrong on a real screen. Can shrink charts BELOW sensing_gui.
        py's own BASELINE_CHART_HEIGHT default if the window's other
        chrome alone doesn't leave room even for that -- see
        BASELINE_CHART_HEIGHT/MIN_CHART_HEIGHT for why that's necessary,
        not just a fallback for some unrealistic edge case.

        baseline_min_height is the window's minimum height already measured
        with charts at BASELINE_CHART_HEIGHT -- i.e. everything the window
        needs at that known reference chart size. Since the finger panels
        sit in a 2-row grid, growing (or shrinking) each chart/gauge by
        `delta` px changes the window's total minimum height by `2 * delta`
        (both rows move together, all 4 panels being identical), so solving
        `available == baseline_min_height + 2 * delta` for delta gives the
        exact chart height that makes the window fit precisely.
        """
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return BASELINE_CHART_HEIGHT
        available_h = screen.availableGeometry().height() - SCREEN_CHROME_MARGIN_PX
        delta = (available_h - baseline_min_height) / 2
        chart_h = BASELINE_CHART_HEIGHT + delta
        return int(max(MIN_CHART_HEIGHT, min(chart_h, FINGER_CHART_MAX_HEIGHT)))

    def _enlarge_sensing_panels(self, chart_height: int) -> None:
        """Grows each finger panel's force-history chart/gauge (created by
        the just-run base-class _setup_ui()) to `chart_height` tall (see
        _compute_safe_chart_height for how that's picked) -- widths are
        unaffected by the vertical-fit problem this solves, so they stay at
        their own fixed constants. Reaches into FingerPanel's _chart/_gauge
        widget instances directly rather than adding a resize hook to
        sensing_gui.py, so that module's own layout/behaviour stays
        completely unchanged for its standalone use.
        """
        for panel in self._panels:
            panel._chart.setMinimumSize(FINGER_CHART_MIN_WIDTH, chart_height)
            panel._gauge.setMinimumSize(FINGER_GAUGE_MIN_WIDTH, chart_height)

    def _hide_event_log(self) -> None:
        """Hides sensing_gui.MainWindow's "Event Log" panel to reclaim
        the vertical space the enlarged force-history charts need (see
        FINGER_CHART_MAX_HEIGHT) -- reached via self._log (the QPlainTextEdit
        sensing_gui.py stores on self; its immediate parent is the
        Event Log QGroupBox) rather than by editing that module, so the
        standalone sensing tool keeps its own Event Log untouched.
        _append_log still writes into the (now hidden) widget and still
        prints every line to the console, so slip/recalibration messages
        remain visible there even with the on-screen panel hidden.
        """
        log_box = self._log.parentWidget()
        if log_box is not None:
            log_box.hide()

    def _hide_total_label(self) -> None:
        """Hides sensing_gui.MainWindow's "Total Grip Force" label
        (self._total_lbl) at the top of the window -- per-request, this
        window shows only each finger's own reading, no combined total.
        Left in place (not removed) since hsg.MainWindow's _on_connected/
        _reset_connection_ui still write to it; hiding it here, the same
        way _hide_event_log hides the Event Log panel, keeps that
        base-class code completely unchanged. This module's own _on_tick
        override (see below) also no longer computes or writes to it.
        """
        self._total_lbl.hide()

    def _hide_diagnostics(self) -> None:
        """Hides sensing_gui.MainWindow's "Diagnostics (raw Contact toggle
        rate / hysteresis streaks / glitches)" panel (self._diag_lbl) --
        per-request, not used by this tool. Reached via
        self._diag_lbl.parentWidget() (its QGroupBox), same "hide, don't
        edit sensing_gui.py" pattern as _hide_event_log/_hide_total_label
        above -- _update_diagnostics still runs and updates the (now
        hidden) label every tick, so nothing about the underlying
        computation changes, just what's drawn.
        """
        diag_box = self._diag_lbl.parentWidget()
        if diag_box is not None:
            diag_box.hide()

    def _set_control_buttons_enabled(self, enabled: bool) -> None:
        for btn in self._finger_buttons + self._all_buttons:
            btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Window 2/3 -- open/raise the separate Grip Overview and Direction
    # Overview windows (see their class docstrings above). Both keep
    # updating via _on_tick's apply_results() calls below regardless of
    # visibility, so whatever they show is already current the moment
    # they're raised, not stale from before they were last opened.
    # ------------------------------------------------------------------

    def _open_grip_window(self) -> None:
        self._grip_window.show()
        self._grip_window.raise_()
        self._grip_window.activateWindow()

    def _open_direction_window(self) -> None:
        self._direction_window.show()
        self._direction_window.raise_()
        self._direction_window.activateWindow()

    def _open_record_window(self) -> None:
        self._record_window.show()
        self._record_window.raise_()
        self._record_window.activateWindow()

    # ------------------------------------------------------------------
    # Connection lifecycle -- establishes/tears down the shared servo link
    # alongside the base class's existing sensing connection handling.
    # ------------------------------------------------------------------

    def _on_connected(self, port: str) -> None:
        super()._on_connected(port)
        try:
            hc.connect(self._engine.ctrl)
            self._servo_connected = True
            self._set_control_buttons_enabled(True)
            self._append_log("Servo control link established (shared connection).")
        except Exception as exc:
            self._servo_connected = False
            self._append_log(f"Servo control link failed: {exc}")

    def _on_disconnect_clicked(self) -> None:
        super()._on_disconnect_clicked()
        self._servo_connected = False
        self._set_control_buttons_enabled(False)
        self._reset_baseline_btn.setEnabled(False)
        self._abort_force_grip("disconnected")
        self._grip_window.reset()
        self._direction_window.reset()

    def _on_disconnected(self) -> None:
        super()._on_disconnected()
        self._servo_connected = False
        self._set_control_buttons_enabled(False)
        self._reset_baseline_btn.setEnabled(False)
        self._abort_force_grip("disconnected")
        self._grip_window.reset()
        self._direction_window.reset()

    def _on_error(self, msg: str) -> None:
        super()._on_error(msg)
        self._servo_connected = False
        self._set_control_buttons_enabled(False)
        self._reset_baseline_btn.setEnabled(False)
        self._abort_force_grip("connection error")
        self._grip_window.reset()
        self._direction_window.reset()

    def _finish_calibration(self) -> None:
        super()._finish_calibration()
        self._reset_baseline_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Reset Baseline -- on-demand fresh baseline capture (see
    # HandSensingEngine.start_manual_recalibration), reusing the same
    # ~BASELINE_DURATION_S capture the one-time connect-time calibration
    # runs, rather than waiting on the periodic automatic recalibration.
    # ------------------------------------------------------------------

    def _on_reset_baseline_clicked(self) -> None:
        # A finger currently confirmed-in-contact is normally excluded from
        # this capture (see start_manual_recalibration's own docstring for
        # why: averaging its samples over this window would bake a
        # mid-press reading into its "resting" baseline) -- UNLESS the
        # "Force" checkbox is checked, which recalibrates every finger
        # unconditionally, in-contact or not (see that checkbox's own
        # tooltip for when that's actually appropriate). Read live from
        # self._engine.finger_states, not from any single displayed
        # FingerResult, since that's the same state _process_one_finger
        # itself is driven by.
        in_contact = {i for i, fs in enumerate(self._engine.finger_states) if fs.confirmed_contact}
        forced = self._force_reset_checkbox.isChecked()
        excluded = set() if forced else set(in_contact)

        if not forced and len(excluded) >= hsg.NUM_FINGERS:
            msg = ("Reset Baseline: release all fingers first (all 4 currently in "
                   "contact), or check \"Force\" to override.")
            self._reset_status_lbl.setText(msg)
            self._append_log(msg)
            return

        self._reset_baseline_btn.setEnabled(False)
        self._reset_status_lbl.setText("Recalibrating baseline…")
        # Only meaningful when forced (otherwise these fingers are simply
        # excluded, not force-included) -- carried through as a plain
        # closure variable, not returned by start_manual_recalibration
        # itself, since the engine has no concept of "forced" at all, only
        # exclude_indices.
        forced_in_contact = in_contact if forced else set()
        self._engine.start_manual_recalibration(
            exclude_indices=excluded,
            on_done=lambda counts, excl: self._on_manual_recal_done(counts, excl, forced_in_contact))

    def _on_manual_recal_done(self, counts: list[int], excluded: set[int],
                               forced_in_contact: "set[int]" = frozenset()) -> None:
        self._reset_baseline_btn.setEnabled(True)
        n_done = hsg.NUM_FINGERS - len(excluded)
        # Same "weak" threshold the one-time connect-time calibration
        # already flags a low sample count against (see hsg.MainWindow.
        # _finish_calibration) -- only checked for fingers this call
        # actually recalibrated, since an excluded finger's count is
        # meaningless (its samples were captured but never used).
        weak = [i for i in range(hsg.NUM_FINGERS)
                if i not in excluded and counts[i] < hsg.MIN_BASELINE_SAMPLES]

        if excluded:
            names = ", ".join(f"Finger {i + 1}" for i in sorted(excluded))
            msg = (f"Baseline reset — {n_done} of {hsg.NUM_FINGERS} fingers "
                   f"recalibrated ({names} in contact, skipped).")
        elif forced_in_contact:
            names = ", ".join(f"Finger {i + 1}" for i in sorted(forced_in_contact))
            msg = (f"Baseline reset — {hsg.NUM_FINGERS} fingers recalibrated "
                   f"(forced; {names} were in contact).")
        else:
            msg = f"Baseline reset — {hsg.NUM_FINGERS} fingers recalibrated."
        if weak:
            names = ", ".join(f"Finger {i + 1}" for i in sorted(weak))
            msg += f" Warning: weak sample count for {names}."
        self._reset_status_lbl.setText(msg)
        # Same confirmation mechanism every other recalibration/servo-move
        # status line in this window already uses (see
        # _schedule_post_servo_recalibration) -- printed to console via the
        # (hidden) Event Log widget's own _append_log, not a new one.
        self._append_log(msg)

    # ------------------------------------------------------------------
    # Display tick -- overrides hsg.MainWindow._on_tick to hold a just-
    # moved finger's panel steady through its settle window instead of
    # redrawing it with (still-settling) frames. See _settle_until /
    # _schedule_post_servo_recalibration.
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        """Identical to hsg.MainWindow._on_tick, except a finger currently
        inside its post-move settle window is skipped when handing results
        to FingerPanel.apply_result -- its panel simply keeps showing
        whatever it last displayed instead of this tick's frame.

        self._engine.poll() is still called exactly as in the base class,
        so hysteresis, slip detection, and recalibration sampling all keep
        running unaffected for every finger, including the settling one --
        only what gets DRAWN is held back.
        """
        results = self._engine.poll()
        if results is None:
            return
        now = time.time()
        total = 0.0
        for fr in results:
            # Always recorded, independent of the settle-window display skip
            # below -- the Force-Target Grip loop needs the engine's true
            # current fr.force even for a finger whose PANEL is holding a
            # stale frame right now (see self._latest_results' own comment).
            self._latest_results[fr.index] = fr
            if now >= self._settle_until.get(fr.index, 0.0):
                self._panels[fr.index].apply_result(fr)
            total += fr.force
            if fr.slip_message:
                self._append_log(fr.slip_message)
            if fr.recal_message:
                self._append_log(fr.recal_message)

        self._total_lbl.setText(
            f"Total Grip Force: {total:.2f} N   (avg {total / hsg.NUM_FINGERS:.2f} N/finger)")
        self._update_diagnostics(results)
        self._update_grip_level_label(results)

        # Window 2/3 get the FULL, unfiltered results every tick -- not
        # gated by the settle-window skip above, which only holds back
        # what the MAIN panels draw right after a servo move. Neither of
        # these windows was specced to inherit that main-window-specific
        # display choice, and "share the same live data feed" is most
        # directly honoured by handing them exactly what poll() returned.
        self._grip_window.apply_results(results)
        self._direction_window.apply_results(results)

        # Test Data Recording -- see _on_record_toggle_clicked/
        # _on_export_test_data_clicked. Uses the FULL, unfiltered results
        # (same rationale as the two windows above), not gated by the
        # settle-window display skip, so a recorded row always reflects
        # this tick's true engine output regardless of what any panel is
        # currently showing.
        if self._recording_active:
            row = {
                "t": round(now - self._recording_start_time, 3),
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            }
            for fr in results:
                label = hsg.CROSSTALK_FINGER_LABELS[fr.index]
                row[f"{label}_position_pct"] = self._finger_current_position.get(label)
                row[f"{label}_force"] = fr.force
                row[f"{label}_raw_force"] = fr.raw_force
                row[f"{label}_contact"] = fr.contact
                row[f"{label}_slip_active"] = fr.slip_active
                row[f"{label}_slip_event"] = fr.slip_event
                row[f"{label}_slip_message"] = fr.slip_message or ""
            self._recording_rows.append(row)
            self._record_window.status_lbl.setText(
                f"Recording… {len(self._recording_rows)} ticks buffered.")

    # ------------------------------------------------------------------
    # Test Data Recording / Export -- see __init__'s _recording_* fields
    # and _on_tick's buffering block above.
    # ------------------------------------------------------------------

    def _on_record_toggle_clicked(self) -> None:
        if not self._recording_active:
            self._recording_active = True
            self._recording_rows = []
            self._recording_start_time = time.time()
            self._recording_start_iso = datetime.now(timezone.utc).isoformat()
            self._screen_recorder.start()
            self._record_window.record_btn.setText("■ Stop Recording")
            self._record_window.status_lbl.setText("Recording… 0 ticks buffered. (screen recording started)")
            self._record_window.export_btn.setEnabled(False)
            self._append_log("Test data recording: started (screen recording started).")
        else:
            self._recording_active = False
            self._pending_video_path = self._screen_recorder.stop()
            self._record_window.record_btn.setText("● Start Recording")
            n = len(self._recording_rows)
            video_note = " + screen video" if self._pending_video_path is not None else " (screen video FAILED -- see log)"
            self._record_window.status_lbl.setText(
                (f"Stopped — {n} ticks buffered{video_note}, ready to export." if n
                 else "Stopped — nothing buffered.")
            )
            self._record_window.export_btn.setEnabled(n > 0)
            self._append_log(
                f"Test data recording: stopped ({n} ticks, "
                f"{'screen video captured' if self._pending_video_path is not None else 'NO screen video -- capture failed or produced 0 frames'}).")

    def _on_export_test_data_clicked(self) -> None:
        """Saves the buffered recording as a CSV (one row per tick, wide
        per-finger columns -- directly loadable into Excel/pandas for a
        report chart) plus a companion `<name>_meta.json` sidecar: session
        metadata, the live Contact/Slip thresholds actually in effect
        during this recording (read from sensing_gui.py's own module
        constants at export time, so it can never go stale relative to
        whatever was really live), and a pre-extracted table of every
        detected slip event (finger, time, force, message) -- the single
        most report-ready piece of evidence that a slip was correctly
        flagged.
        """
        if not self._recording_rows:
            self._append_log("Export Test Data: nothing recorded.")
            return

        default_name = f"eflesh_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Test Data", default_name, "CSV files (*.csv)")
        if not path_str:
            return
        csv_path = Path(path_str)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")
        meta_path = csv_path.with_name(csv_path.stem + "_meta.json")

        fieldnames = ["t", "timestamp_iso"]
        for label in hsg.CROSSTALK_FINGER_LABELS:
            fieldnames += [f"{label}_position_pct", f"{label}_force", f"{label}_raw_force",
                           f"{label}_contact", f"{label}_slip_active", f"{label}_slip_event",
                           f"{label}_slip_message"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._recording_rows)

        slip_events = []
        for row in self._recording_rows:
            for label in hsg.CROSSTALK_FINGER_LABELS:
                if row.get(f"{label}_slip_event"):
                    slip_events.append({
                        "t": row["t"], "timestamp_iso": row["timestamp_iso"], "finger": label,
                        "force_at_event": row[f"{label}_force"],
                        "message": row[f"{label}_slip_message"],
                    })

        # Screen video, if the recorder captured one -- moved (not copied)
        # next to the CSV, same stem, so the pair is trivially findable
        # later (light_rep1.csv / light_rep1_meta.json / light_rep1.avi).
        # None if the recorder never produced a usable file (e.g. it failed
        # to grab any frames) -- exporting the CSV/meta must still succeed
        # in that case, so this is logged, not raised.
        video_path: Path | None = None
        if self._pending_video_path is not None and self._pending_video_path.exists():
            video_path = csv_path.with_suffix(".avi")
            shutil.move(str(self._pending_video_path), str(video_path))
        self._pending_video_path = None

        meta = {
            "exported_at_iso": datetime.now(timezone.utc).isoformat(),
            "recording_started_at_iso": self._recording_start_iso,
            "n_ticks": len(self._recording_rows),
            "video_file": video_path.name if video_path is not None else None,
            "model_path": str(hsg.MODEL_PATH),
            "live_thresholds": {
                "CONTACT_P_CONFIRM": hsg.CONTACT_P_CONFIRM,
                "CONTACT_P_RELEASE": hsg.CONTACT_P_RELEASE,
                "CONTACT_CONFIRM_FRAMES": hsg.CONTACT_CONFIRM_FRAMES,
                "CONTACT_RELEASE_FRAMES": hsg.CONTACT_RELEASE_FRAMES,
                "SLIP_DROP_FRACTION": hsg.SLIP_DROP_FRACTION,
                "SLIP_WINDOW_S": hsg.SLIP_WINDOW_S,
                "SLIP_DISPLAY_HOLD_S": hsg.SLIP_DISPLAY_HOLD_S,
            },
            "n_slip_events": len(slip_events),
            "slip_events": slip_events,
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        video_log_note = f", {video_path.name}" if video_path is not None else " (NO video -- screen capture failed)"
        self._append_log(
            f"Test data exported: {csv_path.name} ({len(self._recording_rows)} ticks), "
            f"{meta_path.name} ({len(slip_events)} slip events){video_log_note}.")
        self._record_window.status_lbl.setText(
            f"Exported {len(self._recording_rows)} ticks → {csv_path.name}"
            + (f" (+ {video_path.name})" if video_path is not None else ""))

    # ------------------------------------------------------------------
    # Per-finger button handlers
    # ------------------------------------------------------------------

    def _make_handler(self, label: str, action: str, fn, sensing_index: int,
                       percent_getter=None):
        def handler():
            if not self._servo_connected:
                self._append_log(f"{label} {action}: ignored (servo link not connected).")
                return
            # percent_getter (and the Speed dropdown) are read HERE, at click
            # time, not captured when the button was built -- so turning the
            # Travel spinbox or Speed dropdown up/down takes effect on the
            # very next click.
            percent = percent_getter() if percent_getter is not None else None
            try:
                if percent is not None:
                    fn(percent=percent, speed=self._current_speed())
                else:
                    fn()
            except Exception as exc:
                self._append_log(f"{label} {action}: error -- {exc}")
            if percent is not None:
                self._update_position_label(label, action, percent)
            self._schedule_post_servo_recalibration(label, sensing_index)
        return handler

    def _current_speed(self) -> int:
        return self._speed_combo.currentData()

    def _physical_close_open(self, label: str, finger):
        """Returns (physical_close_fn, physical_open_fn) for this finger's
        Finger object. Index's wiring is intentionally swapped
        (Close -> finger.open, Open -> finger.close): hand_controller's
        Index motion math is the original, unreversed form (see
        `index = Finger(...)` in hand_controller.py, no reverse=True), under
        which finger.close()/finger.open() physically move Index opposite to
        their names. Swapping here -- for Index only -- makes the buttons'
        (and Close All/Open All's) labels match Index's actual physical
        motion without touching Middle/Ring/Thumb, whose close/open already
        match their labels correctly."""
        if label == "Index":
            return finger.open, finger.close
        return finger.close, finger.open

    def _update_position_label(self, label: str, action: str, percent: int) -> None:
        """Updates `label`'s "current position %" readout after a Close/Open
        move, on the same 0%=open/100%=closed scale regardless of whether
        the move came from this finger's own Close/Open buttons or Close
        All/Open All -- see POSITION_LABEL_NEUTRAL for the scale itself.
        `action` is the LOGICAL action ("Close"/"Open" -- what the user
        asked for), not which physical finger.close/open function actually
        ran, so Index's swapped wiring (see _physical_close_open) doesn't
        flip its displayed position.

        Also the single place this class pushes a finger's position into
        the crosstalk correction (self._engine.set_finger_position) -- see
        that method's docstring and CROSSTALK_CORRECTION_ENABLED in
        sensing_gui.py. Reusing this existing method (rather than adding a
        second call site per caller) means every path that already updates
        the "current position %" label -- single-finger Close/Open,
        Close All/Open All, and Reset -- keeps the crosstalk correction in
        sync automatically, with nothing to remember to update separately.
        """
        if action == "Close":
            position = POSITION_LABEL_NEUTRAL + percent / 2.0
        elif action == "Open":
            position = POSITION_LABEL_NEUTRAL - percent / 2.0
        else:
            return
        self._finger_position_lbl[label].setText(f"{position:.0f}%")
        self._finger_current_position[label] = position
        self._engine.set_finger_position(label, position)

    def _schedule_post_servo_recalibration(self, label: str, sensing_index: int) -> None:
        """See SERVO_SETTLE_DELAY_MS / HandSensingEngine.request_recalibration.
        Hardware testing showed Force noise/drift after a commanded servo
        move, the same baseline-drift mechanism sensing_gui.py's
        periodic recalibration already targets for human-touch disturbance
        -- this just makes a finger eligible for that recalibration right
        away instead of waiting out the rest of RECALIBRATION_INTERVAL_S
        (commit still requires a genuine stable no-contact window, so this
        cannot recalibrate mid-contact or off a movement that hasn't
        settled yet).

        Requests recalibration for ONLY the sensing finger the moved servo
        finger corresponds to (see FINGER_CONTROLS' sensing_index / the
        confirmed Index=Finger1, Middle=Finger2, Ring=Finger3, Thumb=Finger4
        mapping) -- not all 4, since the other 3 fingers' mux groups are
        unaffected by this move.

        Also starts this finger's settle window (see _settle_until/_on_tick)
        immediately -- i.e. from the moment the move command was sent, not
        from when this delayed recalibration fires -- so its panel holds
        its last (pre-move) reading rather than displaying the movement's
        own mechanical noise for the next PANEL_SETTLE_DELAY_MS (shorter
        than, and independent of, the SERVO_SETTLE_DELAY_MS the
        recalibration request below still waits out -- see
        PANEL_SETTLE_DELAY_MS's own comment for why these were split).
        """
        self._settle_until[sensing_index] = time.time() + PANEL_SETTLE_DELAY_MS / 1000.0
        QtCore.QTimer.singleShot(
            SERVO_SETTLE_DELAY_MS,
            lambda: (self._engine.request_recalibration([sensing_index]),
                     self._append_log(
                         f"{label}: requested post-move baseline recalibration "
                         f"(Finger {sensing_index + 1}).")))

    # ------------------------------------------------------------------
    # Close All / Open All -- all 4 fingers' commands sent back-to-back
    # (sleep_after=False on each, one settle sleep at the end) rather than
    # staggered ALL_SEQUENCE_DELAY_MS apart. The old staggered dispatch was
    # a real, provable cause of a visible staircase-instead-of-parallel
    # close (earlier-dispatched fingers got a head start every step) --
    # live-tested 2026-09-03 (verify_simultaneous_close_fix_live.py, see
    # verified-ground-truth-artifact-critical memory family) confirmed
    # genuinely-simultaneous dispatch works fine on this hardware, including
    # under sustained repeated-step use (record_genuine_grasp_dataset.py's
    # whole approach loop). No longer chained via QTimer.singleShot between
    # fingers -- each finger's write is just a few serial bytes now that the
    # artificial per-finger sleep is gone, so looping over all 4 here is
    # still effectively instant and doesn't freeze the GUI thread.
    # ------------------------------------------------------------------

    def _build_all_fingers_steps(self, action: str, percent: int, speed: int
                                  ) -> list[tuple[str, object, int, str, int]]:
        """Builds a _run_all_sequence steps list that drives all 4 fingers
        in the same logical direction (`action`: "Close"/"Open") to the
        SAME `percent` -- the shared "All Fingers %" field (see
        _on_close_all_clicked/_on_open_all_clicked), which overrides each
        finger's own Travel spinbox for Close All/Open All. `percent` and
        `speed` are read once, at sequence-build time (same as a single
        click), and baked into every finger's step via the `p=`/`s=`
        default-argument capture, so the field/dropdown changing mid-way
        can't retroactively alter an already-queued move. Handles Index's
        swapped physical wiring via _physical_close_open, same as every
        other entry point into Close/Open.
        """
        steps = []
        for label, _, finger, sensing_index in FINGER_CONTROLS:
            physical_close, physical_open = self._physical_close_open(label, finger)
            fn = physical_close if action == "Close" else physical_open
            steps.append((label,
                           (lambda f=fn, p=percent, s=speed: f(percent=p, speed=s, sleep_after=False)),
                           sensing_index, action, percent))
        return steps

    def _build_custom_positions_steps(self, speed: int
                                       ) -> list[tuple[str, object, int, str, int]]:
        """Builds a _run_all_sequence steps list that closes each finger to
        ITS OWN Travel spinbox value (self._finger_percent[label]) --
        unlike _build_all_fingers_steps above (one shared percent for
        every finger, from the "All Fingers %" field), this reads each
        finger's own already-set Travel% instead, same value its
        individual Close button would use. Always the Close direction
        (matching each finger's own Close button's interpretation of its
        Travel value), read at click time same as every other entry point
        here.
        """
        steps = []
        for label, _, finger, sensing_index in FINGER_CONTROLS:
            physical_close, _ = self._physical_close_open(label, finger)
            percent = self._finger_percent[label].value()
            steps.append((label,
                           (lambda f=physical_close, p=percent, s=speed: f(percent=p, speed=s, sleep_after=False)),
                           sensing_index, "Close", percent))
        return steps

    def _on_go_to_custom_positions_clicked(self) -> None:
        steps = self._build_custom_positions_steps(self._current_speed())
        self._run_all_sequence(steps, "Go to Custom Positions")

    def _on_close_all_clicked(self) -> None:
        steps = self._build_all_fingers_steps(
            "Close", self._all_percent_spin.value(), self._current_speed())
        self._run_all_sequence(steps, "Close All")

    def _on_open_all_clicked(self) -> None:
        steps = self._build_all_fingers_steps(
            "Open", self._all_percent_spin.value(), self._current_speed())
        self._run_all_sequence(steps, "Open All")

    def _on_reset_clicked(self) -> None:
        # Hardcoded 100, not self._all_percent_spin.value() -- Reset always
        # opens fully regardless of whatever the All Fingers % field is
        # currently set to (see reset_btn's tooltip). Also resets every
        # finger's own Travel spinbox to 100% -- Index/Thumb default to 70%
        # (see DEFAULT_TRAVEL_PERCENT, to avoid a fresh session's Close All
        # slamming them into each other), which would otherwise keep reading
        # "70%" after Reset even though this always commands a full 100%
        # open, same as Middle/Ring -- misleading since it'd look like only
        # those two fingers opened partway.
        for percent_spin in self._finger_percent.values():
            percent_spin.setValue(100)
        self._all_percent_spin.setValue(100)
        steps = self._build_all_fingers_steps("Open", 100, self._current_speed())
        self._run_all_sequence(steps, "Reset (all fingers fully open)")

    def _run_all_sequence(self, steps: list[tuple[str, object, int, str, int]], seq_label: str) -> None:
        if not self._servo_connected:
            self._append_log(f"{seq_label}: ignored (servo link not connected).")
            return
        self._set_control_buttons_enabled(False)
        self._append_log(f"{seq_label}: starting.")
        for label, fn, sensing_index, action, percent in steps:
            try:
                fn()
            except Exception as exc:
                self._append_log(f"{seq_label}: {label} error -- {exc}")
            self._update_position_label(label, action, percent)
            self._schedule_post_servo_recalibration(label, sensing_index)
        time.sleep(0.01)
        self._set_control_buttons_enabled(True)
        self._append_log(f"{seq_label}: done.")

    # ------------------------------------------------------------------
    # Force-Target Grip -- closed-loop grip driven by live Force instead of
    # a fixed percentage. Reuses the existing per-finger position control
    # (Finger.close via _physical_close_open) and the existing Force
    # pipeline (self._latest_results, fed from self._engine.poll() every
    # tick) -- no changes to sensing_gui.py or the model itself.
    # ------------------------------------------------------------------

    def _on_force_grip_clicked(self) -> None:
        """Starts the closed-loop grip. Target Force / Safety Max are read
        once here, not re-read mid-sequence -- same convention as
        _build_all_fingers_steps' percent/speed capture, so adjusting either
        spinbox mid-grip can't retroactively alter an already-running
        sequence."""
        if not self._servo_connected:
            self._append_log("Grip to Target: ignored (servo link not connected).")
            return
        if self._force_grip_active:
            return

        target_n = self._force_grip_window.target_spin.value()
        max_percent = self._force_grip_window.max_spin.value()
        speed = self._current_speed()

        self._force_grip_active = True
        self._force_grip_state = {
            label: {"percent": 0, "done": False, "sensing_index": sensing_index}
            for label, _, _, sensing_index in FINGER_CONTROLS
        }
        self._set_control_buttons_enabled(False)
        self._force_grip_window.stop_btn.setEnabled(True)
        self._force_grip_window.overall_lbl.setText("Gripping…")
        for label in self._force_grip_state:
            self._force_grip_window.status_lbls[label].setText(f"{label}: 0% — starting…")
        self._append_log(
            f"Grip to Target: starting -- target {target_n:.1f} N, safety max {max_percent}%.")
        self._force_grip_round(target_n, max_percent, speed)

    def _on_force_grip_stop_clicked(self) -> None:
        if not self._force_grip_active:
            return
        for label, st in self._force_grip_state.items():
            if not st["done"]:
                st["done"] = True
                self._force_grip_window.status_lbls[label].setText(
                    f"{label}: {st['percent']}% — stopped (user)")
        self._append_log("Grip to Target: stopped by user.")
        self._finish_force_grip("Grip stopped.")

    def _force_grip_round(self, target_n: float, max_percent: int, speed: int) -> None:
        """One check-then-step round over every finger not yet done.

        A finger already at/above target_n -- including one that was
        already touching something before this round's first move -- is
        marked done WITHOUT ever sending it a command, so it never
        physically jumps. Fingers still short of target and below
        max_percent are handed to _force_grip_step_fingers, which now sends
        all of them back-to-back (sleep_after=False on each, one settle
        sleep at the end) instead of staggering them apart -- see
        _run_all_sequence's comment for why the old per-finger stagger was
        removed (2026-09-03 staircase-closing fix). The NEXT round only
        starts GRIP_STEP_SETTLE_MS after this round's last command, so its
        Force checks read settled values rather than this step's own
        transit noise.
        """
        if not self._force_grip_active:
            return  # aborted (Stop / disconnect) -- see _abort_force_grip

        to_step = []
        for label, st in self._force_grip_state.items():
            if st["done"]:
                continue
            effective_max = _grip_max_percent_for(label, max_percent)
            fr = self._latest_results.get(st["sensing_index"])
            force = fr.force if fr is not None else 0.0
            if force >= target_n:
                st["done"] = True
                self._force_grip_window.status_lbls[label].setText(
                    f"{label}: {st['percent']}% ({force:.1f}N) — target reached, stopped")
                self._append_log(
                    f"Grip to Target: {label} reached target ({force:.1f}N) at {st['percent']}%.")
            elif st["percent"] >= effective_max:
                st["done"] = True
                self._force_grip_window.status_lbls[label].setText(
                    f"{label}: {st['percent']}% ({force:.1f}N) — safety max reached, stopped")
                self._append_log(
                    f"Grip to Target: {label} hit safety max ({effective_max}%) without "
                    f"reaching target (Force {force:.1f}N).")
            else:
                to_step.append((label, st, force))

        if not to_step:
            self._finish_force_grip("Grip complete.")
            return

        self._force_grip_step_fingers(to_step, target_n, max_percent, speed)

    def _force_grip_step_fingers(self, to_step: list, target_n: float,
                                  max_percent: int, speed: int) -> None:
        """Sends every finger in `to_step`'s command back-to-back
        (sleep_after=False on each, one settle sleep at the end -- see
        _run_all_sequence's comment), then waits GRIP_STEP_SETTLE_MS and
        starts the next round."""
        if not self._force_grip_active:
            return

        for label, st, force in to_step:
            st["percent"] = min(_grip_max_percent_for(label, max_percent), st["percent"] + GRIP_STEP_PERCENT)
            finger, sensing_index = _FINGER_CONTROLS_BY_LABEL[label]
            physical_close_fn, _ = self._physical_close_open(label, finger)
            try:
                physical_close_fn(percent=st["percent"], speed=speed, sleep_after=False)
            except Exception as exc:
                self._append_log(f"Grip to Target: {label} error -- {exc}")
            self._update_position_label(label, "Close", st["percent"])
            self._schedule_post_servo_recalibration(label, sensing_index)
            self._force_grip_window.status_lbls[label].setText(
                f"{label}: {st['percent']}% ({force:.1f}N)... still closing")
        time.sleep(0.01)

        QtCore.QTimer.singleShot(
            GRIP_STEP_SETTLE_MS, lambda: self._force_grip_round(target_n, max_percent, speed))

    def _finish_force_grip(self, message: str) -> None:
        self._force_grip_active = False
        self._force_grip_window.stop_btn.setEnabled(False)
        self._set_control_buttons_enabled(True)
        self._force_grip_window.overall_lbl.setText(message)
        self._append_log(f"Grip to Target: {message}")

    def _abort_force_grip(self, reason: str) -> None:
        """Called from the disconnect/error handlers -- stops the sequence
        without touching _set_control_buttons_enabled (the caller already
        handles that for every other control) so this can't re-enable
        buttons a disconnect just disabled."""
        if not self._force_grip_active:
            return
        self._force_grip_active = False
        self._force_grip_window.stop_btn.setEnabled(False)
        for label, st in self._force_grip_state.items():
            if not st["done"]:
                self._force_grip_window.status_lbls[label].setText(
                    f"{label}: {st['percent']}% — aborted ({reason})")
        self._force_grip_window.overall_lbl.setText(f"Grip aborted ({reason}).")
        self._append_log(f"Grip to Target: aborted -- {reason}.")

    def _open_force_grip_window(self) -> None:
        self._force_grip_window.show()
        self._force_grip_window.raise_()
        self._force_grip_window.activateWindow()

    def _update_grip_level_label(self, results: list["hsg.FingerResult"]) -> None:
        """Live LIGHT/MEDIUM/FIRM touch badge in the Force-Target Grip
        window -- runs every tick regardless of whether Grip to Target is
        active, so it reflects a manual press just as well as a closed-loop
        grip. Uses the strongest currently-contacting finger (not a sum/
        average across all 4), since the badge is meant to answer "how hard
        is whatever's being gripped right now being squeezed," not total
        force across possibly-unrelated fingers."""
        touching = [fr for fr in results if fr.contact]
        peak_force = max((fr.force for fr in touching), default=0.0)
        level, color = _grip_level_for_force(peak_force, contact=bool(touching))
        text = level if level == "NO CONTACT" else f"{level} touch  ({peak_force:.2f} N)"
        self._force_grip_window.grip_level_lbl.setText(text)
        self._force_grip_window.grip_level_lbl.setStyleSheet(f"padding:6px; color:{color};")

    # ------------------------------------------------------------------
    # Cleanup -- closes Window 2/3/4 alongside the main window so the whole
    # app actually exits (an orphaned top-level window would otherwise
    # keep QApplication.exec() running) rather than leaving them open with
    # no more data arriving. hsg.MainWindow.closeEvent already stops the
    # poll timer/engine; that happens via super().closeEvent(event) below.
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._grip_window.close()
        self._direction_window.close()
        self._force_grip_window.close()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="eFlesh unified hand sensing + servo control demo")
    parser.add_argument("--port", help="Teensy COM port (auto-discovers if omitted)")
    parser.add_argument("--baud", type=int, default=2_000_000, help="Serial baud rate")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(hsg._dark_palette())

    window = UnifiedMainWindow(port=args.port, baud=args.baud)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
