"""Sensing reader -- thin wrapper around sensing_gui.HandSensingEngine.

Exposes ONLY "give me the latest per-finger Force/Theta/Phi/Contact" (plus
the raw FingerResult list, for the GUI to reuse sensing_gui.FingerPanel's
existing display unmodified -- see its own docstring for why that's not a
"display logic" violation). No knowledge of the load cell or hand-tracking
readers, no saving logic. Imports sensing_gui.py unmodified; does not edit
it, does not subclass its MainWindow (this project's other tools already
do that for a full standalone GUI -- this dashboard wraps the ENGINE only).
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_V1_DIR = SCRIPT_DIR.parent
if str(_MODEL_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_V1_DIR))

from PyQt6 import QtCore

import sensing_gui as sg


class SensingReader(QtCore.QObject):
    """Call poll() exactly ONCE per tick (SessionRecorder's job) --
    HandSensingEngine.poll() returns None on any second call within the
    same incoming frame, so a second caller would silently get nothing.
    Anyone else wanting the latest data should call latest() /
    latest_raw_results() instead, which are always populated with
    whatever poll() last saw."""

    connected = QtCore.pyqtSignal(str)
    disconnected = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._engine = sg.HandSensingEngine(self)
        self._engine.connected.connect(self.connected)
        self._engine.disconnected.connect(self.disconnected)
        self._engine.error.connect(self.error)
        self._engine.status.connect(self.status)
        self._latest: dict[str, dict] = {}
        self._latest_raw: list["sg.FingerResult"] | None = None

    def connect_hardware(self, port: str | None = None, baud: int = 2_000_000) -> None:
        self._engine.start(port=port, baud=baud)

    def finish_calibration(self) -> list[int]:
        """Call once, BASELINE_DURATION_S after `connected` fires -- same
        timing sensing_gui.MainWindow itself uses (see its
        _update_calibration_countdown/_finish_calibration)."""
        return self._engine.finish_calibration()

    def set_finger_position(self, label: str, position_percent: float) -> None:
        """Passthrough to HandSensingEngine.set_finger_position -- MUST be
        called by whatever actually moves a finger's servo, every time it
        does, or the engine's crosstalk correction silently keeps believing
        every finger is still at its last-told position (0% at startup)
        forever. hand_control_gui.py and teleop_data_collection.py already
        do this on their own move paths; this dashboard drives servos
        directly (move_all() in dashboard_gui.py) and needs the same call
        wired in on that path -- see that module's _move_all_and_sync()."""
        self._engine.set_finger_position(label, position_percent)

    @property
    def baseline_duration_s(self) -> float:
        return sg.BASELINE_DURATION_S

    def reset_baseline(self, on_done=None) -> None:
        """On-demand full baseline reset -- same underlying mechanism as
        hand_control_gui.py's own "Reset Baseline" button
        (HandSensingEngine.start_manual_recalibration), reused unmodified
        rather than reimplemented. No exclude_indices: this dashboard has
        no notion of "currently in contact, don't recalibrate against
        this finger" the way the interactive Grip-to-Target flow does, so
        every finger is reset -- callers should ask the user to make sure
        nothing is touching any finger before calling this, same as the
        connect-time calibration already requires."""
        self._engine.start_manual_recalibration(on_done=on_done)

    def _get_latest_frame(self):
        """No public accessor exists on HandSensingEngine for the most
        recent raw frame (poll() consumes it internally and returns only
        processed FingerResults) -- reads the same _latest_frame/_frame_lock
        poll() itself uses, read-only, rather than modifying sensing_gui.py
        to expose it. Same pattern already established by
        teleop_data_collection.py's own _get_latest_frame()."""
        with self._engine._frame_lock:
            return self._engine._latest_frame

    def poll(self) -> dict[str, dict] | None:
        """Returns {label: {"force","raw_force","theta","phi","contact",
        "raw_contact","raw_channels"}} for the newest frame, or None if
        nothing new arrived since the last call.

        "force" is GATED to 0.0 whenever fstate.confirmed_contact is False
        (sensing_gui.py's _process_one_finger, ~line 1813) -- for a finger
        whose Contact classifier rarely/never confirms (Index -- see the
        index-contact-threshold-not-fixable memory: its p_contact peaks
        around 2.2e-28 even during a real grasp, a known training-data gap,
        not something this dashboard's own fixes can touch), "force" reads
        a flat 0.0 for the ENTIRE session regardless of real contact. This
        was traced live during the 2026-09-07 pilot re-run: Index still
        read 0.0 after the set_finger_position fix, load cell nonzero.
        "raw_force" (fr.raw_force) is the model's true, UNGATED regression
        output -- never zeroed by the contact hysteresis -- included here
        so this dashboard's whole point (an independent load-cell ground
        truth to calibrate/fit the eFlesh estimate against) isn't defeated
        for exactly the finger that needs it most. "raw_contact" is the
        classifier's raw pre-hysteresis 0/1 output, same reasoning.
        "raw_channels" is that finger's 15 raw S1X..S5Z magnetometer
        readings straight off this tick's frame (sg.frame_to_raw_values),
        BEFORE the trained model turns them into Force/Theta/Phi at all --
        the actual sensor data, for a from-scratch fit against the load
        cell rather than one that inherits whatever bias/gap already
        exists in the deployed model's own regression (see raw_force's own
        limits, e.g. Index's approach-phase self-motion artifact found in
        the 2026-09-07 pilot)."""
        results = self._engine.poll()
        if results is None:
            return None
        frame = self._get_latest_frame()
        raw_cols = self._engine.predictor.raw_cols
        out = {}
        for fr in results:
            label = sg.CROSSTALK_FINGER_LABELS[fr.index]
            raw_channels = (sg.frame_to_raw_values(frame, raw_cols, fr.index)
                             if frame is not None else {})
            out[label] = {
                "force": fr.force, "raw_force": fr.raw_force,
                "theta": fr.theta, "phi": fr.phi,
                "contact": fr.contact, "raw_contact": fr.raw_contact,
                "raw_channels": raw_channels,
            }
        self._latest = out
        self._latest_raw = results
        return out

    def latest(self) -> dict[str, dict]:
        """Latest known per-finger Force/Theta/Phi/Contact, regardless of
        whether THIS tick had a new frame. What SessionRecorder logs."""
        return self._latest

    def latest_raw_results(self) -> list["sg.FingerResult"] | None:
        """Full FingerResult objects from the last poll() -- for the GUI's
        reused FingerPanel.apply_result(), which needs more than the 4
        summary fields (glitch/slip/history/etc). Not used for logging."""
        return self._latest_raw
