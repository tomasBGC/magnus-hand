"""Grasping research dashboard: combines two independent, read-only data
sources -- the eFlesh sensing pipeline (Force/Theta/Phi/Contact) and an
independent LoadScale load cell -- into one timestamped log per episode,
plus a human-operated "genuine contact" ground-truth marker, and an
automated scenario runner for the load-cell grasp training protocol.

Why this exists: this project has repeatedly found its own AUTOMATED
"verified genuine contact" signals unreliable in ways that took real
hardware time to discover -- most acutely, the 2026-09-02 verification-
artifact incident, where a noisy raw_force estimate alone could fabricate
"verified: True" with zero actual object contact (see the
verified-ground-truth-artifact-critical memory). A human pressing and
holding a button ONLY while they genuinely believe contact is occurring is
a ground-truth signal independent of every sensor/model this project has
had trouble trusting on its own -- and the load cell is a second,
physically independent force measurement, the same role it played for the
original true_zero_dataset before the genuine-grasp protocol swapped that
out for a self-referential check (also flagged in that same memory as part
of what made this class of bug possible in the first place).

2026-09-04 rework (load-cell grasp training protocol, no camera): the
camera/hand-tracking half of this dashboard (Connect Camera, calibrate,
"Drive hand from camera", the video preview) has been REMOVED from this
window -- camera tracking was confirmed jittery enough to compromise the
precise, repeatable finger positioning this protocol needs (see the
tracking-jitter investigation earlier this project). hand_tracking_reader.py,
video_stream_reader.py and model_v1/hand_tracking/hand_tracker_bridge.py are
all left untouched on disk (none of this is deleted, just unused by this
window) in case camera-driven teleop is wanted again for a different tool.
The manual per-finger slider panel added the same day was ALSO removed
shortly after (explicit request -- "I wont use it") -- every finger move
now happens exclusively through the Scenario Runner below.

The Scenario Runner automates one grasp rep at a time: close only the
scenario's grasping fingers in small steps, verify against the LOAD CELL
(not the eFlesh model's own force estimate -- see the module docstring
above for why that self-referential pattern is exactly what this dashboard
exists to avoid) using the same sustained-consecutive-ticks debounce this
project settled on for exactly this kind of noisy-estimator problem, hold,
release. The load cell cannot tell WHICH fingers are touching it (confirmed
2026-09-03) -- scenario correctness is a protocol-discipline property (only
the scenario's own fingers ever move) enforced by the runner's own finger
targeting, not something the runner detects after the fact.

2026-09-04 save-gate (explicit request): a finished rep is NOT auto-
committed to the dataset. It sits as a pending result -- reviewed by the
human, same "don't trust the automated signal alone" principle as the
genuine-contact button -- until "Save Rep" is clicked, which copies the raw
JSONL into training_protocol/confirmed/<scenario>/<level>/ and appends one
row to training_protocol/master_tracker.xlsx (see training_tracker.py).
Running another rep without saving silently discards the pending one (the
raw log still exists in training_protocol/ for debugging, it's just never
copied into confirmed/ or counted in the tracker).

Architecture (see individual reader files for detail on each piece):
  - sensing_reader.SensingReader: wraps sensing_gui.HandSensingEngine.
  - load_cell_reader.LoadCellReader: wraps unified_controller.
    UnifiedController's LoadScale-specific methods. Reports Newtons.
  - session_recorder.SessionRecorder: the only place polling/combining/
    logging logic lives -- writes every rep's raw log into training_dir
    below regardless of whether it later gets saved.
  - training_tracker.py: the master_tracker.xlsx read/write logic.
  This file (dashboard_gui.py) only builds widgets, starts/stops readers on
  button clicks, drives the scenario runner's step timer, and redraws from
  row_ready.

None of sensing_gui.py or unified_controller.py are modified -- only
imported from.

Usage:
    python model_v1/grasping_dashboard/dashboard_gui.py
"""
from __future__ import annotations

import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_V1_DIR = SCRIPT_DIR.parent
if str(_MODEL_V1_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_V1_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

_CROSSTALK_DIR = _MODEL_V1_DIR / "crosstalk"
if str(_CROSSTALK_DIR) not in sys.path:
    sys.path.insert(0, str(_CROSSTALK_DIR))

from PyQt6 import QtCore, QtWidgets

import sensing_gui as sg
from AmazingHand import hand_controller as hc
from record_crosstalk import FINGER_DEFS, move_to_position
from sensing_reader import SensingReader
from load_cell_reader import LoadCellReader
from session_recorder import SessionRecorder
import training_tracker

OUT_DIR = SCRIPT_DIR.parent / "grasping_dashboard_data"

# --- Dedicated folder for this protocol (2026-09-04, explicit request) ----
# Everything the load-cell training protocol produces lives under one
# folder: SessionRecorder's raw per-rep JSONL + session meta (every rep run,
# saved or not -- kept for debugging), confirmed/ (only rows the human
# reviewed and explicitly saved, organized by scenario/level), and the
# master Excel tracker. Nothing from this protocol is written directly into
# OUT_DIR itself any more.
TRAINING_DIR = OUT_DIR / "training_protocol"
CONFIRMED_DIR = TRAINING_DIR / "confirmed"
MASTER_XLSX = TRAINING_DIR / "master_tracker.xlsx"

FINGER_BY_LABEL = {entry[0]: entry for entry in FINGER_DEFS}
REFERENCE = {label: 0.0 for label in sg.CROSSTALK_FINGER_LABELS}

# Index+Thumb physical collision risk -- same value/rationale as
# teleop_data_collection.py's own INDEX_THUMB_COLLISION_CAP_PCT (a real
# object is the expected use case here). Duplicated per this codebase's
# established convention of each control tool owning its own copy.
INDEX_THUMB_COLLISION_CAP_PCT = 70.0

# --- Scenario runner (2026-09-04, load-cell grasp training protocol) ------
# Finger sets + per-scenario closure caps copied from
# true_zero_dataset/record_genuine_grasp_dataset.py's own SCENARIOS (same
# physical hand, same collision considerations -- see that file's own
# comments for why index_thumb/all4/*_thumb-triples get 70% and the rest
# get a more conservative 60%). Only the 10 multi-finger combinations that
# are physically graspable are listed; the load cell replaces the eFlesh
# per-finger raw_force this dict's sibling copy verifies against there.
SCENARIOS: dict[str, dict] = {
    "index_thumb": {"grasping": ["Index", "Thumb"], "close_cap_pct": 70.0},
    "index_middle": {"grasping": ["Index", "Middle"], "close_cap_pct": 70.0},
    "all4": {"grasping": ["Index", "Middle", "Ring", "Thumb"], "close_cap_pct": 70.0},
    "index_ring": {"grasping": ["Index", "Ring"], "close_cap_pct": 60.0},
    "middle_thumb": {"grasping": ["Middle", "Thumb"], "close_cap_pct": 60.0},
    "middle_ring": {"grasping": ["Middle", "Ring"], "close_cap_pct": 60.0},
    "ring_thumb": {"grasping": ["Ring", "Thumb"], "close_cap_pct": 60.0},
    "index_middle_thumb": {"grasping": ["Index", "Middle", "Thumb"], "close_cap_pct": 70.0},
    "index_ring_thumb": {"grasping": ["Index", "Ring", "Thumb"], "close_cap_pct": 70.0},
    "middle_ring_thumb": {"grasping": ["Middle", "Ring", "Thumb"], "close_cap_pct": 60.0},
    "index_middle_ring": {"grasping": ["Index", "Middle", "Ring"], "close_cap_pct": 60.0},
}

# Starting defaults for the load cell's OWN calibration -- deliberately NOT
# copied from record_genuine_grasp_dataset.py's eFlesh-side FORCE_LEVELS
# (those were repeatedly de-rated to work around THAT signal's own closure-
# cap/servo behaviour, e.g. firm 3.0->1.7N -- constraints specific to that
# estimator, not to this physically independent load cell). All three are
# editable in the UI: this is a brand-new piece of hardware with no live-
# tested limits yet, and this project's own history (see that same file's
# comments) is that force-level defaults need retuning after real reps.
# firm: 3.0 -> 2.0N (2026-09-07, explicit decision after index_thumb data
# showed 3.0N was physically unreachable even holding steady -- 3 clean
# reps all capped ~2.1-2.4N. Matches the same fix already made for the
# same reason in the other (eFlesh-side) genuine-grasp protocol.
DEFAULT_FORCE_LEVELS_N: dict[str, float] = {"light": 0.5, "medium": 1.5, "firm": 2.0}

# Per-level fraction of each scenario's own close_cap_pct -- how far the
# grasping fingers are allowed to travel before giving up
# (cap_reached_no_force). 2026-09-07: originally shipped as
# {"light": 0.5, "medium": 0.75, "firm": 1.0} on the (wrong) assumption
# that a lighter target needs proportionally less TRAVEL to reach the
# object, same as it needs less FORCE once touching it. Live-disproven the
# same day: light's 35% cap (index_thumb's 70% base x 0.5) never reached
# the load cell at all (cap_reached_no_force, peak_hold ~0N) -- reaching
# the object is a fixed physical distance independent of target force;
# only force accumulation after contact scales with level. All fractions
# reset to 1.0 (every level gets the scenario's full cap, matching pre-
# 2026-09-07 behaviour) until real reach-distance data across levels
# justifies differentiating them again.
LEVEL_CAP_FRACTION: dict[str, float] = {"light": 1.0, "medium": 1.0, "firm": 1.0}

GRIP_STEP_PERCENT = 3
GRIP_STEP_SETTLE_S = 0.5
HOLD_S = 8.0
RELEASE_SETTLE_S = 1.0
# Same sustained-consecutive-ticks debounce as record_genuine_grasp_dataset.
# py's _force_sustained (see verified-ground-truth-artifact-critical memory
# for why a single-sample check is unsafe against a noisy estimator) --
# applied here to the load cell's reading via SessionRecorder's own tick
# rate (sg.DISPLAY_INTERVAL_S, ~20Hz), fed through _on_row_ready below
# rather than re-polling the hardware a second time per tick.
SUSTAIN_DURATION_S = 0.75
POLL_INTERVAL_S = sg.DISPLAY_INTERVAL_S
SUSTAINED_TICKS = max(1, round(SUSTAIN_DURATION_S / POLL_INTERVAL_S))
NO_CONTACT_FORCE_N_THRESHOLD = 0.15
# 5 reps/combo (slightly more than record_genuine_grasp_dataset.py's own
# --reps-per-level default of 4) -- this is a fresh dataset with the load
# cell as sole verification and a per-rep human save-gate, so a slightly
# larger per-cell minimum is cheap insurance for later statistics. Purely a
# UI default/progress target either way -- the runner never auto-stops a
# session, the human decides when to move to the next combo.
DEFAULT_REPS_PER_COMBO = 5


def clamp_for_collision_safety(target: dict[str, float]) -> tuple[dict[str, float], str | None]:
    """Clamps ONLY Index+Thumb down to the cap when both would exceed it
    together, instead of refusing the whole multi-finger command -- used by
    the scenario runner as defense-in-depth (every SCENARIOS close_cap_pct
    already sits at or below this cap, so this should never actually fire
    in normal operation)."""
    clamped = dict(target)
    ix, th = clamped.get("Index", 0.0), clamped.get("Thumb", 0.0)
    if ix > INDEX_THUMB_COLLISION_CAP_PCT and th > INDEX_THUMB_COLLISION_CAP_PCT:
        clamped["Index"] = INDEX_THUMB_COLLISION_CAP_PCT
        clamped["Thumb"] = INDEX_THUMB_COLLISION_CAP_PCT
        return clamped, (f"Index/Thumb clamped to {INDEX_THUMB_COLLISION_CAP_PCT:.0f}% "
                          f"(would collide at {ix:.0f}%/{th:.0f}%)")
    return clamped, None


def move_all(target: dict[str, float], speed: int) -> None:
    """All 4 fingers' servo commands dispatched back-to-back (sleep_after=
    False), one settle sleep at the end -- NOT four independent single-
    finger calls, which reintroduces the staircase-instead-of-parallel
    close bug root-caused and fixed earlier this same project (see
    staircase-close-fix memory) for exactly this "drive several fingers
    at once" situation. The only way any finger moves in this dashboard --
    used by the scenario runner and by the "Open All Fingers" reset."""
    for label in sg.CROSSTALK_FINGER_LABELS:
        move_to_position(FINGER_BY_LABEL[label], target.get(label, 0.0), speed, sleep_after=False)
    QtCore.QThread.msleep(10)


class MiniFingerBar(QtWidgets.QWidget):
    """Compact per-finger force bar: name + a small progress bar (scaled
    to MAX_DISPLAY_FORCE_N) whose colour reflects Contact. Deliberately NOT
    the big styled sensing_gui.FingerPanel -- see module docstring."""

    MAX_DISPLAY_FORCE_N = 5.0

    def __init__(self, label: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(4)
        name_lbl = QtWidgets.QLabel(label)
        name_lbl.setFixedWidth(42)
        name_lbl.setStyleSheet("font-size:10px;font-weight:bold;")
        layout.addWidget(name_lbl)
        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setFixedHeight(14)
        self._bar.setTextVisible(True)
        self._bar.setFormat("-- N")
        layout.addWidget(self._bar, stretch=1)

    def update_from_result(self, fr: "sg.FingerResult") -> None:
        pct = max(0, min(100, round(fr.force / self.MAX_DISPLAY_FORCE_N * 100)))
        self._bar.setValue(pct)
        self._bar.setFormat(f"{fr.force:.2f} N")
        chunk_color = "#4c4" if fr.contact else "#579"
        self._bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {chunk_color}; }}")


class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grasping Research Dashboard")
        self.setMinimumSize(880, 760)

        self._sensing = SensingReader(self)
        self._sensing.connected.connect(self._on_sensing_connected)
        self._sensing.disconnected.connect(self._on_sensing_disconnected)
        self._sensing.error.connect(self._on_sensing_error)
        self._sensing.status.connect(lambda s: self._append_log(f"Sensing: {s}"))

        self._load_cell = LoadCellReader(data_dir=TRAINING_DIR)

        self._servo_connected = False
        self._commanded_position: dict[str, float] = dict(REFERENCE)

        self._recorder = SessionRecorder(self._sensing, self._load_cell, None, TRAINING_DIR, self)
        self._recorder.row_ready.connect(self._on_row_ready)

        training_tracker.ensure_workbook(MASTER_XLSX, list(SCENARIOS.keys()), list(DEFAULT_FORCE_LEVELS_N.keys()))
        self._confirmed_counts: dict[tuple[str, str], int] = training_tracker.load_confirmed_counts(MASTER_XLSX)
        self._orientation_counts: dict[tuple[str, str], dict[str, int]] = (
            training_tracker.load_orientation_counts(MASTER_XLSX))
        self._pending_rep: dict | None = None

        self._calib_timer = QtCore.QTimer(self)
        self._calib_timer.setSingleShot(True)
        self._calib_timer.timeout.connect(self._finish_calibration)

        self._tick_timer = QtCore.QTimer(self)
        self._tick_timer.timeout.connect(self._recorder.tick)

        # --- Scenario runner state ----------------------------------------
        self._runner_active = False
        self._runner_stop_requested = False
        self._runner_scenario = ""
        self._runner_level = ""
        self._runner_target_n: float | None = None
        self._runner_close_cap_pct = 0.0
        self._runner_orientation = ""
        self._runner_is_control = False
        self._runner_percent = 0.0
        self._runner_phase = ""  # "approach" | "hold" | "release"
        self._runner_stop_reason = ""
        self._runner_hold_deadline = 0.0
        self._runner_release_deadline = 0.0
        self._runner_load_by_phase: dict[str, list[float]] = {"approach": [], "hold": []}
        self._runner_episode_id: int | None = None
        self._runner_step_timer = QtCore.QTimer(self)
        self._runner_step_timer.timeout.connect(self._runner_step)

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # --- Connections row (one-click, no manual ports) -------------------------
        conn_box = QtWidgets.QGroupBox("Data Sources")
        conn_layout = QtWidgets.QHBoxLayout(conn_box)

        sensing_col = QtWidgets.QVBoxLayout()
        sensing_col.addWidget(QtWidgets.QLabel("Sensing (eFlesh):"))
        self._sensing_port_edit = QtWidgets.QLineEdit()
        self._sensing_port_edit.setPlaceholderText("blank = auto-discover")
        sensing_col.addWidget(self._sensing_port_edit)
        self._sensing_connect_btn = QtWidgets.QPushButton("Connect")
        self._sensing_connect_btn.clicked.connect(self._on_sensing_connect_clicked)
        sensing_col.addWidget(self._sensing_connect_btn)
        self._reset_baseline_btn = QtWidgets.QPushButton("Reset Baseline / Force")
        self._reset_baseline_btn.setToolTip(
            "Make sure nothing is touching any finger, then click this -- same "
            "on-demand recalibration as hand_control_gui.py's own Reset Baseline.")
        self._reset_baseline_btn.setEnabled(False)
        self._reset_baseline_btn.clicked.connect(self._on_reset_baseline_clicked)
        sensing_col.addWidget(self._reset_baseline_btn)
        self._sensing_status_lbl = QtWidgets.QLabel("Not connected.")
        self._sensing_status_lbl.setStyleSheet("color:#ccc;font-size:10px;")
        self._sensing_status_lbl.setWordWrap(True)
        sensing_col.addWidget(self._sensing_status_lbl)
        conn_layout.addLayout(sensing_col, stretch=1)

        load_col = QtWidgets.QVBoxLayout()
        load_col.addWidget(QtWidgets.QLabel("Load cell:"))
        self._load_port_edit = QtWidgets.QLineEdit()
        self._load_port_edit.setPlaceholderText("blank = auto-discover")
        load_col.addWidget(self._load_port_edit)
        self._load_connect_btn = QtWidgets.QPushButton("Connect")
        self._load_connect_btn.clicked.connect(self._on_load_connect_clicked)
        load_col.addWidget(self._load_connect_btn)
        tare_row = QtWidgets.QHBoxLayout()
        self._load_force_lbl = QtWidgets.QLabel("Load: -- N")
        self._load_force_lbl.setStyleSheet("font-size:13px;font-weight:bold;color:#fc5;")
        tare_row.addWidget(self._load_force_lbl, stretch=1)
        self._tare_btn = QtWidgets.QPushButton("Tare")
        self._tare_btn.setMaximumWidth(60)
        self._tare_btn.clicked.connect(self._on_tare_clicked)
        tare_row.addWidget(self._tare_btn)
        load_col.addLayout(tare_row)
        self._load_status_lbl = QtWidgets.QLabel("Not connected.")
        self._load_status_lbl.setStyleSheet("color:#ccc;font-size:10px;")
        self._load_status_lbl.setWordWrap(True)
        load_col.addWidget(self._load_status_lbl)
        conn_layout.addLayout(load_col, stretch=1)

        root.addWidget(conn_box)

        # --- Mini finger force bars -----------------------------------------------
        bars_row = QtWidgets.QHBoxLayout()
        self._finger_bars: dict[str, MiniFingerBar] = {}
        for label in sg.CROSSTALK_FINGER_LABELS:
            bar = MiniFingerBar(label)
            self._finger_bars[label] = bar
            bars_row.addWidget(bar, stretch=1)
        root.addLayout(bars_row)

        # --- Scenario runner --------------------------------------------------
        runner_box = QtWidgets.QGroupBox("Scenario Runner (load-cell grasp training protocol)")
        runner_layout = QtWidgets.QVBoxLayout(runner_box)

        pick_row = QtWidgets.QHBoxLayout()
        pick_row.addWidget(QtWidgets.QLabel("Scenario:"))
        self._scenario_combo = QtWidgets.QComboBox()
        self._scenario_combo.addItems(list(SCENARIOS.keys()))
        self._scenario_combo.currentTextChanged.connect(lambda _: self._refresh_progress_label())
        pick_row.addWidget(self._scenario_combo, stretch=1)
        pick_row.addWidget(QtWidgets.QLabel("Level:"))
        self._level_combo = QtWidgets.QComboBox()
        self._level_combo.addItems(list(DEFAULT_FORCE_LEVELS_N.keys()))
        self._level_combo.currentTextChanged.connect(self._on_level_changed)
        pick_row.addWidget(self._level_combo, stretch=1)
        pick_row.addWidget(QtWidgets.QLabel("Target reps:"))
        self._reps_spin = QtWidgets.QSpinBox()
        self._reps_spin.setRange(1, 50)
        self._reps_spin.setValue(DEFAULT_REPS_PER_COMBO)
        self._reps_spin.valueChanged.connect(lambda _: self._refresh_progress_label())
        pick_row.addWidget(self._reps_spin)
        runner_layout.addLayout(pick_row)

        self._progress_lbl = QtWidgets.QLabel("")
        self._progress_lbl.setStyleSheet("color:#9cf;font-size:11px;font-weight:bold;")
        runner_layout.addWidget(self._progress_lbl)

        # Single target-force spinbox (2026-09-07 rework, replacing three
        # always-visible per-level spinboxes) -- driven automatically by the
        # Level combo above (see _on_level_changed), so picking Light/
        # Medium/Firm alone updates the N target with no separate step, and
        # a stray edit (e.g. the pilot session's accidental firm=5N) can't
        # silently persist across a level switch. Still editable for a
        # one-off override on the CURRENT rep only -- switching Level again
        # resets it back to the preset.
        levels_row = QtWidgets.QHBoxLayout()
        levels_row.addWidget(QtWidgets.QLabel("Target force (N):"))
        self._target_force_spin = QtWidgets.QDoubleSpinBox()
        self._target_force_spin.setRange(0.05, 20.0)
        self._target_force_spin.setSingleStep(0.1)
        self._target_force_spin.setDecimals(2)
        self._target_force_spin.setValue(DEFAULT_FORCE_LEVELS_N["light"])
        levels_row.addWidget(self._target_force_spin)
        levels_row.addWidget(QtWidgets.QLabel(
            "(auto-set by Level -- edit to override just this rep)"))
        levels_row.addStretch(1)
        runner_layout.addLayout(levels_row)

        object_row = QtWidgets.QHBoxLayout()
        object_row.addWidget(QtWidgets.QLabel("Object used:"))
        self._object_edit = QtWidgets.QLineEdit()
        self._object_edit.editingFinished.connect(
            lambda: self._recorder.set_object_used(self._object_edit.text()))
        object_row.addWidget(self._object_edit, stretch=1)
        # Load cell orientation (2026-09-07, explicit request): the load
        # cell only registers force pressed against ONE of its two
        # sandwiched surfaces -- which finger's press it can actually see
        # depends entirely on which way round it's mounted. Logged per rep
        # (episode_meta -> every tick, and the confirmed row) so a 0N
        # reading can be told apart from "no contact" vs. "wrong side
        # pressed" after the fact -- see the pilot-run finding this came
        # out of. Not auto-detected; the human sets it to match physical
        # reality before running reps and must change it themselves if they
        # physically flip the cell.
        object_row.addWidget(QtWidgets.QLabel("Load cell facing:"))
        self._orientation_combo = QtWidgets.QComboBox()
        self._orientation_combo.addItems(list(sg.CROSSTALK_FINGER_LABELS))
        self._orientation_combo.setCurrentText("Index")  # today's pilot default
        object_row.addWidget(self._orientation_combo)
        self._control_rep_check = QtWidgets.QCheckBox("Control rep (load cell physically out of reach)")
        self._control_rep_check.setToolTip(
            "Check this ONLY once you have physically moved the load cell out of the "
            "grasping fingers' reach. The runner closes the same fingers to the same "
            "cap WITHOUT stopping early on force, and expects near-zero load -- if "
            "force is registered anyway, the rep is flagged, not silently accepted.")
        self._control_rep_check.toggled.connect(lambda _: self._refresh_progress_label())
        object_row.addWidget(self._control_rep_check)
        runner_layout.addLayout(object_row)

        run_row = QtWidgets.QHBoxLayout()
        self._run_rep_btn = QtWidgets.QPushButton("▶  Run Rep")
        self._run_rep_btn.clicked.connect(self._on_run_rep_clicked)
        run_row.addWidget(self._run_rep_btn)
        self._stop_runner_btn = QtWidgets.QPushButton("■  Stop")
        self._stop_runner_btn.setStyleSheet("color:#c44;")
        self._stop_runner_btn.setEnabled(False)
        self._stop_runner_btn.clicked.connect(self._on_stop_runner_clicked)
        run_row.addWidget(self._stop_runner_btn)
        self._open_all_btn = QtWidgets.QPushButton("⏏  Open All Fingers")
        self._open_all_btn.setEnabled(False)
        self._open_all_btn.clicked.connect(self._on_open_all_clicked)
        run_row.addWidget(self._open_all_btn)
        runner_layout.addLayout(run_row)

        self._runner_status_lbl = QtWidgets.QLabel("Idle.")
        self._runner_status_lbl.setStyleSheet("color:#9cf;font-size:11px;")
        self._runner_status_lbl.setWordWrap(True)
        runner_layout.addWidget(self._runner_status_lbl)

        save_row = QtWidgets.QHBoxLayout()
        self._save_rep_btn = QtWidgets.QPushButton("💾  Save This Rep to Master Data")
        self._save_rep_btn.setStyleSheet("font-weight:bold;color:#4c4;")
        self._save_rep_btn.setEnabled(False)
        self._save_rep_btn.clicked.connect(self._on_save_rep_clicked)
        save_row.addWidget(self._save_rep_btn)
        runner_layout.addLayout(save_row)

        self._pending_status_lbl = QtWidgets.QLabel("No pending rep.")
        self._pending_status_lbl.setStyleSheet("color:#ccc;font-size:10px;")
        self._pending_status_lbl.setWordWrap(True)
        runner_layout.addWidget(self._pending_status_lbl)

        root.addWidget(runner_box)

        # --- Manual episode controls (free-form reps outside the runner) ---------
        episode_box = QtWidgets.QGroupBox("Manual Episode (free-form, outside the runner)")
        episode_layout = QtWidgets.QVBoxLayout(episode_box)

        btn_row = QtWidgets.QHBoxLayout()
        self._start_episode_btn = QtWidgets.QPushButton("▶  Start New Episode")
        self._start_episode_btn.clicked.connect(self._on_start_episode_clicked)
        btn_row.addWidget(self._start_episode_btn)
        self._end_success_btn = QtWidgets.QPushButton("✓  End Episode (Success)")
        self._end_success_btn.setStyleSheet("color:#4c4;")
        self._end_success_btn.clicked.connect(lambda: self._on_end_episode_clicked("success"))
        self._end_success_btn.setEnabled(False)
        btn_row.addWidget(self._end_success_btn)
        self._end_failure_btn = QtWidgets.QPushButton("✗  End Episode (Failure)")
        self._end_failure_btn.setStyleSheet("color:#c44;")
        self._end_failure_btn.clicked.connect(lambda: self._on_end_episode_clicked("failure"))
        self._end_failure_btn.setEnabled(False)
        btn_row.addWidget(self._end_failure_btn)
        episode_layout.addLayout(btn_row)

        self._episode_status_lbl = QtWidgets.QLabel("No episode active.")
        self._episode_status_lbl.setStyleSheet("color:#ccc;font-size:10px;")
        episode_layout.addWidget(self._episode_status_lbl)

        # --- Genuine-contact ground-truth marker ---------------------------------
        self._contact_mark_btn = QtWidgets.QPushButton(
            "HOLD while you genuinely feel contact  (or hold SPACE)")
        self._contact_mark_btn.setStyleSheet(
            "font-size:14px;font-weight:bold;padding:14px;background:#333;color:#eee;")
        self._contact_mark_btn.pressed.connect(lambda: self._recorder.set_genuine_contact_marked(True))
        self._contact_mark_btn.released.connect(lambda: self._recorder.set_genuine_contact_marked(False))
        episode_layout.addWidget(self._contact_mark_btn)

        root.addWidget(episode_box)

        log_box = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(80)
        log_layout.addWidget(self._log)
        root.addWidget(log_box)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._refresh_progress_label()

    def _append_log(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    # ------------------------------------------------------------------
    # Spacebar as an alternative genuine-contact hold control
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat():
            self._recorder.set_genuine_contact_marked(True)
            self._contact_mark_btn.setDown(True)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key.Key_Space and not event.isAutoRepeat():
            self._recorder.set_genuine_contact_marked(False)
            self._contact_mark_btn.setDown(False)
        else:
            super().keyReleaseEvent(event)

    # ------------------------------------------------------------------
    # Sensing connection lifecycle
    # ------------------------------------------------------------------

    def _on_sensing_connect_clicked(self) -> None:
        self._sensing_connect_btn.setEnabled(False)
        port = self._sensing_port_edit.text().strip() or None
        self._sensing_status_lbl.setText("Connecting...")
        self._sensing.connect_hardware(port=port)

    def _on_sensing_connected(self, port: str) -> None:
        self._sensing_status_lbl.setText(f"Connected on {port}. Calibrating baseline...")
        self._append_log(f"Sensing: connected on {port}, calibrating.")
        self._calib_timer.start(int(self._sensing.baseline_duration_s * 1000))

    def _finish_calibration(self) -> None:
        counts = self._sensing.finish_calibration()
        self._sensing_status_lbl.setText(f"Calibrated (samples/finger: {counts}).")
        self._append_log(f"Sensing: calibrated, samples/finger={counts}.")
        self._tick_timer.start(int(sg.DISPLAY_INTERVAL_S * 1000))

        try:
            hc.connect(self._sensing._engine.ctrl)
            for _label, enable_fn, *_ in FINGER_DEFS:
                enable_fn()
            self._servo_connected = True
            self._open_all_btn.setEnabled(True)
            self._reset_baseline_btn.setEnabled(True)
            self._append_log("Servo link established (shared with sensing connection); "
                              "all fingers enabled.")
            self._move_all_and_sync(dict(REFERENCE))
            self._append_log("Hand moved to fully open (reference position).")
        except Exception as exc:
            self._servo_connected = False
            self._append_log(f"Servo link FAILED: {exc} -- sensing data can still be read, "
                              f"just not used to drive the hand.")

    def _on_reset_baseline_clicked(self) -> None:
        self._reset_baseline_btn.setEnabled(False)
        self._sensing_status_lbl.setText(
            f"Resetting baseline ({self._sensing.baseline_duration_s:.0f}s) -- "
            f"make sure nothing is touching any finger...")
        self._append_log("Sensing: resetting baseline -- nothing should be touching any finger.")
        self._sensing.reset_baseline(on_done=self._on_reset_baseline_done)

    def _on_reset_baseline_done(self, counts, excluded) -> None:
        self._reset_baseline_btn.setEnabled(True)
        self._sensing_status_lbl.setText(f"Baseline reset (samples/finger: {counts}).")
        self._append_log(f"Sensing: baseline reset done, samples/finger={counts}.")

    def _on_sensing_disconnected(self) -> None:
        self._tick_timer.stop()
        self._calib_timer.stop()
        self._sensing_connect_btn.setEnabled(True)
        self._sensing_status_lbl.setText("Disconnected.")
        self._append_log("Sensing: disconnected.")
        self._servo_connected = False
        self._open_all_btn.setEnabled(False)
        self._reset_baseline_btn.setEnabled(False)

    def _on_sensing_error(self, msg: str) -> None:
        self._tick_timer.stop()
        self._calib_timer.stop()
        self._sensing_connect_btn.setEnabled(True)
        self._sensing_status_lbl.setText(f"ERROR: {msg}")
        self._append_log(f"Sensing: ERROR -- {msg}")
        self._servo_connected = False
        self._open_all_btn.setEnabled(False)
        self._reset_baseline_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Load cell connection
    # ------------------------------------------------------------------

    def _on_load_connect_clicked(self) -> None:
        port = self._load_port_edit.text().strip() or None
        self._load_connect_btn.setEnabled(False)
        try:
            self._load_cell.connect_hardware(port=port)
            self._load_status_lbl.setText("Connected.")
            self._append_log("Load cell: connected.")
        except Exception as exc:
            self._load_status_lbl.setText(f"FAILED: {exc}")
            self._append_log(f"Load cell: connection FAILED -- {exc}")
            self._load_connect_btn.setEnabled(True)

    def _on_tare_clicked(self) -> None:
        if not self._load_cell.connected:
            self._append_log("Load cell: tare ignored -- not connected.")
            return
        self._load_cell.tare()
        self._append_log("Load cell: tared.")

    # ------------------------------------------------------------------
    # Hand movement -- every finger move in this dashboard goes through
    # here, either from the scenario runner's stepping or "Open All Fingers".
    # ------------------------------------------------------------------

    def _on_open_all_clicked(self) -> None:
        if self._runner_active:
            self._append_log("Open All ignored -- scenario runner is active (Stop it first).")
            return
        if not self._servo_connected:
            self._append_log("Open All: ignored, servo link not connected.")
            return
        try:
            self._move_all_and_sync(dict(REFERENCE))
            self._append_log("Open All: all 4 fingers opened.")
        except Exception as exc:
            self._append_log(f"Open All: servo command FAILED -- {exc}")

    def _move_all_and_sync(self, target: dict[str, float]) -> None:
        """Keeps _commanded_position (the runner's own idea of "where are
        the fingers right now", used to build each step's target) in sync
        with what was actually sent. Also pushes every finger's new
        commanded position into the sensing engine's own crosstalk-
        correction position model (SensingReader.set_finger_position) --
        without this, the engine never learns any finger moved from its
        startup 0% and every rep's crosstalk correction is computed against
        stale, wrong neighbour positions (see the 2026-09-07 pilot-run
        finding: Index reading flat 0.0 and Ring reading spurious force
        during index_thumb reps, traced to exactly this missing call)."""
        move_all(target, hc.DEFAULT_SPEED)
        for label in sg.CROSSTALK_FINGER_LABELS:
            self._commanded_position[label] = target.get(label, 0.0)
            self._sensing.set_finger_position(label, target.get(label, 0.0))

    # ------------------------------------------------------------------
    # Manual episode controls
    # ------------------------------------------------------------------

    def _on_start_episode_clicked(self) -> None:
        episode_id = self._recorder.start_episode()
        self._start_episode_btn.setEnabled(False)
        self._end_success_btn.setEnabled(True)
        self._end_failure_btn.setEnabled(True)
        self._episode_status_lbl.setText(f"Episode {episode_id} RECORDING.")
        self._append_log(f"Episode {episode_id} started.")

    def _on_end_episode_clicked(self, outcome: str) -> None:
        self._recorder.end_episode(outcome)
        self._start_episode_btn.setEnabled(True)
        self._end_success_btn.setEnabled(False)
        self._end_failure_btn.setEnabled(False)
        self._episode_status_lbl.setText(
            f"No episode active. Rows logged this session: {self._recorder.rows_logged_this_session}.")
        self._append_log(f"Episode ended -- {outcome}.")

    # ------------------------------------------------------------------
    # Scenario runner: a QTimer-driven state machine (NOT a blocking loop --
    # a time.sleep()-based loop here would freeze the Qt event loop and stop
    # _tick_timer from firing, which would stop SessionRecorder from logging
    # anything for the whole rep). Stepping runs on its own GRIP_STEP_SETTLE_S
    # timer; the sustained-force check instead rides SessionRecorder's own
    # ~20Hz tick via _on_row_ready, matching SUSTAINED_TICKS' original
    # ~20Hz assumption exactly rather than re-deriving it at a slower cadence.
    # ------------------------------------------------------------------

    def _on_run_rep_clicked(self) -> None:
        if self._runner_active:
            return
        if not self._servo_connected:
            self._append_log("Scenario runner: cannot start -- servo link not connected.")
            return
        if not self._load_cell.connected:
            self._append_log("Scenario runner: cannot start -- load cell not connected.")
            return
        if self._pending_rep is not None:
            self._append_log("Scenario runner: previous rep was NOT saved -- discarding it.")
            self._pending_rep = None
            self._save_rep_btn.setEnabled(False)
            self._pending_status_lbl.setText("No pending rep.")

        scenario_name = self._scenario_combo.currentText()
        level_name = self._level_combo.currentText()
        self._runner_scenario = scenario_name
        self._runner_level = level_name
        self._runner_is_control = self._control_rep_check.isChecked()
        self._runner_orientation = self._orientation_combo.currentText()
        self._runner_target_n = None if self._runner_is_control else self._target_force_spin.value()
        self._runner_close_cap_pct = self._effective_close_cap_pct(
            scenario_name, level_name, self._runner_is_control)
        self._runner_percent = 0.0
        self._runner_phase = "approach"
        self._runner_stop_reason = ""
        self._runner_stop_requested = False
        self._runner_load_by_phase = {"approach": [], "hold": []}

        meta = {
            "scenario": scenario_name, "level": level_name,
            "target_force_n": self._runner_target_n, "is_control": self._runner_is_control,
            "phase": "approach", "load_cell_orientation": self._orientation_combo.currentText(),
        }
        episode_id = self._recorder.start_episode(meta=meta)
        self._runner_episode_id = episode_id

        self._runner_active = True
        self._run_rep_btn.setEnabled(False)
        self._stop_runner_btn.setEnabled(True)
        self._scenario_combo.setEnabled(False)
        self._level_combo.setEnabled(False)
        self._control_rep_check.setEnabled(False)
        self._start_episode_btn.setEnabled(False)

        kind = "CONTROL" if self._runner_is_control else f"target={self._runner_target_n:.2f}N"
        self._append_log(f"Scenario runner: episode {episode_id} -- {scenario_name}/{level_name} ({kind}).")
        self._runner_step_timer.start(int(GRIP_STEP_SETTLE_S * 1000))

    def _on_stop_runner_clicked(self) -> None:
        if not self._runner_active:
            return
        self._runner_stop_requested = True
        self._runner_stop_reason = "user_stopped"
        self._append_log("Scenario runner: STOP requested -- releasing now, rep will NOT be saveable.")
        self._runner_enter_release()

    def _runner_step(self) -> None:
        scenario = SCENARIOS[self._runner_scenario]
        grasping = scenario["grasping"]
        # Level-scaled travel cap (LEVEL_CAP_FRACTION), computed once at
        # _on_run_rep_clicked -- NOT scenario["close_cap_pct"] directly, so
        # a light rep gives up at its own tighter cap rather than the
        # scenario's full safety cap.
        close_cap_pct = self._runner_close_cap_pct

        if self._runner_phase == "approach":
            if self._runner_force_sustained("approach", self._runner_target_n):
                self._runner_stop_reason = "target_force_reached"
                self._runner_enter_hold()
                return
            if self._runner_percent >= close_cap_pct:
                self._runner_stop_reason = "control_full_close" if self._runner_is_control else "cap_reached_no_force"
                self._runner_enter_hold()
                return
            self._runner_percent = min(close_cap_pct, self._runner_percent + GRIP_STEP_PERCENT)
            target = dict(self._commanded_position)
            for f in grasping:
                target[f] = self._runner_percent
            target, note = clamp_for_collision_safety(target)
            if note is not None:
                self._append_log(f"Scenario runner: {note}")
            try:
                self._move_all_and_sync(target)
            except Exception as exc:
                self._append_log(f"Scenario runner: servo command FAILED -- {exc}, aborting rep.")
                self._runner_stop_reason = "servo_error"
                self._runner_finish_rep(aborted=True)
                return

        elif self._runner_phase == "hold":
            if time.time() >= self._runner_hold_deadline:
                self._runner_enter_release()

        elif self._runner_phase == "release":
            if time.time() >= self._runner_release_deadline:
                self._runner_finish_rep(aborted=False)

    def _runner_force_sustained(self, phase: str, target_n: float | None) -> bool:
        if self._runner_is_control or target_n is None:
            return False
        history = self._runner_load_by_phase[phase]
        if len(history) < SUSTAINED_TICKS:
            return False
        return all(v >= target_n for v in history[-SUSTAINED_TICKS:])

    def _runner_enter_hold(self) -> None:
        self._runner_phase = "hold"
        self._recorder.update_episode_meta(phase="hold")
        self._runner_hold_deadline = time.time() + HOLD_S

    def _runner_enter_release(self) -> None:
        self._runner_phase = "release"
        self._recorder.update_episode_meta(phase="release")
        try:
            self._move_all_and_sync(dict(REFERENCE))
        except Exception as exc:
            self._append_log(f"Scenario runner: release move FAILED -- {exc}")
        self._runner_release_deadline = time.time() + RELEASE_SETTLE_S

    @staticmethod
    def _any_window_sustained(values: list[float], threshold: float, window: int) -> bool:
        """True if ANY `window`-consecutive run of `values` is entirely
        >= threshold -- not just the tail. Added 2026-09-07 after the
        load cell's hand-held-not-fixture-mounted reality (user constraint,
        not fixable) was confirmed to introduce real, sometimes large
        operator grip-fatigue/drift over an 8s hold -- several reps showed
        genuine, clearly sustained contact for a good stretch of the hold
        that had already decayed by the very end purely from the human
        hand tiring, not the robot fingers releasing. Checking only the
        final SUSTAINED_TICKS (the previous behaviour) systematically
        penalized exactly those genuine touches. Still requires the same
        number of CONSECUTIVE ticks as before -- a single noise spike still
        can't pass -- just no longer requires that run to be the last one."""
        if len(values) < window:
            return False
        for i in range(len(values) - window + 1):
            if all(v >= threshold for v in values[i:i + window]):
                return True
        return False

    def _runner_finish_rep(self, aborted: bool) -> None:
        self._runner_step_timer.stop()
        aborted = aborted or self._runner_stop_requested
        hold_ticks = self._runner_load_by_phase["hold"]
        peak_force_n = max(hold_ticks, default=0.0)
        if aborted:
            verified = False
        elif self._runner_is_control:
            verified = all(v < NO_CONTACT_FORCE_N_THRESHOLD for v in hold_ticks)
        else:
            # Two independent paths to "verified", either is sufficient.
            # (1) target was already reached DURING approach (the original
            # check) -- hold then just needs to still show real contact
            # (the low NO_CONTACT_FORCE_N_THRESHOLD bar), not necessarily
            # still above the full target (force can settle down a bit
            # once the fingers stop closing and that's still a real touch).
            # (2) NEW (2026-09-07, found live): target was NOT reached
            # during the noisier closing motion before the scenario's
            # travel cap was hit (stop_reason=="cap_reached_no_force"), but
            # the load then sustains AT the full target throughout the
            # stable hold window regardless -- confirmed live on 4/5
            # index_thumb/medium reps (e.g. rep002: hold min 1.61N vs a
            # 1.5N target, 128/128 hold ticks above it) that were real,
            # physically genuine touches wrongly marked "failure" solely
            # because the approach-phase sustained-check didn't have
            # enough closing steps left before the cap. Mirrors this
            # project's standing sustained-ticks-not-a-single-sample
            # principle, just applied to the hold window too instead of
            # only the approach window.
            sustained_in_hold_at_no_contact_bar = self._any_window_sustained(
                hold_ticks, NO_CONTACT_FORCE_N_THRESHOLD, SUSTAINED_TICKS)
            reached_during_approach = (self._runner_stop_reason == "target_force_reached"
                                        and sustained_in_hold_at_no_contact_bar)
            sustained_in_hold_at_target = (
                self._runner_target_n is not None
                and self._any_window_sustained(hold_ticks, self._runner_target_n, SUSTAINED_TICKS))
            verified = reached_during_approach or sustained_in_hold_at_target
            if sustained_in_hold_at_target and not reached_during_approach:
                self._runner_stop_reason = "target_reached_during_hold"

        outcome = "success" if verified else "failure"
        stop_reason = self._runner_stop_reason or "aborted"
        result = {
            "final_percent": self._runner_percent, "stop_reason": stop_reason,
            "peak_hold_force_n": peak_force_n, "verified": verified,
        }
        self._recorder.end_episode(outcome, result=result)

        self._append_log(
            f"Scenario runner: rep finished -- {outcome} "
            f"(stop_reason={stop_reason}, peak_hold={peak_force_n:.2f}N)"
            f"{' -- ABORTED, not saveable' if aborted else ''}.")

        self._runner_active = False
        self._run_rep_btn.setEnabled(True)
        self._stop_runner_btn.setEnabled(False)
        self._scenario_combo.setEnabled(True)
        self._level_combo.setEnabled(True)
        self._control_rep_check.setEnabled(True)
        self._start_episode_btn.setEnabled(True)
        self._runner_status_lbl.setText(
            f"Idle. Last: {self._runner_scenario}/{self._runner_level} -- {outcome}.")

        if aborted:
            self._pending_rep = None
            self._save_rep_btn.setEnabled(False)
            self._pending_status_lbl.setText("No pending rep (last rep was aborted).")
        else:
            self._pending_rep = {
                "scenario": self._runner_scenario, "level": self._runner_level,
                "is_control": self._runner_is_control, "target_force_n": self._runner_target_n,
                "load_cell_orientation": self._runner_orientation,
                "episode_id": self._runner_episode_id, "session_id": self._recorder.session_id,
                "outcome": outcome, **result,
            }
            self._save_rep_btn.setEnabled(True)
            self._pending_status_lbl.setText(
                f"Pending: {self._runner_scenario}/{self._runner_level} -- {outcome} "
                f"(peak_hold={peak_force_n:.2f}N). Click Save to keep it, or Run Rep again to discard.")

    # ------------------------------------------------------------------
    # Save gate: master_tracker.xlsx + confirmed/ are the dataset. A
    # finished rep only reaches either once a human clicks this -- see
    # module docstring's "save-gate" section for why.
    # ------------------------------------------------------------------

    def _progress_key(self) -> tuple[str, str]:
        scenario = self._scenario_combo.currentText()
        level = "control" if self._control_rep_check.isChecked() else self._level_combo.currentText()
        return (scenario, level)

    def _on_level_changed(self, level_name: str) -> None:
        """Selecting a Level auto-fills the target-force spinbox from
        DEFAULT_FORCE_LEVELS_N -- see that spinbox's own comment for why
        this replaced three always-visible, independently-driftable
        spinboxes (2026-09-07 explicit request)."""
        if level_name in DEFAULT_FORCE_LEVELS_N:
            self._target_force_spin.setValue(DEFAULT_FORCE_LEVELS_N[level_name])
        self._refresh_progress_label()

    def _effective_close_cap_pct(self, scenario_name: str, level_name: str, is_control: bool) -> float:
        """How far the grasping fingers are allowed to travel this rep --
        see LEVEL_CAP_FRACTION's own comment. Control reps always use the
        scenario's FULL cap regardless of level, since a control rep's
        whole point is confirming near-zero force even at full closure."""
        base_cap = SCENARIOS[scenario_name]["close_cap_pct"]
        if is_control:
            return base_cap
        return base_cap * LEVEL_CAP_FRACTION.get(level_name, 1.0)

    def _refresh_progress_label(self) -> None:
        scenario, level = self._progress_key()
        done = self._confirmed_counts.get((scenario, level), 0)
        target = self._reps_spin.value()
        done_str = f"{done}/{target} confirmed" if done < target else f"{done}/{target} confirmed (target met)"
        # Orientation breakdown (2026-09-07, explicit request after the
        # "forgot to switch the dropdown" mistake happened twice) -- shown
        # live so a lopsided count (e.g. all 5 still on one finger) is
        # visible WHILE collecting this combo, not just discovered after.
        orientation_counts = self._orientation_counts.get((scenario, level), {})
        orientation_str = ", ".join(f"{o}:{c}" for o, c in sorted(orientation_counts.items()))
        suffix = f"  [{orientation_str}]" if orientation_str else ""
        self._progress_lbl.setText(f"{scenario} / {level}: {done_str}{suffix}")

    def _on_save_rep_clicked(self) -> None:
        pending = self._pending_rep
        if pending is None:
            return
        scenario, level = pending["scenario"], ("control" if pending["is_control"] else pending["level"])
        raw_path = TRAINING_DIR / f"session_{pending['session_id']}_episode{pending['episode_id']}.jsonl"
        if not raw_path.exists():
            self._append_log(f"Save Rep: FAILED -- raw log not found at {raw_path}.")
            return

        rep_index = self._confirmed_counts.get((scenario, level), 0) + 1
        dest_dir = CONFIRMED_DIR / scenario / level
        dest_dir.mkdir(parents=True, exist_ok=True)
        timestamp_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = dest_dir / f"rep{rep_index:03d}_{timestamp_compact}.jsonl"
        try:
            shutil.copy2(raw_path, dest_path)
        except OSError as exc:
            self._append_log(f"Save Rep: FAILED to copy raw log -- {exc}")
            return

        row = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "scenario": pending["scenario"], "level": pending["level"],
            "is_control": pending["is_control"], "rep_index_for_combo": rep_index,
            "target_force_n": pending["target_force_n"], "peak_hold_force_n": pending["peak_hold_force_n"],
            "outcome": pending["outcome"], "verified": pending["verified"],
            "stop_reason": pending["stop_reason"], "final_percent": pending["final_percent"],
            "object_used": self._object_edit.text(),
            "load_cell_orientation": pending["load_cell_orientation"],
            "confirmed_jsonl": str(dest_path.relative_to(TRAINING_DIR)),
        }
        try:
            training_tracker.append_rep_row(MASTER_XLSX, row)
        except Exception as exc:
            self._append_log(f"Save Rep: copied JSONL but FAILED to update master_tracker.xlsx -- {exc}")
            return

        self._confirmed_counts[(scenario, level)] = rep_index
        orientation = pending["load_cell_orientation"]
        combo_orientations = self._orientation_counts.setdefault((scenario, level), {})
        combo_orientations[orientation] = combo_orientations.get(orientation, 0) + 1
        self._append_log(f"Save Rep: confirmed as {scenario}/{level} rep {rep_index} "
                          f"(facing {orientation}) -- {dest_path.name}.")
        self._pending_rep = None
        self._save_rep_btn.setEnabled(False)
        self._pending_status_lbl.setText("No pending rep.")
        self._refresh_progress_label()

    # ------------------------------------------------------------------
    # Live display -- driven ONLY by SessionRecorder.row_ready
    # ------------------------------------------------------------------

    def _on_row_ready(self, row: dict) -> None:
        raw_results = self._sensing.latest_raw_results()
        if raw_results is not None:
            for fr in raw_results:
                label = sg.CROSSTALK_FINGER_LABELS[fr.index]
                self._finger_bars[label].update_from_result(fr)

        force = row.get("load_cell_force")
        self._load_force_lbl.setText(f"Load: {force:.2f} N" if force is not None else "Load: -- N")

        if self._runner_active and force is not None and self._runner_phase in self._runner_load_by_phase:
            self._runner_load_by_phase[self._runner_phase].append(force)

        if self._runner_active:
            target_str = "CONTROL" if self._runner_is_control else f"{self._runner_target_n:.2f}N"
            load_str = f"{force:.2f}N" if force is not None else "--N"
            self._runner_status_lbl.setText(
                f"RUNNING: {self._runner_scenario}/{self._runner_level} ({target_str})  "
                f"phase={self._runner_phase}  step={self._runner_percent:.0f}%  load={load_str}")

        if row.get("episode_id") is not None:
            marker = "GENUINE CONTACT MARKED" if row.get("genuine_contact_marked") else ""
            self._episode_status_lbl.setText(
                f"Episode {row['episode_id']} RECORDING -- rows: {self._recorder.rows_logged_this_session}  {marker}")

    def closeEvent(self, event) -> None:
        self._runner_step_timer.stop()
        self._recorder.end_session()
        if self._servo_connected:
            try:
                move_all(REFERENCE, hc.DEFAULT_SPEED)
            except Exception:
                pass
        self._load_cell.close()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
