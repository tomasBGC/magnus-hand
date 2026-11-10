"""Launches the Grasping Research Dashboard with the load-cell Contact
classifier CANDIDATE (active_model_load_cell_contact_candidate.pkl, built by
retrain_contact_with_load_cell_data.py) driving the live sensing pipeline,
instead of the deployed active_model.pkl -- for live validation on fresh,
real hardware reps the candidate has never seen (the offline holdout in the
retrain script only ever replayed already-logged ticks; this drives the
actual robot hand live).

Never touches active_model.pkl or the real training dataset:
  - sensing_gui.MODEL_PATH is monkeypatched to point at the candidate bundle
    before HandSensingEngine is constructed. HandSensingEngine.__init__
    reads MODEL_PATH as a bare module-global at call time (not at import
    time), so reassigning it here is resolved correctly with zero file
    swapping and zero risk to the deployed model on disk.
  - dashboard_gui's TRAINING_DIR/CONFIRMED_DIR/MASTER_XLSX are monkeypatched
    to a separate live_validation_candidate/ folder, so this session's reps
    never mix into or get counted as part of the real training dataset
    (grasping_dashboard_data/training_protocol/) built from genuine data
    collection.

Everything else -- Scenario Runner, load-cell verification, orientation
tracking, raw_channels/raw_force logging -- is identical to the normal
dashboard, so the resulting JSONL directly compares the candidate's live
Contact/Force predictions (sensing.<Finger>.contact/raw_contact/force/
raw_force) against the same trusted, independent load-cell ground truth
used throughout this whole protocol.

Usage: uv run python model_v1/grasping_dashboard/live_validate_candidate_model.py
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_V1_DIR = SCRIPT_DIR.parent
for p in (str(MODEL_V1_DIR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

CANDIDATE_PATH = MODEL_V1_DIR / "active_model_load_cell_contact_candidate.pkl"
if not CANDIDATE_PATH.exists():
    raise SystemExit(
        f"Candidate model not found at {CANDIDATE_PATH} -- "
        "run retrain_contact_with_load_cell_data.py first.")

import sensing_gui as sg  # noqa: E402
sg.MODEL_PATH = str(CANDIDATE_PATH)
print(f"[live_validate_candidate_model] sensing_gui.MODEL_PATH overridden to:\n  {CANDIDATE_PATH}\n"
      "active_model.pkl on disk is NOT touched.")

import dashboard_gui as dg  # noqa: E402

VALIDATION_DIR = MODEL_V1_DIR / "grasping_dashboard_data" / "live_validation_candidate"
dg.TRAINING_DIR = VALIDATION_DIR
dg.CONFIRMED_DIR = VALIDATION_DIR / "confirmed"
dg.MASTER_XLSX = VALIDATION_DIR / "master_tracker.xlsx"
print(f"[live_validate_candidate_model] session data redirected to:\n  {VALIDATION_DIR}\n"
      "(kept separate from the real training dataset -- never merged automatically).")

from PyQt6 import QtWidgets  # noqa: E402


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = dg.DashboardWindow()
    window.setWindowTitle(window.windowTitle() + "  [CANDIDATE MODEL -- live_cell_contact_candidate]")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
