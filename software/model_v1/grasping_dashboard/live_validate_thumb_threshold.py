"""Launches the Grasping Research Dashboard to live-validate the Thumb-
specific Contact-confirm threshold (sensing_gui.py's
CONTACT_P_CONFIRM_PER_FINGER["Thumb"]=0.10 / CONTACT_P_RELEASE_PER_FINGER
["Thumb"]=0.05, added 2026-09-07 alongside the load-cell classifier
retrain) against real hardware -- the offline holdout predicted recall
0.974 at zero false positives (783 held-out ticks, 6 reps never used in
training), but that number has never been checked live. See
thumb_threshold_live_validation_report.md for the full writeup this
session's data feeds into.

Unlike live_validate_candidate_model.py (which swapped in a not-yet-
deployed CANDIDATE classifier bundle), there is no separate model file
under test here -- the retrained Contact classifier is already
active_model.pkl (the default sg.MODEL_PATH), and the only thing this
script needs to verify is that the THRESHOLD override is actually the one
in effect, since that lives in sensing_gui.py as a plain module constant
that's trivially easy to silently drift from (e.g. a stale checkout, or a
future edit to the global default). Refuses to open the window at all if
the override doesn't match what's under test, rather than silently running
a session that wouldn't actually exercise the fix.

Session data is redirected to its own live_validation_thumb_threshold/
folder -- kept separate from the real training dataset AND the earlier
live_validation_candidate/ session (which tested the classifier itself,
before this threshold change existed -- see that folder's own session
timestamp, 2026-09-07 19:40, against this threshold's 20:32 save time).

Usage: uv run python model_v1/grasping_dashboard/live_validate_thumb_threshold.py
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_V1_DIR = SCRIPT_DIR.parent
for p in (str(MODEL_V1_DIR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import sensing_gui as sg  # noqa: E402

EXPECTED_CONFIRM = 0.10
EXPECTED_RELEASE = 0.05
_actual_confirm = sg.CONTACT_P_CONFIRM_PER_FINGER.get("Thumb")
_actual_release = sg.CONTACT_P_RELEASE_PER_FINGER.get("Thumb")
if _actual_confirm != EXPECTED_CONFIRM or _actual_release != EXPECTED_RELEASE:
    raise SystemExit(
        "Thumb threshold override NOT as expected -- refusing to run a "
        "validation session that wouldn't actually exercise the fix under "
        f"test.\n  CONTACT_P_CONFIRM_PER_FINGER['Thumb'] = {_actual_confirm!r} "
        f"(expected {EXPECTED_CONFIRM})\n"
        f"  CONTACT_P_RELEASE_PER_FINGER['Thumb'] = {_actual_release!r} "
        f"(expected {EXPECTED_RELEASE})")
print("[live_validate_thumb_threshold] Confirmed Thumb threshold override IS "
      f"active: CONFIRM={_actual_confirm}, RELEASE={_actual_release} "
      f"(global default for other fingers stays {sg.CONTACT_P_CONFIRM}/{sg.CONTACT_P_RELEASE}).")
print(f"[live_validate_thumb_threshold] Model in use: {sg.MODEL_PATH} "
      "(the currently DEPLOYED active_model.pkl -- no candidate swap needed, "
      "only the threshold is under test here).")

import dashboard_gui as dg  # noqa: E402

VALIDATION_DIR = MODEL_V1_DIR / "grasping_dashboard_data" / "live_validation_thumb_threshold"
dg.TRAINING_DIR = VALIDATION_DIR
dg.CONFIRMED_DIR = VALIDATION_DIR / "confirmed"
dg.MASTER_XLSX = VALIDATION_DIR / "master_tracker.xlsx"
print(f"[live_validate_thumb_threshold] session data redirected to:\n  {VALIDATION_DIR}\n"
      "(kept separate from the real training dataset and the earlier candidate-classifier "
      "validation session).")

from PyQt6 import QtWidgets  # noqa: E402


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = dg.DashboardWindow()
    window.setWindowTitle(window.windowTitle() + "  [THUMB THRESHOLD VALIDATION -- P_CONFIRM=0.10/P_RELEASE=0.05]")
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
