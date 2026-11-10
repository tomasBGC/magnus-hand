"""Shared constants/paths for the crosstalk pipeline (build -> train -> evaluate).

Kept entirely separate from model_v1's main force/direction pipeline
(build_delta_dataset.py, train_model_contact.py, etc.) -- this module is
only imported by scripts inside model_v1/crosstalk/.
"""
from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
DATASET_CSV = SCRIPT_DIR / "crosstalk_dataset.csv"
MODEL_JSON = SCRIPT_DIR / "crosstalk_model_v1.json"
REPORT_MD = SCRIPT_DIR / "crosstalk_model_report.md"

NUM_SENSORS = 5
# S1X..S5Z -- same raw-channel naming convention as model_v1's own
# raw_cols (see live_predictor_test.py's frame_to_raw_values).
RAW_COLS = [f"S{n}{axis}" for n in range(1, NUM_SENSORS + 1) for axis in "XYZ"]

# Same Index/Middle/Ring/Thumb <-> mux-group mapping confirmed in
# model_v1/hand_control_gui.py's FINGER_CONTROLS and already used by
# record_crosstalk.py.
FINGER_LABELS = ["Index", "Middle", "Ring", "Thumb"]

POSITIONS = [0, 25, 50, 75, 100]
NONZERO_POSITIONS = [25, 50, 75, 100]

# Effect-size / noise-floor ratio above which a source finger's crosstalk
# on an affected finger is flagged "significant" (see
# evaluate_crosstalk_model.py and README.md's additive-assumption section).
# This is a simple, documented screening heuristic given only 20
# recordings -- not a formal statistical significance test.
SIGNIFICANT_SNR_THRESHOLD = 3.0
