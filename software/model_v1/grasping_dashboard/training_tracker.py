"""Master Excel tracker for the load-cell grasp training protocol
(dashboard_gui.py's Scenario Runner). One row per CONFIRMED rep -- a rep the
user reviewed and explicitly clicked "Save Rep" for, never an unreviewed
run. Kept as its own small module (rather than inlined into
dashboard_gui.py) so the openpyxl-specific read/write/formula logic doesn't
bloat the GUI file, matching this codebase's convention of one concern per
file (see e.g. load_cell_reader.py, session_recorder.py).

Two sheets:
  - "Reps": one row per confirmed rep, append-only -- the actual dataset
    index. Columns match exactly what a later analysis script would want
    without re-parsing the raw JSONL files.
  - "Summary": a scenario x level count matrix built from COUNTIFS formulas
    against the Reps sheet -- Excel recalculates these itself every time
    the file is opened or the Reps sheet changes, so it stays "live" with
    zero extra code on this end; this module only ever writes it once, the
    first time the workbook is created.
"""
from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

REPS_SHEET = "Reps"
SUMMARY_SHEET = "Summary"

REPS_COLUMNS = [
    "timestamp_iso", "scenario", "level", "is_control", "rep_index_for_combo",
    "target_force_n", "peak_hold_force_n", "outcome", "verified", "stop_reason",
    "final_percent", "object_used", "load_cell_orientation", "confirmed_jsonl",
]


def _build_summary_formulas(ws_summary: Worksheet, scenarios: list[str], levels: list[str]) -> None:
    """(scenario rows) x (level + control columns), each cell a COUNTIFS
    against the Reps sheet -- Excel recalculates these live, this function
    only runs once at workbook creation."""
    ws_summary.cell(row=1, column=1, value="scenario")
    columns = list(levels) + ["control"]
    for c, level in enumerate(columns, start=2):
        ws_summary.cell(row=1, column=c, value=level)
    scenario_col = get_column_letter(REPS_COLUMNS.index("scenario") + 1)
    level_col = get_column_letter(REPS_COLUMNS.index("level") + 1)
    control_col = get_column_letter(REPS_COLUMNS.index("is_control") + 1)
    for r, scenario in enumerate(scenarios, start=2):
        ws_summary.cell(row=r, column=1, value=scenario)
        for c, level in enumerate(columns, start=2):
            if level == "control":
                formula = (f'=COUNTIFS({REPS_SHEET}!{scenario_col}:{scenario_col},A{r},'
                           f'{REPS_SHEET}!{control_col}:{control_col},TRUE)')
            else:
                formula = (f'=COUNTIFS({REPS_SHEET}!{scenario_col}:{scenario_col},A{r},'
                           f'{REPS_SHEET}!{level_col}:{level_col},"{level}",'
                           f'{REPS_SHEET}!{control_col}:{control_col},FALSE)')
            ws_summary.cell(row=r, column=c, value=formula)


def ensure_workbook(path: Path, scenarios: list[str], levels: list[str]) -> None:
    """Creates the workbook with both sheets if it doesn't exist yet.
    Never overwrites an existing file -- scenarios/levels are only used for
    the Summary sheet's initial layout on first creation."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws_reps = wb.active
    ws_reps.title = REPS_SHEET
    ws_reps.append(REPS_COLUMNS)
    ws_summary = wb.create_sheet(SUMMARY_SHEET)
    _build_summary_formulas(ws_summary, scenarios, levels)
    wb.save(path)


def append_rep_row(path: Path, row: dict) -> None:
    """Appends one confirmed rep to the Reps sheet and saves. Assumes
    ensure_workbook() has already been called (raises if the file is
    missing rather than silently creating a header-less/summary-less file)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- call ensure_workbook() first.")
    wb = load_workbook(path)
    ws = wb[REPS_SHEET]
    ws.append([row.get(col, "") for col in REPS_COLUMNS])
    wb.save(path)


def load_confirmed_counts(path: Path) -> dict[tuple[str, str], int]:
    """Reads the Reps sheet and returns {(scenario, level_or_'control'): count}
    -- used at dashboard startup so rep-progress-per-combo survives across
    app restarts (the Excel file, not in-memory state, is the source of
    truth for "how many have we already collected")."""
    counts: dict[tuple[str, str], int] = {}
    if not path.exists():
        return counts
    wb = load_workbook(path, read_only=True, data_only=True)
    if REPS_SHEET not in wb.sheetnames:
        return counts
    ws = wb[REPS_SHEET]
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    try:
        scenario_idx = header.index("scenario")
        level_idx = header.index("level")
        control_idx = header.index("is_control")
    except ValueError:
        return counts
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[scenario_idx] is None:
            continue
        key_level = "control" if row[control_idx] else row[level_idx]
        key = (row[scenario_idx], key_level)
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_orientation_counts(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    """Reads the Reps sheet and returns {(scenario, level_or_'control'):
    {orientation: count}} -- same shape/purpose as load_confirmed_counts,
    split further by load_cell_orientation. Added 2026-09-07 after the same
    "forgot to switch the orientation dropdown for the back half of a
    level's reps" mistake happened twice in a row (middle_thumb, then
    middle_ring) -- lets the dashboard show a live per-orientation
    breakdown so a lopsided count (e.g. 5/0 instead of the intended 3/2) is
    visible before all reps for that combo are done, not caught after."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    if not path.exists():
        return counts
    wb = load_workbook(path, read_only=True, data_only=True)
    if REPS_SHEET not in wb.sheetnames:
        return counts
    ws = wb[REPS_SHEET]
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    try:
        scenario_idx = header.index("scenario")
        level_idx = header.index("level")
        control_idx = header.index("is_control")
        orientation_idx = header.index("load_cell_orientation")
    except ValueError:
        return counts
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[scenario_idx] is None:
            continue
        key_level = "control" if row[control_idx] else row[level_idx]
        key = (row[scenario_idx], key_level)
        orientation = row[orientation_idx] or "?"
        combo = counts.setdefault(key, {})
        combo[orientation] = combo.get(orientation, 0) + 1
    return counts
