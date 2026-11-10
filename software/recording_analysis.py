"""
Automatic Post-Recording Analysis Pipeline
============================================
Runs once, automatically, every time a recording is stopped and saved
(hooked into UnifiedController._save_to_json — see unified_controller.py).
No GUI interaction is required and none of this depends on the Data
Analysis tab being open.

For the recording just saved, this module:
  1. Exports four PNG plots + four matching CSVs (the underlying plotted
     data) into the same folder as the recording's JSON file.
  2. Returns a short quality-check text summary for immediate display in
     the GUI.

Note: master_characterisation.csv and characterisation_summary.csv are NOT
touched automatically. They are updated on demand via update_summary_files()
below, wired to the "Update Summary Files" button in the Data Analysis tab.
That function scans the recordings folder, detects individual press events
within each not-yet-processed recording (each recording is one continuous
stream containing 3 presses — light/medium/hard — not 3 separate files),
appends one row per press to characterisation_summary.csv, and one row per
individual sample (every timestamped snapshot, not just presses) to
master_characterisation.csv, skipping recordings already reflected in each
file.

Magnetic unit note (verified, not assumed)
-------------------------------------------
Firmware: MagneticSensors/MCU/lib/eflesh/eflesh.cpp, collectGroup():
    MLX90393::txyz conv = g.mlx[i].convertRaw(raw);
    g.data[i].x = conv.x;  g.data[i].y = conv.y;  g.data[i].z = conv.z;
convertRaw() (MagneticSensors/MCU/lib/MLX90393/MLX90393.cpp) multiplies the
raw ADC counts by the MLX90393 datasheet sensitivity constants
(base_xy_sens_hc0 = 0.196, base_z_sens_hc0 = 0.316 uT/LSB) scaled by the
configured gain and resolution. So the Teensy already emits calibrated
X/Y/Z in **microtesla (uT)** over serial — these are not raw sensor counts.
MagneticSensors/eflesh_controller.py (_parse_one_frame) unpacks those
floats verbatim with no further scaling, and SensorReading.magnitude /
unified_controller.py's SensorSnapshot both compute
magnitude = sqrt(x**2 + y**2 + z**2) directly from them. Therefore the
"magnitude" stored in every recording JSON, and used throughout this
module, is genuinely in microtesla (uT) — not an arbitrary/raw unit.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# ============================================================================
# Constants
# ============================================================================

GF_TO_N = 0.00981   # gram-force -> Newtons

# HX711 load cell readings are noise/glitches above this cap (seen up to
# 20,000+ g in testing) and must be discarded before any plotting/export/
# press-detection — not just clipped.
MAX_VALID_FORCE_N = 20.0
MAX_VALID_FORCE_G = MAX_VALID_FORCE_N / GF_TO_N   # ~2038.7 g

MAG_UNIT  = "µT"          # confirmed calibrated microtesla, see module docstring
MAG_LABEL = f"Magnetic Flux Density ({MAG_UNIT})"

NUM_SENSORS = 5
GROUP_INDEX = 0     # matches the group the interactive Data Analysis tab plots

PRESS_FORCE_THRESHOLD_N = 0.3     # a press is any contiguous run with force > this
MAX_PRESSES_PER_RECORDING = 3     # keep only the top-N highest-peak-force presses
SENSOR_NOISE_THRESHOLD = 10.0     # magnitude swing (uT) above which a sensor counts as "responding"

SUMMARY_CSV_NAME = "characterisation_summary.csv"
MASTER_CSV_NAME = "master_characterisation.csv"


def _sanitize_unit(unit: str | None) -> str:
    """Normalise a free-text Unit field value into a filename-safe slug.

    Recordings made before the Unit field existed have no "unit" metadata
    key at all, and typing "unit1" (or leaving the field blank) is treated
    the same way — all three map to the original characterisation_summary.csv
    / master_characterisation.csv so the first fingertip's data is untouched.
    """
    return re.sub(r"[^\w\-]", "_", (unit or "").strip()).strip("_")


def _is_default_unit(unit_slug: str) -> bool:
    return unit_slug == "" or unit_slug.lower() == "unit1"


def _unit_label(unit_slug: str) -> str:
    return unit_slug if unit_slug else "unit1"


def _summary_csv_name(unit_slug: str) -> str:
    if _is_default_unit(unit_slug):
        return SUMMARY_CSV_NAME
    return f"characterisation_summary_{unit_slug}.csv"


def _master_csv_name(unit_slug: str) -> str:
    if _is_default_unit(unit_slug):
        return MASTER_CSV_NAME
    return f"master_characterisation_{unit_slug}.csv"

# Colourblind-friendly palette (Okabe & Ito, 2008) — used consistently
# across every plot this module produces.
_SENSOR_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
_XYZ_COLORS    = {'X': '#0072B2', 'Y': '#E69F00', 'Z': '#009E73'}

SUMMARY_COLUMNS = (
    ["Recording Filename", "Face", "Theta", "Press Number", "Peak Force (N)"]
    + [f"Peak Magnitude Sensor {i} ({MAG_UNIT})" for i in range(1, NUM_SENSORS + 1)]
)

MASTER_COLUMNS = (
    ["Row Number", "Recording Name", "Time", "Theta", "Phi", "Force"]
    + [f"S{i}{axis}" for i in range(1, NUM_SENSORS + 1) for axis in ("X", "Y", "Z")]
)


# ============================================================================
# Data extraction
# ============================================================================

def _valid_force_g(raw) -> float:
    """Reject impossible HX711 glitch readings (load cell maxes out at 1000 g)."""
    val = raw if raw is not None else 0.0
    return val if abs(val) <= MAX_VALID_FORCE_G else float("nan")


def _extract_series(data: dict):
    """Pull time/force/magnitude/xyz arrays for Group 1 (group_index 0) out of a recording dict."""
    snaps = data.get("snapshots", [])
    if not snaps:
        return None

    t0 = snaps[0]["timestamp"]
    times = np.array([s["timestamp"] - t0 for s in snaps])
    force_n = np.array([_valid_force_g(s.get("force_g")) * GF_TO_N for s in snaps])

    n_s = 0
    for s in snaps:
        for g in s.get("groups", []):
            if g.get("group_index") == GROUP_INDEX:
                n_s = max(n_s, len(g.get("sensors", [])))
    n_s = min(n_s, NUM_SENSORS)

    mags = np.zeros((len(snaps), n_s))
    xyz  = np.zeros((len(snaps), n_s, 3))
    for i, s in enumerate(snaps):
        for g in s.get("groups", []):
            if g.get("group_index") == GROUP_INDEX:
                for j, sensor in enumerate(g.get("sensors", [])[:n_s]):
                    mags[i, j]   = sensor.get("magnitude", 0.0)
                    xyz[i, j, 0] = sensor.get("x", 0.0)
                    xyz[i, j, 1] = sensor.get("y", 0.0)
                    xyz[i, j, 2] = sensor.get("z", 0.0)

    return times, force_n, mags, xyz


# ============================================================================
# Plot styling (academic / publication style, shared across all 4 plots)
# ============================================================================

def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.35, color='#cccccc', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#222222', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.set_title(title, fontsize=12, fontweight='bold', color='#111111', pad=10)
    ax.set_xlabel(xlabel, fontsize=11, color='#111111')
    ax.set_ylabel(ylabel, fontsize=11, color='#111111')


def _new_figure(rec_name: str, figsize=(8, 5)) -> Figure:
    fig = Figure(figsize=figsize, facecolor='white', dpi=150)
    fig.text(0.5, 0.995, rec_name, ha='center', va='top',
              fontsize=9, color='#666666')
    return fig


def _save_fig(fig: Figure, out_png: Path):
    canvas = FigureCanvasAgg(fig)
    fig.savefig(out_png, dpi=150, facecolor='white')


def _legend(ax, **kwargs):
    kwargs.setdefault('fontsize', 9)
    ax.legend(facecolor='white', edgecolor='#999999',
              labelcolor='#111111', **kwargs)


# ============================================================================
# The four plots + four CSVs
# ============================================================================

def _export_force_vs_time(rec_name, out_png, out_csv, times, force_n):
    fig = _new_figure(rec_name)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.96)
    ax.plot(times, force_n, color=_SENSOR_COLORS[0], linewidth=1.6)
    _style_ax(ax, "Force vs Time", "Time (s)", "Force (N)")
    _save_fig(fig, out_png)

    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(["Time (s)", "Force (N)"])
        for t, f in zip(times, force_n):
            w.writerow([f"{t:.6f}", f"{f:.6f}"])


def _export_magnitude_vs_time(rec_name, out_png, out_csv, times, mags):
    n_s = mags.shape[1]
    fig = _new_figure(rec_name)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.96)
    for j in range(n_s):
        ax.plot(times, mags[:, j], color=_SENSOR_COLORS[j % len(_SENSOR_COLORS)],
                linewidth=1.3, label=f"Sensor {j + 1}")
    _style_ax(ax, "Magnetic Flux Density vs Time", "Time (s)", MAG_LABEL)
    if n_s:
        _legend(ax, loc='upper right', ncol=min(n_s, 5))
    _save_fig(fig, out_png)

    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(["Time (s)"] + [f"Magnitude Sensor {j + 1} ({MAG_UNIT})" for j in range(n_s)])
        for i, t in enumerate(times):
            w.writerow([f"{t:.6f}"] + [f"{mags[i, j]:.4f}" for j in range(n_s)])


def _export_force_vs_magnitude(rec_name, out_png, out_csv, force_n, mags):
    n_s = mags.shape[1]
    fig = _new_figure(rec_name)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.96)
    for j in range(n_s):
        ax.scatter(mags[:, j], force_n, color=_SENSOR_COLORS[j % len(_SENSOR_COLORS)],
                   s=8, alpha=0.65, label=f"Sensor {j + 1}")
    _style_ax(ax, "Force vs Magnetic Flux Density", MAG_LABEL, "Force (N)")
    if n_s:
        _legend(ax, loc='upper right', ncol=min(n_s, 5))
    _save_fig(fig, out_png)

    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(["Force (N)"] + [f"Magnitude Sensor {j + 1} ({MAG_UNIT})" for j in range(n_s)])
        for i, f in enumerate(force_n):
            w.writerow([f"{f:.6f}"] + [f"{mags[i, j]:.4f}" for j in range(n_s)])


def _export_xyz_components(rec_name, out_png, out_csv, times, xyz):
    n_s = xyz.shape[1]
    rows = max(n_s, 1)
    fig = _new_figure(rec_name, figsize=(9, 2.1 * rows + 1.4))
    fig.subplots_adjust(top=0.84, bottom=0.07, left=0.13, right=0.97, hspace=0.45)
    axes = fig.subplots(rows, 1, sharex=True) if rows > 1 else [fig.add_subplot(111)]
    if rows == 1:
        axes = [axes] if not isinstance(axes, (list, np.ndarray)) else list(axes)

    fig.suptitle("X, Y, Z Components vs Time", fontsize=12, fontweight='bold',
                 color='#111111', y=0.975)
    fig.text(0.03, 0.5, MAG_LABEL, va='center', ha='center', rotation='vertical',
             fontsize=11, color='#111111')

    handles = None
    for k in range(rows):
        ax = axes[k]
        is_last = k == rows - 1
        _style_ax(ax, f"Sensor {k + 1}", "Time (s)" if is_last else "", "")
        if not is_last:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
        if k < n_s:
            for c, axis_name in enumerate(['X', 'Y', 'Z']):
                ax.plot(times, xyz[:, k, c], color=_XYZ_COLORS[axis_name],
                        linewidth=1.1, label=f"{axis_name} Component")
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', ncol=3, fontsize=9,
                   facecolor='white', edgecolor='#999999', labelcolor='#111111',
                   bbox_to_anchor=(0.97, 0.895))
    _save_fig(fig, out_png)

    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        header = ["Time (s)"]
        for k in range(n_s):
            header += [f"Sensor {k + 1} X ({MAG_UNIT})",
                       f"Sensor {k + 1} Y ({MAG_UNIT})",
                       f"Sensor {k + 1} Z ({MAG_UNIT})"]
        w.writerow(header)
        for i, t in enumerate(times):
            row = [f"{t:.6f}"]
            for k in range(n_s):
                row += [f"{xyz[i, k, 0]:.4f}", f"{xyz[i, k, 1]:.4f}", f"{xyz[i, k, 2]:.4f}"]
            w.writerow(row)


def _export_all(rec_dir: Path, rec_name: str, times, force_n, mags, xyz):
    _export_force_vs_time(
        rec_name, rec_dir / f"{rec_name}_force_vs_time.png",
        rec_dir / f"{rec_name}_force_vs_time.csv", times, force_n)
    _export_magnitude_vs_time(
        rec_name, rec_dir / f"{rec_name}_magnitude_vs_time.png",
        rec_dir / f"{rec_name}_magnitude_vs_time.csv", times, mags)
    _export_force_vs_magnitude(
        rec_name, rec_dir / f"{rec_name}_force_vs_magnitude.png",
        rec_dir / f"{rec_name}_force_vs_magnitude.csv", force_n, mags)
    _export_xyz_components(
        rec_name, rec_dir / f"{rec_name}_xyz_components.png",
        rec_dir / f"{rec_name}_xyz_components.csv", times, xyz)


# ============================================================================
# Press-event detection
# ============================================================================

def detect_press_events(force_n: np.ndarray,
                        threshold: float = PRESS_FORCE_THRESHOLD_N) -> list[tuple[int, int, int]]:
    """Return one (start_idx, end_idx, peak_idx) per contiguous rise-and-fall
    above *threshold* — i.e. one entry per press event."""
    events = []
    n = len(force_n)
    above = force_n > threshold
    i = 0
    while i < n:
        if above[i]:
            start = i
            while i < n and above[i]:
                i += 1
            end = i - 1
            peak_idx = start + int(np.argmax(force_n[start:end + 1]))
            events.append((start, end, peak_idx))
        else:
            i += 1
    return events


# ============================================================================
# characterisation_summary.csv
# ============================================================================

def _build_summary_rows(json_path: Path, data: dict, force_n: np.ndarray,
                        mags: np.ndarray, events: list[tuple[int, int, int]]) -> list[dict]:
    metadata = data.get("metadata", {})
    face  = metadata.get("face", "")
    theta = metadata.get("theta", "")
    n_s = mags.shape[1] if mags.ndim == 2 else 0

    # Rank all detected presses by peak force (highest first) and keep only
    # the top MAX_PRESSES_PER_RECORDING — no validation against an expected count.
    ranked = sorted(events, key=lambda e: force_n[e[0]:e[1] + 1].max(), reverse=True)
    top_events = ranked[:MAX_PRESSES_PER_RECORDING]

    rows = []
    for order, (start, end, _peak_idx) in enumerate(top_events, start=1):
        row = {
            "Recording Filename": json_path.name,
            "Face": face,
            "Theta": theta,
            "Press Number": order,
            "Peak Force (N)": f"{force_n[start:end + 1].max():.6f}",
        }
        for j in range(NUM_SENSORS):
            key = f"Peak Magnitude Sensor {j + 1} ({MAG_UNIT})"
            row[key] = f"{mags[start:end + 1, j].max():.4f}" if j < n_s else ""
        rows.append(row)

    if not rows:
        # Still log the recording even with zero detected presses, so it's
        # easy to spot in the summary rather than silently missing.
        row = {
            "Recording Filename": json_path.name,
            "Face": face,
            "Theta": theta,
            "Press Number": 0,
            "Peak Force (N)": "",
        }
        for j in range(NUM_SENSORS):
            row[f"Peak Magnitude Sensor {j + 1} ({MAG_UNIT})"] = ""
        rows.append(row)

    return rows


def _append_summary_csv(summary_csv_path: Path, rows: list[dict]):
    file_exists = summary_csv_path.exists()
    with open(summary_csv_path, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================================
# master_characterisation.csv
# ============================================================================

def _count_existing_master_rows(master_csv_path: Path) -> int:
    """Number of data rows already written (excludes header), so Row Number
    keeps incrementing across recordings instead of resetting."""
    if not master_csv_path.exists():
        return 0
    with open(master_csv_path, 'r', newline='', encoding='utf-8') as fh:
        n_lines = sum(1 for _ in csv.reader(fh))
    return max(n_lines - 1, 0)


def _build_master_rows(rec_name: str, metadata: dict, times: np.ndarray,
                       force_n: np.ndarray, xyz: np.ndarray, start_row: int) -> list[dict]:
    """One row per sample timestamp (not per press). Samples whose force
    reading was discarded by the 20N glitch filter (NaN in force_n) are
    skipped entirely, per the same glitch-discard policy used elsewhere."""
    theta = metadata.get("theta", "")
    phi = metadata.get("phi", "")
    n_s = xyz.shape[1] if xyz.ndim == 3 else 0

    rows = []
    row_num = start_row
    for i, t in enumerate(times):
        f = force_n[i]
        if np.isnan(f):
            continue
        row_num += 1
        row = {
            "Row Number": row_num,
            "Recording Name": rec_name,
            "Time": f"{t:.6f}",
            "Theta": theta,
            "Phi": phi,
            "Force": f"{f:.6f}",
        }
        for j in range(NUM_SENSORS):
            if j < n_s:
                row[f"S{j + 1}X"] = f"{xyz[i, j, 0]:.4f}"
                row[f"S{j + 1}Y"] = f"{xyz[i, j, 1]:.4f}"
                row[f"S{j + 1}Z"] = f"{xyz[i, j, 2]:.4f}"
            else:
                row[f"S{j + 1}X"] = ""
                row[f"S{j + 1}Y"] = ""
                row[f"S{j + 1}Z"] = ""
        rows.append(row)

    return rows


def _append_master_csv(master_csv_path: Path, rows: list[dict]):
    file_exists = master_csv_path.exists()
    with open(master_csv_path, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=MASTER_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================================
# Quality-check text (immediate GUI feedback)
# ============================================================================

def _quality_check_text(force_n: np.ndarray, mags: np.ndarray,
                        events: list[tuple[int, int, int]]) -> str:
    n_events = len(events)
    header = f"{n_events} press{'es' if n_events != 1 else ''} detected"

    if len(force_n) and not np.all(np.isnan(force_n)):
        force_line = f"Force range: {np.nanmin(force_n):.3f} to {np.nanmax(force_n):.3f} N"
    else:
        force_line = "Force range: no data"

    responding, flat = [], []
    n_s = mags.shape[1] if mags.ndim == 2 else 0
    for j in range(n_s):
        col = mags[:, j]
        change = float(col.max() - col.min()) if len(col) else 0.0
        if change > SENSOR_NOISE_THRESHOLD:
            responding.append((j + 1, change))
        else:
            flat.append(j + 1)

    resp_str = ("Sensor " + ", ".join(f"{n} (+{c:.0f})" for n, c in responding)
               if responding else "none")
    flat_str = ("Sensor " + ", ".join(str(n) for n in flat) if flat else "none")
    sensor_line = f"Responding: {resp_str} | Flat: {flat_str}"

    return "\n".join([header, force_line, sensor_line])


# ============================================================================
# Summary-file bookkeeping (which recordings have already been added)
# ============================================================================

def _existing_summary_names(summary_csv_path: Path) -> set[str]:
    """Recording filenames already present in characterisation_summary.csv."""
    if not summary_csv_path.exists():
        return set()
    with open(summary_csv_path, 'r', newline='', encoding='utf-8') as fh:
        return {row.get("Recording Filename", "") for row in csv.DictReader(fh)}


def _existing_master_names(master_csv_path: Path) -> set[str]:
    """Recording names already present in master_characterisation.csv."""
    if not master_csv_path.exists():
        return set()
    with open(master_csv_path, 'r', newline='', encoding='utf-8') as fh:
        return {row.get("Recording Name", "") for row in csv.DictReader(fh)}


# ============================================================================
# Public entry points
# ============================================================================

def process_recording(json_path: Path, rec_dir: Path, data: dict[str, Any]) -> str:
    """Run the automatic pipeline for one just-saved recording.

    Exports the 4 plots + 4 CSVs into *rec_dir* and returns a short
    quality-check text summary for immediate display in the GUI.

    Does NOT touch master_characterisation.csv or characterisation_summary.csv
    — see update_summary_files() for that.
    """
    series = _extract_series(data)
    if series is None:
        return "No snapshots recorded — nothing to analyse."
    times, force_n, mags, xyz = series

    rec_name = json_path.stem
    _export_all(rec_dir, rec_name, times, force_n, mags, xyz)

    events = detect_press_events(force_n)
    return _quality_check_text(force_n, mags, events)


def update_summary_files(recordings_root: Path, unit: str | None = None) -> str:
    """Scan *recordings_root* for recording JSON files (one per subfolder) and
    append rows — for the given *unit* only — to that unit's
    characterisation_summary / master_characterisation CSVs.

    *unit* is the free-text value of the GUI's Unit field. Blank or "unit1"
    (case-insensitive) routes to the original characterisation_summary.csv /
    master_characterisation.csv, matching recordings with no "unit" metadata
    (made before this field existed) or "unit1". Any other value routes to
    characterisation_summary_{unit}.csv / master_characterisation_{unit}.csv,
    matching only recordings whose metadata "unit" equals that value — so
    each fingertip's data stays in its own files.

    Safe to call repeatedly: recordings already present (checked by
    Recording Filename / Recording Name) are skipped, so no duplicate rows
    are ever written. Returns a short confirmation message.
    """
    recordings_root = Path(recordings_root)
    unit_slug = _sanitize_unit(unit)
    is_default = _is_default_unit(unit_slug)
    summary_csv_path = recordings_root / _summary_csv_name(unit_slug)
    master_csv_path = recordings_root / _master_csv_name(unit_slug)

    existing_summary = _existing_summary_names(summary_csv_path)
    existing_master = _existing_master_names(master_csv_path)

    def _matches_unit(metadata: dict) -> bool:
        rec_unit_slug = _sanitize_unit(metadata.get("unit"))
        if is_default:
            return _is_default_unit(rec_unit_slug)
        return rec_unit_slug == unit_slug

    n_new = 0
    for json_path in sorted(recordings_root.glob("*/*.json")):
        rec_name = json_path.stem

        try:
            with open(json_path, encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception:
            continue

        metadata = data.get("metadata", {})
        if not _matches_unit(metadata):
            continue

        in_summary = json_path.name in existing_summary
        in_master = rec_name in existing_master
        if in_summary and in_master:
            continue

        series = _extract_series(data)
        if series is None:
            continue
        times, force_n, mags, xyz = series

        if not in_summary:
            events = detect_press_events(force_n)
            rows = _build_summary_rows(json_path, data, force_n, mags, events)
            _append_summary_csv(summary_csv_path, rows)
            existing_summary.add(json_path.name)

        if not in_master:
            start_row = _count_existing_master_rows(master_csv_path)
            master_rows = _build_master_rows(
                rec_name, metadata, times, force_n, xyz, start_row)
            _append_master_csv(master_csv_path, master_rows)
            existing_master.add(rec_name)

        n_new += 1

    label = _unit_label(unit_slug)
    if n_new == 0:
        return f"No new recordings found ({label})."
    return (f"Updated ({label}): {n_new} new recording{'s' if n_new != 1 else ''} "
            f"added → {summary_csv_path.name} / {master_csv_path.name}")
