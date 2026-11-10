"""
Unified Sensor Controller
==========================
Backend that orchestrates simultaneous data collection from:
  - eFlesh magnetic sensor array  (EfleshController)
  - LoadScale force sensor        (LoadScaleReader)

Recording is *event-driven*: every incoming magnetic frame triggers a
snapshot, and the LoadScale is polled at that moment.  This means the
recording rate naturally matches the magnetic sensor's native frame
rate — no data is lost and no artificial sleep loop is used. Although
the force MCU reports at a lower rate, force datais polled at the same
rate at the magnetic sensor, results in a stair-step effect in the force readings.

Produces timestamped JSON records with per-group magnetic readings
(x, y, z per sensor), force data, and user-supplied metadata.

Usage:
    from unified_controller import UnifiedController

    ctrl = UnifiedController(data_dir="path/to/output")   # required
    ctrl.set_metadata(theta=45.0, phi=30.0)
    ctrl.connect_all()

    ctrl.start_recording()          # one snapshot per magnetic frame
    # ... time passes ...
    path = ctrl.stop_recording()    # saves JSON automatically

    # Or single snapshot:
    snapshot = ctrl.capture_snapshot()
    ctrl.save_snapshot(snapshot)
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from MagneticSensors.eflesh_controller import (
    NUM_GROUPS,
    NUM_SENSORS,
    EfleshController,
    FrameData,
)
from LoadScale.load_scale import LoadScaleReader
from recording_analysis import process_recording, update_summary_files


# ============================================================================
# Data model
# ============================================================================

@dataclass
class SensorSnapshot:
    """Single sensor's reading at a point in time."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    magnitude: float = 0.0


@dataclass
class GroupSnapshot:
    """One mux group (5 sensors) at a point in time."""
    group_index: int = 0
    sensors: list[SensorSnapshot] = field(default_factory=lambda: [
        SensorSnapshot() for _ in range(NUM_SENSORS)
    ])


def _frame_to_groups(frame: FrameData) -> list[GroupSnapshot]:
    """Convert a FrameData into a list of GroupSnapshot."""
    groups = []
    for gi in range(NUM_GROUPS):
        sensors = []
        for si in range(NUM_SENSORS):
            r = frame.groups[gi][si]
            sensors.append(SensorSnapshot(
                x=r.x, y=r.y, z=r.z, magnitude=r.magnitude,
            ))
        groups.append(GroupSnapshot(group_index=gi, sensors=sensors))
    return groups


@dataclass
class DataRecord:
    """A single timestamped record containing all sensor data + metadata."""
    timestamp: float = 0.0
    timestamp_iso: str = ""
    force: Optional[float] = None          # LoadScale reading (grams)
    groups: list[GroupSnapshot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Unified Controller
# ============================================================================

class UnifiedController:
    """Manages both the magnetic sensor array and the load-scale sensor.

    Recording is event-driven: a snapshot is captured every time a
    magnetic frame arrives.
    The LoadScale force reading is polled at that moment — no separate
    polling thread, no sleep loop, no data loss.

    Callbacks (set before calling connect_*):
        on_status(msg: str)              – status / info messages
        on_error(msg: str)               – non-fatal errors
        on_mag_frame(frame: FrameData)   – raw magnetic frame (optional)
        on_record(record: DataRecord)    – each recorded snapshot
    """

    def __init__(self, data_dir: str | Path):
        """*data_dir* — required path where JSON recordings are saved.
        The directory is created if it does not exist.
        """
        # --- sensor controllers ---
        self._mag = EfleshController()
        self._load: LoadScaleReader | None = None

        # --- output directory ---
        self._data_dir = Path(data_dir).resolve()
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # --- recording state (event-driven, no separate thread) ---
        self._recording = False
        self._records: list[DataRecord] = []
        self._metadata: dict[str, Any] = {}
        self._mag_connected = False

        # latest values (lock for thread safety)
        self._latest_frame: FrameData | None = None
        self._latest_force: float | None = None
        self._lock = threading.Lock()

        # --- callbacks ---
        self.on_status: Callable[[str], None] | None = None
        self.on_error: Callable[[str], None] | None = None
        self.on_mag_frame: Callable[[FrameData], None] | None = None
        self.on_record: Callable[[DataRecord], None] | None = None
        self.on_load_serial_line: Callable[[str], None] | None = None
        self.on_quality_check: Callable[[str], None] | None = None

        # --- wire magnetic callbacks ---
        self._mag.on_frame = self._on_mag_frame
        self._mag.on_status = lambda m: self._emit(self.on_status, f"[Mag] {m}")
        self._mag.on_error = lambda m: self._emit(self.on_error, f"[Mag] {m}")
        self._mag.on_connected = lambda p: self._on_mag_connected(p)
        self._mag.on_disconnected = lambda: self._on_mag_disconnected()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        """Directory where JSON recordings are saved."""
        return self._data_dir

    def set_data_dir(self, path: str | Path) -> None:
        """Change the output directory without reconnecting sensors.

        The new directory is created if it does not exist.
        Safe to call at any time — before or after connection, during recording.
        """
        self._data_dir = Path(path).resolve()
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._emit(self.on_status, f"Output directory changed to: {self._data_dir}")

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def mag_connected(self) -> bool:
        return self._mag_connected

    @property
    def load_connected(self) -> bool:
        return self._load is not None

    @property
    def magnetic_controller(self) -> EfleshController:
        """The underlying EfleshController, e.g. to share its serial
        connection with another subsystem (see AmazingHand/hand_controller.py)."""
        return self._mag

    @property
    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def latest_force(self) -> float | None:
        with self._lock:
            return self._latest_force

    @property
    def latest_frame(self) -> FrameData | None:
        with self._lock:
            return self._latest_frame

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_metadata(self, **kwargs: Any) -> None:
        """Set per-run metadata fields (e.g. theta=45.0, phi=30.0, label='test').

        Metadata is embedded in every snapshot recorded from this point onward.
        """
        with self._lock:
            self._metadata.update(kwargs)
        self._emit(self.on_status, f"Metadata updated: {self._metadata}")

    def clear_metadata(self) -> None:
        """Remove all per-run metadata."""
        with self._lock:
            self._metadata.clear()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect_magnetic(self, port: str | None = None, baud: int = 2_000_000) -> None:
        """Connect to the eFlesh magnetic sensor array."""
        self._emit(self.on_status, "Connecting to magnetic sensor array...")
        self._mag.start(port=port, baud=baud)

    def connect_loadscale(self, port: str | None = None) -> None:
        """Connect to the LoadScale force sensor.

        Must be called *after* connect_magnetic() if auto-excluding the
        magnetic sensor's port is desired (they share serial discovery).

        Raises RuntimeError if no LoadScale device is found.
        """
        exclude = {self._mag.connected_port} if self._mag.connected_port else set()
        self._emit(self.on_status, "Connecting to LoadScale force sensor...")
        if self._load is not None:
            self._load.close()
        self._load = LoadScaleReader(serial_port=port, exclude_ports=exclude)
        self._load.on_line = self._on_load_serial_line
        # Replay boot lines captured during handshake
        for line in self._load._boot_lines:
            self._on_load_serial_line(line)
        self._emit(self.on_status, f"LoadScale connected on {self._load.serial_connection.port}")

    def connect_all(self,
                    mag_port: str | None = None,
                    load_port: str | None = None,
                    mag_baud: int = 2_000_000) -> None:
        """Connect to both sensors.

        Connects magnetic first so LoadScale can exclude that port.
        Raises RuntimeError if either sensor cannot be found/connected.
        """
        self.connect_magnetic(port=mag_port, baud=mag_baud)
        self._wait_for_magnetic_connection(timeout_s=5.0)
        try:
            self.connect_loadscale(port=load_port)
        except Exception:
            self._mag.stop()
            self._mag_connected = False
            raise

    def disconnect_all(self) -> None:
        """Disconnect from both sensors."""
        self.stop_recording()
        self._mag.stop()
        if self._load:
            self._load.close()
            self._load = None
        self._mag_connected = False
        self._emit(self.on_status, "All sensors disconnected.")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Begin recording.

        A snapshot is captured on every incoming magnetic frame, so the
        recording rate equals the magnetic sensor's native frame rate.
        LoadScale force data is included if connected, otherwise force is None.

        Raises RuntimeError if magnetic sensor is not connected.
        """
        if not self._mag_connected:
            raise RuntimeError("Cannot start recording: magnetic sensor not connected.")
        with self._lock:
            if self._recording:
                return
            self._records.clear()
            self._recording = True
        self._emit(self.on_status, "Recording started...")

    def stop_recording(self, filename_stem: str | None = None) -> str | None:
        """Stop recording and save accumulated data to a JSON file.

        Args:
            filename_stem: Optional custom filename stem (without extension).
                           Used verbatim (sanitized) as the saved file's name.
                           If omitted, an auto-generated timestamped name is used.

        Returns the path to the saved file, or None if no data was recorded.
        """
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            records_to_save = list(self._records)
            metadata_snapshot = dict(self._metadata)

        path = self._save_to_json(
            records=records_to_save,
            metadata=metadata_snapshot,
            filename_stem=filename_stem,
        )
        n = len(records_to_save)
        self._emit(self.on_status, f"Recording stopped — {n} snapshots saved → {path}")
        return path

    def capture_snapshot(self) -> DataRecord:
        """Capture a single snapshot now (does not require recording mode)."""
        return self._build_record()

    def save_snapshot(self, record: DataRecord,
                      filepath: str | Path | None = None) -> str:
        """Save a single snapshot to a JSON file."""
        if filepath is None:
            filepath = self._generate_filename("snapshot")
        filepath = Path(filepath)
        record_dict = self._record_to_dict(record)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record_dict, f, indent=2)
        return str(filepath)

    # ------------------------------------------------------------------
    # Internal: callbacks from magnetic controller
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Load cell calibration helpers
    # ------------------------------------------------------------------

    def poll_force(self) -> float | None:
        """Explicitly poll the load cell and update the cached force reading.

        Call this from a GUI timer to get live readings without needing the
        magnetic sensor to be running.
        """
        if self._load is None:
            return None
        force = self._load.read_force()
        if force is not None:
            with self._lock:
                self._latest_force = force
        return force

    @property
    def loadscale_cal_factor(self) -> float | None:
        return self._load.cal_factor if self._load else None

    def tare_loadscale(self) -> None:
        """Send tare command to the LoadScale."""
        if self._load is None:
            raise RuntimeError("LoadScale not connected.")
        self._load.tare()
        self._emit(self.on_status, "[Load] Tare command sent.")

    def set_loadscale_cal_factor(self, factor: float) -> bool:
        """Send a calibration factor to the LoadScale and save to EEPROM."""
        if self._load is None:
            raise RuntimeError("LoadScale not connected.")
        ok = self._load.set_calibration_factor(factor)
        if ok:
            self._emit(self.on_status, f"[Load] Calibration factor set: {factor:.6f}")
        else:
            self._emit(self.on_error, "[Load] Calibration factor save timed out.")
        return ok

    def compute_and_apply_cal(self, known_mass_g: float) -> bool:
        """Compute a new calibration factor from the current reading and apply it.

        Formula: new_factor = latest_reading * current_factor / known_mass_g
        """
        if self._load is None:
            raise RuntimeError("Load cell not connected.")
        if self._load.cal_factor is None:
            raise RuntimeError(
                "Calibration factor unknown — set it manually first, "
                "or reconnect so the firmware reports it."
            )
        force = self.poll_force()
        if force is None:
            raise RuntimeError("No force reading from load cell — check connection.")
        if known_mass_g <= 0:
            raise ValueError("Known mass must be > 0.")
        new_factor = (force * self._load.cal_factor) / known_mass_g
        return self.set_loadscale_cal_factor(new_factor)

    def _on_load_serial_line(self, line: str) -> None:
        self._emit(self.on_load_serial_line, line)

    def _on_mag_frame(self, frame: FrameData) -> None:
        """Called by EfleshController for every complete binary frame.

        Updates the latest frame cache and, if recording, captures a
        snapshot (event-driven, no sleep).
        """
        with self._lock:
            self._latest_frame = frame
            should_record = self._recording

        # Forward to external listener (e.g. GUI for live display)
        if self.on_mag_frame:
            self.on_mag_frame(frame)

        # --- event-driven recording ---
        if should_record:
            try:
                record = self._build_record()
            except Exception as exc:
                self._emit(self.on_error, f"Failed to build record: {exc}")
                return
            with self._lock:
                if not self._recording:
                    return
                self._records.append(record)
            self._emit(self.on_record, record)

    def _on_mag_connected(self, port: str) -> None:
        self._mag_connected = True
        self._emit(self.on_status, f"Magnetic sensor array connected on {port}")

    def _on_mag_disconnected(self) -> None:
        self._mag_connected = False
        self._emit(self.on_status, "Magnetic sensor array disconnected.")

    def _build_record(self) -> DataRecord:
        """Construct a DataRecord from the latest sensor readings.

        Precondition: both sensors must be connected.
        """
        now = time.time()

        if self._load is None:
            raise RuntimeError("Cannot build record: LoadScale is not connected.")

        # read force from LoadScale (guaranteed connected)
        force = self._load.read_force()
        with self._lock:
            self._latest_force = force

        # grab latest magnetic frame and metadata snapshot
        with self._lock:
            frame = self._latest_frame
            metadata = dict(self._metadata)

        groups = _frame_to_groups(frame) if frame is not None else []

        return DataRecord(
            timestamp=now,
            timestamp_iso=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            force=force,
            groups=groups,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal: serialisation
    # ------------------------------------------------------------------

    def _record_to_dict(self, record: DataRecord) -> dict[str, Any]:
        """Convert a DataRecord to a plain dict for JSON serialisation."""
        return {
            "timestamp": record.timestamp,
            "timestamp_iso": record.timestamp_iso,
            "force_g": record.force,
            "groups": [
                {
                    "group_index": g.group_index,
                    "sensors": [asdict(s) for s in g.sensors],
                }
                for g in record.groups
            ],
            "metadata": record.metadata,
        }

    def _generate_filename(self, prefix: str = "recording") -> Path:
        """Generate a unique timestamped filename."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._data_dir / f"{prefix}_{ts}.json"

    def _save_to_json(self,
                      records: list[DataRecord],
                      metadata: dict[str, Any],
                      filename_stem: str | None = None) -> str:
        """Save recorded snapshots to a JSON file.

        Each recording gets its own subfolder under data_dir (named after the
        recording's filename stem) containing the JSON file plus the
        auto-generated force/magnetic PNG + CSV plots (see recording_analysis.py).
        """
        if filename_stem:
            stem = re.sub(r"[^\w\-]", "_", filename_stem).strip("_")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            theta = metadata.get("theta")
            phi = metadata.get("phi")
            if theta is not None and phi is not None:
                stem = f"eflesh_{ts}_t{theta:.0f}_p{phi:.0f}"
            else:
                stem = f"eflesh_{ts}"

        rec_dir = self._data_dir / stem
        rec_dir.mkdir(parents=True, exist_ok=True)
        filepath = rec_dir / f"{stem}.json"

        output = {
            "recording_info": {
                "total_snapshots": len(records),
                "recorded_at_iso": datetime.now(timezone.utc).isoformat(),
                "sensors": {
                    "magnetic": self._mag_connected,
                    "loadscale": self.load_connected,
                },
            },
            "metadata": metadata,
            "snapshots": [self._record_to_dict(r) for r in records],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        if records:
            try:
                quality_text = process_recording(filepath, rec_dir, output)
                self._emit(self.on_quality_check, quality_text)
            except Exception as exc:
                self._emit(self.on_error, f"Analysis pipeline failed: {exc}")

        return str(filepath)

    def update_summary_files(self, unit: str | None = None) -> str:
        """Scan the recordings folder and append any recordings for *unit*
        not yet reflected in that unit's master_characterisation /
        characterisation_summary CSVs (see recording_analysis.update_summary_files
        for the unit → filename mapping).

        Manually triggered (e.g. from the Data Analysis tab's "Update Summary
        Files" button) — recordings are no longer added to these files
        automatically on Stop Recording.
        """
        return update_summary_files(self._data_dir, unit=unit)

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit(callback: Callable | None, *args: Any) -> None:
        if callback:
            try:
                callback(*args)
            except Exception:
                pass

    def _wait_for_magnetic_connection(self, timeout_s: float = 5.0) -> None:
        """Wait until the magnetic controller reports connected."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._mag_connected:
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for magnetic sensor connection.")


# ============================================================================
# Standalone CLI usage
# ============================================================================

def main() -> None:
    """Quick CLI test: connect, record 5 seconds, save."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Unified sensor data collector")
    parser.add_argument("--mag-port", help="Magnetic sensor COM port")
    parser.add_argument("--load-port", help="LoadScale COM port")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Recording duration in seconds")
    parser.add_argument("--theta", type=float, default=0.0, help="Theta angle (degrees)")
    parser.add_argument("--phi", type=float, default=0.0, help="Phi angle (degrees)")
    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for JSON files (required)")
    args = parser.parse_args()

    ctrl = UnifiedController(data_dir=args.output_dir)
    ctrl.on_status = print
    ctrl.on_error = print

    ctrl.set_metadata(theta=args.theta, phi=args.phi)

    print("Connecting...")
    ctrl.connect_all(mag_port=args.mag_port, load_port=args.load_port)

    # Wait for connections to stabilise
    time.sleep(2.0)

    if not ctrl.mag_connected:
        print("ERROR: Magnetic sensor not connected. Exiting.")
        ctrl.disconnect_all()
        sys.exit(1)

    if not ctrl.load_connected:
        print("ERROR: LoadScale not connected. Exiting.")
        ctrl.disconnect_all()
        sys.exit(1)

    print(f"Recording — press Ctrl+C to stop (max {args.duration:.0f}s)...")
    ctrl.start_recording()
    try:
        time.sleep(args.duration)
    finally:
        path = ctrl.stop_recording()

    if path:
        print(f"Saved {ctrl.record_count} snapshots → {path}")
    else:
        print("No data recorded.")

    ctrl.disconnect_all()


if __name__ == "__main__":
    main()
