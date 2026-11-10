"""Load cell reader -- thin wrapper around unified_controller.py's
UnifiedController / LoadScaleReader logic.

Exposes ONLY "give me the latest load cell force reading". No knowledge of
the sensing or hand-tracking readers, no display logic, no saving logic.
Imports unified_controller.py unmodified; does not edit it, and never
calls its magnetic-sensor or recording/snapshot methods -- SensingReader
and SessionRecorder already own those concerns respectively, independently.
"""
from __future__ import annotations

from pathlib import Path

from unified_controller import UnifiedController

# unified_controller.py's LoadScaleReader reports mass in GRAMS (see its own
# `force: Optional[float]  # LoadScale reading (grams)` comment); this
# reader converts to Newtons (standard gravity) at the boundary, once, so
# every consumer (SessionRecorder's logged rows, the GUI display) sees
# Newtons consistently -- matching the unit every OTHER force value in this
# project (sensing_gui.py's Force/raw_force) already uses, rather than
# mixing grams and Newtons across the same logged row.
GRAMS_TO_NEWTONS = 9.80665 / 1000.0


class LoadCellReader:
    """poll() actively reads the load cell (matching UnifiedController.
    poll_force()'s own "call this from a GUI timer" contract) and should be
    called once per tick by SessionRecorder. latest() returns the cached
    value without triggering a new read. Both return Newtons."""

    def __init__(self, data_dir: str | Path):
        # data_dir is UnifiedController's constructor requirement (used
        # internally for its own snapshot-filename generation, a feature
        # this reader never calls) -- harmless to pass grasping_dashboard's
        # own data directory here.
        self._ctrl = UnifiedController(data_dir=data_dir)
        self._connected = False

    def connect_hardware(self, port: str | None = None) -> None:
        self._ctrl.connect_loadscale(port=port)
        self._connected = True

    @property
    def connected(self) -> bool:
        return self._connected

    def poll(self) -> float | None:
        """Actively reads the load cell and updates the cached value.
        Returns Newtons, or None if not connected or no reading available."""
        if not self._connected:
            return None
        grams = self._ctrl.poll_force()
        return grams * GRAMS_TO_NEWTONS if grams is not None else None

    def latest(self) -> float | None:
        """Cached last reading (Newtons), no new hardware read."""
        if not self._connected:
            return None
        grams = self._ctrl.latest_force
        return grams * GRAMS_TO_NEWTONS if grams is not None else None

    def tare(self) -> None:
        if self._connected:
            self._ctrl.tare_loadscale()

    def close(self) -> None:
        if self._connected and self._ctrl._load is not None:
            self._ctrl._load.close()
        self._connected = False
