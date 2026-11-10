"""SessionRecorder: the one place this dashboard's logic lives. Ticks at
the same rate as the existing sensing loop (~20Hz, sensing_gui.
DISPLAY_INTERVAL_S), polls all three readers, combines their latest values
into one timestamped row, appends it to the active episode's log file (if
any), and emits it for the GUI to display. The GUI has no polling logic of
its own -- it only reacts to row_ready and forwards button presses here.

Ground-truth marking: set_genuine_contact_marked(True/False) is the single
source of truth for the "genuine_contact_marked" field stamped on every
row while it's active -- the independent, human-judged verification signal
this whole dashboard exists to provide (see module-level docstring in
dashboard_gui.py for why: this project has repeatedly found its automated
"verified contact" signals unreliable -- see the verified-ground-truth-
artifact-critical incident -- a human holding a button while they
genuinely feel contact is a completely independent check against that).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from PyQt6 import QtCore


class SessionRecorder(QtCore.QObject):
    row_ready = QtCore.pyqtSignal(dict)

    def __init__(self, sensing_reader, load_cell_reader, hand_tracking_reader,
                 out_dir: str | Path, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._sensing = sensing_reader
        self._load_cell = load_cell_reader
        self._hand_tracking = hand_tracking_reader
        self._out_dir = Path(out_dir)

        self._session_id: str | None = None
        self._session_start_iso: str | None = None
        self._session_end_iso: str | None = None
        self._session_meta_path: Path | None = None
        self._object_used: str = ""
        self._episode_outcomes: dict[int, str] = {}

        self._episode_id: int | None = None
        self._episode_counter = 0
        self._log_file = None
        self._episode_meta: dict = {}

        self._genuine_contact_marked = False
        self._rows_logged_this_session = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, object_used: str = "") -> str:
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_start_iso = datetime.now(timezone.utc).isoformat()
        self._object_used = object_used
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._session_meta_path = self._out_dir / f"session_{self._session_id}_meta.json"
        self._write_meta()
        return self._session_id

    def end_session(self) -> None:
        if self._episode_id is not None:
            self.end_episode("aborted")
        self._session_end_iso = datetime.now(timezone.utc).isoformat()
        self._write_meta()

    def set_object_used(self, object_used: str) -> None:
        self._object_used = object_used
        self._write_meta()

    def _write_meta(self) -> None:
        if self._session_meta_path is None:
            return
        meta = {
            "session_id": self._session_id,
            "session_start_iso": self._session_start_iso,
            "session_end_iso": self._session_end_iso,
            "object_used": self._object_used,
            "episode_outcomes": self._episode_outcomes,
        }
        self._session_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def episode_active(self) -> bool:
        return self._episode_id is not None

    @property
    def rows_logged_this_session(self) -> int:
        return self._rows_logged_this_session

    def start_episode(self, meta: dict | None = None) -> int:
        """meta: optional scenario/level/target_force_n/is_control info (the
        scenario runner's own bookkeeping, added 2026-09-04 for the load-
        cell training protocol) -- stamped onto every row logged during this
        episode via the "episode_meta" field, and merged into this episode's
        entry in episode_outcomes at end_episode() time. Manual Start
        Episode clicks just pass nothing, same as before."""
        if self._session_id is None:
            self.start_session()
        self._episode_counter += 1
        self._episode_id = self._episode_counter
        self._episode_meta = dict(meta) if meta else {}
        path = self._out_dir / f"session_{self._session_id}_episode{self._episode_id}.jsonl"
        self._log_file = path.open("a", encoding="utf-8")
        return self._episode_id

    def update_episode_meta(self, **kwargs) -> None:
        """Merge additional fields into the active episode's meta (e.g. the
        scenario runner advancing "phase" from approach->hold->release) --
        takes effect on the NEXT tick's row, same as any other live state."""
        self._episode_meta.update(kwargs)

    def end_episode(self, outcome: str, result: dict | None = None) -> None:
        if self._episode_id is None:
            return
        self._episode_outcomes[self._episode_id] = {
            "outcome": outcome, **self._episode_meta, **(result or {}),
        }
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self._episode_id = None
        self._episode_meta = {}
        self._write_meta()

    # ------------------------------------------------------------------
    # Ground-truth marking
    # ------------------------------------------------------------------

    def set_genuine_contact_marked(self, active: bool) -> None:
        self._genuine_contact_marked = bool(active)

    @property
    def genuine_contact_marked(self) -> bool:
        return self._genuine_contact_marked

    # ------------------------------------------------------------------
    # The tick
    # ------------------------------------------------------------------

    def tick(self) -> dict:
        """Call once per timer interval (see dashboard_gui.py's tick
        timer, at sensing_gui.DISPLAY_INTERVAL_S). SensingReader.poll() is
        called HERE ONLY -- see that reader's own docstring for why calling
        it more than once per tick is unsafe."""
        sensing_new = self._sensing.poll()
        sensing = sensing_new if sensing_new is not None else self._sensing.latest()
        load_cell_force = self._load_cell.poll() if self._load_cell is not None else None
        hand_tracking_position = self._hand_tracking.latest() if self._hand_tracking is not None else {}

        row = {
            "t": time.time(),
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "episode_id": self._episode_id,
            "sensing": sensing,
            "load_cell_force": load_cell_force,
            "hand_tracking_position": dict(hand_tracking_position),
            "genuine_contact_marked": self._genuine_contact_marked,
            "episode_meta": dict(self._episode_meta) if self._episode_id is not None else {},
        }

        if self._episode_id is not None and self._log_file is not None:
            self._log_file.write(json.dumps(row) + "\n")
            self._log_file.flush()
            self._rows_logged_this_session += 1

        self.row_ready.emit(row)
        return row
