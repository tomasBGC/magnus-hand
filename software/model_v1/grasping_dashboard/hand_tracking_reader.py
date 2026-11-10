"""Hand-tracking reader -- wraps the SAME UDP listener protocol already
built in teleop_data_collection.py's Hand Tracking panel (a QUdpSocket
bound to a port, receiving {"Index": pct, "Middle": pct, "Ring": pct,
"Thumb": pct} JSON datagrams from model_v1/hand_tracking/
hand_tracker_bridge.py).

That listener lives as instance methods on teleop_data_collection.py's
TeleopMainWindow (a GUI class that ALSO drives the hand's servos from it,
which this dashboard must not do -- per the required reader architecture,
this is a passive OBSERVER of tracked hand position, not a controller).
Since there is no standalone function/class in that file to import (the
logic is a few lines embedded in GUI event handlers), this reader
re-implements the same small protocol independently rather than importing
from or modifying that file -- same wire format, same default port, so
model_v1/hand_tracking/hand_tracker_bridge.py works unmodified as the
sender for either consumer.

Exposes ONLY "give me the latest per-finger tracked position". No
knowledge of the sensing or load cell readers, no display logic, no
saving logic, and NEVER sends anything to the hand.
"""
from __future__ import annotations

import json

from PyQt6 import QtCore, QtNetwork

FINGER_LABELS = ["Index", "Middle", "Ring", "Thumb"]
DEFAULT_UDP_PORT = 9999  # matches hand_tracker_bridge.py's own default
DEFAULT_CONTROL_PORT = 9997  # matches hand_tracker_bridge.py's --control-port default


class HandTrackingReader(QtCore.QObject):
    """latest() returns the most recently received datagram's per-finger
    percentages (0-100, 0=open/100=closed, same convention as
    hand_tracker_bridge.py's own calibrated output), or {} if nothing has
    arrived yet. Purely event-driven (UDP arrival) -- there is no
    meaningful separate poll() the way the other two readers have, since a
    camera frame's tracked position either has or hasn't arrived; callers
    on a fixed tick (SessionRecorder) should just call latest() every
    tick and accept a slightly-stale value between camera frames."""

    datagram_received = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._socket: QtNetwork.QUdpSocket | None = None
        self._latest: dict[str, float] = {}

    @property
    def listening(self) -> bool:
        return self._socket is not None

    def start_listening(self, port: int = DEFAULT_UDP_PORT) -> bool:
        if self._socket is not None:
            self.stop_listening()
        sock = QtNetwork.QUdpSocket(self)
        if not sock.bind(QtNetwork.QHostAddress.SpecialAddress.LocalHost, port):
            self.error.emit(f"Failed to bind UDP port {port}: {sock.errorString()}")
            return False
        sock.readyRead.connect(self._on_ready_read)
        self._socket = sock
        return True

    def stop_listening(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _on_ready_read(self) -> None:
        sock = self._socket
        if sock is None:
            return
        parsed = None
        while sock.hasPendingDatagrams():
            datagram, _host, _port = sock.readDatagram(sock.pendingDatagramSize())
            try:
                parsed = json.loads(bytes(datagram).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue  # malformed datagram, ignore and keep whatever's still pending
        if parsed is None:
            return  # only the LATEST datagram in this batch matters
        self._latest = {label: float(parsed[label]) for label in FINGER_LABELS if label in parsed}
        self.datagram_received.emit(self._latest)

    def latest(self) -> dict[str, float]:
        return self._latest

    # ------------------------------------------------------------------
    # Calibration control (2026-09-04): the bridge's own 'o'/'f' keypress
    # calibration only works if ITS separate native window has OS focus --
    # easy to miss, and impossible to give focus to from here anyway (this
    # is a different process). Sending it a tiny control datagram has the
    # identical effect (see hand_tracker_bridge.py's own control-socket
    # handling) without needing that window focused at all. This is a
    # one-off command, not a continuous read -- same "a reader can still
    # expose a small control action" precedent as LoadCellReader.tare().
    # ------------------------------------------------------------------

    def send_calibrate_open(self, host: str = "127.0.0.1", port: int = DEFAULT_CONTROL_PORT) -> None:
        self._send_control(b"CALIBRATE_OPEN", host, port)

    def send_calibrate_fist(self, host: str = "127.0.0.1", port: int = DEFAULT_CONTROL_PORT) -> None:
        self._send_control(b"CALIBRATE_FIST", host, port)

    def _send_control(self, payload: bytes, host: str, port: int) -> None:
        sock = QtNetwork.QUdpSocket(self)
        sock.writeDatagram(payload, QtNetwork.QHostAddress(host), port)
        sock.close()
