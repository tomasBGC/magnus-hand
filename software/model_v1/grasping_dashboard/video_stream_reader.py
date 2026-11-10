"""Video stream reader -- TCP client for hand_tracker_bridge.py's live
camera preview (length-prefixed JPEG frames: 4-byte big-endian length,
then that many bytes of JPEG). Decoded via Qt's own built-in JPEG codec
(QPixmap.loadFromData) so this process needs no OpenCV/PIL of its own --
the same reason that bridge runs in a separate Python 3.12 venv in the
first place (see model_v1/hand_tracking/README.md).

Exposes ONLY "give me the latest frame as a QPixmap" (via a signal, since
frames arrive asynchronously whenever the camera produces one -- there is
no meaningful synchronous poll() the other readers have). No display
logic (the GUI decides where/how to show it), no saving logic, no
knowledge of the other readers.
"""
from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtNetwork

DEFAULT_VIDEO_PORT = 9998  # matches hand_tracker_bridge.py's own default


class VideoStreamReader(QtCore.QObject):
    frame_received = QtCore.pyqtSignal(QtGui.QPixmap)
    connected = QtCore.pyqtSignal()
    disconnected = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._socket: QtNetwork.QTcpSocket | None = None
        self._buffer = QtCore.QByteArray()
        self._expected_len: int | None = None
        self._latest_pixmap: QtGui.QPixmap | None = None

    @property
    def is_connected(self) -> bool:
        return (self._socket is not None
                and self._socket.state() == QtNetwork.QAbstractSocket.SocketState.ConnectedState)

    def connect_to_bridge(self, host: str = "127.0.0.1", port: int = DEFAULT_VIDEO_PORT) -> None:
        if self._socket is not None:
            self.disconnect_from_bridge()
        sock = QtNetwork.QTcpSocket(self)
        sock.connected.connect(self.connected)
        sock.disconnected.connect(self._on_disconnected)
        sock.errorOccurred.connect(lambda _e: self.error.emit(sock.errorString()))
        sock.readyRead.connect(self._on_ready_read)
        self._socket = sock
        self._buffer.clear()
        self._expected_len = None
        sock.connectToHost(host, port)

    def disconnect_from_bridge(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._buffer.clear()
        self._expected_len = None

    def _on_disconnected(self) -> None:
        self.disconnected.emit()

    def _on_ready_read(self) -> None:
        sock = self._socket
        if sock is None:
            return
        self._buffer.append(sock.readAll())
        while True:
            if self._expected_len is None:
                if len(self._buffer) < 4:
                    return
                length_bytes = bytes(self._buffer[:4])
                self._expected_len = int.from_bytes(length_bytes, "big")
                self._buffer = self._buffer[4:]
            if len(self._buffer) < self._expected_len:
                return
            jpeg_bytes = bytes(self._buffer[:self._expected_len])
            self._buffer = self._buffer[self._expected_len:]
            self._expected_len = None
            pixmap = QtGui.QPixmap()
            if pixmap.loadFromData(jpeg_bytes, "JPG"):
                self._latest_pixmap = pixmap
                self.frame_received.emit(pixmap)

    def latest_pixmap(self) -> QtGui.QPixmap | None:
        return self._latest_pixmap
