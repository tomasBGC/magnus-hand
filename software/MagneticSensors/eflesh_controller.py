"""
eFlesh Hardware Controller
===========================
Standalone serial controller for the eFlesh 4×5 multiplexed sensor array.

Handles device discovery via handshake protocol, text-header parsing,
and binary frame streaming.  Emits data through plain callbacks so it
can be reused in any Python application — GUI, CLI, logging, etc.

Protocol:
  1. MCU broadcasts  "EFLESH_HSK\\n"  every ~150 ms
  2. Controller sends "EFLESH_ACK\\n"
  3. MCU responds     "EFLESH_RDY\\n"  → handshake complete
  4. MCU sends text header, ending with "Starting data stream (binary)"
  5. MCU streams 324-byte binary frames

Usage (CLI / headless):
    from eflesh_controller import EfleshController

    def on_frame(frame):
        print(f"Frame {frame.seq}: group0[0] x={frame.groups[0][0].x:.1f}")

    ctrl = EfleshController()
    ctrl.on_frame = on_frame
    ctrl.on_status = print
    ctrl.start()          # auto-discovers and connects
    # ... Ctrl-C to stop ...
    ctrl.stop()

Usage (with Qt GUI):
    See eflesh_gui.py for a complete Qt integration example.
"""

from __future__ import annotations

import struct
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import serial
import serial.tools.list_ports

# ============================================================================
# Protocol Constants
# ============================================================================

NUM_GROUPS   = 4
NUM_SENSORS  = 5
FRAME_SIZE   = 4 + NUM_GROUPS * NUM_SENSORS * 16   # 2 magic + 2 seq + 320 data = 324
MAGIC_BYTE_0 = 0xAA
MAGIC_BYTE_1 = 0x55

# Handshake strings (must match lib/handshake/handshake.h)
HANDSHAKE_HSK = "EFLESH_HSK"
HANDSHAKE_ACK = "EFLESH_ACK"
HANDSHAKE_RDY = "EFLESH_RDY"
HANDSHAKE_HSK_B = HANDSHAKE_HSK.encode("ascii")
HANDSHAKE_RDY_B = HANDSHAKE_RDY.encode("ascii")

# Text header sentinel
HEADER_SENTINEL = "Starting data stream"

# How long to wait for HSK broadcast on a candidate port (seconds)
DISCOVERY_TIMEOUT = 1.5


def _find_subsequence(buf: bytearray, token: bytes) -> int:
    """Return index of token in buf, or -1 if not present."""
    return bytes(buf).find(token)


def _find_magic(buf: bytes | bytearray) -> int:
    """Return index of frame magic bytes 0xAA 0x55, or -1 if absent."""
    for i in range(len(buf) - 1):
        if buf[i] == MAGIC_BYTE_0 and buf[i + 1] == MAGIC_BYTE_1:
            return i
    return -1


# ============================================================================
# Data Model
# ============================================================================

@dataclass
class SensorReading:
    """One sensor's calibrated reading (t, x, y, z)."""
    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5


@dataclass
class FrameData:
    """Complete frame from the MCU: 4 groups × 5 sensors."""
    seq: int = 0
    timestamp: float = 0.0
    groups: list[list[SensorReading]] = field(default_factory=lambda: [
        [SensorReading() for _ in range(NUM_SENSORS)]
        for _ in range(NUM_GROUPS)
    ])


# ============================================================================
# Controller
# ============================================================================

class EfleshController:
    """Manages serial connection, handshake, and frame streaming.

    Callbacks (set before calling start()):
        on_frame(frame: FrameData)   — called for each complete binary frame
        on_status(message: str)      — status / info messages
        on_error(message: str)       — non-fatal errors
        on_connected(port: str)      — called after successful handshake
        on_disconnected()            — called when connection drops
        on_header_line(line: str)    — called for each text header line (optional)
    """

    def __init__(self):
        # Public callbacks — set these before start()
        self.on_frame:       Callable[[FrameData], None] | None = None
        self.on_status:      Callable[[str], None]       | None = None
        self.on_error:       Callable[[str], None]       | None = None
        self.on_connected:   Callable[[str], None]       | None = None
        self.on_disconnected: Callable[[], None]          | None = None
        self.on_header_line:  Callable[[str], None]       | None = None

        # Internal state
        self._ser:     serial.Serial | None = None
        self._thread:  threading.Thread | None = None
        self._running = False
        self._port:    str | None = None
        self._baud:    int = 2000000

        # Stats (read-only after start)
        self.frame_count: int = 0
        self.last_seq:    int = -1
        self.dropped:     int = 0
        self.bytes_received: int = 0
        self.sync_losses: int = 0
        self.parse_errors: int = 0
        self._binary_stash: bytes = b""  # pre-read binary data from reconnect fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, port: str | None = None, baud: int = 2000000):
        """Connect to the eFlesh device.

        If *port* is None, auto-discovers the device by scanning all
        available COM ports and performing the handshake.

        If *port* is given, connects directly (skips discovery).
        """
        if self._running:
            return

        self._baud = baud
        self.frame_count = 0
        self.last_seq = -1
        self.dropped = 0
        self.bytes_received = 0
        self.sync_losses = 0
        self.parse_errors = 0
        self._binary_stash = b""

        if port is not None:
            self._port = port
            self._emit_status(f"Connecting to {port}...")
        else:
            self._emit_status("Auto-discovering eFlesh device...")
            discovered = self._discover(baud)
            if discovered is None:
                self._emit_error("No eFlesh device found on any COM port.")
                return
            self._port = discovered

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Disconnect and stop the reader thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._emit_disconnected()
        self._emit_status("Disconnected.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def connected_port(self) -> str | None:
        return self._port

    @property
    def serial_connection(self) -> serial.Serial | None:
        """The open pyserial connection to the Teensy, once connected.

        Shared with other subsystems that talk to the same board over the
        same USB link (e.g. AmazingHand servo control) instead of opening a
        second connection to the same COM port. Reading is owned exclusively
        by this controller's background thread — other users of this
        connection must only write to it."""
        return self._ser

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    def _discover(self, baud: int) -> str | None:
        """Scan all available serial ports for an eFlesh device.

        Sends ACK repeatedly (every ~100 ms), responds to HSK broadcasts,
        and falls back to binary-stream detection for already-streaming MCUs.
        """
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            self._emit_status("No serial ports found on this system.")
            return None

        self._emit_status(f"Scanning {len(ports)} port(s) for eFlesh device...")

        for pi in ports:
            port_name = pi.device
            self._emit_status(f"  Trying {port_name}...")

            try:
                s = serial.Serial(port_name, baud, timeout=0.2)
            except Exception:
                continue

            try:
                s.reset_input_buffer()

                deadline = time.time() + DISCOVERY_TIMEOUT
                buf = bytearray()
                last_ack_time = 0.0

                while time.time() < deadline:
                    # Send ACK periodically (every 100 ms) so the MCU
                    # doesn't miss it during boot / enumeration.
                    now = time.time()
                    if now - last_ack_time >= 0.1:
                        s.write((HANDSHAKE_ACK + "\n").encode("ascii"))
                        s.flush()
                        last_ack_time = now

                    if s.in_waiting:
                        chunk = s.read(s.in_waiting)
                        buf.extend(chunk)

                        # --- Binary-stream fallback (MCU already streaming) ---
                        if _find_magic(buf) >= 0:
                            s.close()
                            self._emit_status(
                                f"  Found eFlesh device on {port_name} (binary stream detected)"
                            )
                            return port_name

                        # Match protocol tokens in raw bytes so interleaved
                        # binary payload cannot hide valid handshake strings.
                        if _find_subsequence(buf, HANDSHAKE_RDY_B) >= 0:
                            s.close()
                            self._emit_status(
                                f"  Found eFlesh device on {port_name}"
                            )
                            return port_name
                        if _find_subsequence(buf, HANDSHAKE_HSK_B) >= 0:
                            s.write((HANDSHAKE_ACK + "\n").encode("ascii"))
                            s.flush()
                            last_ack_time = time.time()

                        if len(buf) > 2048:
                            buf = buf[-512:]
                    else:
                        time.sleep(0.02)

                s.close()
            except Exception:
                try:
                    s.close()
                except Exception:
                    pass

        return None

    # ------------------------------------------------------------------
    # Main reader loop (runs in background thread)
    # ------------------------------------------------------------------

    def _run(self):
        """Background thread: open serial, handshake, read header, parse frames.

        Supports reconnection: if the MCU is already streaming binary frames
        (e.g. after a disconnect/reconnect), the handshake is skipped and we
        jump straight to frame parsing.
        """
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=0.2)
            self._ser.reset_input_buffer()
        except Exception as e:
            self._emit_error(f"Serial error: {e}")
            self._running = False
            return

        self._emit_status(f"Opened {self._port}, performing handshake...")

        # --- Handshake (with binary-stream fallback for reconnect) ---
        handshake_ok = self._do_handshake()

        if handshake_ok:
            self._emit_connected(self._port)

            # When _do_handshake detected a binary stream (reconnect),
            # it stashed frame data in _binary_stash.  Only skip the
            # header read when the stash actually contains binary data
            # (starts with magic bytes); otherwise the stash is just
            # leftover text that _read_header() should consume normally.
            _stash_is_binary = (
                len(self._binary_stash) >= 2
                and self._binary_stash[0] == MAGIC_BYTE_0
                and self._binary_stash[1] == MAGIC_BYTE_1
            )
            if not _stash_is_binary:
                if not self._read_header():
                    self._running = False
                    return
        else:
            # Handshake failed — check if MCU is already streaming binary frames
            if self._detect_binary_stream():
                self._emit_status(
                    "MCU already streaming — skipping handshake & header."
                )
                self._emit_connected(self._port)
            else:
                self._emit_error(
                    "Handshake failed and no binary stream detected. "
                    "Is the device connected and running?"
                )
                self._running = False
                return

        # --- Parse binary frames ---
        self._emit_status("Receiving binary frames...")
        self._parse_frames()

        # --- Cleanup ---
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def _do_handshake(self) -> bool:
        """Perform the eFlesh handshake on an already-open serial port.

        Sends ACK repeatedly until RDY is received, or a binary stream is
        detected (reconnect fast-path).  Any leftover bytes consumed during
        the handshake are stashed in ``_binary_stash`` so downstream frame
        parsing doesn't lose data.

        Returns True on success.
        """
        buf = bytearray()
        deadline = time.time() + 5.0  # generous timeout
        last_ack_send = 0.0

        self._ser.reset_input_buffer()

        while time.time() < deadline and self._running:
            # Send ACK periodically in case MCU hasn't seen it yet
            now = time.time()
            if now - last_ack_send >= 0.12:
                self._ser.write((HANDSHAKE_ACK + "\n").encode("ascii"))
                self._ser.flush()
                last_ack_send = now

            # Wait a bit for response
            time.sleep(0.02)

            if self._ser.in_waiting:
                chunk = self._ser.read(self._ser.in_waiting)
                buf.extend(chunk)

                # --- Binary-stream fast-path (MCU already streaming) ---
                magic_idx = _find_magic(buf)
                if magic_idx >= 0:
                    # Stash everything from the magic bytes onward so
                    # _parse_frames can consume it immediately.
                    self._binary_stash = bytes(buf[magic_idx:])
                    self._emit_status("Handshake OK — binary stream detected (reconnect).")
                    return True

                if _find_subsequence(buf, HANDSHAKE_RDY_B) >= 0:
                    # Stash any leftover bytes (e.g. the start of the
                    # next binary frame) so they aren't lost.
                    if len(buf) > 0:
                        self._binary_stash = bytes(buf)
                    self._emit_status("Handshake OK — device ready.")
                    return True
                if _find_subsequence(buf, HANDSHAKE_HSK_B) >= 0:
                    self._emit_status("  (saw HSK broadcast)")

                # Keep buffer bounded
                if len(buf) > 8192:
                    buf = buf[-4096:]

        self._emit_error("Handshake failed — no RDY response from device.")
        return False

    def _read_header(self) -> bool:
        """Read the text header from the device until the binary stream sentinel.

        Also handles the reconnect case: if the MCU is already streaming binary
        frames (e.g. after an in-loop handshake response), the first bytes will
        be magic bytes 0xAA 0x55 rather than ASCII text.  In that case we stash
        the data and return immediately.

        Returns True if the sentinel was found (or binary data detected).
        """
        buf = bytearray()
        deadline = time.time() + 5.0

        while time.time() < deadline and self._running:
            if self._ser.in_waiting:
                chunk = self._ser.read(self._ser.in_waiting)
                buf.extend(chunk)

                # --- Reconnect fast-path: binary data already flowing ---
                magic_idx = _find_magic(buf)
                if magic_idx >= 0:
                    self._binary_stash = bytes(buf[magic_idx:])
                    self._emit_status("  (detected binary stream — reconnect)")
                    return True

                while b'\n' in buf:
                    idx = buf.find(b'\n')
                    line_bytes = buf[:idx]
                    buf = buf[idx + 1:]

                    # Guard: if we somehow see magic bytes mid-stream, treat as reconnect
                    line_magic_idx = _find_magic(line_bytes)
                    if line_magic_idx >= 0:
                        self._binary_stash = bytes(line_bytes[line_magic_idx:]) + bytes(buf)
                        self._emit_status("  (detected binary stream — reconnect)")
                        return True

                    try:
                        line = line_bytes.decode("ascii", errors="replace").strip()
                    except Exception:
                        line = line_bytes.decode("utf-8", errors="replace").strip()

                    if line:
                        self._emit_header_line(line)
                        self._emit_status(f"  {line}")

                    if HEADER_SENTINEL in line:
                        return True

                if len(buf) > 4096:
                    buf = buf[-2048:]

            time.sleep(0.02)

        self._emit_error("Header read timed out — never saw binary stream sentinel.")
        return False

    def _detect_binary_stream(self) -> bool:
        """Check whether the MCU is already streaming binary frames.

        Used as a fallback when the handshake fails (e.g. on reconnect).
        Looks for the magic bytes 0xAA 0x55 in incoming data.
        """
        buf = bytearray()
        deadline = time.time() + 2.0

        while time.time() < deadline and self._running:
            if self._ser.in_waiting:
                chunk = self._ser.read(self._ser.in_waiting)
                buf.extend(chunk)

                # Look for two consecutive magic bytes anywhere in the buffer
                for i in range(len(buf) - 1):
                    if buf[i] == MAGIC_BYTE_0 and buf[i + 1] == MAGIC_BYTE_1:
                        # Found a binary frame — push data back into the
                        # serial buffer isn't possible, but _parse_frames
                        # will pick up from here since we leave data in
                        # the OS serial buffer + whatever we already read.
                        # Actually, we've already consumed this data.  We'll
                        # inject it into _parse_frames by stashing it.
                        self._binary_stash = buf[i:]  # consumed by _parse_frames
                        return True
            else:
                time.sleep(0.05)

        return False

    def _parse_frames(self):
        """Read and parse binary frames in a loop."""
        buf = bytearray()

        # Consume any pre-read binary data from the reconnect fallback
        if self._binary_stash:
            buf.extend(self._binary_stash)
            self.bytes_received += len(self._binary_stash)
            self._binary_stash = b""

        last_diag_time = time.time()

        while self._running:
            if self._ser.in_waiting:
                chunk = self._ser.read(self._ser.in_waiting)
                buf.extend(chunk)
                self.bytes_received += len(chunk)
            else:
                time.sleep(0.001)
                continue

            # Periodic diagnostics — report if no frames are being decoded
            now = time.time()
            if now - last_diag_time >= 2.0:
                if self.frame_count == 0 and self.bytes_received > 0:
                    self._emit_status(
                        f"Receiving binary frames... "
                        f"({self.bytes_received} B rcvd, "
                        f"{self.sync_losses} sync losses, "
                        f"{self.parse_errors} parse errs, "
                        f"0 frames decoded)"
                    )
                last_diag_time = now

            while len(buf) >= FRAME_SIZE and self._running:
                # Find sync
                sync_idx = -1
                for i in range(len(buf) - 1):
                    if buf[i] == MAGIC_BYTE_0 and buf[i + 1] == MAGIC_BYTE_1:
                        sync_idx = i
                        break

                if sync_idx < 0:
                    self.sync_losses += 1
                    buf = buf[-1:]
                    break
                if sync_idx > 0:
                    buf = buf[sync_idx:]
                if len(buf) < FRAME_SIZE:
                    break

                try:
                    frame = self._parse_one_frame(buf[:FRAME_SIZE])
                    buf = buf[FRAME_SIZE:]
                    if frame is not None:
                        frame.timestamp = time.time()
                        self._on_frame_internal(frame)
                except Exception:
                    self.parse_errors += 1
                    buf = buf[1:]   # skip bad byte, resync
                    break

    # ------------------------------------------------------------------
    # Frame parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_one_frame(raw: bytes) -> FrameData | None:
        """Parse a single 324-byte binary frame."""
        if len(raw) < FRAME_SIZE:
            return None
        if raw[0] != MAGIC_BYTE_0 or raw[1] != MAGIC_BYTE_1:
            return None

        seq = struct.unpack_from('<H', raw, 2)[0]
        frame = FrameData(seq=seq)

        offset = 4
        for g in range(NUM_GROUPS):
            for s in range(NUM_SENSORS):
                t, x, y, z = struct.unpack_from('<ffff', raw, offset)
                frame.groups[g][s] = SensorReading(t=t, x=x, y=y, z=z)
                offset += 16

        return frame

    def _on_frame_internal(self, frame: FrameData):
        """Track stats, then fire callback."""
        self.frame_count += 1

        # Drop detection
        if self.last_seq >= 0:
            expected = (self.last_seq + 1) & 0xFFFF
            if frame.seq != expected:
                drops = (frame.seq - expected) & 0xFFFF
                self.dropped += drops
        self.last_seq = frame.seq

        if self.on_frame:
            self.on_frame(frame)

    # ------------------------------------------------------------------
    # Callback helpers
    # ------------------------------------------------------------------

    def _emit_status(self, msg: str):
        if self.on_status:
            self.on_status(msg)

    def _emit_error(self, msg: str):
        if self.on_error:
            self.on_error(msg)

    def _emit_connected(self, port: str):
        if self.on_connected:
            self.on_connected(port)

    def _emit_disconnected(self):
        if self.on_disconnected:
            self.on_disconnected()

    def _emit_header_line(self, line: str):
        if self.on_header_line:
            self.on_header_line(line)
