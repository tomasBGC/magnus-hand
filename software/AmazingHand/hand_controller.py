# This file is derived from Pollen Robotics' AmazingHand
# (https://github.com/pollen-robotics/AmazingHand), which is licensed under the
# Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
# It has been modified: the servo command logic and register formulas of the
# upstream PythonExample/AmazingHand_FingerTest.py and the rustypot
# Scs0009PyController were re-implemented here to send commands over the eFlesh
# sensor board's existing USB serial link instead of a direct servo-bus
# connection. See NOTICE.md in the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. Unless required by
# applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied.

"""
AmazingHand Finger Control
===========================
Feetech SCS0009 finger control, relayed through the same USB serial
connection the eFlesh magnetic sensor array already uses (EfleshController /
MagneticSensors/MCU/src/main.cpp) -- not a second, competing connection to
the same COM port, and not a direct rustypot/Waveshare COM port connection.

Command logic (write_torque_enable / write_goal_speed / write_goal_position,
np.deg2rad conversion, CLOSE/OPEN angle offsets) mirrors the official
pollen-robotics/AmazingHand PythonExample/AmazingHand_FingerTest.py script.
Register addresses and the raw-value <-> angle formulas come from that
project's rustypot Scs0009PyController (src/servo/feetech/scs0009.rs), not
invented here:
    torque_enable : addr 40, 1 byte
    goal_position : addr 42, i16, raw = int(1024 * angle_rad / rad(300) + 511)  (truncated toward zero)
    goal_speed    : addr 46, u16, raw = int(value / (rad(300) / 1024))
Both 2-byte registers go out big-endian on the wire (SCS0009-specific; the
double byte-swap in rustypot's conversion + write path nets out to MSB-first,
unlike the little-endian STS3215).

Usage:
    from MagneticSensors.eflesh_controller import EfleshController
    from AmazingHand.hand_controller import connect, enable_index, close_index, open_index

    mag = EfleshController()
    mag.on_connected = lambda port: print(f"Connected on {port}")
    mag.start()                     # begins streaming sensor frames
    # ... wait until mag.connected_port is set ...
    connect(mag)                    # shares mag's serial connection
    enable_index()
    close_index()

    # Or via UnifiedController:
    #   ctrl = UnifiedController(data_dir=...)
    #   ctrl.connect_all()
    #   connect(ctrl.magnetic_controller)

Safety: each finger is driven only by an explicit function call. There is no
loop or "all fingers" helper here on purpose -- test one finger at a time on
the physically mounted hand before wiring up the rest.
"""

import struct
import threading
import time

import numpy as np
import serial

REG_TORQUE_ENABLE = 40
REG_GOAL_POSITION = 42
REG_GOAL_SPEED = 46

FRAME_SYNC = bytes([0xFA, 0xCE])

ANGLE_RANGE_RAD = np.deg2rad(300.0)
ANGLE_STEPS = 1024
POSITION_CENTER = 511

CLOSE_ANGLE_OFFSET = 90
OPEN_ANGLE_OFFSET = -30

# Servo speed unit is rad/s in raw units (see _speed_raw); 3 is a
# conservative default so a first test on a newly mounted finger doesn't
# slam the servos to end-of-travel at full speed.
DEFAULT_SPEED = 3


def _position_raw(angle_rad):
    raw = int(ANGLE_STEPS * angle_rad / ANGLE_RANGE_RAD + POSITION_CENTER)
    return raw & 0xFFFF


def _speed_raw(value):
    step = ANGLE_RANGE_RAD / ANGLE_STEPS
    raw = int(value / step)
    return raw & 0xFFFF


class TeensyServoLink:
    """Sends register-write frames over a serial connection that some other
    controller already owns and is reading from (e.g. EfleshController).

    This link is write-only: it never reads from the connection, so it
    cannot race with that controller's background reader thread. Writes are
    still serialized with a lock in case multiple threads call into this
    link (e.g. a GUI thread and a script)."""

    def __init__(self, serial_connection: serial.Serial):
        if serial_connection is None or not serial_connection.is_open:
            raise RuntimeError(
                "Serial connection is not open -- connect the sensor controller first, then share its connection here."
            )
        self.ser = serial_connection
        self._write_lock = threading.Lock()

    def _send(self, servo_id, addr, data_bytes):
        length = len(data_bytes)
        checksum = (servo_id + addr + length + sum(data_bytes)) & 0xFF
        frame = FRAME_SYNC + bytes([servo_id, addr, length]) + data_bytes + bytes([checksum])
        with self._write_lock:
            self.ser.write(frame)

    def write_torque_enable(self, servo_id, value):
        self._send(servo_id, REG_TORQUE_ENABLE, bytes([value & 0xFF]))

    def write_goal_speed(self, servo_id, value):
        self._send(servo_id, REG_GOAL_SPEED, struct.pack(">H", _speed_raw(value)))

    def write_goal_position(self, servo_id, angle_rad):
        self._send(servo_id, REG_GOAL_POSITION, struct.pack(">H", _position_raw(angle_rad)))


class Finger:
    """One AmazingHand finger = two SCS0009 servos with individual
    MiddlePos calibration offsets, as in AmazingHand_FingerTest.py."""

    def __init__(self, link, id_1, id_2, middle_pos_1, middle_pos_2, reverse=False):
        self.link = link
        self.id_1, self.id_2 = id_1, id_2
        self.middle_pos_1, self.middle_pos_2 = middle_pos_1, middle_pos_2
        self._dir = -1 if reverse else 1

    def enable_torque(self):
        self.link.write_torque_enable(self.id_1, 1)
        self.link.write_torque_enable(self.id_2, 1)

    def close(self, percent=100, speed=DEFAULT_SPEED, sleep_after=True):
        percent = max(0, min(100, percent))
        offset = CLOSE_ANGLE_OFFSET * (percent / 100.0)
        self.link.write_goal_speed(self.id_1, speed)
        self.link.write_goal_speed(self.id_2, speed)
        pos_1 = np.deg2rad(self.middle_pos_1 + self._dir * offset)
        pos_2 = np.deg2rad(self.middle_pos_2 - self._dir * offset)
        self.link.write_goal_position(self.id_1, pos_1)
        self.link.write_goal_position(self.id_2, pos_2)
        if sleep_after:
            time.sleep(0.01)

    def open(self, percent=100, speed=DEFAULT_SPEED, sleep_after=True):
        percent = max(0, min(100, percent))
        offset = OPEN_ANGLE_OFFSET * (percent / 100.0)
        self.link.write_goal_speed(self.id_1, speed)
        self.link.write_goal_speed(self.id_2, speed)
        pos_1 = np.deg2rad(self.middle_pos_1 + self._dir * offset)
        pos_2 = np.deg2rad(self.middle_pos_2 - self._dir * offset)
        self.link.write_goal_position(self.id_1, pos_1)
        self.link.write_goal_position(self.id_2, pos_2)
        if sleep_after:
            time.sleep(0.01)


MIDDLE_POS = [-78, 77, -3, 4, 28, -15, 15, -15]

_link: TeensyServoLink | None = None

index = Finger(None, 1, 2, MIDDLE_POS[0], MIDDLE_POS[1])
middle = Finger(None, 3, 4, MIDDLE_POS[2], MIDDLE_POS[3])
ring = Finger(None, 5, 6, MIDDLE_POS[4], MIDDLE_POS[5])
thumb = Finger(None, 7, 8, MIDDLE_POS[6], MIDDLE_POS[7])


def connect(mag_controller_or_serial):
    """Share an already-open serial connection instead of opening a new one.

    Accepts anything exposing a `.serial_connection` property (an
    EfleshController, or UnifiedController.magnetic_controller), or a raw
    serial.Serial directly. The sensor controller must already be connected
    (its serial_connection must be open) before calling this."""
    ser = getattr(mag_controller_or_serial, "serial_connection", mag_controller_or_serial)
    global _link
    _link = TeensyServoLink(ser)
    index.link = middle.link = ring.link = thumb.link = _link
    return _link


def enable_index():
    index.enable_torque()


def close_index(percent=100):
    index.close(percent)


def open_index(percent=100):
    index.open(percent)


def enable_middle():
    middle.enable_torque()


def close_middle(percent=100):
    middle.close(percent)


def open_middle(percent=100):
    middle.open(percent)


def enable_ring():
    ring.enable_torque()


def close_ring(percent=100):
    ring.close(percent)


def open_ring(percent=100):
    ring.open(percent)


def enable_thumb():
    thumb.enable_torque()


def close_thumb(percent=100):
    thumb.close(percent)


def open_thumb(percent=100):
    thumb.open(percent)
