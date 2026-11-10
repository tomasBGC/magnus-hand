"""LoadScale force/mass sensor reader with serial handshake."""

import logging
import re
import serial
from serial.tools import list_ports
import time
from queue import Full
from typing import Callable, Iterable, Optional, Set

logger = logging.getLogger(__name__)

_CAL_FACTOR_RE = re.compile(
    r"Calibration factor(?:\s+loaded from EEPROM)?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_VALUE_SAVED_RE = re.compile(r"Value\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+saved")


class LoadScaleReader:
    """Interface to the LoadScale sensor via serial connection with handshake."""

    DEVICE_CODE = "LDSCL-001"
    BAUD_RATE = 57600

    def __init__(
        self,
        serial_port: Optional[str] = None,
        baud_rate: int = BAUD_RATE,
        exclude_ports: Optional[Iterable[str]] = None,
    ):
        """
        Initialize the LoadScale reader with serial connection.
        
        Args:
            serial_port: Preferred serial port (will auto-discover if None)
            baud_rate: Serial communication rate
        """
        self.baud_rate = baud_rate
        self._excluded_ports: Set[str] = {port for port in (exclude_ports or []) if port}
        self.cal_factor: Optional[float] = None
        self._boot_lines: list[str] = []
        self.on_line: Optional[Callable[[str], None]] = None
        self._calibrating = False   # blocks read_force() during cal commands
        self.serial_connection = self._connect_loadscale(serial_port, baud_rate)
        self._last_force_value: Optional[float] = None

    def _try_handshake(
        self, port: str, baud_rate: int, handshake_timeout: float = 5.0
    ) -> Optional[serial.Serial]:
        """
        Establish a serial connection and perform handshake with the LoadScale.
        
        Returns:
            Opened serial connection if successful, None otherwise.
        """
        try:
            connection = serial.Serial(port, baud_rate, timeout=0.25, write_timeout=1)
        except serial.SerialException:
            return None

        # MCU boards reset when the serial port opens.
        time.sleep(2.0)
        connection.reset_input_buffer()

        deadline = time.time() + handshake_timeout
        while time.time() < deadline:
            line = connection.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            self._boot_lines.append(line)
            cal_match = _CAL_FACTOR_RE.search(line)
            if cal_match:
                try:
                    self.cal_factor = float(cal_match.group(1))
                except ValueError:
                    pass
            if line == self.DEVICE_CODE:
                connection.write(b"1\n")
                return connection

        connection.close()
        return None

    def _connect_loadscale(
        self, preferred_port: Optional[str], baud_rate: int
    ) -> serial.Serial:
        """
        Attempt to connect to the LoadScale sensor.
        
        Tries the preferred port first, then scans all available ports.
        
        Raises:
            RuntimeError: If no LoadScale is found on any port.
        """
        candidate_ports = []

        if preferred_port:
            candidate_ports.append(preferred_port)

        for port_info in list_ports.comports():
            if port_info.device not in candidate_ports:
                candidate_ports.append(port_info.device)

        candidate_ports = [port for port in candidate_ports if port not in self._excluded_ports]

        for port in candidate_ports:
            connection = self._try_handshake(port, baud_rate)
            if connection is not None:
                logger.info("Connected to LoadScale on %s", port)
                return connection

        raise RuntimeError(
            f"Could not find LoadScale sensor. Expected handshake '{self.DEVICE_CODE}' on ports: {candidate_ports}"
        )

    def read_force(self) -> Optional[float]:
        """
        Read the latest force/mass value from the sensor.
        
        Returns:
            Force/mass value in grams, or None if no valid data received.
        """
        if self._calibrating:
            return self._last_force_value

        try:
            # Drain all currently queued lines and keep the newest numeric value.
            # This avoids returning stale values when callers poll slower than the
            # MCU serial stream (e.g. frame-based capture loops).
            latest_value: Optional[float] = None
            while self.serial_connection.in_waiting > 0:
                line = self.serial_connection.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    latest_value = float(line)
                except ValueError:
                    # Parse cal factor updates from firmware messages (e.g. post-save confirmation)
                    cal_match = _CAL_FACTOR_RE.search(line)
                    if cal_match:
                        try:
                            self.cal_factor = float(cal_match.group(1))
                        except ValueError:
                            pass
                    saved_match = _VALUE_SAVED_RE.search(line)
                    if saved_match:
                        try:
                            self.cal_factor = float(saved_match.group(1))
                        except ValueError:
                            pass
                    if self.on_line:
                        self.on_line(line)
                    continue

            if latest_value is not None:
                self._last_force_value = latest_value
                return latest_value
        except (serial.SerialException, OSError):
            pass

        return self._last_force_value

    def tare(self):
        """Send tare command to reset the scale to zero."""
        try:
            self.serial_connection.write(b"t\n")
            self.serial_connection.flush()
        except (serial.SerialException, OSError):
            pass

    def set_calibration_factor(self, calibration_factor: float, timeout_s: float = 30.0) -> bool:
        """
        Send calibration factor to the LoadScale and save it to EEPROM.
        
        This triggers the MCU's calibration factor update procedure:
        1. Send 'c' to enter calibration mode
        2. Send the calibration factor value
        3. Confirm with 'y' to save to EEPROM
        
        Args:
            calibration_factor: The new calibration factor (raw_reading / known_mass)
            timeout_s: Timeout for the EEPROM save operation
            
        Returns:
            True if calibration was saved, False if operation failed or timed out
        """
        self._calibrating = True
        try:
            # Enter calibration mode
            self.serial_connection.write(b"c\n")
            self.serial_connection.flush()
            time.sleep(0.5)

            # Clear input buffer to remove prompts
            self.serial_connection.reset_input_buffer()
            time.sleep(0.2)

            # Send calibration factor
            cal_str = f"{calibration_factor:.6f}\n"
            self.serial_connection.write(cal_str.encode())
            self.serial_connection.flush()
            time.sleep(0.5)

            # Confirm save to EEPROM
            self.serial_connection.write(b"y\n")
            self.serial_connection.flush()

            # Wait for confirmation — we own the serial port here (_calibrating=True
            # prevents read_force() from consuming this data concurrently).
            start_time = time.time()
            while time.time() - start_time < timeout_s:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode("utf-8", errors="ignore").strip()
                    if self.on_line and line:
                        self.on_line(line)
                    if "saved" in line.lower() or "eeprom" in line.lower():
                        self.cal_factor = calibration_factor
                        return True
                time.sleep(0.1)

            logger.warning("Calibration command timed out waiting for EEPROM confirmation.")
            return False

        except (serial.SerialException, OSError) as e:
            logger.error("Failed to set calibration factor: %s", e)
            return False
        finally:
            self._calibrating = False

    def close(self):
        """Close the serial connection."""
        if self.serial_connection is not None:
            try:
                self.serial_connection.close()
            except (serial.SerialException, OSError):
                pass


def loadscalegrab(
    stop_event,
    q,
    serial_port: Optional[str] = None,
    exclude_ports: Optional[Iterable[str]] = None,
    control_q=None,
):
    """
    Worker process to continuously read force data from LoadScale and queue it.

    Force data is queued for the main process to consume (e.g. for logging,
    recording, or real-time display).

    Args:
        stop_event: Multiprocessing event to signal shutdown
        q: Queue to put force readings into
        serial_port: Preferred serial port for LoadScale connection
        exclude_ports: Serial ports to skip during auto-discovery
        control_q: Queue for runtime control messages (reserved for future use)
    """
    reader = LoadScaleReader(serial_port=serial_port, exclude_ports=exclude_ports)

    try:
        while not stop_event.is_set():
            # Read force value
            force_value = reader.read_force()
            
            if force_value is not None:
                timestamp = time.time()
                data = {
                    "timestamp": timestamp,
                    "force": force_value,
                }

                # Queue the data for main process
                try:
                    q.put(data, block=False)
                except Full:
                    pass

            time.sleep(0.01)  # Poll every 10ms

    finally:
        reader.close()
