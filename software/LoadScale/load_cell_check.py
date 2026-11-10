"""Simple LoadScale troubleshooting utility with live diagnostics.

Features:
- Auto-connect using the existing LoadScale handshake ("LDSCL-001")
- Live weight stream display from the MCU serial output
- Tare button
- Set calibration factor button (uses existing 'c' flow)
- Compute + apply calibration from known mass
- Displays inferred calibration factor and estimated raw counts

Note:
- The current MCU firmware does not expose HX711 tare offset over serial,
  so this tool reports tare offset as unavailable.
"""

from __future__ import annotations

import argparse
import queue
import re
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Optional

import serial
from serial.tools import list_ports


DEVICE_CODE = "LDSCL-001"
BAUD_RATE = 57600

_CAL_FACTOR_RE = re.compile(r"Calibration factor loaded from EEPROM:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_VALUE_SAVED_RE = re.compile(r"Value\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+saved")


@dataclass
class LiveState:
    port: str
    connected: bool = False
    latest_value_g: Optional[float] = None
    min_value_g: Optional[float] = None
    max_value_g: Optional[float] = None
    sample_count: int = 0
    cal_factor: Optional[float] = None
    tare_offset: Optional[float] = None
    last_text: str = ""

    def push_value(self, value: float) -> None:
        self.latest_value_g = value
        self.sample_count += 1
        if self.min_value_g is None or value < self.min_value_g:
            self.min_value_g = value
        if self.max_value_g is None or value > self.max_value_g:
            self.max_value_g = value


def _try_connect_port(port: str, timeout_s: float = 6.0) -> tuple[Optional[serial.Serial], Optional[float], list[str]]:
    """Try one serial port and perform LoadScale handshake.

    Returns:
        (connection, inferred_cal_factor, startup_lines)
    """
    startup_lines: list[str] = []
    try:
        conn = serial.Serial(port, BAUD_RATE, timeout=0.25, write_timeout=1)
    except serial.SerialException:
        return None, None, startup_lines

    time.sleep(2.0)  # Allow MCU reset and boot messages.
    conn.reset_input_buffer()

    inferred_cal: Optional[float] = None
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        raw = conn.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        startup_lines.append(line)
        cal_match = _CAL_FACTOR_RE.search(line)
        if cal_match:
            try:
                inferred_cal = float(cal_match.group(1))
            except ValueError:
                pass

        if line == DEVICE_CODE:
            conn.write(b"1\n")
            conn.flush()
            return conn, inferred_cal, startup_lines

    conn.close()
    return None, None, startup_lines


def connect_loadscale(preferred_port: Optional[str] = None) -> tuple[serial.Serial, str, Optional[float], list[str]]:
    candidate_ports: list[str] = []
    if preferred_port:
        candidate_ports.append(preferred_port)

    for p in list_ports.comports():
        if p.device not in candidate_ports:
            candidate_ports.append(p.device)

    for port in candidate_ports:
        conn, inferred_cal, startup_lines = _try_connect_port(port)
        if conn is not None:
            return conn, port, inferred_cal, startup_lines

    raise RuntimeError(f"Could not find LoadScale sensor. Checked ports: {candidate_ports}")


class LoadCellCheckApp:
    def __init__(self, root: tk.Tk, serial_port: Optional[str] = None):
        self.root = root
        self.root.title("Load Cell Check")
        self.root.geometry("760x540")

        self._serial_port_pref = serial_port
        self._conn: Optional[serial.Serial] = None
        self._stop_event = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._ui_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._state = LiveState(port=serial_port or "(auto)")

        self._build_ui()
        self._connect()
        self._schedule_ui_poll()

    def _build_ui(self) -> None:
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(frm)
        top.pack(fill=tk.X)

        self.lbl_conn = ttk.Label(top, text="Connection: disconnected")
        self.lbl_conn.pack(side=tk.LEFT)

        ttk.Button(top, text="Reconnect", command=self._reconnect).pack(side=tk.RIGHT)

        metrics = ttk.LabelFrame(frm, text="Live Metrics", padding=10)
        metrics.pack(fill=tk.X, pady=(8, 8))

        self.var_latest = tk.StringVar(value="--")
        self.var_minmax = tk.StringVar(value="--")
        self.var_samples = tk.StringVar(value="0")
        self.var_cal = tk.StringVar(value="unknown")
        self.var_tare_offset = tk.StringVar(value="N/A (not exposed by firmware)")
        self.var_raw_est = tk.StringVar(value="--")

        self._metric_row(metrics, "Latest reading (g):", self.var_latest, 0)
        self._metric_row(metrics, "Min/Max (g):", self.var_minmax, 1)
        self._metric_row(metrics, "Samples:", self.var_samples, 2)
        self._metric_row(metrics, "Calibration factor:", self.var_cal, 3)
        self._metric_row(metrics, "Tare offset (raw):", self.var_tare_offset, 4)
        self._metric_row(metrics, "Estimated raw delta counts:", self.var_raw_est, 5)

        actions = ttk.LabelFrame(frm, text="Actions", padding=10)
        actions.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(actions, text="Tare", command=self._cmd_tare).grid(row=0, column=0, padx=4, pady=4, sticky="w")

        ttk.Label(actions, text="Set cal factor:").grid(row=0, column=1, padx=4, sticky="e")
        self.ent_cal = ttk.Entry(actions, width=12)
        self.ent_cal.grid(row=0, column=2, padx=4, sticky="w")
        ttk.Button(actions, text="Apply", command=self._cmd_set_cal).grid(row=0, column=3, padx=4, sticky="w")

        ttk.Label(actions, text="Known mass (g):").grid(row=1, column=1, padx=4, sticky="e")
        self.ent_known_mass = ttk.Entry(actions, width=12)
        self.ent_known_mass.grid(row=1, column=2, padx=4, sticky="w")
        ttk.Button(actions, text="Compute + Apply", command=self._cmd_compute_apply_cal).grid(
            row=1, column=3, padx=4, sticky="w"
        )

        ttk.Label(
            actions,
            text="Tip: Place known mass first, then click Compute + Apply.",
        ).grid(row=2, column=0, columnspan=4, padx=4, pady=(6, 0), sticky="w")

        logbox = ttk.LabelFrame(frm, text="Serial Log", padding=10)
        logbox.pack(fill=tk.BOTH, expand=True)

        self.txt_log = tk.Text(logbox, height=12, wrap="word")
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _metric_row(parent: ttk.LabelFrame, label: str, value_var: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
        ttk.Label(parent, textvariable=value_var).grid(row=row, column=1, sticky="w", pady=2)

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.txt_log.insert(tk.END, f"[{ts}] {msg}\n")
        self.txt_log.see(tk.END)

    def _connect(self) -> None:
        self._disconnect()
        try:
            conn, port, inferred_cal, startup_lines = connect_loadscale(self._serial_port_pref)
        except Exception as exc:
            self._state = LiveState(port=self._serial_port_pref or "(auto)", connected=False)
            self._update_metrics()
            self._log(f"Connect failed: {exc}")
            return

        self._conn = conn
        self._state = LiveState(port=port, connected=True, cal_factor=inferred_cal)

        for line in startup_lines[-15:]:
            self._log(f"BOOT: {line}")

        self._stop_event.clear()
        self._rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
        self._rx_thread.start()

        self._update_metrics()
        self._log(f"Connected on {port}")

    def _disconnect(self) -> None:
        self._stop_event.set()
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=0.5)
        self._rx_thread = None

        if self._conn is not None:
            try:
                self._conn.close()
            except serial.SerialException:
                pass
            self._conn = None

    def _reconnect(self) -> None:
        self._log("Reconnecting...")
        self._connect()

    def _send_line(self, line: str) -> None:
        if self._conn is None:
            raise RuntimeError("Not connected")
        self._conn.write((line + "\n").encode("utf-8"))
        self._conn.flush()

    def _rx_worker(self) -> None:
        while not self._stop_event.is_set():
            if self._conn is None:
                return
            try:
                raw = self._conn.readline()
            except (serial.SerialException, OSError):
                self._ui_queue.put(("status", "Serial read error; disconnected"))
                return

            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            self._ui_queue.put(("line", line))

    def _handle_serial_line(self, line: str) -> None:
        self._state.last_text = line

        cal_match = _CAL_FACTOR_RE.search(line)
        if cal_match:
            try:
                self._state.cal_factor = float(cal_match.group(1))
            except ValueError:
                pass

        saved_match = _VALUE_SAVED_RE.search(line)
        if saved_match:
            try:
                self._state.cal_factor = float(saved_match.group(1))
            except ValueError:
                pass

        try:
            value = float(line)
            self._state.push_value(value)
        except ValueError:
            pass

        self._log(line)
        self._update_metrics()

    def _update_metrics(self) -> None:
        conn_txt = f"Connection: {'connected' if self._state.connected else 'disconnected'} ({self._state.port})"
        self.lbl_conn.configure(text=conn_txt)

        if self._state.latest_value_g is None:
            self.var_latest.set("--")
            self.var_minmax.set("--")
            self.var_raw_est.set("--")
        else:
            self.var_latest.set(f"{self._state.latest_value_g:.4f}")
            if self._state.min_value_g is not None and self._state.max_value_g is not None:
                self.var_minmax.set(f"{self._state.min_value_g:.4f} / {self._state.max_value_g:.4f}")
            else:
                self.var_minmax.set("--")

            if self._state.cal_factor is not None:
                raw_delta = self._state.latest_value_g * self._state.cal_factor
                self.var_raw_est.set(f"{raw_delta:.2f}")
            else:
                self.var_raw_est.set("unknown (cal factor not known)")

        self.var_samples.set(str(self._state.sample_count))

        if self._state.cal_factor is None:
            self.var_cal.set("unknown")
        else:
            self.var_cal.set(f"{self._state.cal_factor:.6f}")

        if self._state.tare_offset is None:
            self.var_tare_offset.set("N/A (not exposed by firmware)")
        else:
            self.var_tare_offset.set(str(self._state.tare_offset))

    def _schedule_ui_poll(self) -> None:
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "line":
                    self._handle_serial_line(payload)
                elif kind == "status":
                    self._log(payload)
        except queue.Empty:
            pass

        self.root.after(100, self._schedule_ui_poll)

    def _cmd_tare(self) -> None:
        try:
            self._send_line("t")
            self._log("CMD: tare")
        except Exception as exc:
            messagebox.showerror("Tare failed", str(exc))

    def _cmd_set_cal(self) -> None:
        text = self.ent_cal.get().strip()
        if not text:
            messagebox.showwarning("Missing value", "Enter a calibration factor first.")
            return

        try:
            value = float(text)
        except ValueError:
            messagebox.showerror("Invalid value", "Calibration factor must be a number.")
            return

        try:
            self._send_line("c")
            time.sleep(0.3)
            self._send_line(f"{value:.6f}")
            time.sleep(0.3)
            self._send_line("y")
            self._state.cal_factor = value
            self._update_metrics()
            self._log(f"CMD: set cal factor -> {value:.6f}")
        except Exception as exc:
            messagebox.showerror("Set calibration failed", str(exc))

    def _cmd_compute_apply_cal(self) -> None:
        known_mass_text = self.ent_known_mass.get().strip()
        if not known_mass_text:
            messagebox.showwarning("Missing value", "Enter known mass in grams.")
            return

        try:
            known_mass = float(known_mass_text)
        except ValueError:
            messagebox.showerror("Invalid value", "Known mass must be a number.")
            return

        if known_mass <= 0:
            messagebox.showerror("Invalid value", "Known mass must be > 0.")
            return

        if self._state.cal_factor is None:
            messagebox.showerror(
                "Unknown current factor",
                "Current calibration factor is unknown. Reconnect and ensure startup line includes factor, "
                "or set factor manually first.",
            )
            return

        readings: list[float] = []
        start = time.time()
        duration_s = 2.0
        while time.time() - start < duration_s:
            if self._state.latest_value_g is not None:
                readings.append(self._state.latest_value_g)
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.05)

        if not readings:
            messagebox.showerror("No readings", "No live readings available for calibration.")
            return

        avg = sum(readings) / len(readings)
        new_factor = (avg * self._state.cal_factor) / known_mass

        if new_factor == 0:
            messagebox.showerror("Bad calibration", "Computed factor is zero, aborting.")
            return

        self.ent_cal.delete(0, tk.END)
        self.ent_cal.insert(0, f"{new_factor:.6f}")

        try:
            old_factor = self._state.cal_factor
            self._send_line("c")
            time.sleep(0.3)
            self._send_line(f"{new_factor:.6f}")
            time.sleep(0.3)
            self._send_line("y")
            self._state.cal_factor = new_factor
            self._update_metrics()
            self._log(
                f"Computed factor from mass={known_mass:.3f}g, avg={avg:.6f}g, "
                f"old={old_factor:.6f} -> new={new_factor:.6f}"
            )
        except Exception as exc:
            messagebox.showerror("Calibration apply failed", str(exc))

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self) -> None:
        self._disconnect()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple LoadScale troubleshooting utility")
    parser.add_argument("--port", type=str, default=None, help="Preferred serial port (auto-discover if omitted)")
    args = parser.parse_args()

    root = tk.Tk()
    app = LoadCellCheckApp(root, serial_port=args.port)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
