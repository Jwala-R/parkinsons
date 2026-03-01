"""
arduino/comm.py — Live Arduino communication over USB Serial.

Serial protocol (Arduino → Python, 60 Hz, one line per sample):
    ACC_X,ACC_Y,ACC_Z,GYR_X,GYR_Y,GYR_Z,PRESSURE\n
    Example: 0.1234,-0.5678,9.8012,0.0023,-0.0041,0.0011,102.3

Haptic command (Python → Arduino):
    H<strength>\n   e.g.  H200\n  (PWM 0-255, then Arduino pulses for HAPTIC_DURATION_MS)
    H0\n            stop

Arduino wiring reference:
    SDA/SCL (A4/A5) → MPU-6050 I2C
    A0               → FSR pressure sensor (voltage divider to 5V)
    D9 (PWM)         → ERM/LRA motor driver IN
    USB              → Serial to Python

Arduino sketch snippet (for reference):
    void loop() {
        // read MPU-6050 via Wire, read analogRead(A0) for pressure
        Serial.print(ax,4); Serial.print(","); ...  Serial.println(pressure);
        if (Serial.available()) {
            String cmd = Serial.readStringUntil('\n');
            if (cmd.startsWith("H")) {
                int pwm = cmd.substring(1).toInt();
                analogWrite(9, pwm);
                delay(HAPTIC_DURATION_MS);
                analogWrite(9, 0);
            }
        }
        delay(1000/60);
    }
"""

import serial
import time


class ArduinoComm:
    """
    Manages a bidirectional USB-Serial connection to an Arduino.

    All public methods are safe to call from a background thread EXCEPT
    send_haptic(), which should only be called from the main thread to
    avoid concurrent write collisions.
    """

    def __init__(self, port: str, baud: int, timeout: float = 0.01):
        self._port = port
        self._baud = baud
        self._timeout = timeout
        self._ser: serial.Serial | None = None
        self.parse_errors = 0

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open the serial port. Returns True on success."""
        try:
            self._ser = serial.Serial(self._port, self._baud,
                                      timeout=self._timeout)
            time.sleep(2.0)  # allow Arduino to reset after DTR toggle
            self._ser.reset_input_buffer()
            return True
        except serial.SerialException as e:
            print(f"[ArduinoComm] Failed to open {self._port}: {e}")
            self._ser = None
            return False

    def disconnect(self):
        """Close the serial port gracefully."""
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ── Reading ────────────────────────────────────────────────────────────

    def read_frame(self) -> dict | None:
        """
        Non-blocking read of one sensor frame.

        Returns a dict:
            {"acc": [ax, ay, az], "gyro": [gx, gy, gz], "pressure": float}
        or None if no data is available or the line is malformed.
        """
        if not self.is_connected:
            return None
        try:
            raw = self._ser.readline()
            if not raw:
                return None
            parts = raw.decode("ascii", errors="ignore").strip().split(",")
            if len(parts) != 7:
                self.parse_errors += 1
                return None
            vals = [float(p) for p in parts]
            return {
                "acc":      [vals[0], vals[1], vals[2]],
                "gyro":     [vals[3], vals[4], vals[5]],
                "pressure": vals[6],
            }
        except (ValueError, UnicodeDecodeError, serial.SerialException):
            self.parse_errors += 1
            return None

    # ── Writing ────────────────────────────────────────────────────────────

    def send_haptic(self, strength: int, duration_ms: int = 500):
        """
        Send a haptic command to the Arduino.

        strength: PWM value 0-255 (0 = off)
        duration_ms: ignored here (Arduino handles timing internally)
        """
        if not self.is_connected:
            return
        try:
            cmd = f"H{int(strength)}\n".encode("ascii")
            self._ser.write(cmd)
        except serial.SerialException as e:
            print(f"[ArduinoComm] Write error: {e}")
