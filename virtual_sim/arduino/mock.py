"""
arduino/mock.py — Synthetic sensor data for demo mode (no Arduino required).

Generates physiologically plausible IMU data:
  - Normal walking: sinusoidal acc_y at 1.8 Hz (arm swing), gravity on acc_z
  - FoG episode:    high-frequency tremor at 5 Hz on all axes (freeze band)

FoG episodes are injected automatically every MOCK_FOG_INTERVAL seconds
for MOCK_FOG_DURATION seconds. The 5 Hz signal pushes spectral power into
the freeze band (3-8 Hz) which the ML Freeze Index will detect.

The MockArduino has the same public interface as ArduinoComm so the rest of
the system works identically in demo mode.
"""

import time
import math
import random
from virtual_sim.config import (
    MOCK_FOG_INTERVAL, MOCK_FOG_DURATION, SAMPLING_RATE
)


class MockArduino:
    """
    Drop-in replacement for ArduinoComm that generates synthetic data.

    fast_forward: if True, every call to read_frame() advances a virtual
    clock by 1/SAMPLING_RATE seconds and always returns a frame (no rate
    limiting). Use this for headless tests that need to run faster than
    real-time. Default False (real-time 60 Hz rate limiting).
    """

    def __init__(self, fast_forward: bool = False):
        self._start_time = None
        self._last_frame_time = None
        self._frame_dt = 1.0 / SAMPLING_RATE
        self._fast_forward = fast_forward
        self._virtual_t = 0.0          # virtual clock for fast_forward mode
        self.parse_errors = 0

    # ── Connection (always succeeds) ───────────────────────────────────────

    def connect(self) -> bool:
        self._start_time = time.monotonic()
        self._last_frame_time = self._start_time
        self._virtual_t = 0.0
        return True

    def disconnect(self):
        pass

    @property
    def is_connected(self) -> bool:
        return self._start_time is not None

    # ── Reading ────────────────────────────────────────────────────────────

    def read_frame(self) -> dict | None:
        """
        Returns one synthetic frame.

        In normal mode: rate-limited to SAMPLING_RATE Hz by wall clock.
        In fast_forward mode: always returns a frame, advances virtual clock.
        """
        if self._fast_forward:
            t = self._virtual_t
            self._virtual_t += self._frame_dt
        else:
            now = time.monotonic()
            if now - self._last_frame_time < self._frame_dt:
                return None
            self._last_frame_time = now
            t = now - self._start_time

        is_fog = self._is_fog_episode(t)
        if is_fog:
            return self._fog_frame(t)
        return self._walking_frame(t)

    # ── Writing (no-op with console log) ──────────────────────────────────

    def send_haptic(self, strength: int, duration_ms: int = 500):
        if strength > 0:
            print(f"[MockArduino] Haptic: PWM={strength}, dur={duration_ms}ms")

    # ── Private helpers ────────────────────────────────────────────────────

    def _is_fog_episode(self, t: float) -> bool:
        """True during the FoG window in each MOCK_FOG_INTERVAL cycle."""
        phase = t % MOCK_FOG_INTERVAL
        return phase >= (MOCK_FOG_INTERVAL - MOCK_FOG_DURATION)

    def _walking_frame(self, t: float) -> dict:
        """
        Normal arm swing during walking at 1.8 Hz.
        acc_y oscillates (vertical); acc_z holds gravity (~9.8 m/s^2 or 1g).
        """
        freq = 1.8
        acc_y = 1.5 * math.sin(2 * math.pi * freq * t) + _noise(0.1)
        acc_x = 0.3 * math.sin(2 * math.pi * freq * t + 0.5) + _noise(0.05)
        acc_z = 9.8 + _noise(0.05)
        gyro_x = 0.5 * math.sin(2 * math.pi * freq * t) + _noise(0.02)
        gyro_y = _noise(0.02)
        gyro_z = 0.2 * math.sin(2 * math.pi * freq * t + 1.0) + _noise(0.02)
        return {
            "acc":      [acc_x, acc_y, acc_z],
            "gyro":     [gyro_x, gyro_y, gyro_z],
            "pressure": 90.0 + _noise(5.0),
        }

    def _fog_frame(self, t: float) -> dict:
        """
        FoG: high-frequency tremor at 5 Hz (freeze band 3-8 Hz).
        No forward progression — amplitude is lower and chaotic.
        """
        freq = 5.0
        amp  = 0.4
        acc_x = amp * math.sin(2 * math.pi * freq * t) + _noise(0.15)
        acc_y = amp * math.sin(2 * math.pi * freq * t + 0.3) + _noise(0.15)
        acc_z = 9.8 + amp * math.sin(2 * math.pi * freq * t + 0.6) + _noise(0.1)
        gyro_x = 0.8 * math.sin(2 * math.pi * freq * t) + _noise(0.1)
        gyro_y = 0.8 * math.sin(2 * math.pi * freq * t + 1.0) + _noise(0.1)
        gyro_z = 0.5 * math.sin(2 * math.pi * freq * t + 2.0) + _noise(0.1)
        return {
            "acc":      [acc_x, acc_y, acc_z],
            "gyro":     [gyro_x, gyro_y, gyro_z],
            "pressure": 90.0 + _noise(5.0),
        }


def _noise(scale: float) -> float:
    return random.gauss(0.0, scale)
