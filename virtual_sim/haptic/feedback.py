"""
haptic/feedback.py — Haptic feedback controller.

Decouples FoG detection events from Arduino output.
Enforces a cooldown period between pulses to avoid overwhelming the patient.

Patterns:
  "single"   — one 500ms pulse (default, for walking task)
  "triple"   — three short 150ms pulses with 200ms gaps (rhythmic cueing,
                evidence-supported for gait re-initiation in PD)
  "rhythmic" — single 100ms pulse (meant to be triggered externally at 1 Hz
                by the caller for metronome-style cueing)
"""

import time
import threading
from virtual_sim.config import HAPTIC_STRENGTH, HAPTIC_DURATION_MS, HAPTIC_COOLDOWN_S


class HapticController:
    """
    Thread-safe haptic output manager.

    All patterns run in a background thread so they do not block
    the Panda3D render loop.
    """

    PATTERNS = ("single", "triple", "rhythmic")

    def __init__(self, comm, cooldown_s: float = HAPTIC_COOLDOWN_S):
        """
        comm: ArduinoComm or MockArduino instance.
        """
        self._comm = comm
        self._cooldown_s = cooldown_s
        self._pattern = "single"
        self._last_fired = 0.0
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def trigger(self, fog_score: float = 1.0):
        """
        Trigger haptic feedback if cooldown has elapsed.
        Runs the pattern in a background thread.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_fired < self._cooldown_s:
                return
            self._last_fired = now

        t = threading.Thread(target=self._fire, daemon=True)
        t.start()

    def set_pattern(self, pattern: str):
        """Set the haptic pattern. One of 'single', 'triple', 'rhythmic'."""
        if pattern not in self.PATTERNS:
            raise ValueError(f"Unknown pattern '{pattern}'. Choose from {self.PATTERNS}")
        self._pattern = pattern

    def stop(self):
        """Immediately silence the haptic motor."""
        self._comm.send_haptic(0, 0)

    # ── Private ────────────────────────────────────────────────────────────

    def _fire(self):
        if self._pattern == "single":
            self._comm.send_haptic(HAPTIC_STRENGTH, HAPTIC_DURATION_MS)

        elif self._pattern == "triple":
            # Three short pulses at 1-Hz-like spacing to cue stepping cadence
            for _ in range(3):
                self._comm.send_haptic(HAPTIC_STRENGTH, 150)
                time.sleep(0.35)  # 150ms pulse + 200ms gap

        elif self._pattern == "rhythmic":
            # Single short pulse — call trigger() externally at 1 Hz
            self._comm.send_haptic(HAPTIC_STRENGTH, 100)
