"""
virtual_sim/test_pipeline.py — Headless end-to-end pipeline simulation test.

Runs the full sensor -> FoG detection -> haptic feedback pipeline WITHOUT
opening any Panda3D window. Simulates ~65 seconds of sensor data at 60 Hz,
covering at least 3 FoG episodes (injected every 20s for 5s each).

What this verifies:
  1. MockArduino generates physiologically plausible sensor data
  2. RingBuffer accumulates frames correctly
  3. FogDetector fires inference every STEP_SIZE (60) frames = once per second
  4. ML model produces scores and detects FoG during 5 Hz tremor episodes
  5. HapticController respects the 2s cooldown and calls send_haptic()
  6. Console output shows motion type, FoG score, and haptic events live

Run from the parkinsons/ folder:
    python virtual_sim/test_pipeline.py

No Arduino required. No Panda3D required. Only numpy/scipy/sklearn/joblib.
"""

import sys
import os
import time
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_REPO    = os.path.dirname(_HERE)
_ML_ROOT = os.path.join(_REPO, "ml")
for p in [_REPO, _ML_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from virtual_sim.config import (
    SAMPLING_RATE, N_CHANNELS, WRIST_CH_START, WRIST_CH_END,
    STEP_SIZE, WINDOW_SIZE, FOG_THRESHOLD,
    MOCK_FOG_INTERVAL, MOCK_FOG_DURATION,
    HAPTIC_COOLDOWN_S,
)
from virtual_sim.arduino.mock import MockArduino
from virtual_sim.fog.detector import FogDetector
from virtual_sim.fog.buffer import RingBuffer
from virtual_sim.haptic.feedback import HapticController

# ── ANSI colour helpers ────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_GREY   = "\033[90m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _col(text, code):
    return f"{code}{text}{_RESET}"

# ── Haptic event logger ────────────────────────────────────────────────────────

class LoggingComm:
    """
    Stand-in for ArduinoComm/MockArduino that logs haptic commands and
    also counts them so we can assert they fired.
    """
    def __init__(self):
        self.haptic_count = 0
        self.haptic_log: list[tuple[float, int, int]] = []

    def connect(self):          return True
    def disconnect(self):       pass
    def read_frame(self):       return None
    @property
    def is_connected(self):     return True

    def send_haptic(self, strength: int, duration_ms: int = 500):
        if strength > 0:
            self.haptic_count += 1
            t = time.monotonic()
            self.haptic_log.append((t, strength, duration_ms))
            print(_col(
                f"  [HAPTIC] PWM={strength}  dur={duration_ms}ms  "
                f"(event #{self.haptic_count})",
                _YELLOW
            ))


# ── Score bar renderer ─────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 30) -> str:
    filled = int(round(score * width))
    bar = "#" * filled + "-" * (width - filled)
    if score >= FOG_THRESHOLD:
        colour = _RED
    elif score >= FOG_THRESHOLD * 0.6:
        colour = _YELLOW
    else:
        colour = _GREEN
    return f"|{_col(bar, colour)}| {score:.3f}"


# ── Main simulation ────────────────────────────────────────────────────────────

def run_simulation(duration_s: float = 65.0, realtime: bool = True):
    """
    Run the pipeline for `duration_s` seconds.

    realtime=True  : sleep between frames to match 60 Hz wall clock
    realtime=False : run as fast as possible (stress-test, no sleep)
    """
    print(_col("\n" + "=" * 70, _BOLD))
    print(_col("  FoG Pipeline Simulation Test", _BOLD))
    print(_col("  Parkinson's Neuroplasticity Simulator — Headless Mode", _BOLD))
    print(_col("=" * 70, _BOLD))
    print(f"  Duration:      {duration_s:.0f}s")
    print(f"  Sample rate:   {SAMPLING_RATE} Hz  ({int(duration_s * SAMPLING_RATE):,} total frames)")
    print(f"  Inference:     every {STEP_SIZE} frames (= 1 s)")
    print(f"  FoG episodes:  every {MOCK_FOG_INTERVAL:.0f}s, lasting {MOCK_FOG_DURATION:.0f}s")
    print(f"  Threshold:     {FOG_THRESHOLD}")
    print(f"  Haptic cooldown: {HAPTIC_COOLDOWN_S}s")
    print(_col("=" * 70, _BOLD))
    print()

    # ── Component init ─────────────────────────────────────────────────────
    mock    = MockArduino(fast_forward=not realtime)
    mock.connect()

    haptic_comm = LoggingComm()
    haptic = HapticController(haptic_comm)
    haptic.set_pattern("single")

    print("Loading FoG detection model...")
    detector = FogDetector(threshold=FOG_THRESHOLD)
    print("Model loaded.\n")

    frame_dt   = 1.0 / SAMPLING_RATE
    total_frames  = int(duration_s * SAMPLING_RATE)
    infer_count   = 0
    fog_detections = 0
    frames_processed = 0

    # Track episode boundary for logging
    last_motion_type = None

    print(_col(
        f"{'Time':>6}  {'Motion':<10}  {'Score Bar':^38}  {'Status':<16}",
        _BOLD
    ))
    print("-" * 80)

    sim_start = time.monotonic()

    frame_num = 0
    while frame_num < total_frames:
        loop_start = time.monotonic()

        # ── Get one frame from mock ────────────────────────────────────────
        frame = mock.read_frame()
        if frame is None:
            if realtime:
                time.sleep(0.001)
            continue

        frame_num += 1
        frames_processed += 1

        # Use virtual time (fast_forward) or wall clock (realtime)
        if realtime:
            elapsed = time.monotonic() - sim_start
        else:
            elapsed = mock._virtual_t  # virtual seconds elapsed

        # ── Map to 24-channel array (wrist = channels 18-23) ──────────────
        imu = np.zeros(N_CHANNELS, dtype=np.float32)
        imu[WRIST_CH_START    : WRIST_CH_START + 3] = frame["acc"]
        imu[WRIST_CH_START + 3: WRIST_CH_END]       = frame["gyro"]

        # ── Determine current motion type for display ──────────────────────
        phase = elapsed % MOCK_FOG_INTERVAL
        is_fog_episode = phase >= (MOCK_FOG_INTERVAL - MOCK_FOG_DURATION)
        motion_type = "FoG (5Hz)" if is_fog_episode else "Walking (1.8Hz)"

        # Log when motion type changes
        if motion_type != last_motion_type:
            tag = _col(">>> FoG episode START <<<", _RED + _BOLD) \
                  if is_fog_episode else _col(">>> Normal gait <<<", _GREEN)
            print(f"\n  {tag}\n")
            last_motion_type = motion_type

        # ── Run FoG detector ───────────────────────────────────────────────
        result = detector.push_frame(imu)

        if result is not None:
            score, is_fog = result
            infer_count += 1

            if is_fog:
                fog_detections += 1
                haptic.trigger(score)
                status = _col("FoG DETECTED", _RED + _BOLD)
            else:
                status = _col("Normal Gait ", _GREEN)

            bar = _score_bar(score)
            print(
                f"  {elapsed:>5.1f}s  {motion_type:<10}  {bar}  {status}"
            )

        # ── Rate limiting ──────────────────────────────────────────────────
        if realtime:
            elapsed_this_frame = time.monotonic() - loop_start
            sleep_time = frame_dt - elapsed_this_frame
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ── Wait for haptic threads to finish ─────────────────────────────────
    time.sleep(1.2)

    # ── Summary ───────────────────────────────────────────────────────────
    total_time = time.monotonic() - sim_start
    print("\n" + _col("=" * 70, _BOLD))
    print(_col("  SIMULATION SUMMARY", _BOLD))
    print(_col("=" * 70, _BOLD))
    print(f"  Frames processed:    {frames_processed:,}")
    print(f"  Inferences run:      {infer_count}  (expected ~{int(duration_s)})")
    print(f"  FoG detections:      {_col(str(fog_detections), _RED if fog_detections > 0 else _GREY)}")
    print(f"  Haptic events fired: {_col(str(haptic_comm.haptic_count), _YELLOW)}")
    print(f"  Wall-clock time:     {total_time:.1f}s  "
          f"({'realtime' if realtime else 'fast-forward'})")

    # ── Assertions ────────────────────────────────────────────────────────
    print()
    checks = [
        ("Frames processed",     frames_processed > 0,              f"{frames_processed} > 0"),
        ("Inferences ran",       infer_count >= int(duration_s) - 5, f"{infer_count} >= {int(duration_s)-5}"),
        ("FoG was detected",     fog_detections > 0,                f"{fog_detections} > 0"),
        ("Haptic fired on FoG",  haptic_comm.haptic_count > 0,      f"{haptic_comm.haptic_count} > 0"),
        ("Cooldown respected",   _check_cooldown(haptic_comm.haptic_log), "gaps >= 2.0s"),
    ]
    all_pass = True
    for name, ok, detail in checks:
        tag = _col("PASS", _GREEN) if ok else _col("FAIL", _RED)
        print(f"  [{tag}]  {name:<30}  {detail}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print(_col("  All checks passed. Pipeline is working correctly.", _GREEN + _BOLD))
    else:
        print(_col("  Some checks failed — see above.", _RED + _BOLD))
    print(_col("=" * 70, _BOLD) + "\n")

    return all_pass


def _check_cooldown(log: list[tuple]) -> bool:
    """Verify that consecutive haptic events are at least 1.9s apart."""
    if len(log) < 2:
        return True
    for i in range(1, len(log)):
        gap = log[i][0] - log[i - 1][0]
        if gap < 1.9:   # slight slack for thread timing
            return False
    return True


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FoG pipeline simulation test")
    parser.add_argument("--duration",  type=float, default=65.0,
                        help="Simulation duration in seconds (default 65)")
    parser.add_argument("--fast",      action="store_true",
                        help="Run as fast as possible instead of real-time 60 Hz")
    args = parser.parse_args()

    ok = run_simulation(duration_s=args.duration, realtime=not args.fast)
    sys.exit(0 if ok else 1)
