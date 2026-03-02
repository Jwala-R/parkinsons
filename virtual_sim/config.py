"""
virtual_sim/config.py — All constants for the simulation.

Windowing parameters MUST mirror ml/configs/default.yaml exactly,
otherwise features extracted here will not match what the model was trained on.
"""

# ── Serial ────────────────────────────────────────────────────────────────────
SERIAL_PORT       = "COM3"      # Change to /dev/ttyACM0 on Linux/macOS
BAUD_RATE         = 115200
SERIAL_TIMEOUT    = 0.01        # seconds — non-blocking readline

# ── IMU / Windowing (must match ml/configs/default.yaml) ──────────────────────
SAMPLING_RATE     = 60          # Hz
N_CHANNELS        = 24          # full tensor expected by model
WRIST_CH_START    = 18          # wrist channels 18-23 in the 24-ch layout
WRIST_CH_END      = 24

WINDOW_SIZE       = 120         # samples (2s × 60Hz)
STEP_SIZE         = 60          # 50% overlap → new inference every 1s
RING_BUFFER_CAP   = 180         # 3s at 60 Hz

# ── FoG detection ─────────────────────────────────────────────────────────────
# Calibrated for wrist-only mode using decision_function + wrist Freeze Index.
# Walk scores ~0.50, FoG scores ~0.85-0.90. Threshold 0.52 gives F1=0.848.
FOG_THRESHOLD     = 0.52

import os as _os
_BASE = _os.path.dirname(_os.path.abspath(__file__))
MODEL_PATH  = _os.path.join(_BASE, "..", "ml", "results", "models",
                             "approach_c_outlier_detector.pkl")
BUNDLE_PATH = _os.path.join(_BASE, "..", "ml", "results", "models",
                             "approach_c_inference_bundle.pkl")

# ── Haptic feedback ───────────────────────────────────────────────────────────
HAPTIC_STRENGTH    = 200        # PWM 0-255
HAPTIC_DURATION_MS = 500        # single-pulse width
HAPTIC_COOLDOWN_S  = 2.0        # minimum gap between pulses

# ── Simulation display ────────────────────────────────────────────────────────
WIN_TITLE = "FoG Therapy Simulator"
WIN_W     = 1280
WIN_H     = 720
FPS       = 60

# ── Mock mode ─────────────────────────────────────────────────────────────────
MOCK_MODE          = False      # True = no Arduino required
MOCK_FOG_INTERVAL  = 20.0       # seconds between synthetic FoG episodes
MOCK_FOG_DURATION  = 5.0        # seconds per episode

# ── Task parameters ───────────────────────────────────────────────────────────
WALK_TARGET_STEPS  = 30
WALK_STEP_THRESHOLD = 0.3       # g — arm swing peak to count as step
EAT_TARGET_SCOOPS  = 5
EAT_PRESSURE_MIN   = 30.0       # FSR value below which grip is "lost"
EAT_FEED_TIMEOUT   = 2.0        # seconds to complete feed after scoop

# Whack-a-mole task
WHACK_TARGET_HITS  = 10         # hits required to complete the game
WHACK_PRESSURE_MIN = 40.0       # FSR threshold for a whack (squeeze)
MOLE_VISIBLE_S     = 2.5        # seconds a mole stays up before ducking
MOLE_INTERVAL_S    = 1.2        # seconds between moles
