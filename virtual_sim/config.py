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

# ── BLE mode (Arduino Nano 33 BLE Rev2) ───────────────────────────────────────
# The Arduino firmware must advertise with this name and expose the two UUIDs.
# Set BLE_PAYLOAD_FORMAT = "csv" if the firmware sends comma-separated text
# instead of the default 28-byte little-endian binary struct.
BLE_DEVICE_NAME    = "FoG-Nano"
BLE_IMU_CHAR_UUID  = "12345678-1234-1234-1234-123456789abd"
BLE_HAPTIC_CHAR_UUID = "12345678-1234-1234-1234-123456789abe"
BLE_PAYLOAD_FORMAT = "binary"   # "binary" (28-byte struct) or "csv"

# ── Camera / hand-tracking mode ───────────────────────────────────────────────
CAMERA_DEVICE_ID      = 0       # OpenCV device index (0 = default webcam)
CAMERA_CALIB_FRAMES   = 60      # frames of still hand to capture for baseline
CAMERA_SHOW_PREVIEW   = True    # show annotated webcam window alongside game

# Scaling factors: wrist screen position (rel to calibration, range ~[-1,1])
# is multiplied by these to produce acc_x / acc_y fed into the tasks.
# Eating task:     target_x = acc_x * 0.35  (needs ±4.3 for full ±1.5 range)
#                  target_z = 1.05 + acc_y * 0.08  (needs ±8 for full height)
# Whack-a-mole:    similar proportions.
# Increase these values if the cursor doesn't reach the edges of the play area.
CAMERA_ACCEL_SCALE    = 27.0    # lateral (acc_x): wrist position -> game units
CAMERA_ACCEL_SCALE_Y  = 54.0    # vertical (acc_y): wrist position -> game units
CAMERA_GYRO_SCALE     = 2.0     # radians -> deg/s proxy (not used for movement)

# Pinch gesture thresholds (normalised landmark distance, 0-1).
# CAMERA_PINCH_OPEN  : thumb-index distance treated as "no squeeze" (pressure 0)
# CAMERA_PINCH_CLOSED: distance treated as "full squeeze" (pressure 1023)
CAMERA_PINCH_OPEN     = 0.25
CAMERA_PINCH_CLOSED   = 0.04
CAMERA_PRESSURE_SCALE = 1023.0  # maps 0-1 pinch to FSR pressure range

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
