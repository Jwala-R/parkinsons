"""
fog/detector.py — Real-time FoG detection using the trained Approach C model.

Wraps the ML inference pipeline:
  1. Receives raw (24,) IMU frames via push_frame()
  2. Buffers in RingBuffer
  3. Every STEP_SIZE frames (1 second at 60 Hz), extracts 339 features
  4. Scores using a wrist-aware scoring path (see _infer docstring)
  5. Returns (score, is_fog) tuple

Channel layout expected by the ML model (from fog_star_loader.py):
  channels 0-5:   left ankle  (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
  channels 6-11:  right ankle (same)
  channels 12-17: back/trunk  (same)
  channels 18-23: wrist       (same)  <- Arduino watch provides these

Wrist-only scoring:
  The original score() method uses _normalize_01() which collapses to 0.0
  when called with a single sample (min == max).
  Instead, we use IsolationForest.decision_function() which is already
  calibrated relative to the training threshold (offset_):
    - Positive values -> normal gait (inside the learned normal region)
    - Negative values -> anomalous (FoG)
  We convert this to a [0, 1] score using a sigmoid centred at 0 (the
  decision boundary), so the output is comparable to FOG_THRESHOLD.

  Freeze Index (FI) is computed from wrist vertical acc (channel 19, acc_y)
  when the ankle channels are zero. This is less precise than the ankle-based
  FI the model was trained with, but still captures the 3-8 Hz tremor band.
"""

import sys
import os
import math
import numpy as np
import joblib
from scipy.signal import welch

from virtual_sim.config import (
    MODEL_PATH, BUNDLE_PATH, FOG_THRESHOLD,
    WINDOW_SIZE, STEP_SIZE, N_CHANNELS, SAMPLING_RATE,
    WRIST_CH_START, WRIST_CH_END,
)
from virtual_sim.fog.buffer import RingBuffer

# Add ml/ to sys.path so we can import ml/src without installing it
_ML_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "ml"))
if _ML_ROOT not in sys.path:
    sys.path.insert(0, _ML_ROOT)

from src.data.features import extract_window_features  # noqa: E402

# Sigmoid steepness: k=15 means score crosses 0.5 exactly at the IF boundary,
# with a steep-enough curve to give clear separation between normal and FoG
_SIG_K     = 15.0
# Weight for the Freeze Index component (wrist-based; less reliable than ankle)
_FI_WEIGHT = 0.35
_IF_WEIGHT = 1.0 - _FI_WEIGHT


class FogDetector:
    """
    Real-time FoG detector wrapping the Approach C population model.

    Usage:
        detector = FogDetector()
        # in main loop:
        result = detector.push_frame(imu_24ch_array)
        if result is not None:
            score, is_fog = result
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        bundle_path: str = BUNDLE_PATH,
        threshold: float = FOG_THRESHOLD,
    ):
        print("[FogDetector] Loading model...")
        self.ensemble  = joblib.load(model_path)
        self._detector = self.ensemble.pop_detector_
        self.threshold = threshold
        self._fs       = float(SAMPLING_RATE)

        self._buffer       = RingBuffer()
        self._step_counter = 0
        self.last_score    = 0.0
        self.last_is_fog   = False

        print(f"[FogDetector] Ready. Threshold={threshold:.2f}")

    # ── Public API ─────────────────────────────────────────────────────────

    def push_frame(self, imu_frame: np.ndarray) -> tuple[float, bool] | None:
        """
        Push one 24-channel IMU sample.

        imu_frame: (24,) float32 — all channels in fog_star_loader order.
        Returns (score, is_fog) every STEP_SIZE frames, else None.
        """
        self._buffer.push(imu_frame)
        self._step_counter += 1

        if self._step_counter < STEP_SIZE:
            return None
        self._step_counter = 0

        window = self._buffer.get_window()  # (120, 24) or None
        if window is None:
            return None

        return self._infer(window)

    def push_wrist_frame(self, wrist_6ch: np.ndarray) -> tuple[float, bool] | None:
        """
        Convenience for a wrist-only Arduino.
        wrist_6ch: (6,) [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        """
        full_frame = np.zeros(N_CHANNELS, dtype=np.float32)
        full_frame[WRIST_CH_START:WRIST_CH_END] = wrist_6ch
        return self.push_frame(full_frame)

    # ── Private ────────────────────────────────────────────────────────────

    def _infer(self, window: np.ndarray) -> tuple[float, bool]:
        """
        Score one (120, 24) window.

        Uses decision_function instead of score() to get a meaningful
        single-sample score, then blends with a wrist-based Freeze Index.
        """
        # 1. Feature extraction
        features = extract_window_features(window, fs=self._fs)   # (339,)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_2d = features[np.newaxis, :]                      # (1, 339)

        # 2. IsolationForest decision function
        #    decision_function returns values centred on 0 (training threshold):
        #      > 0  -> inlier  (normal gait)
        #      < 0  -> outlier (FoG)
        #    We negate and apply a sigmoid to map to [0,1].
        feat_scaled = self._detector.scaler_.transform(features_2d)
        df = float(self._detector.iso_forest_.decision_function(feat_scaled)[0])
        if_score = _sigmoid(-df, k=_SIG_K)   # higher = more anomalous

        # 3. Freeze Index from wrist vertical accelerometer (channel 19, acc_y)
        #    The standard FI uses ankle channels 1,7,13 which are zero in
        #    wrist-only mode. We substitute channel 19 (wrist acc_y).
        wrist_acc_y = window[:, 19]           # (120,)
        fi_score = _freeze_index(wrist_acc_y, fs=self._fs)

        # 4. Blend
        score = _IF_WEIGHT * if_score + _FI_WEIGHT * fi_score
        score = float(np.clip(score, 0.0, 1.0))

        is_fog = score >= self.threshold
        self.last_score   = score
        self.last_is_fog  = is_fog
        return score, is_fog


# ── Signal helpers ─────────────────────────────────────────────────────────────

def _sigmoid(x: float, k: float = 15.0) -> float:
    """Numerically stable sigmoid: 1 / (1 + exp(-k*x))."""
    if x >= 0:
        e = math.exp(-k * x)
        return 1.0 / (1.0 + e)
    else:
        e = math.exp(k * x)
        return e / (1.0 + e)


def _freeze_index(acc_y: np.ndarray, fs: float = 60.0) -> float:
    """
    Compute the Freeze Index from a single vertical acceleration signal.

    FI = power(3-8 Hz) / (power(0.5-3 Hz) + power(3-8 Hz))
    Returns a value in [0, 1] where higher = more freeze-like activity.

    Using a ratio to a sum (rather than a bare ratio) keeps the output
    bounded even when locomotion-band power is very small.
    """
    nperseg = min(len(acc_y), 64)
    freqs, psd = welch(acc_y, fs=fs, nperseg=nperseg)

    loco_mask   = (freqs >= 0.5) & (freqs <= 3.0)
    freeze_mask = (freqs >= 3.0) & (freqs <= 8.0)

    loco_power   = float(np.trapz(psd[loco_mask],  freqs[loco_mask]))
    freeze_power = float(np.trapz(psd[freeze_mask], freqs[freeze_mask]))

    total = loco_power + freeze_power
    if total < 1e-10:
        return 0.0
    return freeze_power / total
