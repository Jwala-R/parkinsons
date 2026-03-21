"""
arduino/camera.py — Hand-tracking mode using a webcam + MediaPipe.

Replaces the physical Arduino sensor with computer-vision-derived IMU
equivalents, derived from wrist and hand landmark positions tracked in
real time.  The output dict matches ArduinoComm exactly so the rest of
the simulation (detector, tasks, haptic) is unaffected.

How it works
------------
1. MediaPipe Hands detects 21 landmarks per hand at up to 60 fps.
2. Wrist position (landmark 0) is tracked over consecutive frames.
3. Finite differences on wrist position give a velocity proxy for acc_x/acc_y.
4. Rotation of the wrist-to-middle-finger vector gives a gyro proxy.
5. Pinch distance (thumb tip ↔ index tip) maps to the FSR pressure value.

Calibration
-----------
A short calibration routine (CAMERA_CALIB_FRAMES frames) captures the
neutral hand pose at rest.  The resting wrist centroid and rotation are
stored; all subsequent readings are reported relative to those baselines.
This normalises sensor values across different users and camera positions.

Run standalone to test / re-calibrate:
    python virtual_sim/arduino/camera.py

Installation:
    pip install mediapipe opencv-python

Reference landmarks (MediaPipe Hand):
    0  WRIST
    4  THUMB_TIP
    8  INDEX_FINGER_TIP
    12 MIDDLE_FINGER_TIP
"""

import time
import math
import threading
import queue
from typing import Optional

# Lazy imports so the sim still loads if cv2/mediapipe are absent
_mp_available = False
try:
    import cv2
    import mediapipe as mp
    _mp_available = True
except ImportError:
    pass

from virtual_sim.config import (
    SAMPLING_RATE,
    CAMERA_DEVICE_ID,
    CAMERA_CALIB_FRAMES,
    CAMERA_ACCEL_SCALE,
    CAMERA_ACCEL_SCALE_Y,
    CAMERA_GYRO_SCALE,
    CAMERA_PRESSURE_SCALE,
    CAMERA_PINCH_OPEN,
    CAMERA_PINCH_CLOSED,
)


class CameraHands:
    """
    Webcam-based hand-tracking sensor that mimics the ArduinoComm interface.

    Runs MediaPipe Hands in a background thread.  Frames are enqueued and
    consumed by the main thread via read_frame(), exactly like the serial
    reader thread in ArduinoComm.

    Modes:
        calibrating — first CAMERA_CALIB_FRAMES frames, computing baseline
        running     — normal operation, values relative to calibration
    """

    def __init__(self, device_id: Optional[int] = None):
        self._device_id = device_id if device_id is not None else CAMERA_DEVICE_ID
        self._frame_queue: queue.Queue = queue.Queue(maxsize=60)
        self._stop_event = threading.Event()
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        # Calibration state
        self._calib_samples: list[tuple] = []   # list of (wx,wy,angle) tuples
        self._calib_done = False
        self._baseline_wx = 0.0
        self._baseline_wy = 0.0
        self._baseline_angle = 0.0

        # Previous frame state (for finite differences)
        self._prev_wx: Optional[float] = None
        self._prev_wy: Optional[float] = None
        self._prev_t: Optional[float] = None

        self.parse_errors = 0

        # Latest landmarks exposed for the HUD hand overlay.
        # Written by background thread, read by main thread — a single
        # assignment is atomic in CPython so no lock needed.
        # Value: list of 21 (x, y) tuples in [0,1] normalised coords, or None.
        self.latest_landmarks: Optional[list] = None
        self.calib_done: bool = False   # public mirror of _calib_done
        self.calib_count: int = 0       # how many calibration frames collected

        # Preview frame queue: reader thread puts annotated BGR frames here;
        # a dedicated preview thread calls imshow from its own thread so it
        # never conflicts with Panda3D's Win32 message pump.
        self._preview_queue: queue.Queue = queue.Queue(maxsize=2)
        self._preview_thread: Optional[threading.Thread] = None

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not _mp_available:
            print("[CameraHands] mediapipe or opencv-python not installed.")
            print("              pip install mediapipe opencv-python")
            return False

        # VideoCapture and MediaPipe are initialised inside the background
        # thread so they don't touch the main thread's COM/DirectX context
        # before Panda3D can claim it (Windows-specific conflict).
        self._connected = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()
        print(f"[CameraHands] Starting camera thread for device {self._device_id}.")
        print(f"[CameraHands] Calibrating — hold your hand still for "
              f"{CAMERA_CALIB_FRAMES} frames...")
        return True

    def disconnect(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._preview_thread:
            self._preview_thread.join(timeout=2.0)
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Reading ─────────────────────────────────────────────────────────────

    def read_frame(self) -> Optional[dict]:
        """Non-blocking pop from the landmark queue."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    # ── Writing (no haptic on camera mode) ──────────────────────────────────

    def send_haptic(self, strength: int, duration_ms: int = 500):
        """Camera mode has no haptic output — prints to console only."""
        if strength > 0:
            print(f"[CameraHands] Haptic cue: PWM={strength} (no hardware)")

    # ── Background reader ────────────────────────────────────────────────────

    def _reader_loop(self):
        # Brief pause so Panda3D's WGL/DirectX window init completes before
        # we ask Media Foundation to open the camera on Windows.
        time.sleep(2.0)

        cap = cv2.VideoCapture(self._device_id, cv2.CAP_MSMF)
        if not cap.isOpened():
            print(f"[CameraHands] Cannot open camera {self._device_id}.")
            self._connected = False
            return
        cap.set(cv2.CAP_PROP_FPS, SAMPLING_RATE)
        self._cap = cap

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        frame_dt = 1.0 / SAMPLING_RATE
        last_t = time.monotonic()

        while not self._stop_event.is_set():
            # Rate-limit to SAMPLING_RATE
            now = time.monotonic()
            if now - last_t < frame_dt:
                time.sleep(0.001)
                continue
            last_t = now

            ret, bgr = cap.read()
            if not ret:
                continue

            # Mirror image so hand movement matches screen direction naturally
            bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            frame_data = None
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                self.latest_landmarks = [(p.x, p.y) for p in lm]
                frame_data = self._landmarks_to_frame(lm, now, bgr.shape)
                # Annotate for preview
                mp_draw = mp.solutions.drawing_utils
                mp_styles = mp.solutions.drawing_styles
                for hand_lm in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        bgr, hand_lm,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
            else:
                self.latest_landmarks = None

            # Overlay calibration / status text on preview frame
            if not self._calib_done:
                remaining = CAMERA_CALIB_FRAMES - self.calib_count
                label = f"Calibrating: {self.calib_count}/{CAMERA_CALIB_FRAMES}  ({remaining} left)"
                color = (0, 200, 255)
            elif frame_data is not None:
                ax, ay = frame_data["acc"][0], frame_data["acc"][1]
                p = frame_data["pressure"]
                label = f"acc=({ax:+.2f},{ay:+.2f})  pinch={p:.0f}"
                color = (60, 255, 100)
            else:
                label = "No hand detected"
                color = (100, 100, 100)
            cv2.putText(bgr, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # Push to preview queue (drop if full — never block reader)
            try:
                self._preview_queue.put_nowait(bgr)
            except queue.Full:
                pass

            if frame_data is not None:
                try:
                    self._frame_queue.put_nowait(frame_data)
                except queue.Full:
                    pass

        hands.close()

    # ── Preview window (own thread to avoid Panda3D Win32 conflict) ──────────

    def _preview_loop(self):
        """
        Owns the OpenCV imshow window.  Runs on its own daemon thread so
        cv2's Win32 message handling never touches Panda3D's message pump.
        Starts after a short delay to let Panda3D finish window creation first.
        """
        time.sleep(3.0)   # wait for Panda3D window to fully initialise
        window_name = "Camera — FoG Hand Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 480, 360)

        while not self._stop_event.is_set():
            try:
                frame = self._preview_queue.get(timeout=0.1)
            except queue.Empty:
                cv2.waitKey(1)
                continue
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    # ── Landmark processing ──────────────────────────────────────────────────

    def _landmarks_to_frame(self, lm, t: float, shape: tuple) -> dict:
        h, w = shape[0], shape[1]

        # Wrist position in pixel coords (normalised to [-1, 1])
        wx = (lm[0].x - 0.5) * 2.0   # -1 = left edge, +1 = right edge
        wy = (lm[0].y - 0.5) * 2.0   # -1 = top,       +1 = bottom

        # Wrist-to-middle-finger vector angle (proxy for wrist rotation)
        dx = lm[12].x - lm[0].x
        dy = lm[12].y - lm[0].y
        angle = math.atan2(dy, dx)    # radians

        # Calibration phase
        if not self._calib_done:
            return self._calibrate(wx, wy, angle)

        # Values relative to calibration baseline
        rel_wx = wx - self._baseline_wx
        rel_wy = wy - self._baseline_wy
        rel_angle = _angle_diff(angle, self._baseline_angle)

        # Direct position mapping: wrist screen position drives acc_x/acc_y
        # directly, scaled so the full hand travel range covers the full game
        # world range.  Tasks use acc_x*0.35 and acc_y*0.08 for position, so
        # we scale to ±4.5 (x) and ±13 (y) to fill the playable area.
        # rel_wx/rel_wy are in [-1, 1] relative to the calibration centre.
        acc_x = rel_wx * CAMERA_ACCEL_SCALE         # left/right
        acc_y = -rel_wy * CAMERA_ACCEL_SCALE_Y    # flip Y: hand up = spoon up
        acc_z = 9.8  # gravity constant (no depth estimation)

        # Gyro proxy from wrist rotation angle (not rate — keeps it stable)
        gyro_z = rel_angle * CAMERA_GYRO_SCALE
        gyro_x = 0.0
        gyro_y = 0.0

        # Pinch pressure: distance between thumb tip (4) and index tip (8)
        pinch_d = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        # Map: open (CAMERA_PINCH_OPEN) -> 0, closed (CAMERA_PINCH_CLOSED) -> 1023
        t_pinch = 1.0 - _clamp(
            (pinch_d - CAMERA_PINCH_CLOSED) / (CAMERA_PINCH_OPEN - CAMERA_PINCH_CLOSED),
            0.0, 1.0
        )
        pressure = t_pinch * CAMERA_PRESSURE_SCALE

        return {
            "acc":      [acc_x, acc_y, acc_z],
            "gyro":     [gyro_x, gyro_y, gyro_z],
            "pressure": pressure,
        }

    def _calibrate(self, wx: float, wy: float, angle: float) -> Optional[dict]:
        """
        Collect calibration samples.  Returns None until done.
        On completion returns the neutral (zero-motion) frame.
        """
        self._calib_samples.append((wx, wy, angle))
        n = len(self._calib_samples)
        self.calib_count = n
        remaining = CAMERA_CALIB_FRAMES - n
        if remaining % 10 == 0 and remaining > 0:
            print(f"[CameraHands] Calibrating... {remaining} frames remaining")

        if n >= CAMERA_CALIB_FRAMES:
            xs, ys, angles = zip(*self._calib_samples)
            self._baseline_wx = sum(xs) / n
            self._baseline_wy = sum(ys) / n
            self._baseline_angle = _mean_angle(list(angles))
            self._calib_done = True
            self.calib_done = True
            self._prev_wx = self._baseline_wx
            self._prev_wy = self._baseline_wy
            self._prev_t = time.monotonic()
            print("[CameraHands] Calibration complete. Starting detection.")
            return {
                "acc":      [0.0, 0.0, 9.8],
                "gyro":     [0.0, 0.0, 0.0],
                "pressure": 0.0,
            }
        return None



# ── Helpers ──────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _angle_diff(a: float, b: float) -> float:
    """Shortest signed difference between two angles in radians."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _mean_angle(angles: list[float]) -> float:
    """Circular mean of a list of angles (radians)."""
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    return math.atan2(sin_sum / len(angles), cos_sum / len(angles))


# ── Standalone calibration / test ────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.split("virtual_sim")[0])

    cam = CameraHands()
    if not cam.connect():
        print("Could not open camera. Check CAMERA_DEVICE_ID in config.py.")
        sys.exit(1)

    print("Reading frames (Ctrl-C to stop)...")
    try:
        while True:
            f = cam.read_frame()
            if f:
                ax, ay, az = f["acc"]
                gx, gy, gz = f["gyro"]
                p = f["pressure"]
                print(f"acc=({ax:+.3f},{ay:+.3f},{az:+.3f}) "
                      f"gyro=({gx:+.3f},{gy:+.3f},{gz:+.3f}) "
                      f"pressure={p:.1f}")
            time.sleep(1 / SAMPLING_RATE)
    except KeyboardInterrupt:
        pass
    finally:
        cam.disconnect()
