"""
fog/buffer.py — Thread-safe circular buffer for IMU frames.

Accumulates raw (24,) IMU samples and provides the most recent
WINDOW_SIZE samples as a (WINDOW_SIZE, N_CHANNELS) array on demand.
"""

import threading
from collections import deque
import numpy as np

from virtual_sim.config import WINDOW_SIZE, RING_BUFFER_CAP, N_CHANNELS


class RingBuffer:
    """
    Thread-safe circular buffer for raw IMU frames.

    push() is called from the serial reader thread.
    get_window() is called from the ML inference path (main thread).
    """

    def __init__(self, capacity: int = RING_BUFFER_CAP):
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray):
        """
        Add one (N_CHANNELS,) sample to the buffer.
        Oldest sample is automatically evicted when capacity is reached.
        """
        with self._lock:
            self._buf.append(frame.astype(np.float32))

    def get_window(self) -> np.ndarray | None:
        """
        Return the most recent WINDOW_SIZE samples as (WINDOW_SIZE, N_CHANNELS).
        Returns None if fewer than WINDOW_SIZE samples have been collected yet.
        """
        with self._lock:
            if len(self._buf) < WINDOW_SIZE:
                return None
            # deque[-WINDOW_SIZE:] — take the last WINDOW_SIZE elements
            return np.array(list(self._buf)[-WINDOW_SIZE:], dtype=np.float32)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
