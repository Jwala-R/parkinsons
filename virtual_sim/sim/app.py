"""
sim/app.py — Main Panda3D ShowBase application.

Orchestrates all subsystems:
  - Serial reader thread (reads Arduino/Mock at 60 Hz into a queue)
  - FoG detector (drains queue, runs inference every 1 second)
  - Active task (scene, motion mapping, progress tracking)
  - Avatar state machine
  - HUD overlay
  - Haptic feedback

Threading notes:
  - All Panda3D calls MUST happen on the main thread.
  - The serial reader thread ONLY puts frames into frame_queue.
  - The Panda3D taskMgr drains the queue each frame on the main thread.
"""

import queue
import threading
import time
import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight, DirectionalLight, LVector3,
    LColor, WindowProperties,
)

from virtual_sim.config import (
    WIN_TITLE, WIN_W, WIN_H, FPS, N_CHANNELS,
    WRIST_CH_START, WRIST_CH_END, MOCK_MODE,
)
from virtual_sim.sim.hud import HUD
from virtual_sim.sim.avatar import Avatar


class SimApp(ShowBase):
    """
    Main Panda3D application.

    Parameters
    ----------
    comm       : ArduinoComm or MockArduino
    detector   : FogDetector
    haptic     : HapticController
    task       : BaseTask subclass (WalkingTask or EatingTask)
    live_mode  : bool — True if connected to real Arduino
    """

    def __init__(self, comm, detector, haptic, task,
                 live_mode: bool = False, mode: str = "demo"):
        super().__init__()

        self._comm = comm
        self._detector = detector
        self._haptic = haptic
        self._task = task
        # Support both old bool kwarg and new string mode kwarg
        if mode != "demo":
            self._mode = mode
        else:
            self._mode = "live" if live_mode else "demo"

        self._frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self._fog_score = 0.0
        self._is_fog = False
        self._last_frame_data: dict | None = None
        self._task_complete_shown = False

        # ── Window setup ───────────────────────────────────────────────────
        props = WindowProperties()
        props.setTitle(WIN_TITLE)
        props.setSize(WIN_W, WIN_H)
        props.setOrigin(100, 50)   # force top-left position so it appears on screen
        props.setUndecorated(False)
        props.setForeground(True)
        self.win.requestProperties(props)

        self.setFrameRateMeter(True)
        self.setBackgroundColor(0.12, 0.12, 0.14, 1)   # dark grey default bg
        self.disableMouse()  # we drive the camera manually
        self.clock.setMode(self.clock.MLimited)
        self.clock.setFrameRate(FPS)

        # ── Lighting ───────────────────────────────────────────────────────
        ambient = AmbientLight("ambient")
        ambient.setColor(LColor(0.4, 0.4, 0.4, 1))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(LColor(0.9, 0.9, 0.8, 1))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(45, -45, 0)
        self.render.setLight(sun_np)

        # ── Scene + avatar ─────────────────────────────────────────────────
        self._task.setup_scene(self.render, self.loader)
        self._avatar = Avatar(self.render, start_pos=self._task.avatar_start_pos)

        # Position camera per task
        self._task.setup_camera(self.camera)

        # ── HUD ────────────────────────────────────────────────────────────
        self._hud = HUD(self)

        # ── Serial reader thread ───────────────────────────────────────────
        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._serial_reader, daemon=True
        )
        self._reader_thread.start()

        # ── Panda3D tasks ──────────────────────────────────────────────────
        self.taskMgr.add(self._update, "sim_update")

        # ESC to quit
        self.accept("escape", self._shutdown)

    # ── Main update task ───────────────────────────────────────────────────

    def _update(self, task: Task) -> int:
        # Drain frame queue (may contain several frames if serial burst arrived)
        while not self._frame_queue.empty():
            frame_data = self._frame_queue.get_nowait()
            self._last_frame_data = frame_data

            imu = self._parse_to_array(frame_data)
            result = self._detector.push_frame(imu)
            if result is not None:
                self._fog_score, self._is_fog = result
                if self._is_fog:
                    self._haptic.trigger(self._fog_score)

        # Update task and avatar every frame
        self._task.update(self._last_frame_data, self._fog_score, self._is_fog)
        self._avatar.update(self._last_frame_data, self._fog_score)

        # Follow avatar with camera (walking task only — eating task fixes camera)
        self._task.update_camera(self.camera, self._avatar)

        # HUD
        self._hud.update(
            self._fog_score,
            self._is_fog,
            self._task.progress,
            mode=self._mode,
        )

        # Camera mode: draw live hand skeleton overlay + calibration screen
        if self._mode == "camera":
            from virtual_sim.config import CAMERA_CALIB_FRAMES
            lm = getattr(self._comm, "latest_landmarks", None)
            cd = getattr(self._comm, "calib_done", False)
            cc = getattr(self._comm, "calib_count", 0)
            self._hud.update_hand(lm, cd, calib_count=cc,
                                  calib_total=CAMERA_CALIB_FRAMES)

        # Check task completion (guard so we only fire once)
        if self._task.is_complete and not self._task_complete_shown:
            self._task_complete_shown = True
            self._on_task_complete()

        return Task.cont

    # ── Serial reader thread ───────────────────────────────────────────────

    def _serial_reader(self):
        """Background thread: read frames from Arduino/Mock and enqueue them."""
        while not self._stop_event.is_set():
            frame = self._comm.read_frame()
            if frame is not None and not self._frame_queue.full():
                self._frame_queue.put_nowait(frame)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _parse_to_array(self, frame_data: dict) -> np.ndarray:
        """
        Map Arduino frame dict to the 24-channel array the ML model expects.
        Wrist data fills channels 18-23; all others remain zero.
        """
        v = np.zeros(N_CHANNELS, dtype=np.float32)
        v[WRIST_CH_START:WRIST_CH_START + 3] = frame_data["acc"]
        v[WRIST_CH_START + 3:WRIST_CH_END]   = frame_data["gyro"]
        return v

    def _on_task_complete(self):
        """Show completion overlay and quit after a few seconds."""
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        OnscreenText(
            text="Task Complete!",
            pos=(0, 0),
            scale=0.15,
            fg=(0.2, 1.0, 0.2, 1.0),
            align=TextNode.ACenter,
        )
        self.taskMgr.doMethodLater(4.0, lambda t: self._shutdown(), "quit_delay")

    def _shutdown(self):
        self._stop_event.set()
        self._haptic.stop()
        self._comm.disconnect()
        self.taskMgr.remove("sim_update")
        self.userExit()
