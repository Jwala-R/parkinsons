"""
sim/hud.py — Heads-up display for the FoG therapy simulator.

Draws on top of the 3D scene using Panda3D's 2D overlay system:
  - FoG risk bar (top-left, green → red)
  - Task progress (top-right)
  - Status text (bottom-center): "Normal Gait" / "FoG Detected"
  - Screen border flash (red) at 2 Hz during FoG

All elements live in aspect2d (Panda3D's 2D overlay node).
Coordinates: aspect2d runs from -ratio..+ratio horizontally and -1..+1 vertically.
"""

import math
import time
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectFrame import DirectFrame
from panda3d.core import TextNode, LColor, LineSegs, NodePath


class HUD:
    """
    Manages all 2D overlay elements.

    Call update(fog_score, is_fog, progress) every frame.
    Call destroy() on cleanup.
    """

    def __init__(self, base):
        """base: the Panda3D ShowBase instance."""
        self._base = base
        self._start_time = time.monotonic()

        # ── FoG score label ───────────────────────────────────────────────
        self._score_label = OnscreenText(
            text="FoG Risk",
            pos=(-1.6, 0.88),
            scale=0.06,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=False,
        )

        # FoG risk bar background (grey)
        self._bar_bg = DirectFrame(
            frameColor=(0.3, 0.3, 0.3, 0.8),
            frameSize=(0, 0.5, -0.04, 0.04),
            pos=(-1.6, 0, 0.80),
        )
        # FoG risk bar fill (green → red) — starts at zero width
        self._bar_fill = DirectFrame(
            frameColor=(0.2, 0.8, 0.2, 1.0),
            frameSize=(0, 0.001, -0.04, 0.04),
            pos=(-1.6, 0, 0.80),
        )

        self._score_value = OnscreenText(
            text="0.00",
            pos=(-1.08, 0.78),
            scale=0.055,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True,
        )

        # ── Task progress (top-right) ─────────────────────────────────────
        self._progress_text = OnscreenText(
            text="",
            pos=(1.3, 0.88),
            scale=0.06,
            fg=(1, 1, 1, 1),
            align=TextNode.ARight,
            mayChange=True,
        )

        # ── Status text (bottom-center) ───────────────────────────────────
        self._status_text = OnscreenText(
            text="Normal Gait",
            pos=(0.0, -0.88),
            scale=0.07,
            fg=(0.2, 1.0, 0.2, 1.0),
            align=TextNode.ACenter,
            mayChange=True,
        )

        # ── Mode indicator (bottom-right) ─────────────────────────────────
        self._mode_text = OnscreenText(
            text="DEMO",
            pos=(1.55, -0.92),
            scale=0.05,
            fg=(0.8, 0.8, 0.2, 1.0),
            align=TextNode.ARight,
            mayChange=True,
        )

        # ── Screen-edge flash border (hidden by default) ──────────────────
        self._border_top = DirectFrame(
            frameColor=(1, 0, 0, 0),
            frameSize=(-1.8, 1.8, -0.03, 0.03),
            pos=(0, 0, 0.97),
        )
        self._border_bot = DirectFrame(
            frameColor=(1, 0, 0, 0),
            frameSize=(-1.8, 1.8, -0.03, 0.03),
            pos=(0, 0, -0.97),
        )
        self._border_left = DirectFrame(
            frameColor=(1, 0, 0, 0),
            frameSize=(-0.03, 0.03, -1.0, 1.0),
            pos=(-1.57, 0, 0),
        )
        self._border_right = DirectFrame(
            frameColor=(1, 0, 0, 0),
            frameSize=(-0.03, 0.03, -1.0, 1.0),
            pos=(1.57, 0, 0),
        )
        self._borders = [
            self._border_top, self._border_bot,
            self._border_left, self._border_right,
        ]

        # ── Camera hand skeleton overlay (bottom-right) ───────────────────
        # Drawn as LineSegs in aspect2d.  Only visible in camera mode.
        self._hand_node: NodePath | None = None
        self._hand_label = OnscreenText(
            text="",
            pos=(1.55, -0.60),
            scale=0.045,
            fg=(0.4, 0.9, 1.0, 1.0),
            align=TextNode.ARight,
            mayChange=True,
        )

        # ── Camera calibration overlay (shown until calibration completes) ─
        self._calib_overlay_bg = DirectFrame(
            frameColor=(0.0, 0.0, 0.0, 0.78),
            frameSize=(-2.0, 2.0, -1.1, 1.1),
            pos=(0, 0, 0),
        )
        self._calib_title = OnscreenText(
            text="Camera Calibration",
            pos=(0, 0.35),
            scale=0.11,
            fg=(0.4, 0.9, 1.0, 1.0),
            align=TextNode.ACenter,
            mayChange=False,
        )
        self._calib_instruction = OnscreenText(
            text="Hold your hand still in front of the camera",
            pos=(0, 0.18),
            scale=0.07,
            fg=(1.0, 1.0, 1.0, 1.0),
            align=TextNode.ACenter,
            mayChange=False,
        )
        self._calib_progress_bg = DirectFrame(
            frameColor=(0.3, 0.3, 0.3, 1.0),
            frameSize=(-0.6, 0.6, -0.03, 0.03),
            pos=(0, 0, 0.0),
        )
        self._calib_progress_fill = DirectFrame(
            frameColor=(0.3, 0.85, 0.4, 1.0),
            frameSize=(-0.6, -0.6, -0.03, 0.03),  # zero width initially
            pos=(0, 0, 0.0),
        )
        self._calib_count_text = OnscreenText(
            text="0 / 60",
            pos=(0, -0.10),
            scale=0.065,
            fg=(0.8, 0.8, 0.8, 1.0),
            align=TextNode.ACenter,
            mayChange=True,
        )
        self._calib_hand_status = OnscreenText(
            text="Waiting for hand...",
            pos=(0, -0.23),
            scale=0.06,
            fg=(1.0, 0.8, 0.3, 1.0),
            align=TextNode.ACenter,
            mayChange=True,
        )
        self._calib_elements = [
            self._calib_overlay_bg, self._calib_title, self._calib_instruction,
            self._calib_progress_bg, self._calib_progress_fill,
            self._calib_count_text, self._calib_hand_status,
        ]
        self._calib_visible = False  # hidden until camera mode activates
        self._set_calib_visible(False)

    # ── Per-frame update ───────────────────────────────────────────────────

    _MODE_LABELS = {
        "demo":   "DEMO",
        "live":   "LIVE",
        "ble":    "BLE",
        "camera": "CAM",
    }

    def update(self, fog_score: float, is_fog: bool, progress: dict,
               live_mode: bool = False, mode: str = "demo"):
        """
        Update all HUD elements.

        progress: {"current": int, "total": int, "label": str}
        mode: one of "demo", "live", "ble", "camera"
        """
        self._update_score_bar(fog_score)
        self._update_progress(progress)
        self._update_status(is_fog)
        self._update_border_flash(is_fog)
        # Support old bool kwarg for backwards compatibility
        if live_mode and mode == "demo":
            mode = "live"
        self._mode_text.setText(self._MODE_LABELS.get(mode, mode.upper()))

    def _set_calib_visible(self, visible: bool):
        fn = "show" if visible else "hide"
        for el in self._calib_elements:
            getattr(el, fn)()
        self._calib_visible = visible

    def update_hand(self, landmarks, calib_done: bool = True,
                    calib_count: int = 0, calib_total: int = 60):
        """
        Drive the calibration overlay and hand skeleton.

        landmarks   : list of 21 (x,y) tuples in [0,1] normalised coords, or None.
        calib_done  : False = still calibrating, show full-screen overlay.
        calib_count : how many calibration frames collected so far.
        calib_total : total frames needed (CAMERA_CALIB_FRAMES).
        """
        # Remove previous skeleton node
        if self._hand_node is not None:
            self._hand_node.removeNode()
            self._hand_node = None

        if not calib_done:
            # Show calibration overlay
            if not self._calib_visible:
                self._set_calib_visible(True)

            # Progress bar
            frac = min(1.0, calib_count / max(1, calib_total))
            width = frac * 1.2   # bar spans -0.6 to +0.6 = 1.2 units total
            self._calib_progress_fill["frameSize"] = (-0.6, -0.6 + width, -0.03, 0.03)
            # Colour: orange -> green as it fills
            r = 1.0 - frac * 0.7
            g = 0.4 + frac * 0.5
            self._calib_progress_fill["frameColor"] = (r, g, 0.2, 1.0)

            self._calib_count_text.setText(f"{calib_count} / {calib_total}")

            if landmarks is None:
                self._calib_hand_status.setText("No hand detected — show hand to camera")
                self._calib_hand_status["fg"] = (1.0, 0.4, 0.3, 1.0)
            else:
                self._calib_hand_status.setText("Hand detected — hold still!")
                self._calib_hand_status["fg"] = (0.3, 1.0, 0.4, 1.0)
                # Draw skeleton in centre of overlay during calibration
                self._hand_node = self._draw_hand_skeleton(landmarks, centre=True)
            return

        # Calibration done — hide overlay
        if self._calib_visible:
            self._set_calib_visible(False)

        if landmarks is None:
            self._hand_label.setText("No hand")
            return

        self._hand_label.setText("Hand OK")
        self._hand_node = self._draw_hand_skeleton(landmarks)

    def _draw_hand_skeleton(self, landmarks: list, centre: bool = False) -> NodePath:
        """
        Render 21 hand landmarks as a skeleton.

        centre=False: small overlay in bottom-right corner (normal gameplay).
        centre=True:  large skeleton centred on screen (calibration overlay).
        """
        if centre:
            BOX_X0, BOX_X1 = -0.55, 0.55
            BOX_Y0, BOX_Y1 = -0.50, 0.16
        else:
            BOX_X0, BOX_X1 = 0.85, 1.58
            BOX_Y0, BOX_Y1 = -0.95, -0.55

        def to_screen(lm_x, lm_y):
            # MediaPipe: x=0 left, x=1 right; y=0 top, y=1 bottom
            sx = BOX_X0 + lm_x * (BOX_X1 - BOX_X0)
            sy = BOX_Y1 - lm_y * (BOX_Y1 - BOX_Y0)   # flip Y
            return sx, sy

        # MediaPipe hand connection pairs
        connections = [
            (0,1),(1,2),(2,3),(3,4),           # thumb
            (0,5),(5,6),(6,7),(7,8),            # index
            (0,9),(9,10),(10,11),(11,12),        # middle
            (0,13),(13,14),(14,15),(15,16),      # ring
            (0,17),(17,18),(18,19),(19,20),      # pinky
            (5,9),(9,13),(13,17),               # palm
        ]

        ls = LineSegs()
        ls.setThickness(1.5)
        ls.setColor(0.3, 0.85, 1.0, 0.85)

        for a, b in connections:
            ax, ay = to_screen(landmarks[a][0], landmarks[a][1])
            bx, by = to_screen(landmarks[b][0], landmarks[b][1])
            ls.moveTo(ax, 0, ay)
            ls.drawTo(bx, 0, by)

        # Dot at each landmark
        ls.setThickness(4.0)
        ls.setColor(1.0, 1.0, 0.3, 1.0)
        for lx, ly in landmarks:
            sx, sy = to_screen(lx, ly)
            ls.moveTo(sx - 0.005, 0, sy)
            ls.drawTo(sx + 0.005, 0, sy)

        node = self._base.aspect2d.attachNewNode(ls.create())
        return node

    # ── Cleanup ────────────────────────────────────────────────────────────

    def destroy(self):
        for elem in [
            self._score_label, self._bar_bg, self._bar_fill,
            self._score_value, self._progress_text, self._status_text,
            self._mode_text,
        ] + self._borders:
            elem.destroy()

    # ── Private helpers ────────────────────────────────────────────────────

    def _update_score_bar(self, score: float):
        width = max(0.0, min(1.0, score)) * 0.5
        self._bar_fill["frameSize"] = (0, width, -0.04, 0.04)

        # Colour: green (low) → yellow (mid) → red (high)
        r = min(1.0, score * 2.0)
        g = min(1.0, (1.0 - score) * 2.0)
        self._bar_fill["frameColor"] = (r, g, 0.1, 1.0)
        self._score_value.setText(f"{score:.2f}")

    def _update_progress(self, progress: dict):
        if not progress:
            return
        label   = progress.get("label", "")
        current = progress.get("current", 0)
        total   = progress.get("total", 1)
        self._progress_text.setText(f"{label}: {current}/{total}")

    def _update_status(self, is_fog: bool):
        if is_fog:
            self._status_text.setText("FoG Detected -- Cue Sent")
            self._status_text["fg"] = (1.0, 0.2, 0.2, 1.0)
        else:
            self._status_text.setText("Normal Gait")
            self._status_text["fg"] = (0.2, 1.0, 0.2, 1.0)

    def _update_border_flash(self, is_fog: bool):
        if not is_fog:
            alpha = 0.0
        else:
            # 2 Hz sine wave pulse
            t = time.monotonic() - self._start_time
            alpha = max(0.0, math.sin(2 * math.pi * 2.0 * t))
        color = (1.0, 0.0, 0.0, alpha * 0.7)
        for border in self._borders:
            border["frameColor"] = color
