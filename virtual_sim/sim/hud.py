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
from panda3d.core import TextNode, LColor


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

    # ── Per-frame update ───────────────────────────────────────────────────

    def update(self, fog_score: float, is_fog: bool, progress: dict, live_mode: bool = False):
        """
        Update all HUD elements.

        progress: {"current": int, "total": int, "label": str}
        """
        self._update_score_bar(fog_score)
        self._update_progress(progress)
        self._update_status(is_fog)
        self._update_border_flash(is_fog)
        self._mode_text.setText("LIVE" if live_mode else "DEMO")

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
