"""
tasks/walking.py — Walking task: walk down a corridor and reach the door.

Scene: a 3D corridor viewed from a 3rd-person follow camera.
Goal:  take TARGET_STEPS steps to reach the door at the far end.

Step detection:
  A "step" is counted when the wrist_acc_y magnitude rises above
  WALK_STEP_THRESHOLD (0.3 g), with a minimum 0.4s between steps
  to avoid double-counting.

FoG behaviour:
  - Avatar stops advancing
  - Corridor flickers red (handled by HUD border flash)
  - Text overlay: "Frozen — cue sent"
  - Step counter pauses

Camera: follows avatar from behind at a fixed offset.
"""

import time
import math
import numpy as np
from panda3d.core import (
    NodePath, LColor, AmbientLight, DirectionalLight,
    CardMaker, LPoint3,
)
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from virtual_sim.tasks.base_task import BaseTask
from virtual_sim.config import WALK_TARGET_STEPS, WALK_STEP_THRESHOLD


class WalkingTask(BaseTask):
    """
    3D corridor scene. Patient walks (arm-swing) toward a door.
    """

    # Camera offset behind and above the avatar
    _CAM_OFFSET = (0, -8, 3)

    def __init__(self):
        self._steps = 0
        self._complete = False
        self._last_step_time = 0.0
        self._step_cooldown = 0.4   # seconds between steps

        # Exponential moving average of acc_y for baseline
        self._acc_y_ema = 0.0
        self._ema_alpha = 0.05

        self._door_node: NodePath | None = None
        self._door_open = False
        self._door_open_time = 0.0

        self._fog_label: OnscreenText | None = None

    # ── BaseTask interface ─────────────────────────────────────────────────

    @property
    def avatar_start_pos(self) -> tuple:
        return (0, 0, 0)

    def setup_scene(self, render: NodePath, loader):
        """Build corridor geometry: floor, two walls, ceiling, and a door."""
        self._render = render

        # Floor
        _make_box(render, w=4, d=0.2, l=60, pos=(0, 28, -0.1),
                  color=(0.55, 0.50, 0.45, 1))
        # Ceiling
        _make_box(render, w=4, d=0.2, l=60, pos=(0, 28, 3.1),
                  color=(0.9, 0.9, 0.9, 1))
        # Left wall
        _make_box(render, w=0.2, d=3, l=60, pos=(-2.1, 28, 1.5),
                  color=(0.7, 0.65, 0.6, 1))
        # Right wall
        _make_box(render, w=0.2, d=3, l=60, pos=(2.1, 28, 1.5),
                  color=(0.7, 0.65, 0.6, 1))

        # Door at the far end (Y = 57)
        self._door_node = render.attachNewNode("door_group")
        self._door_node.setPos(0, 57, 0)
        # Door frame
        _make_box(self._door_node, w=2.0, d=3.0, l=0.2, pos=(0, 0, 1.5),
                  color=(0.45, 0.28, 0.1, 1))
        # Door panel (brown, swings open)
        self._door_panel = self._door_node.attachNewNode("door_panel")
        _make_box(self._door_panel, w=1.8, d=2.8, l=0.1, pos=(0, 0, 1.4),
                  color=(0.55, 0.35, 0.15, 1))

        # Progress milestones (floor marks every 10 steps → every ~6 world units)
        step_dist = 57.0 / WALK_TARGET_STEPS
        for i in range(10, WALK_TARGET_STEPS, 10):
            y = i * step_dist
            _make_box(render, w=3.8, d=0.05, l=0.3, pos=(0, y, 0.01),
                      color=(0.9, 0.7, 0.2, 0.8))

    def setup_camera(self, camera: NodePath):
        ox, oy, oz = self._CAM_OFFSET
        camera.setPos(ox, oy, oz)
        camera.lookAt(LPoint3(0, 10, 1.5))

    def update_camera(self, camera: NodePath, avatar):
        """Follow avatar from behind."""
        ax, ay, az = avatar.position
        ox, oy, oz = self._CAM_OFFSET
        camera.setPos(ax + ox, ay + oy, az + oz)
        camera.lookAt(LPoint3(ax, ay + 6, az + 1.0))

    def update(self, frame_data: dict | None, fog_score: float, is_fog: bool):
        if is_fog:
            self._show_fog_label()
        else:
            self._hide_fog_label()

        if frame_data is None or is_fog:
            return

        acc_y = frame_data["acc"][1]

        # Update baseline EMA
        self._acc_y_ema = (
            self._ema_alpha * acc_y + (1.0 - self._ema_alpha) * self._acc_y_ema
        )
        deviation = abs(acc_y - self._acc_y_ema)

        # Step detection
        now = time.monotonic()
        if (deviation > WALK_STEP_THRESHOLD
                and now - self._last_step_time > self._step_cooldown):
            self._steps += 1
            self._last_step_time = now

            # Open door when goal reached
            if self._steps >= WALK_TARGET_STEPS and not self._door_open:
                self._door_open = True
                self._door_open_time = now
                self._complete = True

        # Animate door opening
        if self._door_open:
            elapsed = time.monotonic() - self._door_open_time
            angle = min(90.0, elapsed * 60)  # 1.5s to fully open
            if self._door_panel:
                self._door_panel.setH(angle)

    @property
    def progress(self) -> dict:
        return {"current": self._steps, "total": WALK_TARGET_STEPS, "label": "Steps"}

    @property
    def is_complete(self) -> bool:
        return self._complete

    # ── Private ────────────────────────────────────────────────────────────

    def _show_fog_label(self):
        if self._fog_label is None:
            self._fog_label = OnscreenText(
                text="Movement frozen -- rhythmic cue sent",
                pos=(0, 0.15),
                scale=0.07,
                fg=(1.0, 0.3, 0.3, 1.0),
                align=TextNode.ACenter,
            )

    def _hide_fog_label(self):
        if self._fog_label is not None:
            self._fog_label.destroy()
            self._fog_label = None


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _make_box(parent: NodePath, w: float, d: float, l: float,
              pos: tuple, color: tuple) -> NodePath:
    """
    Attach a textured box to parent at pos with given dimensions and colour.
    Uses Panda3D's built-in egg geometry from the models library.
    Falls back to a CardMaker quad if the box model is unavailable.
    """
    from panda3d.core import CardMaker
    cm = CardMaker("box_face")
    cm.setFrame(-w / 2, w / 2, -d / 2, d / 2)
    node = parent.attachNewNode(cm.generate())
    node.setPos(*pos)
    node.setScale(1, l, 1)
    node.setColor(*color)
    # Make double-sided
    from panda3d.core import CullFaceAttrib
    node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
    return node
