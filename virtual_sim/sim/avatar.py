"""
sim/avatar.py — Avatar controller for the Panda3D 3D scene.

The avatar is a simple humanoid NodePath built from Panda3D primitives
(no external .egg model required). It has a 3-state machine:

  WALKING   → figure moves forward, bob animation driven by wrist acc_y
  FROZEN    → figure halts, rapid micro-jitter overlay, red colour tint
  RECOVERING → figure resumes slowly for 1.5 seconds before returning to WALKING

Motion mapping from sensor data:
  wrist_acc_y (ch 19, vertical)  → forward velocity + vertical body bob
  wrist_acc_x (ch 18, lateral)   → side lean (NodePath.setR rotation ±5°)
"""

import math
import time
from panda3d.core import NodePath, LColor
from direct.actor.Actor import Actor


_WALKING    = "walking"
_FROZEN     = "frozen"
_RECOVERING = "recovering"

_RECOVER_DURATION = 1.5   # seconds
_FORWARD_SCALE    = 0.12  # world units per g of acc_y
_LEAN_SCALE       = 5.0   # degrees per g of acc_x
_BOB_SCALE        = 0.04  # world units of vertical bob per g


class Avatar:
    """
    Simple avatar built from a Panda3D NodePath box (placeholder geometry).
    Replace _build_geometry() with Actor.loadModel() to use a real .egg model.
    """

    def __init__(self, render: NodePath, start_pos=(0, 10, 0)):
        self._render = render
        self._state = _WALKING
        self._recover_timer = 0.0
        self._walk_phase = 0.0
        self._last_update = time.monotonic()

        self._root = render.attachNewNode("avatar_root")
        self._root.setPos(*start_pos)
        self._build_geometry()

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, frame_data: dict | None, fog_score: float):
        """
        Update avatar state and position.

        frame_data: {"acc": [x,y,z], "gyro": [x,y,z], ...} or None
        fog_score:  float in [0, 1]
        """
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now

        self._update_state(fog_score, dt)

        if frame_data is None:
            return

        acc_x = frame_data["acc"][0]
        acc_y = frame_data["acc"][1]

        if self._state == _WALKING:
            self._walk_phase += dt * 2.0  # ~2 Hz animation cycle
            # Advance forward based on arm-swing magnitude
            forward = abs(acc_y) * _FORWARD_SCALE * dt * 30
            self._root.setY(self._root.getY() + forward)
            # Vertical bob
            bob = math.sin(self._walk_phase * math.pi) * _BOB_SCALE
            self._root.setZ(self._root.getZ() + bob * 0.1)
            # Lateral lean
            lean = acc_x * _LEAN_SCALE
            self._root.setR(lean)
            self._body.setColorScale(1, 1, 1, 1)

        elif self._state == _FROZEN:
            # Micro-jitter to convey tremor
            import random
            jitter = 0.01
            self._root.setX(self._root.getX() + random.uniform(-jitter, jitter))
            # Red tint
            t_pulse = (math.sin(time.monotonic() * 10) + 1) * 0.5
            self._body.setColorScale(1.0, 0.3 + 0.2 * t_pulse, 0.3, 1.0)

        elif self._state == _RECOVERING:
            # Gradually resume walking animation
            self._walk_phase += dt * 0.5
            self._body.setColorScale(1.0, 0.7, 0.7, 1.0)

    @property
    def position(self):
        return self._root.getPos()

    @property
    def state(self) -> str:
        return self._state

    def destroy(self):
        self._root.removeNode()

    # ── Private ────────────────────────────────────────────────────────────

    def _update_state(self, fog_score: float, dt: float):
        from virtual_sim.config import FOG_THRESHOLD
        if fog_score >= FOG_THRESHOLD:
            self._state = _FROZEN
            self._recover_timer = 0.0
        elif self._state == _FROZEN and fog_score < FOG_THRESHOLD * 0.7:
            self._state = _RECOVERING
            self._recover_timer = _RECOVER_DURATION
        elif self._state == _RECOVERING:
            self._recover_timer -= dt
            if self._recover_timer <= 0:
                self._state = _WALKING

    def _build_geometry(self):
        """
        Build a placeholder avatar from Panda3D boxes.
        Structure: torso (tall box) + head (small box) on top.
        Replace with Actor.loadModel('avatar.egg') for a real model.
        """
        from panda3d.core import GeomNode

        # Torso
        self._body = self._root.attachNewNode("body")
        _attach_box(self._body, w=0.3, h=0.1, d=0.6, color=(0.4, 0.6, 1.0, 1.0))
        self._body.setZ(0.9)

        # Head
        head = self._body.attachNewNode("head")
        _attach_box(head, w=0.2, h=0.2, d=0.2, color=(0.9, 0.75, 0.65, 1.0))
        head.setZ(0.4)


def _attach_box(parent: NodePath, w: float, h: float, d: float, color):
    """Attach a simple coloured box to a NodePath using Panda3D's CardMaker / GeomNode."""
    from panda3d.core import CardMaker, LPoint3
    # Use a simple card as a stand-in — a proper box would use GeomVertexData
    # but this avoids needing external assets.
    cm = CardMaker("box")
    cm.setFrame(-w / 2, w / 2, -d / 2, d / 2)
    card = parent.attachNewNode(cm.generate())
    card.setColor(*color)
    return card
