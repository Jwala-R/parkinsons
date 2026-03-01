"""
tasks/eating.py — Eating task: practice scooping food and bringing it to the mouth.

Scene: kitchen table viewed from a seated perspective.
Goal:  complete TARGET_SCOOPS successful scoop -> feed cycles.

Motion mapping:
  acc_x (lateral wrist tilt) -> spoon cursor left/right
  acc_y (vertical wrist swing) -> spoon cursor up/down
  Exponential moving average smooths jitter noise.

Grip detection:
  If FSR pressure drops below EAT_PRESSURE_MIN the grip is "lost".

Scoop detection:
  Spoon cursor enters the bowl highlight region.
Feed detection:
  Spoon cursor reaches the mouth region within EAT_FEED_TIMEOUT seconds
  of a successful scoop.

FoG behaviour:
  Spoon jitters, table shakes, red overlay text appears.
  HapticController triple-pulse fires.
"""

import time
import math
import random

from panda3d.core import (
    NodePath, CardMaker, CullFaceAttrib, LPoint3,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode,
    TransparencyAttrib,
)
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectFrame import DirectFrame
from panda3d.core import TextNode

from virtual_sim.tasks.base_task import BaseTask
from virtual_sim.config import (
    EAT_TARGET_SCOOPS, EAT_PRESSURE_MIN, EAT_FEED_TIMEOUT,
)


class EatingTask(BaseTask):
    """
    Kitchen table scene. Patient moves wrist to guide a spoon
    from a bowl to a mouth target.
    """

    def __init__(self):
        self._scoops = 0
        self._complete = False
        self._has_scoop = False
        self._scoop_time = 0.0

        # Smoothed spoon position in world coordinates (table surface ~z=0.85)
        # Start above the bowl so it is immediately visible
        self._spoon_x = 0.0
        self._spoon_z = 1.15
        self._ema_alpha = 0.15

        self._spoon_node: NodePath | None = None
        self._table_node: NodePath | None = None
        self._bowl_highlight: NodePath | None = None
        self._mouth_highlight: NodePath | None = None

        self._fog_label: OnscreenText | None = None
        self._feedback_label: OnscreenText | None = None
        self._fog_shake_t = 0.0
        self._feedback_until = 0.0   # wall-clock time when label clears
        self._last_update = time.monotonic()

    # ── BaseTask interface ─────────────────────────────────────────────────

    @property
    def avatar_start_pos(self) -> tuple:
        return (-3, 5, 0)

    def setup_scene(self, render: NodePath, loader):
        self._render = render

        # ── Background / wall ──────────────────────────────────────────────
        # A warm off-white back wall
        wall = _flat_quad(render, w=8, h=4)
        wall.setPos(0, 5, 1.5)
        wall.setColor(0.92, 0.88, 0.82, 1)

        # Floor
        floor = _flat_quad(render, w=8, h=6)
        floor.setPos(0, 3, 0)
        floor.setP(-90)         # rotate to horizontal
        floor.setColor(0.60, 0.50, 0.40, 1)

        # ── Table ──────────────────────────────────────────────────────────
        self._table_node = render.attachNewNode("table")

        # Table top (solid box, 3 wide x 2 deep x 0.08 thick)
        top = _solid_box(self._table_node, 3.0, 2.0, 0.08)
        top.setPos(0, 3.0, 0.80)
        top.setColor(0.62, 0.38, 0.18, 1)

        # Table legs (4 x thin box)
        for lx, ly in [(-1.35, 2.1), (1.35, 2.1), (-1.35, 3.9), (1.35, 3.9)]:
            leg = _solid_box(self._table_node, 0.1, 0.1, 0.80)
            leg.setPos(lx, ly, 0.40)
            leg.setColor(0.45, 0.28, 0.12, 1)

        # ── Bowl ──────────────────────────────────────────────────────────
        # Ellipsoid approximation: a flat disc on the table
        bowl_rim = _solid_box(self._table_node, 0.50, 0.50, 0.06)
        bowl_rim.setPos(0, 3.0, 0.87)
        bowl_rim.setColor(0.90, 0.88, 0.80, 1)

        bowl_inside = _solid_box(self._table_node, 0.38, 0.38, 0.03)
        bowl_inside.setPos(0, 3.0, 0.90)
        bowl_inside.setColor(0.80, 0.60, 0.30, 1)  # food colour

        # Highlight ring: turns bright when spoon is near bowl
        self._bowl_highlight = _solid_box(render, 0.56, 0.56, 0.02)
        self._bowl_highlight.setPos(0, 3.0, 0.92)
        self._bowl_highlight.setColor(0.2, 0.9, 0.2, 0.0)
        self._bowl_highlight.setTransparency(TransparencyAttrib.MAlpha)

        # ── Mouth target ───────────────────────────────────────────────────
        # A plate/face stand-in on the right side of the table
        mouth_plate = _solid_box(render, 0.35, 0.35, 0.35)
        mouth_plate.setPos(1.4, 3.0, 1.45)
        mouth_plate.setColor(0.95, 0.75, 0.65, 1)

        mouth_lips = _solid_box(render, 0.22, 0.06, 0.06)
        mouth_lips.setPos(1.4, 2.95, 1.30)
        mouth_lips.setColor(0.85, 0.40, 0.40, 1)

        # Highlight: turns bright when spoon reaches mouth
        self._mouth_highlight = _solid_box(render, 0.42, 0.42, 0.42)
        self._mouth_highlight.setPos(1.4, 3.0, 1.45)
        self._mouth_highlight.setColor(0.2, 0.5, 1.0, 0.0)
        self._mouth_highlight.setTransparency(TransparencyAttrib.MAlpha)

        # Interaction regions (world-space X, Z bounds — Y is fixed along table)
        self._bowl_region  = {"x": (-0.28, 0.28),  "z": (0.84, 1.10)}
        self._mouth_region = {"x": (1.08, 1.72),   "z": (1.18, 1.72)}

        # ── Spoon ──────────────────────────────────────────────────────────
        # A clearly visible elongated box, starts above the bowl
        self._spoon_node = render.attachNewNode("spoon")
        spoon_handle = _solid_box(self._spoon_node, 0.06, 0.06, 0.30)
        spoon_handle.setPos(0, 0, 0)
        spoon_handle.setColor(0.80, 0.72, 0.22, 1)

        spoon_head = _solid_box(self._spoon_node, 0.12, 0.12, 0.06)
        spoon_head.setPos(0, 0, 0.18)
        spoon_head.setColor(0.85, 0.78, 0.30, 1)

        self._spoon_node.setPos(0, 3.05, 1.15)

        # Label over the bowl and mouth so the patient knows what they are
        self._bowl_label = OnscreenText(
            text="BOWL", pos=(-0.45, -0.10), scale=0.055,
            fg=(0.9, 0.8, 0.3, 1), align=TextNode.ACenter,
            mayChange=False,
        )
        self._mouth_label = OnscreenText(
            text="MOUTH", pos=(0.80, 0.22), scale=0.055,
            fg=(0.9, 0.6, 0.5, 1), align=TextNode.ACenter,
            mayChange=False,
        )

        # Scoop feedback label (bottom-left, shows briefly on success)
        self._feedback_label = OnscreenText(
            text="", pos=(-1.3, -0.72), scale=0.065,
            fg=(0.3, 1.0, 0.4, 1.0), align=TextNode.ALeft,
            mayChange=True,
        )

    def setup_camera(self, camera: NodePath):
        """
        Seated perspective: slightly elevated and back, looking at the table
        at a comfortable angle that shows the bowl, mouth, and spoon clearly.
        """
        camera.setPos(0, -1.0, 2.2)
        camera.lookAt(LPoint3(0.4, 3.5, 0.9))

    def update_camera(self, camera: NodePath, avatar):
        pass  # Fixed camera for eating task

    def update(self, frame_data: dict | None, fog_score: float, is_fog: bool):
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now

        # ── FoG visual effects ─────────────────────────────────────────────
        if is_fog:
            self._show_fog_label()
            self._fog_shake_t += dt
            if self._table_node:
                self._table_node.setX(math.sin(self._fog_shake_t * 20) * 0.018)
            self._spoon_x += random.uniform(-0.04, 0.04)
            self._spoon_z += random.uniform(-0.025, 0.025)
        else:
            self._hide_fog_label()
            self._fog_shake_t = 0.0
            if self._table_node:
                self._table_node.setX(0)

        # ── Feedback label timer ───────────────────────────────────────────
        if self._feedback_label and now >= self._feedback_until and self._feedback_until > 0:
            self._feedback_label.setText("")
            self._feedback_until = 0.0

        # ── Sensor-driven spoon movement ───────────────────────────────────
        if frame_data is not None:
            # Grip check
            pressure = frame_data.get("pressure", 100.0)
            if self._has_scoop and pressure < EAT_PRESSURE_MIN:
                self._has_scoop = False
                self._feedback_label.setText("Grip lost!")
                self._feedback_until = now + 1.5

            if not is_fog:
                acc_x = frame_data["acc"][0]   # lateral tilt  -> left/right
                acc_y = frame_data["acc"][1]   # vertical swing -> up/down

                # Scale sensor range to comfortable table-space range
                target_x = acc_x * 0.35         # ±9 m/s^2 -> ±3.15 world units
                target_z = 1.05 + acc_y * 0.08  # centred above table

                self._spoon_x = (
                    self._ema_alpha * target_x
                    + (1 - self._ema_alpha) * self._spoon_x
                )
                self._spoon_z = (
                    self._ema_alpha * target_z
                    + (1 - self._ema_alpha) * self._spoon_z
                )

        self._update_spoon_node()
        self._update_highlights()
        if frame_data is not None:
            self._check_interactions(now)

        # Scoop timeout
        if self._has_scoop and now - self._scoop_time > EAT_FEED_TIMEOUT:
            self._has_scoop = False
            self._feedback_label.setText("Too slow! Try again.")
            self._feedback_until = now + 1.5

    @property
    def progress(self) -> dict:
        return {
            "current": self._scoops,
            "total": EAT_TARGET_SCOOPS,
            "label": "Spoonfuls",
        }

    @property
    def is_complete(self) -> bool:
        return self._complete

    # ── Private ────────────────────────────────────────────────────────────

    def _update_spoon_node(self):
        if not self._spoon_node:
            return
        sx = max(-1.5, min(1.5, self._spoon_x))
        sz = max(0.90, min(2.0, self._spoon_z))
        self._spoon_node.setPos(sx, 3.05, sz)

        # Spoon colour: gold when carrying food, silver otherwise
        loaded = (0.95, 0.78, 0.25, 1)
        empty  = (0.85, 0.85, 0.90, 1)
        self._spoon_node.setColor(*(loaded if self._has_scoop else empty))

    def _update_highlights(self):
        sx, sz = self._spoon_x, self._spoon_z

        # Bowl highlight: glow green when spoon is near bowl and no scoop yet
        near_bowl = (
            self._bowl_region["x"][0] <= sx <= self._bowl_region["x"][1]
            and self._bowl_region["z"][0] <= sz <= self._bowl_region["z"][1]
        )
        if self._bowl_highlight:
            a = 0.55 if (near_bowl and not self._has_scoop) else 0.0
            self._bowl_highlight.setColor(0.2, 0.9, 0.2, a)

        # Mouth highlight: glow blue when spoon (with food) is near mouth
        near_mouth = (
            self._mouth_region["x"][0] <= sx <= self._mouth_region["x"][1]
            and self._mouth_region["z"][0] <= sz <= self._mouth_region["z"][1]
        )
        if self._mouth_highlight:
            a = 0.45 if (near_mouth and self._has_scoop) else 0.0
            self._mouth_highlight.setColor(0.2, 0.5, 1.0, a)

    def _check_interactions(self, now: float):
        sx, sz = self._spoon_x, self._spoon_z

        if not self._has_scoop:
            br = self._bowl_region
            if br["x"][0] <= sx <= br["x"][1] and br["z"][0] <= sz <= br["z"][1]:
                self._has_scoop = True
                self._scoop_time = now
                self._feedback_label.setText("Scooped! Bring to mouth.")
                self._feedback_until = now + 2.0
        else:
            mr = self._mouth_region
            if mr["x"][0] <= sx <= mr["x"][1] and mr["z"][0] <= sz <= mr["z"][1]:
                self._scoops += 1
                self._has_scoop = False
                self._feedback_label.setText(
                    f"Spoonful {self._scoops}/{EAT_TARGET_SCOOPS}!"
                )
                self._feedback_until = now + 1.8
                if self._scoops >= EAT_TARGET_SCOOPS:
                    self._complete = True

    def _show_fog_label(self):
        if self._fog_label is None:
            self._fog_label = OnscreenText(
                text="Movement frozen -- rhythmic cue sent",
                pos=(0, 0.15),
                scale=0.07,
                fg=(1.0, 0.3, 0.3, 1.0),
                shadow=(0, 0, 0, 0.6),
                shadowOffset=(0.04, 0.04),
                align=TextNode.ACenter,
            )

    def _hide_fog_label(self):
        if self._fog_label is not None:
            self._fog_label.destroy()
            self._fog_label = None


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _solid_box(parent: NodePath, w: float, d: float, h: float) -> NodePath:
    """
    Build a textured solid box (6 faces) as a GeomNode and attach it to parent.
    w = width (X), d = depth (Y), h = height (Z).
    The box is centred at the origin of the returned NodePath.
    """
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("box", fmt, Geom.UHStatic)
    vdata.setNumRows(24)

    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")

    hw, hd, hh = w / 2, d / 2, h / 2

    # Each face: 4 vertices, normal, white colour (tinted via setColor later)
    faces = [
        # (4 vertices, normal)
        # Front  (+Y)
        ([(-hw, hd, -hh), (hw, hd, -hh), (hw, hd, hh), (-hw, hd, hh)], (0, 1, 0)),
        # Back   (-Y)
        ([(hw, -hd, -hh), (-hw, -hd, -hh), (-hw, -hd, hh), (hw, -hd, hh)], (0, -1, 0)),
        # Right  (+X)
        ([(hw, -hd, -hh), (hw, hd, -hh), (hw, hd, hh), (hw, -hd, hh)],  (1, 0, 0)),
        # Left   (-X)
        ([(-hw, hd, -hh), (-hw, -hd, -hh), (-hw, -hd, hh), (-hw, hd, hh)], (-1, 0, 0)),
        # Top    (+Z)
        ([(-hw, -hd, hh), (hw, -hd, hh), (hw, hd, hh), (-hw, hd, hh)],  (0, 0, 1)),
        # Bottom (-Z)
        ([(-hw, hd, -hh), (hw, hd, -hh), (hw, -hd, -hh), (-hw, -hd, -hh)], (0, 0, -1)),
    ]

    for verts, normal in faces:
        for v in verts:
            vw.addData3(*v)
            nw.addData3(*normal)
            cw.addData4(1, 1, 1, 1)

    tris = GeomTriangles(Geom.UHStatic)
    for i in range(6):
        base = i * 4
        tris.addVertices(base, base + 1, base + 2)
        tris.addVertices(base, base + 2, base + 3)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("box")
    node.addGeom(geom)
    return parent.attachNewNode(node)


def _flat_quad(parent: NodePath, w: float, h: float) -> NodePath:
    """A single flat quad (CardMaker) for walls and floor."""
    cm = CardMaker("quad")
    cm.setFrame(-w / 2, w / 2, -h / 2, h / 2)
    node = parent.attachNewNode(cm.generate())
    node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
    return node
