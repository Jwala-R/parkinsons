"""
tasks/whackamole.py -- Whack-a-Mole therapy game for wrist/watch sensors.

Scene: top-down game board with a 3x3 grid of mole holes.
Goal:  whack WHACK_TARGET_HITS moles before time runs out.

Wrist/watch motion mapping:
  acc_x (lateral tilt)   -> cursor left/right across the grid
  acc_y (vertical swing) -> cursor up/down across the grid
  Exponential moving average smooths jitter.

Hit detection:
  FSR pressure sensor sends a pulse (pressure >= WHACK_PRESSURE_MIN)
  while the cursor is over an active mole -> score.

FoG behaviour:
  During a FoG episode the cursor jitters and moles stop popping.
  Haptic triple-pulse fires.  A red overlay warns the patient.

Mole timing:
  Active moles pop up for MOLE_VISIBLE_S seconds, then duck.
  A new mole appears after MOLE_INTERVAL_S.
"""

import time
import math
import random  # used for mole selection

from panda3d.core import (
    NodePath, LPoint3,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode,
    TransparencyAttrib, CardMaker, CullFaceAttrib,
)
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from virtual_sim.tasks.base_task import BaseTask
from virtual_sim.config import (
    WHACK_TARGET_HITS,
    WHACK_PRESSURE_MIN,
    MOLE_VISIBLE_S,
    MOLE_INTERVAL_S,
)


# Grid layout: 3x3, world-space positions (x, z) on the board face
_GRID_COLS = 3
_GRID_ROWS = 3
_CELL_SIZE = 0.80        # spacing between mole centres
_BOARD_Y   = 2.0         # how far into the scene the board sits


def _grid_positions():
    """Return list of (world_x, world_z) for each of the 9 mole holes."""
    positions = []
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            x = (col - 1) * _CELL_SIZE        # -0.8, 0, +0.8
            z = (1 - row) * _CELL_SIZE         # +0.8, 0, -0.8
            positions.append((x, z))
    return positions


class WhackAMoleTask(BaseTask):
    """
    Top-down whack-a-mole board.  Patient tilts wrist to aim the mallet
    cursor and squeezes the watch FSR sensor to whack the mole.
    """

    def __init__(self):
        self._hits       = 0
        self._complete   = False

        # Cursor position (world x, z on the board plane)
        self._cur_x = 0.0
        self._cur_z = 0.0
        self._ema_alpha = 0.18

        # Mole state: index in 0-8, None = no active mole
        self._active_mole: int | None = None
        self._mole_pop_time  = 0.0   # when this mole appeared
        self._next_mole_time = 0.0   # wall-clock when next mole pops

        # Hit debounce: ignore pressure while debounce active
        self._last_hit_time = 0.0
        self._HIT_DEBOUNCE  = 0.35   # seconds

        # Previous pressure sample (edge detection)
        self._prev_pressure = 0.0

        # Feedback label (brief "HIT!" or "Miss!" display)
        self._feedback_label: OnscreenText | None = None
        self._feedback_until = 0.0

        # Scene nodes
        self._hole_nodes: list[NodePath] = []   # 9 hole caps
        self._mole_nodes: list[NodePath] = []   # 9 mole heads
        self._cursor_node: NodePath | None = None
        self._board_node:  NodePath | None = None

        self._grid_pos = _grid_positions()       # list of (x, z)
        self._last_update = time.monotonic()

    # ── BaseTask interface ─────────────────────────────────────────────────

    @property
    def avatar_start_pos(self) -> tuple:
        return (0, -2, 0)

    def setup_scene(self, render: NodePath, loader):
        self._render = render

        # ── Background ────────────────────────────────────────────────────
        bg = _flat_quad(render, w=7, h=5)
        bg.setPos(0, _BOARD_Y + 0.5, 0.5)
        bg.setColor(0.18, 0.22, 0.30, 1)   # dark arcade blue

        floor = _flat_quad(render, w=7, h=4)
        floor.setPos(0, _BOARD_Y - 1, -0.5)
        floor.setP(-90)
        floor.setColor(0.12, 0.14, 0.18, 1)

        # ── Game board (green felt surface) ───────────────────────────────
        self._board_node = render.attachNewNode("board")
        board_top = _solid_box(self._board_node, 3.0, 0.12, 3.0)
        board_top.setPos(0, _BOARD_Y, 0)
        board_top.setColor(0.15, 0.55, 0.20, 1)   # felt green

        # Board rim
        rim = _solid_box(self._board_node, 3.20, 0.16, 3.20)
        rim.setPos(0, _BOARD_Y - 0.02, 0)
        rim.setColor(0.40, 0.22, 0.08, 1)          # wood brown

        # Score banner background
        banner = _flat_quad(render, w=3.0, h=0.45)
        banner.setPos(0, _BOARD_Y - 0.05, 1.78)
        banner.setColor(0.08, 0.08, 0.10, 0.85)
        banner.setTransparency(TransparencyAttrib.MAlpha)

        # ── Mole holes and mole heads ─────────────────────────────────────
        for i, (gx, gz) in enumerate(self._grid_pos):
            world_x = gx
            world_y = _BOARD_Y + 0.065    # just above board surface
            world_z = gz

            # Hole (dark disc sunken into board)
            hole = _solid_box(self._board_node, 0.42, 0.10, 0.42)
            hole.setPos(world_x, world_y - 0.03, world_z)
            hole.setColor(0.06, 0.18, 0.06, 1)
            self._hole_nodes.append(hole)

            # Mole head (brown sphere approximation — a box for now)
            mole = render.attachNewNode(f"mole_{i}")
            mole_body = _solid_box(mole, 0.30, 0.30, 0.30)
            mole_body.setPos(0, 0, 0)
            mole_body.setColor(0.55, 0.35, 0.12, 1)   # brown

            # Mole eyes (two small white boxes)
            for ex in (-0.07, 0.07):
                eye = _solid_box(mole, 0.07, 0.07, 0.07)
                eye.setPos(ex, -0.16, 0.06)
                eye.setColor(1.0, 1.0, 1.0, 1)
                pupil = _solid_box(mole, 0.04, 0.04, 0.04)
                pupil.setPos(ex, -0.19, 0.06)
                pupil.setColor(0.1, 0.05, 0.02, 1)

            # Mole nose
            nose = _solid_box(mole, 0.08, 0.08, 0.05)
            nose.setPos(0, -0.17, -0.04)
            nose.setColor(0.80, 0.40, 0.40, 1)

            # Start underground (hidden below board)
            mole.setPos(world_x, world_y, world_z - 0.80)
            self._mole_nodes.append(mole)

        # ── Cursor / mallet ───────────────────────────────────────────────
        self._cursor_node = render.attachNewNode("cursor")

        # Mallet head
        head = _solid_box(self._cursor_node, 0.22, 0.22, 0.18)
        head.setPos(0, 0, 0)
        head.setColor(0.85, 0.70, 0.15, 1)   # gold mallet

        # Mallet handle
        handle = _solid_box(self._cursor_node, 0.06, 0.06, 0.38)
        handle.setPos(0, 0, 0.28)
        handle.setColor(0.60, 0.38, 0.14, 1)

        # Cursor highlight ring (semi-transparent, shows which hole is targeted)
        self._cursor_ring = _solid_box(render, 0.52, 0.04, 0.52)
        self._cursor_ring.setPos(0, _BOARD_Y + 0.08, 0)
        self._cursor_ring.setColor(1.0, 1.0, 0.3, 0.6)
        self._cursor_ring.setTransparency(TransparencyAttrib.MAlpha)

        self._cursor_node.setPos(0, _BOARD_Y + 0.35, 0)

        # ── HUD labels ────────────────────────────────────────────────────
        # Feedback (brief WHACK!/Miss!) — centre screen, clear of HUD status bar
        self._feedback_label = OnscreenText(
            text="", pos=(0, -0.68), scale=0.085,
            fg=(0.3, 1.0, 0.4, 1.0), align=TextNode.ACenter,
            mayChange=True,
        )

        # Instruction line sits just above the HUD status bar row
        self._aim_label = OnscreenText(
            text="Tilt wrist to aim  |  Squeeze to whack!",
            pos=(0, -0.78), scale=0.046,
            fg=(0.75, 0.85, 1.0, 1.0), align=TextNode.ACenter,
            mayChange=False,
        )

        # Schedule first mole appearance
        self._next_mole_time = time.monotonic() + 1.0

    def setup_camera(self, camera: NodePath):
        """
        Top-front perspective: high and slightly back, looking down at the board.
        """
        camera.setPos(0, -1.8, 3.8)
        camera.lookAt(LPoint3(0, _BOARD_Y, 0))

    def update_camera(self, camera: NodePath, avatar):
        pass   # fixed camera

    def update(self, frame_data: dict | None, fog_score: float, is_fog: bool):
        now = time.monotonic()
        dt  = now - self._last_update
        self._last_update = now

        # ── Feedback label timer ──────────────────────────────────────────
        if self._feedback_label and self._feedback_until > 0 and now >= self._feedback_until:
            self._feedback_label.setText("")
            self._feedback_until = 0.0

        # ── Wrist motion -> cursor ────────────────────────────────────────
        if frame_data is not None:
            acc_x = frame_data["acc"][0]
            acc_y = frame_data["acc"][1]

            # Map sensor range to grid range (±_CELL_SIZE * 1.1 world units)
            target_x = acc_x * 0.30
            target_z = acc_y * 0.30

            self._cur_x = self._ema_alpha * target_x + (1 - self._ema_alpha) * self._cur_x
            self._cur_z = self._ema_alpha * target_z + (1 - self._ema_alpha) * self._cur_z

        # ── FSR pressure -> whack ─────────────────────────────────────────
        if frame_data is not None:
            pressure = frame_data.get("pressure", 0.0)
            # Rising edge: pressure just crossed threshold
            hit_edge = (pressure >= WHACK_PRESSURE_MIN
                        and self._prev_pressure < WHACK_PRESSURE_MIN)
            self._prev_pressure = pressure

            if hit_edge:
                self._try_whack(now)

        # ── Mole lifecycle ────────────────────────────────────────────────
        # Retire mole if it has been up too long
        if self._active_mole is not None:
            if now - self._mole_pop_time > MOLE_VISIBLE_S:
                self._duck_mole(self._active_mole)
                self._active_mole = None
                self._next_mole_time = now + MOLE_INTERVAL_S * 0.5

        # Pop a new mole
        if self._active_mole is None and now >= self._next_mole_time:
            self._pop_random_mole(now)

        # ── Update positions ──────────────────────────────────────────────
        self._update_cursor()
        self._update_mole_animations(dt)

    @property
    def progress(self) -> dict:
        return {
            "current": self._hits,
            "total":   WHACK_TARGET_HITS,
            "label":   "Whacks",
        }

    @property
    def is_complete(self) -> bool:
        return self._complete

    # ── Private helpers ────────────────────────────────────────────────────

    def _try_whack(self, now: float):
        """Check if cursor is over an active mole and score if so."""
        if now - self._last_hit_time < self._HIT_DEBOUNCE:
            return

        self._last_hit_time = now

        if self._active_mole is None:
            # Swung at air
            self._feedback_label.setText("Miss!")
            self._feedback_label.setFg((1.0, 0.5, 0.2, 1.0))
            self._feedback_until = now + 0.8
            return

        # Check cursor proximity to active mole
        mx, mz = self._grid_pos[self._active_mole]
        dist = math.hypot(self._cur_x - mx, self._cur_z - mz)

        if dist <= _CELL_SIZE * 0.52:
            # HIT
            self._hits += 1
            self._duck_mole(self._active_mole)
            self._active_mole = None
            self._next_mole_time = now + MOLE_INTERVAL_S

            self._feedback_label.setText("WHACK!" if self._hits % 3 != 0 else "COMBO!")
            self._feedback_label.setFg((0.3, 1.0, 0.4, 1.0))
            self._feedback_until = now + 0.9

            if self._hits >= WHACK_TARGET_HITS:
                self._complete = True
                self._feedback_label.setText("All done! Great job!")
                self._feedback_label.setFg((1.0, 0.95, 0.3, 1.0))
                self._feedback_until = now + 5.0
        else:
            # Near miss
            self._feedback_label.setText("Miss!")
            self._feedback_label.setFg((1.0, 0.5, 0.2, 1.0))
            self._feedback_until = now + 0.8

    def _pop_random_mole(self, now: float):
        """Choose a random hole and start popping the mole."""
        idx = random.randrange(len(self._mole_nodes))
        self._active_mole    = idx
        self._mole_pop_time  = now
        # Moles animate upward in _update_mole_animations

    def _duck_mole(self, idx: int):
        """Immediately hide the mole back underground."""
        gx, gz = self._grid_pos[idx]
        world_y = _BOARD_Y + 0.065
        self._mole_nodes[idx].setPos(gx, world_y, gz - 0.80)

    def _update_cursor(self):
        # Clamp cursor to board extent
        cx = max(-_CELL_SIZE * 1.3, min(_CELL_SIZE * 1.3, self._cur_x))
        cz = max(-_CELL_SIZE * 1.3, min(_CELL_SIZE * 1.3, self._cur_z))

        if self._cursor_node:
            self._cursor_node.setPos(cx, _BOARD_Y + 0.35, cz)

        # Snap cursor ring to nearest grid cell
        best_i, best_d = 0, float("inf")
        for i, (gx, gz) in enumerate(self._grid_pos):
            d = math.hypot(cx - gx, cz - gz)
            if d < best_d:
                best_d, best_i = d, i

        if self._cursor_ring:
            gx, gz = self._grid_pos[best_i]
            self._cursor_ring.setPos(gx, _BOARD_Y + 0.085, gz)
            # Ring glows bright yellow when over active mole
            if self._active_mole == best_i:
                self._cursor_ring.setColor(1.0, 0.3, 0.3, 0.75)
            else:
                self._cursor_ring.setColor(1.0, 1.0, 0.3, 0.45)

    def _update_mole_animations(self, dt: float):
        """Smoothly pop up the active mole; rest stay underground."""
        board_y = _BOARD_Y + 0.065

        for i, mole in enumerate(self._mole_nodes):
            gx, gz = self._grid_pos[i]
            cur_z = mole.getZ()

            if i == self._active_mole:
                # Target: fully popped up
                target_z = gz + 0.05
                new_z = cur_z + (target_z - cur_z) * min(1.0, dt * 8.0)
            else:
                # Target: underground
                target_z = gz - 0.80
                new_z = cur_z + (target_z - cur_z) * min(1.0, dt * 12.0)

            mole.setPos(gx, board_y, new_z)


# ── Geometry helpers (shared with eating.py) ───────────────────────────────────

def _solid_box(parent: NodePath, w: float, d: float, h: float) -> NodePath:
    """
    Build a solid coloured box (6 faces) attached to parent.
    w = width (X), d = depth (Y), h = height (Z). Centred at origin.
    """
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("box", fmt, Geom.UHStatic)
    vdata.setNumRows(24)

    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")

    hw, hd, hh = w / 2, d / 2, h / 2

    faces = [
        ([(-hw, hd, -hh), ( hw, hd, -hh), ( hw, hd,  hh), (-hw, hd,  hh)],  (0,  1, 0)),
        ([( hw,-hd, -hh), (-hw,-hd, -hh), (-hw,-hd,  hh), ( hw,-hd,  hh)],  (0, -1, 0)),
        ([( hw,-hd, -hh), ( hw, hd, -hh), ( hw, hd,  hh), ( hw,-hd,  hh)],  (1,  0, 0)),
        ([(-hw, hd, -hh), (-hw,-hd, -hh), (-hw,-hd,  hh), (-hw, hd,  hh)],  (-1, 0, 0)),
        ([(-hw,-hd,  hh), ( hw,-hd,  hh), ( hw, hd,  hh), (-hw, hd,  hh)],  (0,  0, 1)),
        ([(-hw, hd, -hh), ( hw, hd, -hh), ( hw,-hd, -hh), (-hw,-hd, -hh)],  (0,  0,-1)),
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
    """A single flat quad (CardMaker) for background surfaces."""
    cm = CardMaker("quad")
    cm.setFrame(-w / 2, w / 2, -h / 2, h / 2)
    node = parent.attachNewNode(cm.generate())
    node.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
    return node
