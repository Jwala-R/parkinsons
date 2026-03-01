"""
tasks/base_task.py — Abstract interface that all task scenes must implement.
"""

from abc import ABC, abstractmethod
from panda3d.core import NodePath


class BaseTask(ABC):
    """
    Abstract base class for therapy task scenes.

    Each task is responsible for:
      - Building its own 3D scene geometry (setup_scene)
      - Positioning the starting camera (setup_camera)
      - Optionally following the avatar each frame (update_camera)
      - Updating game-logic state based on sensor data + FoG state
      - Reporting progress for the HUD
    """

    @property
    @abstractmethod
    def avatar_start_pos(self) -> tuple:
        """World-space (x, y, z) where the avatar spawns."""

    @abstractmethod
    def setup_scene(self, render: NodePath, loader):
        """
        Build all scene geometry and attach it under `render`.
        Called once during SimApp.__init__().
        """

    @abstractmethod
    def setup_camera(self, camera: NodePath):
        """Set initial camera position/angle for this task."""

    def update_camera(self, camera: NodePath, avatar):
        """
        Called every frame to optionally follow the avatar.
        Default: do nothing (fixed camera).
        """

    @abstractmethod
    def update(self, frame_data: dict | None, fog_score: float, is_fog: bool):
        """
        Update task state.

        frame_data: {"acc": [x,y,z], "gyro": [x,y,z], "pressure": float} or None
        fog_score:  float in [0, 1]
        is_fog:     bool
        """

    @property
    @abstractmethod
    def progress(self) -> dict:
        """
        Returns HUD progress info:
            {"current": int, "total": int, "label": str}
        """

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        """True when the task goal has been reached."""
