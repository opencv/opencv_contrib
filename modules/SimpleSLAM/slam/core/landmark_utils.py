from __future__ import annotations

"""
landmark_utils.py
~~~~~~~~~~~~~~~~~
Classes and helper functions for managing 3‑D landmarks and camera poses
in an incremental VO / SLAM pipeline.

* MapPoint ─ encapsulates a single 3‑D landmark.
* Map      ─ container for all landmarks + camera trajectory.
* triangulate_points ─ convenience wrapper around OpenCV triangulation.

The module is intentionally lightweight and free of external dependencies
beyond NumPy + OpenCV; this makes it easy to unit‑test without a heavy
visualisation stack.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from scipy.spatial import cKDTree

# datatype test for feature descriptors
def _canon_desc(desc):
    try:
        import torch
        if isinstance(desc, torch.Tensor):
            desc = desc.detach().to("cpu")
            if desc.dtype != torch.uint8:
                desc = desc.float()
            desc = desc.contiguous().numpy()
    except Exception:
        pass
    d = np.asarray(desc)
    if d.dtype == np.uint8:                 # ORB/AKAZE (binary)
        return d.reshape(-1)
    d = d.astype(np.float32, copy=False)    # ALIKE/LightGlue (float)
    n = np.linalg.norm(d) + 1e-8
    return (d / n).reshape(-1)

# --------------------------------------------------------------------------- #
#  MapPoint
# --------------------------------------------------------------------------- #
@dataclass
class MapPoint:
    """A single triangulated 3‑D landmark.

    Parameters
    ----------
    id
        Unique landmark identifier.
    position
        3‑D position in world coordinates (shape ``(3,)``).
    keyframe_idx
        Index of the keyframe that first observed / created this landmark.
    colour
        RGB colour (linear, 0‑1) associated with the point (shape ``(3,)``).
    observations
        List of observations in the form ``(frame_idx, kp_idx, descriptor)`` where
        * ``keyframe_idx`` – Keyframe - idx where the keypoint was detected.
        * ``kp_idx``    – index of the keypoint inside that frame.
        * ``descriptor`` – feature descriptor as a 1‑D ``np.ndarray``.
    """
    id: int
    position: np.ndarray  # shape (3,)
    keyframe_idx: int = -1
    colour: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))    # (3,) in **linear** RGB 0-1
    observations: List[Tuple[int, int, np.ndarray]] = field(default_factory=list)  # (keyframe_idx, kp_idx, descriptor)

    def add_observation(self, keyframe_idx: int, kp_idx: int, descriptor: np.ndarray) -> None:
        """Register that *kp_idx* in *keyframe_idx* observes this landmark."""
        self.observations.append((keyframe_idx, kp_idx, _canon_desc(descriptor)))


# --------------------------------------------------------------------------- #
#  Map container
# --------------------------------------------------------------------------- #
class Map:
    """A minimalistic map: 3‑D points + camera trajectory."""

    def __init__(self) -> None:
        self.points: Dict[int, MapPoint] = {}
        self.keyframe_indices: List[int] = []
        self.poses: List[np.ndarray] = []  # List of 4×4 camera‑to‑world matrices (World-from-Camera)
        self._next_pid: int = 0

    # ---------------- Camera trajectory ---------------- #
    def add_pose(self, pose_c_w: np.ndarray, is_keyframe: bool) -> None:
        """Append a 4×4 *pose_c_w* (camera‑to‑world) to the trajectory.""" ## CHANGED TO CAMERA-FROM-WORLD
        assert pose_c_w.shape == (4, 4), "Pose must be 4×4 homogeneous matrix"
        self.poses.append(pose_c_w.copy())
        if is_keyframe:
            self.keyframe_indices.append(len(self.poses) - 1)

    # ---------------- Landmarks ------------------------ #
    def add_points(self, pts3d: np.ndarray, colours: Optional[np.ndarray] = None,
                   keyframe_idx: int = -1) -> List[int]:
        """Add a set of 3‑D points and return the list of newly assigned ids."""
        if pts3d.ndim != 2 or pts3d.shape[1] != 3:
            raise ValueError("pts3d must be (N,3)")
        new_ids: List[int] = []

        colours = colours if colours is not None else np.ones_like(pts3d)
        for p, c in zip(pts3d, colours):
            pid = self._next_pid
            self.points[pid] = MapPoint(
                pid,
                p.astype(np.float64),
                keyframe_idx,
                c.astype(np.float32),
            )
            new_ids.append(pid)
            self._next_pid += 1
        return new_ids


    # ---------------- Convenience accessors ------------ #
    def get_point_array(self) -> np.ndarray:
        """Return all landmark positions as an (N,3) array (N may be 0)."""
        if not self.points:
            return np.empty((0, 3))
        return np.stack([mp.position for mp in self.points.values()], axis=0)
    
    def get_color_array(self) -> np.ndarray:
        if not self.points:
            return np.empty((0, 3), np.float32)
        return np.stack([mp.colour for mp in self.points.values()])

    def point_ids(self) -> List[int]:
        return list(self.points.keys())

    def __len__(self) -> int:
        return len(self.points)
    
    # ---------------- Merging landmarks ---------------- #
    def fuse_closeby_duplicate_landmarks(self, radius: float = 0.05) -> None:
        """Average-merge landmarks whose centres are closer than ``radius``."""

        if len(self.points) < 2:
            return

        ids = list(self.points.keys())
        pts = np.stack([self.points[i].position for i in ids])
        tree = cKDTree(pts)
        pairs = sorted(tree.query_pairs(radius))

        removed: set[int] = set()
        for i, j in pairs:
            ida, idb = ids[i], ids[j]
            if idb in removed or ida in removed:
                continue
            pa = self.points[ida].position
            pb = self.points[idb].position
            self.points[ida].position = (pa + pb) * 0.5
            removed.add(idb)

        for idx in removed:
            self.points.pop(idx, None)

