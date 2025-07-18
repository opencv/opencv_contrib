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

# --------------------------------------------------------------------------- #
#  MapPoint
# --------------------------------------------------------------------------- #
@dataclass
class MapPoint:
    """A single triangulated 3‑D landmark."""
    id: int
    position: np.ndarray  # shape (3,)
    keyframe_idx: int = -1
    colour: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))    # (3,) in **linear** RGB 0-1
    observations: List[Tuple[int, int]] = field(default_factory=list)  # (frame_idx, kp_idx)

    def add_observation(self, frame_idx: int, kp_idx: int) -> None:
        """Register that *kp_idx* in *frame_idx* observes this landmark."""
        self.observations.append((frame_idx, kp_idx))


# --------------------------------------------------------------------------- #
#  Map container
# --------------------------------------------------------------------------- #
class Map:
    """A minimalistic map: 3‑D points + camera trajectory."""

    def __init__(self) -> None:
        self.points: Dict[int, MapPoint] = {}
        self.poses: List[np.ndarray] = []  # List of 4×4 camera‑to‑world matrices
        self._next_pid: int = 0

    # ---------------- Camera trajectory ---------------- #
    def add_pose(self, pose_w_c: np.ndarray) -> None:
        """Append a 4×4 *pose_w_c* (camera‑to‑world) to the trajectory."""
        assert pose_w_c.shape == (4, 4), "Pose must be 4×4 homogeneous matrix"
        self.poses.append(pose_w_c.copy())

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

    # def add_points(self,
    #                xyz   : np.ndarray,             # (M,3) float32/64
    #                rgb   : np.ndarray | None = None  # (M,3) float32 in [0,1]
    #                ) -> list[int]:
    #     """
    #     Insert M new landmarks and return their integer ids.
    #     `rgb` may be omitted – we then default to light-grey.
    #     """
    #     xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    #     if rgb is None:
    #         rgb = np.full_like(xyz, 0.8, dtype=np.float32)   # light grey
    #     else:
    #         rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)

    #     assert xyz.shape == rgb.shape, \
    #         "xyz and rgb must have the same length"

    #     ids = list(range(self._next_pid, self._next_pid + len(xyz)))
    #     self._next_pid += len(xyz)

    #     for pid, p, c in zip(ids, xyz, rgb):
    #         self.points[pid] = MapPoint(pid, p, c)

    #     return ids


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


    def align_points_to_map(self, pts: np.ndarray, radius: float = 0.05) -> np.ndarray:
        """Rigidly align ``pts`` to existing landmarks using nearest neighbours."""
        map_pts = self.get_point_array()
        if len(map_pts) < 3 or len(pts) < 3:
            return pts

        tree = cKDTree(map_pts)
        src, dst = [], []
        for p in pts:
            idxs = tree.query_ball_point(p, radius)
            if idxs:
                src.append(p)
                dst.append(map_pts[idxs[0]])

        if len(src) < 3:
            return pts

        from .pnp_utils import align_point_clouds
        R, t = align_point_clouds(np.asarray(src), np.asarray(dst))
        pts_aligned = (R @ pts.T + t[:, None]).T
        return pts_aligned


# --------------------------------------------------------------------------- #
#  Geometry helpers (stay here to avoid cyclic imports)
# --------------------------------------------------------------------------- #
def triangulate_points(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Triangulate corresponding *pts1* ↔ *pts2* given (R, t).

    Parameters
    ----------
    K
        3×3 camera intrinsic matrix.
    R, t
        Rotation + translation from *view‑1* to *view‑2*.
    pts1, pts2
        Nx2 arrays of pixel coordinates (dtype float32/float64).
    Returns
    -------
    pts3d
        Nx3 array in *view‑1* camera coordinates (not yet in world frame).
    """
    if pts1.shape != pts2.shape:
        raise ValueError("pts1 and pts2 must be the same shape")
    if pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("pts1/pts2 must be (N,2)")

    proj1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    proj2 = K @ np.hstack((R, t.reshape(3, 1))) # Equivalent to proj1 @ get_homo_from_pose_rt(R, t)

    pts4d_h = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T) # TODO: do triangulation from scratch for N observations
    pts3d = (pts4d_h[:3] / pts4d_h[3]).T  # → (N,3)
    return pts3d
