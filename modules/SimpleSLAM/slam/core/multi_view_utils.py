# slam/core/multi_view_utils.py
"""Utilities for deferred (multi-view) triangulation in the SLAM pipeline."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from .landmark_utils import Map


# --------------------------------------------------------------------------- #
#  Robust linear triangulation across ≥ 2 views
# --------------------------------------------------------------------------- #
def multi_view_triangulation(
    K: np.ndarray,
    poses_w_c: List[np.ndarray],              # M × 4×4  (cam→world)
    pts2d: np.ndarray,                        # M × 2    (pixels)
    *,
    min_depth: float,
    max_depth: float,
    max_rep_err: float,
    eps: float = 1e-6
) -> Optional[np.ndarray]:
    """Return xyz _w or **None** if cheirality / depth / reprojection checks fail."""
    assert len(poses_w_c) == len(pts2d) >= 2, "Need ≥ 2 consistent views"

    # Build A (2 M × 4)
    A = []
    for T_w_c, (u, v) in zip(poses_w_c, pts2d):
        P = K @ np.linalg.inv(T_w_c)[:3, :4]
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.stack(A)

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < eps:                         # degenerate solution
        return None
    X = X_h[:3] / X_h[3]

    # Cheats: cheirality, depth & reprojection
    reproj, depths = [], []
    for T_w_c, (u, v) in zip(poses_w_c, pts2d):
        pc = (np.linalg.inv(T_w_c) @ np.append(X, 1.0))[:3] # TODO pc = (np.linalg.inv(T_w_c) @ np.append(X, 1.0))[:3] Not sure if this is correct
        if pc[2] <= 0:                             # behind the camera
            print("Cheirality check failed:", pc)
            return None
        depths.append(pc[2])

        uv_hat = (K @ pc)[:2] / pc[2]
        reproj.append(np.linalg.norm(uv_hat - (u, v)))

    if not (min_depth <= np.mean(depths) <= max_depth):
        print(f"Depth check failed: {np.mean(depths)} not in [{min_depth}, {max_depth}]")
        return None
    if np.mean(reproj) > max_rep_err:
        print(f"Reprojection error check failed: {np.mean(reproj)} > {max_rep_err}")
        return None
    return X


# --------------------------------------------------------------------------- #
#  Track manager – accumulates 2-D key-frame observations
# --------------------------------------------------------------------------- #
@dataclass
class _Obs:
    kf_idx: int
    kp_idx: int
    uv: Tuple[float, float]


class MultiViewTriangulator:
    """
    Accumulate feature tracks (key-frames only) and triangulate once a track
    appears in ≥ `min_views` distinct key-frames.
    """

    def __init__(self,
                 K: np.ndarray,
                 *,
                 min_views:    int,
                 merge_radius: float,
                 max_rep_err:  float,
                 min_depth:    float,
                 max_depth:    float):
        # All thresholds come from the caller – no magic numbers inside.
        self.K            = K
        self.min_views    = max(2, min_views)
        self.merge_radius = merge_radius
        self.max_rep_err  = max_rep_err
        self.min_depth    = min_depth
        self.max_depth    = max_depth

        self._track_obs: Dict[int, List[_Obs]] = {}
        self._kf_poses:  Dict[int, np.ndarray] = {}
        self._kf_imgs:  Dict[int, np.ndarray]  = {}        # BGR uint8
        self._triangulated: set[int]           = set()

    # ------------------------------------------------------------------ #
    def add_keyframe(self,
                     frame_idx: int,
                     pose_w_c: np.ndarray,
                     kps: List,                       # List[cv2.KeyPoint]
                     track_map: Dict[int, int],
                     img_bgr: np.ndarray) -> None:
        """Register observations (and keep the *full-res* image for colour sampling)."""
        self._kf_poses[frame_idx] = pose_w_c.copy()
        self._kf_imgs[frame_idx]  = img_bgr            # shallow copy is fine
        for kp_idx, tid in track_map.items():
            u, v = kps[kp_idx].pt
            self._track_obs.setdefault(tid, []).append(_Obs(frame_idx, kp_idx, (u, v)))

    # ------------------------------------------------------------------ #
    def triangulate_ready_tracks(self, world_map: Map) -> List[int]:
        """Triangulate mature tracks, insert them into the map, and return new ids."""
        new_ids: List[int] = []

        for tid, obs in list(self._track_obs.items()):
            if tid in self._triangulated or len(obs) < self.min_views:
                continue
            
            obs_sorted = sorted(obs, key=lambda o: o.kf_idx)
            poses, pts2d = [], []
            for o in obs_sorted:
                pose = self._kf_poses.get(o.kf_idx)
                if pose is None:
                    break
                poses.append(pose)
                pts2d.append(o.uv)
            else:
                # print(f"Triangulating track {tid} with {len(obs)} observations")
                X = multi_view_triangulation(
                    self.K, poses, np.float32(pts2d),
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                    max_rep_err=self.max_rep_err,
                )
                # print(" Triangulated 3D point:", X)
                if X is None:
                    continue

                # --------- colour sampling (pick first obs with an image) -------
                rgb = (1.0, 1.0, 1.0)                    # default white
                for o in obs_sorted:
                    img = self._kf_imgs.get(o.kf_idx)
                    if img is None:
                        continue
                    h, w, _ = img.shape
                    x, y = int(round(o.uv[0])), int(round(o.uv[1]))
                    if 0 <= x < w and 0 <= y < h:
                        b, g, r = img[y, x]
                        rgb = (r / 255.0, g / 255.0, b / 255.0)
                        break

                # --------------- map insertion (+ optional merging) -------------
                X = world_map.align_points_to_map(
                    X[None, :], radius=self.merge_radius
                )[0]
                pid = world_map.add_points(X[None, :], np.float32([[*rgb]]))[0]
                for o in obs_sorted:
                    world_map.points[pid].add_observation(o.kf_idx, o.kp_idx)

                new_ids.append(pid)
                self._triangulated.add(tid)
                self._track_obs.pop(tid, None)           # free memory

        return new_ids
