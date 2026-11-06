# """
# Synthetic‐data unit‑tests for *ba_utils.py*
# ===========================================

# These tests generate a small 3‑D scene, propagate a pin‑hole camera
# through a known motion, add optional noise to the **initial** geometry
# and verify that each bundle‑adjustment helper implemented in
# *ba_utils.py*

#     • two_view_ba
#     • pose_only_ba
#     • local_bundle_adjustment

# reduces the mean reprojection RMSE.

# **Pose convention – T_cw (camera-from-world)(world→camera)**
# ----------------------------------------
# Your SLAM pipeline stores a pose as the rigid-body transform **from the
# camera frame to the world frame**.  To project a world point X_w into a
# camera at T_wc we therefore use

# ```
# X_c = R_wcᵀ · (X_w − t_wc)
# ```

# Both the synthetic generator and the RMSE metric below follow that
# convention so that the tests are consistent with run-time code.

# Requires
# --------
# * OpenCV ≥ 4 (for `cv2.KeyPoint`, `cv2.Rodrigues`)
# * `pyceres` + `pycolmap` (same as *ba_utils*)

# Run with *pytest* or plain *unittest*:

# ```bash
# python -m pytest test_ba_utils_fixed.py     # preferred
# # – or –
# python test_ba_utils_fixed.py               # falls back to unittest.main()
# ```
# """

from __future__ import annotations
import math
import unittest
from typing import List, Tuple, Dict

import cv2
import numpy as np

# import the module under test
import slam.core.ba_utils as bau

# TODO: Create Visualization for points and camera poses

# ------------------------------------------------------------
# Minimal SLAM‑like data containers understood by ba_utils
# ------------------------------------------------------------
class MapPoint:
    """Light‑weight replacement for *MapPoint*."""

    def __init__(self, position: np.ndarray):
        self.position: np.ndarray = position.astype(np.float64)
        # list[(frame_idx, kp_idx)]
        self.observations: List[Tuple[int, int, np.ndarray]] = []


class WorldMap:
    """Holds camera poses (T_cw) and 3‑D points."""

    def __init__(self):
        self.poses: List[np.ndarray] = []  # each 4×4 SE(3)
        self.points: Dict[int, MapPoint] = {}  # pid → MapPoint


# ------------------------------------------------------------
# Pose conversion utilities
# ------------------------------------------------------------
# TODO: Code duplication with slam/core/pose_utils.py, refactor
def T_wc_to_T_cw(T_wc: np.ndarray) -> np.ndarray:
    """Convert camera-to-world pose to world-to-camera pose."""
    T_cw = np.eye(4, dtype=np.float64)
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    T_cw[:3, :3] = R_wc.T
    T_cw[:3, 3] = -R_wc.T @ t_wc
    return T_cw


def T_cw_to_T_wc(T_cw: np.ndarray) -> np.ndarray:
    """Convert world-to-camera pose to camera-to-world pose."""
    T_wc = np.eye(4, dtype=np.float64)
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    T_wc[:3, :3] = R_cw.T
    T_wc[:3, 3] = -R_cw.T @ t_cw
    return T_wc


# ------------------------------------------------------------
# Synthetic scene generator
# ------------------------------------------------------------
WIDTH, HEIGHT = 1280, 960
FX = FY = 800.0
CX, CY = WIDTH / 2.0, HEIGHT / 2.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float64)


def _yaw_to_R(yaw_deg: float) -> np.ndarray:
    """Rotation around *y* axis (right‑handed, degrees → 3×3)."""
    theta = math.radians(yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def generate_scene(
    n_frames: int,
    n_points: int = 50,
    dx_per_frame: float = 0.10,
    yaw_per_frame_deg: float = 2.0,
    pix_noise: float = 5.0,
    pose_trans_noise: float = 0.5,
    pose_rot_noise_deg: float = 15.0,
    point_noise: float = 0.05,
    *,
    add_noise: bool = True,
):
    """Return *(world_map, keypoints)* with optional noisy initial estimates.

    Set *add_noise=False* (or individual *_noise parameters to 0) to
    create a perfect initialisation that should converge in a single
    BA iteration.  Ground‑truth is used only to generate measurements –
    the initial geometry fed to BA is perturbed only when noise is
    requested.
    
    NOTE: This function generates T_wc poses but converts them to T_cw 
    for storage in world_map to match ba_utils expectations.
    """

    # When add_noise=False force all noise parameters to zero
    if not add_noise:
        pix_noise = pose_trans_noise = pose_rot_noise_deg = point_noise = 0.0

    rng = np.random.default_rng(42)

    # --- ground‑truth 3‑D points -------------------------------------
    pts_gt = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, n_points),  # X
            rng.uniform(-0.7, 0.7, n_points),  # Y
            rng.uniform(4.0, 8.0, n_points),  # Z
        )
    )

    # --- ground‑truth camera poses (camera-to-world T_wc for projection) ---
    poses_gt_wc: List[np.ndarray] = []
    for i in range(n_frames):
        R = _yaw_to_R(i * yaw_per_frame_deg)
        t = np.array([i * dx_per_frame, 0.0, 0.0], np.float64)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = t
        poses_gt_wc.append(T_wc)

    # --- create (possibly noisy) *initial* map -----------------------
    wmap = WorldMap()
    keypoints: List[List[cv2.KeyPoint]] = [[] for _ in range(n_frames)]

    # a) camera poses --------------------------------------------------
    for T_wc_gt in poses_gt_wc:
        # Apply noise to T_wc
        t_noise = rng.normal(0.0, pose_trans_noise, 3)
        axis = rng.normal(0.0, 1.0, 3)
        axis /= np.linalg.norm(axis)
        angle = math.radians(pose_rot_noise_deg) * rng.normal()
        R_noise, _ = cv2.Rodrigues(axis * angle)

        T_wc_noisy = np.eye(4, dtype=np.float64)
        T_wc_noisy[:3, :3] = R_noise @ T_wc_gt[:3, :3]
        T_wc_noisy[:3, 3] = T_wc_gt[:3, 3] + t_noise
        
        # Convert T_wc to T_cw for storage (ba_utils expects T_cw)
        T_cw_noisy = T_wc_to_T_cw(T_wc_noisy)
        wmap.poses.append(T_cw_noisy)

    # b) points + observations ---------------------------------------
    for pid, X_w in enumerate(pts_gt):
        X_init = X_w + rng.normal(0.0, point_noise, 3)
        mp = MapPoint(X_init)
        wmap.points[pid] = mp

        for f_idx, T_wc in enumerate(poses_gt_wc):
            # Project through **ground‑truth** T_wc pose to create measurement
            R_wc, t_wc = T_wc[:3, :3], T_wc[:3, 3]
            X_c = R_wc.T @ (X_w - t_wc)  # world → camera frame

            Z = X_c[2]
            if Z <= 0:  # behind camera
                continue

            # Homogeneous pixel coordinates via intrinsics
            uv_h = K @ X_c
            u = uv_h[0] / Z
            v = uv_h[1] / Z

            if not (0.0 <= u < WIDTH and 0.0 <= v < HEIGHT):
                continue

            # add pixel noise (measurement noise, not to the *initial* estimate)
            u_meas = u + rng.normal(0.0, pix_noise)
            v_meas = v + rng.normal(0.0, pix_noise)

            kp = cv2.KeyPoint(float(u_meas), float(v_meas), 1)
            kp_idx = len(keypoints[f_idx])
            keypoints[f_idx].append(kp)
            mp.observations.append((f_idx, kp_idx, np.random.rand(128)))  # dummy descriptor

    return wmap, keypoints


# ------------------------------------------------------------
# Utility: reprojection RMSE
# ------------------------------------------------------------

def reproj_rmse(wmap: WorldMap, keypoints, frames: List[int] | None = None) -> float:
    """
    Compute reprojection RMSE.
    wmap.poses are assumed to be T_cw (world-to-camera) as expected by ba_utils.
    """
    sq_err = 0.0
    count = 0
    frames = set(range(len(keypoints))) if frames is None else set(frames)

    for mp in wmap.points.values():
        for f_idx, kp_idx, descriptor in mp.observations:
            if f_idx not in frames:
                continue
            kp = keypoints[f_idx][kp_idx]
            u_m, v_m = kp.pt

            # wmap.poses[f_idx] is T_cw (world-to-camera)
            T_cw = wmap.poses[f_idx]
            X_w = mp.position
            R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
            X_c = R_cw @ X_w + t_cw  # world → camera using T_cw
            
            Z = X_c[2]
            if Z <= 0:
                continue

            uv_h = K @ X_c
            u_p = uv_h[0] / Z
            v_p = uv_h[1] / Z

            sq_err += (u_p - u_m) ** 2 + (v_p - v_m) ** 2
            count += 2

    return math.sqrt(sq_err / count) if count else 0.0


# ------------------------------------------------------------
# Unit‑tests
# ------------------------------------------------------------
class TestBundleAdjustment(unittest.TestCase):
    def test_two_view_ba(self):
        wmap, kps = generate_scene(n_frames=2, add_noise=False)
        e0 = reproj_rmse(wmap, kps)
        self.assertAlmostEqual(e0, 0.0, places=3, msg="Initial reprojection error should be zero when add_noise=False")
        bau.two_view_ba(wmap, K, kps, max_iters=30)
        e1 = reproj_rmse(wmap, kps)
        print(f"two_view_ba: e0={e0:.6f}, e1={e1:.6f}")
        self.assertLessEqual(e1, e0 + 1e-9, msg=f"two_view_ba failed: {e0:.6f} → {e1:.6f}")

    def test_two_view_ba_with_noise(self):
        wmap, kps = generate_scene(n_frames=2, add_noise=True)
        e0 = reproj_rmse(wmap, kps)
        bau.two_view_ba(wmap, K, kps, max_iters=30)
        e1 = reproj_rmse(wmap, kps)
        print(f"two_view_ba with noise: e0={e0:.6f}, e1={e1:.6f}")
        self.assertLess(e1, e0, msg=f"two_view_ba failed to reduce error: {e0:.6f} → {e1:.6f}")

    def test_pose_only_ba(self):
        wmap, kps = generate_scene(n_frames=3, add_noise=False)
        e0 = reproj_rmse(wmap, kps, frames=[2])
        self.assertAlmostEqual(e0, 0.0, places=3)
        bau.pose_only_ba(wmap, K, kps, frame_idx=2, max_iters=15)
        e1 = reproj_rmse(wmap, kps, frames=[2])
        print(f"pose_only_ba: e0={e0:.6f}, e1={e1:.6f}")
        self.assertLessEqual(e1, e0 + 1e-9, msg=f"pose_only_ba failed: {e0:.6f} → {e1:.6f}")

    def test_pose_only_ba_with_noise(self):
        wmap, kps = generate_scene(n_frames=3, add_noise=True)
        e0 = reproj_rmse(wmap, kps, frames=[2])
        bau.pose_only_ba(wmap, K, kps, frame_idx=2, max_iters=15)
        e1 = reproj_rmse(wmap, kps, frames=[2])
        print(f"pose_only_ba with noise: e0={e0:.6f}, e1={e1:.6f}")
        self.assertLess(e1, e0, msg=f"pose_only_ba failed to reduce error: {e0:.6f} → {e1:.6f}")

    def test_local_bundle_adjustment(self):
        wmap, kps = generate_scene(n_frames=10, add_noise=False)
        e0 = reproj_rmse(wmap, kps)
        self.assertAlmostEqual(e0, 0.0, places=3, msg="Initial reprojection error should be zero when add_noise=False")
        bau.local_bundle_adjustment(wmap, K, kps, center_kf_idx=9, window_size=8, max_iters=25)
        e1 = reproj_rmse(wmap, kps)
        print(f"local_bundle_adjustment: e0={e0:.6f}, e1={e1:.6f}")
        self.assertLessEqual(e1, e0 + 1e-9, msg=f"local_bundle_adjustment failed: {e0:.6f} → {e1:.6f}")

    def test_local_bundle_adjustment_with_noise(self):
        wmap, kps = generate_scene(n_frames=10, add_noise=True)
        e0 = reproj_rmse(wmap, kps)
        bau.local_bundle_adjustment(wmap, K, kps, center_kf_idx=9, window_size=8, max_iters=25)
        e1 = reproj_rmse(wmap, kps)
        print(f"local_bundle_adjustment with noise: e0={e0:.6f}, e1={e1:.6f}")
        self.assertLess(e1, e0, msg=f"local_bundle_adjustment failed to reduce error: {e0:.2f} → {e1:.2f}")


if __name__ == "__main__":
    # Fallback to unittest runner when executed directly
    unittest.main(verbosity=2)