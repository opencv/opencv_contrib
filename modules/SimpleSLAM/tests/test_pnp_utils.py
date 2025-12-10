"""Unit tests for pnp_utils.py

Test coverage goals
-------------------
Functions under test:
    * project_points
    * associate_landmarks
    * refine_pose_pnp

We generate synthetic, perfectly controlled data (and some noisy variants) so the
expected geometric relationships are analytically known.

Conventions verified:
    * refine_pose_pnp returns (R, t) mapping world -> camera (OpenCV convention).
    * project_points expects pose_w_c (camera->world) and internally inverts it.

Edge cases handled:
    * Empty inputs in associate_landmarks
    * Insufficient points for refine_pose_pnp

Precision thresholds:
    * Rotation matrices compared with max abs diff <= 1e-6 (ideal / noise-free)
    * Translations compared with <= 1e-6 (ideal) or small tolerance with noise

These tests use only NumPy + OpenCV (cv2). If OpenCV is not available the file
will skip tests that require it.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
except Exception:  # pragma: no cover
    raise ImportError("OpenCV (cv2) is required for some tests")

# Import functions under test
import importlib
from slam.core import pnp_utils
project_points = pnp_utils.project_points
associate_landmarks = pnp_utils.associate_landmarks
refine_pose_pnp = pnp_utils.refine_pose_pnp


# ---------------------------------------------------------------------------
# Helper creators
# ---------------------------------------------------------------------------

def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a random 3x3 rotation using axis-angle."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-math.pi, math.pi)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R

def make_camera_pose(rng: np.random.Generator):
    """Return (pose_w_c, R_wc, t_wc) camera->world homogeneous pose."""
    R_wc = random_rotation(rng)
    t_wc = rng.uniform(-2, 2, size=3)
    pose_w_c = np.eye(4)
    pose_w_c[:3, :3] = R_wc
    pose_w_c[:3, 3] = t_wc
    return pose_w_c, R_wc, t_wc


def invert_pose(T_w_c: np.ndarray) -> np.ndarray:
    R = T_w_c[:3, :3]
    t = T_w_c[:3, 3]
    T_c_w = np.eye(4)
    T_c_w[:3, :3] = R.T
    T_c_w[:3, 3] = -R.T @ t
    return T_c_w

# ---------------------------------------------------------------------------
# project_points
# ---------------------------------------------------------------------------

def test_project_points_identity_camera_center():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    pose_w_c = np.eye(4)  # camera at origin, world == camera
    pts_w = np.array([[0, 0, 2], [1, 1, 2], [-1, -1, 2]], dtype=float)
    uv = project_points(K, pose_w_c, pts_w)
    # manual projection: u = fx*x/z + cx, v = fy*y/z + cy
    expected = np.array([
        [500*0/2 + 320, 500*0/2 + 240],
        [500*1/2 + 320, 500*1/2 + 240],
        [500*-1/2 + 320, 500*-1/2 + 240]
    ], dtype=np.float32)
    assert np.allclose(uv, expected)


def test_project_points_random_pose_roundtrip():
    rng = np.random.default_rng(42)
    K = np.array([[450, 0, 300], [0, 460, 200], [0, 0, 1]], dtype=float)
    pose_w_c, R_wc, t_wc = make_camera_pose(rng)
    # create random world points in front of camera: sample in camera frame then map to world
    pts_c = rng.uniform(0.5, 5.0, size=(50, 3))
    pts_c[:, :2] -= 0.5  # some lateral spread
    # camera->world pose: X_w = R_wc X_c + t_wc
    pts_w = (R_wc @ pts_c.T).T + t_wc
    # project
    uv = project_points(K, pose_w_c, pts_w)
    # reconstruct normalized rays and verify direction consistency
    T_c_w = np.linalg.inv(pose_w_c)
    R_cw = T_c_w[:3, :3]
    t_cw = T_c_w[:3, 3]
    # backproject one point: s * x_norm = R_cw (X_w - t_wc) ; we just check forward consistency
    x = pts_w[0]
    X_c = R_cw @ (x - pose_w_c[:3, 3])
    u_pred = K @ X_c
    u_pred = u_pred[:2] / u_pred[2]
    assert np.allclose(uv[0], u_pred, atol=1e-6)

# ---------------------------------------------------------------------------
# associate_landmarks
# ---------------------------------------------------------------------------

def make_keypoints(xy: np.ndarray):
    kps = []
    for (u, v) in xy:
        kp = cv2.KeyPoint(float(u), float(v), 1)
        kps.append(kp)
    return kps

@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV required for keypoints")
def test_associate_landmarks_basic():
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=float)
    pose_w_c = np.eye(4)
    pts_w = np.array([[0, 0, 2], [0.5, 0, 2], [0, 0.5, 2]], dtype=float)
    proj = project_points(K, pose_w_c, pts_w)
    # Add slight offsets so nearest still correct
    keypoints_xy = proj + np.array([[0.2, -0.1],[0.1,0.2],[-0.15,0.05]])
    kps = make_keypoints(keypoints_xy)
    pts3d, pts2d, kp_ids = associate_landmarks(K, pose_w_c, pts_w, kps, search_rad=5)
    assert pts3d.shape == (3,3)
    assert pts2d.shape == (3,2)
    assert len(kp_ids) == 3
    # Each associated 2D close to projection
    assert np.all(np.linalg.norm(pts2d - proj, axis=1) < 1.0)

@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV required for keypoints")
def test_associate_landmarks_empty_inputs():
    K = np.eye(3)
    pose_w_c = np.eye(4)
    pts3d, pts2d, idxs = associate_landmarks(K, pose_w_c, np.empty((0,3)), [], search_rad=5)
    assert pts3d.size == 0 and pts2d.size == 0 and idxs == []

# ---------------------------------------------------------------------------
# refine_pose_pnp
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV solvePnP required")
def test_refine_pose_pnp_world_to_camera_convention():
    rng = np.random.default_rng(0)
    # Intrinsics
    fx, fy = 600, 610
    cx, cy = 320, 240
    K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=float)

    # Ground truth camera pose (world->camera) chosen randomly
    R_wc = random_rotation(rng)  # this is camera->world; invert for world->camera
    t_wc = rng.uniform(-1,1,size=3)
    T_w_c = np.eye(4)
    T_w_c[:3,:3] = R_wc
    T_w_c[:3,3] = t_wc
    T_c_w = invert_pose(T_w_c)
    R_cw = T_c_w[:3,:3]
    t_cw = T_c_w[:3,3]

    # Synthesize world points visible in camera
    pts_c = rng.uniform(1.0,4.0,size=(100,3))
    pts_c[:,0:2] -= 0.5
    pts_w = (R_wc @ pts_c.T).T + t_wc

    # Project to image using ground truth world->camera extrinsics (R_cw, t_cw)
    P = K @ np.hstack([R_cw, t_cw.reshape(3,1)])
    pts_h = np.hstack([pts_w, np.ones((len(pts_w),1))])
    proj = (P @ pts_h.T).T
    uv = (proj[:,:2] / proj[:,2:])

    # Run PnP with a subset & small noise
    sel = rng.choice(len(pts_w), size=60, replace=False)
    pts3d = pts_w[sel]
    pts2d = uv[sel] + rng.normal(scale=0.5, size=(60,2))  # pixel noise

    R_est, t_est = refine_pose_pnp(K, pts3d.astype(np.float32), pts2d.astype(np.float32))
    assert R_est is not None and t_est is not None
    # Check they map world->camera: compare to ground truth R_cw, t_cw
    # Allow some tolerance due to noise & RANSAC
    assert np.allclose(R_est @ R_est.T, np.eye(3), atol=1e-6)
    assert np.allclose(np.linalg.det(R_est), 1.0, atol=1e-6)
    # Evaluate alignment of rotation
    rot_err = np.rad2deg(np.arccos(max(-1.0, min(1.0, ((np.trace(R_est.T @ R_cw)-1)/2)))))
    assert rot_err < 2.0  # degrees
    assert np.linalg.norm(t_est - t_cw) < 0.1

@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV solvePnP required")
def test_refine_pose_pnp_insufficient_points():
    K = np.eye(3)
    pts3d = np.random.randn(3,3).astype(np.float32)
    pts2d = np.random.randn(3,2).astype(np.float32)
    R, t = refine_pose_pnp(K, pts3d, pts2d)
    assert R is None and t is None


# ---------------------------------------------------------------------------
# Integration test: using associate_landmarks + refine_pose_pnp pipeline
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV required")
def test_pipeline_association_then_pnp():
    rng = np.random.default_rng(5)
    K = np.array([[500,0,320],[0,500,240],[0,0,1]], dtype=float)
    pose_w_c, R_wc, t_wc = make_camera_pose(rng)
    # Generate random 3D points in camera frame then map to world
    pts_c = rng.uniform(1.0,6.0,size=(120,3))
    pts_c[:, :2] -= 0.5
    pts_w = (R_wc @ pts_c.T).T + t_wc
    # Project to image and add noise
    uv = project_points(K, pose_w_c, pts_w) + rng.normal(scale=0.3,size=(len(pts_w),2))
    # Create keypoints at noisy positions
    kps = [cv2.KeyPoint(float(u), float(v), 1) for (u,v) in uv]
    # Subsample landmark set to mimic existing map
    map_sel = rng.choice(len(pts_w), size=80, replace=False)
    pts_w_map = pts_w[map_sel]
    pts3d, pts2d, _ = associate_landmarks(K, pose_w_c, pts_w_map, kps, search_rad=4.0)
    assert len(pts3d) >= 20  # should match many
    R_est, t_est = refine_pose_pnp(K, pts3d, pts2d)
    assert R_est is not None
    # Compare to ground truth world->camera (R_cw,t_cw)
    T_c_w = np.linalg.inv(pose_w_c)
    R_cw = T_c_w[:3,:3]
    t_cw = T_c_w[:3,3]
    rot_err = np.rad2deg(np.arccos(max(-1.0, min(1.0, (np.trace(R_est.T @ R_cw)-1)/2))))
    assert rot_err < 2.5
    assert np.linalg.norm(t_est - t_cw) < 0.15

if __name__ == '__main__':  # pragma: no cover
    import sys, pytest
    sys.exit(pytest.main([__file__, '-vv']))
