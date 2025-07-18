import numpy as np
from slam.core.multi_view_utils import multi_view_triangulation

def build_camera(pose_w_c, K):
    """Return 3×4 projection matrix from camera→world pose."""
    return K @ np.linalg.inv(pose_w_c)[:3, :4]

def random_pose(tx=0.0, ty=0.0, tz=0.0):
    """Simple axis-aligned translation pose_w_c."""
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T

def test_triangulation_noise_free():
    # ---------- synthetic scene --------------------------------------
    fx = fy = 500.0
    cx = cy = 320.0
    K  = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0,  0,  1]])

    # camera 0 at origin, camera 1 translated 1 m on +X, camera 2 on +Y
    poses = [random_pose(0, 0, 0),
             random_pose(1, 0, 0),
             random_pose(0, 1, 0)]

    # ground-truth world point (in front of all cameras)
    X_w_gt = np.array([2.0, 1.5, 8.0])

    # synthetic pixel observations
    pts2d = []
    for T in poses:
        pc = (np.linalg.inv(T) @ np.append(X_w_gt, 1))[:3]
        uv = (K @ pc)[:2] / pc[2]
        pts2d.append(uv)

    # ---------- call the function ------------------------------------
    X_w_est = multi_view_triangulation(
        K, poses, np.float32(pts2d),
        min_depth=0.5, max_depth=50.0, max_rep_err=0.5)

    assert X_w_est is not None, "Triangulation unexpectedly failed"
    # sub-millimetre accuracy in noise-free synthetic setting
    assert np.allclose(X_w_est, X_w_gt, atol=1e-3)

def test_triangulation_with_pixel_noise():
    np.random.seed(42)
    fx = fy = 600.0
    cx = cy = 320.0
    K  = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0,  0,  1]])

    poses = [random_pose(0, 0, 0),
             random_pose(1, 0.2, 0),
             random_pose(-0.3, 1, 0),
             random_pose(0.5, -0.1, 0.3)]

    X_w_gt = np.array([-1.5, 0.8, 6.5])
    pts2d = []
    for T in poses:
        pc  = (np.linalg.inv(T) @ np.append(X_w_gt, 1))[:3]
        uv  = (K @ pc)[:2] / pc[2]
        uv += np.random.normal(scale=0.4, size=2)  # add 0.4-px Gaussian noise
        pts2d.append(uv)

    X_w_est = multi_view_triangulation(
        K, poses, np.float32(pts2d),
        min_depth=0.5, max_depth=50.0, max_rep_err=2.0)

    assert X_w_est is not None, "Triangulation failed with moderate noise"
    # centimetre-level accuracy is fine here
    assert np.linalg.norm(X_w_est - X_w_gt) < 0.05
