# slam/core/ba_utils.py

import cv2
import numpy as np
import pyceres
from pycolmap import cost_functions, CameraModelId

def _pose_to_quat_trans(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4×4 camera-to-world pose matrix into
    a quaternion (w, x, y, z) and a translation vector (3,).
    """
    R = T[:3, :3]
    aa, _ = cv2.Rodrigues(R)           # rotation vector = axis * angle
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        axis = aa.flatten() / theta
        qw = np.cos(theta / 2.0)
        sin_half = np.sin(theta / 2.0)
        qx, qy, qz = axis * sin_half
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    trans = T[:3, 3].astype(np.float64).copy()
    return quat, trans

def _quat_trans_to_pose(quat: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 4×4 pose matrix from a quaternion (w, x, y, z)
    and translation vector (3,).
    """
    qw, qx, qy, qz = quat
    # build rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def run_bundle_adjustment(
    world_map,
    K: np.ndarray,
    keypoints: list[list[cv2.KeyPoint]],
    *,
    fix_first_pose: bool = True,
    loss: str = 'huber',
    huber_thr: float = 1.0,
    max_iters: int = 40
):
    """
    Jointly refine all camera poses (world_map.poses) and 3D points
    (world_map.points) using a simple PINHOLE model. `keypoints`
    is a list of length N_poses, each entry a List[cv2.KeyPoint].
    """
    # -- 1) Intrinsics block ------------------------------------------------
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    intr = np.array([fx, fy, cx, cy], dtype=np.float64)

    problem = pyceres.Problem()
    problem.add_parameter_block(intr, 4)
    problem.set_parameter_block_constant(intr)

    # -- 2) Pose parameter blocks (quaternion + translation) -----------------
    quat_params: list[np.ndarray] = []
    trans_params: list[np.ndarray] = []
    for T in world_map.poses:
        quat, trans = _pose_to_quat_trans(T)
        quat_params.append(quat)
        trans_params.append(trans)

    # add quaternion blocks with manifold
    for q in quat_params:
        problem.add_parameter_block(q, 4)
        problem.set_manifold(q, pyceres.EigenQuaternionManifold())
    # add translation blocks
    for t in trans_params:
        problem.add_parameter_block(t, 3)

    if fix_first_pose:
        problem.set_parameter_block_constant(quat_params[0])
        problem.set_parameter_block_constant(trans_params[0])

    # -- 3) 3D point parameter blocks ----------------------------------------
    point_ids = world_map.point_ids()
    point_params = [
        world_map.points[pid].position.copy().astype(np.float64)
        for pid in point_ids
    ]
    for X in point_params:
        problem.add_parameter_block(X, 3)

    # -- 4) Loss function ----------------------------------------------------
    loss_fn = pyceres.HuberLoss(huber_thr) if loss.lower() == 'huber' else None

    # -- 5) Add reprojection residuals --------------------------------------
    for j, pid in enumerate(point_ids):
        X_block = point_params[j]
        mp = world_map.points[pid]
        for frame_idx, kp_idx in mp.observations:
            u, v = keypoints[frame_idx][kp_idx].pt
            uv = np.array([u, v], dtype=np.float64)

            cost = cost_functions.ReprojErrorCost(
                CameraModelId.PINHOLE,
                uv
            )
            # order: [quat, translation, point3D, intrinsics]
            problem.add_residual_block(
                cost,
                loss_fn,
                [
                    quat_params[frame_idx],
                    trans_params[frame_idx],
                    X_block,
                    intr
                ]
            )

    # -- 6) Solve ------------------------------------------------------------
    options = pyceres.SolverOptions()
    options.max_num_iterations = max_iters
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    print(summary.BriefReport())

    # -- 7) Write optimized values back into world_map -----------------------
    # Update poses
    for i, (q, t) in enumerate(zip(quat_params, trans_params)):
        world_map.poses[i][:] = _quat_trans_to_pose(q, t)
    # Update points
    for pid, X in zip(point_ids, point_params):
        world_map.points[pid].position[:] = X
