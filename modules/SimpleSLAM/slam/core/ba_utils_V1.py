# -*- coding: utf-8 -*-
"""
ba_utils.py
===========

Light-weight Bundle-Adjustment helpers built on **pyceres**.
Implements
  • two_view_ba(...)
  • pose_only_ba(...)
  • local_bundle_adjustment(...)
and keeps the older run_bundle_adjustment (full BA) for
back-compatibility.

A *blue-print* for global_bundle_adjustment is included but not wired.
"""
from __future__ import annotations
import cv2
import numpy as np
import pyceres
from pycolmap import cost_functions, CameraModelId

# --------------------------------------------------------------------- #
#  Small pose ⇄ parameter converters
# --------------------------------------------------------------------- #
def _pose_to_quat_trans(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R   = T[:3, :3]
    aa, _ = cv2.Rodrigues(R)
    theta = np.linalg.norm(aa)
    if theta < 1e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0], np.float64)
    else:
        axis = aa.flatten() / theta
        s    = np.sin(theta / 2.0)
        quat = np.array([np.cos(theta / 2.0),
                         axis[0] * s,
                         axis[1] * s,
                         axis[2] * s], np.float64)
    trans = T[:3, 3].astype(np.float64).copy()
    return quat, trans


def _quat_trans_to_pose(quat: np.ndarray, t: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)
    T          = np.eye(4, dtype=np.float64)
    T[:3, :3]  = R
    T[:3, 3]   = t
    return T

# --------------------------------------------------------------------- #
#  Shared helper to add one reprojection residual
# --------------------------------------------------------------------- #
def _add_reproj_edge(problem, loss_fn,
                     kp_uv: tuple[float, float],
                     quat_param, trans_param,
                     point_param, intr_param):
    """Create a pyceres residual block for one observation."""
    cost = cost_functions.ReprojErrorCost(
        CameraModelId.PINHOLE,
        np.asarray(kp_uv, np.float64)
    )
    problem.add_residual_block(
        cost, loss_fn,
        [quat_param, trans_param, point_param, intr_param]
    )

# --------------------------------------------------------------------- #
#  1) Two-view BA  (bootstrap refinement)
# --------------------------------------------------------------------- #
def two_view_ba(world_map, K, keypoints, max_iters: int = 20):
    """
    Refine the two initial camera poses + all bootstrap landmarks.

    Assumes `world_map` has exactly *two* poses (T_0w, T_1w) and that
    each MapPoint already stores **two** observations (frame-0 and
    frame-1).  Called once right after initialisation.
    """
    assert len(world_map.poses) == 2, "two_view_ba expects exactly 2 poses"

    _core_ba(world_map, K, keypoints,
             opt_kf_idx=[0, 1],
             fix_kf_idx=[],
             max_iters=max_iters,
             info_tag="[2-view BA]")


# --------------------------------------------------------------------- #
#  2) Pose-only BA   (current frame refinement)
# --------------------------------------------------------------------- #
def pose_only_ba(world_map, K, keypoints,
                 frame_idx: int, max_iters: int = 8,
                 huber_thr: float = 2.0):
    """
    Optimise **only one pose** (SE3) while keeping all 3-D points fixed.
    Mimics ORB-SLAM's `Optimizer::PoseOptimization`.
    """
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intr = np.array([fx, fy, cx, cy], np.float64)

    quat, trans = _pose_to_quat_trans(world_map.poses[frame_idx])

    problem = pyceres.Problem()
    problem.add_parameter_block(quat, 4)
    problem.add_parameter_block(trans, 3)
    problem.set_manifold(quat, pyceres.EigenQuaternionManifold())
    problem.add_parameter_block(intr, 4)
    problem.set_parameter_block_constant(intr)

    loss_fn = pyceres.HuberLoss(huber_thr)

    for pid, mp in world_map.points.items():
        for f_idx, kp_idx in mp.observations:
            if f_idx != frame_idx:
                continue
            u, v = keypoints[f_idx][kp_idx].pt
            _add_reproj_edge(problem, loss_fn,
                             (u, v), quat, trans, mp.position, intr)

    if problem.num_residual_blocks() < 10:
        print(f"POSE-ONLY BA skipped – not enough residuals")
        return  # too few observations

    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iters
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, problem, summary)

    world_map.poses[frame_idx][:] = _quat_trans_to_pose(quat, trans)
    # print(f"[Pose-only BA] iters={summary.iterations_used}"
    #       f"  inliers={problem.num_residual_blocks()}")
    print(f"[Pose-only BA] iters={summary.num_successful_steps}"
          f"  inliers={problem.num_residual_blocks()}")


# --------------------------------------------------------------------- #
#  3) Local BA  (sliding window)
# --------------------------------------------------------------------- #
def local_bundle_adjustment(world_map, K, keypoints,
                            center_kf_idx: int,
                            window_size: int = 8,
                            max_points  : int = 3000,
                            max_iters   : int = 15):
    """
    Optimise the *last* `window_size` key-frames around
    `center_kf_idx` plus all landmarks they observe.
    Older poses are kept fixed (gauge).
    """
    first_opt = max(0, center_kf_idx - window_size + 1)
    opt_kf    = list(range(first_opt, center_kf_idx + 1))
    fix_kf    = list(range(0, first_opt))

    _core_ba(world_map, K, keypoints,
             opt_kf_idx=opt_kf,
             fix_kf_idx=fix_kf,
             max_points=max_points,
             max_iters=max_iters,
             info_tag=f"[Local BA (kf {center_kf_idx})]")


# --------------------------------------------------------------------- #
#  4) Global BA  (blue-print only)
# --------------------------------------------------------------------- #
def global_bundle_adjustment_blueprint(world_map, K, keypoints):
    """
    *** NOT WIRED YET ***

    Outline:
      • opt_kf_idx = all key-frames
      • fix_kf_idx = []  (maybe fix the very first to anchor gauge)
      • run _core_ba(...) with a robust kernel
      • run asynchronously (thread) and allow early termination
        if tracking thread needs the map
    """
    raise NotImplementedError


# --------------------------------------------------------------------- #
#  Shared low-level BA engine
# --------------------------------------------------------------------- #
def _core_ba(world_map, K, keypoints,
             *,
             opt_kf_idx: list[int],
             fix_kf_idx: list[int],
             max_points: int | None = None,
             max_iters : int = 20,
             info_tag  : str = ""):
    """
    Generic sparse BA over a **subset** of poses + points.
    """
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    intr = np.array([fx, fy, cx, cy], np.float64)

    problem = pyceres.Problem()
    # TODO DONT add intrinsics if they are fixed
    problem.add_parameter_block(intr, 4)
    problem.set_parameter_block_constant(intr)

    # --- pose blocks ---------------------------------------------------
    #TODO why for looop, optimize relative poses instead of absolute, 
    quat_params, trans_params = {}, {}
    for k in opt_kf_idx:
        quat, tr = _pose_to_quat_trans(world_map.poses[k])
        quat_params[k] = quat
        trans_params[k] = tr
        problem.add_parameter_block(quat, 4)
        problem.set_manifold(quat, pyceres.EigenQuaternionManifold())
        problem.add_parameter_block(tr, 3)

    for k in fix_kf_idx:
        quat, tr = _pose_to_quat_trans(world_map.poses[k])
        quat_params[k] = quat
        trans_params[k] = tr
        problem.add_parameter_block(quat, 4)
        problem.add_parameter_block(tr, 3)
        problem.set_parameter_block_constant(quat)
        problem.set_parameter_block_constant(tr)

    # --- point blocks --------------------------------------------------
    loss_fn = pyceres.HuberLoss(1.0)
    added_pts = 0
    for pid, mp in world_map.points.items():
        # keep only points seen by at least one optimisable KF
        if not any(f in opt_kf_idx for f, _ in mp.observations):
            continue
        if max_points and added_pts >= max_points:
            continue
        problem.add_parameter_block(mp.position, 3)
        added_pts += 1

        for f_idx, kp_idx in mp.observations:
            if f_idx not in opt_kf_idx and f_idx not in fix_kf_idx:
                continue
            u, v = keypoints[f_idx][kp_idx].pt
            _add_reproj_edge(problem, loss_fn,
                             (u, v),
                             quat_params[f_idx],
                             trans_params[f_idx],
                             mp.position,
                             intr)
    print(problem.num_residual_blocks(), "residuals added")
    if problem.num_residual_blocks() < 10:
        print(f"{info_tag} skipped – not enough residuals")
        return

    # --- solve ---------------------------------------------------------
    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iters
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, problem, summary)

    # --- write poses back ---------------------------------------------
    for k in opt_kf_idx:
        world_map.poses[k][:] = _quat_trans_to_pose(
            quat_params[k], trans_params[k])

    iters = (getattr(summary, "iterations_used", getattr(summary, "num_successful_steps", getattr(summary, "num_iterations", None))))
    print(f"{info_tag}  iters={iters}  "
          f"χ²={summary.final_cost:.2f}  "
          f"res={problem.num_residual_blocks()}")
