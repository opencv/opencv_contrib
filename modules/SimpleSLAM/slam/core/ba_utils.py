# -*- coding: utf-8 -*-
"""
ba_utils.py
===========

ALWAYS PASS POSE as T_wc (world→camera) (camera-from-world) pose convention to pyceres.

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
import math
from scipy.spatial.transform import Rotation as R

# TODO: USES T_cw (camera-from-world) convention for storing poses in the map, because of PyCERES, REMEMBER TO CONVERT, use below functions
from slam.core.pose_utils import _pose_inverse, _pose_to_quat_trans, _quat_trans_to_pose

# TODO : UNCOMMENT below block IF YOU USE USES World-from-camera (camera-to-world) pose convention T_wc (camera→world) for storing poses in the map.
# from slam.core.pose_utils import (
#     _pose_inverse,           # T ↔ T⁻¹
#     _pose_to_quat_trans as _cw_to_quat_trans,
#     _quat_trans_to_pose as _quat_trans_to_cw,
# )
# def _pose_to_quat_trans(T_wc):
#     """
#     Convert **camera→world** pose T_wc to the (q_cw, t_cw) parameterisation
#     expected by COLMAP/pyceres residuals.
#     """
#     return _cw_to_quat_trans(_pose_inverse(T_wc))


# def _quat_trans_to_pose(q_cw, t_cw):
#     """
#     Convert optimised (q_cw, t_cw) back to a **camera→world** SE3 matrix.
#     """
#     return _pose_inverse(_quat_trans_to_cw(q_cw, t_cw))

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
# --- ba_utils.py: entry points now use kfs instead of a separate keypoints list ---
def two_view_ba(world_map, K, kfs, max_iters: int = 20):
    """
    Refine the two initial camera poses + all bootstrap landmarks.
    Expects KF indices 0 and 1 to exist.
    """
    assert len(world_map.poses) >= 2, "two_view_ba expects at least 2 poses"
    _core_ba(world_map, K, kfs,
             opt_kf_idx=[0, 1],
             fix_kf_idx=[],
             max_iters=max_iters,
             info_tag="[2-view BA]")

# --------------------------------------------------------------------- #
#  2) Pose-only BA   (current frame refinement)
# --------------------------------------------------------------------- #
def pose_only_ba(world_map, K, kfs,
                 kf_idx: int, max_iters: int = 8,
                 huber_thr: float = 2.0):
    """
    Optimise only one **keyframe** pose while keeping all 3-D points fixed.
    """
    import pyceres
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    intr = np.array([fx, fy, cx, cy], np.float64)

    # read pose from keyframe (authoritative), fall back to map if needed
    Tcw = kfs[kf_idx].pose
    quat, trans = _pose_to_quat_trans(Tcw)

    problem = pyceres.Problem()
    problem.add_parameter_block(quat, 4)
    problem.set_manifold(quat, pyceres.EigenQuaternionManifold())
    problem.add_parameter_block(trans, 3)

    problem.add_parameter_block(intr, 4)
    problem.set_parameter_block_constant(intr)

    loss_fn = pyceres.HuberLoss(huber_thr)

    # add residuals for observations in this KF
    added = 0
    for mp in world_map.points.values():
        # point is constant in pose-only BA
        problem.add_parameter_block(mp.position, 3)
        problem.set_parameter_block_constant(mp.position)
        for f_idx, kp_idx, _desc in mp.observations:
            if f_idx != kf_idx: 
                continue
            u, v = kfs[f_idx].kps[kp_idx].pt
            _add_reproj_edge(problem, loss_fn, (u, v), quat, trans, mp.position, intr)
            added += 1

    if added < 10:
        print("POSE-ONLY BA skipped – not enough residuals")
        return

    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iters
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, problem, summary)

    # write back to both KF and Map (keep consistent)
    new_Tcw = _quat_trans_to_pose(quat, trans)
    kfs[kf_idx].pose = new_Tcw
    if len(world_map.poses) > kf_idx:
        world_map.poses[kf_idx][:] = new_Tcw
    print(f"[Pose-only BA] iters={getattr(summary,'num_successful_steps',0)}  residuals={added}")

# --------------------------------------------------------------------- #
#  3) Local BA  (sliding window)
# --------------------------------------------------------------------- #
def local_bundle_adjustment(world_map, K, kfs,
                            center_kf_idx: int,
                            window_size: int = 6,
                            max_points  : int = 10000,
                            max_iters   : int = 15):
    """
    Optimise a sliding window of keyframes around center_kf_idx plus
    all landmarks they observe. Older KFs are fixed (gauge).
    """
    first_opt = max(1, center_kf_idx - window_size + 1)
    opt_kf    = list(range(first_opt, center_kf_idx + 1))
    fix_kf    = list(range(0, first_opt))
    print(f"opt_kf={opt_kf}, fix_kf={fix_kf}, center_kf_idx={center_kf_idx}")

    _core_ba(world_map, K, kfs,
             opt_kf_idx = opt_kf,
             fix_kf_idx = fix_kf,
             max_points = max_points,
             max_iters  = max_iters,
             info_tag   = f"[Local BA @ KF {center_kf_idx}]")

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
def _core_ba(world_map, K, kfs,
             *,
             opt_kf_idx: list[int],
             fix_kf_idx: list[int],
             max_points: int | None = None,
             max_iters : int = 20,
             info_tag  : str = ""):
    """
    Sparse BA over a subset of poses (keyframes) + points.
    Poses come from kfs[idx].pose; residual UVs from kfs[idx].kps[kp_idx].pt
    """
    import pyceres
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    intr = np.array([fx, fy, cx, cy], np.float64)

    problem = pyceres.Problem()
    loss_fn = pyceres.HuberLoss(2.0)

    problem.add_parameter_block(intr, 4)
    problem.set_parameter_block_constant(intr)

    # pose blocks (opt + fixed) from Keyframe poses
    quat_params, trans_params = {}, {}
    for k in opt_kf_idx:
        quat, tr = _pose_to_quat_trans(kfs[k].pose)
        quat_params[k] = quat; trans_params[k] = tr
        problem.add_parameter_block(quat, 4)
        problem.set_manifold(quat, pyceres.EigenQuaternionManifold())
        problem.add_parameter_block(tr, 3)

    for k in fix_kf_idx:
        quat, tr = _pose_to_quat_trans(kfs[k].pose)
        quat_params[k] = quat; trans_params[k] = tr
        problem.add_parameter_block(quat, 4)
        problem.set_manifold(quat, pyceres.EigenQuaternionManifold())
        problem.add_parameter_block(tr, 3)
        problem.set_parameter_block_constant(quat)
        problem.set_parameter_block_constant(tr)

    # point blocks + residuals
    added_pts = 0
    added_res = 0
    for mp in world_map.points.values():
        # keep only points seen by at least one optimisable KF
        if not any(f in opt_kf_idx for f, _, _ in mp.observations):
            continue
        if max_points and added_pts >= max_points:
            continue

        problem.add_parameter_block(mp.position, 3)
        added_pts += 1

        for f_idx, kp_idx, _desc in mp.observations:
            if (f_idx not in opt_kf_idx) and (f_idx not in fix_kf_idx):
                continue
            u, v = kfs[f_idx].kps[kp_idx].pt
            _add_reproj_edge(problem, loss_fn,
                             (u, v),
                             quat_params[f_idx],
                             trans_params[f_idx],
                             mp.position,
                             intr)
            added_res += 1

    if added_res < 10:
        print(f"{info_tag} skipped – not enough residuals")
        return

    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iters
    opts.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, problem, summary)

    # write poses back → Keyframes + Map
    for k in opt_kf_idx:
        new_Tcw = _quat_trans_to_pose(quat_params[k], trans_params[k])
        kfs[k].pose = new_Tcw
        if len(world_map.poses) > k:
            world_map.poses[k][:] = new_Tcw

    iters = getattr(summary, "iterations_used",
             getattr(summary, "num_successful_steps",
             getattr(summary, "num_iterations", None)))
    print(f"{info_tag} iters={iters}  χ²={summary.final_cost:.2f}  residuals={added_res}")