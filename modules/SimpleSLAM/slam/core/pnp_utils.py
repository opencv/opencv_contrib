"""
pnp_utils.py
~~~~~~~~~~~~
Light-weight helpers for

* projecting world landmarks into an image,
* finding 3-D ↔ 2-D correspondences,
* robust PnP pose refinement, and
* optional landmark ↔ key-point bookkeeping.

All functions are Numpy-only apart from OpenCV calls; no extra deps.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Sequence

# --------------------------------------------------------------- #
#  Projection helper
# --------------------------------------------------------------- #
def project_points(
    K: np.ndarray,
    pose_w_c: np.ndarray,          # 4×4 camera-to-world
    pts_w: np.ndarray              # N×3 (world)
) -> np.ndarray:                   # N×2 float32 (pixels)
    """Project *pts_w* into the camera given `pose_w_c`."""
    T_c_w = np.linalg.inv(pose_w_c)                # world → camera
    P = K @ T_c_w[:3, :4]                          # 3×4
    pts_h = np.hstack([pts_w, np.ones((len(pts_w), 1))]).T   # 4×N
    uvw = P @ pts_h                                # 3×N
    return (uvw[:2] / uvw[2]).T.astype(np.float32) # N×2

# --------------------------------------------------------------- #
#  3-D ↔ 2-D association
# --------------------------------------------------------------- #
def associate_landmarks(
    K:            np.ndarray,
    pose_w_c:     np.ndarray,
    pts_w:        np.ndarray,      # N×3  (world coords of map)
    kps:          Sequence[cv2.KeyPoint],
    search_rad:   float = 10.0
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Simple projection-based data association:

    1.  Project every landmark into the current image.
    2.  Take the *closest* key-point within `search_rad` px.
        (no descriptor needed → fast, deterministic)

    Returns
    -------
    pts3d   : M×3  world points that found a match
    pts2d   : M×2  pixel coords of the matched key-points
    kp_idx  : list of length *M* with the matching cv2.KeyPoint indices
    """
    if len(pts_w) == 0 or len(kps) == 0:
        return np.empty((0, 3)), np.empty((0, 2)), []

    proj = project_points(K, pose_w_c, pts_w)      # N×2
    kp_xy = np.float32([kp.pt for kp in kps])      # K×2

    pts3d, pts2d, kp_ids = [], [], []
    used = set()
    for i, (u, v) in enumerate(proj):
        d2 = np.sum((kp_xy - (u, v)) ** 2, axis=1)
        j = np.argmin(d2)
        if d2[j] < search_rad ** 2 and j not in used:
            pts3d.append(pts_w[i])
            pts2d.append(kp_xy[j])
            kp_ids.append(j)
            used.add(j)

    if not pts3d:
        return np.empty((0, 3)), np.empty((0, 2)), []
    return np.float32(pts3d), np.float32(pts2d), kp_ids

# TODO: Try to use EPnP 
# --------------------------------------------------------------- #
#  Robust PnP RANSAC  +  LM refinement
# --------------------------------------------------------------- #
def refine_pose_pnp(
    K:        np.ndarray,
    pts3d:    np.ndarray,          # M×3
    pts2d:    np.ndarray           # M×2
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """AP3P + RANSAC → LM-refined Returns (R, t) that map **world → camera**."""
    if len(pts3d) < 4:
        return None, None

    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        reprojectionError=3.0,
        iterationsCount=100,
        flags=cv2.SOLVEPNP_AP3P
    )
    if not ok or inl is None or len(inl) < 4:
        return None, None

    # final bundle-free local optimisation
    cv2.solvePnPRefineLM(
        pts3d[inl[:, 0]], pts2d[inl[:, 0]],
        K, None, rvec, tvec
    )
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()  # R: 3×3, t: 3×1, # 1-D (3,) instead of (3,1)

# --------------------------------------------------------------- #
#  3-D ↔ 3-D alignment
# --------------------------------------------------------------- #
def align_point_clouds(
    src: np.ndarray,  # N×3
    dst: np.ndarray   # N×3
) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid alignment **src → dst** (no scale).

    Returns rotation ``R`` and translation ``t`` such that

    ``dst ≈ (R @ src.T + t).T``
    """

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src/dst must both be (N,3)")

    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)

    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_dst - R @ centroid_src
    return R, t
