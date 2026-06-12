# slam/core/pnp_utils.py
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
from scipy.spatial import cKDTree

log = logging.getLogger("pnp")

# ---------- small utilities ----------

def _pose_inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def _pose_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def predict_pose_const_vel(Tcw_prevprev: np.ndarray, Tcw_prev: np.ndarray) -> np.ndarray:
    """Constant-velocity on SE(3) for T_cw (camera-from-world).
       T_pred = T_prev * inv(T_prevprev) * T_prev
    """
    return Tcw_prev @ _pose_inverse(Tcw_prevprev) @ Tcw_prev

def project_point(K: np.ndarray, T_cw: np.ndarray, Xw: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (u,v) and depth z in the current camera."""
    Xc = T_cw[:3, :3] @ Xw + T_cw[:3, 3]
    z = float(Xc[2])
    if z <= 1e-8:
        return np.array([-1.0, -1.0], dtype=np.float32), z
    uv = (K @ (Xc / z))[:2]
    return uv.astype(np.float32), z

def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    # ORB/BRIEF descriptors: uint8 row vectors
    return int(cv2.norm(a, b, cv2.NORM_HAMMING))

def _choose_mp_descriptor(observations: List[Tuple[int, int, np.ndarray]]) -> Optional[np.ndarray]:
    if not observations:
        return None
    # use most-recent observation's descriptor
    return observations[-1][2]

@dataclass
class Matches2D3D:
    pts3d: np.ndarray          # (N,3) world
    pts2d: np.ndarray          # (N,2) image
    kp_indices: List[int]      # indices into current frame keypoints
    mp_ids:   List[int]        # matched map point ids


def _as_np1d(x):
    if x is None:
        return None
    a = np.asarray(x)
    return a.reshape(-1)

def _kp_coords(kps):
    """Accepts List[cv2.KeyPoint] or ndarray (N,2). Returns float32 (N,2)."""
    if isinstance(kps, (list, tuple)) and len(kps) > 0 and hasattr(kps[0], "pt"):
        return np.float32([kp.pt for kp in kps])
    kps = np.asarray(kps)
    assert kps.ndim == 2 and kps.shape[1] >= 2, "kps must be (N,2)"
    return kps[:, :2].astype(np.float32, copy=False)

def _desc_distance(a, b, metric="auto"):
    """
    Robust descriptor distance that supports:
    - binary (uint8) -> Hamming
    - float (float32/64) -> L2 (default) or cosine if metric='cosine'
    """
    a = _as_np1d(a); b = _as_np1d(b)
    if a is None or b is None:
        return np.inf
    if a.shape[0] != b.shape[0]:
        return np.inf

    if metric == "auto":
        metric = "hamming" if a.dtype == np.uint8 and b.dtype == np.uint8 else "l2"

    if metric == "hamming":
        # ensure uint8 contiguous
        a = a.astype(np.uint8, copy=False)
        b = b.astype(np.uint8, copy=False)
        return float(cv2.norm(a, b, cv2.NORM_HAMMING))

    if metric == "cosine":
        a = a.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return np.inf
        # convert similarity -> distance in [0,2]
        cos_sim = float(np.dot(a, b) / (na * nb))
        return 1.0 - cos_sim

    # default L2
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    return float(np.linalg.norm(a - b))

# TODO Optimal way would be to add a parameter to class map_point that keeps a running average of descriptor(Also know as reference descriptor)
def _best_mp_distance_to_cur_desc(mp, d_cur, max_obs_check: int = 6) -> float:
    """Return the smallest distance between the current descriptor and any
    of the map point's stored observation descriptors (check last N)."""
    if not mp.observations:
        return float("inf")
    # check only the most recent few to keep it fast
    obs_iter = mp.observations[-max_obs_check:]
    best = float("inf")
    for _, _, d in obs_iter:
        if d is None:
            continue
        best = min(best, _desc_distance(d, d_cur, metric="auto"))
    return best


def reproject_and_match_2d3d(world_map,
                              K: np.ndarray,
                              Tcw_pred: np.ndarray,
                              kps_cur,                 # list[cv2.KeyPoint] OR ndarray (N,2)
                              des_cur: np.ndarray,     # ORB: uint8; ALIKE: float32
                              img_w: int, img_h: int,
                              radius_px: float = 12.0,
                              max_hamm: int = 64,
                              max_l2: float = 0.8, # TODO Magic Variable
                              use_cosine: bool = False):
    """
    Reproject landmarks and match within a window around projections.
    Works with ORB/BRIEF (uint8) and ALIKE/LightGlue-style float descriptors.
    """
    # bail early if no features/descriptors
    if des_cur is None or (hasattr(des_cur, "size") and des_cur.size == 0):
        return Matches2D3D(np.zeros((0,3),np.float32), np.zeros((0,2),np.float32), [], [])
    if not world_map.points:
        return Matches2D3D(np.zeros((0,3),np.float32), np.zeros((0,2),np.float32), [], [])

    pts2d_cur = _kp_coords(kps_cur)
    tree = cKDTree(pts2d_cur)

    used_kps = set()
    pts3d, pts2d, kpids, mpids = [], [], [], []

    # pick metric once based on current descriptor dtype
    cur_is_binary = (des_cur.dtype == np.uint8)
    metric = "hamming" if cur_is_binary else ("cosine" if use_cosine else "l2")
    thr = max_hamm if metric == "hamming" else max_l2

    for mp_id, mp in world_map.points.items():
        # project
        uv, z = project_point(K, Tcw_pred, mp.position)
        if z <= 0:
            continue
        u, v = float(uv[0]), float(uv[1])
        if u < 0 or v < 0 or u >= img_w or v >= img_h:
            continue

        # small-window search
        cand_idx = tree.query_ball_point([u, v], r=radius_px)
        if not cand_idx:
            continue

        # descriptor of this map point (use most recent obs)
        mp_desc = _choose_mp_descriptor(mp.observations)
        if mp_desc is None:
            continue

        best_i, best_d = -1, 1e9
        for i in cand_idx:
            if i in used_kps:
                continue
            d = _best_mp_distance_to_cur_desc(mp, des_cur[i], max_obs_check=6)
            if d < best_d:
                best_d, best_i = d, i

        if best_i < 0 or best_d > thr:
            continue

        used_kps.add(best_i)
        pts3d.append(mp.position.astype(np.float32))
        pts2d.append(pts2d_cur[best_i])
        kpids.append(best_i)
        mpids.append(mp_id)

    if not pts3d:
        return Matches2D3D(np.zeros((0,3),np.float32), np.zeros((0,2),np.float32), [], [])

    return Matches2D3D(
        np.asarray(pts3d, np.float32),
        np.asarray(pts2d, np.float32),
        kpids, mpids
    )


def solve_pnp_ransac(pts3d: np.ndarray,
                     pts2d: np.ndarray,
                     K: np.ndarray,
                     ransac_px: float,
                     Tcw_init: Optional[np.ndarray] = None,
                     iters: int = 200,
                     conf: float = 0.999) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Return (T_cw, inlier_mask) or (None, empty)"""
    if len(pts3d) < 4:
        return None, np.zeros((0,), dtype=bool)

    use_guess = Tcw_init is not None
    rvec0, tvec0 = None, None
    if use_guess:
        R0, t0 = Tcw_init[:3, :3], Tcw_init[:3, 3]
        rvec0, _ = cv2.Rodrigues(R0)
        tvec0 = t0.reshape(3, 1)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        useExtrinsicGuess=bool(use_guess),
        iterationsCount=int(iters),
        reprojectionError=float(ransac_px),
        confidence=float(conf),
        flags=cv2.SOLVEPNP_ITERATIVE,
        rvec=rvec0, tvec=tvec0
    )
    if not ok or inliers is None or len(inliers) < 4:
        return None, np.zeros((len(pts3d),), dtype=bool)

    R, _ = cv2.Rodrigues(rvec)
    Tcw = _pose_from_Rt(R, tvec.reshape(3))
    mask = np.zeros((len(pts3d),), dtype=bool)
    mask[inliers.ravel().astype(int)] = True
    return Tcw, mask

# ---------- optional: tiny debug overlay ----------

def draw_reprojection_debug(img_bgr: np.ndarray,
                            K: np.ndarray,
                            Tcw: np.ndarray,
                            matches: Matches2D3D,
                            inlier_mask: np.ndarray) -> np.ndarray:
    vis = img_bgr.copy()
    for (Xw, uv, ok) in zip(matches.pts3d, matches.pts2d, inlier_mask):
        # draw observed point
        cv2.circle(vis, tuple(np.int32(uv)), 3, (0, 255, 0) if ok else (0, 0, 255), -1)
        # draw reprojected point from estimated pose
        uvp, z = project_point(K, Tcw, Xw.astype(np.float64))
        if z > 0:
            cv2.circle(vis, tuple(np.int32(uvp)), 2, (255, 255, 0), 1)
            cv2.line(vis, tuple(np.int32(uv)), tuple(np.int32(uvp)), (255, 255, 0), 1)
    return vis
