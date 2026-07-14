"""
two_view_bootstrap.py  (verbose, self-contained)
------------------------------------------------
ORB-style two-view bootstrap for monocular SLAM with helpful logging.

What it does:
- Competes Homography (H) vs Fundamental (F) on comparable residuals.
- Recovers a valid (R, t) with cheirality + parallax checks.
- Optionally returns a robust inlier mask aligned with your matches.
- Builds the initial map (KF0 = I, KF1 = [R|t]), triangulates once,
  depth-filters, and adds landmarks + observations.

How to use in main:
1) Use `evaluate_two_view_bootstrap_with_masks` to decide on a pair.
2) Call `bootstrap_two_view_map(...)` to create KF0+KF1 and landmarks.

Author: ChatGPT assistant
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
import numpy as np
import cv2
import logging

from slam.core.pose_utils import _pose_rt_to_homogenous

# ---------------- logging ----------------
logger = logging.getLogger("two_view_bootstrap")
# if not logger.handlers:
#     _h = logging.StreamHandler()
#     _fmt = logging.Formatter("[%(levelname)s] %(name)s:%(funcName)s: %(message)s")
#     _h.setFormatter(_fmt)
#     logger.addHandler(_h)
# logger.setLevel(logging.INFO)  # override from your main if you want DEBUG

# ---------------- data types ----------------

class TwoViewModel(Enum):
    HOMOGRAPHY = auto()
    FUNDAMENTAL = auto()

@dataclass
class InitParams:
    ransac_px: float = 1.5
    chi2_H: float = 5.99       # chi^2 95% for 2 DoF (px^2)
    chi2_F: float = 3.84       # chi^2 95% for 1 DoF (px^2)
    min_pts_for_tests: int = 60
    min_posdepth: float = 0.90
    min_parallax_deg: float = 1.5
    score_ratio_H: float = 0.45  # prefer H if S_H/(S_H+S_F) > 0.45

@dataclass
class TwoViewScores:
    S_H: float
    S_F: float
    ratio_H: float

@dataclass
class TwoViewPose:
    model: TwoViewModel
    R: np.ndarray          # 3x3
    t: np.ndarray          # 3x1
    posdepth: float        # fraction in front of both cams
    parallax_deg: float    # median triangulation angle

@dataclass
class TwoViewDecision:
    pose: TwoViewPose
    inlier_mask: np.ndarray  # boolean mask aligned with pts_ref/pts_cur (N,)

# ---------------- residuals & scoring ----------------

def symmetric_transfer_errors_H(H: np.ndarray,
                                pts_ref: np.ndarray,
                                pts_cur: np.ndarray) -> np.ndarray:
    """Squared symmetric transfer error for a homography H."""
    x1 = cv2.convertPointsToHomogeneous(pts_ref)[:, 0, :].T  # 3xN
    x2 = cv2.convertPointsToHomogeneous(pts_cur)[:, 0, :].T
    Hx1 = H @ x1
    Hinv = np.linalg.inv(H)
    Hinvx2 = Hinv @ x2
    p2 = (Hx1[:2] / (Hx1[2] + 1e-12)).T
    p1 = (Hinvx2[:2] / (Hinvx2[2] + 1e-12)).T
    e12 = np.sum((pts_cur - p2) ** 2, axis=1)
    e21 = np.sum((pts_ref - p1) ** 2, axis=1)
    d2 = e12 + e21
    logger.debug("H symmetric errors: med=%.3f px^2, 75p=%.3f px^2",
                 float(np.median(d2)), float(np.percentile(d2, 75)))
    return d2

def sampson_distances_F(F: np.ndarray,
                        pts_ref: np.ndarray,
                        pts_cur: np.ndarray) -> np.ndarray:
    """Sampson distance for a fundamental matrix F."""
    x1 = cv2.convertPointsToHomogeneous(pts_ref)[:, 0, :]
    x2 = cv2.convertPointsToHomogeneous(pts_cur)[:, 0, :]
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    num = (np.sum(x2 * (F @ x1.T).T, axis=1)) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12
    d2 = num / den
    logger.debug("F Sampson distances: med=%.3f, 75p=%.3f",
                 float(np.median(d2)), float(np.percentile(d2, 75)))
    return d2

def truncated_inlier_score(residuals_sq: np.ndarray, chi2_cutoff: float) -> float:
    """ORB-style truncated linear score: sum(max(0, chi2 - d^2))."""
    S = float(np.maximum(0.0, chi2_cutoff - residuals_sq).sum())
    logger.debug("Truncated score @chi2=%.2f → S=%.1f (inlier-like=%d/%d)",
                 chi2_cutoff, S, int((residuals_sq < chi2_cutoff).sum()), residuals_sq.size)
    return S

def compute_model_scores(H: Optional[np.ndarray],
                         F: Optional[np.ndarray],
                         pts_ref: np.ndarray,
                         pts_cur: np.ndarray,
                         params: InitParams) -> TwoViewScores:
    S_H = truncated_inlier_score(symmetric_transfer_errors_H(H, pts_ref, pts_cur), params.chi2_H) if H is not None else 0.0
    S_F = truncated_inlier_score(sampson_distances_F(F, pts_ref, pts_cur), params.chi2_F)          if F is not None else 0.0
    ratio_H = S_H / (S_H + S_F + 1e-12)
    logger.info("Scores  S_H=%.1f  S_F=%.1f  → ratio_H=%.3f", S_H, S_F, ratio_H)
    return TwoViewScores(S_H=S_H, S_F=S_F, ratio_H=ratio_H)

# ---------------- triangulation validation ----------------

def triangulation_metrics(K: np.ndarray,
                          R: np.ndarray,
                          t: np.ndarray,
                          pts_ref: np.ndarray,
                          pts_cur: np.ndarray) -> Tuple[float, float, int]:
    """Return (posdepth_fraction, median_parallax_deg, N_points_used)."""
    if len(pts_ref) < 2:
        return 0.0, 0.0, 0
    p1n = cv2.undistortPoints(pts_ref.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    p2n = cv2.undistortPoints(pts_cur.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, p1n.T, p2n.T)
    X = (Xh[:3] / (Xh[3] + 1e-12)).T  # Nx3

    z1 = X[:, 2]
    X2 = (R @ X.T + t.reshape(3, 1)).T
    z2 = X2[:, 2]
    posdepth = float(np.mean((z1 > 0) & (z2 > 0)))

    C1 = np.zeros((1, 3))
    C2 = (-R.T @ t).reshape(1, 3)
    v1 = X - C1
    v2 = X - C2
    cosang = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12)
    parallax_deg = float(np.degrees(np.median(np.arccos(np.clip(cosang, -1.0, 1.0)))))

    logger.debug("Triangulation metrics: posdepth=%.3f  parallax_med=%.2f°  N=%d",
                 posdepth, parallax_deg, len(X))
    return posdepth, parallax_deg, len(X)

def validate_two_view_pose(K: np.ndarray,
                           R: np.ndarray,
                           t: np.ndarray,
                           pts_ref: np.ndarray,
                           pts_cur: np.ndarray,
                           params: InitParams) -> Tuple[bool, float, float]:
    posdepth, parallax_deg, N = triangulation_metrics(K, R, t, pts_ref, pts_cur)
    ok = (N >= params.min_pts_for_tests and
          posdepth >= params.min_posdepth and
          parallax_deg >= params.min_parallax_deg)
    logger.info("Validate pose: ok=%s  N=%d  posdepth=%.3f  parallax=%.2f°  (req: N≥%d, pos≥%.2f, par≥%.2f°)",
                ok, N, posdepth, parallax_deg, params.min_pts_for_tests, params.min_posdepth, params.min_parallax_deg)
    return ok, posdepth, parallax_deg

# ---------------- pose recovery for each model ----------------

def recover_pose_from_homography(K: np.ndarray,
                                 H: np.ndarray,
                                 pts_ref: np.ndarray,
                                 pts_cur: np.ndarray,
                                 params: InitParams) -> Optional[TwoViewPose]:
    ret = cv2.decomposeHomographyMat(H, K)
    if ret is None or len(ret) < 4:
        logger.warning("Homography decomposition failed.")
        return None
    _, Rs, ts, _ = ret
    logger.info("Homography decomposition → %d candidates", len(Rs))
    best: Optional[TwoViewPose] = None
    best_key = (-1.0, -1.0)  # maximize (posdepth, parallax)

    for idx, (R, t) in enumerate(zip(Rs, ts)):
        t = t.reshape(3, 1)
        t = t / (np.linalg.norm(t) + 1e-12)  # scale-free
        ok, pd, ang = validate_two_view_pose(K, R, t, pts_ref, pts_cur, params)
        logger.info("  H-cand #%d: ok=%s  posdepth=%.3f  parallax=%.2f°", idx, ok, pd, ang)
        if ok and (pd, ang) > best_key:
            best = TwoViewPose(TwoViewModel.HOMOGRAPHY, R, t, pd, ang)
            best_key = (pd, ang)
    if best is None:
        logger.info("No homography candidate passed validation.")
    else:
        logger.info("Chosen H-candidate: posdepth=%.3f  parallax=%.2f°", best.posdepth, best.parallax_deg)
    return best

def recover_pose_from_fundamental(K: np.ndarray,
                                  F: np.ndarray,
                                  pts_ref: np.ndarray,
                                  pts_cur: np.ndarray,
                                  params: InitParams) -> Optional[TwoViewPose]:
    E = K.T @ F @ K
    ok, R, t, mask = cv2.recoverPose(E, pts_ref, pts_cur, K)
    ninl = int(np.count_nonzero(mask)) if mask is not None else 0
    logger.info("recoverPose(E): ok=%s  inliers=%d", ok, ninl)
    if not ok or mask is None or ninl < params.min_pts_for_tests:
        logger.info("F/E rejected: not enough inliers for validation.")
        return None
    inl = mask.ravel().astype(bool)
    ok2, pd, ang = validate_two_view_pose(K, R, t, pts_ref[inl], pts_cur[inl], params)
    if ok2:
        logger.info("F/E accepted: posdepth=%.3f  parallax=%.2f°", pd, ang)
        return TwoViewPose(TwoViewModel.FUNDAMENTAL, R, t, pd, ang)
    logger.info("F/E rejected after validation.")
    return None

# ---------------- top-level model selection ----------------

def evaluate_two_view_bootstrap(K: np.ndarray,
                                pts_ref: np.ndarray,
                                pts_cur: np.ndarray,
                                params: InitParams = InitParams()
                                ) -> Optional[TwoViewPose]:
    """Pick H vs F with comparable residuals, then recover a valid (R,t)."""
    H, maskH = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, params.ransac_px)
    F, maskF = cv2.findFundamentalMat(pts_ref, pts_cur, cv2.FM_RANSAC, params.ransac_px)

    nH = int(maskH.sum()) if (maskH is not None and maskH.size) else 0
    nF = int(maskF.sum()) if (maskF is not None and maskF.size) else 0
    logger.info("RANSAC: H-inliers=%d (th=%.2f px), F-inliers=%d (th=%.2f px)",
                nH, params.ransac_px, nF, params.ransac_px)

    if H is None and F is None:
        logger.info("Both H and F estimation failed → reject pair.")
        return None

    scores = compute_model_scores(H, F, pts_ref, pts_cur, params)

    if scores.ratio_H > params.score_ratio_H and H is not None:
        logger.info("Model selection: prefer HOMOGRAPHY (ratio_H=%.3f > %.2f)",
                    scores.ratio_H, params.score_ratio_H)
        pose = recover_pose_from_homography(K, H, pts_ref, pts_cur, params)
        if pose is not None:
            return pose
        logger.info("H path failed validation → trying F/E fallback.")
    else:
        logger.info("Model selection: prefer FUNDAMENTAL/E (ratio_H=%.3f ≤ %.2f)",
                    scores.ratio_H, params.score_ratio_H)

    if F is not None:
        pose = recover_pose_from_fundamental(K, F, pts_ref, pts_cur, params)
        if pose is not None:
            return pose

    logger.info("Pair rejected: ambiguous or too weak for initialization.")
    return None

# ---------------- masks for the chosen model ----------------
  
def _final_inlier_mask_for_model(model: TwoViewModel,
                                 pts_ref: np.ndarray,
                                 pts_cur: np.ndarray,
                                 K: np.ndarray,
                                 R: np.ndarray,
                                 t: np.ndarray,
                                 ransac_px: float) -> np.ndarray:
    """
    Make a robust inlier mask aligned with pts_ref/pts_cur for the chosen model.
    F/E: intersect F-RANSAC inliers with recoverPose mask.
    H:   use H-RANSAC inliers.
    """
    def _as_bool(m):
        if m is None:
            return None
        v = np.asarray(m).ravel()
        # treat any nonzero as True (works for 1 and 255)
        return (v.astype(np.uint8) > 0)

    if model is TwoViewModel.FUNDAMENTAL:
        F, maskF = cv2.findFundamentalMat(pts_ref, pts_cur, cv2.FM_RANSAC, ransac_px)
        if F is None or maskF is None:
            return np.zeros(len(pts_ref), dtype=bool)
        E = K.T @ F @ K
        _, _, _, maskRP = cv2.recoverPose(E, pts_ref, pts_cur, K)
        mF  = _as_bool(maskF)
        mRP = _as_bool(maskRP)
        return mF if mRP is None else (mF & mRP)
    else:
        H, maskH = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, ransac_px)
        if H is None or maskH is None:
            return np.zeros(len(pts_ref), dtype=bool)
        return _as_bool(maskH)


def evaluate_two_view_bootstrap_with_masks(K: np.ndarray,
                                           pts_ref: np.ndarray,
                                           pts_cur: np.ndarray,
                                           params: InitParams = InitParams()
                                           ) -> Optional[TwoViewDecision]:
    """Same as evaluate_two_view_bootstrap, but also returns a robust inlier mask."""
    pose = evaluate_two_view_bootstrap(K, pts_ref, pts_cur, params)
    if pose is None:
        return None
    mask = _final_inlier_mask_for_model(pose.model, pts_ref, pts_cur, K, pose.R, pose.t, params.ransac_px)
    return TwoViewDecision(pose=pose, inlier_mask=mask.astype(bool))

# ---------------- map building (non-redundant) ----------------

def _triangulate_points_cv(K: np.ndarray,
                           R: np.ndarray,
                           t: np.ndarray,
                           pts_ref: np.ndarray,
                           pts_cur: np.ndarray) -> np.ndarray:
    """Triangulate in the reference camera frame (world := cam0)."""
    p1n = cv2.undistortPoints(pts_ref.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    p2n = cv2.undistortPoints(pts_cur.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    P1  = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2  = np.hstack([R, t.reshape(3, 1)])
    Xh  = cv2.triangulatePoints(P1, P2, p1n.T, p2n.T)
    X   = (Xh[:3] / (Xh[3] + 1e-12)).T  # Nx3 in cam0/world
    return X

def bootstrap_two_view_map(K: np.ndarray,
                           kp_ref, desc_ref,
                           kp_cur, desc_cur,
                           matches,            # list[cv2.DMatch] between (ref,cur)
                           args,
                           world_map,
                           params: InitParams = InitParams(),
                           decision: Optional[TwoViewDecision] = None):
    """
    Build the initial map from one accepted two-view pair.

    If you already ran the gate, pass its 'decision' to avoid recomputation.
    Otherwise this function will run the gate internally.

    Side effects:
      - Inserts KF0 (Tcw = I) and KF1 (Tcw = [R|t]) into world_map.
      - Triangulates, depth-filters, and adds points + observations.

    Returns: (success: bool, T0_cw: 4x4, T1_cw: 4x4)
    """
    if len(matches) < 50:
        logger.info("[BOOTSTRAP] Not enough matches for init (%d < 50).", len(matches))
        return False, None, None

    # Build float arrays aligned with 'matches'
    pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches])
    pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in matches])

    # 1) Model selection + (R,t) + inlier mask
    if decision is None:
        decision = evaluate_two_view_bootstrap_with_masks(K, pts_ref, pts_cur, params)
        if decision is None:
            logger.info("[BOOTSTRAP] Pair rejected by gate; aborting.")
            return False, None, None
    pose = decision.pose
    mask = decision.inlier_mask.astype(bool)
    ninl = int(mask.sum())
    logger.info("[BOOTSTRAP] Using model=%s with %d inliers.", pose.model.name, ninl)

    if ninl < params.min_pts_for_tests:
        logger.info("[BOOTSTRAP] Too few inliers after gating (%d < %d).", ninl, params.min_pts_for_tests)
        return False, None, None

    # 2) Triangulate once on the final inliers
    p0 = pts_ref[mask]
    p1 = pts_cur[mask]
    Xw = _triangulate_points_cv(K, pose.R, pose.t, p0, p1)  # world := cam0

    # 3) Cheirality + depth range gating (both views)
    z0 = Xw[:, 2]
    X1 = (pose.R @ Xw.T + pose.t.reshape(3, 1)).T
    z1 = X1[:, 2]

    min_d = float(getattr(args, "min_depth", 0.0))
    max_d = float(getattr(args, "max_depth", 1e6))
    ok = (z0 > min_d) & (z0 < max_d) & (z1 > min_d) & (z1 < max_d)
    Xw = Xw[ok]
    logger.info("[BOOTSTRAP] Triangulated=%d  kept=%d after depth filter [%.3g, %.3g].",
                len(p0), len(Xw), min_d, max_d)
    if len(Xw) < 80:
        logger.info("[BOOTSTRAP] Not enough 3D points to seed the map (%d < 80).", len(Xw))
        return False, None, None

    # 4) Insert KF0 (I) and KF1 ([R|t]) exactly once, in this order
    T0_cw = np.eye(4, dtype=np.float64)
    T1_cw = _pose_rt_to_homogenous(pose.R, pose.t)
    # world_map.add_pose(T0_cw, is_keyframe=True)   # KF0
    # world_map.add_pose(T1_cw, is_keyframe=True)   # KF1

    # 5) Add 3D points and observations
    cols = np.full((len(Xw), 3), 0.7, dtype=np.float32)  # grey
    ids  = world_map.add_points(Xw, cols, keyframe_idx=0)

    # map inlier indices back to kp indices
    qidx = np.int32([m.queryIdx for m in matches])
    tidx = np.int32([m.trainIdx  for m in matches])
    sel  = np.where(mask)[0][ok]  # indices into 'matches' of inliers that passed depth

    for pid, i0, i1 in zip(ids, qidx[sel], tidx[sel]):
        world_map.points[pid].add_observation(0, i0, desc_ref[i0])  # KF0
        world_map.points[pid].add_observation(1, i1, desc_cur[i1])  # KF1

    logger.info("[BOOTSTRAP] Map initialised: %d landmarks, 2 keyframes (KF0=I, KF1=[R|t]).", len(ids))
    return True, T0_cw, T1_cw

# ---------------- convenience ----------------

def pts_from_matches(kps_ref, kps_cur, matches):
    pts_ref = np.float32([kps_ref[m.queryIdx].pt for m in matches])
    pts_cur = np.float32([kps_cur[m.trainIdx].pt for m in matches])
    return pts_ref, pts_cur
