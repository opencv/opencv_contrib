# keyframe.py
from dataclasses import dataclass
import cv2
import numpy as np
import lz4.frame
from typing import List, Tuple
from slam.core.features_utils import feature_matcher, filter_matches_ransac

# --------------------------------------------------------------------------- #
#  Dataclass
# --------------------------------------------------------------------------- #
@dataclass
class Keyframe:
    idx:   int                    # global frame index
    frame_idx: int                # actual frame number (0-based), where this KF was created
    path:  str                    # "" for in-memory frames
    kps:   list[cv2.KeyPoint]
    desc:  np.ndarray
    pose: np.ndarray              # 4×4 camera-from-world (T_cw) pose
    thumb: bytes                  # lz4-compressed JPEG


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def make_thumb(bgr, hw=(640, 360)):
    th = cv2.resize(bgr, hw)
    ok, enc = cv2.imencode('.jpg', th,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return lz4.frame.compress(enc.tobytes()) if ok else b''

# keyframe_utils.py

def _rot_deg_from_Tcw(Tcw_prev: np.ndarray, Tcw_curr: np.ndarray) -> float:
    """Angular change between two camera-from-world poses (degrees)."""
    R1, R2 = Tcw_prev[:3, :3], Tcw_curr[:3, :3]
    R = R2 @ R1.T
    # clamp for numerical safety
    cos = max(min((np.trace(R) - 1.0) * 0.5, 1.0), -1.0)
    return float(np.degrees(np.arccos(cos)))

def is_new_keyframe(
    frame_no: int,
    matches_to_kf: list,
    kp_curr: list[cv2.KeyPoint],
    kp_kf: list[cv2.KeyPoint],
    Tcw_prev_kf: np.ndarray | None,
    Tcw_curr: np.ndarray | None,
    *,
    kf_cooldown: int = 5,
    kf_min_inliers: int = 125,
    kf_min_ratio: float = 0.35,
    kf_max_disp: float = 30.0,
    kf_min_rot_deg: float = 8.0,
    last_kf_frame_no: int = -999
) -> bool:
    """Decide whether current frame should be promoted to a keyframe.

    Gate by age, then trigger on any of: weak track, large image motion, or rotation.
    All thresholds are intentionally conservative—tune per dataset.
    """
    # ---- 1) age gate ----
    age = frame_no - last_kf_frame_no
    if age < kf_cooldown:
        return False

    # ---- 2) compute signals ----
    n_inl = len(matches_to_kf)
    n_ref = max(1, len(kp_kf))
    inlier_ratio = n_inl / n_ref

    if n_inl > 0:
        disp = np.hypot(
            np.array([kp_curr[m.trainIdx].pt[0] - kp_kf[m.queryIdx].pt[0] for m in matches_to_kf]),
            np.array([kp_curr[m.trainIdx].pt[1] - kp_kf[m.queryIdx].pt[1] for m in matches_to_kf])
        )
        mean_disp = float(np.mean(disp))
        med_disp  = float(np.median(disp))
    else:
        mean_disp = med_disp = 0.0

    rot_deg = 0.0
    if Tcw_prev_kf is not None and Tcw_curr is not None:
        rot_deg = _rot_deg_from_Tcw(Tcw_prev_kf, Tcw_curr)

    # ---- 3) triggers ----
    weak_track  = (n_inl < kf_min_inliers) or (inlier_ratio < kf_min_ratio)
    large_flow  = (med_disp > kf_max_disp)   # median more robust than mean
    view_change = (rot_deg > kf_min_rot_deg)

    return weak_track or large_flow or view_change

def select_keyframe(
    args,
    seq: List[str],
    frame_idx: int,
    img2, kp2, des2,
    Tcw_curr,
    matcher,
    kfs: List[Keyframe],
    last_kf_frame_no: int
) -> Tuple[List[Keyframe], int]:
    """
    Decide whether to add a new Keyframe at this iteration.

    Parameters
    ----------
    args
        CLI namespace (provides use_lightglue, ransac_thresh, kf_* params).
    seq
        Original sequence list (so we can grab file paths if needed).
    frame_idx
        zero-based index of the *first* of the pair.  Keyframes use i+1 as frame number.
    img2
        BGR image for frame i+1 (for thumbnail if we promote).
    kp2, des2
        KPs/descriptors of frame i+1.
    pose2
        Current camera-to-world pose estimate for frame i+1 (4×4).  May be None.
    matcher
        Either the OpenCV BF/FLANN matcher or the LightGlue matcher.
    kfs
        Current list of Keyframe objects.
    last_kf_frame_no
        Frame number (1-based) of the last keyframe added; or -inf if none.

    Returns
    -------
    kfs
        Possibly-extended list of Keyframe objects.
    last_kf_frame_no
        Updated last keyframe frame number (still the same if we didn’t add one).
    """
    frame_no = frame_idx + 1
    if not kfs:
        return kfs, last_kf_frame_no

    prev_kf = kfs[-1]

    raw_matches = feature_matcher(args, prev_kf.kps, kp2, prev_kf.desc, des2, matcher)
    matches = filter_matches_ransac(prev_kf.kps, kp2, raw_matches, args.ransac_thresh)

    if is_new_keyframe(
        frame_no, matches, kp2, prev_kf.kps,
        Tcw_prev_kf=prev_kf.pose, Tcw_curr=Tcw_curr,
        kf_cooldown=args.kf_cooldown,
        kf_min_inliers=args.kf_min_inliers,
        kf_min_ratio=getattr(args, "kf_min_ratio", 0.35),
        kf_max_disp=args.kf_max_disp,
        kf_min_rot_deg=getattr(args, "kf_min_rot_deg", 8.0),
        last_kf_frame_no=last_kf_frame_no,
    ):
        thumb = make_thumb(img2, tuple(args.kf_thumb_hw))
        path  = seq[frame_idx + 1] if isinstance(seq[frame_idx + 1], str) else ""
        seq_id = len(kfs)
        kfs.append(Keyframe(seq_id, frame_no, path, kp2, des2, Tcw_curr, thumb))
        last_kf_frame_no = frame_no

    return kfs, last_kf_frame_no
