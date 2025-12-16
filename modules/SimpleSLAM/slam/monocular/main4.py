# main.py
"""
Monocular VO/SLAM main with:
  • PnP tracking + Multi-View Triangulation
  • Frame-by-frame track overlay
  • NEW: Last-3 Keyframes triptych (OpenCV window, reliable)
      - New landmarks (just triangulated) = GREEN dots
      - Cross-KF polyline turns RED once a landmark is seen in ≥ --track_maturity KFs
      - Per-panel overlay: "KF #i | features: N"
  • Optional 2-D Matplotlib trajectory plotter (stable refresh)

Notes:
  - Keyframe thumbnails in your Keyframe dataclass are lz4-compressed JPEG bytes.
    We now decode those safely for visualization.  (see _decode_thumb)
"""

import argparse
from copy import deepcopy
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Iterable, Set

import lz4.frame  # for decoding KF thumbnails

from slam.core.pose_utils import _pose_inverse, _pose_rt_to_homogenous

from slam.core.dataloader import (
    load_sequence,
    load_frame_pair,
    load_calibration,
    load_groundtruth,
)

from slam.core.features_utils import (
    init_feature_pipeline,
    feature_extractor,
    feature_matcher,
    filter_matches_ransac,
)

from slam.core.keyframe_utils import (
    Keyframe,
    select_keyframe,
    make_thumb,   # retained, but we keep our own BGR thumbs for GUI
)

from slam.core.visualization_utils import (
    draw_tracks,
    Visualizer3D,
    TrajectoryPlotter,
)

from slam.core.trajectory_utils import compute_gt_alignment
from slam.core.landmark_utils import Map
from slam.core.triangulation_utils import (
    update_and_prune_tracks,
    MultiViewTriangulator,
    triangulate_points,
)
from slam.core.pnp_utils import refine_pose_pnp
from slam.core.ba_utils import (
    two_view_ba,
    pose_only_ba,
    local_bundle_adjustment,
)

# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Feature tracking with key-frames")
    p.add_argument('--dataset',
                   choices=['kitti', 'malaga', 'tum-rgbd', 'custom'],
                   required=True)
    p.add_argument('--base_dir', default='../Dataset')
    # feature/detector settings
    p.add_argument('--detector', choices=['orb', 'sift', 'akaze'],
                   default='orb')
    p.add_argument('--matcher', choices=['bf'], default='bf')
    p.add_argument('--use_lightglue', action='store_true')
    p.add_argument('--min_conf', type=float, default=0.7,
                   help='Minimum LightGlue confidence for a match')
    # runtime
    p.add_argument('--fps', type=float, default=10)
    # RANSAC
    p.add_argument('--ransac_thresh', type=float, default=3.0)
    # key-frame params
    p.add_argument('--kf_max_disp', type=float, default=45)
    p.add_argument('--kf_min_inliers', type=float, default=150)
    p.add_argument('--kf_cooldown', type=int, default=5)
    p.add_argument('--kf_thumb_hw', type=int, nargs=2, default=[640, 360])

    # visualization toggles
    p.add_argument("--no_viz3d", action="store_true", help="Disable 3-D visualization window")
    p.add_argument('--no_plot2d', action='store_true', help='Disable the Matplotlib 2-D trajectory plotter')

    # triangulation depth filtering
    p.add_argument("--min_depth", type=float, default=0.60)
    p.add_argument("--max_depth", type=float, default=50.0)
    p.add_argument('--mvt_rep_err', type=float, default=30.0,
                   help='Max mean reprojection error (px) for multi-view triangulation')

    #  PnP / map-maintenance
    p.add_argument('--pnp_min_inliers', type=int, default=20)
    p.add_argument('--proj_radius',     type=float, default=12.0)
    p.add_argument('--merge_radius',    type=float, default=0.10)

    # Bundle Adjustment
    p.add_argument('--local_ba_window', type=int, default=6, help='Window size (number of keyframes) for local BA')

    # NEW: track maturity + triptych toggle
    p.add_argument('--track_maturity', type=int, default=6,
                   help='#KFs after which a track is rendered in RED')
    p.add_argument('--no_triptych', action='store_true',
                   help='Disable the last-3-keyframes triptych overlay')

    return p


# --------------------------------------------------------------------------- #
#  Bootstrap initialisation
# --------------------------------------------------------------------------- #
def try_bootstrap(K, kp0, descs0, kp1, descs1, matches, args, world_map):
    """Return (success, T1_wc) and add initial landmarks."""
    if len(matches) < 50:
        return False, None

    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

    E, inl_E = cv2.findEssentialMat(
        pts0, pts1, K, cv2.RANSAC, 0.999, args.ransac_thresh)
    H, inl_H = cv2.findHomography(pts0, pts1, cv2.RANSAC, args.ransac_thresh)

    use_E = (inl_E is not None and inl_H is not None and inl_E.sum() > inl_H.sum()) or (inl_E is not None and H is None)
    if use_E and E is None:
        return False, None
    if not use_E and H is None:
        return False, None

    if use_E:
        _, R, t, inl_pose = cv2.recoverPose(E, pts0, pts1, K)
        mask = (inl_E.ravel() & inl_pose.ravel()).astype(bool)
    else:
        print("[BOOTSTRAP] Using Homography for initialisation")
        R, t = cv2.decomposeHomographyMat(H, K)[1:3]
        mask = inl_H.ravel().astype(bool)

    p0 = pts0[mask]
    p1 = pts1[mask]
    pts3d = triangulate_points(K, R, t, p0, p1)

    z0 = pts3d[:, 2]
    pts3d_cam1 = (R @ pts3d.T + t.reshape(3, 1)).T
    z1 = pts3d_cam1[:, 2]

    ok = ((z0 > args.min_depth) & (z0 < args.max_depth) &
          (z1 > args.min_depth) & (z1 < args.max_depth))
    pts3d = pts3d[ok]
    print(f"[BOOTSTRAP] Triangulated {len(pts3d)} points. Inliers after depth: {ok.sum()}")

    if len(pts3d) < 80:
        print("[BOOTSTRAP] Not enough points to bootstrap the map.")
        return False, None

    T1_cw = _pose_rt_to_homogenous(R, t)  # camera-from-world
    T1_wc = _pose_inverse(T1_cw)          # world-from-camera

    world_map.add_pose(T1_wc, is_keyframe=True)

    cols = np.full((len(pts3d), 3), 0.7)
    ids = world_map.add_points(pts3d, cols, keyframe_idx=0)

    inlier_kp_idx = np.where(mask)[0][ok]
    qidx = np.int32([m.queryIdx for m in matches])
    tidx = np.int32([m.trainIdx for m in matches])
    sel  = np.where(mask)[0][ok]

    for pid, i0, i1 in zip(ids, qidx[sel], tidx[sel]):
        world_map.points[pid].add_observation(0, i0, descs0[i0])
        world_map.points[pid].add_observation(1, i1, descs1[i1])

    print(f"[BOOTSTRAP] Map initialised with {len(ids)} landmarks.")
    return True, T1_wc


# --------------------------------------------------------------------------- #
#  PnP + reprojection debug
# --------------------------------------------------------------------------- #
def visualize_pnp_reprojection(img_bgr, K, T_wc, pts3d_w, pts2d_px, inlier_mask=None,
                               win_name="PnP debug", thickness=2):
    img = img_bgr.copy()
    T_cw = np.linalg.inv(T_wc)
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cw)
    tvec = t_cw.reshape(3, 1)

    proj, _ = cv2.projectPoints(pts3d_w.astype(np.float32), rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)

    if inlier_mask is None:
        inlier_mask = np.ones(len(proj), dtype=bool)

    for (u_meas, v_meas), (u_proj, v_proj), ok in zip(pts2d_px, proj, inlier_mask):
        color = (0, 255, 0) if ok else (0, 0, 255)  # green inlier, red outlier
        cv2.circle(img, (int(round(u_proj)), int(round(v_proj))), 4, (255, 0, 0), -1)
        cv2.circle(img, (int(round(u_meas)), int(round(v_meas))), 4, color, -1)
        if ok:
            cv2.line(img,
                     (int(round(u_meas)), int(round(v_meas))),
                     (int(round(u_proj)), int(round(v_proj))),
                     color, thickness)

    cv2.imshow(win_name, img)
    cv2.waitKey(1)
    return img


def track_with_pnp(K,
                   kp_prev, kp_cur, desc_prev, desc_cur, matches, img2,
                   frame_no,
                   Twc_prev,
                   world_map, args):
    if len(world_map.points) < 4 or len(kp_cur) == 0:
        print(f"[PNP] Not enough data at frame {frame_no} "
              f"(points={len(world_map.points)}, kps={len(kp_cur)})")
        return False, None, set()

    mp_ids = world_map.point_ids()
    pts_w  = world_map.get_point_array().astype(np.float32)
    if len(mp_ids) != len(pts_w):
        print("[PNP] Map containers out of sync; skipping frame.")
        return False, None, set()

    from slam.core.pnp_utils import project_points
    proj_px = project_points(K, Twc_prev, pts_w)
    kp_xy   = np.float32([kp.pt for kp in kp_cur])

    def _associate(search_rad_px: float):
        used_kp = set()
        obj_pts, img_pts = [], []
        obj_pids, kp_ids = [], []
        r2 = search_rad_px * search_rad_px
        for i, (u, v) in enumerate(proj_px):
            d2 = np.sum((kp_xy - (u, v))**2, axis=1)
            j  = int(np.argmin(d2))
            if d2[j] < r2 and j not in used_kp:
                obj_pts.append(pts_w[i])
                img_pts.append(kp_xy[j])
                obj_pids.append(mp_ids[i])
                kp_ids.append(j)
                used_kp.add(j)
        if len(obj_pts) == 0:
            return (np.empty((0, 3), np.float32), np.empty((0, 2), np.float32),
                    [], [])
        return (np.asarray(obj_pts, np.float32),
                np.asarray(img_pts, np.float32),
                obj_pids, kp_ids)

    pts3d, pts2d, obj_pids, kp_ids = _associate(args.proj_radius)
    if len(pts3d) < max(4, args.pnp_min_inliers // 2):
        pts3d, pts2d, obj_pids, kp_ids = _associate(args.proj_radius * 1.5)
    if len(pts3d) < 4:
        print(f"[PNP] Not enough 2D–3D correspondences (found {len(pts3d)}) at frame {frame_no}")
        return False, None, set()

    R_cw, tvec = refine_pose_pnp(K, pts3d, pts2d)
    if R_cw is None:
        print(f"[PNP] RANSAC/LM failed at frame {frame_no}")
        return False, None, set()
    T_cw = _pose_rt_to_homogenous(R_cw, tvec)
    Twc_cur = _pose_inverse(T_cw)

    rvec, _ = cv2.Rodrigues(R_cw)
    proj_refined, _ = cv2.projectPoints(pts3d.astype(np.float32), rvec, tvec, K, None)
    proj_refined = proj_refined.reshape(-1, 2)
    reproj_err = np.linalg.norm(proj_refined - pts2d, axis=1)
    inlier_mask = reproj_err < float(3)
    num_inl = int(np.count_nonzero(inlier_mask))
    if num_inl < args.pnp_min_inliers:
        print(f"[PNP] Too few inliers after refine ({num_inl}<{args.pnp_min_inliers}) at frame {frame_no}")
        return False, None, set()

    print(f"[PNP] Pose @ frame {frame_no} refined with {num_inl} inliers")

    visualize_pnp_reprojection(
        img2, K, Twc_cur,
        pts3d_w=pts3d[inlier_mask],
        pts2d_px=pts2d[inlier_mask],
        inlier_mask=np.ones(num_inl, dtype=bool),
        win_name="PnP reprojection"
    )
    return True, Twc_cur, (obj_pids, kp_ids, inlier_mask)


# --------------------------------------------------------------------------- #
#  NEW — Keyframe track visualizations
# --------------------------------------------------------------------------- #
def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def _put_text_with_outline(img: np.ndarray, text: str, org: Tuple[int, int],
                           scale: float = 0.7, color=(255, 255, 255),
                           thickness: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def _decode_thumb(thumb_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode lz4-compressed JPEG bytes (Keyframe.thumb) into a BGR image.
    """
    if not isinstance(thumb_bytes, (bytes, bytearray, memoryview)):
        return None
    try:
        raw = lz4.frame.decompress(thumb_bytes)   # JPEG bytes
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def _collect_obs_for_kf(world_map: Map, kf_idx: int) -> Dict[int, int]:
    """Return {pid -> kp_idx} for all points that have an observation in keyframe `kf_idx`."""
    out: Dict[int, int] = {}
    for pid, mp in world_map.points.items():
        for fidx, kpi, _ in mp.observations:
            if fidx == kf_idx:
                out[pid] = kpi
                break
    return out  # observations are stored as (frame_idx, kp_idx, descriptor). :contentReference[oaicite:2]{index=2}

def _track_length_in_keyframes(world_map: Map, pid: int) -> int:
    if pid not in world_map.points:
        return 0
    kf_set = {f for (f, _, _) in world_map.points[pid].observations}
    return len(kf_set)

def render_kf_triptych(
    kf_triplet: List[Tuple[int, np.ndarray, List[cv2.KeyPoint]]],
    world_map: Map,
    *,
    maturity_thresh: int = 6,
    new_ids: Optional[Iterable[int]] = None,
    title: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Render last-3 keyframes side-by-side, overlaying features and cross-KF tracks.

    kf_triplet: [(kf_pose_idx, bgr_image, keypoints), ...]  len = 1..3
    """
    if not kf_triplet:
        return None

    new_ids_set: Set[int] = set(new_ids) if new_ids is not None else set()

    # Normalise sizes (use first as reference)
    H0, W0 = kf_triplet[0][1].shape[:2]
    panels = []
    for kf_idx, img, _ in kf_triplet:
        img = _ensure_bgr(img)
        if img is None:
            continue
        if img.shape[:2] != (H0, W0):
            img = cv2.resize(img, (W0, H0), interpolation=cv2.INTER_AREA)
        panels.append((kf_idx, img.copy()))
    if not panels:
        return None

    # Precompute observations for each KF
    obs_by_kf: Dict[int, Dict[int, int]] = {kf: _collect_obs_for_kf(world_map, kf) for kf, _ in panels}

    # Draw points on each panel
    for (kf_idx, img), (_, _, kps) in zip(panels, kf_triplet):
        obs = obs_by_kf.get(kf_idx, {})
        feat_count = len(obs)
        _put_text_with_outline(img, f"KF #{kf_idx} | features: {feat_count}", (10, 24), 0.8, (255, 255, 255), 2)
        for pid, kp_idx in obs.items():
            if kp_idx < 0 or kp_idx >= len(kps):  # defensive
                continue
            x, y = map(int, np.round(kps[kp_idx].pt))
            color = (0, 255, 0) if pid in new_ids_set else (220, 220, 220)
            r = 4 if pid in new_ids_set else 3
            cv2.circle(img, (x, y), r, color, -1)

    # Build canvas and paste panels
    widths = [p[1].shape[1] for p in panels]
    height = H0
    canvas = np.zeros((height, sum(widths), 3), dtype=np.uint8)
    x0 = 0
    x_offsets = []
    for (_, img) in panels:
        w = img.shape[1]
        canvas[:, x0:x0+w] = img
        x_offsets.append(x0)
        x0 += w

    # Draw cross-KF polylines for landmarks that appear in ≥2 of the panels
    track_map: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {}
    for pidx, (kf_idx, _img) in enumerate(panels):
        obs = obs_by_kf.get(kf_idx, {})
        kps = kf_triplet[pidx][2]
        for pid, kp_idx in obs.items():
            if kp_idx < 0 or kp_idx >= len(kps):
                continue
            pt = tuple(map(int, np.round(kps[kp_idx].pt)))
            track_map.setdefault(pid, []).append((pidx, pt))

    for pid, samples in track_map.items():
        if len(samples) < 2:
            continue
        samples.sort(key=lambda t: t[0])
        pts = []
        for pidx, (x, y) in samples:
            pts.append([x + x_offsets[pidx], y])
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        L = _track_length_in_keyframes(world_map, pid)
        colour = (0, 0, 255) if L >= maturity_thresh else (0, 255, 0)
        cv2.polylines(canvas, [pts], isClosed=False, color=colour, thickness=2, lineType=cv2.LINE_AA)

    if title:
        _put_text_with_outline(canvas, title, (10, height - 10), 0.8, (255, 255, 255), 2)

    return canvas


# --------------------------------------------------------------------------- #
#  Main processing loop
# --------------------------------------------------------------------------- #
def main():
    args = _build_parser().parse_args()

    # --- Data loading ---
    seq = load_sequence(args)
    calib       = load_calibration(args)
    groundtruth = load_groundtruth(args)
    K = calib["K_l"]
    P = calib["P_l"]

    # ------ build 4×4 GT poses + alignment matrix (once) ----------------
    gt_T = None
    if groundtruth is not None:
        gt_T = np.pad(groundtruth, ((0, 0), (0, 1), (0, 0)), constant_values=0.0)
        gt_T[:, 3, 3] = 1.0
        _ = compute_gt_alignment(gt_T)  # retained if you want aligned GT later

    # --- feature pipeline ---
    detector, matcher = init_feature_pipeline(args)

    mvt = MultiViewTriangulator(
        K,
        min_views=3,
        merge_radius=args.merge_radius,
        max_rep_err=args.mvt_rep_err,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )

    # --- tracking state ---
    prev_map, tracks = {}, {}
    next_track_id = 0
    initialised = False

    world_map = Map()
    Twc_cur_pose = np.eye(4)
    world_map.add_pose(Twc_cur_pose, is_keyframe=True)
    viz3d = None if args.no_viz3d else Visualizer3D(color_axis="y")

    plot2d = None if args.no_plot2d else TrajectoryPlotter()   # Matplotlib plotter window. :contentReference[oaicite:3]{index=3}

    kfs: list[Keyframe] = []
    last_kf_frame_no = -999
    frame_keypoints: List[List[cv2.KeyPoint]] = []
    frame_keypoints.append([])

    total = len(seq) - 1

    # For triptych: map[kf_pose_idx] -> (thumb_bgr, kps)
    kf_vis: Dict[int, Tuple[np.ndarray, List[cv2.KeyPoint]]] = {}

    new_ids: List[int] = []  # new landmarks added at the most recent KF

    # Prepare a named, resizable window for the triptych (stable)
    TRIP_WIN = "Keyframes (last 3) — feature tracks"
    if not args.no_triptych:
        cv2.namedWindow(TRIP_WIN, cv2.WINDOW_NORMAL)

    for i in tqdm(range(total), desc='Tracking'):
        img1, img2 = load_frame_pair(args, seq, i)

        # --- feature extraction / matching ---
        kp1, des1 = feature_extractor(args, img1, detector)
        kp2, des2 = feature_extractor(args, img2, detector)
        matches = feature_matcher(args, kp1, kp2, des1, des2, matcher)
        matches = filter_matches_ransac(kp1, kp2, matches, args.ransac_thresh)

        if len(matches) < 12:
            print(f"[WARN] Not enough matches at frame {i}. Skipping.")
            continue

        # ---------------- Frame-by-frame track overlay ----------------
        frame_no = i + 1
        curr_map, tracks, next_track_id = update_and_prune_tracks(
            matches, prev_map, tracks, kp2, frame_no, next_track_id)
        prev_map = curr_map

        vis_img = img2.copy()
        draw_tracks(vis_img, tracks, frame_no, max_age=15, sample_rate=3, max_tracks=300)
        cv2.imshow("Frame tracks (age-coloured)", vis_img)

        # ---------------- Key-frame decision ----------------
        if i == 0:
            kfs.append(Keyframe(idx=0, frame_idx=1,
                                path=seq[0] if isinstance(seq[0], str) else "",
                                kps=kp1, desc=des1,
                                pose=Twc_cur_pose,
                                thumb=make_thumb(img1, tuple(args.kf_thumb_hw))))
            last_kf_frame_no = kfs[-1].frame_idx
            is_kf = False
            continue
        else:
            prev_len = len(kfs)
            kfs, last_kf_frame_no = select_keyframe(
                args, seq, i, img2, kp2, des2, Twc_cur_pose, matcher, kfs, last_kf_frame_no)
            is_kf = len(kfs) > prev_len

        if is_kf:
            frame_keypoints.append(kp2.copy())

        # ---------------- Bootstrap ----------------
        if not initialised:
            if len(kfs) < 2:
                continue
            bootstrap_matches = feature_matcher(args, kfs[0].kps, kfs[-1].kps,
                                                kfs[0].desc, kfs[-1].desc, matcher)
            bootstrap_matches = filter_matches_ransac(kfs[0].kps, kfs[-1].kps,
                                                      bootstrap_matches, args.ransac_thresh)
            ok, Twc_temp_pose = try_bootstrap(
                K, kfs[0].kps, kfs[0].desc, kfs[-1].kps, kfs[-1].desc, bootstrap_matches, args, world_map)
            if ok:
                frame_keypoints[0] = kfs[0].kps.copy()
                frame_keypoints[-1] = kfs[-1].kps.copy()
                initialised = True
                Twc_cur_pose = world_map.poses[-1].copy()
                continue
            else:
                print("****************** BOOTSTRAP FAILED **************")
                continue

        # ---------------- Tracking (PnP) ----------------
        Twc_pose_prev = Twc_cur_pose.copy()
        ok_pnp, Twc_cur_pose, assoc = track_with_pnp(
            K, kp1, kp2, des1, des2, matches,
            frame_no=i + 1, img2=img2,
            Twc_prev=Twc_pose_prev,
            world_map=world_map,
            args=args
        )

        # Fallback to 2-D-2-D only if PnP failed
        if not ok_pnp:
            print(f"[WARN] PnP failed at frame {i}. Using 2D-2D tracking.")
            last_kf = kfs[-1] if not is_kf else (kfs[-2] if len(kfs) > 1 else kfs[0])

            tracking_matches = feature_matcher(args, last_kf.kps, kp2, last_kf.desc, des2, matcher)
            tracking_matches = filter_matches_ransac(last_kf.kps, kp2, tracking_matches, args.ransac_thresh)

            pts0 = np.float32([last_kf.kps[m.queryIdx].pt for m in tracking_matches])
            pts1 = np.float32([kp2[m.trainIdx].pt  for m in tracking_matches])
            E, mask = cv2.findEssentialMat(pts0, pts1, K, cv2.RANSAC, 0.999, args.ransac_thresh)
            if E is None:
                continue
            _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K)
            T_rel = _pose_rt_to_homogenous(R, t)
            Twc_cur_pose = last_kf.pose @ np.linalg.inv(T_rel)

        if is_kf:
            world_map.add_pose(Twc_cur_pose, is_keyframe=True)

            # Book-keep PnP inliers as observations on the new KF
            if ok_pnp and assoc is not None:
                obj_pids, kp_ids, inlier_mask = assoc
                kf_idx = len(world_map.poses) - 1   # index of newly added KF
                for ok_m, pid, kp_idx in zip(inlier_mask, obj_pids, kp_ids):
                    if ok_m and pid in world_map.points:
                        world_map.points[pid].add_observation(kf_idx, kp_idx, des2[kp_idx])

        # ---------------- Map Growth via Multi-View Triangulation -------------
        if is_kf:
            kf_pose_idx = len(world_map.poses) - 1

            # Build a BGR thumbnail for GUI (avoid bytes-in-GUI issues)
            w, h = tuple(args.kf_thumb_hw)  # [W, H] by convention in your code. :contentReference[oaicite:4]{index=4}
            thumb_bgr = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
            kf_vis[kf_pose_idx] = (thumb_bgr, kp2)

            # (Still store compressed thumb in Keyframe via select_keyframe)
            mvt.add_keyframe(
                frame_idx=kf_pose_idx,
                Twc_pose=Twc_cur_pose,
                kps=kp2,
                track_map=curr_map,
                img_bgr=img2,
                descriptors=des2
            )
            new_mvt_ids = mvt.triangulate_ready_tracks(world_map)
            new_ids = list(new_mvt_ids)

        # ---------------- Local Bundle Adjustment (optional) -------------------
        if is_kf and (len(kfs) % args.local_ba_window == 0):
            center_kf_idx = len(world_map.poses) - 1
            print(f"[BA] Running local BA around key-frame {center_kf_idx} (window size = {args.local_ba_window})")
            _ = deepcopy(world_map)
            local_bundle_adjustment(
                world_map, K, frame_keypoints,
                center_kf_idx=center_kf_idx,
                window_size=args.local_ba_window)

        # ---------------- 2D path plot (Matplotlib) -------------------
        def _scale_translation(pos3: np.ndarray, scale: float) -> np.ndarray:
            """
            Scale a 3D *position* (translation) by 'scale' for plotting.
            NOTE: Do NOT multiply the whole 4x4 pose by a scalar—only scale the translation component.
            """
            return np.asarray(pos3, dtype=float) * float(scale)
        
        if plot2d is not None:
            est_pos = Twc_cur_pose[:3, 3]
            gt_pos  = None
            if gt_T is not None and i + 1 < len(gt_T):
                p_gt = gt_T[i + 1, :3, 3]                     # raw GT
                # gt_pos = apply_alignment(p_gt, R_align, t_align)
                gt_pos = _scale_translation(p_gt, 0.3)
            plot2d.append(est_pos, gt_pos, mirror_x=False)
            # tiny event loop tick for reliable redraw across backends
            import matplotlib.pyplot as _plt
            _plt.pause(0.001)  # <-- prevents the “small black rectangle” issue in many setups. :contentReference[oaicite:5]{index=5}

        # ---------------- 3D viz (Open3D) ----------------------
        if viz3d is not None:
            viz3d.update(world_map, new_ids)

        # ---------------- NEW: "last 3 KFs" triptych -----------
        if is_kf and not args.no_triptych:
            last_kf_indices = sorted(kf_vis.keys())[-3:]
            triplet = []
            for idx in last_kf_indices:
                thumb, kps = kf_vis[idx]
                # defensive: if thumb somehow missing, try decoding stored Keyframe.thumb
                if thumb is None or thumb.size == 0:
                    # find KF order to map idx→Keyframe
                    # (poses are appended in order, so i-th KF has pose index world_map.keyframe_indices[i])
                    # but we cached thumbs; this is a fallback for robustness
                    for kf in kfs:
                        if kf.pose is not None and idx < len(world_map.poses):
                            dec = _decode_thumb(kf.thumb)  # lz4-compressed JPEG → BGR. :contentReference[oaicite:6]{index=6}
                            if dec is not None:
                                w, h = tuple(args.kf_thumb_hw)
                                thumb = cv2.resize(dec, (w, h), interpolation=cv2.INTER_AREA)
                                break
                triplet.append((idx, thumb, kps))
            trip = render_kf_triptych(
                triplet, world_map,
                maturity_thresh=args.track_maturity,
                new_ids=new_ids,
                title=f"New points: green || Mature tracks (>= {args.track_maturity}) : red"
            )
            if trip is not None:
                # ensure the window is large enough to see everything
                H, W = trip.shape[:2]
                cv2.resizeWindow(TRIP_WIN, max(960, int(W)), max(360, int(H)))
                cv2.imshow(TRIP_WIN, trip)

        # ---------------- UI keys ------------------------------
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
