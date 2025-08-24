# main.py
"""
WITHOUT MULTI-VIEW TRIANGULATION - Use 2 different stratergy for motion estimation

Entry-point: high-level processing loop
--------------------------------------
$ python main.py --dataset kitti --base_dir ../Dataset


The core loop now performs:
  1) Feature detection + matching (OpenCV or LightGlue)
  2) Essential‑matrix estimation + pose recovery
  3) Landmarks triangulation (with Z‑filtering)
  4) Pose integration (camera trajectory in world frame)
  5) Optional 3‑D visualisation via Open3D 

The script shares most command‑line arguments with the previous version
but adds `--no_viz3d` to disable the 3‑D window.

"""
import argparse
import cv2
import lz4.frame
import numpy as np
from tqdm import tqdm
from typing import List

from slam.core.dataloader import (
                            load_sequence, 
                            load_frame_pair, 
                            load_calibration, 
                            load_groundtruth)

from slam.core.features_utils import (
                                init_feature_pipeline, 
                                feature_extractor, 
                                feature_matcher, 
                                filter_matches_ransac)

from slam.core.keyframe_utils import (
    Keyframe, 
    update_and_prune_tracks, 
    select_keyframe, 
    make_thumb)

from slam.core.visualization_utils import draw_tracks, Visualizer3D, TrajectoryPlotter
from slam.core.trajectory_utils import compute_gt_alignment, apply_alignment
from slam.core.landmark_utils import Map, triangulate_points
from slam.core.pnp_utils import associate_landmarks, refine_pose_pnp
from slam.core.ba_utils_OG import run_bundle_adjustment
from slam.core.ba_utils import (
    two_view_ba,
    pose_only_ba,
    local_bundle_adjustment
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
    p.add_argument('--ransac_thresh', type=float, default=1.0)
    # key-frame params
    p.add_argument('--kf_max_disp', type=float, default=45)
    p.add_argument('--kf_min_inliers', type=float, default=150)
    p.add_argument('--kf_cooldown', type=int, default=3)
    p.add_argument('--kf_thumb_hw', type=int, nargs=2,
                   default=[640, 360])
    
    # 3‑D visualisation toggle
    p.add_argument("--no_viz3d", action="store_true", help="Disable 3‑D visualization window")
    # triangulation depth filtering
    p.add_argument("--min_depth", type=float, default=1.0)
    p.add_argument("--max_depth", type=float, default=50.0)

    #  PnP / map-maintenance
    p.add_argument('--pnp_min_inliers', type=int, default=30)
    p.add_argument('--proj_radius',     type=float, default=3.0)
    p.add_argument('--merge_radius',    type=float, default=0.10)

    return p


# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #

def _pose_inv(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return 4×4 inverse of a rigid‑body transform specified by R|t."""
    Rt = R.T
    tinv = -Rt @ t
    T = np.eye(4)
    T[:3, :3] = Rt
    T[:3, 3] = tinv.ravel()
    return T

def get_homo_from_pose_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Homogeneous matrix  **T_c1→c2**  from the relative pose (R, t)
       returned by `recoverPose`."""
    T = np.eye(4)
    T[:3, :3]  = R
    T[:3, 3]   = t.ravel()
    return T

def try_bootstrap(K, kp0, kp1, matches, args, world_map):
    """Return (success, T_cam0_w, T_cam1_w) and add initial landmarks."""
    if len(matches) < 50:
        return False, None, None

    # 1. pick a model: Essential (general) *or* Homography (planar)
    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

    E, inl_E = cv2.findEssentialMat(
        pts0, pts1, K, cv2.RANSAC, 0.999, args.ransac_thresh)

    H, inl_H = cv2.findHomography(
        pts0, pts1, cv2.RANSAC, args.ransac_thresh)

    # score = #inliers (ORB-SLAM uses a more elaborate model-selection score)
    use_E = (inl_E.sum() > inl_H.sum())

    if use_E and E is None:
        return False, None, None
    if not use_E and H is None:
        return False, None, None

    if use_E:
        _, R, t, inl_pose = cv2.recoverPose(E, pts0, pts1, K)
        mask = (inl_E.ravel() & inl_pose.ravel()).astype(bool)
    else:
        R, t = cv2.decomposeHomographyMat(H, K)[1:3]  # take the best hypothesis
        mask = inl_H.ravel().astype(bool)

    # 2. triangulate those inliers – exactly once
    p0 = pts0[mask]
    p1 = pts1[mask]
    pts3d = triangulate_points(K, R, t, p0, p1)
    z = pts3d[:, 2]
    ok = (z > args.min_depth) & (z < args.max_depth)
    pts3d = pts3d[ok]

    if len(pts3d) < 80:
        return False, None, None

    # 3. fill the map
    T0_w = np.eye(4)
    T1_w = np.eye(4)
    T1_w[:3, :3] = R.T
    T1_w[:3, 3]  = ( -R.T @ t ).ravel()

    # world_map.reset()                      # fresh map
    # world_map.add_pose(T0_w)
    world_map.add_pose(T1_w)

    cols = np.full((len(pts3d), 3), 0.7)   # grey – colour is optional here
    ids = world_map.add_points(pts3d, cols, keyframe_idx=1)

    # -----------------------------------------------
    # add (frame_idx , kp_idx) pairs for each new MP
    # -----------------------------------------------
    inlier_kp_idx = np.where(mask)[0][ok]   # kp indices that survived depth
    for pid, kp_idx in zip(ids, inlier_kp_idx):
        world_map.points[pid].add_observation(0, kp_idx)   # img0 side
        world_map.points[pid].add_observation(1, kp_idx)   # img1 side


    print(f"[BOOTSTRAP] Map initialised with {len(ids)} landmarks.")
    return True, T0_w, T1_w


def solve_pnp_step(K, pose_pred, world_map, kp, args):
    """Return (success, refined_pose, used_kp_indices)."""
    pts_w = world_map.get_point_array()
    pts3d, pts2d, used = associate_landmarks(
        K, pose_pred, pts_w, kp, args.proj_radius)

    if len(pts3d) < args.pnp_min_inliers:
        return False, pose_pred, used

    R, t = refine_pose_pnp(K, pts3d, pts2d)
    if R is None:
        return False, pose_pred, used

    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3]  = -R.T @ t
    print(f"[PNP] Pose refined with {len(pts3d)} inliers.")
    return True, T, used


def triangulate_new_points(K, pose_prev, pose_cur,
                           kp_prev, kp_cur, matches,
                           used_cur_idx, args, world_map, img_cur):
    """
    Triangulate *all* keypoints that are ❶ un-tracked in the current
    key-frame and ❷ fall inside the [min_depth , max_depth] band.

    The former parallax (bearing-angle) gate is gone, so we return more
    candidate landmarks per key-frame.  Depth filtering is kept to
    avoid the most obvious degeneracies (z≈0 or way too far).
    """
    # ------------------------------------------------------------------
    # 1) keep only keypoints that PnP did **not** consume
    # ------------------------------------------------------------------
    fresh = [m for m in matches if m.trainIdx not in used_cur_idx]
    if not fresh:
        return []                       # nothing new to add this time

    p0 = np.float32([kp_prev[m.queryIdx].pt for m in fresh])
    p1 = np.float32([kp_cur[m.trainIdx].pt  for m in fresh])

    # ------------------------------------------------------------------
    # 2) triangulate once per key-frame pair
    # ------------------------------------------------------------------
    rel       = np.linalg.inv(pose_prev) @ pose_cur
    R_rel     = rel[:3, :3]
    t_rel     = rel[:3, 3]

    pts3d_cam = triangulate_points(K, R_rel, t_rel, p0, p1)

    # ------------------------------------------------------------------
    # 3) depth sanity check (keep only z within user bounds)
    # ------------------------------------------------------------------
    z          = pts3d_cam[:, 2]
    depth_ok   = (z > args.min_depth) & (z < args.max_depth)
    if not depth_ok.any():
        return []                       # everything was outside bounds

    pts3d_cam  = pts3d_cam[depth_ok]
    fresh      = [m for m, ok in zip(fresh, depth_ok) if ok]

    # ------------------------------------------------------------------
    # 4) colour sampling for the surviving points (optional but nice)
    # ------------------------------------------------------------------
    h, w, _ = img_cur.shape
    pix  = [kp_cur[m.trainIdx].pt for m in fresh]
    cols = []
    for (u, v) in pix:
        x, y = int(round(u)), int(round(v))
        if 0 <= x < w and 0 <= y < h:
            b, g, r = img_cur[y, x]
            cols.append((r, g, b))
        else:
            cols.append((255, 255, 255))
    cols = np.float32(cols) / 255.0

    # ------------------------------------------------------------------
    # 5) transform to world frame and commit to the map
    # ------------------------------------------------------------------
    pts3d_w = pose_prev @ np.hstack([pts3d_cam, np.ones((len(pts3d_cam), 1))]).T
    pts3d_w = pts3d_w.T[:, :3]

    new_ids = world_map.add_points(pts3d_w, cols,
                                   keyframe_idx=len(world_map.poses) - 1)
    print(f"[TRIANGULATION] Added {len(new_ids)} new landmarks.")
    return new_ids, fresh

# def triangulate_new_points(K, pose_prev, pose_cur,
#                            kp_prev, kp_cur, matches,
#                            used_cur_idx, args, world_map, img_cur):
#     """Triangulate only *unmatched* keypoints and add them if baseline is good."""
#     fresh = [m for m in matches if m.trainIdx not in used_cur_idx]
#     if not fresh:
#         return []

#     p0 = np.float32([kp_prev[m.queryIdx].pt for m in fresh])
#     p1 = np.float32([kp_cur[m.trainIdx].pt  for m in fresh])

#     # -- baseline / parallax test (ORB-SLAM uses θ ≈ 1°)
#     rel = np.linalg.inv(pose_prev) @ pose_cur
#     R_rel, t_rel = rel[:3, :3], rel[:3, 3]
#     cos_thresh = np.cos(np.deg2rad(1.0))

#     rays0 = cv2.undistortPoints(p0.reshape(-1, 1, 2), K, None).reshape(-1, 2)
#     rays1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
#     rays0 = np.hstack([rays0, np.ones((len(rays0), 1))])
#     rays1 = np.hstack([rays1, np.ones((len(rays1), 1))])

#     ang_ok = np.einsum('ij,ij->i', rays0, (R_rel @ rays1.T).T) < cos_thresh
#     if not ang_ok.any():
#         return []

#     p0, p1 = p0[ang_ok], p1[ang_ok]
#     fresh  = [m for m, keep in zip(fresh, ang_ok) if keep]

#     pts3d = triangulate_points(K, R_rel, t_rel, p0, p1)
#     z = pts3d[:, 2]
#     depth_ok = (z > args.min_depth) & (z < args.max_depth)
#     pts3d = pts3d[depth_ok]
#     fresh  = [m for m, keep in zip(fresh, depth_ok) if keep]

#     if not len(pts3d):
#         return []

#     # colour sampling
#     h, w, _ = img_cur.shape
#     pix = [kp_cur[m.trainIdx].pt for m in fresh]
#     cols = []
#     for (u, v) in pix:
#         x, y = int(round(u)), int(round(v))
#         if 0 <= x < w and 0 <= y < h:
#             b, g, r = img_cur[y, x]
#             cols.append((r, g, b))
#         else:
#             cols.append((255, 255, 255))
#     cols = np.float32(cols) / 255.0

#     pts3d_w = pose_prev @ np.hstack([pts3d, np.ones((len(pts3d), 1))]).T
#     pts3d_w = pts3d_w.T[:, :3]
#     ids = world_map.add_points(pts3d_w, cols, keyframe_idx=len(world_map.poses)-1)
#     return ids


# --------------------------------------------------------------------------- #
#  Main processing loop
# --------------------------------------------------------------------------- #
def main():
    PAUSED = False
    args = _build_parser().parse_args()

    # --- Data loading ---
    seq = load_sequence(args)
    calib       = load_calibration(args)        # dict with K_l, P_l, ...
    groundtruth = load_groundtruth(args)        # None or Nx3x4 array
    K = calib["K_l"]  # intrinsic matrix for left camera
    P = calib["P_l"]  # projection matrix for left camera

    # ------ build 4×4 GT poses + alignment matrix (once) ----------------
    gt_T = None
    R_align = t_align = None
    if groundtruth is not None:
        gt_T = np.pad(groundtruth, ((0, 0), (0, 1), (0, 0)), constant_values=0.0)
        gt_T[:, 3, 3] = 1.0                             # homogeneous 1s
        R_align, t_align = compute_gt_alignment(gt_T)

    # --- feature pipeline (OpenCV / LightGlue) ---
    detector, matcher = init_feature_pipeline(args)

    # --- tracking state ---
    prev_map, tracks = {}, {}
    next_track_id = 0
    initialised = False
    tracking_lost = False


    world_map = Map()
    # cur_pose = np.eye(4)  # camera‑to‑world (identity at t=0)
    # world_map.add_pose(cur_pose) #TODO CODE IS INITIALIZING SOMEWHERE ELSE IN BOOTSTRAP CHECK IT
    viz3d = None if args.no_viz3d else Visualizer3D(color_axis="y")
    plot2d = TrajectoryPlotter()           

    kfs: list[Keyframe] = []
    last_kf_idx = -999

    # TODO FOR BUNDLE ADJUSTMENT
    frame_keypoints: List[List[cv2.KeyPoint]] = []  #CHANGE
    frame_keypoints.append([])   # placeholder to keep indices aligned

    # --- visualisation ---
    achieved_fps = 0.0
    last_time = cv2.getTickCount() / cv2.getTickFrequency()

    # cv2.namedWindow('Feature Tracking', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Feature Tracking', 1200, 600)

    total = len(seq) - 1
    


    new_ids:  list[int] = [] # CHANGE

    for i in tqdm(range(total), desc='Tracking'):
        # --- load image pair ---
        img1, img2 = load_frame_pair(args, seq, i)

        # --- feature extraction / matching ---
        kp1, des1 = feature_extractor(args, img1, detector)
        kp2, des2 = feature_extractor(args, img2, detector)
        matches = feature_matcher(args, kp1, kp2, des1, des2, matcher)
        # --- filter matches with RANSAC ---
        matches = filter_matches_ransac(kp1, kp2, matches, args.ransac_thresh)

        if len(matches) < 12:
            print(f"[WARN] Not enough matches at frame {i}. Skipping.")
            continue
        
        # ---------------- Key-frame decision -------------------------- # TODO make every frame a keyframe
        frame_keypoints.append(kp2)
        if i == 0:
            kfs.append(Keyframe(0, seq[0] if isinstance(seq[0], str) else "",
                        kp1, des1, make_thumb(img1, tuple(args.kf_thumb_hw))))
            last_kf_idx = 0
            prev_len = len(kfs)
            is_kf = False
            continue
        else:
            prev_len = len(kfs)
            kfs, last_kf_idx = select_keyframe(
                args, seq, i, img2, kp2, des2, matcher, kfs, last_kf_idx)
            is_kf = len(kfs) > prev_len

        print("len(kfs) = ", len(kfs), "last_kf_idx = ", last_kf_idx)
        # ------------------------------------------------ bootstrap ------------------------------------------------
        if not initialised:
            if len(kfs) < 2:
                continue 
            bootstrap_matches = feature_matcher(args, kfs[0].kps, kfs[-1].kps, kfs[0].desc, kfs[-1].desc, matcher)
            bootstrap_matches = filter_matches_ransac(kfs[0].kps, kfs[-1].kps, bootstrap_matches, args.ransac_thresh)
            ok, T0, T1 = try_bootstrap(K, kfs[0].kps, kfs[-1].kps, bootstrap_matches, args, world_map)
            if ok:
                frame_keypoints[0] = kfs[0].kps        # BA (img1 is frame-0)
                frame_keypoints[1] = kfs[1].kps        # BA (img2 is frame-1)
                # print("POSES: " ,world_map.poses)
                two_view_ba(world_map, K, frame_keypoints, max_iters=25) # BA

                initialised = True
                cur_pose = T1               # we are at frame i+1
                pose_prev = T0
                continue
            else:
                continue                # keep trying with next frame

        # ------------------------------------------------ tracking -------------------------------------------------
        pose_pred = cur_pose.copy()         # CV-predict could go here
        ok_pnp, cur_pose, used_idx = solve_pnp_step(
            K, pose_pred, world_map, kfs[-1].kps, args) # last keyframe

        if not ok_pnp:                      # fallback to 2-D-2-D if PnP failed
            pts0 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts1 = np.float32([kp2[m.trainIdx].pt  for m in matches])
            E, mask = cv2.findEssentialMat(pts0, pts1, K, cv2.RANSAC,
                                        0.999, args.ransac_thresh)
            if E is None:
                tracking_lost = True
                continue
            _, R, t, mpose = cv2.recoverPose(E, pts0, pts1, K)
            cur_pose = cur_pose @ get_homo_from_pose_rt(R, t)
            tracking_lost = False

        world_map.add_pose(cur_pose)        # always push *some* pose

        # pose_only_ba(world_map, K, frame_keypoints,    # FOR BA
        #      frame_idx=len(world_map.poses)-1)

        # ------------------------------------------------ map growth ------------------------------------------------
        if is_kf:
            new_ids, fresh = triangulate_new_points(
                K, pose_prev, cur_pose,
                kp1, kp2, matches,
                used_idx, args, world_map, img2
            )

            print(last_kf_idx, "LAST --------------------CURRENT", len(kfs))
            # last_kf_idx = len(world_map.poses) - 1          # current KF’s index

            for pid, m in zip(new_ids, fresh):
                world_map.points[pid].add_observation(last_kf_idx - 1, m.queryIdx)
                world_map.points[pid].add_observation(last_kf_idx,     m.trainIdx)

            if new_ids:
                print(f"Added {len(new_ids)} new landmarks.")

            pose_prev = cur_pose.copy()

            # local_bundle_adjustment(
            #     world_map, K, frame_keypoints,
            #     center_kf_idx=last_kf_idx,
            #     window_size=5 )


        # if is_kf and len(world_map.points) > 100:      # light heuristic
        #     run_bundle_adjustment(world_map, K, frame_keypoints,
        #                         fix_first_pose=True,
        #                         max_iters=20)


        # --- 2-D path plot (cheap) ----------------------------------------------
        est_pos = cur_pose[:3, 3]
        gt_pos  = None
        if gt_T is not None and i + 1 < len(gt_T):
            p_gt = gt_T[i + 1, :3, 3]                     # raw GT
            gt_pos = apply_alignment(p_gt, R_align, t_align)
        plot2d.append(est_pos, gt_pos, mirror_x=True)


        # # --- 2-D track maintenance (for GUI only) ---
        # frame_no = i + 1
        # prev_map, tracks, next_track_id = update_and_prune_tracks(
        #     matches, prev_map, tracks, kp2, frame_no, next_track_id)


        # --- 3-D visualisation ---
        if viz3d is not None:
            viz3d.update(world_map, new_ids)


        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('p') and viz3d is not None:
            viz3d.paused = not viz3d.paused


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()