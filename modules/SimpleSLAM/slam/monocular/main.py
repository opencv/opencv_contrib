# main.py
"""
WITHOUT MULTI-VIEW TRIANGULATION

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

    world_map = Map()
    cur_pose = np.eye(4)  # camera‑to‑world (identity at t=0)
    world_map.add_pose(cur_pose)
    viz3d = None if args.no_viz3d else Visualizer3D(color_axis="y")
    plot2d = TrajectoryPlotter()           

    kfs: list[Keyframe] = []
    last_kf_idx = -999

    frame_keypoints: List[List[cv2.KeyPoint]] = []  #CHANGE
    # --- visualisation ---
    achieved_fps = 0.0
    last_time = cv2.getTickCount() / cv2.getTickFrequency()

    # cv2.namedWindow('Feature Tracking', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Feature Tracking', 1200, 600)

    total = len(seq) - 1

    for i in tqdm(range(total), desc='Tracking'):
        # --- load image pair ---
        img1, img2 = load_frame_pair(args, seq, i)

        # --- feature extraction / matching ---
        kp1, des1 = feature_extractor(args, img1, detector)
        kp2, des2 = feature_extractor(args, img2, detector)
        matches = feature_matcher(args, kp1, kp2, des1, des2, matcher)
        # --- filter matches with RANSAC ---
        # matches = filter_matches_ransac(kp1, kp2, matches, args.ransac_thresh)

        if len(matches) < 12:
            print(f"[WARN] Not enough matches at frame {i}. Skipping.")
            continue
        
        # store keypoints for *every* new pose:
        if i == 0:
            # kp1 is the keypoints for the very first pose (index 0)
            frame_keypoints.append(kp1)
        # kp2 will correspond to the pose you'll add below at index i+1
        frame_keypoints.append(kp2)

        # ---------------- Key-frame decision -------------------------- #
        if i == 0:
            kfs.append(Keyframe(0, seq[0] if isinstance(seq[0], str) else "",
                        kp1, des1, make_thumb(img1, tuple(args.kf_thumb_hw))))
            last_kf_idx = 0
            is_kf = False
        else:
            prev_len = len(kfs)
            kfs, last_kf_idx = select_keyframe(
                args, seq, i, img2, kp2, des2, matcher, kfs, last_kf_idx)
            is_kf = len(kfs) > prev_len


        # --- #TODO: INITIALIZATION Relative pose from Essential matrix --- 
        # TODO missing triangulation step here, also should be in if statement only in beginning or loss of keyframe
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, mask_E = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=args.ransac_thresh,
        )
        if E is None or E.shape != (3, 3):
            print(f"[WARN] Essential matrix failed at frame {i}.")
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
        mask = (mask_E.ravel() & mask_pose.ravel()).astype(bool)
        pts1_in, pts2_in = pts1[mask], pts2[mask]


        # -- Predict pose from last step (VO) --
        pose_prev = cur_pose.copy()
        pose_pred = cur_pose @ get_homo_from_pose_rt(R, t)


        # -----------------------------PNP
        cur_pose = pose_pred
        used_kp2: list[int] = []
        new_ids:  list[int] = []

        if is_kf:
            # -- PnP refinement using existing landmarks --
            pts_w = world_map.get_point_array()
            pts3d, pts2d, used_kp2 = associate_landmarks(K, pose_pred, pts_w, kp2, args.proj_radius)

            if len(pts3d) >= args.pnp_min_inliers:
                R_pnp, t_pnp = refine_pose_pnp(K, pts3d, pts2d)
                if R_pnp is not None:
                    cur_pose = np.eye(4)
                    cur_pose[:3, :3] = R_pnp.T
                    cur_pose[:3, 3] = -R_pnp.T @ t_pnp
                else:
                    cur_pose = pose_pred
            else:
                cur_pose = pose_pred

            # --- Triangulate new landmarks only for keyframes ---
            fresh = [m for m in matches if m.trainIdx not in set(used_kp2)]
            if fresh:
                pts1_new = np.float32([kp1[m.queryIdx].pt for m in fresh])
                pts2_new = np.float32([kp2[m.trainIdx].pt for m in fresh])

                rel = np.linalg.inv(pose_prev) @ cur_pose
                R_rel, t_rel = rel[:3, :3], rel[:3, 3]
                pts3d_cam = triangulate_points(K, R_rel, t_rel, pts1_new, pts2_new)
                z = pts3d_cam[:, 2]
                valid = (z > args.min_depth) & (z < args.max_depth)
                pts3d_cam = pts3d_cam[valid]

                if len(pts3d_cam):
                    pts3d_world = (
                        pose_prev
                        @ np.hstack([pts3d_cam, np.ones((len(pts3d_cam), 1))]).T
                    ).T[:, :3]

                    # ------------ sample RGB from img2 -------------
                    pix = [kp2[m.trainIdx].pt for m in np.asarray(fresh)[valid]]
                    cols = []
                    h, w, _ = img2.shape
                    for (u, v) in pix:
                        x, y = int(round(u)), int(round(v))
                        if 0 <= x < w and 0 <= y < h:
                            b, g, r = img2[y, x]          # BGR uint8
                            cols.append((r, g, b))        # keep RGB order
                        else:
                            cols.append((255, 255, 255))  # white fallback
                    cols = np.float32(cols) / 255.0        # → 0-1

                    # coarse rigid alignment of the *new* cloud to the map
                    pts3d_world = world_map.align_points_to_map(pts3d_world, radius=args.merge_radius * 2.0)  # TODO: HYPER PARAMETER CHECK -MAGIC VARIABLE- radius can be loose here
                    frame_no = i + 1
                    new_ids = world_map.add_points(
                        pts3d_world, cols, keyframe_idx=frame_no
                    )
                    for pid, m in zip(new_ids, np.asarray(fresh)[valid]):
                        world_map.points[pid].add_observation(frame_no, m.trainIdx)

            if i % 10 == 0:
                world_map.fuse_closeby_duplicate_landmarks(args.merge_radius)


        world_map.add_pose(cur_pose)

        if is_kf and len(world_map.points) > 100:      # light heuristic
            run_bundle_adjustment(world_map, K, frame_keypoints,
                                fix_first_pose=True,
                                max_iters=20)


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
        # # ---------------------- GUI FOR 2D feature Tracking -------------------------- #
        # vis = draw_tracks(img2.copy(), tracks, frame_no)
        # for t in prev_map.keys():
        #     cv2.circle(vis, tuple(map(int, kp2[t].pt)), 3, (0, 255, 0), -1)

        # cv2.putText(vis, f"KF idx: {last_kf_idx}  |  total KFs: {len(kfs)}",
        #             (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 255, 0), 2)

        # # thumb strip (last 4)
        # thumbs = [cv2.imdecode(
        #           np.frombuffer(lz4.frame.decompress(k.thumb), np.uint8),
        #           cv2.IMREAD_COLOR) for k in kfs[-4:]]
        # bar = (np.hstack(thumbs) if thumbs else
        #        np.zeros((*args.kf_thumb_hw[::-1], 3), np.uint8))
        # cv2.imshow('Keyframes', bar)

        # cv2.putText(vis,
        #             (f"Frame {frame_no}/{total} | "
        #              f"Tracks: {len(tracks)} | "
        #              f"FPS: {achieved_fps:.1f}"),
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #             (255, 255, 255), 2)

        # cv2.imshow('Feature Tracking', vis)
        # wait_ms = int(1000 / args.fps) if args.fps > 0 else 1
        # key = cv2.waitKey(0 if PAUSED else wait_ms) & 0xFF
        # if key == 27:  # ESC to exit
        #     break
        # elif key == ord('p'):  # 'p' to toggle pause
        #     PAUSED = not PAUSED
        #     continue

        # update FPS
        # now = cv2.getTickCount() / cv2.getTickFrequency()
        # achieved_fps = 1.0 / (now - last_time)
        # last_time = now

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()