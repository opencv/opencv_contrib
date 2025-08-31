# main.py
"""
WITH MULTI-VIEW TRIANGULATION - New PnP

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
from copy import deepcopy
import cv2
import lz4.frame
import numpy as np
from tqdm import tqdm
from typing import List

from slam.core.pose_utils import _pose_inverse, _pose_rt_to_homogenous

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
    select_keyframe, 
    make_thumb)

from slam.core.visualization_utils import draw_tracks, Visualizer3D, Trajectory2D, VizUI
from slam.core.trajectory_utils import compute_gt_alignment, apply_alignment
from slam.core.landmark_utils import Map

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s:%(funcName)s: %(message)s") # INFO for clean summary, DEBUG for deep dive
# logging.getLogger("two_view_bootstrap").setLevel(logging.DEBUG)
# logging.getLogger("pnp").setLevel(logging.DEBUG)
# logging.getLogger("multi_view").setLevel(logging.DEBUG)
# logging.getLogger("triangulation").setLevel(logging.DEBUG)
log = logging.getLogger("main")


from slam.core.two_view_bootstrap import (
    InitParams,
    pts_from_matches,                        # used to build point arrays
    evaluate_two_view_bootstrap_with_masks,  # NEW (returns inlier mask)
    bootstrap_two_view_map                   # NEW (builds KF0+KF1+points)
)

from slam.core.pnp_utils import (
    predict_pose_const_vel,
    reproject_and_match_2d3d,
    solve_pnp_ransac,
    draw_reprojection_debug
)

from slam.core.triangulation_utils import (
    triangulate_between_kfs_2view
)

from slam.core.ba_utils import pose_only_ba, local_bundle_adjustment


class BootstrapState:
        def __init__(self):
            self.has_ref = False
            self.kps_ref = None
            self.des_ref = None
            self.img_ref = None
            self.frame_id_ref = -1
        def seed(self, kps, des, img, frame_id):
            self.has_ref = True
            self.kps_ref, self.des_ref = kps, des
            self.img_ref, self.frame_id_ref = img, frame_id
            log.info(f"[Init] Seeded reference @frame={frame_id} (kps={len(kps)})")
        def clear(self):
            log.info("[Init] Clearing reference (bootstrap succeeded).")
            self.__init__()

def _refresh_ref_needed(matches, min_matches=80, max_age=30, cur_id=0, ref_id=0):
    too_few = len(matches) < min_matches
    too_old = (cur_id - ref_id) > max_age
    if too_few: log.info(f"[Init] Refresh ref: few matches ({len(matches)}<{min_matches})")
    if too_old: log.info(f"[Init] Refresh ref: age={cur_id-ref_id}>{max_age}")
    return too_few or too_old

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
    p.add_argument('--ransac_thresh', type=float, default=2.5)

    # key-frame params
    p.add_argument('--kf_max_disp', type=float, default=45)
    p.add_argument('--kf_min_inliers', type=float, default=150)
    p.add_argument('--kf_min_ratio', type=float, default=0.35,
               help='Min inlier ratio (to prev KF kps) before promoting KF')
    p.add_argument('--kf_min_rot_deg', type=float, default=8.0,
               help='Min rotation (deg) wrt prev KF to trigger KF')
    p.add_argument('--kf_cooldown', type=int, default=5)
    p.add_argument('--kf_thumb_hw', type=int, nargs=2,
                   default=[640, 360])
    
    # 3‑D visualisation toggle
    p.add_argument("--no_viz3d", action="store_true", help="Disable 3‑D visualization window")

    # triangulation depth filtering
    p.add_argument("--min_depth", type=float, default=0.40)
    p.add_argument("--max_depth", type=float, default=float('inf'))
    p.add_argument('--mvt_rep_err', type=float, default=2.0,
               help='Max mean reprojection error (px) for multi-view triangulation')

    #  PnP / map-maintenance
    p.add_argument('--pnp_min_inliers', type=int, default=20)
    p.add_argument('--proj_radius',     type=float, default=12.0)
    p.add_argument('--merge_radius',    type=float, default=0.10)

    # Bundle Adjustment
    p.add_argument('--local_ba_window', type=int, default=6, help='Window size (number of keyframes) for local BA')

    return p


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


    
    bs = BootstrapState()

    # --- tracking state ---
    prev_map, tracks = {}, {}
    next_track_id = 0
    initialised = False
    tracking_lost = False

    # --- World Map Initialization ---
    world_map = Map()
    Tcw_cur_pose = np.eye(4)  # camera-from-world (identity at t=0)

    # --- Visualization  ---
    viz3d = None if args.no_viz3d else Visualizer3D(color_axis="y")
    traj2d = Trajectory2D(gt_T_list=gt_T if groundtruth is not None else None)
    ui = VizUI()  # p: pause/resume, n: step, q/Esc: quit

    # --- Keyframe Initialization ---
    kfs: list[Keyframe] = []
    last_kf_frame_no = -999

    # --- visualisation ---
    achieved_fps = 0.0
    last_time = cv2.getTickCount() / cv2.getTickFrequency()

    new_ids:  list[int] = [] # list of new IDs for showing them in 3D viz
    total = len(seq) - 1

    for i in tqdm(range(total), desc='Tracking'):

        # --- load image pair ---
        img1, img2 = load_frame_pair(args, seq, i)

        # --- feature extraction / matching ---
        kp1, des1 = feature_extractor(args, img1, detector)
        kp2, des2 = feature_extractor(args, img2, detector)

        # --------------------------------------------------------------------------- #
        # ------------------------ Bootstrap (delayed two-view) ------------------------------------------------
        # --------------------------------------------------------------------------- #
        if not initialised:
            # 1) Seed a reference frame the first time we get here
            if not bs.has_ref:
                # use the *earlier* image/features as reference; your loop already has img1/img2, kp1/des1 & kp2/des2
                bs.seed(kp1, des1, img1, frame_id=i)   # frame_id = i for img1
                continue

            # 2) Match reference ↔ current using your existing pipeline
            matches_bs = feature_matcher(args, bs.kps_ref, kp2, bs.des_ref, des2, matcher)
            matches_bs = filter_matches_ransac(bs.kps_ref, kp2, matches_bs, args.ransac_thresh)
            log.info(f"[Init] Matches ref→cur: raw={len(matches_bs)}  ransac_th={args.ransac_thresh:.2f}px")

            # Optional: refresh reference if matches are weak or ref is stale
            if _refresh_ref_needed(matches_bs, min_matches=80, max_age=30, cur_id=i+1, ref_id=bs.frame_id_ref):
                bs.seed(kp2, des2, img2, frame_id=i+1)
                continue

            # 3) Build point arrays and evaluate the pair
            pts_ref, pts_cur = pts_from_matches(bs.kps_ref, kp2, matches_bs)
            init_params = InitParams(
                ransac_px=float(args.ransac_thresh),
                min_posdepth=0.90,
                min_parallax_deg=1.5,
                score_ratio_H=0.45
            )
            # Decide with masks (no redundancy later)
            decision = evaluate_two_view_bootstrap_with_masks(K, pts_ref, pts_cur, init_params)
            if decision is None:
                log.info("[Init] Pair rejected → waiting for a better one.")
                continue

            log.info(f"[Init] Accepted pair via {'H' if decision.pose.model.name=='HOMOGRAPHY' else 'F/E'}: "
                    f"posdepth={decision.pose.posdepth:.3f}, parallax={decision.pose.parallax_deg:.2f}°")

            # Build the initial map (KF0 = I, KF1 = [R|t]) once
            ok, T0_cw, T1_cw = bootstrap_two_view_map(
                K,
                bs.kps_ref, bs.des_ref,   # reference frame kps/desc
                kp2, des2,                # current frame kps/desc
                matches_bs,
                args,
                world_map,
                params=init_params,
                decision=decision         # pass decision to avoid recomputing/gating
            )
            if not ok:
                log.warning("[Init] bootstrap_two_view_map() failed — keep searching.")
                continue
            
            # -------------BOOKKEEPING for Map and Keyframes---------------------
            world_map.add_pose(T0_cw, is_keyframe=True)   # KF0
            world_map.add_pose(T1_cw, is_keyframe=True)   # KF1

            # --- 2D Trajectory visualization bookkeeping ---
            traj2d.push(bs.frame_id_ref, T0_cw)
            traj2d.push(i + 1,          T1_cw)
            
            # --- Create the two initial Keyframes to mirror world_map KFs ---
            try:
                # Be explicit about indices: world_map added KF0 first, then KF1.
                # Keep kfs indices aligned (0, 1) to avoid confusion later.
                if len(kfs) != 0:
                    log.warning("[Init] Expected empty kfs at bootstrap, found len=%d. "
                                "Proceeding but indices may misalign.", len(kfs))

                kf0_idx = 0 if len(kfs) == 0 else len(kfs)
                kf1_idx = kf0_idx + 1

                # Resolve frame indices and image paths
                ref_fidx = bs.frame_id_ref                         # you seeded with this earlier
                cur_fidx = i + 1                                   # kp2/img2 correspond to i+1
                ref_path = seq[ref_fidx] if isinstance(seq[ref_fidx], str) else ""
                cur_path = seq[cur_fidx] if isinstance(seq[cur_fidx], str) else ""

                # Build thumbnails (guard if you disabled UI)
                thumb0 = make_thumb(bs.img_ref, tuple(args.kf_thumb_hw)) if 'make_thumb' in globals() else None
                thumb1 = make_thumb(img2,       tuple(args.kf_thumb_hw)) if 'make_thumb' in globals() else None

                # Create KF0 = Identity pose
                kf0 = Keyframe(idx=kf0_idx, frame_idx=ref_fidx, path=ref_path, kps=bs.kps_ref, desc=bs.des_ref, pose=T0_cw, thumb=thumb0)

                # Create KF1 = bootstrap pose
                kf1 = Keyframe(idx=kf1_idx, frame_idx=cur_fidx, path=cur_path, kps=kp2, desc=des2, pose=T1_cw, thumb=thumb1)

                kfs.extend([kf0, kf1])
                last_kf_frame_no = cur_fidx   # the most recently added keyframe
                log.info("[Init] Inserted initial keyframes: KF0(frame=%d, idx=%d) & KF1(frame=%d, idx=%d).",
                        ref_fidx, kf0_idx, cur_fidx, kf1_idx)


                # and also seed their mutual matches (helps get first 3-view tracks quickly)
                raw01 = feature_matcher(args, kfs[0].kps, kfs[1].kps, kfs[0].desc, kfs[1].desc, matcher)
                # --- END --- 

            except Exception as e:
                log.exception("[Init] Failed to create initial keyframes: %s", e)

            # SET INITIALISED FLAG
            initialised = True
            print("-----BOOTSTRAPPED SUCCESSFULLY-----")
            
            if viz3d:
                # highlight everything once (all points are new at bootstrap)
                viz3d.update(world_map, new_ids=world_map.point_ids())

            Tcw_cur_pose = T1_cw.copy()   # current camera pose (optional)
            bs.clear()
            continue
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----

        # ------------------------------------------------------------------- #
        # --------------------- Frame-to-Map Tracking ----------------------- #
        # ------------------------------------------------------------------- #
        # When initialised, use constant-velocity prediction, reproject, search, PnP.
        if initialised:
            # 1) Predict pose with constant velocity using last two poses
            if len(world_map.poses) >= 2:
                Tcw_prevprev = world_map.poses[-2]
                Tcw_prev     = world_map.poses[-1]
                Tcw_pred     = predict_pose_const_vel(Tcw_prevprev, Tcw_prev)
            else:
                Tcw_pred     = Tcw_cur_pose

            # 2) Reproject active points and do small-window matching
            H, W = img2.shape[:2]
            m23d = reproject_and_match_2d3d(
                world_map, K, Tcw_pred, kp2, des2, W, H,
                radius_px=float(args.proj_radius),  # CLI param
                max_hamm=64
            )

            log.debug(f"[Track] candidates for PnP: {len(m23d.pts3d)}")

            # 3) PnP+RANSAC
            if len(m23d.pts3d) >= int(args.pnp_min_inliers):
                Tcw_est, inl_mask = solve_pnp_ransac(
                    m23d.pts3d, m23d.pts2d, K,
                    ransac_px=float(args.ransac_thresh),
                    Tcw_init=Tcw_pred,
                    iters=300, conf=0.999
                )
                ninl = int(inl_mask.sum()) if inl_mask.size else 0
                if Tcw_est is not None and ninl >= int(args.pnp_min_inliers):
                    Tcw_cur_pose = Tcw_est
                    world_map.add_pose(Tcw_cur_pose, is_keyframe=False)

                    # --- 2D Trajectory visualization bookkeeping ---
                    traj2d.push(i + 1, Tcw_cur_pose)

                    log.info(f"[Track] PnP inliers={ninl}  (th={args.ransac_thresh:.1f}px)")

                    # Optional: quick visual sanity overlay (comment out if headless)
                    try:
                        dbg = draw_reprojection_debug(img2, K, Tcw_cur_pose, m23d, inl_mask)
                        cv2.imshow("Track debug", dbg)
                        # if cv2.waitKey(int(1000.0/max(args.fps, 1e-6))) == 27:
                        #     break # TODO REMOVE
                    except Exception:
                        pass

                else:
                    log.warning(f"[Track] PnP failed or too few inliers (got {ninl}). Tracking lost?")
                    tracking_lost = True
            else:
                log.warning(f"[Track] Too few 2D–3D matches for PnP ({len(m23d.pts3d)} < {args.pnp_min_inliers}).")
                tracking_lost = True

            # (Optional) if tracking_lost: trigger relocalization here in the future.
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----
        
        
        # ------------------------------------------------------------------- #
        # ---------------------     Keyframe Selection  --------------------- #
        # ------------------------------------------------------------------- #
        prev_len = len(kfs)
        kfs, last_kf_frame_no = select_keyframe(
            args, seq, i, img2, kp2, des2, Tcw_cur_pose, matcher, kfs, last_kf_frame_no
        )
        is_kf = (len(kfs) > prev_len)
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----

        # ------------------------------------------------------------------- #
        # --------------------- Map Growth (Triangulation) --------------------- #
        # ------------------------------------------------------------------- #
        if is_kf and len(kfs) >= 2:
            prev_kf = kfs[-2]
            curr_kf = kfs[-1]
            new_ids = triangulate_between_kfs_2view(
                args, K, world_map, prev_kf, curr_kf, matcher, log,
                use_parallax_gate=True, parallax_min_deg=2.0,
                reproj_px_max=float(args.ransac_thresh)
            )
            if new_ids:
                log.info("[Map] Triangulated %d new points after KF %d.", len(new_ids), curr_kf.idx)

                # TODO pose only BA should happen on every tracking frame and not here, but well, if it works we don't touch it
                # --- BA pass 1: fast pose-only BA on current KF (uses its own obs) ---
                try:
                    pose_only_ba(world_map, K, kfs, kf_idx=curr_kf.idx, max_iters=8, huber_thr=2.0)
                except Exception as e:
                    log.warning(f"[BA] Pose-only BA failed: {e}")

                # --- BA pass 2: local BA on a small KF window ---
                try:
                    local_bundle_adjustment(
                        world_map, K, kfs,
                        center_kf_idx=curr_kf.idx,
                        window_size=int(getattr(args, "local_ba_window", 6)),
                        max_points=10000,
                        max_iters=15
                    )
                except Exception as e:
                    log.warning(f"[BA] Local BA failed: {e}")

                if viz3d:
                    viz3d.update(world_map, new_ids=new_ids)
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----


        # --------------------------------------MISCELLANEOUS INFO ---------------------------------------------------
        # print(f"[traj2d] est={len(traj2d.est_xyz)} gt={len(traj2d.gt_xyz)}")
        # print(f"Frame {i+1}/{total}  |  FPS: {achieved_fps:.1f}  |  KFs: {len(kfs)}  |  Map points: {len(world_map.points)}")
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----

        
        # ------------------------------------------------------------------- #
        # --------------------- Visualization ----------------------- #
        # ------------------------------------------------------------------- #
        if viz3d:
            viz3d.update(world_map, new_ids=new_ids)
        # --- draw & UI control (end of iteration) ---
        traj2d.draw(paused=ui.paused)

        ui.poll(1)  # non-blocking poll
        if ui.should_quit():
            break

        # If currently paused: block here until resume or do single-step
        if ui.paused:
            did_step = ui.wait_if_paused()
            # if user pressed 'n', did_step=True -> allow this iteration to exit;
            # next iteration will immediately pause again (nice for stepping).
        # --x------x----------x----------x----------x----x----x-- END -x----------x-----------x----------x----
    
    if viz3d:
        viz3d.close()    

if __name__ == '__main__':
    main()