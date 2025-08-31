# --------------------------------------------------------------------------- #
#  BA  reprojection debug helpers
# --------------------------------------------------------------------------- #
import cv2
import numpy as np
from copy import deepcopy
from typing import Tuple, List

def _gather_3d_2d(world_map, keypoints, frame_idx: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect   (pts3d_w , measured_px) pairs for *one* frame.
    """
    pts3d, uv = [], []
    for mp in world_map.points.values():
        for f_idx, kp_idx, _ in mp.observations:
            if f_idx == frame_idx:
                pts3d.append(mp.position)
                uv.append(keypoints[f_idx][kp_idx].pt)
                break
    if not pts3d:          # nothing observed by this camera
        return np.empty((0, 3)), np.empty((0, 2))
    return np.asarray(pts3d, np.float32), np.asarray(uv, np.float32)


def _project(K, T_wc, pts3d_w):
    """
    Vectorised pin-hole projection  (world ➜ camera ➜ image).
    """
    T_cw = np.linalg.inv(T_wc)
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cw)
    proj, _ = cv2.projectPoints(pts3d_w, rvec, t_cw.reshape(3, 1), K, None)
    return proj.reshape(-1, 2)


def draw_ba_reprojection(img_bgr, K,
                         T_wc_before, T_wc_after,
                         pts3d_before, pts3d_after, pts2d_meas,
                         win_name="BA reprojection", rad=4) -> np.ndarray:
    """
    Overlay *measured* KPs (white crosses), projections *before* BA (red),
    projections *after* BA (green).   An arrow connects before → after so
    you can see the correction visually.
    """
    img = img_bgr.copy()
    proj_before = _project(K, T_wc_before, pts3d_before)
    proj_after  = _project(K, T_wc_after,  pts3d_after)

    for (u_meas, v_meas), (u_b, v_b), (u_a, v_a) in zip(pts2d_meas,
                                                        proj_before,
                                                        proj_after):
        p_meas = (int(round(u_meas)), int(round(v_meas)))
        p_b    = (int(round(u_b)), int(round(v_b)))
        p_a    = (int(round(u_a)), int(round(v_a)))

        cv2.drawMarker(img, p_meas, (255,255,255), cv2.MARKER_CROSS, rad*2, 1)
        cv2.circle(img,   p_b, rad, (0,0,255),  -1)     # red  = before BA
        cv2.circle(img,   p_a, rad, (0,255,0), 2)      # green = after BA
        cv2.arrowedLine(img, p_b, p_a, (255,255,0), 1, tipLength=0.25)

    cv2.imshow(win_name, img)
    cv2.waitKey(1)
    return img

# --------------------------------------------------------------------------- #
#  BA-window visualiser (all key-frames in the sliding window)
# --------------------------------------------------------------------------- #
def visualize_ba_window(seq, args,
                        K,
                        world_map_before, world_map_after,
                        keypoints,
                        opt_kf_idx: list[int],
                        rad_px: int = 4) -> None:
    """
    For every key-frame in *opt_kf_idx* show an OpenCV window with:
        • measured kp (white cross),
        • projection **before** BA   (red dot),
        • projection **after**  BA   (green dot),
        • arrow before → after       (cyan).

    Each window is titled "BA KF <i>" – hit any key to close all.
    """
    from slam.core.dataloader import load_frame_pair   # avoids cyclic imports

    for k in opt_kf_idx:
        # --- 1) pull rgb -------------------------------------------------------
        # `load_frame_pair(args, seq, i)` returns (img_i , img_{i+1});
        # the KF itself is at i+1  → ask for i = k-1
        img_bgr, _ = load_frame_pair(args, seq, k-1)

        # --- 2) collect correspondences ---------------------------------------
        pts3d_b, uv = _gather_3d_2d(world_map_before, keypoints, k)
        if len(pts3d_b) == 0:          # nothing observed by this KF
            continue
        pts3d_a, _  = _gather_3d_2d(world_map_after,  keypoints, k)

        # Quantitative Error check
        err_b = np.linalg.norm(_project(K, world_map_before.poses[k], pts3d_b) - uv, axis=1)
        err_a = np.linalg.norm(_project(K, world_map_after.poses[k],  pts3d_a) - uv, axis=1)
        print(f'KF {k}:  mean reproj error  {err_b.mean():5.2f} px  →  {err_a.mean():5.2f} px')


        # --- 3) overlay -------------------------------------------------------
        draw_ba_reprojection(img_bgr, K,
                             world_map_before.poses[k],
                             world_map_after.poses[k],
                             pts3d_b, pts3d_a, uv,
                             win_name=f"BA KF {k}", rad=rad_px)
    cv2.waitKey(0)           # press any key once all windows are up
    cv2.destroyAllWindows()
