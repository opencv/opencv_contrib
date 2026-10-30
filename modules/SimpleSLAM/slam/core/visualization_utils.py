# visualization_utils.py
from __future__ import annotations
"""
visualization_utils.py
~~~~~~~~~~~~~~~~~~~~~~
Clean, modular utilities for
* drawing 2‑D track overlays with OpenCV, and
* rendering a coloured 3‑D map in an **Open3D** window.

Main entry‑points
-----------------
``draw_tracks(img, tracks, frame_no)``
    Overlay recent feature tracks.

``Visualizer3D``
    Live window that shows
      • the SLAM point cloud, colour‑coded along an axis or PCA‑auto;
      • the camera trajectory (blue line);
      • new landmarks highlighted (default bright‑green).
    Supports WASDQE first‑person navigation when the Open3D build exposes
    key‑callback APIs.
"""

from typing import Dict, List, Tuple, Optional, Literal
import warnings, threading, cv2, numpy as np

import cv2
import numpy as np

try:
    import open3d as o3d  # type: ignore
except Exception as exc:  # pragma: no cover
    o3d = None  # type: ignore
    _OPEN3D_ERR = exc
else:
    _OPEN3D_ERR = None

from slam.core.landmark_utils import Map
import numpy as np

ColourAxis = Literal["x", "y", "z", "auto"]

# --------------------------------------------------------------------------- #
#  3‑D visualiser  (Open3D only)                                             #
# --------------------------------------------------------------------------- #

class Visualizer3D:
    """Open3D window that shows the coloured point-cloud and camera path."""

    def __init__(
        self,
        color_axis: ColourAxis = "z",
        *,
        new_colour: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        window_size: Tuple[int, int] = (1280, 720),
        nav_step: float = 0.25,
    ) -> None:
        self.backend  = "none"
        self._closed  = False
        self._lock    = threading.Lock()
        self.paused   = False

        self.color_axis = color_axis
        self.new_colour = np.asarray(new_colour, dtype=np.float32)
        self.nav_step   = nav_step

        # scalar-to-colour normalisation
        self._v_min: Optional[float] = None
        self._v_max: Optional[float] = None
        self._pc_vec: Optional[np.ndarray] = None  # PCA axis for "auto"

        # ------------------------------------------------------------------ #
        #  Open3D init
        # ------------------------------------------------------------------ #
        if o3d is None:
            warnings.warn(f"[Visualizer3D] Open3D missing → window disabled ({_OPEN3D_ERR})")
            return

        vis_cls = (
            o3d.visualization.VisualizerWithKeyCallback
            if hasattr(o3d.visualization, "VisualizerWithKeyCallback")
            else o3d.visualization.Visualizer
        )
        self.vis = vis_cls()
        self.vis.create_window("SLAM Map", width=window_size[0], height=window_size[1])
        self.backend = "open3d"

        self.pcd   = o3d.geometry.PointCloud()
        self.lines = o3d.geometry.LineSet()
        self._first = True

        if isinstance(self.vis, o3d.visualization.VisualizerWithKeyCallback):
            self._bind_nav_keys()

        print(f"[Visualizer3D] ready | colour_axis={self.color_axis}")

    # ------------------------------------------------------------------ #
    #  WASDQE first-person navigation
    # ------------------------------------------------------------------ #
    def _bind_nav_keys(self):
        vc = self.vis.get_view_control()

        def translate(delta: np.ndarray):
            cam = vc.convert_to_pinhole_camera_parameters()
            T = np.eye(4);  T[:3, 3] = delta * self.nav_step
            cam.extrinsic = T @ cam.extrinsic
            vc.convert_from_pinhole_camera_parameters(cam)
            return False

        key_map = {
            ord("W"): np.array([0, 0, -1]),
            ord("S"): np.array([0, 0,  1]),
            ord("A"): np.array([-1, 0, 0]),
            ord("D"): np.array([ 1, 0, 0]),
            ord("Q"): np.array([0,  1, 0]),
            ord("E"): np.array([0, -1, 0]),
        }
        for k, v in key_map.items():
            self.vis.register_key_callback(k, lambda _v, vec=v: translate(vec))

    # ------------------------------------------------------------------ #
    #  Helpers for colour mapping
    # ------------------------------------------------------------------ #
    def _compute_scalar(self, pts: np.ndarray) -> np.ndarray:
        if self.color_axis in ("x", "y", "z"):
            return pts[:, {"x": 0, "y": 1, "z": 2}[self.color_axis]]
        if self._pc_vec is None:                 # first call → PCA axis
            centred = pts - pts.mean(0)
            _, _, vh = np.linalg.svd(centred, full_matrices=False)
            self._pc_vec = vh[0]
        return pts @ self._pc_vec

    def _normalise(self, scalars: np.ndarray) -> np.ndarray:
        if self._v_min is None:                  # initialise 5th–95th perc.
            self._v_min, self._v_max = np.percentile(scalars, [5, 95])
        else:                                    # expand running min / max
            self._v_min = min(self._v_min, scalars.min())
            self._v_max = max(self._v_max, scalars.max())
        return np.clip((scalars - self._v_min) / (self._v_max - self._v_min + 1e-6), 0, 1)

    def _colormap(self, norm: np.ndarray) -> np.ndarray:
        try:
            import matplotlib.cm as cm
            return cm.get_cmap("turbo")(norm)[:, :3]
        except Exception:
            # fall-back: simple HSV → RGB
            h = (1 - norm) * 240
            c = np.ones_like(h); m = np.zeros_like(h)
            x = c * (1 - np.abs((h / 60) % 2 - 1))
            return np.select(
                [h < 60, h < 120, h < 180, h < 240, h < 300, h >= 300],
                [
                    np.stack([c, x, m], 1), np.stack([x, c, m], 1),
                    np.stack([m, c, x], 1), np.stack([m, x, c], 1),
                    np.stack([x, m, c], 1), np.stack([c, m, x], 1),
                ])

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #
    def update(self, slam_map: Map, new_ids: Optional[List[int]] = None):
        if self.backend != "open3d" or len(slam_map.points) == 0:
            return
        if self.paused:
            with self._lock:
                self.vis.poll_events(); self.vis.update_renderer()
            return

        # -------------------------- build numpy arrays -------------------------
        pts = slam_map.get_point_array()
        col = slam_map.get_color_array() if hasattr(slam_map, "get_color_array") else None
        if col is None or len(col) == 0:            # legacy maps without colour
            scal  = self._compute_scalar(pts)
            col   = self._colormap(self._normalise(scal))
        else:
            col = col.astype(np.float32)

        # pts = slam_map.get_point_array()
        # col = slam_map.get_color_array() if hasattr(slam_map, "get_color_array") else None

        # use_fallback = (
        #     col is None
        #     or len(col) == 0
        #     or (isinstance(col, np.ndarray) and col.ndim == 2 and col.shape[1] == 3
        #         and np.allclose(col.std(axis=0), 0, atol=1e-6))  # colours are (nearly) uniform
        # )

        # if use_fallback:
        #     # vertical gradient (axis set by color_axis, you pass "y" at construction)
        #     scal = self._compute_scalar(pts)
        #     col  = self._colormap(self._normalise(scal)).astype(np.float32)
        # else:
        #     col = col.astype(np.float32)


        # keep arrays in sync (pad / trim)
        if len(col) < len(pts):
            diff = len(pts) - len(col)
            col  = np.vstack([col, np.full((diff, 3), 0.8, np.float32)])
        elif len(col) > len(pts):
            col  = col[: len(pts)]

        # highlight newly-added landmarks
        if new_ids:
            id_to_i = {pid: i for i, pid in enumerate(slam_map.point_ids())}
            for nid in new_ids:
                if nid in id_to_i:
                    col[id_to_i[nid]] = self.new_colour

        # ----------------------- Open3D geometry update ------------------------
        with self._lock:
            self.pcd.points  = o3d.utility.Vector3dVector(pts)
            self.pcd.colors  = o3d.utility.Vector3dVector(col)

            self._update_lineset(slam_map)

            if self._first:
                self.vis.add_geometry(self.pcd); self.vis.add_geometry(self.lines)
                self._first = False
            else:
                self.vis.update_geometry(self.pcd); self.vis.update_geometry(self.lines)

            self.vis.poll_events(); self.vis.update_renderer()

    # ------------------------------------------------------------------ #
    #  Blue camera trajectory poly-line
    # ------------------------------------------------------------------ #
    def _update_lineset(self, slam_map: Map):
        if len(slam_map.poses) < 2:
            return
        path = np.asarray([p[:3, 3] for p in slam_map.poses], np.float32)
        self.lines.points = o3d.utility.Vector3dVector(path)
        self.lines.lines  = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(path) - 1)])
        self.lines.colors = o3d.utility.Vector3dVector(np.tile([[0, 0, 1]], (len(path) - 1, 1)))

    # ------------------------------------------------------------------ #
    #  Clean shutdown
    # ------------------------------------------------------------------ #
    def close(self):
        if self.backend == "open3d" and not self._closed:
            self.vis.destroy_window(); self._closed = True
# --------------------------------------------------------------------------- #
#  2‑D overlay helpers
# --------------------------------------------------------------------------- #

def draw_tracks(
    vis: np.ndarray,
    tracks: Dict[int, List[Tuple[int, int, int]]],
    current_frame: int,
    max_age: int = 10,
    sample_rate: int = 5,
    max_tracks: int = 100,
) -> np.ndarray:
    """Draw ageing feature tracks as fading polylines.

    Parameters
    ----------
    vis          : BGR uint8 image (modified *in‑place*)
    tracks       : {track_id: [(frame_idx,x,y), ...]}
    current_frame: index of the frame being drawn
    max_age      : only show segments younger than this (#frames)
    sample_rate  : skip tracks where `track_id % sample_rate != 0` to avoid clutter
    max_tracks   : cap total rendered tracks for speed
    """
    recent = [
        (tid, pts)
        for tid, pts in tracks.items()
        if current_frame - pts[-1][0] <= max_age
    ]
    recent.sort(key=lambda x: x[1][-1][0], reverse=True)

    drawn = 0
    for tid, pts in recent:
        if drawn >= max_tracks:
            break
        if tid % sample_rate:
            continue
        pts = [p for p in pts if current_frame - p[0] <= max_age]
        for j in range(1, len(pts)):
            _, x0, y0 = pts[j - 1]
            _, x1, y1 = pts[j]
            ratio = (current_frame - pts[j - 1][0]) / max_age
            colour = (0, int(255 * (1 - ratio)), int(255 * ratio))
            cv2.line(vis, (x0, y0), (x1, y1), colour, 2)
        drawn += 1
    return vis


# --------------------------------------------------------------------------- #
#  Lightweight 2-D trajectory plotter (Matplotlib)
# --------------------------------------------------------------------------- #

# ---- Trajectory 2D (x–z) + simple pause UI ----
class Trajectory2D:
    def __init__(self, gt_T_list=None, win="Trajectory 2D (x–z)"):
        self.win = win
        self.gt_T = gt_T_list  # list of 4x4 Twc (or None)
        self.est_xyz = []      # estimated camera centers (world)
        self.gt_xyz  = []      # paired GT centers
        self.align_ok = False
        self.s = 1.0
        self.R = np.eye(3)
        self.t = np.zeros(3)

        # ensure a real window exists (Qt backend is happier with this)
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 500, 500)

    @staticmethod
    def _cam_center_from_Tcw(Tcw: np.ndarray) -> np.ndarray:
        R, t = Tcw[:3, :3], Tcw[:3, 3]
        return (-R.T @ t).astype(np.float64)

    def _maybe_update_alignment(self, Kpairs=60):
        if len(self.est_xyz) < 6 or len(self.gt_xyz) < 6:
            return
        X = np.asarray(self.gt_xyz[-Kpairs:], np.float64)   # GT
        Y = np.asarray(self.est_xyz[-Kpairs:], np.float64)  # EST
        muX, muY = X.mean(0), Y.mean(0)
        X0, Y0 = X - muX, Y - muY
        cov = (Y0.T @ X0) / X.shape[0]
        U, S, Vt = np.linalg.svd(cov)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        varY = (Y0**2).sum() / X.shape[0]
        s = (S * np.diag(D)).sum() / (varY + 1e-12)
        t = muX - s * (R @ muY)
        self.s, self.R, self.t = float(s), R, t
        self.align_ok = True

    def push(self, frame_idx: int, Tcw: np.ndarray):
        self.est_xyz.append(self._cam_center_from_Tcw(Tcw))
        if self.gt_T is not None and 0 <= frame_idx < len(self.gt_T):
            self.gt_xyz.append(self.gt_T[frame_idx][:3, 3].astype(np.float64))
        self._maybe_update_alignment(Kpairs=min(100, len(self.est_xyz)))

    def draw(self, paused=False, size=(720, 720), margin=60):
        W, H = size
        canvas = np.full((H, W, 3), 255, np.uint8)

        # nothing to draw yet
        if not self.est_xyz:
            cv2.imshow(self.win, canvas)
            return

        E = np.asarray(self.est_xyz, np.float64)
        if self.align_ok:
            E = (self.s * (self.R @ E.T)).T + self.t

        curves = [("estimate", E, (255, 0, 0))]  # BGR: blue
        have_gt = len(self.gt_xyz) > 0
        if have_gt:
            G = np.asarray(self.gt_xyz, np.float64)
            curves.append(("ground-truth", G, (0, 0, 255)))    # red
            allpts = np.vstack([E[:, [0, 2]], G[:, [0, 2]]])
        else:
            allpts = E[:, [0, 2]]

        # ----- nice bounds with padding -----
        minx, minz = allpts.min(axis=0); maxx, maxz = allpts.max(axis=0)
        pad_x = 0.05 * max(1e-6, maxx - minx)
        pad_z = 0.05 * max(1e-6, maxz - minz)
        minx -= pad_x; maxx += pad_x
        minz -= pad_z; maxz += pad_z

        spanx = max(maxx - minx, 1e-6)
        spanz = max(maxz - minz, 1e-6)
        sx = (W - 2 * margin) / spanx
        sz = (H - 2 * margin) / spanz
        s = min(sx, sz)

        def to_px(x, z):
            u = int((x - minx) * s + margin)
            v = int((maxz - z) * s + margin)  # flip z for image y
            return u, v

        # ----- grid + ticks (matplotlib-ish) -----
        def linspace_ticks(a, b, m=6):
            # simple ticks; good enough for live viz
            return np.linspace(a, b, m)

        ticks_x = linspace_ticks(minx, maxx, 6)
        ticks_z = linspace_ticks(minz, maxz, 6)

        # draw background grid
        for x in ticks_x:
            u0, v0 = to_px(x, minz)
            u1, v1 = to_px(x, maxz)
            cv2.line(canvas, (u0, v0), (u1, v1), (235, 235, 235), 1, cv2.LINE_AA)
        for z in ticks_z:
            u0, v0 = to_px(minx, z)
            u1, v1 = to_px(maxx, z)
            cv2.line(canvas, (u0, v0), (u1, v1), (235, 235, 235), 1, cv2.LINE_AA)

        # axis box
        cv2.rectangle(canvas, (margin-1, margin-1), (W-margin, H-margin), (200, 200, 200), 1)

        # tick labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        for x in ticks_x:
            u, v = to_px(x, minz)
            cv2.putText(canvas, f"{x:.1f}", (u-14, H - margin + 20), font, 0.4, (0,0,0), 1, cv2.LINE_AA)
        for z in ticks_z:
            u, v = to_px(minx, z)
            cv2.putText(canvas, f"{z:.1f}", (margin - 50, v+4), font, 0.4, (0,0,0), 1, cv2.LINE_AA)

        # axis labels
        cv2.putText(canvas, "x", (W//2, H - 10), font, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "z", (10, H//2), font, 0.6, (0,0,0), 1, cv2.LINE_AA)

        # ----- draw curves -----
        for name, C, color in curves:
            if C.shape[0] < 2:
                # single point
                u, v = to_px(C[-1, 0], C[-1, 2])
                cv2.circle(canvas, (u, v), 3, color, -1, cv2.LINE_AA)
            else:
                pts = [to_px(x, z) for (x, _, z) in C]
                for p, q in zip(pts[:-1], pts[1:]):
                    cv2.line(canvas, p, q, color, 2, cv2.LINE_AA)
                cv2.circle(canvas, pts[-1], 3, color, -1, cv2.LINE_AA)

        # legend
        legend_x, legend_y = margin + 10, margin + 20
        for idx, (name, _, color) in enumerate(curves):
            y = legend_y + idx * 20
            cv2.line(canvas, (legend_x, y), (legend_x + 30, y), color, 3, cv2.LINE_AA)
            cv2.putText(canvas, name, (legend_x + 40, y + 4), font, 0.5, (60,60,60), 1, cv2.LINE_AA)

        # title and paused hint
        cv2.putText(canvas, "Trajectory 2D (x–z)", (margin, 30), font, 0.7, (0,0,0), 2, cv2.LINE_AA)
        if paused:
            cv2.putText(canvas, "PAUSED  [p: resume | n: step | q/Esc: quit]",
                        (margin, H - 15), font, 0.5, (20,20,20), 2, cv2.LINE_AA)

        cv2.imshow(self.win, canvas)


# --------------------------------------------------------------------------- #
#  UI for Pausing and stepping through the pipeline
# --------------------------------------------------------------------------- #
class VizUI:
    """Tiny UI state for pausing/stepping the pipeline."""
    def __init__(self, pause_key='p', step_key='n', quit_keys=('q', 27)):
        self.pause_key = self._to_code(pause_key)
        self.step_key  = self._to_code(step_key)
        self.quit_keys = {self._to_code(k) for k in (quit_keys if isinstance(quit_keys, (tuple, list, set)) else (quit_keys,))}
        self.paused = False
        self._request_quit = False
        self._do_step = False

    @staticmethod
    def _to_code(k):
        return k if isinstance(k, int) else ord(k)

    def poll(self, delay_ms=1):
        k = cv2.waitKey(delay_ms) & 0xFF
        if k == 255:  # no key
            return
        if k in self.quit_keys:
            self._request_quit = True
            return
        if k == self.pause_key:
            self.paused = not self.paused
            self._do_step = False
            return
        if k == self.step_key:
            self._do_step = True
            return

    def should_quit(self):
        return self._request_quit

    def wait_if_paused(self):
        """Block while paused, but allow 'n' to step one iteration."""
        if not self.paused:
            return False  # not blocking
        while True:
            k = cv2.waitKey(30) & 0xFF
            if k == self.pause_key:  # resume
                self.paused = False
                return False
            if k in self.quit_keys:
                self._request_quit = True
                return False
            if k == self.step_key:
                # allow one iteration to run, remain paused afterward
                self._do_step = True
                return True  # consume one step

    def consume_step(self):
        """Return True once if a single-step was requested."""
        if self._do_step:
            self._do_step = False
            return True
        return False

