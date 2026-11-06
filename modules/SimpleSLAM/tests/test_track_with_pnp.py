# tests/test_track_with_pnp.py
import os
import sys
import pathlib
import importlib
import numpy as np
import cv2

# --- Headless / GUI-off for tests ---
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass

# --- Put repo root (which contains 'slam/') on sys.path ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Import the module under test (supports both layouts) ---
try:
    main = importlib.import_module("slam.monocular.main")
except ImportError:
    # Fallback if you run tests inside slam/monocular/
    main = importlib.import_module("main")

track_with_pnp = getattr(main, "track_with_pnp")

# Disable visualization during tests
def _no_viz(*args, **kwargs):
    return
if hasattr(main, "visualize_pnp_reprojection"):
    setattr(main, "visualize_pnp_reprojection", _no_viz)

# ---------- Minimal stubs for world map / map points ----------
class _DummyPoint:
    def __init__(self, pid, xyz, desc=None):
        self.id = pid
        self.xyz = np.asarray(xyz, dtype=np.float32)
        self.observations = []
        self.rep_desc = np.zeros(32, np.uint8) if desc is None else np.asarray(desc, np.uint8)

    def add_observation(self, frame_no, kp_idx, desc):
        self.observations.append((int(frame_no), int(kp_idx)))
        self.rep_desc = np.asarray(desc, np.uint8)

class _DummyMap:
    def __init__(self, pts_w):
        self.points = [_DummyPoint(i, p) for i, p in enumerate(pts_w)]

    def point_ids(self):
        return list(range(len(self.points)))

    def get_point_array(self):
        return np.asarray([p.xyz for p in self.points], dtype=np.float32)

# --------------------- Helpers for the test ---------------------
def _K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1]], dtype=np.float64)

def _Twc(R=np.eye(3), t=np.zeros(3)):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T

def _project_points(K, Twc, pts_w):
    Tcw = np.linalg.inv(Twc)
    P = (K @ Tcw[:3, :4])
    pts_h = np.hstack([pts_w, np.ones((len(pts_w), 1))]).T
    uvw = P @ pts_h
    uv = (uvw[:2] / uvw[2]).T
    return uv.astype(np.float32)

def _make_keypoints(uv):
    return [cv2.KeyPoint(float(u), float(v), 1) for (u, v) in uv]

class _Args:
    def __init__(self,
                 proj_radius=12.0,
                 pnp_min_inliers=30,
                 ransac_thresh=3.0,
                 pnp_inlier_px=3.0):
        self.proj_radius = float(proj_radius)
        self.pnp_min_inliers = int(pnp_min_inliers)
        self.ransac_thresh = float(ransac_thresh)
        # some versions of main look for this explicitly
        self.pnp_inlier_px = float(pnp_inlier_px)

# --------------------------- Tests -----------------------------
def test_track_with_pnp_happy_path():
    rng = np.random.default_rng(42)
    K = _K()

    # Landmarks in world (3–8 m ahead)
    N = 150
    X = rng.uniform(-2.0, 2.0, size=N)
    Y = rng.uniform(-0.5, 0.5, size=N)
    Z = rng.uniform(3.0, 8.0, size=N)
    pts_w = np.stack([X, Y, Z], axis=1).astype(np.float32)

    # Prev pose at origin; GT pose: small forward motion + 1° yaw
    Twc_prev = _Twc()
    yaw = np.deg2rad(1.0)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]], dtype=np.float64)
    t_gt = np.array([0.02, -0.01, 0.12], dtype=np.float64)
    Twc_gt = _Twc(Rz, t_gt)

    # Synthetic current keypoints (GT projection + pixel noise)
    uv = _project_points(K, Twc_gt, pts_w)
    uv += rng.normal(0.0, 0.5, size=uv.shape).astype(np.float32)
    kp_cur = _make_keypoints(uv)

    # Prev-frame placeholders
    kp_prev = []
    desc_dim = 32
    desc_cur = rng.integers(0, 256, size=(len(kp_cur), desc_dim), dtype=np.uint8)
    desc_prev = np.empty((0, desc_dim), dtype=np.uint8)
    matches = []

    world_map = _DummyMap(pts_w)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)
    args = _Args()

    ok, Twc_est, used_idx = track_with_pnp(
        K,
        kp_prev, kp_cur, desc_prev, desc_cur, matches, img2,
        frame_no=1,
        Twc_prev=Twc_prev,
        world_map=world_map, args=args
    )

    assert ok, "PnP tracking should succeed"
    assert isinstance(Twc_est, np.ndarray) and Twc_est.shape == (4, 4)

    # Translation error (allow a few cm due to pixel noise)
    t_err = np.linalg.norm(Twc_est[:3, 3] - Twc_gt[:3, 3])
    assert t_err < 0.05, f"translation error too high: {t_err:.3f} m"

    # Rotation error
    dR = Twc_gt[:3, :3].T @ Twc_est[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    assert ang < 2.0, f"rotation error too high: {ang:.2f} deg"

    # Inliers / bookkeeping
    assert isinstance(used_idx, set)
    assert len(used_idx) >= args.pnp_min_inliers
    obs_total = sum(len(p.observations) for p in world_map.points)
    assert obs_total >= len(used_idx)

def test_track_with_pnp_not_enough_data():
    K = _K()
    world_map = _DummyMap(pts_w=np.empty((0, 3), dtype=np.float32))
    args = _Args()

    ok, Twc_est, used_idx = track_with_pnp(
        K,
        kp_prev=[], kp_cur=[], desc_prev=np.empty((0, 32), np.uint8),
        desc_cur=np.empty((0, 32), np.uint8), matches=[], img2=np.zeros((10, 10), np.uint8),
        frame_no=0, Twc_prev=_Twc(), world_map=world_map, args=args
    )

    assert not ok
    assert Twc_est is None
    assert isinstance(used_idx, set) and len(used_idx) == 0
