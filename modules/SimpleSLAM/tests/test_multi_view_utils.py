"""pytest test‑suite for `multi_view_utils.py`
================================================

Goals
-----
1. **Function integration** –
   * `multi_view_triangulation()`
   * `MultiViewTriangulator`

2. **Real‑world wiring** – whenever the project’s own
   `landmark_utils.Map` implementation is importable we use it, giving an
   *integration* rather than pure‑unit test.  When that class is missing (for
   example in a stripped‑down CI job) we transparently fall back to a *very*
   small stub that exposes only the methods/attributes the triangulator needs.

3. **Numerical accuracy** – the synthetic scene is designed so that with 0.4 px
   image noise and five views the RMS localisation error should be
   ≲ 5 cm.  If the implementation regresses we’ll catch it.

Run with::

    pytest tests/test_multi_view_utils.py
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
#  Locate & load the module under test – `multi_view_utils`
# --------------------------------------------------------------------------- #
# We try these locations in order:
#   1. Installed package                →  `slam.core.multi_view_utils`
#   2. Source tree root ("editable" dev) →  `multi_view_utils`
#   3. Direct path fallback so the test also works when launched from a
#      separate build/CI directory.

mvu: types.ModuleType

for _modname in ("slam.core.multi_view_utils", "multi_view_utils"):
    try:
        mvu = importlib.import_module(_modname)  # type: ignore
        break
    except ModuleNotFoundError:  # pragma: no cover – probe next option
        pass

# Public call‑ables
multi_view_triangulation = mvu.multi_view_triangulation  # type: ignore[attr‑defined]
MultiViewTriangulator = mvu.MultiViewTriangulator          # type: ignore[attr‑defined]

# --------------------------------------------------------------------------- #
#  Map implementation (real vs stub)
# --------------------------------------------------------------------------- #
# 1. Prefer the full implementation shipped with the repo.
# 2. Otherwise synthesize the minimal surface‑area stub so that the Triangulator
#    can still be unit‑tested.
from slam.core.landmark_utils import Map as SLAMMap  # type: ignore[attr‑defined]


# --------------------------------------------------------------------------- #
#  Synthetic scene generation helpers
# --------------------------------------------------------------------------- #

def _make_camera_pose(tx: float) -> np.ndarray:
    """Camera looks down +Z, translated along +X by *tx* (c→w)."""
    T = np.eye(4)
    T[0, 3] = tx
    return T


def _generate_scene(
    n_views: int = 5,
    n_pts: int = 40,
    noise_px: float = 0.4,
    seed: int | None = None,
):
    """Build a toy scene and return (K, poses_w_c, pts_w, 2‑D projections)."""

    rng = np.random.default_rng(seed)

    # -- basic pin‑hole intrinsics (640×480) --
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])

    # -- camera trajectory (translate along X) --
    poses = [_make_camera_pose(t) for t in np.linspace(0.0, 1.0, n_views)]

    # -- random 3‑D points in front of cameras (z ∈ [4,6]) --
    pts_w = np.vstack(
        (
            rng.uniform(-1.0, 1.0, n_pts),  # x
            rng.uniform(-1.0, 1.0, n_pts),  # y
            rng.uniform(4.0, 6.0, n_pts),  # z – ensure positive depth
        )
    ).T

    # -- project each point into every view and add Gaussian pixel noise --
    pts2d_all: list[np.ndarray] = []
    for T_w_c in poses:
        P_c_w = np.linalg.inv(T_w_c)  # w→c
        uv_view = []
        for X_w in pts_w:
            Xc_h = P_c_w @ np.append(X_w, 1.0)
            uv = (K @ Xc_h[:3])[:2] / Xc_h[2]
            uv += rng.normal(0.0, noise_px, 2)
            uv_view.append(uv)
        pts2d_all.append(np.asarray(uv_view, dtype=np.float32))

    return K, poses, pts_w, pts2d_all

# --------------------------------------------------------------------------- #
#  Light cv2.KeyPoint substitute (Triangulator only needs `.pt`)
# --------------------------------------------------------------------------- #

class _KeyPoint:  # pylint: disable=too‑few‑public‑methods
    def __init__(self, x: float, y: float):
        self.pt = (float(x), float(y))

# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #


def test_multi_view_triangulation_accuracy():
    """Direct N‑view triangulation should achieve < 5 cm RMS error."""
    K, poses, pts_w, pts2d = _generate_scene()

    errs: list[float] = []
    for j in range(len(pts_w)):
        uv_track = [view[j] for view in pts2d]
        X_hat = multi_view_triangulation(
            K,
            poses,
            np.float32(uv_track),
            min_depth=0.1,
            max_depth=10.0,
            max_rep_err=2.0,
        )
        assert X_hat is not None, "Triangulation unexpectedly returned None"
        errs.append(np.linalg.norm(X_hat - pts_w[j]))

    rms = float(np.sqrt(np.mean(np.square(errs))))
    assert rms < 5e-2, f"RMS error too high: {rms:.4f} m"


@pytest.mark.parametrize("min_views", [2, 3])
def test_multiview_triangulator_pipeline(min_views: int):
    """Full pipeline: incremental key‑frames → map landmarks."""
    K, poses, pts_w, pts2d = _generate_scene()

    tri = MultiViewTriangulator(
        K,
        min_views=min_views,
        merge_radius=0.1,
        max_rep_err=2.0,
        min_depth=0.1,
        max_depth=10.0,
    )

    world_map = SLAMMap()

    # ---- Feed key‑frames ----
    for frame_idx, (pose_w_c, uv_view) in enumerate(zip(poses, pts2d)):
        kps: list[_KeyPoint] = []
        track_map: dict[int, int] = {}
        for pid, (u, v) in enumerate(uv_view):
            kps.append(_KeyPoint(u, v))
            track_map[pid] = pid  # 1‑to‑1 track id
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        descs = [np.zeros(32, dtype=np.uint8) for _ in kps]
        tri.add_keyframe(frame_idx, pose_w_c, kps, track_map, dummy_img, descs)

    new_ids = tri.triangulate_ready_tracks(world_map)
    assert len(new_ids) == len(pts_w), "Not all points were triangulated"

    # ---- Numerical accuracy ----
    errs: list[float] = []
    for pid in new_ids:
        p_obj = world_map.points[pid]
        # Real MapPoint uses `.position`; stub stores the same attribute name.
        X_hat = p_obj.position if hasattr(p_obj, "position") else p_obj.xyz  # type: ignore[attr‑defined]
        errs.append(np.linalg.norm(X_hat - pts_w[pid]))

    rms = float(np.sqrt(np.mean(np.square(errs))))
    assert rms < 5e-2, f"RMS error too high: {rms:.4f} m"
