# tests/test_landmark_utils.py
"""
Unit-tests for landmark_utils.py.

We assume:
  * pytest, numpy and scipy are available.
  * landmark_utils.py sits on the Python path.
"""
from __future__ import annotations

import sys
import types
import numpy as np
import pytest

import slam.core.landmark_utils as lu

# --------------------------------------------------------------------------- #
#  Helpers / fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_points() -> np.ndarray:
    """A simple, well-conditioned 3-D point cloud (float64)."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def map_with_points(sample_points) -> lu.Map:
    """A map pre-populated with 4 landmarks."""
    m = lu.Map()
    m.add_points(sample_points)
    return m


# --------------------------------------------------------------------------- #
#  MapPoint
# --------------------------------------------------------------------------- #
def test_mappoint_add_observation():
    mp = lu.MapPoint(id=0, position=np.zeros(3))

    # --- new: fabricate two dummy 32-byte descriptors -------------------- #
    desc1 = np.arange(32, dtype=np.uint8)
    desc2 = np.arange(32, dtype=np.uint8)[::-1]

    mp.add_observation(frame_idx=5, kp_idx=17, descriptor=desc1)
    mp.add_observation(frame_idx=6, kp_idx=3,  descriptor=desc2)

    # (frame_idx, kp_idx) bookkeeping should still be correct
    assert [(f, k) for f, k, _ in mp.observations] == [(5, 17), (6, 3)]

    # descriptors must be stored verbatim (and by value, not reference)
    np.testing.assert_array_equal(mp.observations[0][2], desc1)
    np.testing.assert_array_equal(mp.observations[1][2], desc2)

# --------------------------------------------------------------------------- #
#  Map-level functionality
# --------------------------------------------------------------------------- #
def test_add_points_and_accessors(sample_points):
    m = lu.Map()

    ids = m.add_points(sample_points)  # default colours
    assert ids == list(range(4)), "Returned ids should be consecutive from 0"
    assert len(m) == 4, "__len__ should reflect number of landmarks"

    # geometry
    np.testing.assert_array_equal(m.get_point_array(), sample_points)

    # default colours should be all-ones, float32
    cols = m.get_color_array()
    assert cols.dtype == np.float32
    np.testing.assert_array_equal(cols, np.ones_like(sample_points, dtype=np.float32))

    # explicit colours
    custom_cols = np.random.rand(4, 3).astype(np.float32)
    m2 = lu.Map()
    m2.add_points(sample_points, colours=custom_cols)
    np.testing.assert_array_equal(m2.get_color_array(), custom_cols)


def test_add_points_invalid_shape():
    m = lu.Map()
    bad = np.zeros((5, 2))
    with pytest.raises(ValueError):
        m.add_points(bad)


def test_add_pose_and_bad_shape():
    m = lu.Map()
    good_pose = np.eye(4)
    m.add_pose(good_pose)
    assert len(m.poses) == 1
    np.testing.assert_array_equal(m.poses[0], good_pose)

    bad_pose = np.eye(3)
    with pytest.raises(AssertionError):
        m.add_pose(bad_pose)


def test_point_ids_and_length(sample_points):
    m = lu.Map()
    m.add_points(sample_points)
    assert m.point_ids() == [0, 1, 2, 3]
    assert len(m) == 4


# --------------------------------------------------------------------------- #
#  Duplicate-landmark fusion
# --------------------------------------------------------------------------- #
def test_fuse_closeby_duplicate_landmarks_merging():
    m = lu.Map()
    pts = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],  # within 5 cm of the first ⇒ should merge
            [1.00, 0.00, 0.00],
        ]
    )
    m.add_points(pts)

    m.fuse_closeby_duplicate_landmarks(radius=0.05)

    assert len(m) == 2, "Two nearby points should have merged into one"

    # the surviving landmark at id=0 should now sit at their mean position
    np.testing.assert_allclose(
        m.points[0].position, np.array([0.01, 0.0, 0.0]), atol=1e-6
    )


def test_fuse_closeby_duplicate_landmarks_no_merging(sample_points):
    m = lu.Map()
    m.add_points(sample_points)
    m.fuse_closeby_duplicate_landmarks(radius=0.01)  # radius too small
    assert len(m) == 4, "No points should merge when radius is tiny"
