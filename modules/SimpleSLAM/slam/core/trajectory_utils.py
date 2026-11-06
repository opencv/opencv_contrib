"""
trajectory_utils.py
~~~~~~~~~~~~~~~~~~~
Small utilities for aligning, transforming and plotting camera
trajectories.

This file contains *no* project-specific imports, so it can be reused
from notebooks or quick scripts without pulling the whole SLAM stack.
"""
from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
#  Alignment helpers
# ------------------------------------------------------------------ #
def compute_gt_alignment(gt_T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (R_align, t_align) such that

        p_slam = R_align @ p_gt + t_align

    maps *ground-truth* positions into the SLAM world used in this
    project (camera-0 frame at t=0).

    Parameters
    ----------
    gt_T : (N,4,4)  homogeneous ground-truth poses, first one defines the
           reference frame.

    Notes
    -----
    *KITTI* supplies (3×4) poses whose *first* matrix is *already*
    camera-0 wrt world, therefore the alignment is an identity – but we
    keep it generic so other datasets work too.
    """
    if gt_T.ndim != 3 or gt_T.shape[1:] != (4, 4):
        raise ValueError("gt_T must be (N,4,4)")
    R0, t0 = gt_T[0, :3, :3], gt_T[0, :3, 3]
    R_align = np.eye(3) @ R0.T
    t_align = -R_align @ t0
    return R_align, t_align


def apply_alignment(p_gt: np.ndarray,
                    R_align: np.ndarray,
                    t_align: np.ndarray) -> np.ndarray:
    """Transform *one* 3-D point from GT to SLAM coordinates."""
    return R_align @ p_gt + t_align
