import numpy as np
from slam.core.pose_utils import _pose_inverse, project_to_SO3

def test_pose_inverse():
    R = project_to_SO3(np.random.randn(3,3))  # produce a valid rotation
    t = np.random.randn(3)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t
    T_inv = _pose_inverse(T)
    I = T_inv @ T
    assert np.allclose(I, np.eye(4), atol=1e-10)
