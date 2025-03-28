import numpy as np
from cv2.sfm import triangulatePoints
import pytest

def test_triangulate_simple():
    M1 = np.eye(3, 4, dtype=np.float32)  # Identity camera
    M2 = np.array([[1,0,0,1], [0,1,0,0], [0,0,1,0]], dtype=np.float32)  # Translated
    
    points2d = [
        np.array([[100, 200]], dtype=np.float32),  # View 1
        np.array([[50, 200]], dtype=np.float32)    # View 2
    ]
    
    points3d = triangulatePoints(points2d, [M1, M2])
    expected = np.array([[1.0, 2.0, 2.0]])
    assert np.allclose(points3d, expected, atol=1e-6)

def test_weights():
    M1 = np.eye(3, 4, dtype=np.float32)
    M2 = np.eye(3, 4, dtype=np.float32)  # Same camera (degenerate case)
    
    # Without weights: Should average positions
    points2d = [
        np.array([[10, 20]]),
        np.array([[30, 20]])  # Outlier
    ]
    
    # With weights: Downweight outlier
    points3d_noweights = triangulatePoints(points2d, [M1, M2])
    points3d_weighted = triangulatePoints(points2d, [M1, M2], weights=[1.0, 0.1])
    
    # Weighted result should be closer to first point
    assert abs(points3d_weighted[0,0] - 10) < abs(points3d_noweights[0,0] - 10)

def test_multiple_points():
    # Test batch processing of multiple points
    M1 = np.eye(3, 4)
    M2 = np.array([[1,0,0,1], [0,1,0,0], [0,0,1,0]])
    
    points2d = [
        np.array([[100, 200], [150, 250]]),
        np.array([[50, 200], [75, 250]])
    ]
    
    points3d = triangulatePoints(points2d, [M1, M2])
    assert points3d.shape == (2, 3)
    assert np.allclose(points3d[0], [1.0, 2.0, 2.0])

def test_input_validation():
    # Test invalid inputs
    with pytest.raises(ValueError):
        triangulatePoints([np.zeros((1,2))], [np.eye(3,4)])  # Only 1 view
    
    with pytest.raises(ValueError):
        triangulatePoints(
            [np.zeros((2,2)), np.zeros((3,2))],  # Mismatched point counts
            [np.eye(3,4), np.eye(3,4)]
        )
