import numpy as np
import cv2

def triangulatePoints(points2d, projections, weights=None):
    """
    Triangulates 3D points from 2D correspondences using Direct Linear Transform (DLT).
    
    Args:
        points2d: List of 2D points (each as Nx2 array or Nx1x2/1xNx2 Mat).
        projections: List of 3x4 projection matrices.
        weights: Optional list of weights per view (default: uniform).
    
    Returns:
        points3d: Nx3 array of triangulated points.
    
    Raises:
        ValueError: If inputs are invalid or incompatible.
    """
    if len(points2d) != len(projections):
        raise ValueError("points2d and projections count must match.")
    
    # Convert inputs to standardized NumPy arrays
    pts_clean = [np.asarray(p).reshape(-1, 2) for p in points2d]
    n_points = len(pts_clean[0])
    projs = [np.asarray(M, dtype=np.float64) for M in projections]
    
    if weights is None:
        weights = [1.0] * len(projections)
    weights = np.asarray(weights, dtype=np.float64)
    
    points3d = np.zeros((n_points, 3), dtype=np.float64)
    
    for i in range(n_points):
        # Build equation system for the i-th point
        A = []
        for w, p, M in zip(weights, pts_clean, projs):
            x, y = p[i]
            A.append(w * (x * M[2] - M[0]))
            A.append(w * (y * M[2] - M[1]))
        A = np.stack(A, axis=0)
        
        # Solve via SVD (min ||Ax|| s.t. ||x||=1)
        _, _, Vh = np.linalg.svd(A)
        pt_homo = Vh[-1, :4]
        points3d[i] = (pt_homo[:3] / pt_homo[3]).reshape(3)  # Dehomogenize
    
    return points3d
