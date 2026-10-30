import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

def adjust_bundle(camera_array, points_3d, points_2d, camera_indices, point_indices, num_cameras, num_points):
    """Intializes all the class attributes and instance variables.
        Write the specifications for each variable:

        cameraArray with shape (n_cameras, 9) contains initial estimates of parameters for all cameras.
                First 3 components in each row form a rotation vector,
                next 3 components form a translation vector,
                then a focal distance and two distortion parameters.

        points_3d with shape (n_points, 3)
                contains initial estimates of point coordinates in the world frame.

        camera_ind with shape (n_observations,)
                contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.

        point_ind with shape (n_observations,)
                contatins indices of points (from 0 to n_points - 1) involved in each observation.

        points_2d with shape (n_observations, 2)
                contains measured 2-D coordinates of points projected on images in each observations.
    """
    def rotate(points, rot_vecs):
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def project(points, cameraArray):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        # f = cameraArray[:, 6]
        # k1 = cameraArray[:, 7]
        # k2 = cameraArray[:, 8]
        # n = np.sum(points_proj ** 2, axis=1)
        # r = 1 + k1 * n + k2 * n ** 2
        # points_proj *= (r * f)[:, np.newaxis]
        return points_proj

    def fun(params, n_cameras, num_points, camera_indices, point_indices, points_2d):
        camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = params[n_cameras * 9:].reshape((num_points, 3))
        points_proj = project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        n = numCameras * 9 + numPoints * 3
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(cameraIndices.size)
        for s in range(9):
            A[2 * i, cameraIndices * 9 + s] = 1
            A[2 * i + 1, cameraIndices * 9 + s] = 1
        for s in range(3):
            A[2 * i, numCameras * 9 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 9 + pointIndices * 3 + s] = 1
        return A
    
    def optimizedParams(params, num_cameras, num_points):
        camera_params = params[:num_cameras * 9].reshape((num_cameras, 9))
        points_3d = params[num_cameras * 9:].reshape((num_points, 3))
        return camera_params, points_3d
    
    x0 = np.hstack((camera_array.ravel(), points_3d.ravel()))
    A = bundle_adjustment_sparsity(num_cameras, num_points, camera_indices, point_indices)
    res = least_squares(fun, x0, ftol=1e-2, method='lm',
                        args=(num_cameras, num_points, camera_indices, point_indices, points_2d))
    camera_params, points_3d = optimizedParams(res.x, num_cameras, num_points)
    return camera_params, points_3d