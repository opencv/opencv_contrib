import numpy as np
from scipy.spatial.transform import Rotation


def project_to_SO3(M):
    """
    Helper function - Projects a near-rotation 3x3 matrix M onto SO(3) via SVD.
    Returns the closest rotation (Frobenius norm).
    """
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def _pose_inverse(T, validate=True):
    """
    Invert a 4x4 homogeneous rigid transform using SciPy. Return 4×4 inverse of a rigid‑body transform specified by R|t

    Parameters
    ----------
    T : array-like (4,4)
        Homogeneous transform [[R, t],[0,0,0,1]].
    validate : bool
        If True, re-project rotation onto SO(3) via SciPy (robust to mild drift).

    Returns
    -------
    T_inv : (4,4) ndarray
        Inverse transform.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4,4):
        raise ValueError("T must be 4x4.")
    Rmat = T[:3,:3]
    t    = T[:3, 3]

    if validate:
        # Produces closest rotation in Frobenius norm
        # Re-projects matrix to ensure perfect orthonormality
        Rmat = project_to_SO3(Rmat)

    R_inv = Rmat.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3,:3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def _pose_rt_to_homogenous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert relative pose (R, t) to Homogeneous matrix  **T_c1→c2**."""
    T = np.eye(4)
    T[:3, :3]  = R
    T[:3, 3]   = t.ravel()
    return T

# ********************************************************************************************
# BA HELPER FUNCTIONS : Small pose ⇄ parameter converters
# ********************************************************************************************

def _pose_to_quat_trans(T, ordering="xyzw"): # pyceres uses maybe (xyzw), optionally you could use "wxyz" 
    """
    Convert a camera-from-world pose matrix T_cw (4x4) into (quat_cw, t_cw)

    Parameters
    ----------
    T : (4,4) array-like
        Homogeneous transform.
    ordering : str
        "wxyz" (default) or "xyzw".
    Returns
    -------
    q : (4,) ndarray
        Quaternion in requested ordering, unit norm (w>=0 if wxyz).
    t : (3,) ndarray
        Translation vector.
    """
    T = np.asarray(T, dtype=float)
    assert T.shape == (4,4)
    Rmat = T[:3,:3]
    t = T[:3,3].copy()
    
    # Re-orthonormalize (optional but good hygiene)
    U, _, Vt = np.linalg.svd(Rmat)
    Rmat = U @ Vt
    if np.linalg.det(Rmat) < 0:     # enforce right-handed
        U[:, -1] *= -1
        Rmat = U @ Vt
    
    rot = Rotation.from_matrix(Rmat)
    # SciPy gives quaternions in (x, y, z, w)
    q_xyzw = rot.as_quat()
    
    # Normalize (usually already unit)
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    
    if ordering.lower() == "xyzw":
        q = q_xyzw
        # optional consistent sign: enforce w>=0
        if q[-1] < 0: q = -q
    else:  # wxyz
        w = q_xyzw[3]
        q = np.array([w, *q_xyzw[:3]])
        if q[0] < 0: q = -q
    return q, t

def _quat_trans_to_pose(q, t, ordering="xyzw"):
    q = np.asarray(q, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    if ordering.lower() == "wxyz":
        w, x, y, z = q
        q_xyzw = np.array([x, y, z, w])
    else:
        x, y, z, w = q
        q_xyzw = q
    # Normalize in case
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    Rmat = Rotation.from_quat(q_xyzw).as_matrix()
    T = np.eye(4)
    T[:3,:3] = Rmat
    T[:3,3] = t
    return T
