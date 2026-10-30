# dataloader.py

import os
import glob
import cv2
import numpy as np
import pickle
from typing import List, Optional, Dict, Tuple
import pandas as pd

def _parse_tum_rgb_list(txt_path: str, seq_dir: str) -> Tuple[List[str], List[float]]:
    """Parse TUM rgb.txt returning (paths, timestamps)."""
    paths, stamps = [], []
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ts_str, rel = line.strip().split()
            stamps.append(float(ts_str))
            paths.append(os.path.join(seq_dir, rel))
    return paths, stamps

def load_sequence(args) -> List[str]:
    """
    Return a list `seq` of either on-disk image paths or in-memory BGR frames.
    (Exactly as before—only left / monocular images.)
    """
    # TODO: also have to add the ability to load a video file
    # TODO: add the ability to load an entire dataset various sequences
    prefix = os.path.join(args.base_dir, args.dataset)

    if args.dataset == 'kitti':
        img_dir, pat = os.path.join(prefix, '05', 'image_0'), '*.png'
        seq = sorted(glob.glob(os.path.join(img_dir, pat)))

    elif args.dataset == 'parking':
        img_dir, pat = os.path.join(prefix, 'images'), '*.png'
        seq = sorted(glob.glob(os.path.join(img_dir, pat)))

    elif args.dataset == 'malaga':
        img_dir = os.path.join(prefix,
            'malaga-urban-dataset-extract-07_rectified_800x600_Images')
        seq = sorted(glob.glob(os.path.join(img_dir, '*_left.jpg')))

    elif args.dataset == 'tum-rgbd':
        img_dir = os.path.join(prefix, 'rgbd_dataset_freiburg1_room', 'rgb')
        seq = sorted(glob.glob(os.path.join(img_dir, '*.png')))

    elif args.dataset == 'custom':
        vid = os.path.join(prefix, 'custom_compress.mp4')
        cap = cv2.VideoCapture(vid)
        seq = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            seq.append(fr)
        cap.release()

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if len(seq) < 2:
        raise RuntimeError("Dataset must contain at least two frames.")
    return seq


def load_frame_pair(args, seq, i):
    """
    Convenience wrapper: returns BGR frame i and i+1.
    Works for both path-based and in-memory sequences.
    """
    if args.dataset == 'custom':  # frames already in‐memory
        return seq[i], seq[i + 1]
    return cv2.imread(seq[i]), cv2.imread(seq[i + 1])


# --------------------------------------------------------------------------- #
#  New: stereo image paths
# --------------------------------------------------------------------------- #
# USAGE: right_seq   = load_stereo_paths(args)       # unused for now
def load_stereo_paths(args) -> List[str]:
    """
    Return sorted list of right-camera image paths
    if the dataset provides them; else an empty list.
    """
    prefix = os.path.join(args.base_dir, args.dataset)
    if args.dataset == 'kitti':
        right_dir, pat = os.path.join(prefix, '05', 'image_1'), '*.png'
        return sorted(glob.glob(os.path.join(right_dir, pat)))
    if args.dataset == 'malaga':
        right_dir = os.path.join(prefix,
            'malaga-urban-dataset-extract-07_rectified_800x600_Images')
        return sorted(glob.glob(os.path.join(right_dir, '*_right.jpg')))
    # parking, tum‐rgbd, custom have no right camera
    return []


# --------------------------------------------------------------------------- #
#  New: calibration loaders
# --------------------------------------------------------------------------- #
def load_calibration(args) -> Dict[str, np.ndarray]:
    """
    Returns a dict with keys:
      'K_l', 'P_l',        ← left intrinsics 3×3 and 3×4
      'K_r', 'P_r'         ← right intrinsics & proj (if available)
    """
    # TODO: add TUM RGB-D calibration
    prefix = os.path.join(args.base_dir, args.dataset)

    if args.dataset == 'kitti':
        return _calib_kitti()
    if args.dataset == 'malaga':
        return _calib_malaga()
    if args.dataset == "tum-rgbd":
        return _calib_tum()
    if args.dataset == 'custom':
        calib_path = os.path.join(prefix, 'calibration.pkl')
        return _calib_custom(calib_path)

    raise ValueError(f"No calibration loader for {args.dataset}")


def _calib_kitti() -> Dict[str, np.ndarray]:
    params_l = np.fromstring(
        "7.070912e+02 0.000000e+00 6.018873e+02 0.000000e+00 "
        "0.000000e+00 7.070912e+02 1.831104e+02 0.000000e+00 "
        "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00",
        sep=' ', dtype=np.float64).reshape(3, 4)
    K_l = params_l[:3, :3]

    params_r = np.fromstring(
        "7.070912e+02 0.000000e+00 6.018873e+02 -3.798145e+02 "
        "0.000000e+00 7.070912e+02 1.831104e+02 0.000000e+00 "
        "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00",
        sep=' ', dtype=np.float64).reshape(3, 4)
    K_r = params_r[:3, :3]

    return {'K_l': K_l, 'P_l': params_l,
            'K_r': K_r, 'P_r': params_r}

def _calib_malaga() -> Dict[str, np.ndarray]:
    # left intrinsics & proj (from your rough code)
    K_l = np.array([
        [795.11588, 0.0,       517.12973],
        [0.0,       795.11588, 395.59665],
        [0.0,       0.0,       1.0     ]
    ])
    P_l = np.hstack([K_l, np.zeros((3,1))])
    # Right camera: assume identity extrinsics for now
    return {'K_l': K_l, 'P_l': P_l, 'K_r': K_l.copy(), 'P_r': P_l.copy()}

def _calib_tum():
    K = np.array([[517.3, 0., 318.6],
                  [0., 516.5, 255.3],
                  [0.,   0.,    1. ]])
    D = np.array([ 0.2624, -0.9531, 0.0054, 0.0026, -1.1633 ])   # d0-d4
    P = np.hstack([K, np.zeros((3,1))])
    return {"K_l": K, "P_l": P, "D_l": D, "K_r":None, "P_r":None, "D_r":None}


def _calib_custom(calib_path: str) -> Dict[str, np.ndarray]:
    with open(calib_path, 'rb') as f:
        camera_matrix, *_ = pickle.load(f)
    P = np.hstack([camera_matrix, np.zeros((3,1))])
    return {'K_l': camera_matrix, 'P_l': P, 'K_r': None, 'P_r': None}


# --------------------------------------------------------------------------- #
#  New: ground‐truth loaders
# --------------------------------------------------------------------------- #

def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert unit quaternion to 3×3 rotation matrix (TUM order)."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )

def _align_tum_gt(rgb_ts: List[float], gt_ts: List[float], poses: List[np.ndarray]) -> List[np.ndarray]:
    """Align ground‑truth poses to rgb timestamps via nearest‑neighbour search."""
    aligned = []
    j = 0
    for t in rgb_ts:
        # advance j until gt_ts[j] >= t
        while j + 1 < len(gt_ts) and gt_ts[j] < t:
            j += 1
        # choose closer of j and j‑1
        if j == 0:
            aligned.append(poses[0])
        else:
            if abs(gt_ts[j] - t) < abs(gt_ts[j - 1] - t):
                aligned.append(poses[j])
            else:
                aligned.append(poses[j - 1])
    return aligned


def load_groundtruth(args) -> Optional[np.ndarray]:
    """
    Returns an array of shape [N×3×4] with ground‐truth camera poses,
    or None if no GT is available for this dataset.
    """
    # TODO: add the ability to load the GT generally from a file - ie just give a path of dataset and the function will parse it
    prefix = os.path.join(args.base_dir, args.dataset)

    if args.dataset == 'kitti':
        poses = np.loadtxt(os.path.join(prefix, 'poses/05.txt'))
        return poses.reshape(-1, 3, 4)

    if args.dataset == 'malaga':
        seq = load_sequence(args)
        filepath = os.path.join(
            prefix,
            'malaga-urban-dataset-extract-07_all-sensors_GPS.txt'
        )
        return _malaga_get_gt(filepath, seq)
    
    if args.dataset == "tum-rgbd":
        seq_dir = os.path.join(prefix, "rgbd_dataset_freiburg1_room")
        rgb_list = os.path.join(seq_dir, "rgb.txt")
        gt_file = os.path.join(seq_dir, "groundtruth.txt")

        # --- read rgb timestamps --- #
        _, rgb_ts = _parse_tum_rgb_list(rgb_list, seq_dir)

        # --- read ground‑truth trajectory --- #
        gt_ts, poses = [], []
        with open(gt_file, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                items = line.strip().split()
                ts = float(items[0])
                tx, ty, tz = map(float, items[1:4])
                qx, qy, qz, qw = map(float, items[4:8])
                R = _quat_to_rot(qx, qy, qz, qw)
                P = np.hstack([R, np.array([[tx], [ty], [tz]], dtype=np.float64)])
                gt_ts.append(ts)
                poses.append(P)

        aligned = _align_tum_gt(rgb_ts, gt_ts, poses)
        return np.stack(aligned, axis=0)  # [N×3×4]
    
    else:
        print(f"No ground truth available for dataset: {args.dataset}")
        pass

    return None

# --------------------------------------------------------------------------- #
#  Malaga-urban ground truth helper functions
# --------------------------------------------------------------------------- #

def _malaga_get_gt(
    filepath: str,
    seq: List[str]
) -> np.ndarray:
    """
    Load Malaga-urban GPS/INS ground truth and align it to the left-image sequence.
    Returns [N×3×4] pose array (no file writes).
    """
    # 1) read and trim the GPS log
    col_names = [
        "Time","Lat","Lon","Alt","fix","sats","speed","dir",
        "LocalX","LocalY","LocalZ","rawlogID","GeocenX","GeocenY","GeocenZ",
        "GPSX","GPSY","GPSZ","GPSVX","GPSVY","GPSVZ","LocalVX","LocalVY","LocalVZ","SATTime"
    ]
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        comment='%',
        header=None,
        names=col_names
    )
    df = df[["Time","LocalX","LocalY","LocalZ"]].sort_values(by="Time").reset_index(drop=True)
    times = df["Time"].values
    t0, t1 = times[0], times[-1]

    # 2) keep only images whose timestamp falls within the GT interval
    valid_seq = []
    for img in seq:
        ts = extract_file_timestamp(img)
        if t0 <= ts <= t1:
            valid_seq.append(img)
    seq[:] = valid_seq  # trim the sequence in place

    # 3) build the pose list
    poses = []
    for img in seq:
        ts = extract_file_timestamp(img)
        position = get_position_at_time(ts, df)   # [-y, z, x]
        P4 = np.eye(4, dtype=np.float64)
        P4[:3, 3] = position
        poses.append(P4[:3,:4])

    return np.stack(poses, axis=0)  # [N×3×4]


def extract_file_timestamp(filepath: str) -> float:
    """
    Extract the floating‐point timestamp from a Malaga filename:
    e.g. '..._1234567890.123_left.jpg' → 1234567890.123
    """
    name = os.path.basename(filepath)
    parts = name.split("_")
    return float(parts[2])


def get_position_at_time(timestamp: float, df: pd.DataFrame) -> np.ndarray:
    """
    Linearly interpolate the (LocalX,LocalY,LocalZ) at the given timestamp.
    Returns a length-3 vector [-y, z, x] to match camera axes.
    """
    times = df["Time"].values
    idx = np.searchsorted(times, timestamp)
    idx = np.clip(idx, 1, len(times)-1)
    t_prev, t_next = times[idx-1], times[idx]
    row_prev, row_next = df.iloc[idx-1], df.iloc[idx]
    x0, y0, z0 = row_prev[["LocalX","LocalY","LocalZ"]]
    x1, y1, z1 = row_next[["LocalX","LocalY","LocalZ"]]

    # use ASCII variable name 'alpha' instead of a Greek symbol
    alpha = (timestamp - t_prev) / (t_next - t_prev) if (t_next != t_prev) else 0.0

    x = x0 + alpha * (x1 - x0)
    y = y0 + alpha * (y1 - y0)
    z = z0 + alpha * (z1 - z0)
    return np.array([-y, z, x], dtype=np.float64)