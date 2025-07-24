# features_utils.py
import cv2
import numpy as np
from typing import List, Tuple
# optional LightGlue imports guarded by try/except
try:
    import torch
    from lightglue import LightGlue, ALIKED
    from lightglue.utils import rbd , load_image
except ImportError:
    raise ImportError('LightGlue unavailable, please install it using instructions on https://github.com/cvg/LightGlue')



# --------------------------------------------------------------------------- #
#  Initialisation helpers
# --------------------------------------------------------------------------- #
def init_feature_pipeline(args):
    """
    Instantiate detector & matcher according to CLI arguments.
    Returns (detector, matcher)
    """
    if args.use_lightglue:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = ALIKED(max_num_keypoints=2048).eval().to(device)
        matcher  = LightGlue(features='aliked').eval().to(device)
    else:
        detector = _get_opencv_detector(args.detector)
        matcher  = _get_opencv_matcher(args.matcher, args.detector)
    return detector, matcher


def _get_opencv_detector(detector_type, max_features=6000):
    if detector_type == 'orb':
        return cv2.ORB_create(max_features)
    if detector_type == 'sift':
        return cv2.SIFT_create()
    if detector_type == 'akaze':
        return cv2.AKAZE_create()
    raise ValueError(f"Unsupported detector: {detector_type}")


def _get_opencv_matcher(matcher_type, detector_type):
    if matcher_type != 'bf':
        raise ValueError(f"Unsupported matcher: {matcher_type}")
    norm = cv2.NORM_HAMMING if detector_type in ['orb', 'akaze'] else cv2.NORM_L2
    return cv2.BFMatcher(norm, crossCheck=True)

# --------------------------------------------------------------------------- #
#  Individual feature extraction and matching functions
# --------------------------------------------------------------------------- #

def _convert_lg_kps_to_opencv(kp0: torch.Tensor) -> List[cv2.KeyPoint]:
    cv_kp0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0]
    return cv_kp0

def _convert_opencv_to_lg_kps(cv_kp: List[cv2.KeyPoint]) -> torch.Tensor:
    """
    Convert a list of cv2.KeyPoint into a tensor of shape [M×2] 
    containing (x, y) coordinates for LightGlue.
    Args:
        cv_kp: list of cv2.KeyPoint
    Returns:
        Tensor of shape [M, 2], dtype=torch.float32
    """
    coords = [(kp.pt[0], kp.pt[1]) for kp in cv_kp]
    if not coords:
        return torch.empty((0, 2), dtype=torch.float32)
    return torch.tensor(coords, dtype=torch.float32)


def _convert_lg_matches_to_opencv(matches_raw: torch.Tensor) -> List[cv2.DMatch]:
    pairs = matches_raw.cpu().numpy().tolist()
    cv_matches = [cv2.DMatch(int(i), int(j), 0, 0.0) for i, j in pairs]
    return cv_matches

def feature_extractor(args, img: np.ndarray, detector):
    """
    Extract features from a single image.
    Returns a FrameFeatures with OpenCV-compatible keypoints and descriptors,
    and optional LightGlue tensors if using LightGlue.
    """

    if args.use_lightglue:
        t0 = _bgr_to_tensor(img)
        feats = detector.extract(t0)
        feats = rbd(feats)
        lg_kps = feats['keypoints']
        kp0 = _convert_lg_kps_to_opencv(lg_kps)
        des0 = feats['descriptors']
        return kp0, des0
    
    else:
        kp0, des0 = detector.detectAndCompute(img, None)
        if des0 is None:
            return [], []
        return kp0, des0


def feature_matcher(args, kp0, kp1, des0, des1, matcher):
    """    Match features between two FrameFeatures and return OpenCV-compatible matches."""
    # optionally filter matches from LightGlue by confidence
    if args.use_lightglue:
        # LightGlue matching
        # kp0, kp1: List[cv2.KeyPoint]
        # des0, des1: torch.Tensor [M×D], [N×D]
        device = des0.device
        lg_kp0 = _convert_opencv_to_lg_kps(kp0).to(device).unsqueeze(0)   # [1×M×2]
        lg_kp1 = _convert_opencv_to_lg_kps(kp1).to(device).unsqueeze(0)   # [1×N×2]
        lg_desc0 = des0.unsqueeze(0)                          # [1×M×D]
        lg_desc1 = des1.unsqueeze(0)                          # [1×N×D]

        raw = matcher({
            'image0': {'keypoints': lg_kp0, 'descriptors': lg_desc0},
            'image1': {'keypoints': lg_kp1, 'descriptors': lg_desc1}
        })
        raw = rbd(raw)
        matches_raw = raw['matches']
        # LightGlue may return confidence either as 'scores' or 'confidence'.
        # Avoid `or` with tensors to prevent ambiguity errors.
        conf = raw['scores'] if 'scores' in raw else raw.get('confidence')
        if conf is not None:
            mask = conf > args.min_conf
            matches_raw = matches_raw[mask]
        # convert index pairs to cv2.DMatch list
        cv_matches = _convert_lg_matches_to_opencv(matches_raw)
        return cv_matches
    else:
        # OpenCV matching
        matches = matcher.match(des0, des1)
        return sorted(matches, key=lambda m: m.distance)


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def filter_matches_ransac(kp1, kp2, matches, thresh=1.0):
    """
    Drop outliers using the fundamental matrix + RANSAC.
    """
    if len(matches) < 8:
        return matches

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    _, mask = cv2.findFundamentalMat(pts1, pts2,
                                     cv2.FM_RANSAC, thresh, 0.99)
    mask = mask.ravel().astype(bool)
    return [m for m, ok in zip(matches, mask) if ok]


# --------------------------------------------------------------------------- #
#  Convenience method for LightGlue/OpenCV pipeline to detect and match features in two images
# NO LONGER USED IN THE MAIN SLAM PIPELINE, BUT LEFT FOR REFERENCE
#  (e.g. for testing purposes).
# --------------------------------------------------------------------------- #
def _opencv_detect_and_match(img1, img2, detector, matcher):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return [], [], [], [], []        # gracefully handle empty images

    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)

    return kp1, kp2, des1, des2, matches

def _bgr_to_tensor(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.cuda() if torch.cuda.is_available() else tensor


def _convert_lightglue_to_opencv(kp0, kp1, matches):
    n = kp0.shape[0]
    cv_kp0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp0]
    cv_kp1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1]
    cv_matches = [cv2.DMatch(int(i), int(j), 0, 0.0) for i, j in matches]
    return cv_kp0, cv_kp1, cv_matches


def _lightglue_detect_and_match(img1, img2, extractor, matcher):
    t0, t1 = _bgr_to_tensor(img1), _bgr_to_tensor(img2)

    f0, f1   = extractor.extract(t0), extractor.extract(t1)
    matches  = matcher({'image0': f0, 'image1': f1})

    # remove batch dimension
    f0, f1   = rbd(f0), rbd(f1)
    matches  = rbd(matches)

    kp0, kp1 = f0['keypoints'], f1['keypoints']
    des0, des1 = f0['descriptors'], f1['descriptors']
    cv_kp0, cv_kp1, cv_matches = _convert_lightglue_to_opencv(kp0, kp1,
                                                              matches['matches'])
    return cv_kp0, cv_kp1, des0, des1, cv_matches

def detect_and_match(img1, img2, detector, matcher, args):
    """
    Front-end entry. Chooses OpenCV or LightGlue depending on CLI flag.
    """
    if args.use_lightglue:
        return _lightglue_detect_and_match(img1, img2, detector, matcher)
    return _opencv_detect_and_match(img1, img2, detector, matcher)
