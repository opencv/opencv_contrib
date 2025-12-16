import os
import glob
import argparse
import time
from tqdm import tqdm
from dataclasses import dataclass

import cv2
import numpy as np


# Optional LightGlue imports
try:
    import torch
    from lightglue import LightGlue, ALIKED
    from lightglue.utils import rbd , load_image
except ImportError:
    raise ImportError('LightGlue unavailable, please install it using instructions on https://github.com/cvg/LightGlue')


def get_detector(detector_type, max_features=6000):
    if detector_type == 'orb':
        return cv2.ORB_create(max_features)
    elif detector_type == 'sift':
        return cv2.SIFT_create()
    elif detector_type == 'akaze':
        return cv2.AKAZE_create()
    raise ValueError(f"Unsupported detector: {detector_type}")

def get_matcher(matcher_type, detector_type=None):
    if matcher_type == 'bf':
        norm = cv2.NORM_HAMMING if detector_type in ['orb','akaze'] else cv2.NORM_L2
        return cv2.BFMatcher(norm, crossCheck=True)
    raise ValueError(f"Unsupported matcher: {matcher_type}")

def opencv_detector_and_matcher(img1,img2,detector,matcher):
    kp1,des1=detector.detectAndCompute(img1,None)
    kp2,des2=detector.detectAndCompute(img2,None)
    if des1 is None or des2 is None:
        return [],[],[]
    matches=matcher.match(des1,des2)
    return kp1, kp2, des1, des2, sorted(matches,key=lambda m:m.distance)

def bgr_to_tensor(image):
    """Convert a OpenCV‐style BGR uint8 into (3,H,W) torch tensor in [0,1] ."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0)
    return tensor.cuda() if torch.cuda.is_available() else tensor

def tensor_to_bgr(img_tensor):
    """Convert a (1,3,H,W) or (3,H,W) torch tensor in [0,1] to RGB → OpenCV‐style BGR uint8 image"""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img_np = img_tensor.permute(1,2,0).cpu().numpy()
    img_np = (img_np * 255).clip(0,255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def convert_lightglue_to_opencv(keypoints0, keypoints1, matches):
        """
        Convert filtered LightGlue keypoints and matches into OpenCV-compatible KeyPoint and DMatch objects.
        Here, keypoints0 and keypoints1 are assumed to already be filtered (only matched keypoints).
        
        Returns:
            opencv_kp0: List of cv2.KeyPoint objects for image0.
            opencv_kp1: List of cv2.KeyPoint objects for image1.
            opencv_matches: List of cv2.DMatch objects where each match is simply (i,i).
        """
        # TODO: convert descriptors as well
        n_matches = keypoints0.shape[0]  # number of matched keypoints
        opencv_kp0 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints0]
        opencv_kp1 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints1]

        opencv_matches = [
        cv2.DMatch(int(q), int(t), 0, 0.0) for q, t in matches
        ]

        return opencv_kp0, opencv_kp1, opencv_matches

def lightglue_match(img1, img2, extractor, matcher, min_conf=0.0):
        """
        Uses ALIKED and LightGlue to extract features and match them,
        then converts the outputs to OpenCV-compatible formats.
        Returns:
            pos_kp0: numpy array of keypoint coordinates for image0 (shape (K,2)).
            pos_kp1: numpy array of keypoint coordinates for image1 (shape (K,2)).
            opencv_matches: list of cv2.DMatch objects.
            opencv_kp0: list of cv2.KeyPoint objects for image0.
            opencv_kp1: list of cv2.KeyPoint objects for image1.
        """ 
        # TODO: add min_conf filtering
        # TODO: return descriptors as well
        # Convert image to torch tensor
        img1_tensor = bgr_to_tensor(img1)
        img2_tensor = bgr_to_tensor(img2)
        feats0 = extractor.extract(img1_tensor)
        feats1 = extractor.extract(img2_tensor)
        matches_out = matcher({'image0': feats0, 'image1': feats1})
        # Remove batch dimension using rbd from lightglue.utils
        feats0 = rbd(feats0)
        feats1 = rbd(feats1)
        matches_out = rbd(matches_out)
        
        # Get matches and keypoints as numpy arrays.
        matches = matches_out['matches']  # shape (K,2)
        # keypoints returned by LightGlue are numpy arrays of (x,y)
        keypoints0 = feats0['keypoints']
        keypoints1 = feats1['keypoints']
        descriptors0 = feats0['descriptors']  # shape (K,D)
        descriptors1 = feats1['descriptors']  # shape (K,D)    
        print("discriptor types:", type(descriptors0), type(descriptors1))

        
        # Convert the keypoints and matches to OpenCV formats.
        opencv_kp0, opencv_kp1, opencv_matches = convert_lightglue_to_opencv(keypoints0, keypoints1, matches)
        
        return opencv_kp0, opencv_kp1, descriptors0, descriptors1, opencv_matches

def update_and_prune_tracks(matches, prev_map, tracks, kp_curr, frame_idx, next_track_id, prune_age=30):
    """
    Update feature tracks given a list of matches, then prune stale ones.

    Args:
        matches        : List[cv2.DMatch] between previous and current keypoints.
        prev_map       : Dict[int, int] mapping prev-frame kp index -> track_id.
        tracks         : Dict[int, List[Tuple[int, int, int]]], 
                         each track_id → list of (frame_idx, x, y).
        kp_curr        : List[cv2.KeyPoint] for current frame.
        frame_idx      : int, index of the current frame (1-based or 0-based).
        next_track_id  : int, the next unused track ID.
        prune_age      : int, max age (in frames) before a track is discarded.

    Returns:
        curr_map       : Dict[int, int] mapping curr-frame kp index → track_id.
        tracks         : Updated tracks dict.
        next_track_id  : Updated next unused track ID.
    """
    curr_map = {}

    # 1) Assign each match to an existing or new track_id
    for m in matches:
        q = m.queryIdx    # keypoint idx in previous frame
        t = m.trainIdx    # keypoint idx in current frame

        # extract integer pixel coords
        x, y = int(kp_curr[t].pt[0]), int(kp_curr[t].pt[1])

        if q in prev_map:
            # continue an existing track
            tid = prev_map[q]
        else:
            # start a brand-new track
            tid = next_track_id
            tracks[tid] = []
            next_track_id += 1

        # map this current keypoint to the track
        curr_map[t] = tid
        # append the new observation
        tracks[tid].append((frame_idx, x, y))

    # 2) Prune any track not seen in the last `prune_age` frames
    for tid, pts in list(tracks.items()):
        last_seen_frame = pts[-1][0]
        if (frame_idx - last_seen_frame) > prune_age:
            del tracks[tid]

    return curr_map, tracks, next_track_id

def filter_matches_ransac(kp1, kp2, matches, thresh):
    """
    Filter a list of cv2.DMatch objects using RANSAC on the fundamental matrix.

    Args:
        kp1      (List[cv2.KeyPoint]): KeyPoints from the first image.
        kp2      (List[cv2.KeyPoint]): KeyPoints from the second image.
        matches  (List[cv2.DMatch])   : Initial matches between kp1 and kp2.
        thresh   (float)              : RANSAC inlier threshold (pixels).

    Returns:
        List[cv2.DMatch]: Only those matches deemed inliers by RANSAC.
    """
    # Need at least 8 points for a fundamental matrix
    if len(matches) < 8:
        return matches

    # Build corresponding point arrays
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Run RANSAC to find inlier mask
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=thresh,
        confidence=0.99
    )
    mask = mask.ravel().astype(bool)

    # Keep only the inlier matches
    inliers = [m for m, ok in zip(matches, mask) if ok]
    return inliers

def draw_tracks(vis, tracks, current_frame, max_age=10, sample_rate=5, max_tracks=1000):
    """
    Draw each track's path with color decaying from green (new) to red (old).
    Only draw up to max_tracks most recent tracks, and sample every sample_rate-th track.
    """
    recent=[(tid,pts) for tid,pts in tracks.items() if pts and current_frame-pts[-1][0]<=max_age]
    recent.sort(key=lambda x:x[1][-1][0],reverse=True)
    drawn=0
    for tid,pts in recent:
        if drawn>=max_tracks: break
        if tid%sample_rate!=0: continue
        pts=[p for p in pts if current_frame-p[0]<=max_age]
        for j in range(1,len(pts)):
            frame_idx,x0,y0=pts[j-1]
            _,x1,y1=pts[j]
            age=current_frame-frame_idx
            ratio=age/max_age
            b=0
            g=int(255*(1-ratio))
            r=int(255*ratio)
            cv2.line(vis,(int(x0),int(y0)),(int(x1),int(y1)),(b,g,r),2)
        drawn+=1
    return vis

def dataloader(args):
    """
    Load the dataset and return the detector, matcher, and sequence of images.
    """
    # init modules once
    if args.use_lightglue:
        detector=ALIKED(max_num_keypoints=2048).eval().cuda()
        matcher=LightGlue(features='aliked').eval().cuda()
    else:
        detector=get_detector(args.detector)
        matcher=get_matcher(args.matcher,args.detector)

    # load sequence
    prefix=os.path.join(args.base_dir, args.dataset)
    is_custom=False
    if args.dataset=='kitti': img_dir,pat=os.path.join(prefix,'05','image_0'),'*.png'
    elif args.dataset=='parking': img_dir,pat=os.path.join(prefix,'images'),'*.png'
    elif args.dataset=='malaga': img_dir,pat =os.path.join(prefix,'malaga-urban-dataset-extract-07_rectified_800x600_Images'),'*_left.jpg'
    elif args.dataset=='tum-rgbd': img_dir,pat =os.path.join(prefix,'rgbd_dataset_freiburg1_room', 'rgb'),'*.png'
    else:
        vid=os.path.join(prefix,'custom_compress.mp4')
        cap=cv2.VideoCapture(vid)
        seq=[]
        while True:
            ok,fr=cap.read()
            if not ok: break; seq.append(fr)
        cap.release(); is_custom=True

    # load images    
    if not is_custom: seq=sorted(glob.glob(os.path.join(img_dir,pat)))

    return detector, matcher, seq

def load_frame_pair(args,seq, i):
    """
    Load a pair of consecutive frames from the sequence.
    Args:
        seq (list): List of image paths or frames.
        i (int): Index of the current frame.
    Returns:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
    """
    # load images
    if args.dataset=='custom': 
        img1, img2 = seq[i], seq[i+1]
    else: 
        img1 = cv2.imread(seq[i])
        img2=cv2.imread(seq[i+1])

    return img1, img2

def detect_and_match(args, img1, img2, detector, matcher):
    """
    Load two consecutive images from the sequence and match features using the specified detector and matcher.
    Args:
        args (argparse.Namespace): Command line arguments.
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        detector: Feature detector object.
        matcher: Feature matcher object.
    Returns:
        kp_map1 (list): Keypoints from the first image.
        kp_map2 (list): Keypoints from the second image.
        matches (list): List of matched features.
    """
    # match features
    if args.use_lightglue:
        kp_map1, kp_map2, des1, des2, matches = lightglue_match(img1,img2,detector,matcher)
    else:
        kp_map1, kp_map2, des1, des2, matches = opencv_detector_and_matcher(img1, img2, detector, matcher)
    
    return kp_map1, kp_map2, des1, des2, matches


@dataclass
class Keyframe:
    idx: int                   # global frame index
    path: str                  # on-disk image file  OR "" if custom video
    kps:  list[cv2.KeyPoint]   # keypoints (needed for geometric checks)
    desc: np.ndarray           # descriptors (uint8/float32)
    thumb: bytes               # lz4-compressed thumbnail for UI


def main():
    parser=argparse.ArgumentParser("Feature tracking with RANSAC filtering")
    parser.add_argument('--dataset',choices=['kitti','malaga','tum-rgb','custom'],required=True)
    parser.add_argument('--base_dir',default='.././Dataset')
    parser.add_argument('--detector',choices=['orb','sift','akaze'],default='orb')
    parser.add_argument('--matcher',choices=['bf'],default='bf')
    parser.add_argument('--use_lightglue',action='store_true')
    parser.add_argument('--fps',type=float,default=10)
    # --- RANSAC related CLI flags -------------------------------------------
    parser.add_argument('--ransac_thresh',type=float,default=1.0, help='RANSAC threshold for fundamental matrix')
    # --- Key-frame related CLI flags -----------------------------------------
    parser.add_argument('--kf_max_disp',  type=float, default=30,
                        help='Min avg. pixel displacement wrt last keyframe')
    parser.add_argument('--kf_min_ratio', type=float, default=0.5,
                    help='Min surviving-match ratio wrt last keyframe')
    parser.add_argument('--kf_cooldown', type=int, default=5,
                        help='Frames to wait before next keyframe check')

    args=parser.parse_args()

    # initialize detector and matcher
    detector, matcher, seq = dataloader(args)

    # tracking data
    track_id=0; prev_map={}; tracks={}
    cv2.namedWindow('Feature Tracking',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Feature Tracking',1200,600)

    total=len(seq)-1; prev_time=time.time(); achieved_fps=0.0
    for i in tqdm(range(total),desc='Tracking'):
        # load frame pair
        img1, img2 = load_frame_pair(args, seq, i)

        # Detect and match features
        kp_map1, kp_map2, des1, des2, matches = detect_and_match(args, img1, img2, detector, matcher)

        # filter with RANSAC
        matches = filter_matches_ransac(kp_map1, kp_map2, matches, args.ransac_thresh)
        
        # update & prune tracks
        frame_no = i + 1
        prev_map, tracks, track_id = update_and_prune_tracks(matches, prev_map, tracks, kp_map2, frame_no, track_id, prune_age=30)
        
        # draw
        vis= img2.copy()
        vis=draw_tracks(vis,tracks,i+1)
        for t,tid in prev_map.items():
            cv2.circle(vis,tuple(map(int, kp_map2[t].pt)),3,(0,255,0),-1)

        text=f"Frame {i+1}/{total} | Tracks: {len(tracks)} | FPS: {achieved_fps:.1f}"
        cv2.putText(vis,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        # show
        interval_ms=int(1000/args.fps) if args.fps>0 else 0
        key=cv2.waitKey(interval_ms)
        cv2.imshow('Feature Tracking',vis)
        # measure fps
        now=time.time();dt=now-prev_time
        if dt>0: achieved_fps=1.0/dt
        prev_time=now
        if key&0xFF==27: break
    cv2.destroyAllWindows()

if __name__=='__main__': 
    main()
