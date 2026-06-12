import cv2
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'WXAgg', etc.
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.optimize import least_squares
import pandas as pd
import pickle


# New imports for LightGlue integration
import torch
from lightglue import LightGlue, ALIKED  # using ALIKED features
from lightglue.utils import load_image, rbd

class StructureFromMotion:
    def __init__(self, prefix, thumpup_pos_thresh, thumpup_rot_thresh, use_lightglue=True):
        # Set this flag to True to use ALIKED+LightGlue instead of the OpenCV pipeline.
        self.use_lightglue = use_lightglue

        # If using LightGlue, initialize its extractor and matcher.
        if self.use_lightglue:
            self.extractor = ALIKED(max_num_keypoints=2048).eval().cuda()
            # Here we choose 'disk' as the matching features for LightGlue.
            self.matcher = LightGlue(features='disk').eval().cuda()

        # (Rest of your __init__ code remains unchanged...)
        self.isStereo = False
        self.THUMPUP_POS_THRESHOLD = thumpup_pos_thresh
        self.THUMPUP_ROT_THRESHOLD = thumpup_rot_thresh
        
        # ================== DETERMINE DATASET ==================
        if 'kitti' in prefix:
            self.dataset = 'kitti'
            self.image_list = sorted(glob.glob(os.path.join(prefix, '05/image_0', '*.png')))
            self.image_list_r = sorted(glob.glob(os.path.join(prefix, '05/image_1', '*.png')))
            self.K_3x3, self.Proj, self.K_r_3x3, self.Proj_r = self.load_calibration_kitti()
            self.gt = np.loadtxt(os.path.join(prefix, 'poses/05.txt')).reshape(-1, 3, 4)
        elif 'parking' in prefix:
            self.dataset = 'parking'
            self.image_list = sorted(glob.glob(os.path.join(prefix, 'images', '*.png')))
            self.gt = np.loadtxt(os.path.join(prefix, 'poses.txt')).reshape(-1, 3, 4)
            self.K_3x3, self.Proj = self.load_calibration_parking()
        elif 'malaga' in prefix:
            self.dataset = 'malaga'
            self.image_list = sorted(glob.glob(
                os.path.join(prefix, 'malaga-urban-dataset-extract-07_rectified_1024x768_Images',
                             '*_left.jpg')))
            self.image_list_r = sorted(glob.glob(
                os.path.join(prefix, 'malaga-urban-dataset-extract-07_rectified_1024x768_Images',
                             '*_right.jpg')))
            self.K_3x3, self.Proj, self.K_r_3x3, self.Proj_r = self.load_calibration_malaga()
            self.gt = self.malaga_get_gt(
                os.path.join(prefix, 'malaga-urban-dataset-extract-07_all-sensors_GPS.txt'),
                os.path.join(prefix, 'poses')
            )
        elif 'custom' in prefix:
            self.dataset = 'custom'
            self.image_list = []
            calib_path = os.path.join(prefix, 'calibration.pkl')
            if not os.path.exists(calib_path):
                raise FileNotFoundError(f"Calibration file not found at: {calib_path}")
            self.K_3x3, self.Proj = self.load_calibration_custom(calib_path)
            video_path = os.path.join(prefix, 'custom_compress.mp4')
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found at: {video_path}")
            cap = cv2.VideoCapture(video_path)
            self.frames = []
            success, frame = cap.read()
            while success:
                self.frames.append(frame)
                success, frame = cap.read()
            cap.release()
            self.image_list = list(range(len(self.frames)))
            possible_gt_path = os.path.join(prefix, 'groundtruth.txt')
            if os.path.exists(possible_gt_path):
                try:
                    self.gt = np.loadtxt(possible_gt_path).reshape(-1, 3, 4)
                except:
                    print("Ground truth found but could not be loaded. Skipping.")
                    self.gt = None
            else:
                print("No ground truth for custom video. Ground truth set to None.")
                self.gt = None
        else:
            raise NotImplementedError('Dataset not implemented:', prefix)
        
        # # ================== FEATURE DETECTORS (OpenCV fallback) ==================
        self.MAX_FEATURES = 6000
        matcher_type = "akaze"  # or "orb", "sift"
        if matcher_type == "orb":
            self.fd = cv2.ORB_create(self.MAX_FEATURES)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher_type == "sift":
            self.fd = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif matcher_type == "akaze":
            self.fd = cv2.AKAZE_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            raise ValueError(f"Unsupported matcher, {matcher_type}")
        
        self.RANSAC_THRESHOLD = 1.0

        if self.isStereo:
            self.MIN_DISP = 0
            self.NUM_DISP = 32
            self.BLOCK_SIZE = 11
            P1 = self.BLOCK_SIZE * self.BLOCK_SIZE * 8
            P2 = self.BLOCK_SIZE * self.BLOCK_SIZE * 32
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=self.MIN_DISP,
                numDisparities=self.NUM_DISP,
                blockSize=self.BLOCK_SIZE,
                P1=P1,
                P2=P2
            )

    # ---------------------------------------------------------------------
    # CALIBRATION LOADERS
    # ---------------------------------------------------------------------
    def load_calibration_kitti(self):
        # left
        params_left = np.fromstring(
            "7.070912e+02 0.000000e+00 6.018873e+02 0.000000e+00 "
            "0.000000e+00 7.070912e+02 1.831104e+02 0.000000e+00 "
            "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00",
            dtype=np.float64, sep=' '
        ).reshape(3, 4)
        K_left_3x3 = params_left[:3, :3]

        # right
        params_right = np.fromstring(
            "7.070912e+02 0.000000e+00 6.018873e+02 -3.798145e+02 "
            "0.000000e+00 7.070912e+02 1.831104e+02 0.000000e+00 "
            "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00",
            dtype=np.float64, sep=' '
        ).reshape(3, 4)
        K_right_3x3 = params_right[:3, :3]
        return K_left_3x3, params_left, K_right_3x3, params_right

    def load_calibration_parking(self):
        P = np.array([
            331.37,    0.0,    320.0,   0.0,
            0.0,    369.568,   240.0,   0.0,
            0.0,      0.0,      1.0,    0.0
        ]).reshape(3, 4)
        K_3x3 = P[:3, :3]
        return K_3x3, P

    def load_calibration_malaga(self):
        K_left_3x3 = np.array([
            [795.11588, 0.0,       517.12973],
            [0.0,       795.11588, 395.59665],
            [0.0,       0.0,       1.0     ]
        ])
        Proj_l = np.array([
            [795.11588, 0.0,       517.12973, 0.0],
            [0.0,       795.11588, 395.59665, 0.0],
            [0.0,       0.0,       1.0,       0.0]
        ])

        # Right camera extrinsics
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R_right = self.quaternion_to_rotation_matrix(q)
        t_right = np.array([0.119471, 0.0, 0.0]).reshape(3, 1)
        
        inv_homo_proj = np.linalg.inv(self.get_4by4_homo_proj(R_right, t_right))
        Proj_r = Proj_l @ inv_homo_proj

        # For the right camera
        K_right_3x3 = K_left_3x3.copy()
        return K_left_3x3, Proj_l, K_right_3x3, Proj_r

    def load_calibration_custom(self, calib_path):
        with open(calib_path, 'rb') as f:
            calibration = pickle.load(f)
        camera_matrix_3x3 = calibration[0]  # shape (3,3)
        # distCoeffs = calibration[1]      # shape (5,) or similar, if needed

        # Build a 3×4 projection for the left camera
        Proj = np.hstack([camera_matrix_3x3, np.zeros((3, 1))])
        return camera_matrix_3x3, Proj

    # ---------------------------------------------------------------------
    # MALAGA GT UTILS
    # ---------------------------------------------------------------------
    def malaga_get_gt(self, filepath, gt_filepath):
        col_names = [
            "Time","Lat","Lon","Alt","fix","sats","speed","dir",
            "LocalX","LocalY","LocalZ","rawlogID","GeocenX","GeocenY","GeocenZ",
            "GPSX","GPSY","GPSZ","GPSVX","GPSVY","GPSVZ","LocalVX","LocalVY","LocalVZ","SATTime"
        ]
        df = pd.read_csv(filepath, sep='\s+', comment='%', header=None, names=col_names)
        df = df[["Time", "LocalX", "LocalY", "LocalZ"]].sort_values(by="Time").reset_index(drop=True)
        
        times = df["Time"].values
        first_time, last_time = times[0], times[-1]
        
        # Remove frames that have no GT
        rows_to_del = []
        for i in range(len(self.image_list)):
            f = self.image_list[i]
            timestamp = self.extract_file_timestamp(f)
            if timestamp < first_time or timestamp > last_time:
                rows_to_del.append(i)
        self.image_list = np.delete(self.image_list, rows_to_del)
        self.image_list_r = np.delete(self.image_list_r, rows_to_del)

        gt = []
        for f in self.image_list:
            timestamp = self.extract_file_timestamp(f)
            position = self.get_position_at_time(timestamp, df)
            pose = np.eye(4)
            pose[:3, 3] = position
            gt.append(pose)

        gt = np.array(gt)
        # Truncate to Nx3 if that is what your pipeline expects
        gt = gt[:, :3]

        np.save(os.path.join(gt_filepath + ".npy"), gt)
        np.savetxt(os.path.join(gt_filepath + ".txt"), gt.reshape(gt.shape[0], -1), fmt="%.5f")
        return gt

    def extract_file_timestamp(self, filepath):
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        time_part = parts[2]
        return float(time_part)
    
    def get_position_at_time(self, timestamp, df):
        times = df["Time"].values
        idx = np.searchsorted(times, timestamp)
        t0 = times[idx-1]
        t1 = times[idx]
        row0 = df.iloc[idx-1]
        row1 = df.iloc[idx]
        x0, y0, z0 = row0["LocalX"], row0["LocalY"], row0["LocalZ"]
        x1, y1, z1 = row1["LocalX"], row1["LocalY"], row1["LocalZ"]

        alpha = (timestamp - t0)/(t1 - t0) if t1 != t0 else 0
        x = x0 + alpha*(x1 - x0)
        y = y0 + alpha*(y1 - y0)
        z = z0 + alpha*(z1 - z0)
        return [-y, z, x]

    # ---------------------------------------------------------------------
    # GENERAL UTILS
    # ---------------------------------------------------------------------
    def quaternion_to_rotation_matrix(self, q):
        a, b, c, d = q
        a2, b2, c2, d2 = a*a, b*b, c*c, d*d
        R = np.array([
            [a2 + b2 - c2 - d2,   2*b*c - 2*a*d,     2*b*d + 2*a*c],
            [2*b*c + 2*a*d,       a2 - b2 + c2 - d2, 2*c*d - 2*a*b],
            [2*b*d - 2*a*c,       2*c*d + 2*a*b,     a2 - b2 - c2 + d2]
        ])
        return R

    def get_4by4_homo_proj(self, R, t):
        P = np.eye(4)
        P[:3, :3] = R
        P[:3, 3]  = t.ravel()
        return P

    # ---------------------------------------------------------------------
    # READING IMAGES
    # ---------------------------------------------------------------------
    def read_image(self, idx):
        if self.dataset == 'custom':
            return self.frames[idx]
        else:
            return cv2.imread(self.image_list[idx])

    # ---------------------------------------------------------------------
    # FEATURE DETECTION & MATCHING
    # ---------------------------------------------------------------------
    def feature_detection(self, image):
        kp, des = self.fd.detectAndCompute(image, mask=None)
        return kp, des
    
    def feature_matching(self, des1, des2, kp1, kp2):
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        m_kp1 = np.array([kp1[m.queryIdx] for m in matches])
        m_kp2 = np.array([kp2[m.trainIdx] for m in matches])
        return m_kp1, m_kp2, matches

    # ---------------------------------------------------------------------
    # POSE ESTIMATION (2D-2D)
    # ---------------------------------------------------------------------
    def pose_estimation(self, pos_m_kp1, pos_m_kp2):
        E, _ = cv2.findEssentialMat(
            pos_m_kp1, pos_m_kp2,
            cameraMatrix=self.K_3x3,
            method=cv2.RANSAC,
            threshold=self.RANSAC_THRESHOLD
        )
        _, R, t, _ = cv2.recoverPose(E, pos_m_kp1, pos_m_kp2, cameraMatrix=self.K_3x3)
        return R, t
    
    def pose_estimation(self, pos_m_kp1, pos_m_kp2):
        # If the keypoints are torch tensors on GPU, move them to CPU and convert to numpy
        if isinstance(pos_m_kp1, torch.Tensor):
            pts1 = pos_m_kp1.cpu().numpy().astype(np.float32)
        else:
            pts1 = np.asarray(pos_m_kp1, dtype=np.float32)

        if isinstance(pos_m_kp2, torch.Tensor):
            pts2 = pos_m_kp2.cpu().numpy().astype(np.float32)
        else:
            pts2 = np.asarray(pos_m_kp2, dtype=np.float32)

        
        # Pass the camera matrix and additional parameters as positional arguments
        E, _ = cv2.findEssentialMat(
            pts1, pts2,
            cameraMatrix=self.K_3x3,
            method=cv2.RANSAC,
            threshold=self.RANSAC_THRESHOLD
        )
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, cameraMatrix=self.K_3x3)
        return R, t


    # ---------------------------------------------------------------------
    # TRIANGULATION
    # ---------------------------------------------------------------------
    def triangulate(self, R, t, pos_m_kp1, pos_m_kp2):
        P1 = self.Proj
        P2 = self.Proj @ self.get_4by4_homo_proj(R, t)

        # Ensure the keypoints are numpy arrays of type float32
        if isinstance(pos_m_kp1, torch.Tensor):
            pts1 = pos_m_kp1.cpu().numpy().astype(np.float32)
        else:
            pts1 = np.asarray(pos_m_kp1, dtype=np.float32)

        if isinstance(pos_m_kp2, torch.Tensor):
            pts2 = pos_m_kp2.cpu().numpy().astype(np.float32)
        else:
            pts2 = np.asarray(pos_m_kp2, dtype=np.float32)

        # cv2.triangulatePoints expects points in shape (2, N)
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T  # Convert homogeneous coordinates to 3D
        return points_3d

    # ---------------------------------------------------------------------
    # STEREO UTILITIES
    # ---------------------------------------------------------------------
    def compute_stereo_disparity(self, img_l, img_r):
        gray_left  = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        return disparity

    def apply_disparity_check(self, q, disp, min_disp, max_disp):
        q_idx = q.astype(int)
        disp_vals = disp.T[q_idx[:,0], q_idx[:,1]]
        mask = (disp_vals > min_disp) & (disp_vals < max_disp)
        return disp_vals, mask
    
    def calculate_right_features(self, q1, q2, disparity1, disparity2, min_disp=0.0, max_disp=100.0):
        disparity1_vals, mask1 = self.apply_disparity_check(q1, disparity1, min_disp, max_disp)
        disparity2_vals, mask2 = self.apply_disparity_check(q2, disparity2, min_disp, max_disp)
        in_bounds = mask1 & mask2

        q1_l = q1[in_bounds]
        q2_l = q2[in_bounds]
        disp1 = disparity1_vals[in_bounds]
        disp2 = disparity2_vals[in_bounds]
        
        # Right coords
        q1_r = np.copy(q1_l)
        q2_r = np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        return q1_l, q1_r, q2_l, q2_r, in_bounds

    def get_stereo_3d_pts(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate from left and right images for two frames (q1_l, q1_r) and (q2_l, q2_r).
        Then we can do solvePnPRansac on Q1, etc.
        """
        Q1_4d = cv2.triangulatePoints(self.Proj, self.Proj_r, q1_l.T, q1_r.T)
        Q1_3d = (Q1_4d[:3] / Q1_4d[3]).T

        Q2_4d = cv2.triangulatePoints(self.Proj, self.Proj_r, q2_l.T, q2_r.T)
        Q2_3d = (Q2_4d[:3] / Q2_4d[3]).T
        return Q1_3d, Q2_3d

    def project_3d_to_2d(self, X_3D, R, t):
        # Compute the projection matrix for the given rotation and translation.
        P = self.Proj @ self.get_4by4_homo_proj(R, t)
        # Convert 3D points to homogeneous coordinates.
        X_3D_hom = np.hstack([X_3D, np.ones((X_3D.shape[0], 1))])
        # Project the 3D points.
        X_pix = X_3D_hom @ P.T
        # Normalize to get pixel coordinates.
        u = X_pix[:, 0] / X_pix[:, 2]
        v = X_pix[:, 1] / X_pix[:, 2]
        return np.vstack((u, v)).T

    # ---------------------------------------------------------------------
    # NONLINEAR REFINEMENT
    # ---------------------------------------------------------------------
    def refine(self, pos_m_kp1, pos_m_kp2, points_3d, R, t, verbose=False):
        def reprojection_resid(params, pos_m_kp1, pos_m_kp2, points_3d):
            R_vec = params[:3]
            t_vec = params[3:].reshape(3,1)
            R_opt, _ = cv2.Rodrigues(R_vec)
            
            # Project points for camera 1 and camera 2.
            proj1 = self.project_3d_to_2d(points_3d, np.eye(3), np.zeros((3,1)))
            proj2 = self.project_3d_to_2d(points_3d, R_opt, t_vec)
            
            # Ensure the keypoint arrays are numpy arrays on the CPU.
            if isinstance(pos_m_kp1, torch.Tensor):
                pos_m_kp1 = pos_m_kp1.cpu().numpy().astype(np.float32)
            else:
                pos_m_kp1 = np.asarray(pos_m_kp1, dtype=np.float32)
            if isinstance(pos_m_kp2, torch.Tensor):
                pos_m_kp2 = pos_m_kp2.cpu().numpy().astype(np.float32)
            else:
                pos_m_kp2 = np.asarray(pos_m_kp2, dtype=np.float32)
            
            err1 = (pos_m_kp1 - proj1).ravel()
            err2 = (pos_m_kp2 - proj2).ravel()
            return np.concatenate((err1, err2))

        # Get initial parameters from the current pose.
        r_init = cv2.Rodrigues(R)[0].ravel()
        t_init = t.ravel()
        params_init = np.hstack([r_init, t_init])

        result = least_squares(
            reprojection_resid,
            params_init,
            args=(pos_m_kp1, pos_m_kp2, points_3d),
            method='lm',
            ftol=1e-6, xtol=1e-6, gtol=1e-6
        )
        
        # Check if the optimization was successful; if not, return the original R, t.
        if not result.success:
            print("Least squares refinement did not converge. Returning original pose.")
            return R, t

        r_refined = result.x[:3]
        t_refined = result.x[3:].reshape(3, 1)
        R_refined, _ = cv2.Rodrigues(r_refined)
        
        if verbose:
            print("[Refine] r_init:", r_init, "-> r_refined:", r_refined)
            print("[Refine] t_init:", t_init, "-> t_refined:", t_refined.ravel())
        
        return R_refined, t_refined


    # ---------------------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------------------
    def vis_init(self):
        plt.ion()  # interactive mode on
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0], projection='3d')
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        if self.dataset == 'kitti':
            ax4.view_init(elev=15, azim=-90)
        elif self.dataset == 'parking':
            ax4.view_init(elev=-80, azim=-90)
        
        # Basic range
        x_min, x_max = -400, 400
        y_min, y_max = -400, 400
        z_min, z_max = -400, 400

        return ax1, ax2, ax3, ax4, ax5, ax6, x_min, x_max, y_min, y_max, z_min, z_max
    
    def vis(self, vis_param, image1, image2, kp1, kp2, all_points_3d, all_poses, matches, i, show):
        ax1, ax2, ax3, ax4, ax5, ax6, x_min, x_max, y_min, y_max, z_min, z_max = vis_param
        
        c_pose = all_poses[-1]
        points_3d = all_points_3d[-1]
        # transform local points to world
        points_3d_world = (c_pose[:, :3] @ points_3d.T + c_pose[:, 3:]).T

        ax1.clear()
        ax1.imshow(image1, cmap='gray')
        ax1.axis('off')

        outimg1 = cv2.drawKeypoints(image1, kp1, None)
        ax2.clear()
        ax2.imshow(outimg1)
        ax2.axis('off')

        matched_img = cv2.drawMatches(image1, kp1, image2, kp2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        ax3.clear()
        ax3.imshow(matched_img)
        ax3.axis('off')

        ax4.clear()
        ax4.set_xlim([x_min + c_pose[0, -1], x_max + c_pose[0, -1]])
        ax4.set_ylim([y_min + c_pose[1, -1], y_max + c_pose[1, -1]])
        ax4.set_zlim([z_min + c_pose[2, -1], z_max + c_pose[2, -1]])
        ax4.scatter(points_3d_world[:,0], points_3d_world[:,1], points_3d_world[:,2], s=10, c='green')
        ax4.scatter(c_pose[0, -1], c_pose[1, -1], c_pose[2, -1], c='red', label='Camera')
        ax4.legend()

        ax5.clear()
        ax5.grid()
        ax5.set_xlim([x_min + c_pose[0, -1], x_max + c_pose[0, -1]])
        ax5.set_ylim([z_min + c_pose[2, -1], z_max + c_pose[2, -1]])
        arr_poses = np.array(all_poses)
        ax5.scatter(arr_poses[:, 0, -1], arr_poses[:, 2, -1], s=10, c='red', label='Estimated camera')
        
        # if GT exists
        if self.gt is not None and i < self.gt.shape[0]:
            ax5.scatter(self.gt[:i, 0, -1], self.gt[:i, 2, -1], s=10, c='black', label='GT camera')

        ax5.scatter(points_3d_world[:,0], points_3d_world[:,2], s=10, c='green', alpha=0.5)
        ax5.legend()

        ax6.clear()
        ax6.plot(range(len(all_points_3d)), [p.shape[0] for p in all_points_3d], label='num matched keypoints')
        ax6.grid()
        ax6.legend()

        plt.tight_layout()
        if show:
            # pass
            plt.pause(0.01)
            plt.show()

    # ---------------------------------------------------------------------
    # LOSSES
    # ---------------------------------------------------------------------
    def compute_ate(self, est_pose, gt_pose):
        t_est = est_pose[:, 3]
        t_gt = gt_pose[:, 3]
        return np.linalg.norm(t_est - t_gt)

    def compute_rte(self, est_pose_old, est_pose_new, gt_pose_old, gt_pose_new):
        old_est_t = est_pose_old[:, 3]
        new_est_t = est_pose_new[:, 3]
        old_gt_t  = gt_pose_old[:, 3]
        new_gt_t  = gt_pose_new[:, 3]
        rel_est = new_est_t - old_est_t
        rel_gt  = new_gt_t - old_gt_t
        return np.linalg.norm(rel_est - rel_gt)

    # ---------------------------------------------------------------------
    # INITIALIZE (KEYFRAMES)
    # ---------------------------------------------------------------------
    def initialize(self, keyframe_filename):
        if os.path.exists(keyframe_filename):
            return

        c_pose = np.hstack([np.eye(3), np.zeros((3,1))])  # 3x4
        all_poses = [c_pose]
        keyframes = [0]

        print("Finding keyframes")
        initial_frame = True
        for id_n_keyframe in tqdm(range(1, len(self.image_list))):
            id_c_keyframe = keyframes[-1]
            pose_c_keyframe = all_poses[id_c_keyframe]

            img1 = self.read_image(id_c_keyframe)
            img2 = self.read_image(id_n_keyframe)

            # Use LightGlue if enabled, otherwise use OpenCV pipeline.
            if self.use_lightglue:
                pos_m_kp1, pos_m_kp2, matches, kp1, kp2 = self.lightglue_match(img1, img2)
            else:
                kp1, des1 = self.feature_detection(img1)
                kp2, des2 = self.feature_detection(img2)
                m_kp1, m_kp2, matches = self.feature_matching(des1, des2, kp1, kp2)
                pos_m_kp1 = np.array([kp.pt for kp in m_kp1])
                pos_m_kp2 = np.array([kp.pt for kp in m_kp2])

            # Pose estimation, triangulation, refinement, etc. (unchanged)
            if initial_frame or not self.isStereo:
                R, t = self.pose_estimation(pos_m_kp1, pos_m_kp2)
                initial_frame = False
            else:
                # stereo approach (unchanged)
                q1_l, q1_r, q2_l, q2_r, mask_stereo = self.calculate_right_features(
                    pos_m_kp1, pos_m_kp2,
                    disparity1, disparity2
                )
                Q1, Q2 = self.get_stereo_3d_pts(q1_l, q1_r, q2_l, q2_r)
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=Q1.reshape(-1,1,3).astype(np.float32),
                    imagePoints=q2_l.reshape(-1,1,2).astype(np.float32),
                    cameraMatrix=self.K_3x3,
                    distCoeffs=None,
                    iterationsCount=500,
                    reprojectionError=5.0,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.reshape(3,1)
                else:
                    print("PnP RANSAC failed... (Stereo approach)")

            pts_3d = self.triangulate(R, t, pos_m_kp1, pos_m_kp2)
            R, t = self.refine(pos_m_kp1, pos_m_kp2, pts_3d, R, t, verbose=False)
            pts_3d = self.triangulate(R, t, pos_m_kp1, pos_m_kp2)
            inv_pose = np.linalg.inv(self.get_4by4_homo_proj(R, t))
            c_pose = c_pose @ inv_pose
            all_poses.append(c_pose)

            dist_from_keyframe = np.linalg.norm(c_pose[:3, -1] - pose_c_keyframe[:3, -1])
            good_depth_pts = pts_3d[pts_3d[:,2] > 0, 2]
            avg_depth = np.mean(good_depth_pts) if len(good_depth_pts) else 1e-6

            rel_R = np.linalg.inv(pose_c_keyframe[:3, :3]) @ c_pose[:3, :3]
            rod = cv2.Rodrigues(rel_R)[0]
            deg = np.abs(rod * 180.0 / np.pi)
            ratio = dist_from_keyframe / avg_depth

            if ratio > self.THUMPUP_POS_THRESHOLD or np.any(deg > self.THUMPUP_ROT_THRESHOLD):
                print(f"   [Keyframe] Dist: {dist_from_keyframe:.3f}, AvgDepth: {avg_depth:.3f}, ratio={ratio:.3f}, deg={deg.squeeze()}, frame={id_n_keyframe}")
                keyframes.append(id_n_keyframe)

        np.save(keyframe_filename, np.array(keyframes))



    # ---------------------------------------------------------------------
    # MAIN RUN
    # ---------------------------------------------------------------------
    def run(self, keyframe_filename, checkpoints=20, dynamic_vis=True):
        import matplotlib.pyplot as plt

        if os.path.exists(keyframe_filename):
            keyframes = np.load(keyframe_filename)
            print(f"loaded keyframes from file: {keyframes}")
        else:
            keyframes = np.arange(len(self.image_list))
            print("keyframe file not found, using all frames")

        vis_param = self.vis_init() if dynamic_vis else None

        c_pose = np.hstack((np.eye(3), np.zeros((3, 1))))
        all_poses = [c_pose]
        all_points_3d = []
        initial_frame = True
        cumulative_loss_loc = 0

        iter_count = len(keyframes)
        for i in tqdm(range(1, iter_count)):
            idx1 = keyframes[i - 1]
            idx2 = keyframes[i]
            img1 = self.read_image(idx1)
            img2 = self.read_image(idx2)

            if self.isStereo:
                img1_r = cv2.imread(self.image_list_r[idx1])
                img2_r = cv2.imread(self.image_list_r[idx2])
                disparity1 = self.compute_stereo_disparity(img1, img1_r)
                disparity2 = self.compute_stereo_disparity(img2, img2_r)

            if self.use_lightglue:
                pos_m_kp1, pos_m_kp2, matches, kp1, kp2 = self.lightglue_match(img1, img2)
            else:
                kp1, des1 = self.feature_detection(img1)
                kp2, des2 = self.feature_detection(img2)
                m_kp1, m_kp2, matches = self.feature_matching(des1, des2, kp1, kp2)
                pos_m_kp1 = np.array([kp.pt for kp in m_kp1])
                pos_m_kp2 = np.array([kp.pt for kp in m_kp2])

            if initial_frame or not self.isStereo:
                R, t = self.pose_estimation(pos_m_kp1, pos_m_kp2)
                initial_frame = False
            else:
                q1_l, q1_r, q2_l, q2_r, in_bounds = self.calculate_right_features(
                    pos_m_kp1, pos_m_kp2,
                    disparity1, disparity2
                )
                Q1, Q2 = self.get_stereo_3d_pts(q1_l, q1_r, q2_l, q2_r)
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    Q1.reshape(-1,1,3).astype(np.float32),
                    q2_l.reshape(-1,1,2).astype(np.float32),
                    self.K_3x3, None,
                    iterationsCount=2000,
                    reprojectionError=5.0,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.reshape(3,1)
                else:
                    print("[Warning] Stereo PnP RANSAC failed to find a valid pose.")

            points_3d = self.triangulate(R, t, pos_m_kp1, pos_m_kp2)
            R, t = self.refine(pos_m_kp1, pos_m_kp2, points_3d, R, t)
            points_3d = self.triangulate(R, t, pos_m_kp1, pos_m_kp2)
            all_points_3d.append(points_3d)

            inv_pose = np.linalg.inv(self.get_4by4_homo_proj(R, t))
            c_pose = c_pose @ inv_pose
            all_poses.append(c_pose)

            ate_loss = -1
            rte_loss = -1
            if (self.gt is not None) and (idx2 < self.gt.shape[0]):
                gt_pose_current = self.gt[idx2]
                diff_mean = np.mean(np.abs(gt_pose_current[:, 3] - c_pose[:, 3]))
                cumulative_loss_loc += diff_mean
                ate_loss = self.compute_ate(c_pose, gt_pose_current)

            # --- Visualization
            if dynamic_vis:
                self.vis(vis_param, img1, img2, kp1, kp2, all_points_3d, all_poses, matches, idx1, show=True)

            # --- Save screenshots every 'checkpoints' frames
            if (i % checkpoints == 0) or (i == iter_count - 1):
                if dynamic_vis:
                    self.vis(vis_param, img1, img2, kp1, kp2, all_points_3d, all_poses, matches, idx1, show=False)

                    out_dir = os.path.join(output_dir, self.dataset)
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)

                    if ate_loss == -1:
                        fname = f"{idx1}_noGT.png"
                    else:
                        fname = f"{idx1}_{(cumulative_loss_loc / i):.2f}_{ate_loss:.2f}.png"

                    plt.savefig(os.path.join(out_dir, fname), dpi=100)

            # ----------------------- NEW HELPER FUNCTIONS -----------------------

    def numpy_to_tensor(self, image):
        """Converts a BGR numpy image (as read by cv2) into a torch tensor of shape (1, 3, H, W)
        normalized in [0,1] and moved to GPU if available."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.cuda() if torch.cuda.is_available() else tensor

    def convert_lightglue_to_opencv(self, keypoints0, keypoints1, matches):
        """
        Convert filtered LightGlue keypoints and matches into OpenCV-compatible KeyPoint and DMatch objects.
        Here, keypoints0 and keypoints1 are assumed to already be filtered (only matched keypoints).
        
        Returns:
            opencv_kp0: List of cv2.KeyPoint objects for image0.
            opencv_kp1: List of cv2.KeyPoint objects for image1.
            opencv_matches: List of cv2.DMatch objects where each match is simply (i,i).
        """
        n_matches = keypoints0.shape[0]  # number of matched keypoints
        opencv_kp0 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints0]
        opencv_kp1 = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints1]
        opencv_matches = []
        for i in range(n_matches):
            # Here, we assume that the i-th keypoint in keypoints0 matches the i-th in keypoints1.
            dmatch = cv2.DMatch(i, i, 0, 0.0)
            opencv_matches.append(dmatch)
        return opencv_kp0, opencv_kp1, opencv_matches



    def lightglue_match(self, img1, img2):
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
        # Convert image to torch tensor
        img1_tensor = self.numpy_to_tensor(img1)
        img2_tensor = self.numpy_to_tensor(img2)
        feats0 = self.extractor.extract(img1_tensor)
        feats1 = self.extractor.extract(img2_tensor)
        matches_out = self.matcher({'image0': feats0, 'image1': feats1})
        # Remove batch dimension using rbd from lightglue.utils
        feats0 = rbd(feats0)
        feats1 = rbd(feats1)
        matches_out = rbd(matches_out)
        
        # Get matches and keypoints as numpy arrays.
        matches = matches_out['matches']  # shape (K,2)
        # keypoints returned by LightGlue are numpy arrays of (x,y)
        keypoints0 = feats0['keypoints'][matches[..., 0]]
        keypoints1 = feats1['keypoints'][matches[..., 1]]
        
        # Convert the keypoints and matches to OpenCV formats.
        opencv_kp0, opencv_kp1, opencv_matches = self.convert_lightglue_to_opencv(keypoints0, keypoints1, matches)
        
        return keypoints0, keypoints1, opencv_matches, opencv_kp0, opencv_kp1


    # ----------------------- OVERRIDDEN METHODS IN initialize() and run() -----------------------

    
    
# ================================
# Example main script usage
# ================================
if __name__ == "__main__":
    thumpup_pos_thresh = 0.015
    thumpup_rot_thresh = 1.5
    dataset = "kitti"  # can be "kitti", "parking", "malaga", or "custom"

    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    if dataset == "malaga":
        keyframe_filename = f"./output/malaga_keyframes_{thumpup_pos_thresh}_{thumpup_rot_thresh}.npy"
        sfm = StructureFromMotion("./Dataset/malaga-urban-dataset-extract-07", thumpup_pos_thresh, thumpup_rot_thresh)
    elif dataset == "parking":
        keyframe_filename = f"./output/parking_keyframes_{thumpup_pos_thresh}_{thumpup_rot_thresh}.npy"
        sfm = StructureFromMotion("./Dataset/parking", thumpup_pos_thresh, thumpup_rot_thresh)
    elif dataset == "kitti":
        keyframe_filename = f"./output/kitti_keyframes_{thumpup_pos_thresh}_{thumpup_rot_thresh}.npy"
        sfm = StructureFromMotion("./Dataset/kitti", thumpup_pos_thresh, thumpup_rot_thresh)
    elif dataset == "custom":
        keyframe_filename = f"./output/custom_keyframes_{thumpup_pos_thresh}_{thumpup_rot_thresh}.npy"
        sfm = StructureFromMotion("./Dataset/custom", thumpup_pos_thresh, thumpup_rot_thresh)
    else:
        raise ValueError("Unknown dataset")

    sfm.initialize(keyframe_filename)
    sfm.run(keyframe_filename, checkpoints=20, dynamic_vis=True)

    # If you want to block at the end (so the figure remains):
    # plt.show(block=True)
