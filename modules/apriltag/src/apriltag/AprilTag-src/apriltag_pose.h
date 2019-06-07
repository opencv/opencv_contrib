#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#include "apriltag.h"
#include "common/matd.h"

typedef struct {
    apriltag_detection_t* det;
    double tagsize; // In meters.
    double fx; // In pixels.
    double fy; // In pixels.
    double cx; // In pixels.
    double cy; // In pixels.
} apriltag_detection_info_t;

typedef struct {
    matd_t* R;
    matd_t* t;
} apriltag_pose_t;

/**
 * Estimate pose of the tag using the homography method described in [1].
 * @outparam pose
 */
void estimate_pose_for_tag_homography(
        apriltag_detection_info_t* info,
        apriltag_pose_t* pose);

/**
 * Estimate pose of the tag. This returns one or two possible poses for the
 * tag, along with the object-space error of each.
 *
 * This uses the homography method described in [1] for the initial estimate.
 * Then Orthogonal Iteration [2] is used to refine this estimate. Then [3] is
 * used to find a potential second local minima and Orthogonal Iteration is
 * used to refine this second estimate.
 *
 * [1]: E. Olson, “Apriltag: A robust and flexible visual fiducial system,” in
 *      2011 IEEE International Conference on Robotics and Automation,
 *      May 2011, pp. 3400–3407.
 * [2]: Lu, G. D. Hager and E. Mjolsness, "Fast and globally convergent pose
 *      estimation from video images," in IEEE Transactions on Pattern Analysis
 *      and Machine Intelligence, vol. 22, no. 6, pp. 610-622, June 2000.
 *      doi: 10.1109/34.862199
 * [3]: Schweighofer and A. Pinz, "Robust Pose Estimation from a Planar Target,"
 *      in IEEE Transactions on Pattern Analysis and Machine Intelligence,
 *      vol. 28, no. 12, pp. 2024-2030, Dec. 2006.  doi: 10.1109/TPAMI.2006.252
 *
 * @outparam err1, pose1, err2, pose2
 */
void estimate_tag_pose_orthogonal_iteration(
        apriltag_detection_info_t* info,
        double* err1,
        apriltag_pose_t* pose1,
        double* err2,
        apriltag_pose_t* pose2,
        int nIters);

/**
 * Estimate tag pose.
 * This method is an easier to use interface to estimate_tag_pose_orthogonal_iteration.
 *
 * @outparam pose 
 * @return Object-space error of returned pose.
 */
double estimate_tag_pose(apriltag_detection_info_t* info, apriltag_pose_t* pose);

#ifdef __cplusplus
}
#endif

