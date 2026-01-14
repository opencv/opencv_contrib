#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "opencv2/slam/keyframe.hpp"
#include "opencv2/slam/map.hpp"

namespace cv {
namespace vo {

struct PoseGraphEdge {
    int i = -1;   // from keyframe id
    int j = -1;   // to keyframe id
    cv::Mat R_ij; // 3x3, camera->world of relative? stored as Rwc form of transform from i to j (R_ij is R_w component of T_ij when composed as X_j = R_ij * X_i + t_ij)
    cv::Mat t_ij; // 3x1, translation in world frame of i (same convention as above)
    double weight = 1.0;
};

// Bundle Adjustment Optimizer using OpenCV-based Levenberg-Marquardt
// Note: For production, should use g2o or Ceres for better performance
class Optimizer {
public:
    Optimizer();

    // Local Bundle Adjustment
    // Optimizes a window of recent keyframes and all observed map points
    // fixedKFs: indices of keyframes to keep fixed during optimization
#if defined(HAVE_G2O)
    static void localBundleAdjustmentG2O(
        std::vector<KeyFrame> &keyframes,
        std::vector<MapPoint> &mappoints,
        const std::vector<int> &localKfIndices,
        const std::vector<int> &fixedKfIndices,
        double fx, double cx, double cy,
        int iterations = 10);
#endif

#if defined(HAVE_SFM)
    static void localBundleAdjustmentSFM(
        std::vector<KeyFrame> &keyframes,
        std::vector<MapPoint> &mappoints,
        const std::vector<int> &localKfIndices,
        const std::vector<int> &fixedKfIndices,
        double fx, double fy, double cx, double cy,
        int iterations = 10);
#endif
    // Pose-only optimization (optimize camera pose given fixed 3D points)
    static bool optimizePose(
        KeyFrame &kf,
        const std::vector<MapPoint> &mappoints,
        const std::vector<int> &matchedMpIndices,
        double fx, double fy, double cx, double cy,
        std::vector<bool> &inliers,
        int iterations = 10);

#if defined(HAVE_SFM)
    // Global Bundle Adjustment (expensive, use after loop closure)
    static void globalBundleAdjustmentSFM(
        std::vector<KeyFrame> &keyframes,
        std::vector<MapPoint> &mappoints,
        double fx, double fy, double cx, double cy,
        int iterations = 20);
#endif

    // Pose-graph optimization (loop-closure constraints).
    // Edges use module pose convention: keyframe pose is (R_w, C_w).
    // The relative constraint T_ij (R_ij, t_ij) represents the expected transform from i to j:
    //   R_pred = R_i^T * R_j, t_pred = R_i^T * (C_j - C_i);
    // Residual is formed on SE3 using small-angle approximation.
    static void poseGraphOptimize(
        std::vector<KeyFrame> &keyframes,
        const std::vector<PoseGraphEdge> &edges,
        const std::vector<int> &fixedKfIds,
        int iterations = 10,
        double step = 0.5);

private:
    // Compute reprojection error and Jacobian
    static double computeReprojectionError(
        const Point3d &point3D,
        const Mat &R, const Mat &t,
        const Point2f &observed,
        double fx, double fy, double cx, double cy,
        Mat &jacobianPose,
        Mat &jacobianPoint);

    // Project 3D point to image
    static Point2f project(
        const Point3d &point3D,
        const Mat &R, const Mat &t,
        double fx, double fy, double cx, double cy);
};

} // namespace vo
} // namespace cv