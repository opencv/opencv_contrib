#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "opencv2/slam/keyframe.hpp"
#include "opencv2/slam/map.hpp"

namespace cv {
namespace vo {

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
        double fx, double fy, double cx, double cy,
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