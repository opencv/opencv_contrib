// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <string>
#include <vector>

namespace cv {
namespace vo {

//! @addtogroup slam
//! @{

/**
 * @brief Tracking state enumeration for VisualOdometry.
 */
enum TrackingState {
    NOT_INITIALIZED = 0,  //!< System not yet initialized
    INITIALIZING = 1,     //!< Waiting for initialization
    TRACKING = 2,         //!< Successfully tracking
    LOST = 3              //!< Tracking lost
};

/**
 * @brief Result returned by VisualOdometry::track() for each processed frame.
 */
struct CV_EXPORTS_W_SIMPLE TrackingResult {
    CV_PROP_RW bool ok = false;               //!< True if pose was successfully estimated
    CV_PROP_RW int state = 0;                 //!< TrackingState as integer
    CV_PROP_RW cv::Mat R_w;                   //!< Rotation matrix (world frame) 3x3
    CV_PROP_RW cv::Mat t_w;                   //!< Translation vector (world frame) 3x1
    CV_PROP_RW int numMatches = 0;            //!< Number of feature matches found
    CV_PROP_RW int numInliers = 0;            //!< Number of inliers after RANSAC
    CV_PROP_RW bool keyframeInserted = false; //!< True if this frame became a keyframe
    CV_PROP_RW int frameId = -1;              //!< Frame ID
    CV_PROP_RW double timestamp = 0.0;        //!< Frame timestamp
};

/**
 * @brief Options for the Visual Odometry frontend (per-frame tracking).
 */
struct CV_EXPORTS_W_SIMPLE VisualOdometryOptions {
    // --- Matching parameters ---
    CV_PROP_RW int minMatches = 15;                 //!< Minimum matches to attempt pose estimation
    CV_PROP_RW int minInliers = 4;                  //!< Minimum inliers for valid pose
    CV_PROP_RW double minInlierRatio = 0.1;         //!< Minimum inlier ratio
    CV_PROP_RW int maxMatchesKeep = 500;            //!< Max matches to retain after sorting
    CV_PROP_RW double flowWeightLambda = 5.0;       //!< Flow weight lambda for scoring

    // --- Pose estimation thresholds ---
    CV_PROP_RW double diffZeroThresh = 2.0;         //!< Diff zero threshold
    CV_PROP_RW double flowZeroThresh = 0.3;         //!< Flow zero threshold
    CV_PROP_RW double minTranslationNorm = 1e-4;    //!< Min translation norm
    CV_PROP_RW double minRotationRad = 0.5 * CV_PI / 180.0; //!< Min rotation (radians)

    // --- Keyframe insertion policy ---
    CV_PROP_RW int keyframeMinGap = 1;              //!< Minimum frames between keyframes
    CV_PROP_RW int keyframeMaxGap = 8;              //!< Maximum frames before forcing keyframe
    CV_PROP_RW double keyframeMinParallaxPx = 8.0;  //!< Minimum parallax in pixels

    // --- Verbose logging ---
    CV_PROP_RW bool verbose = false;                //!< Enable verbose logging
};

class MapManager;
struct KeyFrame;

/**
 * @brief Visual Odometry frontend for monocular camera pose estimation.
 */
class CV_EXPORTS_W VisualOdometry {
public:
    CV_WRAP explicit VisualOdometry(
        cv::Ptr<cv::Feature2D> detector = cv::ORB::create(),
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING));

    ~VisualOdometry();

    VisualOdometry(const VisualOdometry&) = delete;
    VisualOdometry& operator=(const VisualOdometry&) = delete;

    CV_WRAP void setCameraIntrinsics(double fx, double fy, double cx, double cy);
    CV_WRAP void setWorldScale(double scale);

    CV_WRAP void setOptions(const VisualOdometryOptions& options);
    CV_WRAP VisualOdometryOptions getOptions() const;

    CV_WRAP void reset();

    CV_WRAP TrackingResult track(cv::InputArray frame, double timestamp = 0.0);

    // Advanced C++ API (not wrapped)
    TrackingResult track(cv::InputArray frame, double timestamp, MapManager* map);
    TrackingResult track(cv::InputArray frame, double timestamp, MapManager* map, bool allowMapping);

    CV_WRAP int getState() const;
    CV_WRAP int getFrameId() const;
    CV_WRAP void getCurrentPose(cv::OutputArray R_out, cv::OutputArray t_out) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

//! @}

} // namespace vo
} // namespace cv
