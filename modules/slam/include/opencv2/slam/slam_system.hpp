// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "opencv2/slam/visual_odometry.hpp"

namespace cv {
namespace vo {

//! @addtogroup slam
//! @{

/**
 * @brief System operating mode.
 */
enum Mode {
    MODE_SLAM = 0,          //!< Build/extend map while tracking
    MODE_LOCALIZATION = 1   //!< Localize against a pre-built (frozen) map
};

/**
 * @brief Options for the SlamSystem (backend and system-level).
 */
struct CV_EXPORTS_W_SIMPLE SlamSystemOptions {
    // --- Backend BA parameters ---
    CV_PROP_RW bool enableBackend = true;          //!< Enable backend BA thread
    CV_PROP_RW int backendTriggerInterval = 2;     //!< Keyframes between BA triggers
    CV_PROP_RW int backendWindow = 45;             //!< BA sliding window size
    CV_PROP_RW int backendIterations = 15;         //!< BA iterations per trigger

    // --- Map maintenance ---
    CV_PROP_RW bool enableMapMaintenance = true;   //!< Enable map maintenance
    CV_PROP_RW int maintenanceInterval = 5;        //!< Keyframes between maintenance

    // --- Redundant keyframe culling ---
    CV_PROP_RW int maxKeyframeCullsPerMaintenance = 2;
    CV_PROP_RW int redundantKeyframeMinObs = 80;
    CV_PROP_RW int redundantKeyframeMinPointObs = 3;
    CV_PROP_RW double redundantKeyframeRatio = 0.90;

    // --- Map point culling / retention ---
    CV_PROP_RW int mapMaxPointsKeep = 8000;        //!< Max map points to keep
    CV_PROP_RW double mapMaxReprojErrorPx = 5.0;   //!< Max reprojection error in pixels
    CV_PROP_RW double mapMinFoundRatio = 0.10;     //!< Min found ratio for map points
    CV_PROP_RW int mapMinObservations = 2;         //!< Min observations for map points
};

class CV_EXPORTS_W SlamSystem {
public:
    CV_WRAP SlamSystem();
    CV_WRAP explicit SlamSystem(cv::Ptr<cv::Feature2D> detector,
                                cv::Ptr<cv::DescriptorMatcher> matcher);

    ~SlamSystem();

    SlamSystem(const SlamSystem&) = delete;
    SlamSystem& operator=(const SlamSystem&) = delete;

    CV_WRAP void setFrontendOptions(const VisualOdometryOptions& options);
    CV_WRAP void setSystemOptions(const SlamSystemOptions& options);

    CV_WRAP void setCameraIntrinsics(double fx, double fy, double cx, double cy);
    CV_WRAP void setWorldScale(double scale_m);

    CV_WRAP void setMode(int mode);
    CV_WRAP int getMode() const;

    CV_WRAP VisualOdometryOptions getFrontendOptions() const;
    CV_WRAP SlamSystemOptions getSystemOptions() const;

    CV_WRAP void setEnableBackend(bool enable);
    CV_WRAP void setBackendWindow(int window);
    CV_WRAP void setBackendIterations(int iterations);

    CV_WRAP TrackingResult track(cv::InputArray frame, double timestamp = 0.0);

    CV_WRAP bool saveTrajectoryTUM(const std::string& path) const;
    CV_WRAP bool saveMap(const std::string& path) const;
    CV_WRAP bool loadMap(const std::string& path);

    CV_WRAP void reset();

    const MapManager& getMap() const;
    MapManager& getMapMutable();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

//! @}

} // namespace vo
} // namespace cv
