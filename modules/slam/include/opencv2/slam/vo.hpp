#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

namespace cv {
namespace vo {

struct CV_EXPORTS_W_SIMPLE VisualOdometryOptions {
    CV_PROP_RW int minMatches = 15;
    CV_PROP_RW int minInliers = 4;
    CV_PROP_RW double minInlierRatio = 0.1;
    CV_PROP_RW double diffZeroThresh = 2.0;
    CV_PROP_RW double flowZeroThresh = 0.3;
    CV_PROP_RW double minTranslationNorm = 1e-4;
    CV_PROP_RW double minRotationRad = 0.5 * CV_PI / 180.0;
    CV_PROP_RW int maxMatchesKeep = 500;
    CV_PROP_RW double flowWeightLambda = 5.0;
    CV_PROP_RW bool verbose = false;

    // Keyframe insertion policy
    CV_PROP_RW int keyframeMinGap = 1;   // allow close KFs if有视差
    CV_PROP_RW int keyframeMaxGap = 8;   // force insertion if间隔过大
    CV_PROP_RW double keyframeMinParallaxPx = 8.0;

    // Backend BA cadence
    CV_PROP_RW int backendTriggerInterval = 2;
    CV_PROP_RW bool enableBackend = true;
    CV_PROP_RW int backendWindow = 45;
    CV_PROP_RW int backendIterations = 15;

    // Redundant keyframe culling
    CV_PROP_RW int maxKeyframeCullsPerMaintenance = 2;
    CV_PROP_RW int redundantKeyframeMinObs = 80;
    CV_PROP_RW int redundantKeyframeMinPointObs = 3;
    CV_PROP_RW double redundantKeyframeRatio = 0.90;

    // Map maintenance
    CV_PROP_RW bool enableMapMaintenance = true;
    CV_PROP_RW int maintenanceInterval = 5;

    // Map point culling / retention
    CV_PROP_RW int mapMaxPointsKeep = 8000;
    CV_PROP_RW double mapMaxReprojErrorPx = 5.0;
    CV_PROP_RW double mapMinFoundRatio = 0.10;
    CV_PROP_RW int mapMinObservations = 2;
    // Verbose diagnostics (OpenCV log level is global).
};

class CV_EXPORTS_W VisualOdometry {
public:
    CV_WRAP VisualOdometry(cv::Ptr<cv::Feature2D> feature, cv::Ptr<cv::DescriptorMatcher> matcher);
    // Run with explicit options (keeps compatibility with callers who pass a full options struct)
    CV_WRAP int run(const std::string &imageDir, double scale_m, const VisualOdometryOptions &options);
    // Run using the internally stored options (affected by setters such as setEnableBackend)
    CV_WRAP int run(const std::string &imageDir, double scale_m = 1.0);

    // Convenience setters for backend controls
    CV_WRAP void setEnableBackend(bool enable);
    CV_WRAP void setBackendWindow(int window);
    CV_WRAP void setBackendIterations(int iterations);
private:
    cv::Ptr<cv::Feature2D> feature_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    VisualOdometryOptions options_{}; // stored defaults for setter-based configuration
};

}
} // namespace cv::vo