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
    // Backend/BA controls
    CV_PROP_RW bool enableBackend = true;
    CV_PROP_RW int backendWindow = 5;       // number of latest keyframes for local BA
    CV_PROP_RW int backendIterations = 10;  // BA iterations (if backend enabled)
    // Map maintenance controls
    CV_PROP_RW bool enableMapMaintenance = true;
    CV_PROP_RW int maintenanceInterval = 5; // run cull/descriptor refresh every N keyframes
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