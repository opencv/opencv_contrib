#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

namespace cv {
namespace vo {

struct VisualOdometryOptions {
    int min_matches = 15;
    int min_inliers = 4;
    double min_inlier_ratio = 0.1;
    double diff_zero_thresh = 2.0;
    double flow_zero_thresh = 0.3;
    double min_translation_norm = 1e-4;
    double min_rotation_rad = 0.5 * CV_PI / 180.0;
    int max_matches_keep = 500;
    double flow_weight_lambda = 5.0;
};

class VisualOdometry {
public:
    VisualOdometry(cv::Ptr<cv::Feature2D> feature, cv::Ptr<cv::DescriptorMatcher> matcher);
    int run(const std::string &imageDir, double scale_m = 1.0, const VisualOdometryOptions &options = VisualOdometryOptions());
private:
    cv::Ptr<cv::Feature2D> feature_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

}
} // namespace cv::vo