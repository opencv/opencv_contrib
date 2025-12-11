#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

namespace cv {
namespace vo {

struct CV_EXPORTS_W_SIMPLE VisualOdometryOptions {
    CV_PROP_RW int min_matches = 15;
    CV_PROP_RW int min_inliers = 4;
    CV_PROP_RW double min_inlier_ratio = 0.1;
    CV_PROP_RW double diff_zero_thresh = 2.0;
    CV_PROP_RW double flow_zero_thresh = 0.3;
    CV_PROP_RW double min_translation_norm = 1e-4;
    CV_PROP_RW double min_rotation_rad = 0.5 * CV_PI / 180.0;
    CV_PROP_RW int max_matches_keep = 500;
    CV_PROP_RW double flow_weight_lambda = 5.0;
};

class CV_EXPORTS_W VisualOdometry {
public:
    CV_WRAP VisualOdometry(cv::Ptr<cv::Feature2D> feature, cv::Ptr<cv::DescriptorMatcher> matcher);
    CV_WRAP int run(const std::string &imageDir, double scale_m = 1.0, const VisualOdometryOptions &options = VisualOdometryOptions());
private:
    cv::Ptr<cv::Feature2D> feature_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

}
} // namespace cv::vo