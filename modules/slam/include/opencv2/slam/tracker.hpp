// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <vector>

#include "opencv2/slam/feature.hpp"
#include "opencv2/slam/matcher.hpp"

namespace cv {
namespace vo {

class CV_EXPORTS Tracker {
public:
    Tracker();

    // Process a gray image. imgOut contains visualization (matches or keypoints).
    bool processFrame(const cv::Mat& gray,
                      const std::string& imagePath,
                      cv::Mat& imgOut,
                      cv::Mat& R_out,
                      cv::Mat& t_out,
                      std::string& info);

private:
    FeatureExtractor feat_;
    Matcher matcher_;

    cv::Mat prevGray_, prevDesc_;
    std::vector<cv::KeyPoint> prevKp_;
    int frame_id_;
};

} // namespace vo
} // namespace cv
