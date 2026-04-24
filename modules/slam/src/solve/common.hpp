#ifndef SLAM_SOLVE_UTIL_H
#define SLAM_SOLVE_UTIL_H

#include "type.hpp"

#include <vector>

#include <opencv2/core/types.hpp>

namespace cv::slam {
namespace solve {

void normalize(const std::vector<cv::KeyPoint>& keypts, std::vector<cv::Point2f>& normalized_pts, Mat33_t& transform);

} // namespace solve
} // namespace cv::slam

#endif // SLAM_SOLVE_UTIL_H
