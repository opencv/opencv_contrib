#include "solve/common.hpp"

#include <numeric>

namespace cv::slam {
namespace solve {

void normalize(const std::vector<cv::KeyPoint>& keypts, std::vector<cv::Point2f>& normalized_pts, Mat33_t& transform) {
    const auto num_keypts = keypts.size();

    // compute centroids
    const auto pts_mean = std::accumulate(keypts.begin(), keypts.end(), cv::Point2f{0.0f, 0.0f},
                                          [](const cv::Point2f& acc, const cv::KeyPoint& keypt) { return acc + keypt.pt; })
                          / static_cast<double>(num_keypts);

    // compute average absolute deviations
    cv::Point2f pts_l1_dev{0.0f, 0.0f};
    normalized_pts.resize(num_keypts);
    for (unsigned int idx = 0; idx < num_keypts; ++idx) {
        // convert points to zero-mean distribution
        normalized_pts.at(idx) = keypts.at(idx).pt - pts_mean;
        // accumulate L1 distance from centroid
        pts_l1_dev += cv::Point2f{std::abs(normalized_pts.at(idx).x), std::abs(normalized_pts.at(idx).y)};
    }
    pts_l1_dev /= static_cast<double>(num_keypts);

    // apply normalization
    std::transform(normalized_pts.begin(), normalized_pts.end(), normalized_pts.begin(),
                   [&pts_l1_dev](const cv::Point2f& pt) { return cv::Point2f{pt.x / pts_l1_dev.x, pt.y / pts_l1_dev.y}; });

    // build transformation matrix
    transform << 1.0, 0.0, -pts_mean.x, 0.0, 1.0, -pts_mean.y, 0.0, 0.0, 1.0;
    transform.block<1, 3>(0, 0) /= pts_l1_dev.x;
    transform.block<1, 3>(1, 0) /= pts_l1_dev.y;
}

} // namespace solve
} // namespace cv::slam
