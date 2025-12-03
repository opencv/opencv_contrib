#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace vo {

class PoseEstimator {
public:
    PoseEstimator() = default;
    // Estimate relative pose from matched normalized image points. Returns true if pose recovered.
    bool estimate(const std::vector<Point2f> &pts1,
                  const std::vector<Point2f> &pts2,
                  double fx, double fy, double cx, double cy,
                  Mat &R, Mat &t, Mat &mask, int &inliers);
};

} // namespace vo
} // namespace cv