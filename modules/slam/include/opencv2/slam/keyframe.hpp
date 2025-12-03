#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace cv {
namespace vo {

struct KeyFrame {
    int id = -1;
    Mat image; // optional
    std::vector<KeyPoint> kps;
    Mat desc;
    // pose: rotation and translation to map coordinates
    // X_world = R * X_cam + t
    Mat R_w = Mat::eye(3,3,CV_64F);
    Mat t_w = Mat::zeros(3,1,CV_64F);
};

} // namespace vo
} // namespace cv