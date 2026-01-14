#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace cv {
namespace vo {

struct KeyFrame {
        // Custom constructor for convenient initialization
        KeyFrame(int id_, const Mat& image_, const std::vector<KeyPoint>& kps_, const Mat& desc_, const Mat& R_w_, const Mat& t_w_)
            : id(id_), image(image_.clone()), kps(kps_), desc(desc_.clone()), R_w(R_w_.clone()), t_w(t_w_.clone()) {}
        // Constructor with timestamp
        KeyFrame(int id_, double timestamp_, const Mat& image_, const std::vector<KeyPoint>& kps_, const Mat& desc_, const Mat& R_w_, const Mat& t_w_)
            : id(id_), timestamp(timestamp_), image(image_.clone()), kps(kps_), desc(desc_.clone()), R_w(R_w_.clone()), t_w(t_w_.clone()) {}
    int id = -1;
    double timestamp = 0.0;  //!< Timestamp when this keyframe was captured
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