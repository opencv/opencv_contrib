#ifndef SLAM_DATA_MARKER2D_H
#define SLAM_DATA_MARKER2D_H

#include "type.hpp"
#include <opencv2/core/types.hpp>
#include <Eigen/Core>

namespace cv::slam {
namespace marker_model {
class base;
}

namespace data {

class marker2d {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! constructor
    marker2d(const std::vector<cv::Point2f>& undist_corners, const eigen_alloc_vector<Vec3_t>& bearings,
             const Mat33_t& rot_cm, const Vec3_t& trans_cm, unsigned int id, const std::shared_ptr<marker_model::base>& marker_model,
             const std::vector<cv::Point2f>& dist_corners);

    //! Compute corner positions on the world from camera pose and corner positions on the camera
    eigen_alloc_vector<Vec3_t> compute_corners_pos_w(const Mat44_t& cam_pose_wc, const eigen_alloc_vector<Vec3_t>& corners_pos) const;

    //! undistorted corner points
    std::vector<cv::Point2f> undist_corners_;

    //! bearing of corners
    eigen_alloc_vector<Vec3_t> bearings_;

    //! marker pose (camera -> marker)
    Mat33_t rot_cm_;
    Vec3_t trans_cm_;

    //! marker ID
    unsigned int id_;

    //! marker model
    std::shared_ptr<marker_model::base> marker_model_;

    std::vector<cv::Point2f> dist_corners_; // Keep these to draw the markers
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_MARKER2D_H
