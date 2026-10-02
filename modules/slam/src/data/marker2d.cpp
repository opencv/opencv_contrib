#include "data/marker2d.hpp"

namespace cv::slam {
namespace data {

marker2d::marker2d(const std::vector<cv::Point2f>& undist_corners, const eigen_alloc_vector<Vec3_t>& bearings,
                   const Mat33_t& rot_cm, const Vec3_t& trans_cm, unsigned int id, const std::shared_ptr<marker_model::base>& marker_model,
                   const std::vector<cv::Point2f>& dist_corners)
    : undist_corners_(undist_corners), bearings_(bearings), rot_cm_(rot_cm), trans_cm_(trans_cm), id_(id), marker_model_(marker_model),
      dist_corners_(dist_corners) {}

eigen_alloc_vector<Vec3_t> marker2d::compute_corners_pos_w(const Mat44_t& cam_pose_wc, const eigen_alloc_vector<Vec3_t>& corners_pos) const {
    eigen_alloc_vector<Vec3_t> corners_pos_w;
    for (const Vec3_t& corner_pos : corners_pos) {
        const Mat33_t rot_wc = cam_pose_wc.block<3, 3>(0, 0);
        const Vec3_t trans_wc = cam_pose_wc.block<3, 1>(0, 3);
        corners_pos_w.push_back(rot_wc * (rot_cm_ * corner_pos + trans_cm_) + trans_wc);
    }
    return corners_pos_w;
}

} // namespace data
} // namespace cv::slam
