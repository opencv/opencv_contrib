#include "data/common.hpp"
#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "feature/orb_extractor.hpp"
#include "match/stereo.hpp"

#include <thread>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace data {

frame::frame(unsigned int frame_id, const double timestamp, camera::base* camera, feature::orb_params* orb_params,
             const frame_observation frm_obs, const std::unordered_map<unsigned int, marker2d>& markers_2d)
    : id_(frame_id), timestamp_(timestamp), camera_(camera), orb_params_(orb_params), frm_obs_(frm_obs),
      markers_2d_(markers_2d),
      // Initialize association with 3D points
      landmarks_(std::vector<std::shared_ptr<landmark>>(frm_obs_.undist_keypts_.size(), nullptr)) {}

void frame::set_pose_cw(const Mat44_t& pose_cw) {
    pose_is_valid_ = true;
    pose_cw_ = pose_cw;

    rot_cw_ = pose_cw_.block<3, 3>(0, 0);
    rot_wc_ = rot_cw_.transpose();
    trans_cw_ = pose_cw_.block<3, 1>(0, 3);
    trans_wc_ = -rot_cw_.transpose() * trans_cw_;
}

Mat44_t frame::get_pose_cw() const {
    return pose_cw_;
}

Mat44_t frame::get_pose_wc() const {
    Mat44_t pose_wc = Mat44_t::Identity();
    pose_wc.block<3, 3>(0, 0) = rot_wc_;
    pose_wc.block<3, 1>(0, 3) = trans_wc_;
    return pose_wc;
}

Vec3_t frame::get_trans_wc() const {
    return trans_wc_;
}

Mat33_t frame::get_rot_wc() const {
    return rot_wc_;
}

bool frame::bow_is_available() const {
    return !bow_vec_.empty() && !bow_feat_vec_.empty();
}

void frame::compute_bow(bow_vocabulary* bow_vocab) {
    bow_vocabulary_util::compute_bow(bow_vocab, frm_obs_.descriptors_, bow_vec_, bow_feat_vec_);
}

bool frame::can_observe(const std::shared_ptr<landmark>& lm, const float ray_cos_thr,
                        Vec2_t& reproj, float& x_right, unsigned int& pred_scale_level) const {
    const Vec3_t pos_w = lm->get_pos_in_world();

    const bool in_image = camera_->reproject_to_image(rot_cw_, trans_cw_, pos_w, reproj, x_right);
    if (!in_image) {
        return false;
    }

    const Vec3_t cam_to_lm_vec = pos_w - trans_wc_;
    const auto cam_to_lm_dist = cam_to_lm_vec.norm();
    const auto margin_far = 1.3;
    const auto margin_near = 1.0 / margin_far;
    if (!lm->is_inside_in_orb_scale(cam_to_lm_dist, margin_far, margin_near)) {
        return false;
    }

    const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();
    const auto ray_cos = cam_to_lm_vec.dot(obs_mean_normal) / cam_to_lm_dist;
    if (ray_cos < ray_cos_thr) {
        return false;
    }

    pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, this->orb_params_->num_levels_, this->orb_params_->log_scale_factor_);
    return true;
}

bool frame::has_landmark(const std::shared_ptr<landmark>& lm) const {
    return static_cast<bool>(landmarks_idx_map_.count(lm));
}

void frame::add_landmark(const std::shared_ptr<landmark>& lm, const unsigned int idx) {
    CV_LOG_DEBUG(&g_log_tag, "frame::add_landmark " << id_);
    assert(!has_landmark(lm));
    landmarks_.at(idx) = lm;
    landmarks_idx_map_[lm] = idx;
}

std::shared_ptr<landmark> frame::get_landmark(const unsigned int idx) const {
    return landmarks_.at(idx);
}

void frame::erase_landmark_with_index(const unsigned int idx) {
    assert(landmarks_.at(idx));
    landmarks_idx_map_.erase(landmarks_.at(idx));
    landmarks_.at(idx) = nullptr;
}

void frame::erase_landmark(const std::shared_ptr<landmark>& lm) {
    assert(has_landmark(lm));
    auto idx = landmarks_idx_map_[lm];
    landmarks_idx_map_.erase(lm);
    landmarks_.at(idx) = nullptr;
}

std::vector<std::shared_ptr<landmark>> frame::get_landmarks() const {
    return landmarks_;
}

void frame::erase_landmarks() {
    std::fill(landmarks_.begin(), landmarks_.end(), nullptr);
    landmarks_idx_map_.clear();
}

void frame::set_landmarks(const std::vector<std::shared_ptr<landmark>>& landmarks) {
    erase_landmarks();
    for (unsigned int idx = 0; idx < landmarks.size(); ++idx) {
        const auto& lm = landmarks.at(idx);
        if (lm) {
            add_landmark(lm, idx);
        }
    }
}

std::vector<unsigned int> frame::get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin, const int min_level, const int max_level) const {
    return data::get_keypoints_in_cell(camera_, frm_obs_, ref_x, ref_y, margin, min_level, max_level);
}

Vec3_t frame::triangulate_stereo(const unsigned int idx) const {
    return data::triangulate_stereo(camera_, rot_wc_, trans_wc_, frm_obs_, idx);
}

} // namespace data
} // namespace cv::slam
