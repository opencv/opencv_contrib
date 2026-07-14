#include "camera/base.hpp"
#include "data/common.hpp"
#include "data/frame.hpp"
#include "data/frame_observation.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "match/projection.hpp"
#include "util/angle.hpp"

namespace cv::slam {
namespace match {

unsigned int projection::match_frame_and_landmarks(data::frame& frm,
                                                   const std::vector<std::shared_ptr<data::landmark>>& local_landmarks,
                                                   eigen_alloc_unord_map<unsigned int, Vec2_t>& lm_to_reproj,
                                                   std::unordered_map<unsigned int, float>& lm_to_x_right,
                                                   std::unordered_map<unsigned int, unsigned int>& lm_to_scale,
                                                   const float margin) const {
    unsigned int num_matches = 0;

    // Reproject the 3D points to the frame, then acquire the 2D-3D matches
    for (auto local_lm : local_landmarks) {
        if (!lm_to_reproj.count(local_lm->id_)) {
            continue;
        }
        if (local_lm->will_be_erased()) {
            continue;
        }

        // Acquire keypoints in the cell where the reprojected 3D points exist
        Vec2_t reproj = lm_to_reproj.at(local_lm->id_);
        const auto pred_scale_level = lm_to_scale.at(local_lm->id_);
        const int min_level = std::max(0, static_cast<int>(pred_scale_level) - 1);
        const int max_level = std::min(frm.orb_params_->num_levels_ - 1, pred_scale_level + 1);
        const auto indices_in_cell = frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                               margin * frm.orb_params_->scale_factors_.at(pred_scale_level),
                                                               min_level, max_level);
        if (indices_in_cell.empty()) {
            continue;
        }

        const cv::Mat lm_desc = local_lm->get_descriptor();

        unsigned int best_hamm_dist = MAX_HAMMING_DIST;
        int best_scale_level = -1;
        unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;
        int second_best_scale_level = -1;
        int best_idx = -1;

        for (const auto idx : indices_in_cell) {
            const auto& lm = frm.get_landmark(idx);
            if (lm && lm->has_observation()) {
                continue;
            }

            if (!frm.frm_obs_.stereo_x_right_.empty() && 0 < frm.frm_obs_.stereo_x_right_.at(idx)) {
                const auto reproj_error = std::abs(lm_to_x_right.at(local_lm->id_) - frm.frm_obs_.stereo_x_right_.at(idx));
                if (margin * frm.orb_params_->scale_factors_.at(pred_scale_level) < reproj_error) {
                    continue;
                }
            }

            const cv::Mat& desc = frm.frm_obs_.descriptors_.row(idx);

            const auto dist = compute_descriptor_distance_32(lm_desc, desc);

            if (dist < best_hamm_dist) {
                second_best_hamm_dist = best_hamm_dist;
                best_hamm_dist = dist;
                second_best_scale_level = best_scale_level;
                best_scale_level = frm.frm_obs_.undist_keypts_.at(idx).octave;
                best_idx = idx;
            }
            else if (dist < second_best_hamm_dist) {
                second_best_scale_level = frm.frm_obs_.undist_keypts_.at(idx).octave;
                second_best_hamm_dist = dist;
            }
        }

        if (best_hamm_dist <= HAMMING_DIST_THR_HIGH) {
            // Lowe's ratio test
            if (best_scale_level == second_best_scale_level && best_hamm_dist > lowe_ratio_ * second_best_hamm_dist) {
                continue;
            }

            // Add the matching information
            frm.add_landmark(local_lm, best_idx);
            ++num_matches;
        }
    }

    return num_matches;
}

unsigned int projection::match_current_and_last_frames(data::frame& curr_frm, const data::frame& last_frm, const float margin) const {
    unsigned int num_matches = 0;

    const Mat33_t rot_cw = curr_frm.get_rot_cw();
    const Vec3_t trans_cw = curr_frm.get_trans_cw();

    const Vec3_t trans_wc = -rot_cw.transpose() * trans_cw;

    const Mat33_t rot_lw = last_frm.get_rot_cw();
    const Vec3_t trans_lw = last_frm.get_trans_cw();

    const Vec3_t trans_lc = rot_lw * trans_wc + trans_lw;

    // For non-monocular, check if the z component of the current-to-last translation vector is moving forward
    // The z component is positive going -> moving forward
    const bool assume_forward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                    ? false
                                    : trans_lc(2) > curr_frm.camera_->true_baseline_;
    // The z component is negative going -> moving backward
    const bool assume_backward = (curr_frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                     ? false
                                     : -trans_lc(2) > curr_frm.camera_->true_baseline_;

    // Reproject the 3D points associated to the keypoints of the last frame,
    // then acquire the 2D-3D matches
    for (unsigned int idx_last = 0; idx_last < last_frm.frm_obs_.undist_keypts_.size(); ++idx_last) {
        const auto& lm = last_frm.get_landmark(idx_last);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // 3D point coordinates with the global reference
        const Vec3_t pos_w = lm->get_pos_in_world();

        // Reproject and compute visibility
        Vec2_t reproj;
        float x_right;
        const bool in_image = curr_frm.camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

        // Ignore if it is reprojected outside the image
        if (!in_image) {
            continue;
        }

        // Acquire keypoints in the cell where the reprojected 3D points exist
        const unsigned int last_scale_level = last_frm.frm_obs_.undist_keypts_.at(idx_last).octave;
        int min_level;
        int max_level;
        if (assume_forward) {
            min_level = last_scale_level;
            max_level = std::min(last_frm.orb_params_->num_levels_ - 1, last_scale_level + 1);
        }
        else if (assume_backward) {
            min_level = std::max(0, static_cast<int>(last_scale_level) - 1);
            max_level = last_scale_level;
        }
        else {
            min_level = std::max(0, static_cast<int>(last_scale_level) - 1);
            max_level = std::min(last_frm.orb_params_->num_levels_ - 1, last_scale_level + 1);
        }
        auto indices = curr_frm.get_keypoints_in_cell(reproj(0), reproj(1),
                                                      margin * curr_frm.orb_params_->scale_factors_.at(last_scale_level),
                                                      min_level, max_level);
        if (indices.empty()) {
            continue;
        }

        const auto lm_desc = lm->get_descriptor();

        unsigned int best_hamm_dist = MAX_HAMMING_DIST;
        int best_idx = -1;

        for (const auto curr_idx : indices) {
            const auto& curr_lm = curr_frm.get_landmark(curr_idx);
            if (curr_lm && curr_lm->has_observation()) {
                continue;
            }

            if (!curr_frm.frm_obs_.stereo_x_right_.empty() && curr_frm.frm_obs_.stereo_x_right_.at(curr_idx) > 0) {
                const float reproj_error = std::fabs(x_right - curr_frm.frm_obs_.stereo_x_right_.at(curr_idx));
                if (margin * curr_frm.orb_params_->scale_factors_.at(last_scale_level) < reproj_error) {
                    continue;
                }
            }

            if (check_orientation_ && std::abs(util::angle::diff(last_frm.frm_obs_.undist_keypts_.at(idx_last).angle, curr_frm.frm_obs_.undist_keypts_.at(curr_idx).angle)) > 30.0) {
                continue;
            }

            const auto& desc = curr_frm.frm_obs_.descriptors_.row(curr_idx);

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_hamm_dist) {
                best_hamm_dist = hamm_dist;
                best_idx = curr_idx;
            }
        }

        if (HAMMING_DIST_THR_HIGH < best_hamm_dist) {
            continue;
        }

        // The matching is valid
        curr_frm.add_landmark(lm, best_idx);
        ++num_matches;
    }

    return num_matches;
}

unsigned int projection::match_frame_and_keyframe(data::frame& curr_frm, const std::shared_ptr<data::keyframe>& keyfrm, const std::set<std::shared_ptr<data::landmark>>& already_matched_lms,
                                                  const float margin, const unsigned int hamm_dist_thr) const {
    auto lms = curr_frm.get_landmarks();
    auto num_matches = match_frame_and_keyframe(curr_frm.get_pose_cw(), curr_frm.camera_, curr_frm.frm_obs_, curr_frm.orb_params_, lms, keyfrm, already_matched_lms, margin, hamm_dist_thr);
    curr_frm.set_landmarks(lms);
    return num_matches;
}

unsigned int projection::match_frame_and_keyframe(const Mat44_t& cam_pose_cw,
                                                  const camera::base* camera,
                                                  const data::frame_observation& frm_obs,
                                                  const feature::orb_params* orb_params,
                                                  std::vector<std::shared_ptr<data::landmark>>& frm_landmarks,
                                                  const std::shared_ptr<data::keyframe>& keyfrm,
                                                  const std::set<std::shared_ptr<data::landmark>>& already_matched_lms,
                                                  const float margin, const unsigned int hamm_dist_thr) const {
    unsigned int num_matches = 0;

    const Mat33_t rot_cw = cam_pose_cw.block<3, 3>(0, 0);
    const Vec3_t trans_cw = cam_pose_cw.block<3, 1>(0, 3);
    const Vec3_t cam_center = -rot_cw.transpose() * trans_cw;

    const auto landmarks = keyfrm->get_landmarks();

    // Reproject the 3D points associated to the keypoints of the keyframe,
    // then acquire the 2D-3D matches
    for (unsigned int idx = 0; idx < landmarks.size(); idx++) {
        auto& lm = landmarks.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        // Avoid duplication
        if (already_matched_lms.count(lm)) {
            continue;
        }

        // 3D point coordinates with the global reference
        const Vec3_t pos_w = lm->get_pos_in_world();

        // Reproject and compute visibility
        Vec2_t reproj;
        float x_right;
        const bool in_image = camera->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

        // Ignore if it is reprojected outside the image
        if (!in_image) {
            continue;
        }

        // Check if it's within ORB scale levels
        const Vec3_t cam_to_lm_vec = pos_w - cam_center;
        const auto cam_to_lm_dist = cam_to_lm_vec.norm();
        constexpr auto margin_far = 1.3;
        constexpr auto margin_near = 1.0 / margin_far;
        const auto max_cam_to_lm_dist = margin_far * lm->get_max_valid_distance();
        const auto min_cam_to_lm_dist = margin_near * lm->get_min_valid_distance();

        if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
            continue;
        }

        // Acquire keypoints in the cell where the reprojected 3D points exist
        const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, orb_params->num_levels_, orb_params->log_scale_factor_);
        const int min_level = std::max(0, static_cast<int>(pred_scale_level) - 1);
        const int max_level = std::min(orb_params->num_levels_ - 1, pred_scale_level + 1);
        const auto indices = data::get_keypoints_in_cell(camera, frm_obs, reproj(0), reproj(1),
                                                         margin * orb_params->scale_factors_.at(pred_scale_level),
                                                         min_level, max_level);

        if (indices.empty()) {
            continue;
        }

        const auto lm_desc = lm->get_descriptor();

        unsigned int best_hamm_dist = MAX_HAMMING_DIST;
        int best_idx = -1;

        for (unsigned long curr_idx : indices) {
            if (frm_landmarks.at(curr_idx)) {
                continue;
            }

            if (check_orientation_ && std::abs(util::angle::diff(keyfrm->frm_obs_.undist_keypts_.at(idx).angle, frm_obs.undist_keypts_.at(curr_idx).angle)) > 30.0) {
                continue;
            }

            const auto& desc = frm_obs.descriptors_.row(curr_idx);

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_hamm_dist) {
                best_hamm_dist = hamm_dist;
                best_idx = curr_idx;
            }
        }

        if (hamm_dist_thr < best_hamm_dist) {
            continue;
        }

        // The matching is valid
        frm_landmarks.at(best_idx) = lm;
        num_matches++;
    }

    return num_matches;
}

unsigned int projection::match_by_Sim3_transform(const std::shared_ptr<data::keyframe>& keyfrm, const Mat44_t& Sim3_cw, const std::vector<std::shared_ptr<data::landmark>>& landmarks,
                                                 std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_keyfrm, const float margin) const {
    unsigned int num_matches = 0;

    // Convert Sim3 into SE3
    const Mat33_t s_rot_cw = Sim3_cw.block<3, 3>(0, 0);
    const auto s_cw = std::sqrt(s_rot_cw.block<1, 3>(0, 0).dot(s_rot_cw.block<1, 3>(0, 0)));
    const Mat33_t rot_cw = s_rot_cw / s_cw;
    const Vec3_t trans_cw = Sim3_cw.block<3, 1>(0, 3) / s_cw;
    const Vec3_t cam_center = -rot_cw.transpose() * trans_cw;

    std::set<std::shared_ptr<data::landmark>> already_matched(matched_lms_in_keyfrm.begin(), matched_lms_in_keyfrm.end());
    already_matched.erase(nullptr);

    for (const auto& lm : landmarks) {
        if (lm->will_be_erased()) {
            continue;
        }
        if (already_matched.count(lm)) {
            continue;
        }

        // 3D point coordinates with the global reference
        const Vec3_t pos_w = lm->get_pos_in_world();

        // Reproject and compute visibility
        Vec2_t reproj;
        float x_right;
        const bool in_image = keyfrm->camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

        // Ignore if it is reprojected outside the image
        if (!in_image) {
            continue;
        }

        // Check if it's within ORB scale levels
        const Vec3_t cam_to_lm_vec = pos_w - cam_center;
        const auto cam_to_lm_dist = cam_to_lm_vec.norm();
        constexpr auto margin_far = 1.3;
        constexpr auto margin_near = 1.0 / margin_far;
        const auto max_cam_to_lm_dist = margin_far * lm->get_max_valid_distance();
        const auto min_cam_to_lm_dist = margin_near * lm->get_min_valid_distance();

        if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
            continue;
        }

        // Compute the angle formed by the average observation vector of the 3D points,
        // and discard it if it is wider than the threshold value (60 degrees)
        const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();

        if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist) {
            continue;
        }

        // Acquire keypoints in the cell where the reprojected 3D points exist
        const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm->orb_params_->num_levels_, keyfrm->orb_params_->log_scale_factor_);
        const int min_level = std::max(0, static_cast<int>(pred_scale_level) - 1);
        const int max_level = std::min(keyfrm->orb_params_->num_levels_ - 1, pred_scale_level + 1);
        const auto indices = keyfrm->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm->orb_params_->scale_factors_.at(pred_scale_level), min_level, max_level);

        if (indices.empty()) {
            continue;
        }

        // Find keypoints with the closest descriptor
        const auto lm_desc = lm->get_descriptor();

        unsigned int best_dist = MAX_HAMMING_DIST;
        int best_idx = -1;

        for (const auto idx : indices) {
            if (matched_lms_in_keyfrm.at(idx)) {
                continue;
            }

            const auto& desc = keyfrm->frm_obs_.descriptors_.row(idx);

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_dist) {
                best_dist = hamm_dist;
                best_idx = idx;
            }
        }

        if (HAMMING_DIST_THR_LOW < best_dist) {
            continue;
        }

        matched_lms_in_keyfrm.at(best_idx) = lm;
        ++num_matches;
    }

    return num_matches;
}

unsigned int projection::match_keyframes_mutually(const std::shared_ptr<data::keyframe>& keyfrm_1, const std::shared_ptr<data::keyframe>& keyfrm_2, std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_keyfrm_1,
                                                  const float& s_12, const Mat33_t& rot_12, const Vec3_t& trans_12, const float margin) const {
    // The pose of keyframe 1
    const Mat33_t rot_1w = keyfrm_1->get_rot_cw();
    const Vec3_t trans_1w = keyfrm_1->get_trans_cw();

    // The pose of keyframe 2
    const Mat33_t rot_2w = keyfrm_2->get_rot_cw();
    const Vec3_t trans_2w = keyfrm_2->get_trans_cw();

    // Compute the similarity transformation between the keyframes 1 and 2
    const Mat33_t s_rot_12 = s_12 * rot_12;
    const Mat33_t s_rot_21 = (1.0 / s_12) * rot_12.transpose();
    const Vec3_t trans_21 = -s_rot_21 * trans_12;

    const auto landmarks_1 = keyfrm_1->get_landmarks();
    const auto landmarks_2 = keyfrm_2->get_landmarks();

    // Contain matching information if there are already matches between the keyframes 1 and 2
    std::vector<bool> is_already_matched_in_keyfrm_1(landmarks_1.size(), false);
    std::vector<bool> is_already_matched_in_keyfrm_2(landmarks_2.size(), false);

    for (unsigned int idx_1 = 0; idx_1 < landmarks_1.size(); ++idx_1) {
        auto& lm = matched_lms_in_keyfrm_1.at(idx_1);
        if (!lm) {
            continue;
        }
        const auto idx_2 = lm->get_index_in_keyframe(keyfrm_2);
        if (0 <= idx_2 && idx_2 < static_cast<int>(landmarks_2.size())) {
            is_already_matched_in_keyfrm_1.at(idx_1) = true;
            is_already_matched_in_keyfrm_2.at(idx_2) = true;
        }
    }

    std::vector<int> matched_indices_2_in_keyfrm_1(landmarks_1.size(), -1);
    std::vector<int> matched_indices_1_in_keyfrm_2(landmarks_2.size(), -1);

    // Compute the similarity transformation from the 3D points observed in keyframe 1 to keyframe 2 coordinates,
    // then project the result, and search keypoint matches
    // (world origin -- SE3 -> keyframe 1 -- Sim3 --> keyframe 2)
    // s_rot_21 * (rot_1w * pos_w + trans_1w) + trans_21
    // = s_rot_21 * rot_1w * pos_w + s_rot_21 * trans_1w + trans_21
    {
        const Mat33_t s_rot_21w = s_rot_21 * rot_1w;
        const Vec3_t trans_21w = s_rot_21 * trans_1w + trans_21;
        for (unsigned int idx_1 = 0; idx_1 < landmarks_1.size(); ++idx_1) {
            auto& lm = landmarks_1.at(idx_1);
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            if (is_already_matched_in_keyfrm_1.at(idx_1)) {
                continue;
            }

            // 3D point coordinates with the global reference
            const Vec3_t pos_w = lm->get_pos_in_world();
            const Vec3_t pos_2 = s_rot_21w * pos_w + trans_21w;

            // Reproject and compute visibility
            Vec2_t reproj;
            float x_right;
            const bool in_image = keyfrm_2->camera_->reproject_to_image(s_rot_21w, trans_21w, pos_w, reproj, x_right);

            // Ignore if it is reprojected outside the image
            if (!in_image) {
                continue;
            }

            // Check if it's within ORB scale levels
            const auto cam_to_lm_dist = pos_2.norm();
            constexpr auto margin_far = 1.3;
            constexpr auto margin_near = 1.0 / margin_far;
            const auto max_cam_to_lm_dist = margin_far * lm->get_max_valid_distance();
            const auto min_cam_to_lm_dist = margin_near * lm->get_min_valid_distance();

            if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
                continue;
            }

            // Acquire keypoints in the cell where the reprojected 3D points exist
            const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm_2->orb_params_->num_levels_, keyfrm_2->orb_params_->log_scale_factor_);
            const int min_level = std::max(0, static_cast<int>(pred_scale_level) - 1);
            const int max_level = std::min(keyfrm_2->orb_params_->num_levels_ - 1, pred_scale_level + 1);
            const auto indices = keyfrm_2->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm_2->orb_params_->scale_factors_.at(pred_scale_level), min_level, max_level);

            if (indices.empty()) {
                continue;
            }

            // Find a keypoint with the closest descriptor
            const auto lm_desc = lm->get_descriptor();

            unsigned int best_hamm_dist = MAX_HAMMING_DIST;
            int best_idx_2 = -1;

            for (const auto idx_2 : indices) {
                const auto& desc = keyfrm_2->frm_obs_.descriptors_.row(idx_2);

                const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

                if (hamm_dist < best_hamm_dist) {
                    best_hamm_dist = hamm_dist;
                    best_idx_2 = idx_2;
                }
            }

            if (best_hamm_dist <= HAMMING_DIST_THR_HIGH) {
                matched_indices_2_in_keyfrm_1.at(idx_1) = best_idx_2;
            }
        }
    }

    // Compute the similarity transformation from the 3D points observed in the current keyframe (keyframe 1) to the candidate keyframe (keyframe 2) coordinates, then project the result
    // earch keypoint matches
    // (world origin -- SE3 -> keyframe2 -- Sim3 --> keyframe1)
    // s_rot_12 * (rot_2w * pos_w + trans_2w) + trans_12
    // = s_rot_12 * rot_2w * pos_w + s_rot_12 * trans_2w + trans_12
    {
        const Mat33_t s_rot_12w = s_rot_12 * rot_2w;
        const Vec3_t trans_12w = s_rot_12 * trans_2w + trans_12;
        for (unsigned int idx_2 = 0; idx_2 < landmarks_2.size(); ++idx_2) {
            auto& lm = landmarks_2.at(idx_2);
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            if (is_already_matched_in_keyfrm_2.at(idx_2)) {
                continue;
            }

            // 3D point coordinates with the global reference
            const Vec3_t pos_w = lm->get_pos_in_world();
            const Vec3_t pos_1 = s_rot_12w * pos_w + trans_12w;

            // Reproject and compute visibility
            Vec2_t reproj;
            float x_right;
            const bool in_image = keyfrm_2->camera_->reproject_to_image(s_rot_12w, trans_12w, pos_w, reproj, x_right);

            // Ignore if it is reprojected outside the image
            if (!in_image) {
                continue;
            }

            // Check if it's within ORB scale levels
            const auto cam_to_lm_dist = pos_1.norm();
            constexpr auto margin_far = 1.3;
            constexpr auto margin_near = 1.0 / margin_far;
            const auto max_cam_to_lm_dist = margin_far * lm->get_max_valid_distance();
            const auto min_cam_to_lm_dist = margin_near * lm->get_min_valid_distance();

            if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
                continue;
            }

            // Acquire keypoints in the cell where the reprojected 3D points exist
            const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm_1->orb_params_->num_levels_, keyfrm_1->orb_params_->log_scale_factor_);
            const int min_level = std::max(0, static_cast<int>(pred_scale_level) - 1);
            const int max_level = std::min(keyfrm_1->orb_params_->num_levels_ - 1, pred_scale_level + 1);
            const auto indices = keyfrm_1->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm_1->orb_params_->scale_factors_.at(pred_scale_level), min_level, max_level);

            if (indices.empty()) {
                continue;
            }

            // Find a keypoint with the closest descriptor
            const auto lm_desc = lm->get_descriptor();

            unsigned int best_hamm_dist = MAX_HAMMING_DIST;
            int best_idx_1 = -1;

            for (const auto idx_1 : indices) {
                const auto& desc = keyfrm_1->frm_obs_.descriptors_.row(idx_1);

                const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

                if (hamm_dist < best_hamm_dist) {
                    best_hamm_dist = hamm_dist;
                    best_idx_1 = idx_1;
                }
            }

            if (best_hamm_dist <= HAMMING_DIST_THR_HIGH) {
                matched_indices_1_in_keyfrm_2.at(idx_2) = best_idx_1;
            }
        }
    }

    // Record only the cross-matches
    unsigned int num_matches = 0;
    for (unsigned int i = 0; i < landmarks_1.size(); ++i) {
        const auto idx_2 = matched_indices_2_in_keyfrm_1.at(i);
        if (idx_2 < 0) {
            continue;
        }

        const auto idx_1 = matched_indices_1_in_keyfrm_2.at(idx_2);
        if (idx_1 == static_cast<int>(i)) {
            matched_lms_in_keyfrm_1.at(idx_1) = landmarks_2.at(idx_2);
            ++num_matches;
        }
    }

    return num_matches;
}

} // namespace match
} // namespace cv::slam
