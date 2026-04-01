#include "data/bow_vocabulary.hpp"
#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "match/bow_tree.hpp"
#include "util/angle.hpp"

namespace cv::slam {
namespace match {

unsigned int bow_tree::match_for_triangulation(const std::shared_ptr<data::keyframe>& keyfrm_1,
                                               const std::shared_ptr<data::keyframe>& keyfrm_2,
                                               const Mat33_t& E_12,
                                               std::vector<std::pair<unsigned int, unsigned int>>& matched_idx_pairs,
                                               const float residual_rad_thr) const {
    unsigned int num_matches = 0;

    // Project the center of keyframe 1 to keyframe 2
    // to acquire the epipole coordinates of the candidate keyframe
    const Vec3_t cam_center_1 = keyfrm_1->get_trans_wc();
    const Mat33_t rot_2w = keyfrm_2->get_rot_cw();
    const Vec3_t trans_2w = keyfrm_2->get_trans_cw();
    Vec3_t epiplane_in_keyfrm_2;
    const bool valid_epiplane = keyfrm_2->camera_->reproject_to_bearing(rot_2w, trans_2w, cam_center_1, epiplane_in_keyfrm_2);

    // Acquire the 3D point information of the keframes
    const auto assoc_lms_in_keyfrm_1 = keyfrm_1->get_landmarks();
    const auto assoc_lms_in_keyfrm_2 = keyfrm_2->get_landmarks();

    // Save the matching information
    // Discard the already matched keypoints in keyframe 2
    // to acquire a unique association to each keypoint in keyframe 1
    std::vector<bool> is_already_matched_in_keyfrm_2(keyfrm_2->frm_obs_.undist_keypts_.size(), false);
    // Save the keypoint idx in keyframe 2 which is already associated to the keypoint idx in keyframe 1
    std::vector<int> matched_indices_2_in_keyfrm_1(keyfrm_1->frm_obs_.undist_keypts_.size(), -1);

    data::bow_feature_vector::const_iterator itr_1 = keyfrm_1->bow_feat_vec_.begin();
    data::bow_feature_vector::const_iterator itr_2 = keyfrm_2->bow_feat_vec_.begin();
    const data::bow_feature_vector::const_iterator itr_1_end = keyfrm_1->bow_feat_vec_.end();
    const data::bow_feature_vector::const_iterator itr_2_end = keyfrm_2->bow_feat_vec_.end();

    while (itr_1 != itr_1_end && itr_2 != itr_2_end) {
        // Check if the node numbers of BoW tree match
        if (itr_1->first == itr_2->first) {
            // If the node numbers of BoW tree match,
            // Check in practice if matches exist between keyframes
            const auto& keyfrm_1_indices = itr_1->second;
            const auto& keyfrm_2_indices = itr_2->second;

            for (const auto idx_1 : keyfrm_1_indices) {
                const auto& lm_1 = assoc_lms_in_keyfrm_1.at(idx_1);
                // Ignore if the keypoint of keyframe is associated any 3D points
                if (lm_1) {
                    continue;
                }

                // Check if it's a stereo keypoint or not
                const bool is_stereo_keypt_1 = !keyfrm_1->frm_obs_.stereo_x_right_.empty() && 0 <= keyfrm_1->frm_obs_.stereo_x_right_.at(idx_1);

                // Acquire the keypoints and ORB feature vectors
                const auto& keypt_1 = keyfrm_1->frm_obs_.undist_keypts_.at(idx_1);
                const Vec3_t& bearing_1 = keyfrm_1->frm_obs_.bearings_.at(idx_1);
                const auto& desc_1 = keyfrm_1->frm_obs_.descriptors_.row(idx_1);

                // Find a keypoint in keyframe 2 that has the minimum hamming distance
                unsigned int best_hamm_dist = HAMMING_DIST_THR_LOW;
                int best_idx_2 = -1;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;

                for (const auto idx_2 : keyfrm_2_indices) {
                    // Ignore if the keypoint is associated any 3D points
                    // (because this function is used for triangulation)
                    const auto& lm_2 = assoc_lms_in_keyfrm_2.at(idx_2);
                    if (lm_2) {
                        continue;
                    }

                    // Ignore if matches are already aquired
                    if (is_already_matched_in_keyfrm_2.at(idx_2)) {
                        continue;
                    }

                    if (check_orientation_ && std::abs(util::angle::diff(keypt_1.angle, keyfrm_2->frm_obs_.undist_keypts_.at(idx_2).angle)) > 30.0) {
                        continue;
                    }

                    // Check if it's a stereo keypoint or not
                    const bool is_stereo_keypt_2 = !keyfrm_2->frm_obs_.stereo_x_right_.empty() && 0 <= keyfrm_2->frm_obs_.stereo_x_right_.at(idx_2);

                    // Acquire the keypoints and ORB feature vectors
                    const Vec3_t& bearing_2 = keyfrm_2->frm_obs_.bearings_.at(idx_2);
                    const auto& desc_2 = keyfrm_2->frm_obs_.descriptors_.row(idx_2);

                    // Compute the distance
                    const auto hamm_dist = compute_descriptor_distance_32(desc_1, desc_2);

                    if (HAMMING_DIST_THR_LOW < hamm_dist || best_hamm_dist < hamm_dist) {
                        continue;
                    }

                    if (valid_epiplane && !is_stereo_keypt_1 && !is_stereo_keypt_2) {
                        // Do not use any keypoints near the epipole if both are not stereo keypoints
                        const auto cos_dist = epiplane_in_keyfrm_2.dot(bearing_2);
                        // The threshold of the minimum angle formed by the epipole and the bearing vector is 3.0 degree
                        constexpr double cos_dist_thr = 0.99862953475;

                        // Do not allow to match if the formed angle is narrower that the threshold value
                        if (cos_dist_thr < cos_dist) {
                            continue;
                        }
                    }

                    // Check consistency in Matrix E
                    const bool is_inlier = check_epipolar_constraint(bearing_1, bearing_2, E_12,
                                                                     keyfrm_1->orb_params_->scale_factors_.at(keypt_1.octave),
                                                                     residual_rad_thr);
                    if (is_inlier) {
                        if (hamm_dist < best_hamm_dist) {
                            second_best_hamm_dist = best_hamm_dist;
                            best_hamm_dist = hamm_dist;
                            best_idx_2 = idx_2;
                        }
                        else if (hamm_dist < second_best_hamm_dist) {
                            second_best_hamm_dist = hamm_dist;
                        }
                    }
                }

                if (best_idx_2 < 0) {
                    continue;
                }

                // Ratio test
                if (lowe_ratio_ * second_best_hamm_dist < static_cast<float>(best_hamm_dist)) {
                    continue;
                }

                is_already_matched_in_keyfrm_2.at(best_idx_2) = true;
                matched_indices_2_in_keyfrm_1.at(idx_1) = best_idx_2;
                ++num_matches;
            }

            ++itr_1;
            ++itr_2;
        }
        else if (itr_1->first < itr_2->first) {
            // Since the node number of keyframe 1 is smaller, increment the iterator until the node numbers match
            itr_1 = keyfrm_1->bow_feat_vec_.lower_bound(itr_2->first);
        }
        else {
            // Since the node number of keyframe 2 is smaller, increment the iterator until the node numbers match
            itr_2 = keyfrm_2->bow_feat_vec_.lower_bound(itr_1->first);
        }
    }

    matched_idx_pairs.clear();
    matched_idx_pairs.reserve(num_matches);

    for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_keyfrm_1.size(); ++idx_1) {
        if (matched_indices_2_in_keyfrm_1.at(idx_1) < 0) {
            continue;
        }
        matched_idx_pairs.emplace_back(std::make_pair(idx_1, matched_indices_2_in_keyfrm_1.at(idx_1)));
    }

    return num_matches;
}

unsigned int bow_tree::match_frame_and_keyframe(const std::shared_ptr<data::keyframe>& keyfrm, data::frame& frm, std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm) const {
    unsigned int num_matches = 0;

    matched_lms_in_frm = std::vector<std::shared_ptr<data::landmark>>(frm.frm_obs_.undist_keypts_.size(), nullptr);

    const auto keyfrm_lms = keyfrm->get_landmarks();

    data::bow_feature_vector::const_iterator keyfrm_itr = keyfrm->bow_feat_vec_.begin();
    data::bow_feature_vector::const_iterator frm_itr = frm.bow_feat_vec_.begin();
    const data::bow_feature_vector::const_iterator kryfrm_end = keyfrm->bow_feat_vec_.end();
    const data::bow_feature_vector::const_iterator frm_end = frm.bow_feat_vec_.end();

    while (keyfrm_itr != kryfrm_end && frm_itr != frm_end) {
        // Check if the node numbers of BoW tree match
        if (keyfrm_itr->first == frm_itr->first) {
            // If the node numbers of BoW tree match,
            // Check in practice if matches exist between the frame and keyframe
            const auto& keyfrm_indices = keyfrm_itr->second;
            const auto& frm_indices = frm_itr->second;

            for (const auto keyfrm_idx : keyfrm_indices) {
                // Ignore if the keypoint of keyframe is not associated any 3D points
                auto& lm = keyfrm_lms.at(keyfrm_idx);
                if (!lm) {
                    continue;
                }
                if (lm->will_be_erased()) {
                    continue;
                }

                const auto& keyfrm_desc = keyfrm->frm_obs_.descriptors_.row(keyfrm_idx);

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_frm_idx = -1;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;

                for (const auto frm_idx : frm_indices) {
                    if (matched_lms_in_frm.at(frm_idx)) {
                        continue;
                    }

                    if (check_orientation_ && std::abs(util::angle::diff(keyfrm->frm_obs_.undist_keypts_.at(keyfrm_idx).angle, frm.frm_obs_.undist_keypts_.at(frm_idx).angle)) > 30.0) {
                        continue;
                    }

                    const auto& frm_desc = frm.frm_obs_.descriptors_.row(frm_idx);

                    const auto hamm_dist = compute_descriptor_distance_32(keyfrm_desc, frm_desc);

                    if (hamm_dist < best_hamm_dist) {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = hamm_dist;
                        best_frm_idx = frm_idx;
                    }
                    else if (hamm_dist < second_best_hamm_dist) {
                        second_best_hamm_dist = hamm_dist;
                    }
                }

                if (HAMMING_DIST_THR_LOW < best_hamm_dist) {
                    continue;
                }

                // Ratio test
                if (lowe_ratio_ * second_best_hamm_dist < static_cast<float>(best_hamm_dist)) {
                    continue;
                }

                matched_lms_in_frm.at(best_frm_idx) = lm;

                ++num_matches;
            }

            ++keyfrm_itr;
            ++frm_itr;
        }
        else if (keyfrm_itr->first < frm_itr->first) {
            // Since the node number of the keyframe is smaller, increment the iterator until the node numbers match
            keyfrm_itr = keyfrm->bow_feat_vec_.lower_bound(frm_itr->first);
        }
        else {
            // Since the node number of the frame is smaller, increment the iterator until the node numbers match
            frm_itr = frm.bow_feat_vec_.lower_bound(keyfrm_itr->first);
        }
    }

    return num_matches;
}

unsigned int bow_tree::match_keyframes(const std::shared_ptr<data::keyframe>& keyfrm_1, const std::shared_ptr<data::keyframe>& keyfrm_2, std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_keyfrm_1) const {
    unsigned int num_matches = 0;

    const auto keyfrm_1_lms = keyfrm_1->get_landmarks();
    const auto keyfrm_2_lms = keyfrm_2->get_landmarks();

    matched_lms_in_keyfrm_1 = std::vector<std::shared_ptr<data::landmark>>(keyfrm_1_lms.size(), nullptr);

    // Set 'true' if a keypoint in keyframe 2 is associated to the keypoint in keyframe 1
    // NOTE: the size matches the number of the keypoints in keyframe 2
    std::vector<bool> is_already_matched_in_keyfrm_2(keyfrm_2_lms.size(), false);

    data::bow_feature_vector::const_iterator itr_1 = keyfrm_1->bow_feat_vec_.begin();
    data::bow_feature_vector::const_iterator itr_2 = keyfrm_2->bow_feat_vec_.begin();
    const data::bow_feature_vector::const_iterator itr_1_end = keyfrm_1->bow_feat_vec_.end();
    const data::bow_feature_vector::const_iterator itr_2_end = keyfrm_2->bow_feat_vec_.end();

    while (itr_1 != itr_1_end && itr_2 != itr_2_end) {
        // Check if the node numbers of BoW tree match
        if (itr_1->first == itr_2->first) {
            // If the node numbers of BoW tree match,
            // Check in practice if matches exist between keyframes
            const auto& keyfrm_1_indices = itr_1->second;
            const auto& keyfrm_2_indices = itr_2->second;

            for (const auto idx_1 : keyfrm_1_indices) {
                // Ignore if the keypoint is not associated any 3D points
                // (because this function is used for Sim3 estimation)
                auto& lm_1 = keyfrm_1_lms.at(idx_1);
                if (!lm_1) {
                    continue;
                }
                if (lm_1->will_be_erased()) {
                    continue;
                }

                const auto& desc_1 = keyfrm_1->frm_obs_.descriptors_.row(idx_1);

                unsigned int best_hamm_dist = MAX_HAMMING_DIST;
                int best_idx_2 = -1;
                unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;

                for (const auto idx_2 : keyfrm_2_indices) {
                    // Ignore if the keypoint is not associated any 3D points
                    // (because this function is used for Sim3 estimation)
                    auto& lm_2 = keyfrm_2_lms.at(idx_2);
                    if (!lm_2) {
                        continue;
                    }
                    if (lm_2->will_be_erased()) {
                        continue;
                    }

                    if (is_already_matched_in_keyfrm_2.at(idx_2)) {
                        continue;
                    }

                    if (check_orientation_ && std::abs(util::angle::diff(keyfrm_1->frm_obs_.undist_keypts_.at(idx_1).angle, keyfrm_2->frm_obs_.undist_keypts_.at(idx_2).angle)) > 30.0) {
                        continue;
                    }

                    const auto& desc_2 = keyfrm_2->frm_obs_.descriptors_.row(idx_2);

                    const auto hamm_dist = compute_descriptor_distance_32(desc_1, desc_2);

                    if (hamm_dist < best_hamm_dist) {
                        second_best_hamm_dist = best_hamm_dist;
                        best_hamm_dist = hamm_dist;
                        best_idx_2 = idx_2;
                    }
                    else if (hamm_dist < second_best_hamm_dist) {
                        second_best_hamm_dist = hamm_dist;
                    }
                }

                if (HAMMING_DIST_THR_LOW < best_hamm_dist) {
                    continue;
                }

                // Ratio test
                if (lowe_ratio_ * second_best_hamm_dist < static_cast<float>(best_hamm_dist)) {
                    continue;
                }

                // Record the matching information
                // The index of keyframe 1 matches the best index 2 of keyframe 2
                matched_lms_in_keyfrm_1.at(idx_1) = keyfrm_2_lms.at(best_idx_2);
                // The best index of keyframe 2 already matches the keypoint of keyframe 1
                is_already_matched_in_keyfrm_2.at(best_idx_2) = true;

                num_matches++;
            }

            ++itr_1;
            ++itr_2;
        }
        else if (itr_1->first < itr_2->first) {
            // Since the node number of keyframe 1 is smaller, increment the iterator until the node numbers match
            itr_1 = keyfrm_1->bow_feat_vec_.lower_bound(itr_2->first);
        }
        else {
            // Since the node number of keyframe 2 is smaller, increment the iterator until the node numbers match
            itr_2 = keyfrm_2->bow_feat_vec_.lower_bound(itr_1->first);
        }
    }

    return num_matches;
}

} // namespace match
} // namespace cv::slam
