#include "data/frame.hpp"
#include "match/area.hpp"
#include "util/angle.hpp"

namespace cv::slam {
namespace match {

unsigned int area::match_in_consistent_area(data::frame& frm_1, data::frame& frm_2, std::vector<cv::Point2f>& prev_matched_pts,
                                            std::vector<int>& matched_indices_2_in_frm_1, int margin) {
    unsigned int num_matches = 0;

    matched_indices_2_in_frm_1 = std::vector<int>(frm_1.frm_obs_.undist_keypts_.size(), -1);

    std::vector<unsigned int> matched_dists_in_frm_2(frm_2.frm_obs_.undist_keypts_.size(), MAX_HAMMING_DIST);
    std::vector<int> matched_indices_1_in_frm_2(frm_2.frm_obs_.undist_keypts_.size(), -1);

    for (unsigned int idx_1 = 0; idx_1 < frm_1.frm_obs_.undist_keypts_.size(); ++idx_1) {
        const auto& undist_keypt_1 = frm_1.frm_obs_.undist_keypts_.at(idx_1);
        const auto scale_level_1 = undist_keypt_1.octave;

        // Use only keypoints with the 0-th scale
        if (0 < scale_level_1) {
            continue;
        }

        // Get keypoints in the cells neighboring to the previous match
        const auto indices = frm_2.get_keypoints_in_cell(prev_matched_pts.at(idx_1).x, prev_matched_pts.at(idx_1).y,
                                                         margin, scale_level_1, scale_level_1);
        if (indices.empty()) {
            continue;
        }

        const auto& desc_1 = frm_1.frm_obs_.descriptors_.row(idx_1);

        unsigned int best_hamm_dist = MAX_HAMMING_DIST;
        unsigned int second_best_hamm_dist = MAX_HAMMING_DIST;
        int best_idx_2 = -1;

        for (const auto idx_2 : indices) {
            if (check_orientation_ && std::abs(util::angle::diff(frm_1.frm_obs_.undist_keypts_.at(idx_1).angle, frm_2.frm_obs_.undist_keypts_.at(idx_2).angle)) > 30.0) {
                continue;
            }

            const auto& desc_2 = frm_2.frm_obs_.descriptors_.row(idx_2);

            const auto hamm_dist = compute_descriptor_distance_32(desc_1, desc_2);

            // Ignore if the already-matched point is closer in Hamming space
            if (matched_dists_in_frm_2.at(idx_2) <= hamm_dist) {
                continue;
            }

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
        if (second_best_hamm_dist * lowe_ratio_ < static_cast<float>(best_hamm_dist)) {
            continue;
        }

        // Assuming that the indices 1 and 2 are the best match

        // If a match associated to the best index 2 exists, to overwrite the matching information of the previous index 1 (= prev_idx_1),
        // 'matched_indices_2_in_frm_1.at(prev_idx_1)' must be deleted to overrwrite the updates
        // ('matched_indices_1_in_frm_2.at (best_idx_2)' will be overwritten, so there is no need to delete it)
        const auto prev_idx_1 = matched_indices_1_in_frm_2.at(best_idx_2);
        if (0 <= prev_idx_1) {
            matched_indices_2_in_frm_1.at(prev_idx_1) = -1;
            --num_matches;
        }

        // Record the mutual matching information
        matched_indices_2_in_frm_1.at(idx_1) = best_idx_2;
        matched_indices_1_in_frm_2.at(best_idx_2) = idx_1;
        matched_dists_in_frm_2.at(best_idx_2) = best_hamm_dist;
        ++num_matches;
    }

    // Update the previous matches
    for (unsigned int idx_1 = 0; idx_1 < matched_indices_2_in_frm_1.size(); ++idx_1) {
        if (0 <= matched_indices_2_in_frm_1.at(idx_1)) {
            prev_matched_pts.at(idx_1) = frm_2.frm_obs_.undist_keypts_.at(matched_indices_2_in_frm_1.at(idx_1)).pt;
        }
    }

    return num_matches;
}

} // namespace match
} // namespace cv::slam
