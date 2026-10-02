#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "module/local_map_updater.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace module {

local_map_updater::local_map_updater(const unsigned int max_num_local_keyfrms)
    : max_num_local_keyfrms_(max_num_local_keyfrms) {}

std::vector<std::shared_ptr<data::keyframe>> local_map_updater::get_local_keyframes() const {
    return local_keyfrms_;
}

std::vector<std::shared_ptr<data::landmark>> local_map_updater::get_local_landmarks() const {
    return local_lms_;
}

std::shared_ptr<data::keyframe> local_map_updater::get_nearest_covisibility() const {
    return nearest_covisibility_;
}

bool local_map_updater::acquire_local_map(const std::vector<std::shared_ptr<data::landmark>>& frm_lms) {
    constexpr unsigned int keyframe_id_threshold = 0;
    unsigned int num_temporal_keyfrms = 0;
    const auto local_keyfrms_was_found = find_local_keyframes(frm_lms, keyframe_id_threshold, num_temporal_keyfrms);
    const auto local_lms_was_found = find_local_landmarks(frm_lms);
    return local_keyfrms_was_found && local_lms_was_found;
}

bool local_map_updater::acquire_local_map(const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
                                          const unsigned int keyframe_id_threshold,
                                          unsigned int& num_temporal_keyfrms) {
    num_temporal_keyfrms = 0;
    const auto local_keyfrms_was_found = find_local_keyframes(frm_lms, keyframe_id_threshold, num_temporal_keyfrms);
    const auto local_lms_was_found = find_local_landmarks(frm_lms);
    return local_keyfrms_was_found && local_lms_was_found;
}

bool local_map_updater::find_local_keyframes(const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
                                             const unsigned int keyframe_id_threshold,
                                             unsigned int& num_temporal_keyfrms) {
    const auto num_shared_lms_and_keyfrm = count_num_shared_lms(frm_lms, keyframe_id_threshold);
    if (num_shared_lms_and_keyfrm.empty()) {
        CV_LOG_DEBUG(&g_log_tag, "find_local_keyframes: empty");
        return false;
    }

    std::unordered_set<unsigned int> already_found_keyfrm_ids;
    const auto first_local_keyfrms = find_first_local_keyframes(num_shared_lms_and_keyfrm, keyframe_id_threshold, already_found_keyfrm_ids, num_temporal_keyfrms);
    const auto second_local_keyfrms = find_second_local_keyframes(first_local_keyfrms, keyframe_id_threshold, already_found_keyfrm_ids, num_temporal_keyfrms);
    local_keyfrms_ = first_local_keyfrms;
    std::copy(second_local_keyfrms.begin(), second_local_keyfrms.end(), std::back_inserter(local_keyfrms_));
    return true;
}

auto local_map_updater::count_num_shared_lms(
    const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
    const unsigned int keyframe_id_threshold) const
    -> std::vector<std::pair<unsigned int, std::shared_ptr<data::keyframe>>> {
    std::vector<std::pair<unsigned int, std::shared_ptr<data::keyframe>>> num_shared_lms_and_keyfrm;

    // count the number of sharing landmarks between the current frame and each of the neighbor keyframes
    // key: keyframe, value: number of sharing landmarks
    keyframe_to_num_shared_lms_t keyfrm_to_num_shared_lms;
    for (unsigned int idx = 0; idx < frm_lms.size(); ++idx) {
        auto& lm = frm_lms.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        const auto observations = lm->get_observations();
        for (auto obs : observations) {
            auto keyfrm = obs.first.lock();
            ++keyfrm_to_num_shared_lms[keyfrm];
        }
    }
    int num_temporal_keyfrms = 0;
    for (auto& it : keyfrm_to_num_shared_lms) {
        if (keyframe_id_threshold > 0 && it.first->id_ >= keyframe_id_threshold) {
            ++num_temporal_keyfrms;
        }
        num_shared_lms_and_keyfrm.emplace_back(it.second, it.first);
    }
    constexpr int margin = 5; // Keep a little more than max_num_local_keyfrms_, as keyframes may be deleted.
    if (num_shared_lms_and_keyfrm.size() > max_num_local_keyfrms_ + num_temporal_keyfrms + margin) {
        std::partial_sort(num_shared_lms_and_keyfrm.begin(),
                          num_shared_lms_and_keyfrm.begin() + max_num_local_keyfrms_ + num_temporal_keyfrms + margin,
                          num_shared_lms_and_keyfrm.end(),
                          greater_number_and_id_object_pairs<unsigned int, data::keyframe>());
    }
    else {
        std::sort(num_shared_lms_and_keyfrm.begin(),
                  num_shared_lms_and_keyfrm.end(),
                  greater_number_and_id_object_pairs<unsigned int, data::keyframe>());
    }

    return num_shared_lms_and_keyfrm;
}

auto local_map_updater::find_first_local_keyframes(
    const std::vector<std::pair<unsigned int, std::shared_ptr<data::keyframe>>>& num_shared_lms_and_keyfrm,
    const unsigned int keyframe_id_threshold,
    std::unordered_set<unsigned int>& already_found_keyfrm_ids,
    unsigned int& num_temporal_keyfrms)
    -> std::vector<std::shared_ptr<data::keyframe>> {
    std::vector<std::shared_ptr<data::keyframe>> first_local_keyfrms;
    first_local_keyfrms.reserve(std::min(static_cast<size_t>(max_num_local_keyfrms_), 2 * num_shared_lms_and_keyfrm.size()));

    unsigned int max_num_shared_lms = 0;
    for (auto& keyfrm_and_num_shared_lms : num_shared_lms_and_keyfrm) {
        const auto num_shared_lms = keyfrm_and_num_shared_lms.first;
        const auto& keyfrm = keyfrm_and_num_shared_lms.second;

        if (keyfrm->will_be_erased()) {
            continue;
        }

        first_local_keyfrms.push_back(keyfrm);
        if (keyframe_id_threshold > 0 && keyfrm->id_ >= keyframe_id_threshold) {
            ++num_temporal_keyfrms;
        }

        // avoid duplication
        already_found_keyfrm_ids.insert(keyfrm->id_);

        // update the nearest keyframe
        if (max_num_shared_lms < num_shared_lms) {
            max_num_shared_lms = num_shared_lms;
            nearest_covisibility_ = keyfrm;
        }

        if (max_num_local_keyfrms_ <= first_local_keyfrms.size() + num_temporal_keyfrms) {
            break;
        }
    }

    return first_local_keyfrms;
}

auto local_map_updater::find_second_local_keyframes(const std::vector<std::shared_ptr<data::keyframe>>& first_local_keyframes,
                                                    const unsigned int keyframe_id_threshold,
                                                    std::unordered_set<unsigned int>& already_found_keyfrm_ids,
                                                    unsigned int& num_temporal_keyfrms) const
    -> std::vector<std::shared_ptr<data::keyframe>> {
    std::vector<std::shared_ptr<data::keyframe>> second_local_keyfrms;
    second_local_keyfrms.reserve(4 * first_local_keyframes.size());

    // add the second-order keyframes to the local landmarks
    auto add_second_local_keyframe = [this, &second_local_keyfrms, &already_found_keyfrm_ids,
                                      &num_temporal_keyfrms, keyframe_id_threshold](const std::shared_ptr<data::keyframe>& keyfrm) {
        if (!keyfrm) {
            return false;
        }
        if (keyfrm->will_be_erased()) {
            return false;
        }
        if (keyframe_id_threshold > 0 && keyfrm->id_ >= keyframe_id_threshold) {
            ++num_temporal_keyfrms;
        }
        // avoid duplication
        if (already_found_keyfrm_ids.count(keyfrm->id_)) {
            return false;
        }
        already_found_keyfrm_ids.insert(keyfrm->id_);
        second_local_keyfrms.push_back(keyfrm);
        return true;
    };
    for (auto iter = first_local_keyframes.cbegin(); iter != first_local_keyframes.cend(); ++iter) {
        if (max_num_local_keyfrms_ <= first_local_keyframes.size() + second_local_keyfrms.size() + num_temporal_keyfrms) {
            break;
        }

        const auto& keyfrm = *iter;

        // covisibilities of the neighbor keyframe
        const auto neighbors = keyfrm->graph_node_->get_top_n_covisibilities(10);
        for (const auto& neighbor : neighbors) {
            add_second_local_keyframe(neighbor);
            if (max_num_local_keyfrms_ <= first_local_keyframes.size() + second_local_keyfrms.size() + num_temporal_keyfrms) {
                return second_local_keyfrms;
            }
        }

        // children of the spanning tree
        const auto spanning_children = keyfrm->graph_node_->get_spanning_children();
        for (const auto& child : spanning_children) {
            add_second_local_keyframe(child);
            if (max_num_local_keyfrms_ <= first_local_keyframes.size() + second_local_keyfrms.size() + num_temporal_keyfrms) {
                return second_local_keyfrms;
            }
        }

        // parent of the spanning tree
        const auto& parent = keyfrm->graph_node_->get_spanning_parent();
        add_second_local_keyframe(parent);
    }

    return second_local_keyfrms;
}

bool local_map_updater::find_local_landmarks(const std::vector<std::shared_ptr<data::landmark>>& frm_lms) {
    local_lms_.clear();
    local_lms_.reserve(50 * local_keyfrms_.size());

    std::unordered_set<unsigned int> already_found_lms_ids;
    already_found_lms_ids.reserve(frm_lms.size());
    for (unsigned int idx = 0; idx < frm_lms.size(); ++idx) {
        auto& lm = frm_lms.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        already_found_lms_ids.insert(lm->id_);
    }
    for (const auto& keyfrm : local_keyfrms_) {
        // ZoneScopedN("find_local_landmarks_per_keyfrm");
        const auto& lms = keyfrm->get_landmarks();

        for (const auto& lm : lms) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // avoid duplication
            if (already_found_lms_ids.count(lm->id_)) {
                continue;
            }
            already_found_lms_ids.insert(lm->id_);
            local_lms_.push_back(lm);
        }
    }

    return true;
}

} // namespace module
} // namespace cv::slam
