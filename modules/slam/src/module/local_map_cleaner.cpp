#include "data/keyframe.hpp"
#include "util/yaml.hpp"
#include "data/landmark.hpp"
#include "data/map_database.hpp"
#include "module/local_map_cleaner.hpp"

namespace cv::slam {
namespace module {

local_map_cleaner::local_map_cleaner(const cv::FileNode& yaml_node, data::map_database* map_db, data::bow_database* bow_db)
    : map_db_(map_db), bow_db_(bow_db),
      redundant_obs_ratio_thr_(util::yaml_get_val<double>(yaml_node, "redundant_obs_ratio_thr", 0.9)),
      observed_ratio_thr_(util::yaml_get_val<double>(yaml_node, "observed_ratio_thr", 0.3)),
      num_reliable_keyfrms_(util::yaml_get_val<unsigned int>(yaml_node, "num_reliable_keyfrms", 2)),
      top_n_covisibilities_to_search_(util::yaml_get_val<unsigned int>(yaml_node, "top_n_covisibilities_to_search", 30)) {}

void local_map_cleaner::reset() {
    fresh_landmarks_.clear();
}

unsigned int local_map_cleaner::remove_invalid_landmarks(const unsigned int cur_keyfrm_id) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
    // states of observed landmarks
    enum class lm_state_t { Valid,
                            Invalid,
                            NotClear };

    unsigned int num_removed = 0;
    auto iter = fresh_landmarks_.begin();
    while (iter != fresh_landmarks_.end()) {
        const auto& lm = *iter;

        // decide the state of lms the buffer
        auto lm_state = lm_state_t::NotClear;
        if (lm->will_be_erased()) {
            // in case `lm` will be erased
            // remove `lm` from the buffer
            lm_state = lm_state_t::Valid;
        }
        else if (lm->get_observed_ratio() < observed_ratio_thr_) {
            // if `lm` is not reliable
            // remove `lm` from the buffer and the database
            lm_state = lm_state_t::Invalid;
        }
        else if (num_reliable_keyfrms_ + lm->first_keyfrm_id_ < cur_keyfrm_id) {
            // if the number of the observers of `lm` is sufficient after some keyframes were inserted
            // remove `lm` from the buffer
            lm_state = lm_state_t::Valid;
        }

        // select to remove `lm` according to the state
        if (lm_state == lm_state_t::Valid) {
            iter = fresh_landmarks_.erase(iter);
        }
        else if (lm_state == lm_state_t::Invalid) {
            ++num_removed;
            lm->prepare_for_erasing(map_db_);
            iter = fresh_landmarks_.erase(iter);
        }
        else {
            // hold decision because the state is NotClear
            iter++;
        }
    }

    return num_removed;
}

unsigned int local_map_cleaner::remove_redundant_keyframes(const std::shared_ptr<data::keyframe>& cur_keyfrm) const {
    if (redundant_obs_ratio_thr_ < 0.0 || top_n_covisibilities_to_search_ <= 0) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
    // window size not to remove
    constexpr unsigned int window_size_not_to_remove = 2;
    // if the redundancy ratio of observations is larger than this threshold,
    // the corresponding keyframe will be erased
    unsigned int num_removed = 0;
    // check redundancy for each of the covisibilities
    const auto cur_covisibilities = cur_keyfrm->graph_node_->get_top_n_covisibilities(top_n_covisibilities_to_search_);
    for (const auto& covisibility : cur_covisibilities) {
        // cannot remove the root node
        if (covisibility->graph_node_->is_spanning_root()) {
            continue;
        }
        // cannot remove the recent keyframe(s)
        if (covisibility->id_ <= cur_keyfrm->id_
            && cur_keyfrm->id_ <= covisibility->id_ + window_size_not_to_remove) {
            continue;
        }

        // count the number of redundant observations (num_redundant_obs) and valid observations (num_valid_obs)
        // for the covisibility
        unsigned int num_redundant_obs = 0;
        unsigned int num_valid_obs = 0;
        count_redundant_observations(covisibility, num_valid_obs, num_redundant_obs);

        // if the redundant observation ratio of `covisibility` is larger than the threshold, it will be removed
        if (redundant_obs_ratio_thr_ <= static_cast<float>(num_redundant_obs) / num_valid_obs) {
            ++num_removed;
            const auto cur_landmarks = covisibility->get_landmarks();
            covisibility->prepare_for_erasing(map_db_, bow_db_);
            for (const auto& lm : cur_landmarks) {
                if (!lm) {
                    continue;
                }
                if (lm->will_be_erased()) {
                    continue;
                }
                if (!lm->has_representative_descriptor()) {
                    lm->compute_descriptor();
                }
                if (!lm->has_valid_prediction_parameters()) {
                    lm->update_mean_normal_and_obs_scale_variance();
                }
            }
        }
    }

    return num_removed;
}

void local_map_cleaner::count_redundant_observations(const std::shared_ptr<data::keyframe>& keyfrm, unsigned int& num_valid_obs, unsigned int& num_redundant_obs) const {
    // if the number of keyframes that observes the landmark with more reliable scale than the specified keyframe does,
    // it is considered as redundant
    constexpr unsigned int num_better_obs_thr = 3;

    num_valid_obs = 0;
    num_redundant_obs = 0;

    const auto landmarks = keyfrm->get_landmarks();
    for (unsigned int idx = 0; idx < landmarks.size(); ++idx) {
        const auto& lm = landmarks.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // if depth is within the valid range, it won't be considered
        if (keyfrm->depth_is_available()) {
            assert(!keyfrm->frm_obs_.depths_.empty());
            const auto depth = keyfrm->frm_obs_.depths_.at(idx);
            if (depth < 0.0 || keyfrm->camera_->depth_thr_ < depth) {
                continue;
            }
        }

        ++num_valid_obs;

        // if the number of the obs is smaller than the threshold, cannot remote the observers
        if (lm->num_observations() <= num_better_obs_thr) {
            continue;
        }

        // `keyfrm` observes `lm` with the scale level `scale_level`
        const auto scale_level = keyfrm->frm_obs_.undist_keypts_.at(idx).octave;
        // get observers of `lm`
        const auto observations = lm->get_observations();

        bool obs_by_keyfrm_is_redundant = false;

        // the number of the keyframes that observe `lm` with the more reliable (closer) scale
        unsigned int num_better_obs = 0;

        for (const auto& obs : observations) {
            const auto ngh_keyfrm = obs.first.lock();
            if (*ngh_keyfrm == *keyfrm) {
                continue;
            }

            // `ngh_keyfrm` observes `lm` with the scale level `ngh_scale_level`
            const auto ngh_scale_level = ngh_keyfrm->frm_obs_.undist_keypts_.at(obs.second).octave;

            // compare the scale levels
            if (ngh_scale_level <= scale_level + 1) {
                // the observation by `ngh_keyfrm` is more reliable than `keyfrm`
                ++num_better_obs;
                if (num_better_obs_thr <= num_better_obs) {
                    // if the number of the better observations is greater than the threshold,
                    // consider the observation of `lm` by `keyfrm` is redundant
                    obs_by_keyfrm_is_redundant = true;
                    break;
                }
            }
        }

        if (obs_by_keyfrm_is_redundant) {
            ++num_redundant_obs;
        }
    }
}

} // namespace module
} // namespace cv::slam
