#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "data/bow_database.hpp"
#include "data/bow_vocabulary.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace data {

bow_database::bow_database(bow_vocabulary* bow_vocab)
    : bow_vocab_(bow_vocab) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: data::bow_database");
}

bow_database::~bow_database() {
    clear();
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: data::bow_database");
}

void bow_database::add_keyframe(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);

    // Append keyframe to the corresponding index in keyframes_in_node_ list
    for (const auto& node_id_and_weight : keyfrm->bow_vec_) {
        keyfrms_in_node_[node_id_and_weight.first].push_back(keyfrm);
    }
}

void bow_database::erase_keyframe(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);

    // Delete keyframe from the coresponding index in keyframes_in_node_ list
    for (const auto& node_id_and_weight : keyfrm->bow_vec_) {
        // first: node ID, second: weight
        if (!static_cast<bool>(keyfrms_in_node_.count(node_id_and_weight.first))) {
            continue;
        }
        // Obtain keyframe which shares word
        auto& keyfrms_in_node = keyfrms_in_node_.at(node_id_and_weight.first);

        // std::list::erase only accepts iterator
        for (auto itr = keyfrms_in_node.begin(), lend = keyfrms_in_node.end(); itr != lend; itr++) {
            if (keyfrm->id_ == (*itr)->id_) {
                keyfrms_in_node.erase(itr);
                break;
            }
        }
    }
}

void bow_database::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    CV_LOG_INFO(&g_log_tag, "clear BoW database");
    keyfrms_in_node_.clear();
}

std::vector<std::shared_ptr<keyframe>> bow_database::acquire_keyframes(const bow_vector& bow_vec, const float min_score,
                                                                       const float num_common_words_thr_ratio,
                                                                       const std::set<std::shared_ptr<keyframe>>& keyfrms_to_reject) {
    // Step 1.
    // Count up the number of nodes, words which are shared with query_keyframe, for all the keyframes in DoW database

    const auto num_common_words = compute_num_common_words(bow_vec, keyfrms_to_reject);
    if (num_common_words.empty()) {
        return std::vector<std::shared_ptr<keyframe>>();
    }

    // Set min_num_common_words_thr as 80 percentile of max_num_common_words
    // for the following selection of candidate keyframes.
    // (Delete frames from candidates if it has less shared words than 80% of the max_num_common_words)
    unsigned int max_num_common_words = 0;
    for (const auto& keyfrm_num_common_words_pair : num_common_words) {
        if (max_num_common_words < keyfrm_num_common_words_pair.second) {
            max_num_common_words = keyfrm_num_common_words_pair.second;
        }
    }
    const auto min_num_common_words_thr = static_cast<unsigned int>(num_common_words_thr_ratio * max_num_common_words);

    // Step 2.
    // Collect keyframe candidates which have more shared words than min_num_common_words_thr
    // by calculating similarity score between each candidate and the query keyframe.

    float best_score = min_score;
    const auto scores = compute_scores(num_common_words, bow_vec, min_num_common_words_thr, min_score, best_score);
    if (scores.empty()) {
        return std::vector<std::shared_ptr<keyframe>>();
    }

    std::unordered_set<std::shared_ptr<keyframe>> final_candidates;
    for (const auto& keyfrm_score : scores) {
        const auto keyfrm = keyfrm_score.first;
        final_candidates.insert(keyfrm);
    }
    return std::vector<std::shared_ptr<keyframe>>(final_candidates.begin(), final_candidates.end());
}

std::unordered_map<std::shared_ptr<keyframe>, unsigned int>
bow_database::compute_num_common_words(const bow_vector& bow_vec,
                                       const std::set<std::shared_ptr<keyframe>>& keyfrms_to_reject) const {
    std::unordered_map<std::shared_ptr<keyframe>, unsigned int> num_common_words;

    std::lock_guard<std::mutex> lock(mtx_);

    // Count the number of shared words for keyframes which share the word with the query keyframe
    for (const auto& node_id_and_weight : bow_vec) {
        // first: node ID, second: weight
        // If not in the BoW database, continue
        if (!static_cast<bool>(keyfrms_in_node_.count(node_id_and_weight.first))) {
            continue;
        }
        // Get a keyframe which shares the word (node ID) with the query
        const auto& keyfrms_in_node = keyfrms_in_node_.at(node_id_and_weight.first);
        // For each keyframe, increase shared word number one by one
        for (const auto& keyfrm_in_node : keyfrms_in_node) {
            // If far enough from the query keyframe, store it as the initial loop candidates
            if (!static_cast<bool>(keyfrms_to_reject.count(keyfrm_in_node))) {
                // Initialize if not in num_common_words
                if (!static_cast<bool>(num_common_words.count(keyfrm_in_node))) {
                    num_common_words[keyfrm_in_node] = 0;
                }
                // Count up the number of words
                ++num_common_words.at(keyfrm_in_node);
            }
        }
    }

    return num_common_words;
}

std::unordered_map<std::shared_ptr<keyframe>, float>
bow_database::compute_scores(const std::unordered_map<std::shared_ptr<keyframe>, unsigned int>& num_common_words,
                             const bow_vector& bow_vec,
                             const unsigned int min_num_common_words_thr,
                             const float min_score,
                             float& best_score) const {
    std::unordered_map<std::shared_ptr<keyframe>, float> scores;

    best_score = min_score;

    for (const auto& keyfrm_num_common_words_pair : num_common_words) {
        const auto& keyfrm = keyfrm_num_common_words_pair.first;
        if (min_num_common_words_thr < keyfrm_num_common_words_pair.second) {
            // Calculate similarity score with query keyframe
            // for the keyframes which have more shared words than minimum common words
            const auto score = data::bow_vocabulary_util::score(bow_vocab_, bow_vec, keyfrm->bow_vec_);
            if (min_score > score) {
                continue;
            }
            if (best_score < score) {
                best_score = score;
            }
            // Store score
            scores[keyfrm] = score;
        }
    }

    return scores;
}

} // namespace data
} // namespace cv::slam
