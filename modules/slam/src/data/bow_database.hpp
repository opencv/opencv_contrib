#ifndef SLAM_DATA_BOW_DATABASE_H
#define SLAM_DATA_BOW_DATABASE_H

#include "data/bow_vocabulary.hpp"

#include <mutex>
#include <list>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace cv::slam {
namespace data {

class frame;
class keyframe;

class bow_database {
public:
    /**
     * Constructor
     * @param bow_vocab
     */
    explicit bow_database(bow_vocabulary* bow_vocab);

    /**
     * Destructor
     */
    ~bow_database();

    /**
     * Add a keyframe to the database
     * @param keyfrm
     */
    void add_keyframe(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Erase the keyframe from the database
     * @param keyfrm
     */
    void erase_keyframe(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Clear the database
     */
    void clear();

    /**
     * Acquire keyframes over score
     */
    std::vector<std::shared_ptr<keyframe>> acquire_keyframes(const bow_vector& bow_vec, const float min_score = 0.0f,
                                                             const float num_common_words_thr_ratio = 0.8f,
                                                             const std::set<std::shared_ptr<keyframe>>& keyfrms_to_reject = {});

protected:
    /**
     * Initialize temporary variables
     */
    void initialize();

    /**
     * Compute the number of shared words
     * @param bow_vec
     * @param keyfrms_to_reject
     * @return number of shared words between the query and the each of keyframes contained in the database (key: keyframes that share word with query keyframe, value: number of shared words)
     */
    std::unordered_map<std::shared_ptr<keyframe>, unsigned int>
    compute_num_common_words(const bow_vector& bow_vec,
                             const std::set<std::shared_ptr<keyframe>>& keyfrms_to_reject = {}) const;

    /**
     * Compute scores (scores_) between the query and the each of keyframes contained in the database
     * @param num_common_words
     * @param bow_vec
     * @param min_num_common_words_thr
     * @return similarity scores between the query and the each of keyframes contained in the database (key: keyframes that share word with query keyframe, value: score)
     */
    std::unordered_map<std::shared_ptr<keyframe>, float>
    compute_scores(const std::unordered_map<std::shared_ptr<keyframe>, unsigned int>& num_common_words,
                   const bow_vector& bow_vec,
                   const unsigned int min_num_common_words_thr,
                   const float min_score,
                   float& best_score) const;

    //-----------------------------------------
    // BoW feature vectors

    //! mutex to access BoW database
    mutable std::mutex mtx_;
    //! BoW database
    std::unordered_map<unsigned int, std::list<std::shared_ptr<keyframe>>> keyfrms_in_node_;

    //-----------------------------------------
    // BoW vocabulary

    //! BoW vocabulary
    bow_vocabulary* bow_vocab_;
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_BOW_DATABASE_H
