#ifndef SLAM_MODULE_LOCAL_MAP_UPDATER_H
#define SLAM_MODULE_LOCAL_MAP_UPDATER_H

#include <memory>

namespace cv::slam {

namespace data {
class frame;
class keyframe;
class landmark;
} // namespace data

namespace module {

class local_map_updater {
public:
    using keyframe_to_num_shared_lms_t = nondeterministic::unordered_map<std::shared_ptr<data::keyframe>, unsigned int>;

    //! Constructor
    explicit local_map_updater(const unsigned int max_num_local_keyfrms);

    //! Destructor
    ~local_map_updater() = default;

    //! Get the local keyframes
    std::vector<std::shared_ptr<data::keyframe>> get_local_keyframes() const;

    //! Get the local landmarks
    std::vector<std::shared_ptr<data::landmark>> get_local_landmarks() const;

    //! Get the nearest covisibility
    std::shared_ptr<data::keyframe> get_nearest_covisibility() const;

    //! Acquire the new local map
    bool acquire_local_map(const std::vector<std::shared_ptr<data::landmark>>& frm_lms);
    bool acquire_local_map(const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
                           unsigned int keyframe_id_threshold,
                           unsigned int& num_temporal_keyfrms);

private:
    //! Find the local keyframes
    bool find_local_keyframes(const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
                              unsigned int keyframe_id_threshold,
                              unsigned int& num_temporal_keyfrms);

    //! Count the number of shared landmarks between the current frame and each of the neighbor keyframes
    auto count_num_shared_lms(
        const std::vector<std::shared_ptr<data::landmark>>& frm_lms,
        unsigned int keyframe_id_threshold) const
        -> std::vector<std::pair<unsigned int, std::shared_ptr<data::keyframe>>>;

    //! Find the first-order local keyframes
    auto find_first_local_keyframes(
        const std::vector<std::pair<unsigned int, std::shared_ptr<data::keyframe>>>& keyfrm_weights,
        const unsigned int keyframe_id_threshold,
        std::unordered_set<unsigned int>& already_found_keyfrm_ids,
        unsigned int& num_temporal_keyfrms)
        -> std::vector<std::shared_ptr<data::keyframe>>;

    //! Find the second-order local keyframes
    auto find_second_local_keyframes(const std::vector<std::shared_ptr<data::keyframe>>& first_local_keyframes,
                                     unsigned int keyframe_id_threshold,
                                     std::unordered_set<unsigned int>& already_found_keyfrm_ids,
                                     unsigned int& num_temporal_keyfrms) const
        -> std::vector<std::shared_ptr<data::keyframe>>;

    //! Find the local landmarks
    bool find_local_landmarks(const std::vector<std::shared_ptr<data::landmark>>& frm_lms);

    // maximum number of the local keyframes
    const unsigned int max_num_local_keyfrms_;

    // found local keyframes
    std::vector<std::shared_ptr<data::keyframe>> local_keyfrms_;
    // found local landmarks
    std::vector<std::shared_ptr<data::landmark>> local_lms_;
    // the nearst keyframe in covisibility graph, which will be found in find_first_local_keyframes()
    std::shared_ptr<data::keyframe> nearest_covisibility_;
};

} // namespace module
} // namespace cv::slam

#endif // SLAM_MODULE_LOCAL_MAP_UPDATER_H
