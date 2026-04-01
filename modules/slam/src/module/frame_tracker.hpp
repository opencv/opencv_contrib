#ifndef SLAM_MODULE_FRAME_TRACKER_H
#define SLAM_MODULE_FRAME_TRACKER_H

#include "type.hpp"
#include "optimize/pose_optimizer.hpp"

#include <memory>

namespace cv::slam {

namespace camera {
class base;
} // namespace camera

namespace data {
class frame;
class keyframe;
class bow_database;
} // namespace data

namespace module {

class frame_tracker {
public:
    explicit frame_tracker(camera::base* camera,
                           const std::shared_ptr<optimize::pose_optimizer>& pose_optimizer,
                           const unsigned int num_matches_thr = 20,
                           bool use_fixed_seed = false,
                           float margin = 20.0);

    bool motion_based_track(data::frame& curr_frm, const data::frame& last_frm, const Mat44_t& velocity) const;

    bool bow_match_based_track(data::frame& curr_frm, const data::frame& last_frm, const std::shared_ptr<data::keyframe>& ref_keyfrm) const;

    bool robust_match_based_track(data::frame& curr_frm, const data::frame& last_frm, const std::shared_ptr<data::keyframe>& ref_keyfrm) const;

private:
    unsigned int discard_outliers(const std::vector<bool>& outlier_flags, data::frame& curr_frm) const;

    const camera::base* camera_;
    const unsigned int num_matches_thr_;
    //! Use fixed random seed for RANSAC if true
    const bool use_fixed_seed_;
    //! margin for projection matcher
    const float margin_;

    std::shared_ptr<optimize::pose_optimizer> pose_optimizer_ = nullptr;
};

} // namespace module
} // namespace cv::slam

#endif // SLAM_MODULE_FRAME_TRACKER_H
