#ifndef SLAM_MATCH_ROBUST_H
#define SLAM_MATCH_ROBUST_H

#include "type.hpp"
#include "match/base.hpp"

#include <memory>

namespace cv::slam {

namespace data {
class frame;
struct frame_observation;
class keyframe;
class landmark;
} // namespace data

namespace match {

class robust final : public base {
public:
    explicit robust(const float lowe_ratio, const bool check_orientation)
        : base(lowe_ratio, check_orientation) {}

    ~robust() final = default;

    unsigned int match_for_triangulation(const std::shared_ptr<data::keyframe>& keyfrm_1,
                                         const std::shared_ptr<data::keyframe>& keyfrm_2,
                                         const Mat33_t& E_12,
                                         std::vector<std::pair<unsigned int, unsigned int>>& matched_idx_pairs,
                                         const float residual_rad_thr) const;

    unsigned int match_keyframes(const std::shared_ptr<data::keyframe>& keyfrm1, const std::shared_ptr<data::keyframe>& keyfrm2,
                                 std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm,
                                 bool validate_with_essential_solver = true, bool use_fixed_seed = false) const;

    unsigned int match_frame_and_keyframe(data::frame& frm, const std::shared_ptr<data::keyframe>& keyfrm,
                                          std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm,
                                          bool use_fixed_seed = false) const;

    unsigned int brute_force_match(const data::frame_observation& frm_obs, const std::shared_ptr<data::keyframe>& keyfrm, std::vector<std::pair<int, int>>& matches) const;
};

} // namespace match
} // namespace cv::slam

#endif // SLAM_MATCH_ROBUST_H
