#ifndef SLAM_MATCH_PROJECTION_H
#define SLAM_MATCH_PROJECTION_H

#include "type.hpp"
#include "match/base.hpp"

#include <set>
#include <memory>

namespace cv::slam {

namespace data {
class frame;
struct frame_observation;
class keyframe;
class landmark;
} // namespace data

namespace match {

class projection final : public base {
public:
    explicit projection(const float lowe_ratio = 0.6, const bool check_orientation = true)
        : base(lowe_ratio, check_orientation) {}

    ~projection() final = default;

    
    unsigned int match_frame_and_landmarks(data::frame& frm,
                                           const std::vector<std::shared_ptr<data::landmark>>& local_landmarks,
                                           eigen_alloc_unord_map<unsigned int, Vec2_t>& lm_to_reproj,
                                           std::unordered_map<unsigned int, float>& lm_to_x_right,
                                           std::unordered_map<unsigned int, unsigned int>& lm_to_scale,
                                           const float margin = 5.0) const;

    
    unsigned int match_current_and_last_frames(data::frame& curr_frm, const data::frame& last_frm, const float margin) const;

    
    
    unsigned int match_frame_and_keyframe(data::frame& curr_frm, const std::shared_ptr<data::keyframe>& keyfrm, const std::set<std::shared_ptr<data::landmark>>& already_matched_lms,
                                          const float margin, const unsigned int hamm_dist_thr) const;
    unsigned int match_frame_and_keyframe(const Mat44_t& cam_pose_cw,
                                          const camera::base* camera,
                                          const data::frame_observation& frm_obs,
                                          const feature::orb_params* orb_params,
                                          std::vector<std::shared_ptr<data::landmark>>& frm_landmarks,
                                          const std::shared_ptr<data::keyframe>& keyfrm,
                                          const std::set<std::shared_ptr<data::landmark>>& already_matched_lms,
                                          const float margin, const unsigned int hamm_dist_thr) const;

    
    
    
    unsigned int match_by_Sim3_transform(const std::shared_ptr<data::keyframe>& keyfrm, const Mat44_t& Sim3_cw, const std::vector<std::shared_ptr<data::landmark>>& landmarks,
                                         std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_keyfrm, const float margin) const;

    
    
    unsigned int match_keyframes_mutually(const std::shared_ptr<data::keyframe>& keyfrm_1, const std::shared_ptr<data::keyframe>& keyfrm_2, std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_keyfrm_1,
                                          const float& s_12, const Mat33_t& rot_12, const Vec3_t& trans_12, const float margin) const;
};

} // namespace match
} // namespace cv::slam

#endif // SLAM_MATCH_PROJECTION_H
