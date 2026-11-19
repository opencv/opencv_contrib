#ifndef SLAM_OPTIMIZE_POSE_OPTIMIZER_H
#define SLAM_OPTIMIZE_POSE_OPTIMIZER_H

#include "type.hpp"

namespace cv::slam {

namespace data {
class frame;
struct frame_observation;
class keyframe;
} // namespace data

namespace camera {
class base;
} // namespace camera

namespace feature {
struct orb_params;
} // namespace feature

namespace optimize {

class pose_optimizer {
public:
    /**
     * Perform pose optimization
     * @param frm
     * @return
     */
    virtual unsigned int optimize(const data::frame& frm, Mat44_t& optimized_pose, std::vector<bool>& outlier_flags) const = 0;
    virtual unsigned int optimize(const data::keyframe* keyfrm, Mat44_t& optimized_pose, std::vector<bool>& outlier_flags) const = 0;

    virtual unsigned int optimize(const Mat44_t& cam_pose_cw, const data::frame_observation& frm_obs,
                                  const feature::orb_params* orb_params,
                                  const camera::base* camera,
                                  const std::vector<std::shared_ptr<data::landmark>>& landmarks,
                                  Mat44_t& optimized_pose,
                                  std::vector<bool>& outlier_flags) const = 0;
};

} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_POSE_OPTIMIZER_H
