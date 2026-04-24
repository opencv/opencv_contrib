#ifndef SLAM_OPTIMIZE_POSE_OPTIMIZER_G2O_H
#define SLAM_OPTIMIZE_POSE_OPTIMIZER_G2O_H

#include "optimize/pose_optimizer.hpp"

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

class pose_optimizer_g2o : public pose_optimizer {
public:
    /**
     * Constructor
     * @param num_trials_robust
     * @param num_trials
     * @param num_each_iter
     */
    explicit pose_optimizer_g2o(
        unsigned int num_trials_robust = 2,
        unsigned int num_trials = 2,
        unsigned int num_each_iter = 10);

    /**
     * Destructor
     */
    virtual ~pose_optimizer_g2o() = default;

    /**
     * Perform pose optimization
     * @param frm
     * @return
     */
    unsigned int optimize(const data::frame& frm, Mat44_t& optimized_pose, std::vector<bool>& outlier_flags) const override;
    unsigned int optimize(const data::keyframe* keyfrm, Mat44_t& optimized_pose, std::vector<bool>& outlier_flags) const override;

    unsigned int optimize(const Mat44_t& cam_pose_cw, const data::frame_observation& frm_obs,
                          const feature::orb_params* orb_params,
                          const camera::base* camera,
                          const std::vector<std::shared_ptr<data::landmark>>& landmarks,
                          Mat44_t& optimized_pose,
                          std::vector<bool>& outlier_flags) const override;

private:
    //! Number of robust optimization (with outlier rejection) attempts
    const unsigned int num_trials_robust_ = 2;

    //! Number of optimization (with outlier rejection) attempts
    const unsigned int num_trials_ = 2;

    //! Maximum number of iterations for each optimization
    const unsigned int num_each_iter_ = 10;
};

} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_POSE_OPTIMIZER_G2O_H
