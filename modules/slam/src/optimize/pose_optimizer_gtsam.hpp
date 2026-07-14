#ifndef SLAM_OPTIMIZE_POSE_OPTIMIZER_GTSAM_H
#define SLAM_OPTIMIZE_POSE_OPTIMIZER_GTSAM_H

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

class pose_optimizer_gtsam : public pose_optimizer {
public:
    /**
     * Constructor
     * @param num_iter
     */
    explicit pose_optimizer_gtsam(unsigned int num_iter = 3,
                                  double relative_error_tol = 1e-2,
                                  double lambda_initial = 1e-1,
                                  double lambda_upper_bound = 1e+2,
                                  bool enable_outlier_elimination = true,
                                  const std::string& verbosity = "SILENT");

    /**
     * Destructor
     */
    virtual ~pose_optimizer_gtsam() = default;

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
    const unsigned int num_iter_ = 5;
    const double relative_error_tol_ = 1e-2;
    const double lambda_initial_ = 1e-5;
    const double lambda_upper_bound_ = 1e-2;
    const bool enable_outlier_elimination_ = true;
    const std::string verbosity_ = "SILENT";
};

} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_POSE_OPTIMIZER_GTSAM_H
