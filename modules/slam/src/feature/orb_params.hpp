#ifndef SLAM_FEATURE_ORB_PARAMS_H
#define SLAM_FEATURE_ORB_PARAMS_H

#include <nlohmann/json_fwd.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>

namespace cv::slam {
namespace feature {

struct orb_params {
    orb_params() = delete;

    //! Constructor
    orb_params(const std::string& name, const float scale_factor, const unsigned int num_levels,
               const unsigned int ini_fast_thr, const unsigned int min_fast_thr);
    orb_params(const std::string& name);

    //! Constructor
    explicit orb_params(const YAML::Node& yaml_node);

    //! Destructor
    virtual ~orb_params() = default;

    nlohmann::json to_json() const;

    //! name (id for saving)
    const std::string name_;

    const float scale_factor_ = 1.2;
    const float log_scale_factor_ = std::log(1.2);
    const unsigned int num_levels_ = 8;
    const unsigned int ini_fast_thr_ = 20;
    const unsigned int min_fast_thr_ = 7;

    //! A list of the scale factor of each pyramid layer
    std::vector<float> scale_factors_;
    std::vector<float> inv_scale_factors_;
    //! A list of the sigma of each pyramid layer
    std::vector<float> level_sigma_sq_;
    std::vector<float> inv_level_sigma_sq_;

    //! Calculate scale factors
    static std::vector<float> calc_scale_factors(const unsigned int num_scale_levels, const float scale_factor);

    //! Calculate inverses of scale factors
    static std::vector<float> calc_inv_scale_factors(const unsigned int num_scale_levels, const float scale_factor);

    //! Calculate squared sigmas at all levels
    static std::vector<float> calc_level_sigma_sq(const unsigned int num_scale_levels, const float scale_factor);

    //! Calculate inverses of squared sigmas at all levels
    static std::vector<float> calc_inv_level_sigma_sq(const unsigned int num_scale_levels, const float scale_factor);
};

std::ostream& operator<<(std::ostream& os, const orb_params& oparam);

} // namespace feature
} // namespace cv::slam

#endif // SLAM_FEATURE_ORB_PARAMS_H
