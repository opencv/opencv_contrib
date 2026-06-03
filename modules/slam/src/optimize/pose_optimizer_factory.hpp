#ifndef SLAM_OPTIMIZE_POSE_OPTIMIZER_FACTORY_H
#define SLAM_OPTIMIZE_POSE_OPTIMIZER_FACTORY_H

#include "optimize/pose_optimizer_g2o.hpp"
#ifdef USE_GTSAM
#include "optimize/pose_optimizer_gtsam.hpp"
#endif // USE_GTSAM
#include "type.hpp"
#include "util/yaml.hpp"

#include <memory>

namespace cv::slam {
namespace optimize {

class pose_optimizer_factory {
public:
    static std::unique_ptr<pose_optimizer> create(const cv::FileNode& yaml_node) {
        const auto& backend = util::yaml_get_val<std::string>(yaml_node, "backend", "g2o");
        if (backend == "g2o") {
            cv::FileNode g2o_node = util::yaml_optional_ref(yaml_node, "g2o");
            return std::unique_ptr<pose_optimizer>(new pose_optimizer_g2o(
                util::yaml_get_val<unsigned int>(g2o_node, "num_trials_robust", 2),
                util::yaml_get_val<unsigned int>(g2o_node, "num_trials", 2),
                util::yaml_get_val<unsigned int>(g2o_node, "num_each_iter", 10)));
        }
        else if (backend == "gtsam") {
#ifdef USE_GTSAM
            cv::FileNode gtsam_node = util::yaml_optional_ref(yaml_node, "gtsam");
            auto num_iter = util::yaml_get_val<unsigned int>(gtsam_node, "num_iter", 5);
            auto relative_error_tol = util::yaml_get_val<double>(gtsam_node, "relative_error_tol", 1e-2);
            auto lambda_initial = util::yaml_get_val<double>(gtsam_node, "lambda_initial", 1e-5);
            auto lambda_upper_bound = util::yaml_get_val<double>(gtsam_node, "lambda_upper_bound", 1e-2);
            auto enable_outlier_elimination = util::yaml_get_val<bool>(gtsam_node, "enable_outlier_elimination", true);
            auto verbosity = util::yaml_get_val<std::string>(gtsam_node, "verbosity", "SILENT");
            return std::unique_ptr<pose_optimizer>(new pose_optimizer_gtsam(
                num_iter, relative_error_tol,
                lambda_initial, lambda_upper_bound,
                enable_outlier_elimination, verbosity));
#else
            throw std::runtime_error("gtsam is not enabled");
#endif // USE_GTSAM
        }
        else {
            throw std::runtime_error("Invalid backend");
        }
    }
};

} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_POSE_OPTIMIZER_FACTORY_H
