#include "camera/equirectangular.hpp"
#include "camera/perspective.hpp"
#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "optimize/pose_optimizer_gtsam.hpp"
#include "optimize/internal_gtsam/projection_factor.hpp"
#include "util/converter.hpp"

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <vector>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {
namespace optimize {

pose_optimizer_gtsam::pose_optimizer_gtsam(unsigned int num_iter,
                                           double relative_error_tol,
                                           double lambda_initial,
                                           double lambda_upper_bound,
                                           bool enable_outlier_elimination,
                                           const std::string& verbosity)
    : num_iter_(num_iter), relative_error_tol_(relative_error_tol),
      lambda_initial_(lambda_initial), lambda_upper_bound_(lambda_upper_bound),
      enable_outlier_elimination_(enable_outlier_elimination), verbosity_(verbosity) {}

unsigned int pose_optimizer_gtsam::optimize(const data::frame& frm, Mat44_t& optimized_pose, std::vector<bool>& outlier_flags) const {
    auto num_valid_obs = optimize(frm.get_pose_cw(), frm.frm_obs_, frm.orb_params_, frm.camera_,
                                  frm.get_landmarks(), optimized_pose, outlier_flags);
    return num_valid_obs;
}

unsigned int pose_optimizer_gtsam::optimize(const data::keyframe* keyfrm, Mat44_t& optimized_pose, std::vector<bool>& outlier_flags) const {
    auto num_valid_obs = optimize(keyfrm->get_pose_cw(), keyfrm->frm_obs_, keyfrm->orb_params_, keyfrm->camera_,
                                  keyfrm->get_landmarks(), optimized_pose, outlier_flags);
    return num_valid_obs;
}

unsigned int pose_optimizer_gtsam::optimize(const Mat44_t& cam_pose_cw, const data::frame_observation& frm_obs,
                                            const feature::orb_params* orb_params,
                                            const camera::base* camera,
                                            const std::vector<std::shared_ptr<data::landmark>>& landmarks,
                                            Mat44_t& optimized_pose,
                                            std::vector<bool>& outlier_flags) const {
    // 1. Construct an optimizer

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    unsigned int num_init_obs = 0;

    // Workaround: normalize rotation. (In the g2o version, rotation seems to be normalized by conversion to SE3Quat.)
    Mat33_t rot_cw = cam_pose_cw.block<3, 3>(0, 0);
    Mat44_t normalized_cam_pose_cw = cam_pose_cw;
    normalized_cam_pose_cw.block<3, 3>(0, 0) = util::converter::to_rot_mat(util::converter::to_angle_axis(rot_cw));

    // 2. Convert the frame to the gtsam value, then set it to values
    values.insert(gtsam::Symbol('x', 0),
                  gtsam::Pose3(util::converter::inverse_pose(normalized_cam_pose_cw)));

    const unsigned int num_keypts = frm_obs.undist_keypts_.size();
    outlier_flags.resize(num_keypts);
    std::fill(outlier_flags.begin(), outlier_flags.end(), false);

    // 3. Connect the landmark vertices by using projection edges

    const double huber_k = 1.345;
    const double sq_huber_k = huber_k * huber_k;

    for (unsigned int idx = 0; idx < num_keypts; ++idx) {
        const auto& lm = landmarks.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        ++num_init_obs;

        // Connect the frame and the landmark vertices using the projection edges
        const auto& undist_keypt = frm_obs.undist_keypts_.at(idx);
        const float x_right = frm_obs.stereo_x_right_.empty() ? -1.0f : frm_obs.stereo_x_right_.at(idx);
        const float sigma_sq = orb_params->level_sigma_sq_.at(undist_keypt.octave);
        // Create reprojection edge from keyfrm
        const auto dim = (x_right < 0.0) ? 2 : 3;
        auto gaussian_noise_model = gtsam::noiseModel::Isotropic::Sigma(dim, sigma_sq);
        auto huber_noise_model = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(huber_k), gaussian_noise_model);

        if (x_right < 0.0) {
            gtsam::Point2 measurement(undist_keypt.pt.x, undist_keypt.pt.y);
            switch (camera->model_type_) {
                case camera::model_type_t::Perspective: {
                    auto cam = static_cast<const camera::perspective*>(camera);
                    using monocular_calibration_type = gtsam::Cal3_S2;
                    using monocular_factor_type = internal_gtsam::PoseOptFactor<gtsam::Pose3, gtsam::Point3,
                                                                                monocular_calibration_type, internal_gtsam::PinholeCamera<monocular_calibration_type>>;
                    monocular_calibration_type::shared_ptr monocular_calibration(new monocular_calibration_type(cam->fx_, cam->fy_, 0.0, cam->cx_, cam->cy_));
                    graph.emplace_shared<monocular_factor_type>(
                        lm->get_pos_in_world(), idx, measurement, huber_noise_model, gtsam::Symbol('x', 0), monocular_calibration);
                    break;
                }
                case camera::model_type_t::Equirectangular: {
                    auto cam = static_cast<const camera::equirectangular*>(camera);
                    using monocular_calibration_type = internal_gtsam::SphericalCameraCalibration;
                    using monocular_factor_type = internal_gtsam::PoseOptFactor<gtsam::Pose3, gtsam::Point3,
                                                                                monocular_calibration_type, internal_gtsam::SphericalCamera<monocular_calibration_type>>;
                    boost::shared_ptr<monocular_calibration_type> monocular_calibration(new monocular_calibration_type(cam->rows_, cam->cols_));
                    graph.emplace_shared<monocular_factor_type>(
                        lm->get_pos_in_world(), idx, measurement, huber_noise_model, gtsam::Symbol('x', 0), monocular_calibration);
                    break;
                }
                case camera::model_type_t::Fisheye: {
                    throw std::runtime_error("Invalid model type: " + camera->get_model_type_string());
                }
                case camera::model_type_t::RadialDivision: {
                    throw std::runtime_error("Invalid model type: " + camera->get_model_type_string());
                }
            }
        }
        else {
            switch (camera->model_type_) {
                case camera::model_type_t::Perspective: {
                    auto cam = static_cast<const camera::perspective*>(camera);
                    using stereo_calibration_type = gtsam::Cal3_S2Stereo;
                    using stereo_factor_type = internal_gtsam::StereoPoseOptFactor<gtsam::Pose3, gtsam::Point3,
                                                                                   stereo_calibration_type, gtsam::StereoCamera>;
                    stereo_calibration_type::shared_ptr stereo_calibration(new stereo_calibration_type(cam->fx_, cam->fy_, 0.0, cam->cx_, cam->cy_, cam->true_baseline_));
                    gtsam::StereoPoint2 measurement(undist_keypt.pt.x, x_right, undist_keypt.pt.y);
                    graph.emplace_shared<stereo_factor_type>(
                        lm->get_pos_in_world(), idx, measurement, huber_noise_model, gtsam::Symbol('x', 0), stereo_calibration);
                    break;
                }
                case camera::model_type_t::Equirectangular: {
                    throw std::runtime_error("Invalid model type: " + camera->get_model_type_string());
                    break;
                }
                case camera::model_type_t::Fisheye: {
                    throw std::runtime_error("Invalid model type: " + camera->get_model_type_string());
                }
                case camera::model_type_t::RadialDivision: {
                    throw std::runtime_error("Invalid model type: " + camera->get_model_type_string());
                }
            }
        }
    }

    if (num_init_obs < 5) {
        return 0;
    }

    // 4. Perform robust Bundle Adjustment (BA)
    gtsam::LevenbergMarquardtParams lm_params;
    lm_params.setMaxIterations(num_iter_);
    lm_params.setRelativeErrorTol(relative_error_tol_);
    lm_params.setlambdaInitial(lambda_initial_);
    lm_params.setlambdaUpperBound(lambda_upper_bound_);
    // lm_params.setVerbosity("TERMINATION");
    lm_params.setVerbosityLM(verbosity_);

    unsigned int num_bad_obs = 0;
    if (enable_outlier_elimination_) {
        std::vector<boost::shared_ptr<internal_gtsam::PoseOptFactorBase<gtsam::Pose3, gtsam::Point3>>> outlier_factors(num_keypts, nullptr);
        const unsigned int num_trials = 2;
        for (unsigned int trial = 0; trial < num_trials; ++trial) {
            gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);
            values = optimizer.optimize();

            for (size_t i = 0; i < graph.size(); ++i) {
                const auto& nonlinear_factor = graph.at(i);
                const auto& factor = (outlier_factors.at(i) == nullptr) ? boost::dynamic_pointer_cast<internal_gtsam::PoseOptFactorBase<gtsam::Pose3, gtsam::Point3>>(nonlinear_factor)
                                                                        : outlier_factors.at(i);
                // filter reprojection edge
                if (factor == nullptr) {
                    continue;
                }
                gtsam::Pose3 pose = values.at<gtsam::Pose3>(factor->key());
                gtsam::Values values;
                values.insert(factor->key(), pose);
                const gtsam::Vector b = factor->unwhitenedError(values);
                auto sq_mahalanobis_distance = factor->noiseModel()->squaredMahalanobisDistance(b);

                if (sq_huber_k < sq_mahalanobis_distance) {
                    if (!outlier_flags.at(factor->idx())) {
                        outlier_flags.at(factor->idx()) = true;
                        ++num_bad_obs;
                        outlier_factors[i] = factor;
                        graph.remove(i);
                    }
                }
                else {
                    if (outlier_flags.at(factor->idx())) {
                        outlier_flags.at(factor->idx()) = false;
                        --num_bad_obs;
                        graph[i] = outlier_factors.at(i);
                        outlier_factors[i] = nullptr;
                    }
                }

                if (trial == 0) {
                    if (graph[i]) {
                        graph.remove(i);
                        graph[i] = factor->cloneWithNewNoiseModel(boost::dynamic_pointer_cast<gtsam::noiseModel::Robust>(factor->noiseModel())->noise());
                    }
                }
            }

            if (num_init_obs - num_bad_obs < 5) {
                break;
            }
        }
    }
    else {
        gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);
        values = optimizer.optimize();
    }

    // 5. Update the information

    auto pose = values.at<gtsam::Pose3>(gtsam::Symbol('x', 0));
    optimized_pose = util::converter::inverse_pose(pose.matrix());

    return num_init_obs - num_bad_obs;
}

} // namespace optimize
} // namespace cv::slam
