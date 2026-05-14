#include "camera/equirectangular.hpp"
#include "util/yaml.hpp"
#include "camera/perspective.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/map_database.hpp"
#include "optimize/local_bundle_adjuster_gtsam.hpp"
#include "optimize/internal_gtsam/projection_factor.hpp"
#include "util/converter.hpp"

#include <unordered_map>

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace optimize {

local_bundle_adjuster_gtsam::local_bundle_adjuster_gtsam(const cv::FileNode& yaml_node,
                                                         const unsigned int num_first_iter,
                                                         const unsigned int num_second_iter)
    : num_first_iter_(num_first_iter), num_second_iter_(num_second_iter),
      use_additional_keyframes_for_monocular_(util::yaml_get_val<bool>(yaml_node, "use_additional_keyframes_for_monocular", false)) {
}

void local_bundle_adjuster_gtsam::optimize(data::map_database* map_db,
                                           const std::shared_ptr<cv::slam::data::keyframe>& curr_keyfrm, bool* const force_stop_flag) const {
    // 1. Aggregate the local and fixed keyframes, and local landmarks

    // Correct the local keyframes of the current keyframe
    std::unordered_map<unsigned int, std::shared_ptr<data::keyframe>> local_keyfrms;
    bool has_scale = false;

    local_keyfrms[curr_keyfrm->id_] = curr_keyfrm;
    const auto curr_covisibilities = curr_keyfrm->graph_node_->get_covisibilities();
    for (const auto& local_keyfrm : curr_covisibilities) {
        if (!local_keyfrm) {
            continue;
        }
        if (local_keyfrm->will_be_erased()) {
            continue;
        }
        if (local_keyfrm->graph_node_->is_spanning_root()) {
            continue;
        }
        if (local_keyfrm->id_ < map_db->get_fixed_keyframe_id_threshold()) {
            continue;
        }

        local_keyfrms[local_keyfrm->id_] = local_keyfrm;
        if (local_keyfrm->camera_->setup_type_ != camera::setup_type_t::Monocular) {
            has_scale = true;
        }
    }

    // Correct landmarks seen in local keyframes
    std::unordered_map<unsigned int, std::shared_ptr<data::landmark>> local_lms;

    for (const auto& local_keyfrm : local_keyfrms) {
        const auto landmarks = local_keyfrm.second->get_landmarks();
        for (const auto& local_lm : landmarks) {
            if (!local_lm) {
                continue;
            }
            if (local_lm->will_be_erased()) {
                continue;
            }

            // Avoid duplication
            if (local_lms.count(local_lm->id_)) {
                continue;
            }

            local_lms[local_lm->id_] = local_lm;
        }
    }

    // Fixed keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
    std::unordered_map<unsigned int, std::shared_ptr<data::keyframe>> fixed_keyfrms;

    for (const auto& local_lm : local_lms) {
        const auto observations = local_lm.second->get_observations();
        for (const auto& obs : observations) {
            const auto fixed_keyfrm = obs.first.lock();
            if (!fixed_keyfrm) {
                continue;
            }
            if (fixed_keyfrm->will_be_erased()) {
                continue;
            }

            // Do not add if it's in the local keyframes
            if (local_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            // Avoid duplication
            if (fixed_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            fixed_keyfrms[fixed_keyfrm->id_] = fixed_keyfrm;
        }
    }

    if (use_additional_keyframes_for_monocular_) {
        // Ensure that there are always at least two fixed keyframes
        auto additional_keyfrms_size = 2 - fixed_keyfrms.size();
        if (!has_scale && fixed_keyfrms.size() < 2 && local_keyfrms.size() > additional_keyfrms_size) {
            for (unsigned int i = 0; i < additional_keyfrms_size; ++i) {
                auto itr = local_keyfrms.begin();
                auto keyfrm_id = itr->first;
                auto keyfrm = itr->second;
                local_keyfrms.erase(keyfrm_id);
                fixed_keyfrms[keyfrm_id] = keyfrm;
            }
        }
    }

    // 2. Construct an optimizer

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_estimate;

    // 3. Convert each of the keyframe to the gtsam vertex, then set it to the optimizer

    // Save the converted keyframes
    std::unordered_map<unsigned int, std::shared_ptr<data::keyframe>> all_keyfrms;

    // Set the local keyframes to the optimizer
    for (const auto& id_local_keyfrm_pair : local_keyfrms) {
        const auto& local_keyfrm = id_local_keyfrm_pair.second;

        all_keyfrms.emplace(id_local_keyfrm_pair);
        initial_estimate.insert(gtsam::Symbol('x', id_local_keyfrm_pair.first),
                                gtsam::Pose3(local_keyfrm->get_pose_wc()));
    }

    // Set the fixed keyframes to the optimizer
    for (const auto& id_fixed_keyfrm_pair : fixed_keyfrms) {
        const auto& fixed_keyfrm = id_fixed_keyfrm_pair.second;

        all_keyfrms.emplace(id_fixed_keyfrm_pair);
        auto pose = gtsam::Pose3(fixed_keyfrm->get_pose_wc());
        initial_estimate.insert(gtsam::Symbol('x', id_fixed_keyfrm_pair.first),
                                pose);
        // Fix keyframe
        auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << gtsam::Vector3::Constant(1e-3), gtsam::Vector3::Constant(1e-2))
                .finished());
        graph.addPrior(gtsam::Symbol('x', id_fixed_keyfrm_pair.first), pose, poseNoise);
    }

    // 4. Connect the vertices of the keyframe and the landmark by using an edge of reprojection constraint

    const double huber_k = 1.345;
    constexpr double chi_sq_2D = 5.99146;  // chi-square threshold, 2 dof, 95% confidence
    constexpr double chi_sq_3D = 7.81473;  // chi-square threshold, 3 dof, 95% confidence

    for (const auto& id_local_lm_pair : local_lms) {
        const auto local_lm = id_local_lm_pair.second;
        const auto observations = local_lm->get_observations();
        if (observations.empty()) {
            CV_LOG_WARNING(&g_log_tag, "empty observation");
            continue;
        }

        // Convert the landmark to the gtsam vertex, then set to the optimizer
        auto point = gtsam::Point3(local_lm->get_pos_in_world());
        initial_estimate.insert(gtsam::Symbol('l', id_local_lm_pair.first),
                                point);

        unsigned int num_edges = 0;
        for (const auto& obs : observations) {
            const auto keyfrm = obs.first.lock();
            auto idx = obs.second;
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }
            if (!initial_estimate.exists(gtsam::Symbol('x', keyfrm->id_))) {
                continue;
            }

            const auto& undist_keypt = keyfrm->frm_obs_.undist_keypts_.at(idx);
            const float x_right = keyfrm->frm_obs_.stereo_x_right_.empty() ? -1.0f : keyfrm->frm_obs_.stereo_x_right_.at(idx);
            const float sigma_sq = keyfrm->orb_params_->level_sigma_sq_.at(undist_keypt.octave);
            // NOTE: Currently only Perspective camera model is supported for GTSAM-based local BA.
            // Fisheye, equirectangular, and radial division models are not yet implemented.
            assert(keyfrm->camera_->model_type_ == camera::model_type_t::Perspective);
            // Create reprojection edge from keyfrm
            const auto dim = (x_right < 0.0) ? 2 : 3;
            auto gaussian_noise_model = gtsam::noiseModel::Isotropic::Sigma(dim, sigma_sq);
            auto huber_noise_model = gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(huber_k), gaussian_noise_model);
            if (x_right < 0.0) {
                gtsam::Point2 measurement(undist_keypt.pt.x, undist_keypt.pt.y);
                switch (keyfrm->camera_->model_type_) {
                    case camera::model_type_t::Perspective: {
                        auto cam = static_cast<camera::perspective*>(keyfrm->camera_);
                        using monocular_calibration_type = gtsam::Cal3_S2;
                        using monocular_factor_type = internal_gtsam::ProjectionFactor<gtsam::Pose3, gtsam::Point3,
                                                                                       monocular_calibration_type, internal_gtsam::PinholeCamera<monocular_calibration_type>>;
                        monocular_calibration_type::shared_ptr monocular_calibration(new monocular_calibration_type(cam->fx_, cam->fy_, 0.0, cam->cx_, cam->cy_));
                        graph.emplace_shared<monocular_factor_type>(
                            measurement, huber_noise_model, gtsam::Symbol('x', keyfrm->id_), gtsam::Symbol('l', id_local_lm_pair.first), monocular_calibration);
                        break;
                    }
                    case camera::model_type_t::Equirectangular: {
                        auto cam = static_cast<camera::equirectangular*>(keyfrm->camera_);
                        using monocular_calibration_type = internal_gtsam::SphericalCameraCalibration;
                        using monocular_factor_type = internal_gtsam::ProjectionFactor<gtsam::Pose3, gtsam::Point3,
                                                                                       monocular_calibration_type, internal_gtsam::SphericalCamera<monocular_calibration_type>>;
                        boost::shared_ptr<monocular_calibration_type> monocular_calibration(new monocular_calibration_type(cam->rows_, cam->cols_));
                        graph.emplace_shared<monocular_factor_type>(
                            measurement, huber_noise_model, gtsam::Symbol('x', keyfrm->id_), gtsam::Symbol('l', id_local_lm_pair.first), monocular_calibration);
                        break;
                    }
                    case camera::model_type_t::Fisheye: {
                        throw std::runtime_error("Invalid model type: " + keyfrm->camera_->get_model_type_string());
                    }
                    case camera::model_type_t::RadialDivision: {
                        throw std::runtime_error("Invalid model type: " + keyfrm->camera_->get_model_type_string());
                    }
                }
            }
            else {
                auto cam = static_cast<camera::perspective*>(keyfrm->camera_);
                using stereo_calibration_type = gtsam::Cal3_S2Stereo;
                using stereo_factor_type = internal_gtsam::StereoProjectionFactor<gtsam::Pose3, gtsam::Point3,
                                                                                  stereo_calibration_type, gtsam::StereoCamera>;
                gtsam::StereoPoint2 measurement(undist_keypt.pt.x, x_right, undist_keypt.pt.y);
                stereo_calibration_type::shared_ptr stereo_calibration(new stereo_calibration_type(cam->fx_, cam->fy_, 0.0, cam->cx_, cam->cy_, cam->true_baseline_));
                graph.emplace_shared<stereo_factor_type>(
                    measurement, huber_noise_model, gtsam::Symbol('x', keyfrm->id_), gtsam::Symbol('l', id_local_lm_pair.first), stereo_calibration);
            }
            ++num_edges;
        }

        if (num_edges == 0) {
            initial_estimate.erase(gtsam::Symbol('l', id_local_lm_pair.first));
            CV_LOG_WARNING(&g_log_tag, "lm(" << local_lm->id_ << ") no edges");
        }
    }

    // 5. Perform the first optimization

    if (force_stop_flag && *force_stop_flag) {
        return;
    }

    gtsam::LevenbergMarquardtParams lm_params;
    lm_params.setMaxIterations(num_first_iter_);
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, lm_params);
    gtsam::Values result = optimizer.optimize();

    // 6. Discard outliers, then perform the second optimization

    bool run_robust_BA = true;
    if (force_stop_flag && *force_stop_flag) {
        run_robust_BA = false;
    }

    gtsam::Values final_result = result;

    if (run_robust_BA) {
        gtsam::NonlinearFactorGraph graph2;
        gtsam::Values initial_estimate2 = result;

        // Re-add fixed keyframe priors
        for (const auto& id_fixed_keyfrm_pair : fixed_keyfrms) {
            const auto& fixed_keyfrm = id_fixed_keyfrm_pair.second;
            auto pose = gtsam::Pose3(fixed_keyfrm->get_pose_wc());
            auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << gtsam::Vector3::Constant(1e-3), gtsam::Vector3::Constant(1e-2)).finished());
            graph2.addPrior(gtsam::Symbol('x', id_fixed_keyfrm_pair.first), pose, poseNoise);
        }

        // Clone reprojection factors, replacing Huber with Gaussian for inliers
        for (const auto& nonlinear_factor : graph) {
            const auto& noise_model_factor = boost::dynamic_pointer_cast<gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>>(nonlinear_factor);
            if (!noise_model_factor) {
                continue;
            }

            gtsam::Pose3 pose = result.at<gtsam::Pose3>(noise_model_factor->key1());
            gtsam::Point3 point = result.at<gtsam::Point3>(noise_model_factor->key2());
            gtsam::Values values;
            values.insert(noise_model_factor->key1(), pose);
            values.insert(noise_model_factor->key2(), point);
            const gtsam::Vector b = noise_model_factor->unwhitenedError(values);
            double mahalanobis_distance = std::sqrt(noise_model_factor->noiseModel()->squaredMahalanobisDistance(b));

            bool depth_is_positive = true;
            const auto& projection_factor = boost::dynamic_pointer_cast<internal_gtsam::ProjectionFactorBase<gtsam::Pose3, gtsam::Point3>>(nonlinear_factor);
            if (projection_factor) {
                try {
                    projection_factor->project(pose, point);
                }
                catch (gtsam::CheiralityException& e) {
                    depth_is_positive = false;
                }
                catch (gtsam::StereoCheiralityException& e) {
                    depth_is_positive = false;
                }
            }

            const double chi_sq_threshold = chi_sq_2D;  // monocular uses 2D
            if (chi_sq_threshold < mahalanobis_distance * mahalanobis_distance || !depth_is_positive) {
                continue; // Discard outlier
            }

            // Replace Huber robust kernel with Gaussian noise model
            const auto& robust_model = boost::dynamic_pointer_cast<gtsam::noiseModel::Robust>(noise_model_factor->noiseModel());
            if (robust_model) {
                graph2.add(noise_model_factor->cloneWithNewNoiseModel(robust_model->noise()));
            }
            else {
                graph2.add(nonlinear_factor);
            }
        }

        gtsam::LevenbergMarquardtParams lm_params2;
        lm_params2.setMaxIterations(num_second_iter_);
        gtsam::LevenbergMarquardtOptimizer optimizer2(graph2, initial_estimate2, lm_params2);
        final_result = optimizer2.optimize();
    }

    // 7. Count the outliers

    std::vector<std::pair<std::shared_ptr<data::keyframe>, std::shared_ptr<data::landmark>>> outlier_observations;

    for (const auto& nonlinear_factor : graph) {
        double mahalanobis_distance = -1.0;
        bool depth_is_positive = true;

        // filter prior factor
        const auto& prior_factor = boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Pose3>>(nonlinear_factor);
        if (prior_factor != nullptr) {
            continue;
        }

        // filter reprojection factor
        const auto& noise_model_factor = boost::dynamic_pointer_cast<gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>>(nonlinear_factor);
        if (noise_model_factor != nullptr) {
            gtsam::Pose3 pose = result.at<gtsam::Pose3>(noise_model_factor->key1());
            gtsam::Point3 point = result.at<gtsam::Point3>(noise_model_factor->key2());
            gtsam::Values values;
            values.insert(noise_model_factor->key1(), pose);
            values.insert(noise_model_factor->key2(), point);
            const gtsam::Vector b = noise_model_factor->unwhitenedError(values);
            mahalanobis_distance = std::sqrt(noise_model_factor->noiseModel()->squaredMahalanobisDistance(b));

            const auto& projection_factor = boost::dynamic_pointer_cast<internal_gtsam::ProjectionFactorBase<gtsam::Pose3, gtsam::Point3>>(nonlinear_factor);
            if (projection_factor) {
                try {
                    projection_factor->project(pose, point);
                }
                catch (gtsam::CheiralityException& e) {
                    depth_is_positive = false;
                }
                catch (gtsam::StereoCheiralityException& e) {
                    depth_is_positive = false;
                }
            }
        }

        auto& lm = local_lms.at(gtsam::Symbol(nonlinear_factor->back()).index());
        auto& keyfrm = all_keyfrms.at(gtsam::Symbol(nonlinear_factor->front()).index());
        const double chi_sq_threshold = chi_sq_2D;  // monocular uses 2D
        if (chi_sq_threshold < mahalanobis_distance * mahalanobis_distance || !depth_is_positive) {
            outlier_observations.emplace_back(std::make_pair(keyfrm, lm));
        }
    }

    // 8. Update the information

    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        for (const auto& outlier_obs : outlier_observations) {
            const auto& keyfrm = outlier_obs.first;
            const auto& lm = outlier_obs.second;
            keyfrm->erase_landmark(lm);
            lm->erase_observation(map_db, keyfrm);
            CV_LOG_DEBUG(&g_log_tag, "Erase invalid observation lm=" << lm->id_);
            if (!lm->will_be_erased()) {
                lm->compute_descriptor();
                lm->update_mean_normal_and_obs_scale_variance();
            }
        }

        for (const auto& id_local_keyfrm_pair : local_keyfrms) {
            const auto& local_keyfrm = id_local_keyfrm_pair.second;
            auto pose = final_result.at<gtsam::Pose3>(gtsam::Symbol('x', id_local_keyfrm_pair.first));
            local_keyfrm->set_pose_cw(util::converter::inverse_pose(pose.matrix()));
        }

        for (const auto& id_local_lm_pair : local_lms) {
            const auto& local_lm = id_local_lm_pair.second;
            if (local_lm->will_be_erased()) {
                continue;
            }
            if (!final_result.exists(gtsam::Symbol('l', id_local_lm_pair.first))) {
                continue;
            }

            auto point = final_result.at<gtsam::Point3>(gtsam::Symbol('l', id_local_lm_pair.first));
            local_lm->set_pos_in_world(point);
            local_lm->update_mean_normal_and_obs_scale_variance();
        }
    }
}

} // namespace optimize
} // namespace cv::slam
