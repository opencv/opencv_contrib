#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/marker.hpp"
#include "data/map_database.hpp"
#include "marker_model/base.hpp"
#include "optimize/local_bundle_adjuster_g2o.hpp"
#include "optimize/terminate_action.hpp"
#include "optimize/internal/landmark_vertex_container.hpp"
#include "optimize/internal/marker_vertex_container.hpp"
#include "optimize/internal/se3/shot_vertex_container.hpp"
#include "optimize/internal/se3/reproj_edge_wrapper.hpp"
#include "util/converter.hpp"

#include <unordered_map>

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace optimize {

local_bundle_adjuster_g2o::local_bundle_adjuster_g2o(const YAML::Node& yaml_node,
                                                     const unsigned int num_first_iter,
                                                     const unsigned int num_second_iter)
    : num_first_iter_(num_first_iter), num_second_iter_(num_second_iter),
      use_additional_keyframes_for_monocular_(yaml_node["use_additional_keyframes_for_monocular"].as<bool>(false)) {}

void local_bundle_adjuster_g2o::optimize(data::map_database* map_db,
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

    // Correct markers seen in local keyframes
    std::unordered_map<unsigned int, std::shared_ptr<data::marker>> local_mkrs;

    for (auto local_keyfrm : local_keyfrms) {
        const auto markers = local_keyfrm.second->get_markers();
        for (auto local_mkr : markers) {
            if (!local_mkr) {
                continue;
            }

            // Avoid duplication
            if (local_mkrs.count(local_mkr->id_)) {
                continue;
            }

            local_mkrs[local_mkr->id_] = local_mkr;
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

    std::unique_ptr<g2o::BlockSolverBase> block_solver;
    auto linear_solver = cv::slam::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    block_solver = cv::slam::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    auto terminateAction = new terminate_action;
    terminateAction->setGainThreshold(1e-3);
    optimizer.addPostIterationAction(terminateAction);
    optimizer.setAlgorithm(algorithm);

    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    // 3. Convert each of the keyframe to the g2o vertex, then set it to the optimizer

    // Container of the shot vertices
    auto vtx_id_offset = std::make_shared<unsigned int>(0);
    internal::se3::shot_vertex_container keyfrm_vtx_container(vtx_id_offset, local_keyfrms.size() + fixed_keyfrms.size());
    // Save the converted keyframes
    std::unordered_map<unsigned int, std::shared_ptr<data::keyframe>> all_keyfrms;

    // Set the local keyframes to the optimizer
    for (const auto& id_local_keyfrm_pair : local_keyfrms) {
        const auto& local_keyfrm = id_local_keyfrm_pair.second;

        all_keyfrms.emplace(id_local_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(local_keyfrm, false);
        optimizer.addVertex(keyfrm_vtx);
    }

    // Set the fixed keyframes to the optimizer
    for (const auto& id_fixed_keyfrm_pair : fixed_keyfrms) {
        const auto& fixed_keyfrm = id_fixed_keyfrm_pair.second;

        all_keyfrms.emplace(id_fixed_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(fixed_keyfrm, true);
        optimizer.addVertex(keyfrm_vtx);
    }

    // 4. Connect the vertices of the keyframe and the landmark by using an edge of reprojection constraint

    // Container of the landmark vertices
    internal::landmark_vertex_container lm_vtx_container(vtx_id_offset, local_lms.size());

    // Container of the reprojection edges
    using reproj_edge_wrapper = internal::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(all_keyfrms.size() * local_lms.size());

    // Chi-squared value with significance level of 5%
    // Two degree-of-freedom (n=2)
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // Three degree-of-freedom (n=3)
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (const auto& id_local_lm_pair : local_lms) {
        const auto local_lm = id_local_lm_pair.second;
        const auto observations = local_lm->get_observations();
        if (observations.empty()) {
            CV_LOG_WARNING(&g_log_tag, "empty observation");
            continue;
        }

        // Convert the landmark to the g2o vertex, then set to the optimizer
        auto lm_vtx = lm_vtx_container.create_vertex(local_lm, false);
        optimizer.addVertex(lm_vtx);

        for (const auto& obs : observations) {
            const auto keyfrm = obs.first.lock();
            auto idx = obs.second;
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }
            if (!keyfrm_vtx_container.contain(keyfrm)) {
                continue;
            }

            const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
            const auto& undist_keypt = keyfrm->frm_obs_.undist_keypts_.at(idx);
            const float x_right = keyfrm->frm_obs_.stereo_x_right_.empty() ? -1.0f : keyfrm->frm_obs_.stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->orb_params_->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, local_lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);
        }
    }

    // Container of the reprojection edges for corners of markers
    internal::marker_vertex_container marker_vtx_container(vtx_id_offset, local_mkrs.size());
    std::vector<reproj_edge_wrapper> mkr_reproj_edge_wraps;
    mkr_reproj_edge_wraps.reserve(all_keyfrms.size() * local_mkrs.size());
    std::unordered_set<unsigned int> mkr_has_vtx;

    for (auto& id_local_mkr_pair : local_mkrs) {
        auto mkr = id_local_mkr_pair.second;
        if (!mkr) {
            continue;
        }

        // Use marker only if it was initialized before (or is fixed)
        if (!mkr->keep_fixed_ && !mkr->initialized_before_)
            continue;

        // Indicate that a vertex container will be available, this can be
        // used to guard against a race condition, should initialized_before_
        // get changed in another thread
        mkr_has_vtx.insert(mkr->id_);

        // Convert the corners to the g2o vertex, then set it to the optimizer
        auto corner_vertices = marker_vtx_container.create_vertices(mkr, mkr->keep_fixed_);
        for (unsigned int corner_idx = 0; corner_idx < corner_vertices.size(); ++corner_idx) {
            const auto corner_vtx = corner_vertices[corner_idx];
            optimizer.addVertex(corner_vtx);

            for (const auto& id_keyfrm : mkr->observations_) {
                const auto& keyfrm = id_keyfrm.second;
                if (!keyfrm) {
                    continue;
                }
                if (keyfrm->will_be_erased()) {
                    continue;
                }
                if (!keyfrm_vtx_container.contain(keyfrm)) {
                    continue;
                }

                const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
                const auto& mkr_2d = keyfrm->markers_2d_.at(mkr->id_);
                const auto& undist_pt = mkr_2d.undist_corners_.at(corner_idx);
                const float x_right = -1.0;
                const float inv_sigma_sq = 1.0;

                const auto sqrt_chi_sq = (mkr->keep_fixed_) ? 0.0 : sqrt_chi_sq_2D;

                auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, nullptr, corner_vtx,
                                                            0, undist_pt.x, undist_pt.y, x_right,
                                                            inv_sigma_sq, sqrt_chi_sq, false);
                mkr_reproj_edge_wraps.push_back(reproj_edge_wrap);
                optimizer.addEdge(reproj_edge_wrap.edge_);
            }
        }
    }

    // 5. Perform the first optimization

    if (force_stop_flag && *force_stop_flag) {
        return;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_first_iter_);

    // 6. Discard outliers, then perform the second optimization

    bool run_robust_BA = true;

    if (force_stop_flag && *force_stop_flag) {
        run_robust_BA = false;
    }

    if (run_robust_BA) {
        for (auto& reproj_edge_wrap : reproj_edge_wraps) {
            auto edge = reproj_edge_wrap.edge_;

            const auto& local_lm = reproj_edge_wrap.lm_;
            if (local_lm->will_be_erased()) {
                continue;
            }

            if (reproj_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }

            edge->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(num_second_iter_);
    }

    delete terminateAction;

    // 7. Count the outliers

    std::vector<std::pair<std::shared_ptr<data::keyframe>, std::shared_ptr<data::landmark>>> outlier_observations;
    outlier_observations.reserve(reproj_edge_wraps.size());

    for (auto& reproj_edge_wrap : reproj_edge_wraps) {
        auto edge = reproj_edge_wrap.edge_;

        const auto& local_lm = reproj_edge_wrap.lm_;
        if (local_lm->will_be_erased()) {
            continue;
        }

        if (reproj_edge_wrap.is_monocular_) {
            if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
        else {
            if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
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
            if (!lm->will_be_erased()) {
                lm->compute_descriptor();
                lm->update_mean_normal_and_obs_scale_variance();
            }
        }

        for (const auto& id_local_keyfrm_pair : local_keyfrms) {
            const auto& local_keyfrm = id_local_keyfrm_pair.second;

            auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(local_keyfrm);
            local_keyfrm->set_pose_cw(util::converter::to_eigen_mat(keyfrm_vtx->estimate()));
        }

        for (const auto& id_local_lm_pair : local_lms) {
            const auto& local_lm = id_local_lm_pair.second;
            if (local_lm->will_be_erased()) {
                continue;
            }

            auto lm_vtx = lm_vtx_container.get_vertex(local_lm);
            local_lm->set_pos_in_world(lm_vtx->estimate());
            local_lm->update_mean_normal_and_obs_scale_variance();
        }

        // Also update the marker positions
        for (auto& id_mkr_pair : local_mkrs) {
            auto& mkr = id_mkr_pair.second;
            if (mkr->keep_fixed_) {
                continue;
            }
            if (!mkr->initialized_before_) {
                continue;
            }
            // It's possible that initialized_before_ got changed since it was
            // checked the last time, this actually tests if a vtx was made
            if (mkr_has_vtx.find(mkr->id_) == mkr_has_vtx.end()) {
                continue;
            }

            for (size_t corner_idx = 0; corner_idx < 4; corner_idx++) {
                mkr->corners_pos_w_[corner_idx] = marker_vtx_container.get_vertex(mkr, corner_idx)->estimate();
            }
        }
    }
}

} // namespace optimize
} // namespace cv::slam
