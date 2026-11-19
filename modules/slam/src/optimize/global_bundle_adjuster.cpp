#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/marker.hpp"
#include "data/map_database.hpp"
#include "marker_model/base.hpp"
#include "optimize/global_bundle_adjuster.hpp"
#include "optimize/terminate_action.hpp"
#include "optimize/internal/landmark_vertex_container.hpp"
#include "optimize/internal/marker_vertex_container.hpp"
#include "optimize/internal/se3/shot_vertex_container.hpp"
#include "optimize/internal/se3/reproj_edge_wrapper.hpp"
#include "util/converter.hpp"

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace cv::slam {
namespace optimize {

void optimize_impl(g2o::SparseOptimizer& optimizer,
                   const std::vector<std::shared_ptr<data::keyframe>>& keyfrms,
                   const std::vector<std::shared_ptr<data::landmark>>& lms,
                   const std::vector<std::shared_ptr<data::marker>>& markers,
                   std::vector<bool>& is_optimized_lm,
                   internal::se3::shot_vertex_container& keyfrm_vtx_container,
                   internal::landmark_vertex_container& lm_vtx_container,
                   internal::marker_vertex_container& marker_vtx_container,
                   std::unordered_set<unsigned int>& mkr_has_vtx,
                   unsigned int num_iter,
                   bool use_huber_kernel,
                   bool fix_markers,
                   bool verbose,
                   bool* const force_stop_flag) {
    // 2. Construct an optimizer

    std::unique_ptr<g2o::BlockSolverBase> block_solver;
    auto linear_solver = cv::slam::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>>();
    block_solver = cv::slam::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    optimizer.setAlgorithm(algorithm);

    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    // 3. Convert each of the keyframe to the g2o vertex, then set it to the optimizer

    // Set the keyframes to the optimizer
    for (const auto& keyfrm : keyfrms) {
        if (!keyfrm) {
            continue;
        }
        if (keyfrm->will_be_erased()) {
            continue;
        }

        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(keyfrm, keyfrm->graph_node_->is_spanning_root());
        optimizer.addVertex(keyfrm_vtx);
    }

    // 4. Connect the vertices of the keyframe and the landmark by using reprojection edge

    // Container of the reprojection edges
    using reproj_edge_wrapper = internal::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(10 * lms.size());

    // Chi-squared value with significance level of 5%
    // Two degree-of-freedom (n=2)
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // Three degree-of-freedom (n=3)
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (unsigned int i = 0; i < lms.size(); ++i) {
        const auto& lm = lms.at(i);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        // Convert the landmark to the g2o vertex, then set it to the optimizer
        auto lm_vtx = lm_vtx_container.create_vertex(lm, false);
        optimizer.addVertex(lm_vtx);

        unsigned int num_edges = 0;
        const auto observations = lm->get_observations();
        for (const auto& obs : observations) {
            auto keyfrm = obs.first.lock();
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
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq, use_huber_kernel);
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);
            ++num_edges;
        }

        if (num_edges == 0) {
            optimizer.removeVertex(lm_vtx);
            is_optimized_lm.at(i) = false;
        }
    }

    // Connect marker vertices
    for (unsigned int marker_idx = 0; marker_idx < markers.size(); ++marker_idx) {
        auto mkr = markers.at(marker_idx);
        if (!mkr) {
            continue;
        }

        // Only optimize if requested, and if enough keyframes were available to
        // initialize it correctly
        if (!fix_markers && !mkr->keep_fixed_ && !mkr->initialized_before_)
            continue;

        // Indicate that a vertex container will be available, this can be
        // used to guard against a race condition, should initialized_before_
        // get changed in another thread
        mkr_has_vtx.insert(mkr->id_);

        // Convert the corners to the g2o vertex, then set it to the optimizer
        auto corner_vertices = marker_vtx_container.create_vertices(mkr, fix_markers || mkr->keep_fixed_);
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

                const auto sqrt_chi_sq = (fix_markers || mkr->keep_fixed_) ? 0.0 : sqrt_chi_sq_2D;

                auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, nullptr, corner_vtx,
                                                            0, undist_pt.x, undist_pt.y, x_right,
                                                            inv_sigma_sq, sqrt_chi_sq, false);
                reproj_edge_wraps.push_back(reproj_edge_wrap);
                optimizer.addEdge(reproj_edge_wrap.edge_);
            }
        }
    }

    // 5. Perform optimization

    optimizer.initializeOptimization();
    optimizer.setVerbose(verbose);
    optimizer.optimize(num_iter);

    if (force_stop_flag && *force_stop_flag) {
        return;
    }
}

global_bundle_adjuster::global_bundle_adjuster(
    const unsigned int num_iter,
    const bool use_huber_kernel,
    const bool verbose)
    : num_iter_(num_iter),
      use_huber_kernel_(use_huber_kernel),
      verbose_(verbose) {}

void global_bundle_adjuster::optimize_for_initialization(const std::vector<std::shared_ptr<data::keyframe>>& keyfrms,
                                                         const std::vector<std::shared_ptr<data::landmark>>& lms,
                                                         const std::vector<std::shared_ptr<data::marker>>& markers,
                                                         float gain_threshold,
                                                         bool fix_markers,
                                                         bool* const force_stop_flag) const {
    std::vector<bool> is_optimized_lm(lms.size(), true);

    auto vtx_id_offset = std::make_shared<unsigned int>(0);
    // Container of the shot vertices
    internal::se3::shot_vertex_container keyfrm_vtx_container(vtx_id_offset, keyfrms.size());
    // Container of the landmark vertices
    internal::landmark_vertex_container lm_vtx_container(vtx_id_offset, lms.size());
    // Container of the landmark vertices
    internal::marker_vertex_container marker_vtx_container(vtx_id_offset, markers.size());
    std::unordered_set<unsigned int> mkr_has_vtx;

    g2o::SparseOptimizer optimizer;
    auto terminateAction = new terminate_action;
    terminateAction->setGainThreshold(gain_threshold);
    optimizer.addPostIterationAction(terminateAction);

    optimize_impl(optimizer, keyfrms, lms, markers, is_optimized_lm, keyfrm_vtx_container, lm_vtx_container, marker_vtx_container,
                  mkr_has_vtx, num_iter_, use_huber_kernel_, fix_markers, verbose_, force_stop_flag);

    if (force_stop_flag && *force_stop_flag) {
        return;
    }

    // Extract the result

    for (auto keyfrm : keyfrms) {
        if (keyfrm->will_be_erased()) {
            continue;
        }
        auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
        const auto cam_pose_cw = util::converter::to_eigen_mat(keyfrm_vtx->estimate());

        keyfrm->set_pose_cw(cam_pose_cw);
    }

    for (unsigned int i = 0; i < lms.size(); ++i) {
        if (!is_optimized_lm.at(i)) {
            continue;
        }

        const auto& lm = lms.at(i);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        auto lm_vtx = lm_vtx_container.get_vertex(lm);
        const Vec3_t pos_w = lm_vtx->estimate();

        lm->set_pos_in_world(pos_w);
        lm->update_mean_normal_and_obs_scale_variance();
    }

    for (auto& mkr : markers) {
        if (fix_markers || mkr->keep_fixed_) // No need to update
            continue;
        if (!mkr->initialized_before_)
            continue;
        // It's possible that initialized_before_ got changed since it was
        // checked the last time, this actually tests if a vtx was made
        if (mkr_has_vtx.find(mkr->id_) == mkr_has_vtx.end())
            continue;

        for (size_t corner_idx = 0; corner_idx < 4; corner_idx++) {
            mkr->corners_pos_w_[corner_idx] = marker_vtx_container.get_vertex(mkr, corner_idx)->estimate();
        }
    }
}

bool global_bundle_adjuster::optimize(const std::vector<std::shared_ptr<data::keyframe>>& keyfrms,
                                      std::unordered_set<unsigned int>& optimized_keyfrm_ids,
                                      std::unordered_set<unsigned int>& optimized_landmark_ids,
                                      std::unordered_set<unsigned int>& optimized_marker_ids,
                                      eigen_alloc_unord_map<unsigned int, Vec3_t>& lm_to_pos_w_after_global_BA,
                                      eigen_alloc_unord_map<unsigned int, Mat44_t>& keyfrm_to_pose_cw_after_global_BA,
                                      eigen_alloc_unord_map<unsigned int, std::array<Vec3_t, 4>>& marker_to_pos_w_after_global_BA,
                                      bool* const force_stop_flag) const {
    std::unordered_set<unsigned int> already_found_landmark_ids;
    std::vector<std::shared_ptr<data::landmark>> lms;
    for (const auto& keyfrm : keyfrms) {
        for (const auto& lm : keyfrm->get_landmarks()) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }
            if (already_found_landmark_ids.count(lm->id_)) {
                continue;
            }

            already_found_landmark_ids.insert(lm->id_);
            lms.push_back(lm);
        }
    }

    std::unordered_set<unsigned int> already_found_marker_ids;
    std::vector<std::shared_ptr<data::marker>> markers;
    for (const auto& keyfrm : keyfrms) {
        for (const auto& mkr : keyfrm->get_markers()) {
            if (!mkr) {
                continue;
            }
            if (already_found_marker_ids.count(mkr->id_)) {
                continue;
            }

            already_found_marker_ids.insert(mkr->id_);
            markers.push_back(mkr);
        }
    }

    std::vector<bool> is_optimized_lm(lms.size(), true);

    auto vtx_id_offset = std::make_shared<unsigned int>(0);
    // Container of the shot vertices
    internal::se3::shot_vertex_container keyfrm_vtx_container(vtx_id_offset, keyfrms.size());
    // Container of the landmark vertices
    internal::landmark_vertex_container lm_vtx_container(vtx_id_offset, lms.size());
    // Container of the landmark vertices
    internal::marker_vertex_container marker_vtx_container(vtx_id_offset, markers.size());
    std::unordered_set<unsigned int> mkr_has_vtx;

    g2o::SparseOptimizer optimizer;
    auto terminateAction = new terminate_action;
    terminateAction->setGainThreshold(1e-3);
    optimizer.addPostIterationAction(terminateAction);

    optimize_impl(optimizer, keyfrms, lms, markers, is_optimized_lm, keyfrm_vtx_container, lm_vtx_container, marker_vtx_container,
                  mkr_has_vtx, num_iter_, use_huber_kernel_, false, verbose_, force_stop_flag);

    if (force_stop_flag && *force_stop_flag && !terminateAction->stopped_by_terminate_action_) {
        return false;
    }

    delete terminateAction;

    // Extract the result

    for (auto keyfrm : keyfrms) {
        if (keyfrm->will_be_erased()) {
            continue;
        }
        auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
        const auto cam_pose_cw = util::converter::to_eigen_mat(keyfrm_vtx->estimate());

        keyfrm_to_pose_cw_after_global_BA[keyfrm->id_] = cam_pose_cw;
        optimized_keyfrm_ids.insert(keyfrm->id_);
    }

    for (unsigned int i = 0; i < lms.size(); ++i) {
        if (!is_optimized_lm.at(i)) {
            continue;
        }

        const auto& lm = lms.at(i);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        auto lm_vtx = lm_vtx_container.get_vertex(lm);
        const Vec3_t pos_w = lm_vtx->estimate();

        lm_to_pos_w_after_global_BA[lm->id_] = pos_w;
        optimized_landmark_ids.insert(lm->id_);
    }

    for (auto& mkr : markers) {
        if (mkr->keep_fixed_) // No need to update
            continue;
        if (!mkr->initialized_before_)
            continue;
        // It's possible that initialized_before_ got changed since it was
        // checked the last time, this actually tests if a vtx was made
        if (mkr_has_vtx.find(mkr->id_) == mkr_has_vtx.end())
            continue;

        bool changed = false;
        std::array<Vec3_t, 4> new_pos_corners;
        for (size_t corner_idx = 0; corner_idx < 4; corner_idx++) {
            auto orig_pos = mkr->corners_pos_w_[corner_idx];
            auto vtx = marker_vtx_container.get_vertex(mkr, corner_idx);
            auto new_pos = vtx->estimate();

            if (orig_pos != new_pos) {
                changed = true;
            }

            new_pos_corners[corner_idx] = new_pos;
        }

        if (!changed)
            continue;

        optimized_marker_ids.insert(mkr->id_);
        marker_to_pos_w_after_global_BA[mkr->id_] = new_pos_corners;
    }

    return true;
}

} // namespace optimize
} // namespace cv::slam
