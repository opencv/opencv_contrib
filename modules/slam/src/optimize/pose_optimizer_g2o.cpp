#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "optimize/pose_optimizer_g2o.hpp"
#include "optimize/terminate_action.hpp"
#include "optimize/internal/se3/pose_opt_edge_wrapper.hpp"
#include "util/converter.hpp"

#include <vector>
#include <mutex>

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace cv::slam {
namespace optimize {

pose_optimizer_g2o::pose_optimizer_g2o(const unsigned int num_trials_robust, const unsigned int num_trials, const unsigned int num_each_iter)
    : num_trials_robust_(num_trials_robust), num_trials_(num_trials), num_each_iter_(num_each_iter) {}

unsigned int pose_optimizer_g2o::optimize(const data::frame& frm, Mat44_t& optimized_pose, std::vector<bool>& outlier_flags) const {
    auto num_valid_obs = optimize(frm.get_pose_cw(), frm.frm_obs_, frm.orb_params_, frm.camera_,
                                  frm.get_landmarks(), optimized_pose, outlier_flags);
    return num_valid_obs;
}

unsigned int pose_optimizer_g2o::optimize(const data::keyframe* keyfrm, Mat44_t& optimized_pose, std::vector<bool>& outlier_flags) const {
    auto num_valid_obs = optimize(keyfrm->get_pose_cw(), keyfrm->frm_obs_, keyfrm->orb_params_, keyfrm->camera_,
                                  keyfrm->get_landmarks(), optimized_pose, outlier_flags);
    return num_valid_obs;
}

unsigned int pose_optimizer_g2o::optimize(const Mat44_t& cam_pose_cw, const data::frame_observation& frm_obs,
                                          const feature::orb_params* orb_params,
                                          const camera::base* camera,
                                          const std::vector<std::shared_ptr<data::landmark>>& landmarks,
                                          Mat44_t& optimized_pose,
                                          std::vector<bool>& outlier_flags) const {
    // 1. Construct an optimizer

    auto linear_solver = cv::slam::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = cv::slam::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    auto terminateAction = new terminate_action;
    terminateAction->setGainThreshold(1e-3);
    optimizer.addPostIterationAction(terminateAction);
    optimizer.setAlgorithm(algorithm);

    unsigned int num_init_obs = 0;

    // 2. Convert the frame to the g2o vertex, then set it to the optimizer

    auto frm_vtx = new internal::se3::shot_vertex();
    frm_vtx->setId(0);
    frm_vtx->setEstimate(util::converter::to_g2o_SE3(cam_pose_cw));
    frm_vtx->setFixed(false);
    optimizer.addVertex(frm_vtx);

    const unsigned int num_keypts = frm_obs.undist_keypts_.size();
    outlier_flags.resize(num_keypts);
    std::fill(outlier_flags.begin(), outlier_flags.end(), false);

    // 3. Connect the landmark vertices by using projection edges

    // Container of the reprojection edges
    using pose_opt_edge_wrapper = internal::se3::pose_opt_edge_wrapper;
    std::vector<pose_opt_edge_wrapper> pose_opt_edge_wraps;
    pose_opt_edge_wraps.reserve(num_keypts);

    // Chi-squared value with significance level of 5%
    // Two degree-of-freedom (n=2)
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // Three degree-of-freedom (n=3)
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

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
        const float inv_sigma_sq = orb_params->inv_level_sigma_sq_.at(undist_keypt.octave);
        const auto sqrt_chi_sq = (camera->setup_type_ == camera::setup_type_t::Monocular)
                                     ? sqrt_chi_sq_2D
                                     : sqrt_chi_sq_3D;
        auto pose_opt_edge_wrap = pose_opt_edge_wrapper(camera, frm_vtx, lm->get_pos_in_world(),
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
        pose_opt_edge_wraps.push_back(pose_opt_edge_wrap);
        optimizer.addEdge(pose_opt_edge_wrap.edge_);
    }

    if (num_init_obs < 5) {
        return 0;
    }

    // 4. Perform robust Bundle Adjustment (BA)

    unsigned int num_bad_obs = 0;
    if (num_trials_robust_ == 0) {
        for (auto& pose_opt_edge_wrap : pose_opt_edge_wraps) {
            pose_opt_edge_wrap.edge_->setRobustKernel(nullptr);
        }
    }
    for (unsigned int trial = 0; trial < num_trials_robust_ + num_trials_; ++trial) {
        optimizer.initializeOptimization();
        optimizer.optimize(num_each_iter_);

        num_bad_obs = 0;

        for (auto& pose_opt_edge_wrap : pose_opt_edge_wraps) {
            auto edge = pose_opt_edge_wrap.edge_;

            if (outlier_flags.at(pose_opt_edge_wrap.idx_)) {
                edge->computeError();
            }

            if (pose_opt_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2()) {
                    outlier_flags.at(pose_opt_edge_wrap.idx_) = true;
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    outlier_flags.at(pose_opt_edge_wrap.idx_) = false;
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2()) {
                    outlier_flags.at(pose_opt_edge_wrap.idx_) = true;
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    outlier_flags.at(pose_opt_edge_wrap.idx_) = false;
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }

            if (num_trials_ != 0 && trial + 1 == num_trials_robust_) {
                edge->setRobustKernel(nullptr);
            }
        }

        if (num_init_obs - num_bad_obs < 5) {
            break;
        }
    }

    delete terminateAction;

    // 5. Update the information

    optimized_pose = util::converter::to_eigen_mat(frm_vtx->estimate());

    return num_init_obs - num_bad_obs;
}

} // namespace optimize
} // namespace cv::slam
