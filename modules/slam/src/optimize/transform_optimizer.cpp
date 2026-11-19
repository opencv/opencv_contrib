#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "optimize/transform_optimizer.hpp"
#include "optimize/internal/sim3/transform_vertex.hpp"
#include "optimize/internal/sim3/mutual_reproj_edge_wrapper.hpp"

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace cv::slam {
namespace optimize {

transform_optimizer::transform_optimizer(const bool fix_scale, const unsigned int num_iter)
    : fix_scale_(fix_scale), num_iter_(num_iter) {}

unsigned int transform_optimizer::optimize(const std::shared_ptr<data::keyframe>& keyfrm_1, const std::shared_ptr<data::keyframe>& keyfrm_2,
                                           std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_keyfrm_2,
                                           ::g2o::Sim3& g2o_Sim3_12, const float chi_sq) const {
    const float sqrt_chi_sq = std::sqrt(chi_sq);

    // 1. Construct an optimizer

    auto linear_solver = cv::slam::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto block_solver = cv::slam::make_unique<g2o::BlockSolverX>(std::move(linear_solver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    // 2. Create a Sim3 transformation vertex

    auto Sim3_12_vtx = new internal::sim3::transform_vertex();
    Sim3_12_vtx->setId(0);
    Sim3_12_vtx->setEstimate(g2o_Sim3_12);
    Sim3_12_vtx->setFixed(false);
    Sim3_12_vtx->fix_scale_ = fix_scale_;
    Sim3_12_vtx->rot_1w_ = keyfrm_1->get_rot_cw();
    Sim3_12_vtx->trans_1w_ = keyfrm_1->get_trans_cw();
    Sim3_12_vtx->rot_2w_ = keyfrm_2->get_rot_cw();
    Sim3_12_vtx->trans_2w_ = keyfrm_2->get_trans_cw();
    optimizer.addVertex(Sim3_12_vtx);

    // 3. Add landmarks and constraints

    // Wrapper including the following two edges:
    // - backward: an edge reprojecting 3D point observed keyframe 2 to keyframe 1 (using the camera model of keyframe 1)
    // - forward: an edge reprojecting 3D point observed keyframe 1 to keyframe 2 (using the camera model of keyframe 2)
    using reproj_edge_wrapper = internal::sim3::mutual_reproj_edge_wapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> mutual_edges;
    // The number of matches
    const unsigned int num_matches = matched_lms_in_keyfrm_2.size();
    mutual_edges.reserve(num_matches);

    // All the 3D points observed in keyframe 1
    const auto lms_in_keyfrm_1 = keyfrm_1->get_landmarks();

    // The number of valid matches
    unsigned int num_valid_matches = 0;

    for (unsigned int idx1 = 0; idx1 < num_matches; ++idx1) {
        // Only if matching information exists
        if (!matched_lms_in_keyfrm_2.at(idx1)) {
            continue;
        }

        const auto& lm_1 = lms_in_keyfrm_1.at(idx1);
        const auto& lm_2 = matched_lms_in_keyfrm_2.at(idx1);

        // Only if both of the 3D points are valid
        if (!lm_1 || !lm_2) {
            continue;
        }
        if (lm_1->will_be_erased() || lm_2->will_be_erased()) {
            continue;
        }

        const auto idx2 = lm_2->get_index_in_keyframe(keyfrm_2);

        if (idx2 < 0) {
            continue;
        }

        // Create forward/backward edges, then set them to the optimizer
        reproj_edge_wrapper mutual_edge(keyfrm_1, idx1, lm_1, keyfrm_2, idx2, lm_2, Sim3_12_vtx, sqrt_chi_sq);
        optimizer.addEdge(mutual_edge.edge_12_);
        optimizer.addEdge(mutual_edge.edge_21_);

        ++num_valid_matches;
        mutual_edges.push_back(mutual_edge);
    }

    // 3. Perform optimization

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // 4. Reject outliers

    unsigned int num_outliers = 0;
    for (unsigned int i = 0; i < num_valid_matches; ++i) {
        auto edge_12 = mutual_edges.at(i).edge_12_;
        auto edge_21 = mutual_edges.at(i).edge_21_;

        // Inlier check
        if (edge_12->chi2() < chi_sq && edge_21->chi2() < chi_sq) {
            continue;
        }

        // Outlier rejection
        const auto idx1 = mutual_edges.at(i).idx1_;
        matched_lms_in_keyfrm_2.at(idx1) = nullptr;

        mutual_edges.at(i).set_as_outlier();
        ++num_outliers;
    }

    if (num_valid_matches - num_outliers < 10) {
        return 0;
    }

    // 5. Perform optimization again

    optimizer.initializeOptimization();
    optimizer.optimize(num_iter_);

    // 6. Count the inliers

    unsigned int num_inliers = 0;
    for (unsigned int i = 0; i < num_valid_matches; ++i) {
        auto edge_12 = mutual_edges.at(i).edge_12_;
        auto edge_21 = mutual_edges.at(i).edge_21_;

        // Outlier check
        if (mutual_edges.at(i).is_outlier()) {
            continue;
        }

        // Outlier check
        if (chi_sq < edge_12->chi2() || chi_sq < edge_21->chi2()) {
            // Outlier rejection
            unsigned int idx1 = mutual_edges.at(i).idx1_;
            matched_lms_in_keyfrm_2.at(idx1) = nullptr;
            continue;
        }

        ++num_inliers;
    }

    // 7. Set the result

    g2o_Sim3_12 = Sim3_12_vtx->estimate();

    return num_inliers;
}

} // namespace optimize
} // namespace cv::slam
