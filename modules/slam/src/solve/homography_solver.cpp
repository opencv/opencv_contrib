#include "solve/common.hpp"
#include "solve/homography_solver.hpp"
#include "util/converter.hpp"
#include "util/random_array.hpp"
#include "util/trigonometric.hpp"

namespace cv::slam {
namespace solve {

homography_solver::homography_solver(const std::vector<cv::KeyPoint>& undist_keypts_1, const std::vector<cv::KeyPoint>& undist_keypts_2,
                                     const std::vector<std::pair<int, int>>& matches_12, const float sigma, bool use_fixed_seed)
    : undist_keypts_1_(undist_keypts_1), undist_keypts_2_(undist_keypts_2), matches_12_(matches_12), sigma_(sigma),
      random_engine_(util::create_random_engine(use_fixed_seed)) {}

void homography_solver::find_via_ransac(const unsigned int max_num_iter, const bool recompute) {
    const auto num_matches = static_cast<unsigned int>(matches_12_.size());

    // 0. Normalize keypoint coordinates

    // apply normalization
    std::vector<cv::Point2f> normalized_keypts_1, normalized_keypts_2;
    Mat33_t transform_1, transform_2;
    normalize(undist_keypts_1_, normalized_keypts_1, transform_1);
    normalize(undist_keypts_2_, normalized_keypts_2, transform_2);

    const Mat33_t transform_2_inv = transform_2.inverse();

    // 1. Prepare for RANSAC

    // minimum number of samples (= 4), but we will require more for robustness
    constexpr unsigned int min_set_size = 4;
    if (num_matches < min_set_size * 2) {
        solution_is_valid_ = false;
        return;
    }

    // RANSAC variables
    best_cost_ = std::numeric_limits<float>::max();
    is_inlier_match_ = std::vector<bool>(num_matches, false);

    // minimum set of keypoint matches
    std::vector<cv::Point2f> min_set_keypts_1(min_set_size);
    std::vector<cv::Point2f> min_set_keypts_2(min_set_size);

    // shared variables in RANSAC loop
    // homography matrix from shot 1 to shot 2, and
    // homography matrix from shot 2 to shot 1,
    Mat33_t H_21_in_sac, H_12_in_sac;
    // inlier/outlier flags
    std::vector<bool> is_inlier_match_in_sac(num_matches, false);

    // 2. RANSAC loop

    for (unsigned int iter = 0; iter < max_num_iter; ++iter) {
        // 2-1. Create a minimum set
        const auto indices = util::create_random_array(min_set_size, 0U, num_matches - 1, random_engine_);
        for (unsigned int i = 0; i < min_set_size; ++i) {
            const auto idx = indices.at(i);
            min_set_keypts_1.at(i) = normalized_keypts_1.at(matches_12_.at(idx).first);
            min_set_keypts_2.at(i) = normalized_keypts_2.at(matches_12_.at(idx).second);
        }

        // 2-2. Compute a homography matrix
        Mat33_t normalized_H_21;
        const bool sample_is_not_degenerate = compute_H_21(min_set_keypts_1, min_set_keypts_2, normalized_H_21);
        if (!sample_is_not_degenerate) {
            continue;
        }
        H_21_in_sac = transform_2_inv * normalized_H_21 * transform_1;

        // 2-3. Check inliers and compute a score
        float cost_in_sac;
        unsigned int num_inliers = check_inliers(H_21_in_sac, is_inlier_match_in_sac, cost_in_sac);

        // 2-4. Update the best model
        if (num_inliers > min_set_size && best_cost_ > cost_in_sac) {
            best_cost_ = cost_in_sac;
            best_H_21_ = H_21_in_sac;
            is_inlier_match_ = is_inlier_match_in_sac;
        }
    }

    solution_is_valid_ = best_cost_ < std::numeric_limits<float>::max();

    if (!recompute || !solution_is_valid_) {
        return;
    }

    // 3. Recompute a homography matrix only with the inlier matches

    std::vector<cv::Point2f> inlier_normalized_keypts_1;
    std::vector<cv::Point2f> inlier_normalized_keypts_2;
    inlier_normalized_keypts_1.reserve(matches_12_.size());
    inlier_normalized_keypts_2.reserve(matches_12_.size());
    for (unsigned int i = 0; i < matches_12_.size(); ++i) {
        if (is_inlier_match_.at(i)) {
            inlier_normalized_keypts_1.push_back(normalized_keypts_1.at(matches_12_.at(i).first));
            inlier_normalized_keypts_2.push_back(normalized_keypts_2.at(matches_12_.at(i).second));
        }
    }

    Mat33_t normalized_H_21;
    bool refinement_success = solve::homography_solver::compute_H_21(inlier_normalized_keypts_1, inlier_normalized_keypts_2, normalized_H_21);
    if (refinement_success) {
        best_H_21_ = transform_2_inv * normalized_H_21 * transform_1;
        check_inliers(best_H_21_, is_inlier_match_, best_cost_);
    }
}

bool homography_solver::compute_H_21(const std::vector<cv::Point2f>& keypts_1, const std::vector<cv::Point2f>& keypts_2, Mat33_t& H_21) {
    // https://www.uio.no/studier/emner/matnat/its/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf

    assert(keypts_1.size() == keypts_2.size());

    const auto num_points = keypts_1.size();

    typedef Eigen::Matrix<Mat33_t::Scalar, Eigen::Dynamic, 9> CoeffMatrix;
    CoeffMatrix A(2 * num_points, 9);

    for (unsigned int i = 0; i < num_points; ++i) {
        A.block<1, 3>(2 * i, 0) = Vec3_t::Zero();
        A.block<1, 3>(2 * i, 3) = -util::converter::to_homogeneous(keypts_1.at(i));
        A.block<1, 3>(2 * i, 6) = keypts_2.at(i).y * util::converter::to_homogeneous(keypts_1.at(i));
        A.block<1, 3>(2 * i + 1, 0) = util::converter::to_homogeneous(keypts_1.at(i));
        A.block<1, 3>(2 * i + 1, 3) = Vec3_t::Zero();
        A.block<1, 3>(2 * i + 1, 6) = -keypts_2.at(i).x * util::converter::to_homogeneous(keypts_1.at(i));
    }

    const Eigen::JacobiSVD<CoeffMatrix> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // check if A is degenerate (this can happen if we picked too many collinear points)
    if (svd.rank() < 8) {
        return false;
    }

    const Eigen::Matrix<Mat33_t::Scalar, 9, 1> v = svd.matrixV().col(8);
    // need transpose() because elements are contained as col-major after it was constructed from a pointer
    H_21 = Mat33_t(v.data()).transpose();
    return true;
}

bool homography_solver::decompose(const Mat33_t& H_21, const Mat33_t& cam_matrix_1, const Mat33_t& cam_matrix_2,
                                  eigen_alloc_vector<Mat33_t>& init_rots, eigen_alloc_vector<Vec3_t>& init_transes, eigen_alloc_vector<Vec3_t>& init_normals) {
    // Motion and structure from motion in a piecewise planar environment
    // (Faugeras et al. in IJPRAI 1988)

    init_rots.reserve(8);
    init_transes.reserve(8);
    init_normals.reserve(8);

    const Mat33_t A = cam_matrix_2.inverse() * H_21 * cam_matrix_1;

    // Eq.(7) SVD
    const Eigen::JacobiSVD<Mat33_t> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Mat33_t& U = svd.matrixU();
    const Vec3_t& lambda = svd.singularValues();
    const Mat33_t& V = svd.matrixV();
    const Mat33_t Vt = V.transpose();

    const float d1 = lambda(0);
    const float d2 = lambda(1);
    const float d3 = lambda(2);

    // check rank condition
    if (d1 / d2 < 1.0001 || d2 / d3 < 1.0001) {
        return false;
    }

    // intermediate variable in Eq.(8)
    const float s = U.determinant() * Vt.determinant();

    // x1 and x3 in Eq.(12)
    const float aux_1 = std::sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
    const float aux_3 = std::sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
    const std::array<float, 4> x1s = {{aux_1, aux_1, -aux_1, -aux_1}};
    const std::array<float, 4> x3s = {{aux_3, -aux_3, aux_3, -aux_3}};

    // when d' > 0

    // Eq.(13)
    const float aux_sin_theta = std::sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);
    const float cos_theta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    const std::array<float, 4> aux_sin_thetas = {{aux_sin_theta, -aux_sin_theta, -aux_sin_theta, aux_sin_theta}};

    for (unsigned int i = 0; i < 4; ++i) {
        // Eq.(13)
        Mat33_t aux_rot = Mat33_t::Identity();
        aux_rot(0, 0) = cos_theta;
        aux_rot(0, 2) = -aux_sin_thetas.at(i);
        aux_rot(2, 0) = aux_sin_thetas.at(i);
        aux_rot(2, 2) = cos_theta;
        // Eq.(8)
        const Mat33_t init_rot = s * U * aux_rot * Vt;
        init_rots.push_back(init_rot);

        // Eq.(14)
        Vec3_t aux_trans{x1s.at(i), 0.0, -x3s.at(i)};
        aux_trans *= d1 - d3;
        // Eq.(8)
        const Vec3_t init_trans = U * aux_trans;
        init_transes.emplace_back(init_trans / init_trans.norm());

        // Eq.(9)
        const Vec3_t aux_normal{x1s.at(i), 0.0, x3s.at(i)};
        // Eq.(8)
        Vec3_t init_normal = V * aux_normal;
        if (init_normal(2) < 0) {
            init_normal = -init_normal;
        }
        init_normals.push_back(init_normal);
    }

    // when d' < 0

    // Eq.(13)
    const float aux_sin_phi = std::sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);
    const float cos_phi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    const std::array<float, 4> sin_phis = {{aux_sin_phi, -aux_sin_phi, -aux_sin_phi, aux_sin_phi}};

    for (unsigned int i = 0; i < 4; ++i) {
        // Eq.(15)
        Mat33_t aux_rot = Mat33_t::Identity();
        aux_rot(0, 0) = cos_phi;
        aux_rot(0, 2) = sin_phis.at(i);
        aux_rot(1, 1) = -1;
        aux_rot(2, 0) = sin_phis.at(i);
        aux_rot(2, 2) = -cos_phi;
        // Eq.(8)
        const Mat33_t init_rot = s * U * aux_rot * Vt;
        init_rots.push_back(init_rot);

        // Eq.(16)
        Vec3_t aux_trans{x1s.at(i), 0.0, x3s.at(i)};
        aux_trans(0) = x1s.at(i);
        aux_trans(1) = 0;
        aux_trans(2) = x3s.at(i);
        aux_trans *= d1 + d3;
        // Eq.(8)
        const Vec3_t init_trans = U * aux_trans;
        init_transes.emplace_back(init_trans / init_trans.norm());

        // Eq.(9)
        const Vec3_t aux_normal{x1s.at(i), 0.0, x3s.at(i)};
        Vec3_t init_normal = V * aux_normal;
        if (init_normal(2) < 0) {
            init_normal = -init_normal;
        }
        init_normals.push_back(init_normal);
    }

    return true;
}

unsigned int homography_solver::check_inliers(const Mat33_t& H_21, std::vector<bool>& is_inlier_match, float& cost) {
    unsigned int num_inliers = 0;
    const auto num_matches = matches_12_.size();

    // chi-squared value (p=0.05, n=2)
    constexpr float chi_sq = 5.991;

    is_inlier_match.resize(num_matches);

    const Mat33_t H_12 = H_21.inverse();

    const float sigma_sq = sigma_ * sigma_;

    cost = 0;

    for (unsigned int i = 0; i < num_matches; ++i) {
        const auto& keypt_1 = undist_keypts_1_.at(matches_12_.at(i).first);
        const auto& keypt_2 = undist_keypts_2_.at(matches_12_.at(i).second);

        // 1. Transform to homogeneous coordinates

        const Vec3_t pt_1 = util::converter::to_homogeneous(keypt_1.pt);
        const Vec3_t pt_2 = util::converter::to_homogeneous(keypt_2.pt);

        // 2. Compute error

        Vec3_t transformed_pt_1 = H_21 * pt_1;
        transformed_pt_1 = transformed_pt_1 / transformed_pt_1(2);
        const float dist_sq_1 = (pt_2 - transformed_pt_1).squaredNorm();

        Vec3_t transformed_pt_2 = H_12 * pt_2;
        transformed_pt_2 = transformed_pt_2 / transformed_pt_2(2);
        const float dist_sq_2 = (pt_1 - transformed_pt_2).squaredNorm();

        const float dist_sq = std::max(dist_sq_1, dist_sq_2);

        double thr = chi_sq * sigma_sq;
        if (thr > dist_sq) {
            is_inlier_match.at(i) = true;
            cost += dist_sq;
            num_inliers++;
        }
        else {
            is_inlier_match.at(i) = false;
            cost += thr;
        }
    }

    return num_inliers;
}

} // namespace solve
} // namespace cv::slam
