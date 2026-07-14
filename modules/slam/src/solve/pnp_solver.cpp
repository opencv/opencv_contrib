#include "solve/pnp_solver.hpp"
#include "util/fancy_index.hpp"
#include "util/random_array.hpp"
#include "util/trigonometric.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace solve {

pnp_solver::pnp_solver(const eigen_alloc_vector<Vec3_t>& valid_bearings,
                       const std::vector<int>& octaves,
                       const eigen_alloc_vector<Vec3_t>& valid_points,
                       const std::vector<float>& scale_factors,
                       const unsigned int min_num_inliers,
                       const bool use_fixed_seed,
                       const unsigned int gauss_newton_num_iter)
    : num_matches_(valid_bearings.size()), valid_bearings_(valid_bearings),
      valid_points_(valid_points), min_num_inliers_(min_num_inliers),
      random_engine_(util::create_random_engine(use_fixed_seed)),
      gauss_newton_num_iter_(gauss_newton_num_iter) {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: solve::pnp_solver");

    max_cos_errors_.clear();
    max_cos_errors_.resize(num_matches_);

    constexpr double max_rad_error = 1.0 * M_PI / 180.0;
    for (unsigned int i = 0; i < num_matches_; ++i) {
        // Calculate radial error threshold from each scale factor
        const auto max_rad_error_with_scale = scale_factors.at(octaves.at(i)) * max_rad_error;
        max_cos_errors_.at(i) = util::cos(max_rad_error_with_scale);
    }

    assert(num_matches_ == valid_bearings_.size());
    assert(num_matches_ == octaves.size());
    assert(num_matches_ == valid_points_.size());
    assert(num_matches_ == max_cos_errors_.size());
}

pnp_solver::~pnp_solver() {
    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: solve::pnp_solver");
}

void pnp_solver::find_via_ransac(const unsigned int max_num_iter, const bool recompute) {
    // 1. Prepare for RANSAC

    // minimum number of samples (= 4)
    static constexpr unsigned int min_set_size = 4;
    if (num_matches_ < min_set_size || num_matches_ < min_num_inliers_) {
        solution_is_valid_ = false;
        return;
    }

    // RANSAC variables
    is_inlier_match = std::vector<bool>(num_matches_, false);

    // shared variables in RANSAC loop
    // rotation from world to camera
    Mat33_t rot_cw_in_sac;
    // translation from world to camera
    Vec3_t trans_cw_in_sac;
    // inlier/outlier flags
    std::vector<bool> is_inlier_match_in_sac;

    eigen_alloc_vector<Vec3_t> min_set_bearings;
    eigen_alloc_vector<Vec3_t> min_set_pos_ws;

    // 2. RANSAC loop

    double min_cost = std::numeric_limits<double>::max();
    for (unsigned int iter = 0; iter < max_num_iter; ++iter) {
        // 2-1. Create a minimum set
        const auto random_indices = util::create_random_array(min_set_size, 0U, num_matches_ - 1, random_engine_);
        assert(random_indices.size() == min_set_size);

        min_set_bearings.clear();
        min_set_pos_ws.clear();

        for (const auto i : random_indices) {
            const Vec3_t& bearing = valid_bearings_.at(i);
            const Vec3_t& pos_w = valid_points_.at(i);

            min_set_bearings.push_back(bearing);
            min_set_pos_ws.push_back(pos_w);
        }

        // 2-2. Compute a camera pose
        compute_pose(min_set_bearings, min_set_pos_ws, rot_cw_in_sac, trans_cw_in_sac, gauss_newton_num_iter_);

        // 2-3. Check inliers and compute a score
        double cost = 0.0;
        const auto num_inliers = check_inliers(rot_cw_in_sac, trans_cw_in_sac, is_inlier_match_in_sac, cost);

        // 2-4. Update the best model
        if (num_inliers > min_num_inliers_ && min_cost > cost) {
            min_cost = cost;
            best_rot_cw_ = rot_cw_in_sac;
            best_trans_cw_ = trans_cw_in_sac;
            is_inlier_match = is_inlier_match_in_sac;
        }
    }

    solution_is_valid_ = min_cost < std::numeric_limits<double>::max();

    if (!recompute || !solution_is_valid_) {
        return;
    }

    // 3. Recompute a camera pose only with the inlier matches

    eigen_alloc_vector<Vec3_t> inlier_bearings;
    eigen_alloc_vector<Vec3_t> inlier_pos_ws;
    for (unsigned int i = 0; i < num_matches_; ++i) {
        if (!is_inlier_match.at(i)) {
            continue;
        }
        const Vec3_t& bearing = valid_bearings_.at(i);
        const Vec3_t& pos_w = valid_points_.at(i);
        inlier_bearings.push_back(bearing);
        inlier_pos_ws.push_back(pos_w);
    }

    compute_pose(inlier_bearings, inlier_pos_ws, best_rot_cw_, best_trans_cw_, gauss_newton_num_iter_);
}

unsigned int pnp_solver::check_inliers(const Mat33_t& rot_cw, const Vec3_t& trans_cw, std::vector<bool>& is_inlier, double& cost) {
    unsigned int num_inliers = 0;

    cost = 0.0;
    is_inlier.resize(num_matches_);
    for (unsigned int i = 0; i < num_matches_; ++i) {
        const Vec3_t& pos_w = valid_points_.at(i);
        const Vec3_t& bearing = valid_bearings_.at(i);

        const Vec3_t pos_c = rot_cw * pos_w + trans_cw;

        // Compute cosine similarity between the bearing vector and the position of the 3D point
        const auto cos_angle = pos_c.dot(bearing) / pos_c.norm();

        // The match is inlier if the cosine similarity is less than or equal to the threshold
        if (max_cos_errors_.at(i) < cos_angle) {
            is_inlier.at(i) = true;
            cost += 1 - cos_angle;
            ++num_inliers;
        }
        else {
            cost += 1 - max_cos_errors_.at(i);
            is_inlier.at(i) = false;
        }
    }

    return num_inliers;
}

double pnp_solver::compute_pose(const eigen_alloc_vector<Vec3_t>& bearing_vectors,
                                const eigen_alloc_vector<Vec3_t>& pos_ws,
                                Mat33_t& rot_cw, Vec3_t& trans_cw, const unsigned int num_iter) {
    // EPnP: An AccurateO(n)Solution to the PnP Problem
    // (Lepetit et al. in IJCV 2009)

    const eigen_alloc_vector<Vec3_t> control_points = choose_control_points(pos_ws);
    const eigen_alloc_vector<Vec4_t> alphas = compute_barycentric_coordinates(control_points, pos_ws);

    // Construct M matrix according to Eq.(5) and (6)
    const MatX_t M = compute_M(bearing_vectors, alphas);

    // Compute singular vectors U
    const MatRC_t<12, 12> MtM = M.transpose() * M;
    Eigen::JacobiSVD<MatRC_t<12, 12>> SVD(MtM, Eigen::ComputeFullV | Eigen::ComputeFullU);

    const MatRC_t<12, 12> U = SVD.matrixU();

    // Compute L matrix and rho vector in Eq.(13)
    const MatRC_t<6, 10> L_6x10 = compute_L_6x10(U);
    const MatRC_t<6, 1> Rho = compute_rho(control_points);

    Mat33_t rot_cand;
    Vec3_t trans_cand;
    double reproj_minimum = std::numeric_limits<double>::max();
    for (unsigned int N = 2; N <= 4; N++) {
        // Solve rotation and translation from the Case N = 2 to the Case N = 4
        // The result which achieved the lowest reprojection error will be chosen

        // Solve Eq.(13)
        const Vec4_t betas = find_initial_betas(L_6x10, Rho, N);
        // Minimize Eq.(15)
        const Vec4_t refined_betas = gauss_newton(L_6x10, Rho, betas, num_iter);

        // Eq.(16)
        const eigen_alloc_vector<Vec3_t> ccs = compute_ccs(refined_betas, U);
        const bool bearing_z_sign = bearing_vectors.at(0)(2) > 0;
        const eigen_alloc_vector<Vec3_t> pcs = compute_pcs(alphas, ccs, bearing_z_sign);

        estimate_R_and_t(pos_ws, pcs, rot_cand, trans_cand);

        const auto reproj_error = reprojection_error(pos_ws, bearing_vectors, rot_cand, trans_cand);

        // Take the rotation and translation which archieve the minimum reprojection error
        if (reproj_error < reproj_minimum) {
            reproj_minimum = reproj_error;
            rot_cw = rot_cand;
            trans_cw = trans_cand;
        }
    }
    return reproj_minimum;
}

eigen_alloc_vector<Vec3_t> pnp_solver::choose_control_points(const eigen_alloc_vector<Vec3_t>& pos_ws) {
    const unsigned int num_correspondences = pos_ws.size();
    eigen_alloc_vector<Vec3_t> cws;
    for (unsigned int i = 0; i < 4; i++) {
        cws.emplace_back(Vec3_t{0, 0, 0});
    }

    // Take C0 as the reference points centroid:
    for (unsigned int i = 0; i < num_correspondences; ++i) {
        cws.at(0) += pos_ws.at(i);
    }
    cws.at(0) /= num_correspondences;

    // Take C1, C2, and C3 from PCA on the reference points:
    MatX_t PW0(num_correspondences, 3);

    for (unsigned int i = 0; i < num_correspondences; ++i) {
        PW0.block<1, 3>(i, 0) = pos_ws.at(i) - cws.at(0);
    }

    const MatX_t PW0tPW0 = PW0.transpose() * PW0;
    Eigen::JacobiSVD<MatX_t> SVD(PW0tPW0, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const MatX_t D = SVD.singularValues();
    const MatX_t U = SVD.matrixU();

    for (unsigned int i = 1; i < 4; ++i) {
        const double k = std::sqrt(D(i - 1, 0) / num_correspondences);
        cws.at(i) = cws.at(0) + k * U.block<3, 1>(0, i - 1);
    }
    return cws;
}

eigen_alloc_vector<Vec4_t> pnp_solver::compute_barycentric_coordinates(const eigen_alloc_vector<Vec3_t>& control_points, const eigen_alloc_vector<Vec3_t>& pos_ws) {
    const unsigned int num_correspondences = pos_ws.size();
    // The barycentric coordinates are obtained easily
    // because the positions of C1 to C3 relative to C0 form the orthogonal basis
    Mat33_t CC;
    for (unsigned int i = 0; i < 3; i++) {
        CC.block<3, 1>(0, i) = control_points.at(i + 1) - control_points.at(0);
    }

    // Compute generalized inverse
    Eigen::JacobiSVD<Mat33_t> svd(CC, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vec3_t D = svd.singularValues();
    Mat33_t S;
    S.setZero();
    for (unsigned int i = 0; i < D.size(); ++i) {
        if (D(i) > 1e-6) {
            S(i, i) = 1 / D(i);
        }
        else {
            S(i, i) = 0;
        }
    }
    const Mat33_t CC_inv = svd.matrixV() * S * svd.matrixU().transpose();

    eigen_alloc_vector<Vec4_t> alphas;
    for (unsigned int i = 0; i < num_correspondences; ++i) {
        const Vec3_t pi = pos_ws.at(i);
        Vec4_t alpha;
        alpha.block<3, 1>(1, 0) = (CC_inv * (pi - control_points.at(0)));

        alpha(0) = 1.0f - alpha(1) - alpha(2) - alpha(3);

        alphas.push_back(alpha);
    }
    return alphas;
}

MatX_t pnp_solver::compute_M(const eigen_alloc_vector<Vec3_t>& bearings,
                             const eigen_alloc_vector<Vec4_t>& alphas) {
    const unsigned int num_correspondences = bearings.size();

    MatX_t M(num_correspondences * 2, 12);

    // Fill the M matrix up according to Eq.(5) and (6)
    for (unsigned int j = 0; j < num_correspondences; j++) {
        const Vec4_t alpha = alphas.at(j);
        const Vec3_t bearing = bearings.at(j);
        const auto u = bearing(0) / bearing(2);
        const auto v = bearing(1) / bearing(2);
        for (unsigned int i = 0; i < 4; i++) {
            M(j * 2, 3 * i) = alpha(i);
            M(j * 2, 3 * i + 1) = 0.0;
            M(j * 2, 3 * i + 2) = -alpha(i) * u;

            M(j * 2 + 1, 3 * i) = 0.0;
            M(j * 2 + 1, 3 * i + 1) = alpha(i);
            M(j * 2 + 1, 3 * i + 2) = -alpha(i) * v;
        }
    }

    return M;
}

eigen_alloc_vector<Vec3_t> pnp_solver::compute_ccs(const Vec4_t& betas, const MatX_t& U) {
    eigen_alloc_vector<Vec3_t> ccs;
    for (unsigned int i = 0; i < 4; ++i) {
        ccs.emplace_back(Vec3_t{0, 0, 0});
    }

    // Compute the local control points
    for (unsigned int i = 0; i < 4; i++) {
        for (unsigned int j = 0; j < 4; j++) {
            ccs.at(i) += betas(j) * U.block<3, 1>(3 * i, 11 - j);
        }
    }
    return ccs;
}

eigen_alloc_vector<Vec3_t> pnp_solver::compute_pcs(const eigen_alloc_vector<Vec4_t>& alphas, const eigen_alloc_vector<Vec3_t>& ccs, const bool bearing_z_sign) {
    const unsigned int num_correspondences = alphas.size();
    eigen_alloc_vector<Vec3_t> pcs;
    // Compute local 3D points using the barycentric coordinates and the local control points
    for (unsigned int i = 0; i < num_correspondences; ++i) {
        const Vec4_t a = alphas.at(i);
        pcs.emplace_back(a(0) * ccs.at(0) + a(1) * ccs.at(1) + a(2) * ccs.at(2) + a(3) * ccs.at(3));
    }
    // Invert the local points if they have not the same direction of the bearing vectors
    const bool pc_z_sign = pcs.at(0)(2) > 0;
    if (pc_z_sign != bearing_z_sign) {
        for (Vec3_t& pc : pcs) {
            pc *= -1;
        }
    }
    return pcs;
}

double pnp_solver::reprojection_error(const eigen_alloc_vector<Vec3_t>& pws, const eigen_alloc_vector<Vec3_t>& bearings, const Mat33_t& rot, const Vec3_t& trans) {
    const unsigned int num_correspondences = pws.size();
    double error_sum = 0.0;

    for (unsigned int i = 0; i < num_correspondences; ++i) {
        const Vec3_t pw = pws.at(i);
        const Vec3_t pc = rot * pw + trans;

        const auto cos_angle = pc.dot(bearings.at(i)) / pc.norm();
        error_sum += 1.0 - cos_angle;
    }
    return error_sum / num_correspondences;
}

void pnp_solver::estimate_R_and_t(const eigen_alloc_vector<Vec3_t>& pws, const eigen_alloc_vector<Vec3_t>& pcs, Mat33_t& rot, Vec3_t& trans) {
    const unsigned int num_correspondences = pws.size();
    Vec3_t pc0{0, 0, 0}, pw0{0, 0, 0};

    for (unsigned int i = 0; i < num_correspondences; i++) {
        pc0 += pcs.at(i);
        pw0 += pws.at(i);
    }
    pc0 /= num_correspondences;
    pw0 /= num_correspondences;

    // The correlation matrix of world points to local points
    Mat33_t CM = Mat33_t::Zero();

    for (unsigned int i = 0; i < num_correspondences; i++) {
        const Vec3_t pc = pcs.at(i);
        const Vec3_t pw = pws.at(i);
        CM += (pc - pc0) * (pw - pw0).transpose();
    }

    Eigen::JacobiSVD<MatX_t> SVD(CM, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const MatX_t& CM_u = SVD.matrixU();
    const MatX_t& CM_vt = SVD.matrixV().transpose();

    rot = CM_u * CM_vt;
    const double det = rot.determinant();

    if (det < 0) {
        Mat33_t SGM = Mat33_t::Identity();
        SGM(2, 2) = -1;
        rot = CM_u * SGM * CM_vt;
    }

    trans = pc0 - rot * pw0;
}

Vec4_t pnp_solver::find_initial_betas(const MatRC_t<6, 10>& L_6x10, const Vec6_t& Rho, unsigned int N) {
    assert(2 <= N && N <= 4);
    if (N == 2) {
        return find_initial_betas_2(L_6x10, Rho);
    }
    else if (N == 3) {
        return find_initial_betas_3(L_6x10, Rho);
    }
    else {
        return find_initial_betas_4(L_6x10, Rho);
    }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

Vec4_t pnp_solver::find_initial_betas_2(const MatRC_t<6, 10>& L_6x10, const Vec6_t& Rho) {
    Vec4_t betas;
    MatRC_t<6, 3> L_6x3;

    for (unsigned int i = 0; i < 6; ++i) {
        L_6x3(i, 0) = L_6x10(i, 0);
        L_6x3(i, 1) = L_6x10(i, 1);
        L_6x3(i, 2) = L_6x10(i, 2);
    }

    Eigen::JacobiSVD<MatX_t> SVD(L_6x3, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Vec3_t b3 = SVD.solve(Rho);

    if (b3(0) < 0) {
        betas(0) = std::sqrt(-b3(0));
        betas(1) = (b3(2) < 0) ? std::sqrt(-b3(2)) : 0.0;
    }
    else {
        betas(0) = std::sqrt(b3(0));
        betas(1) = (b3(2) > 0) ? std::sqrt(b3(2)) : 0.0;
    }

    if (b3(1) < 0) {
        betas(0) = -betas(0);
    }

    betas(2) = 0.0;
    betas(3) = 0.0;

    return betas;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

Vec4_t pnp_solver::find_initial_betas_3(const MatRC_t<6, 10>& L_6x10, const Vec6_t& Rho) {
    Vec4_t betas;
    MatRC_t<6, 5> L_6x5(6, 5);

    for (unsigned int i = 0; i < 6; ++i) {
        L_6x5(i, 0) = L_6x10(i, 0);
        L_6x5(i, 1) = L_6x10(i, 1);
        L_6x5(i, 2) = L_6x10(i, 2);
        L_6x5(i, 3) = L_6x10(i, 3);
        L_6x5(i, 4) = L_6x10(i, 4);
    }

    Eigen::JacobiSVD<MatX_t> SVD(L_6x5, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Vec5_t b5 = SVD.solve(Rho);

    if (b5(0) < 0) {
        betas(0) = std::sqrt(-b5(0));
        betas(1) = (b5(2) < 0) ? std::sqrt(-b5(2)) : 0.0;
    }
    else {
        betas(0) = std::sqrt(b5(0));
        betas(1) = (b5(2) > 0) ? std::sqrt(b5(2)) : 0.0;
    }
    if (b5(1) < 0) {
        betas(0) = -betas(0);
    }

    betas(2) = b5(3) / betas(0);
    betas(3) = 0.0;

    return betas;
}

Vec4_t pnp_solver::find_initial_betas_4(const MatRC_t<6, 10>& L_6x10, const Vec6_t& Rho) {
    Vec4_t betas;
    MatRC_t<6, 4> L_6x4;

    for (unsigned int i = 0; i < 6; ++i) {
        L_6x4(i, 0) = L_6x10(i, 0);
        L_6x4(i, 1) = L_6x10(i, 1);
        L_6x4(i, 2) = L_6x10(i, 3);
        L_6x4(i, 3) = L_6x10(i, 6);
    }

    Eigen::JacobiSVD<MatX_t> SVD(L_6x4, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Vec4_t b4 = SVD.solve(Rho);

    if (b4(0) < 0) {
        betas(0) = std::sqrt(-b4(0));
        betas(1) = -b4(1) / betas(0);
        betas(2) = -b4(2) / betas(0);
        betas(3) = -b4(3) / betas(0);
    }
    else {
        betas(0) = std::sqrt(b4(0));
        betas(1) = b4(1) / betas(0);
        betas(2) = b4(2) / betas(0);
        betas(3) = b4(3) / betas(0);
    }
    return betas;
}

MatRC_t<6, 10> pnp_solver::compute_L_6x10(const MatX_t& U) {
    eigen_alloc_vector<eigen_alloc_vector<Vec3_t>> dv;

    // Compute difference vectors for four each candidate control vectors
    for (unsigned int i = 0; i < 4; i++) {
        const MatRC_t<12, 1> u = U.block<12, 1>(0, 11 - i);
        // Difference vectors among four candidate control vectors
        eigen_alloc_vector<Vec3_t> diffs;
        unsigned int a = 0, b = 1;
        for (unsigned int j = 0; j < 6; j++) {
            const Vec3_t diff = u.block<3, 1>(3 * a, 0) - u.block<3, 1>(3 * b, 0);
            diffs.push_back(diff);
            b++;
            if (b > 3) {
                a++;
                b = a + 1;
            }
        }
        dv.push_back(diffs);
    }

    MatRC_t<6, 10> L_6x10;
    for (unsigned int i = 0; i < 6; i++) {
        L_6x10(i, 0) = dv.at(0).at(i).dot(dv.at(0).at(i));
        L_6x10(i, 1) = 2.0f * dv.at(0).at(i).dot(dv.at(1).at(i));
        L_6x10(i, 2) = dv.at(1).at(i).dot(dv.at(1).at(i));
        L_6x10(i, 3) = 2.0f * dv.at(0).at(i).dot(dv.at(2).at(i));
        L_6x10(i, 4) = 2.0f * dv.at(1).at(i).dot(dv.at(2).at(i));
        L_6x10(i, 5) = dv.at(2).at(i).dot(dv.at(2).at(i));
        L_6x10(i, 6) = 2.0f * dv.at(0).at(i).dot(dv.at(3).at(i));
        L_6x10(i, 7) = 2.0f * dv.at(1).at(i).dot(dv.at(3).at(i));
        L_6x10(i, 8) = 2.0f * dv.at(2).at(i).dot(dv.at(3).at(i));
        L_6x10(i, 9) = dv.at(3).at(i).dot(dv.at(3).at(i));
    }
    return L_6x10;
}

Vec6_t pnp_solver::compute_rho(const eigen_alloc_vector<Vec3_t>& control_points) {
    Vec6_t Rho;
    Rho(0) = (control_points.at(0) - control_points.at(1)).squaredNorm();
    Rho(1) = (control_points.at(0) - control_points.at(2)).squaredNorm();
    Rho(2) = (control_points.at(0) - control_points.at(3)).squaredNorm();
    Rho(3) = (control_points.at(1) - control_points.at(2)).squaredNorm();
    Rho(4) = (control_points.at(1) - control_points.at(3)).squaredNorm();
    Rho(5) = (control_points.at(2) - control_points.at(3)).squaredNorm();
    return Rho;
}

void pnp_solver::compute_A_and_b_for_gauss_newton(const MatRC_t<6, 10>& L_6x10, const Vec6_t& Rho,
                                                  const Vec4_t& betas, MatRC_t<6, 4>& A, Vec6_t& b) {
    for (unsigned int i = 0; i < 6; ++i) {
        A(i, 0) = 2 * L_6x10(i, 0) * betas(0) + L_6x10(i, 1) * betas(1)
                  + L_6x10(i, 3) * betas(2) + L_6x10(i, 6) * betas(3);
        A(i, 1) = L_6x10(i, 1) * betas(0) + 2 * L_6x10(i, 2) * betas(1)
                  + L_6x10(i, 4) * betas(2) + L_6x10(i, 7) * betas(3);
        A(i, 2) = L_6x10(i, 3) * betas(0) + L_6x10(i, 4) * betas(1)
                  + 2 * L_6x10(i, 5) * betas(2) + L_6x10(i, 8) * betas(3);
        A(i, 3) = L_6x10(i, 6) * betas(0) + L_6x10(i, 7) * betas(1)
                  + L_6x10(i, 8) * betas(2) + 2 * L_6x10(i, 9) * betas(3);

        b(i, 0) = Rho(i) - (L_6x10(i, 0) * betas(0) * betas(0) + L_6x10(i, 1) * betas(0) * betas(1) + L_6x10(i, 2) * betas(1) * betas(1) + L_6x10(i, 3) * betas(0) * betas(2) + L_6x10(i, 4) * betas(1) * betas(2) + L_6x10(i, 5) * betas(2) * betas(2) + L_6x10(i, 6) * betas(0) * betas(3) + L_6x10(i, 7) * betas(1) * betas(3) + L_6x10(i, 8) * betas(2) * betas(3) + L_6x10(i, 9) * betas(3) * betas(3));
    }
}

Vec4_t pnp_solver::gauss_newton(const MatRC_t<6, 10>& L_6x10, const Vec6_t& Rho, const Vec4_t& initial_betas, const unsigned int num_iter) {
    Vec4_t betas = initial_betas;

    MatRC_t<6, 4> A;
    Vec6_t B;

    for (unsigned int j = 0; j < num_iter; j++) {
        compute_A_and_b_for_gauss_newton(L_6x10, Rho, betas, A, B);

        // Using fastest QR decomposition in Eigen
        betas += A.householderQr().solve(B);
    }
    return betas;
}

} // namespace solve
} // namespace cv::slam
