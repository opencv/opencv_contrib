#include "solve/essential_5pt.hpp"
#include "solve/essential_solver.hpp"
#include "util/converter.hpp"
#include "util/random_array.hpp"
#include "util/trigonometric.hpp"

namespace cv::slam {
namespace solve {

essential_solver::essential_solver(const eigen_alloc_vector<Vec3_t>& bearings_1, const eigen_alloc_vector<Vec3_t>& bearings_2,
                                   const std::vector<std::pair<int, int>>& matches_12, bool use_fixed_seed)
    : bearings_1_(bearings_1), bearings_2_(bearings_2), matches_12_(matches_12),
      random_engine_(util::create_random_engine(use_fixed_seed)) {}

void essential_solver::find_via_ransac(const unsigned int max_num_iter, const bool recompute, const unsigned int min_set_size) {
    const auto num_matches = static_cast<unsigned int>(matches_12_.size());

    // 1. Prepare for RANSAC

    // minimum number of samples
    if (num_matches < min_set_size) {
        solution_is_valid_ = false;
        return;
    }

    // RANSAC variables
    best_cost_ = std::numeric_limits<float>::max();
    unsigned int best_num_inliers = 0;
    is_inlier_match_ = std::vector<bool>(num_matches, false);

    // minimum set of keypoint matches
    eigen_alloc_vector<Vec3_t> min_set_bearings_1(min_set_size);
    eigen_alloc_vector<Vec3_t> min_set_bearings_2(min_set_size);

    // shared variables in RANSAC loop
    // essential matrix from shot 1 to shot 2
    Mat33_t E_21_in_sac;
    // inlier/outlier flags
    std::vector<bool> is_inlier_match_in_sac(num_matches, false);

    // 2. RANSAC loop

    for (unsigned int iter = 0; iter < max_num_iter; iter++) {
        // 2-1. Create a minimum set
        const auto indices = util::create_random_array(min_set_size, 0U, num_matches - 1, random_engine_);
        for (unsigned int i = 0; i < min_set_size; ++i) {
            const auto idx = indices.at(i);
            min_set_bearings_1.at(i) = bearings_1_.at(matches_12_.at(idx).first);
            min_set_bearings_2.at(i) = bearings_2_.at(matches_12_.at(idx).second);
        }

        // 2-2. Compute candidate essential matrices with the minimal solver
        std::vector<Mat33_t> E_mats;
        assert(min_set_size >= 5);
        if (min_set_size < 8) {
            E_mats = compute_E_21_minimal(min_set_bearings_1, min_set_bearings_2);
        }
        else {
            E_mats.push_back(compute_E_21_nonminimal(min_set_bearings_1, min_set_bearings_2));
        }

        // see if any of the candidates are better than best_E_21_
        for (const auto& E_in_sac : E_mats) {
            // 2-3. Check inliers and compute a cost
            float cost_in_sac;
            unsigned int num_inliers = check_inliers(E_in_sac, is_inlier_match_in_sac, cost_in_sac);

            // 2-4. Update the best model
            if (num_inliers > min_set_size && best_cost_ > cost_in_sac) {
                best_cost_ = cost_in_sac;
                best_E_21_ = E_in_sac;
                is_inlier_match_ = is_inlier_match_in_sac;
                best_num_inliers = num_inliers;
            }
        }
    }

    solution_is_valid_ = best_cost_ < std::numeric_limits<float>::max();

    // we need a valid solution with at least 8 inliers to do the refinement
    // since it uses the 8pt algorithm
    if (!recompute || !solution_is_valid_ || best_num_inliers < 8) {
        return;
    }

    // 3. Recompute an essential matrix with only the inlier matches and the
    // non-minimal solver

    eigen_alloc_vector<Vec3_t> inlier_bearing_1;
    eigen_alloc_vector<Vec3_t> inlier_bearing_2;
    inlier_bearing_1.reserve(matches_12_.size());
    inlier_bearing_2.reserve(matches_12_.size());
    for (unsigned int i = 0; i < matches_12_.size(); ++i) {
        if (is_inlier_match_.at(i)) {
            inlier_bearing_1.push_back(bearings_1_.at(matches_12_.at(i).first));
            inlier_bearing_2.push_back(bearings_2_.at(matches_12_.at(i).second));
        }
    }

    best_E_21_ = compute_E_21_nonminimal(inlier_bearing_1, inlier_bearing_2);
    check_inliers(best_E_21_, is_inlier_match_, best_cost_);
}

Mat33_t essential_solver::compute_E_21_nonminimal(const eigen_alloc_vector<Vec3_t>& bearings_1, const eigen_alloc_vector<Vec3_t>& bearings_2) {
    assert(bearings_1.size() == bearings_2.size());

    const auto num_points = bearings_1.size();

    typedef Eigen::Matrix<Mat33_t::Scalar, Eigen::Dynamic, 9> CoeffMatrix;
    CoeffMatrix A(num_points, 9);

    for (unsigned int i = 0; i < num_points; i++) {
        A.block<1, 3>(i, 0) = bearings_2.at(i)(0) * bearings_1.at(i);
        A.block<1, 3>(i, 3) = bearings_2.at(i)(1) * bearings_1.at(i);
        A.block<1, 3>(i, 6) = bearings_2.at(i)(2) * bearings_1.at(i);
    }

    const Eigen::JacobiSVD<CoeffMatrix> init_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Matrix<Mat33_t::Scalar, 9, 1> v = init_svd.matrixV().col(8);
    // need transpose() because elements are contained as col-major after it was constructed from a pointer
    const Mat33_t init_E_21 = Mat33_t(v.data()).transpose();

    const Eigen::JacobiSVD<Mat33_t> svd(init_E_21, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Mat33_t& U = svd.matrixU();
    Vec3_t lambda = svd.singularValues();
    const Mat33_t& V = svd.matrixV();

    lambda(2) = 0.0;

    const Mat33_t E_21 = U * lambda.asDiagonal() * V.transpose();

    return E_21;
}

std::vector<Mat33_t> essential_solver::compute_E_21_minimal(const eigen_alloc_vector<Vec3_t>& x1,
                                                            const eigen_alloc_vector<Vec3_t>& x2) {
    std::vector<Mat33_t> E_mats;
    E_mats.reserve(10);

    // Extract the Nullspace from the epipolar constraint.
    bool success;
    const Eigen::Matrix<double, 9, 4> E_basis = find_nullspace_of_epipolar_constraint(x1, x2, success);
    if (!success) {
        return E_mats;
    }

    // Use the epipolar constaints to build a matrix representing
    // ten, 3rd order polynomial equations in the 3 unknowns x,y,z
    // (this ends up being a lot of polynomial math to get us to a constraint matrix
    // we can solve).
    const Eigen::Matrix<double, 10, 20> constraint_matrix = form_polynomial_constraint_matrix(E_basis);

    // Step 3: Apply Gauss-Jordan Elimination to the constraint matrix.
    Eigen::FullPivLU<Mat10_t> c_lu(constraint_matrix.block<10, 10>(0, 0));
    const Mat10_t eliminated_matrix = c_lu.solve(constraint_matrix.block<10, 10>(0, 10));

    // Solving the eliminated matrix like in the matlab code shown in Stewenius et al.

    // Build the "action matrix"
    Mat10_t action_matrix = Mat10_t::Zero();
    action_matrix.block<3, 10>(0, 0) = eliminated_matrix.block<3, 10>(0, 0);
    action_matrix.row(3) = eliminated_matrix.row(4);
    action_matrix.row(4) = eliminated_matrix.row(5);
    action_matrix.row(5) = eliminated_matrix.row(7);
    action_matrix(6, 0) = -1.0;
    action_matrix(7, 1) = -1.0;
    action_matrix(8, 3) = -1.0;
    action_matrix(9, 6) = -1.0;

    // Get the solutions to the constraint matrix (i.e. the 10 sets of solutions
    // for our 3 unknowns)
    Eigen::EigenSolver<Mat10_t> eigensolver(action_matrix);
    const auto& eig_vecs = eigensolver.eigenvectors();
    const auto& eig_vals = eigensolver.eigenvalues();

    // Build essential matrices by substituting in the real solutions (there can be up to 10
    // since we solved a 10th degree polynomial)
    for (int s = 0; s < 10; ++s) {
        // Only consider real solutions.
        if (eig_vals(s).imag() != 0) {
            continue;
        }
        Mat33_t E;
        Eigen::Map<Vec9_t>(E.data()) = E_basis * eig_vecs.col(s).tail<4>().real();
        E_mats.emplace_back(E.transpose());
    }
    return E_mats;
}

bool essential_solver::decompose(const Mat33_t& E_21, eigen_alloc_vector<Mat33_t>& init_rots, eigen_alloc_vector<Vec3_t>& init_transes) {
    // https://en.wikipedia.org/wiki/Essential_matrix#Determining_R_and_t_from_E

    const Eigen::JacobiSVD<Mat33_t> svd(E_21, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Vec3_t trans = svd.matrixU().col(2);
    trans.normalize();

    Mat33_t W = Mat33_t::Zero();
    W(0, 1) = -1;
    W(1, 0) = 1;
    W(2, 2) = 1;

    Mat33_t rot_1 = svd.matrixU() * W * svd.matrixV().transpose();
    if (rot_1.determinant() < 0) {
        rot_1 *= -1;
    }

    Mat33_t rot_2 = svd.matrixU() * W.transpose() * svd.matrixV().transpose();
    if (rot_2.determinant() < 0) {
        rot_2 *= -1;
    }

    init_rots = {rot_1, rot_1, rot_2, rot_2};
    init_transes = {trans, -trans, trans, -trans};

    return true;
}

Mat33_t essential_solver::create_E_21(const Mat33_t& rot_1w, const Vec3_t& trans_1w, const Mat33_t& rot_2w, const Vec3_t& trans_2w) {
    const Mat33_t rot_21 = rot_2w * rot_1w.transpose();
    const Vec3_t trans_21 = -rot_21 * trans_1w + trans_2w;
    const Mat33_t trans_21_x = util::converter::to_skew_symmetric_mat(trans_21);
    return trans_21_x * rot_21;
}

unsigned int essential_solver::check_inliers(const Mat33_t& E_21, std::vector<bool>& is_inlier_match, float& cost) {
    unsigned int num_inliers = 0;
    const auto num_points = matches_12_.size();

    is_inlier_match.resize(num_points);

    const Mat33_t E_12 = E_21.transpose();

    cost = 0.0;

    // outlier threshold of cosine between a bearing vector and the epipolar plane
    const float cos_angle_thr = util::cos(1.0 * M_PI / 180.0);

    for (unsigned int i = 0; i < num_points; ++i) {
        const auto& bearing_1 = bearings_1_.at(matches_12_.at(i).first);
        const auto& bearing_2 = bearings_2_.at(matches_12_.at(i).second);

        const Vec3_t epiplane_in_2 = E_21 * bearing_1;
        const float cos_in_2 = epiplane_in_2.cross(bearing_2).norm() / epiplane_in_2.norm();

        const Vec3_t epiplane_in_1 = E_12 * bearing_2;
        const float cos_in_1 = epiplane_in_1.cross(bearing_1).norm() / epiplane_in_1.norm();

        float worst_cos_angle = std::min(cos_in_1, cos_in_2);

        if (cos_angle_thr < worst_cos_angle) {
            is_inlier_match.at(i) = true;
            cost += 1.0 - worst_cos_angle;
            num_inliers++;
        }
        else {
            is_inlier_match.at(i) = false;
            cost += 1.0 - cos_angle_thr;
        }
    }

    return num_inliers;
}

} // namespace solve
} // namespace cv::slam
