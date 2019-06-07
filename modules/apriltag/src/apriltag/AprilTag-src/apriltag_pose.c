#include <math.h>
#include <stdio.h>

#include "apriltag_pose.h"
#include "apriltag_math.h"
#include "common/homography.h"
#include "common/image_u8x3.h"


/**
 * Calculate projection operator from image points.
 */
matd_t* calculate_F(matd_t* v) {
    matd_t* outer_product = matd_op("MM'", v, v, v, v);
    matd_t* inner_product = matd_op("M'M", v, v);
    matd_scale_inplace(outer_product, 1.0/inner_product->data[0]);
    matd_destroy(inner_product);
    return outer_product;
}

/**
 * Returns the value of the supplied scalar matrix 'a' and destroys the matrix.
 */
double matd_to_double(matd_t *a)
{
    assert(matd_is_scalar(a));
    double d = a->data[0];
    matd_destroy(a);
    return d;
}

/**
 * @param v Image points on the image plane.
 * @param p Object points in object space.
 * @outparam t Optimal translation.
 * @param R In/Outparam. Should be set to initial guess at R. Will be modified to be the optimal translation.
 * @param n_points Number of points.
 * @param n_steps Number of iterations.
 *
 * @return Object-space error after iteration.
 *
 * Implementation of Orthogonal Iteration from Lu, 2000.
 */
double orthogonal_iteration(matd_t** v, matd_t** p, matd_t** t, matd_t** R, int n_points, int n_steps) {
    matd_t* p_mean = matd_create(3, 1);
    for (int i = 0; i < n_points; i++) {
        matd_add_inplace(p_mean, p[i]);
    }
    matd_scale_inplace(p_mean, 1.0/n_points);

    matd_t** p_res = malloc(sizeof(matd_t *)*n_points);
    for (int i = 0; i < n_points; i++) {
        p_res[i] = matd_op("M-M", p[i], p_mean);
    }

    // Compute M1_inv.
    matd_t** F = malloc(sizeof(matd_t *)*n_points);
    matd_t *avg_F = matd_create(3, 3);
    for (int i = 0; i < n_points; i++) {
        F[i] = calculate_F(v[i]);
        matd_add_inplace(avg_F, F[i]);
    }
    matd_scale_inplace(avg_F, 1.0/n_points);
    matd_t *I3 = matd_identity(3);
    matd_t *M1 = matd_subtract(I3, avg_F);
    matd_t *M1_inv = matd_inverse(M1);
    matd_destroy(avg_F);
    matd_destroy(M1);

    double prev_error = HUGE_VAL;
    // Iterate.
    for (int i = 0; i < n_steps; i++) {
        // Calculate translation.
        matd_t *M2 = matd_create(3, 1);
        for (int j = 0; j < n_points; j++) {
            matd_t* M2_update = matd_op("(M - M)*M*M", F[j], I3, *R, p[j]);
            matd_add_inplace(M2, M2_update);
            matd_destroy(M2_update);
        }
        matd_scale_inplace(M2, 1.0/n_points);
        matd_destroy(*t);
        *t = matd_multiply(M1_inv, M2);
        matd_destroy(M2);

        // Calculate rotation.
        matd_t** q = malloc(sizeof(matd_t *)*n_points);
        matd_t* q_mean = matd_create(3, 1);
        for (int j = 0; j < n_points; j++) {
            q[j] = matd_op("M*(M*M+M)", F[j], *R, p[j], *t);
            matd_add_inplace(q_mean, q[j]);
        }
        matd_scale_inplace(q_mean, 1.0/n_points);

        matd_t* M3 = matd_create(3, 3);
        for (int j = 0; j < n_points; j++) {
            matd_t *M3_update = matd_op("(M-M)*M'", q[j], q_mean, p_res[j]);
            matd_add_inplace(M3, M3_update);
            matd_destroy(M3_update);
        }
        matd_svd_t M3_svd = matd_svd(M3);
        matd_destroy(M3);
        matd_destroy(*R);
        *R = matd_op("M*M'", M3_svd.U, M3_svd.V);
        matd_destroy(M3_svd.U);
        matd_destroy(M3_svd.S);
        matd_destroy(M3_svd.V);
        matd_destroy(q_mean);
        for (int j = 0; j < n_points; j++) {
            matd_destroy(q[j]);
        }

        double error = 0;
        for (int i = 0; i < 4; i++) {
            matd_t* err_vec = matd_op("(M-M)(MM+M)", I3, F[i], *R, p[i], *t);
            error += matd_to_double(matd_op("M'M", err_vec, err_vec));
            matd_destroy(err_vec);
        }
        prev_error = error;

        free(q);
    }

    matd_destroy(I3);
    matd_destroy(M1_inv);
    for (int i = 0; i < n_points; i++) {
        matd_destroy(p_res[i]);
        matd_destroy(F[i]);
    }
    free(p_res);
    free(F);
    matd_destroy(p_mean);
    return prev_error;
}

/**
 * Evaluates polynomial p at x.
 */
double polyval(double* p, int degree, double x) {
    double ret = 0;
    for (int i = 0; i <= degree; i++) {
        ret += p[i]*pow(x, i);
    }
    return ret;
}

/**
 * Numerically solve small degree polynomials. This is a customized method. It
 * ignores roots larger than 1000 and only gives small roots approximately.
 *
 * @param p Array of parameters s.t. p(x) = p[0] + p[1]*x + ...
 * @param degree The degree of p(x).
 * @outparam roots
 * @outparam n_roots
 */
void solve_poly_approx(double* p, int degree, double* roots, int* n_roots) {
    static const int MAX_ROOT = 1000;
    if (degree == 1) {
        if (fabs(p[0]) > MAX_ROOT*fabs(p[1])) {
            *n_roots = 0;
        } else {
            roots[0] = -p[0]/p[1];
            *n_roots = 1;
        }
        return;
    }

    // Calculate roots of derivative.
    double *p_der = malloc(sizeof(double)*degree);
    for (int i = 0; i < degree; i++) {
        p_der[i] = (i + 1) * p[i+1];
    }

    double *der_roots = malloc(sizeof(double)*(degree - 1));
    int n_der_roots;
    solve_poly_approx(p_der, degree - 1, der_roots, &n_der_roots);


    // Go through all possibilities for roots of the polynomial.
    *n_roots = 0;
    for (int i = 0; i <= n_der_roots; i++) {
        double min;
        if (i == 0) {
            min = -MAX_ROOT;
        } else {
            min = der_roots[i - 1];
        }

        double max;
        if (i == n_der_roots) {
            max = MAX_ROOT;
        } else {
            max = der_roots[i];
        }

        if (polyval(p, degree, min)*polyval(p, degree, max) < 0) {
            // We have a zero-crossing in this interval, use a combination of Newton' and bisection.
            // Some thanks to Numerical Recipes in C.

            double lower;
            double upper;
            if (polyval(p, degree, min) < polyval(p, degree, max)) {
                lower = min;
                upper = max;
            } else {
                lower = max;
                upper = min;
            }
            double root = 0.5*(lower + upper);
            double dx_old = upper - lower;
            double dx = dx_old;
            double f = polyval(p, degree, root);
            double df = polyval(p_der, degree - 1, root);

            for (int j = 0; j < 100; j++) {
                if (((f + df*(upper - root))*(f + df*(lower - root)) > 0)
                        || (fabs(2*f) > fabs(dx_old*df))) {
                    dx_old = dx;
                    dx = 0.5*(upper - lower);
                    root = lower + dx;
                } else {
                    dx_old = dx;
                    dx = -f/df;
                    root += dx;
                }

                if (root == upper || root == lower) {
                    break;
                }

                f = polyval(p, degree, root);
                df = polyval(p_der, degree - 1, root);

                if (f > 0) {
                    upper = root;
                } else {
                    lower = root;
                }
            }

            roots[(*n_roots)++] = root;
        } else if(polyval(p, degree, max) == 0) {
            // Double/triple root.
            roots[(*n_roots)++] = max;
        }
    }

    free(der_roots);
    free(p_der);
}

/**
 * Given a local minima of the pose error tries to find the other minima.
 */
matd_t* fix_pose_ambiguities(matd_t** v, matd_t** p, matd_t* t, matd_t* R, int n_points) {
    matd_t* I3 = matd_identity(3);

    // 1. Find R_t
    matd_t* R_t_3 = matd_vec_normalize(t);

    matd_t* e_x = matd_create(3, 1);
    MATD_EL(e_x, 0, 0) = 1;
    matd_t* R_t_1_tmp = matd_op("M-(M'*M)*M", e_x, e_x, R_t_3, R_t_3);
    matd_t* R_t_1 = matd_vec_normalize(R_t_1_tmp);
    matd_destroy(e_x);
    matd_destroy(R_t_1_tmp);

    matd_t* R_t_2 = matd_crossproduct(R_t_3, R_t_1);

    matd_t* R_t = matd_create_data(3, 3, (double[]) {
            MATD_EL(R_t_1, 0, 0), MATD_EL(R_t_1, 0, 1), MATD_EL(R_t_1, 0, 2),
            MATD_EL(R_t_2, 0, 0), MATD_EL(R_t_2, 0, 1), MATD_EL(R_t_2, 0, 2),
            MATD_EL(R_t_3, 0, 0), MATD_EL(R_t_3, 0, 1), MATD_EL(R_t_3, 0, 2)});
    matd_destroy(R_t_1);
    matd_destroy(R_t_2);
    matd_destroy(R_t_3);

    // 2. Find R_z
    matd_t* R_1_prime = matd_multiply(R_t, R);
    double r31 = MATD_EL(R_1_prime, 2, 0);
    double r32 = MATD_EL(R_1_prime, 2, 1);
    double hypotenuse = sqrt(r31*r31 + r32*r32);
    if (hypotenuse < 1e-100) {
        r31 = 1;
        r32 = 0;
        hypotenuse = 1;
    }
    matd_t* R_z = matd_create_data(3, 3, (double[]) {
            r31/hypotenuse, -r32/hypotenuse, 0,
            r32/hypotenuse, r31/hypotenuse, 0,
            0, 0, 1});

    // 3. Calculate parameters of Eos
    matd_t* R_trans = matd_multiply(R_1_prime, R_z);
    double sin_gamma = -MATD_EL(R_trans, 0, 1);
    double cos_gamma = MATD_EL(R_trans, 1, 1);
    matd_t* R_gamma = matd_create_data(3, 3, (double[]) {
            cos_gamma, -sin_gamma, 0,
            sin_gamma, cos_gamma, 0,
            0, 0, 1});

    double sin_beta = -MATD_EL(R_trans, 2, 0);
    double cos_beta = MATD_EL(R_trans, 2, 2);
    double t_initial = atan2(sin_beta, cos_beta);
    matd_destroy(R_trans);

    matd_t** v_trans = malloc(sizeof(matd_t *)*n_points);
    matd_t** p_trans = malloc(sizeof(matd_t *)*n_points);
    matd_t** F_trans = malloc(sizeof(matd_t *)*n_points);
    matd_t* avg_F_trans = matd_create(3, 3);
    for (int i = 0; i < n_points; i++) {
        p_trans[i] = matd_op("M'*M", R_z, p[i]);
        v_trans[i] = matd_op("M*M", R_t, v[i]);
        F_trans[i] = calculate_F(v_trans[i]);
        matd_add_inplace(avg_F_trans, F_trans[i]);
    }
    matd_scale_inplace(avg_F_trans, 1.0/n_points);

    matd_t* G = matd_op("(M-M)^-1", I3, avg_F_trans);
    matd_scale_inplace(G, 1.0/n_points);

    matd_t* M1 = matd_create_data(3, 3, (double[]) {
            0, 0, 2,
            0, 0, 0,
            -2, 0, 0});
    matd_t* M2 = matd_create_data(3, 3, (double[]) {
            -1, 0, 0,
            0, 1, 0,
            0, 0, -1});

    matd_t* b0 = matd_create(3, 1);
    matd_t* b1 = matd_create(3, 1);
    matd_t* b2 = matd_create(3, 1);
    for (int i = 0; i < n_points; i++) {
        matd_t* op_tmp1 = matd_op("(M-M)MM", F_trans[i], I3, R_gamma, p_trans[i]);
        matd_t* op_tmp2 = matd_op("(M-M)MMM", F_trans[i], I3, R_gamma, M1, p_trans[i]);
        matd_t* op_tmp3 = matd_op("(M-M)MMM", F_trans[i], I3, R_gamma, M2, p_trans[i]);

        matd_add_inplace(b0, op_tmp1);
        matd_add_inplace(b1, op_tmp2);
        matd_add_inplace(b2, op_tmp3);

        matd_destroy(op_tmp1);
        matd_destroy(op_tmp2);
        matd_destroy(op_tmp3);
    }
    matd_t* b0_ = matd_multiply(G, b0);
    matd_t* b1_ = matd_multiply(G, b1);
    matd_t* b2_ = matd_multiply(G, b2);

    double a0 = 0;
    double a1 = 0;
    double a2 = 0;
    double a3 = 0;
    double a4 = 0;
    for (int i = 0; i < n_points; i++) {
        matd_t* c0 = matd_op("(M-M)(MM+M)", I3, F_trans[i], R_gamma, p_trans[i], b0_);
        matd_t* c1 = matd_op("(M-M)(MMM+M)", I3, F_trans[i], R_gamma, M1, p_trans[i], b1_);
        matd_t* c2 = matd_op("(M-M)(MMM+M)", I3, F_trans[i], R_gamma, M2, p_trans[i], b2_);

        a0 += matd_to_double(matd_op("M'M", c0, c0));
        a1 += matd_to_double(matd_op("2M'M", c0, c1));
        a2 += matd_to_double(matd_op("M'M+2M'M", c1, c1, c0, c2));
        a3 += matd_to_double(matd_op("2M'M", c1, c2));
        a4 += matd_to_double(matd_op("M'M", c2, c2));

        matd_destroy(c0);
        matd_destroy(c1);
        matd_destroy(c2);
    }

    matd_destroy(b0);
    matd_destroy(b1);
    matd_destroy(b2);
    matd_destroy(b0_);
    matd_destroy(b1_);
    matd_destroy(b2_);

    for (int i = 0; i < n_points; i++) {
        matd_destroy(p_trans[i]);
        matd_destroy(v_trans[i]);
        matd_destroy(F_trans[i]);
    }
    free(p_trans);
    free(v_trans);
    free(F_trans);
    matd_destroy(avg_F_trans);
    matd_destroy(G);


    // 4. Solve for minima of Eos.
    double p0 = a1;
    double p1 = 2*a2 - 4*a0;
    double p2 = 3*a3 - 3*a1;
    double p3 = 4*a4 - 2*a2;
    double p4 = -a3;

    double roots[4];
    int n_roots;
    solve_poly_approx((double []) {p0, p1, p2, p3, p4}, 4, roots, &n_roots);

    double minima[4];
    int n_minima = 0;
    for (int i = 0; i < n_roots; i++) {
        double t1 = roots[i];
        double t2 = t1*t1;
        double t3 = t1*t2;
        double t4 = t1*t3;
        double t5 = t1*t4;
        // Check extrema is a minima.
        if (a2 - 2*a0 + (3*a3 - 6*a1)*t1 + (6*a4 - 8*a2 + 10*a0)*t2 + (-8*a3 + 6*a1)*t3 + (-6*a4 + 3*a2)*t4 + a3*t5 >= 0) {
            // And that it corresponds to an angle different than the known minimum.
            double t = 2*atan(roots[i]);
            // We only care about finding a second local minima which is qualitatively
            // different than the first.
            if (fabs(t - t_initial) > 0.1) {
                minima[n_minima++] = roots[i];
            }
        }
    }

    // 5. Get poses for minima.
    matd_t* ret = NULL;
    if (n_minima == 1) {
        double t = minima[0];
        matd_t* R_beta = matd_copy(M2);
        matd_scale_inplace(R_beta, t);
        matd_add_inplace(R_beta, M1);
        matd_scale_inplace(R_beta, t);
        matd_add_inplace(R_beta, I3);
        matd_scale_inplace(R_beta, 1/(1 + t*t));
        ret = matd_op("M'MMM'", R_t, R_gamma, R_beta, R_z);
        matd_destroy(R_beta);
    } else if (n_minima > 1)  {
        // This can happen if our prior pose estimate was not very good.
        fprintf(stderr, "Error, more than one new minima found.\n");
    }
    matd_destroy(I3);
    matd_destroy(M1);
    matd_destroy(M2);
    matd_destroy(R_t);
    matd_destroy(R_gamma);
    matd_destroy(R_z);
    matd_destroy(R_1_prime);
    return ret;
}

/**
 * Estimate pose of the tag using the homography method.
 */
void estimate_pose_for_tag_homography(apriltag_detection_info_t* info, apriltag_pose_t* solution) {
    double scale = info->tagsize/2.0;

    matd_t *M_H = homography_to_pose(info->det->H, -info->fx, info->fy, info->cx, info->cy);
    MATD_EL(M_H, 0, 3) *= scale;
    MATD_EL(M_H, 1, 3) *= scale;
    MATD_EL(M_H, 2, 3) *= scale;

    matd_t* fix = matd_create(4, 4);
    MATD_EL(fix, 0, 0) = 1;
    MATD_EL(fix, 1, 1) = -1;
    MATD_EL(fix, 2, 2) = -1;
    MATD_EL(fix, 3, 3) = 1;

    matd_t* initial_pose = matd_multiply(fix, M_H);
    matd_destroy(M_H);
    matd_destroy(fix);

    solution->R = matd_create(3, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            MATD_EL(solution->R, i, j) = MATD_EL(initial_pose, i, j);
        }
    }

    solution->t = matd_create(3, 1);
    for (int i = 0; i < 3; i++) {
        MATD_EL(solution->t, i, 0) = MATD_EL(initial_pose, i, 3);
    }
    matd_destroy(initial_pose);
}

/**
 * Estimate tag pose using orthogonal iteration.
 */
void estimate_tag_pose_orthogonal_iteration(
        apriltag_detection_info_t* info,
        double* err1,
        apriltag_pose_t* solution1,
        double* err2,
        apriltag_pose_t* solution2,
        int nIters) {
    double scale = info->tagsize/2.0;
    matd_t* p[4] = {
        matd_create_data(3, 1, (double[]) {-scale, scale, 0}),
        matd_create_data(3, 1, (double[]) {scale, scale, 0}),
        matd_create_data(3, 1, (double[]) {scale, -scale, 0}),
        matd_create_data(3, 1, (double[]) {-scale, -scale, 0})};
    matd_t* v[4];
    for (int i = 0; i < 4; i++) {
        v[i] = matd_create_data(3, 1, (double[]) {
                (info->det->p[i][0] - info->cx)/info->fx, (info->det->p[i][1] - info->cy)/info->fy, 1});
    }

    estimate_pose_for_tag_homography(info, solution1);
    *err1 = orthogonal_iteration(v, p, &solution1->t, &solution1->R, 4, nIters);
    solution2->R = fix_pose_ambiguities(v, p, solution1->t, solution1->R, 4);
    if (solution2->R) {
        solution2->t = matd_create(3, 1);
        *err2 = orthogonal_iteration(v, p, &solution2->t, &solution2->R, 4, nIters);
    } else {
        *err2 = HUGE_VAL;
    }

    for (int i = 0; i < 4; i++) {
        matd_destroy(p[i]);
        matd_destroy(v[i]);
    }
}

/**
 * Estimate tag pose.
 */
double estimate_tag_pose(apriltag_detection_info_t* info, apriltag_pose_t* pose) {
    double err1, err2;
    apriltag_pose_t pose1, pose2;
    estimate_tag_pose_orthogonal_iteration(info, &err1, &pose1, &err2, &pose2, 50);
    if (err1 <= err2) {
        pose->R = pose1.R;
        pose->t = pose1.t;
        if (pose2.R) {
            matd_destroy(pose2.t);
        }
        matd_destroy(pose2.R);
        return err1;
    } else {
        pose->R = pose2.R;
        pose->t = pose2.t;
        matd_destroy(pose1.R);
        matd_destroy(pose1.t);
        return err2;
    }
}
