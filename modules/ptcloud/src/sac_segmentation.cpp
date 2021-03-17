// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/calib3d.hpp" // lmsolver
#include "opencv2/ptcloud/sac_segmentation.hpp" // public api
#include "opencv2/ptcloud/sac_utils.hpp" // api for tests & samples
#include "opencv2/surface_matching/ppf_helpers.hpp" // ply,normals

#include <numeric>
#include <iostream>
#include <limits>
#include <utility>

namespace cv {
namespace ptcloud {

SACModel::SACModel(int typ) {
    type = typ;
    score = std::make_pair(0,1000);
}

inline
bool fequal(float a, float b) {
    return std::abs(a - b) <= std::numeric_limits<float>::epsilon();
}

inline
bool fequal(const Vec3f &p1, const Vec3f &p2) {
    return (fequal(p1[0], p2[0]) && fequal(p1[1], p2[1]) && fequal(p1[2], p2[2]));
}

struct box {
    Point3f m, M;
    double r;
};

static box bbox(const Mat &cloud) {
    box b;
    b.m = Point3f(9999,9999,9999);
    b.M = Point3f(-9999,-9999,-9999);
    for (size_t i=0; i<cloud.total(); i++) {
        const Point3f &p = cloud.at<Point3f>(int(i));
        b.M.x = std::max(b.M.x,p.x);
        b.M.y = std::max(b.M.y,p.y);
        b.M.z = std::max(b.M.z,p.z);
        b.m.x = std::min(b.m.x,p.x);
        b.m.y = std::min(b.m.y,p.y);
        b.m.z = std::min(b.m.z,p.z);
    }
    b.r = (norm(b.M - b.m) / 2);
    return b;
}


template<class T>
static void knuth_shuffle(std::vector<T> &vec) {
    for (size_t t=0; t<vec.size(); t++ ) {
        int r = theRNG().uniform(0, (int)vec.size());
        std::swap(vec[t], vec[r]);
    }
}


static double stopping(size_t inliers, size_t cloud_size, size_t m, double nu=0.99) {
    double e = double(inliers) / cloud_size;
    double a = log(1.0 - nu);
    double b = log(1.0 - std::pow(e,m));
    return a/b;
}

//
// SPRT impl taken from calib3d/usac
//
struct Error_ {
    virtual ~Error_() {}
    // set model to use getError() function
    virtual void setModelParameters (const Mat &model) = 0;
    // returns error of point wih @point_idx w.r.t. model
    virtual float getError (int point_idx) = 0;
};

struct PlaneModel : public Error_ {
    Mat_<Point3f> pts;
    Mat_<double> model;
    PlaneModel(const Mat &cloud) : pts(cloud) {}

    virtual void setModelParameters (const Mat &model_) CV_OVERRIDE {model = model_;}
    virtual float getError (int point_idx)  CV_OVERRIDE {
        Point3f p = pts(point_idx);
        double a = model(0) * p.x;
        double b = model(1) * p.y;
        double c = model(2) * p.z;
        double d = a + b + c + model(3);
        return (float)abs(d);
    }
};

struct SphereModel : public Error_ {
    Mat_<Point3f> pts;
    Mat_<double> model;
    SphereModel(const Mat &cloud) : pts(cloud) {}

    virtual void setModelParameters (const Mat &model_) CV_OVERRIDE {model = model_;}
    virtual float getError (int point_idx)  CV_OVERRIDE {
        Point3f p = pts(point_idx);
        Point3f center(float(model(0)),float(model(1)),float(model(2)));
        double distanceFromCenter  = norm(p - center);
        double distanceFromSurface = fabs(distanceFromCenter - model(3));

        return (float)distanceFromSurface;
    }
};

struct CylinderModel : public Error_ {
    Mat_<Point3f> pts, normals;
    Mat_<double> model;
    double weight;
    Point3d pt_on_axis, axis_dir;
    double rsqr;
    CylinderModel(const Mat &cloud, const Mat &nrm, double weight_) : pts(cloud), normals(nrm), weight(weight_) {}

    virtual void setModelParameters (const Mat &model_) CV_OVERRIDE {
        model = model_;
        pt_on_axis = Point3d(model(0),model(1),model(2));
        axis_dir = Point3d(model(3),model(4),model(5));
        rsqr = model(6)*model(6);
    }
    virtual float getError (int point_idx) CV_OVERRIDE {
        Point3d pt = pts(point_idx);
        Point3d a_cross = axis_dir.cross(pt_on_axis - pt);
        double distanceFromAxis = fabs(a_cross.dot(a_cross) / axis_dir.dot(axis_dir));
        double distanceFromSurface = fabs(distanceFromAxis - rsqr);

        double d_normal = 0.0;
        if ((! normals.empty()) && (weight > 0)) {
            // Calculate the point's projection on the cylinder axis
            double dist = (pt.dot(axis_dir) - pt_on_axis.dot(axis_dir));
            Point3d pt_proj = pt_on_axis + dist * axis_dir;
            Point3d dir = pt - pt_proj;
            dir = dir / sqrt(dir.dot(dir));
            // Calculate the angular distance between the point normal and the (dir=pt_proj->pt) vector
            double rad = normals(point_idx).dot(dir);
            if (rad < -1.0) rad = -1.0;
            if (rad > 1.0) rad = 1.0;
            d_normal = fabs(acos (rad));
            // convert range 0 to PI/2
            d_normal = (std::min) (d_normal, CV_PI - d_normal);
        }

        // calculate overall distance as weighted sum of the two distances.
        return (float)fabs(weight * d_normal + (1 - weight) * distanceFromSurface);
    }
};



struct SPRT_history {
    /*
     * delta:
     * The probability of a data point being consistent
     * with a ‘bad’ model is modeled as a probability of
     * a random event with Bernoulli distribution with parameter
     * δ : p(1|Hb) = δ.

     * epsilon:
     * The probability p(1|Hg) = ε
     * that any randomly chosen data point is consistent with a ‘good’ model
     * is approximated by the fraction of inliers ε among the data
     * points

     * A is the decision threshold, the only parameter of the Adapted SPRT
     */
    double epsilon, delta, A;
    // number of samples processed by test
    int tested_samples; // k
    SPRT_history () : epsilon(0), delta(0), A(0) {
        tested_samples = 0;
    }
};

///////////////////////////////////// SPRT IMPL //////////////////////////////////////////
struct SPRTImpl : public SPRT {
    RNG rng;
    const int points_size;
    int highest_inlier_number, current_sprt_idx; // i
    // time t_M needed to instantiate a model hypothesis given a sample
    // Let m_S be the number of models that are verified per sample
    const double inlier_threshold, norm_thr, one_over_thr, t_M, m_S;

    double lowest_sum_errors, current_epsilon, current_delta, current_A,
            delta_to_epsilon, complement_delta_to_complement_epsilon;

    std::vector<SPRT_history> sprt_histories;

    bool last_model_is_good;

    const double log_eta_0;
    const int sample_size, MAX_ITERATIONS;
    double lambda, sum_errors;
    int tested_inliers;

    SACScore score;
    const ScoreMethod score_type;

    SPRTImpl (int state, int points_size_,
          double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
          double time_sample, double avg_num_models, int sample_size_, int max_iterations_, ScoreMethod score_type_)
        : rng(state),
          points_size(points_size_),
          inlier_threshold (inlier_threshold_),
          norm_thr(inlier_threshold_*9/4),
          one_over_thr (1/norm_thr),
          t_M(time_sample),
          m_S(avg_num_models),
          log_eta_0(log(1-.99/*confidence*/)),
          sample_size (sample_size_),
          MAX_ITERATIONS(max_iterations_),
          score_type(score_type_) {
        // reserve (approximately) some space for sprt vector.
        sprt_histories.reserve(20);

        createTest(prob_pt_of_good_model, prob_pt_of_bad_model);

        highest_inlier_number = 0;
        lowest_sum_errors = std::numeric_limits<double>::max();
        last_model_is_good = false;
    }

    /*
     * add a data point, and recalculate lambda
     * returns: whether to continue evaluating this model
     * @model: model to verify
     *                      p(x(r)|Hb)                  p(x(j)|Hb)
     * lambda(j) = Product (----------) = lambda(j-1) * ----------
     *                      p(x(r)|Hg)                  p(x(j)|Hg)
     * Set j = 1
     * 1.  Check whether j-th data point is consistent with the
     * model
     * 2.  Compute the likelihood ratio λj eq. (1)
     * 3.  If λj >  A, decide the model is ’bad’ (model ”re-jected”),
     * else increment j or continue testing
     * 4.  If j = N the number of correspondences decide model ”accepted”
     */
    bool addDataPoint(int tested_point, double error) CV_OVERRIDE {
        if (tested_point==0) {
            lambda = 1;
            sum_errors = 0;
            tested_inliers = 0;
            lowest_sum_errors = std::numeric_limits<double>::max();
        }
        if (error < inlier_threshold) {
            tested_inliers++;
            lambda *= delta_to_epsilon;
        } else {
            lambda *= complement_delta_to_complement_epsilon;
            // since delta is always higher than epsilon, then lambda can increase only
            // when point is not consistent with model
            if (lambda > current_A) {
                return false;
            }
        }
        if (score_type == ScoreMethod::SCORE_METHOD_MSAC) {
            if (error < norm_thr)
                sum_errors -= (1 - error * one_over_thr);
            if (sum_errors - points_size + tested_point > lowest_sum_errors) {
                return false;
            }
        } else if (score_type == ScoreMethod::SCORE_METHOD_RANSAC) {
            if (tested_inliers + points_size - tested_point < highest_inlier_number) {
                return false;
            }
        }
        return true;
    }

    /* Verifies model and returns model score.
     * Return: true if model is good, false - otherwise.
     */
    bool isModelGood(int tested_point) CV_OVERRIDE {
        last_model_is_good = tested_point == points_size;
        // increase number of samples processed by current test
        sprt_histories[current_sprt_idx].tested_samples++;
        if (last_model_is_good) {
            score.first = tested_inliers;
            if (score_type == ScoreMethod::SCORE_METHOD_MSAC) {
                score.second = sum_errors;
                if (lowest_sum_errors > sum_errors)
                    lowest_sum_errors = sum_errors;
            } else if (score_type == ScoreMethod::SCORE_METHOD_RANSAC)
                score.second = -static_cast<double>(tested_inliers);

            const double new_epsilon = static_cast<double>(tested_inliers) / points_size;
            if (new_epsilon > current_epsilon) {
                highest_inlier_number = tested_inliers; // update max inlier number
                /*
                 * Model accepted and the largest support so far:
                 * design (i+1)-th test (εi + 1= εˆ, δi+1 = δ, i := i + 1).
                 * Store the current model parameters θ
                 */
                createTest(new_epsilon, current_delta);
            }
        } else {
            /*
             * Since almost all tested models are ‘bad’, the probability
             * δ can be estimated as the average fraction of consistent data points
             * in rejected models.
             */
            // add 1 to tested_point, because loop over tested_point starts from 0
            const double delta_estimated = static_cast<double> (tested_inliers) / (tested_point+1);
            if (delta_estimated > 0 && fabs(current_delta - delta_estimated)
                                       / current_delta > 0.05) {
                /*
                 * Model rejected: re-estimate δ. If the estimate δ_ differs
                 * from δi by more than 5% design (i+1)-th test (εi+1 = εi,
                 * δi+1 = δˆ, i := i + 1)
                 */
                createTest(current_epsilon, delta_estimated);
            }
        }
        return last_model_is_good;
    }

    SACScore getScore() CV_OVERRIDE {
        return score;
    }
    /*
     * Termination criterion:
     * l is number of tests
     * n(l) = Product from i = 0 to l ( 1 - P_g (1 - A(i)^(-h(i)))^k(i) )
     * log n(l) = sum from i = 0 to l k(i) * ( 1 - P_g (1 - A(i)^(-h(i))) )
     *
     *        log (n0) - log (n(l-1))
     * k(l) = -----------------------  (9)
     *          log (1 - P_g*A(l)^-1)
     *
     * A is decision threshold
     * P_g is probability of good model.
     * k(i) is number of samples verified by i-th sprt.
     * n0 is typically set to 0.05
     * this equation does not have to be evaluated before nR < n0
     * nR = (1 - P_g)^k
     */
    int update (int inlier_size) CV_OVERRIDE {
        if (sprt_histories.empty())
            return std::min(MAX_ITERATIONS, getStandardUpperBound(inlier_size));

        const double epsilon = static_cast<double>(inlier_size) / points_size; // inlier probability
        const double P_g = pow (epsilon, sample_size); // probability of good sample

        double log_eta_lmin1 = 0;

        int total_number_of_tested_samples = 0;
        const int sprts_size_min1 = static_cast<int>(sprt_histories.size())-1;
        if (sprts_size_min1 < 0) return getStandardUpperBound(inlier_size);
        // compute log n(l-1), l is number of tests
        for (int test = 0; test < sprts_size_min1; test++) {
            log_eta_lmin1 += log (1 - P_g * (1 - pow (sprt_histories[test].A,
             -computeExponentH(sprt_histories[test].epsilon, epsilon,sprt_histories[test].delta))))
                         * sprt_histories[test].tested_samples;
            total_number_of_tested_samples += sprt_histories[test].tested_samples;
        }

        // Implementation note: since η > ηR the equation (9) does not have to be evaluated
        // before ηR < η0 is satisfied.
        if (std::pow(1 - P_g, total_number_of_tested_samples) < log_eta_0)
            return std::min(MAX_ITERATIONS, getStandardUpperBound(inlier_size));
        // use decision threshold A for last test (l-th)
        const double predicted_iters_sprt = (log_eta_0 - log_eta_lmin1) /
                log (1 - P_g * (1 - 1 / sprt_histories[sprts_size_min1].A)); // last A
        //cout << "pred " << predicted_iters_sprt << endl;
        if (std::isnan(predicted_iters_sprt) || std::isinf(predicted_iters_sprt))
            return getStandardUpperBound(inlier_size);

        if (predicted_iters_sprt < 0) return 0;
        // compare with standard upper bound
        if (predicted_iters_sprt < MAX_ITERATIONS)
            return std::min(static_cast<int>(predicted_iters_sprt),
                    getStandardUpperBound(inlier_size));
        return getStandardUpperBound(inlier_size);
    }

private:
    // Saves sprt test to sprt history and update current epsilon, delta and threshold.
    void createTest (double epsilon, double delta) {
        // if epsilon is closed to 1 then set them to 0.99 to avoid numerical problems
        if (epsilon > 0.999999) epsilon = 0.999;
        // delta can't be higher than epsilon, because ratio delta / epsilon will be greater than 1
        if (epsilon < delta) delta = epsilon-0.0001;
        // avoid delta going too high as it is very unlikely
        // e.g., 30% of points are consistent with bad model is not very real
        if (delta   > 0.3) delta = 0.3;

        SPRT_history new_sprt_history;
        new_sprt_history.epsilon = epsilon;
        new_sprt_history.delta = delta;
        new_sprt_history.A = estimateThresholdA (epsilon, delta);

        sprt_histories.emplace_back(new_sprt_history);

        current_A = new_sprt_history.A;
        current_delta = delta;
        current_epsilon = epsilon;

        delta_to_epsilon = delta / epsilon;
        complement_delta_to_complement_epsilon = (1 - delta) / (1 - epsilon);
        current_sprt_idx = static_cast<int>(sprt_histories.size()) - 1;
    }

    /*
    * A(0) = K1/K2 + 1
    * A(n+1) = K1/K2 + 1 + log (A(n))
    * K1 = t_M / P_g
    * K2 = m_S/(P_g*C)
    * t_M is time needed to instantiate a model hypotheses given a sample
    * P_g = epsilon ^ m, m is the number of data point in the Ransac sample.
    * m_S is the number of models that are verified per sample.
    *                   p (0|Hb)                  p (1|Hb)
    * C = p(0|Hb) log (---------) + p(1|Hb) log (---------)
    *                   p (0|Hg)                  p (1|Hg)
    */
    double estimateThresholdA (double epsilon, double delta) {
        const double C = (1 - delta) * log ((1 - delta) / (1 - epsilon)) +
                         delta * (log(delta / epsilon));
        // K = K1/K2 + 1 = (t_M / P_g) / (m_S / (C * P_g)) + 1 = (t_M * C)/m_S + 1
        const double K = t_M * C / m_S + 1;
        double An, An_1 = K;
        // compute A using a recursive relation
        // A* = lim(n->inf)(An), the series typically converges within 4 iterations
        for (int i = 0; i < 10; i++) {
            An = K + log(An_1);
            if (fabs(An - An_1) < FLT_EPSILON)
                break;
            An_1 = An;
        }
        return An;
    }

    inline int getStandardUpperBound(int inlier_size) const {
        const double predicted_iters = log_eta_0 / log(1 - std::pow
                (static_cast<double>(inlier_size) / points_size, sample_size));
        return (! std::isinf(predicted_iters) && predicted_iters < MAX_ITERATIONS) ?
                static_cast<int>(predicted_iters) : MAX_ITERATIONS;
    }
    /*
     * h(i) must hold
     *
     *     δ(i)                  1 - δ(i)
     * ε (-----)^h(i) + (1 - ε) (--------)^h(i) = 1
     *     ε(i)                  1 - ε(i)
     *
     * ε * a^h + (1 - ε) * b^h = 1
     * Has numerical solution.
     */
    static double computeExponentH (double epsilon, double epsilon_new, double delta) {
        const double a = log (delta / epsilon); // log likelihood ratio
        const double b = log ((1 - delta) / (1 - epsilon));

        const double x0 = log (1 / (1 - epsilon_new)) / b;
        const double v0 = epsilon_new * exp (x0 * a);
        const double x1 = log ((1 - 2*v0) / (1 - epsilon_new)) / b;
        const double v1 = epsilon_new * exp (x1 * a) + (1 - epsilon_new) * exp(x1 * b);
        const double h = x0 - (x0 - x1) / (1 + v0 - v1) * v0;

        if (std::isnan(h))
            // The equation always has solution for h = 0
            // ε * a^0 + (1 - ε) * b^0 = 1
            // ε + 1 - ε = 1 -> 1 = 1
            return 0;
        return h;
    }
};

Ptr<SPRT> SPRT::create(int state, int points_size_,
      double inlier_threshold_, double prob_pt_of_good_model, double prob_pt_of_bad_model,
      double time_sample, double avg_num_models, int sample_size_, int max_iterations_, ScoreMethod score_type_) {
    return makePtr<SPRTImpl>(state, points_size_,
      inlier_threshold_, prob_pt_of_good_model, prob_pt_of_bad_model,
      time_sample, avg_num_models, sample_size_, max_iterations_, score_type_);
}

SACScore getInliers(int model_type, const std::vector<double> &coeffs, const Mat &cloud, const Mat &normals, const std::vector<int> &indices, const double threshold, const double best_inliers, std::vector<int>& inliers, double normal_distance_weight_, Ptr<SPRT> sprt) {
    inliers.clear();

    Ptr<Error_> mdl;
    switch (model_type) {
        case PLANE_MODEL: mdl = makePtr<PlaneModel>(cloud); break;
        case SPHERE_MODEL: mdl = makePtr<SphereModel>(cloud); break;
        case CYLINDER_MODEL: mdl = makePtr<CylinderModel>(cloud, normals, normal_distance_weight_); break;
        default: CV_Error(215, format("unsupported model type %d", model_type).c_str());
    }
    mdl->setModelParameters(Mat(coeffs));

    double msac=0; // use same calculation as SPRT
    double norm_thr = (threshold*9/4);
    double one_over_thr = (1/norm_thr);
    size_t num_points = indices.size();

    size_t i;
    for (i=0; i < num_points; i++) {
        if (i == num_points/4) {
            if (inliers.size() < best_inliers/8) {
                break;
            }
        }
        double distance = mdl->getError(indices[i]);

        if (!sprt.empty()) {
            if (! sprt->addDataPoint(int(i),distance)) {
                break;
            }
        }

        if (distance < norm_thr)
            msac -= (1 - distance * one_over_thr);

        if (distance < threshold)
            inliers.emplace_back(indices[i]);
    }

    if (!sprt.empty()) {
        if (sprt->isModelGood(int(i)))
            return sprt->getScore();
    }

    return SACScore((double)inliers.size(), msac);
}


// http://ambrsoft.com/TrigoCalc/Sphere/Spher3D_.htm
static bool getSphereFromPoints(const Mat &cloud, const std::vector<int> &inliers, std::vector<double> &coeffs) {
    const Vec3f* points = cloud.ptr<Vec3f>(0);

    Mat temp(4, 4, CV_32FC1, 1.0f); // last column must be 1
    for (int i = 0; i < 4; i++) {
        size_t point_idx = inliers[i];
        float* tempi = temp.ptr<float>(i);
        tempi[0] = (float) points[point_idx][0];
        tempi[1] = (float) points[point_idx][1];
        tempi[2] = (float) points[point_idx][2];
        tempi[3] = 1;
    }
    double m11 = determinant(temp);
    if (fequal(float(m11), 0)) return false; // no sphere exists

    for (int i = 0; i < 4; i++) {
        size_t point_idx = inliers[i];
        float* tempi = temp.ptr<float>(i);

        tempi[0] = (float) points[point_idx][0] * (float) points[point_idx][0]
                 + (float) points[point_idx][1] * (float) points[point_idx][1]
                 + (float) points[point_idx][2] * (float) points[point_idx][2];
    }
    double m12 = determinant(temp);

    for(int i = 0; i < 4; i++) {
        size_t point_idx = inliers[i];
        float* tempi = temp.ptr<float>(i);

        tempi[1] = tempi[0];
        tempi[0] = (float) points[point_idx][0];
    }
    double m13 = determinant(temp);

    for (int i = 0; i < 4; i++) {
        size_t point_idx = inliers[i];
        float* tempi = temp.ptr<float>(i);

        tempi[2] = tempi[1];
        tempi[1] = (float) points[point_idx][1];
    }
    double m14 = determinant(temp);

    for (int i = 0; i < 4; i++) {
        size_t point_idx = inliers[i];
        float* tempi = temp.ptr<float>(i);

        tempi[0] = tempi[2];
        tempi[1] = (float) points[point_idx][0];
        tempi[2] = (float) points[point_idx][1];
        tempi[3] = (float) points[point_idx][2];
    }
    double m15 = determinant(temp);

    Point3d center(
        0.5 * m12 / m11,
        0.5 * m13 / m11,
        0.5 * m14 / m11);

    double radius = std::sqrt(
        center.x * center.x +
        center.y * center.y +
        center.z * center.z - m15 / m11);

    if (std::isnan(radius)) return false;

    coeffs = std::vector<double> {center.x, center.y, center.z, radius};
    return true;
}

static bool getPlaneFromPoints(const Mat &cloud, const std::vector<int> &inliers, std::vector<double> &coeffs) {
    // REF: https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
    const Vec3f* points = cloud.ptr<Vec3f>(0);
    Vec3d abc;
    Vec3f centroid(0, 0, 0);
    for (size_t idx : inliers)
        centroid += points[idx];

    centroid /= double(inliers.size());

    double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

    for (size_t idx : inliers) {
        Vec3f r = points[idx] - centroid;
        xx += r(0) * r(0);
        xy += r(0) * r(1);
        xz += r(0) * r(2);
        yy += r(1) * r(1);
        yz += r(1) * r(2);
        zz += r(2) * r(2);
    }

    double det_x = yy * zz - yz * yz;
    double det_y = xx * zz - xz * xz;
    double det_z = xx * yy - xy * xy;

    if (det_x > det_y && det_x > det_z) {
        abc = Vec3d(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
    } else if (det_y > det_z) {
        abc = Vec3d(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
    } else {
        abc = Vec3d(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
    }
    double magnitude_abc = sqrt(abc[0]*abc[0] + abc[1]*abc[1] + abc[2]*abc[2]);

    // Return invalid plane if the points don't span a plane.
    if (fequal(float(magnitude_abc), 0))
        return false;

    abc /= magnitude_abc;
    double d = -abc.dot(centroid);

    coeffs = std::vector<double>{abc[0], abc[1], abc[2], d};
    return true;
}

// from 2 points & normals
static bool getCylinderFromPoints(const Mat &cloud, const Mat &normals_cld, const std::vector<int> &inliers, std::vector<double> & model_coefficients) {
    CV_Assert(inliers.size() == 2);

    const Vec3f* points = cloud.ptr<Vec3f>(0);
    const Vec3f* normals = normals_cld.ptr<Vec3f>(0);

    Vec3f p1 = points[inliers[0]];
    Vec3f p2 = points[inliers[1]];

    if (fequal(p1, p2))
        return false;

    Vec3f n1 (normals[inliers[0]]);
    Vec3f n2 (normals[inliers[1]]);
    Vec3f w = n1 + p1 - p2;

    float a = n1.dot(n1);
    float b = n1.dot(n2);
    float c = n2.dot(n2);
    float d = n1.dot(w);
    float e = n2.dot(w);
    float denominator = a*c - b*b;
    float sc, tc;
    // Compute the line parameters of the two closest points
    if (denominator < 1e-8) {          // The lines are almost parallel
        sc = 0.0f;
        tc = (b > c ? d / b : e / c);  // Use the largest denominator
    } else {
        sc = (b*e - c*d) / denominator;
        tc = (a*e - b*d) / denominator;
    }

    // point_on_axis, axis_direction
    Vec3f line_pt  = p1 + n1 + sc * n1;
    Vec3f line_dir = p2 + tc * n2 - line_pt;

    model_coefficients.resize (7);
    // point on line
    model_coefficients[0] = line_pt[0];
    model_coefficients[1] = line_pt[1];
    model_coefficients[2] = line_pt[2];

    double line_len_sqr = line_dir.dot(line_dir);
    double divide_by = std::sqrt (line_len_sqr);

    // direction of line;
    model_coefficients[3] = line_dir[0] / divide_by;
    model_coefficients[4] = line_dir[1] / divide_by;
    model_coefficients[5] = line_dir[2] / divide_by;
    Vec3f line_normal = line_dir.cross(line_pt - p1);

    // radius of cylinder
    double radius_squared = fabs(line_normal.dot(line_normal)) / line_len_sqr;
    if (fequal(float(radius_squared), 0))
        return false;
    model_coefficients[6] = sqrt(radius_squared);

    return (true);
}

// from 3 points
static bool getCylinderFromPoints(const Mat &cloud, const std::vector<int> &inliers, std::vector<double> & model_coefficients) {
    CV_Assert(inliers.size() == 3);

    const Vec3f* points = cloud.ptr<Vec3f>(0);

    // two points form a line on the hull of the cylinder
    Vec3f p1 = points[inliers[0]];
    Vec3f p2 = points[inliers[1]];
    // a third point on the opposite side
    Vec3f p3 = points[inliers[2]];
    // degenerate if 2 points are same
    if (fequal(p1, p2) || fequal(p1, p3) || fequal(p2, p3))
        return false;

    Vec3f dr = p2 - p1;
    normalize(dr, dr);

    // distance from p3 to line p1p2
    double a = p3.dot(dr);
    Vec3f  n = dr - a*p3;
    double r = std::sqrt(n.dot(n));

    // move line_point halfway along the normal (to the center of the cylinder)
    model_coefficients = std::vector<double> {
        p1[0] + n[0]/2, p1[1] + n[1]/2, p1[2]+n[2]/2,
        dr[0],  dr[1],  dr[2],
        r/2
    };
    return (true);
}

// pca on inlier points
// https://pointclouds.org/documentation/centroid_8hpp_source.html#l00485
static std::vector<double> optimizePlane(const Mat &pointcloud,
                                  const std::vector<int> &indices,
                                  const std::vector<double> &old_coeffs = {}) {
    CV_UNUSED(old_coeffs);

    const Point3f *cloud = pointcloud.ptr<Point3f>(0);

    // de-mean before accumulating values
    Point3f centroid(0, 0, 0);
    for (int idx : indices) {
        centroid += cloud[idx];
    }
    centroid /= double(indices.size());

    std::vector<double> accu(6, 0);
    for (const auto &index : indices) {
        const Point3f p = cloud[index] - centroid;
        accu[0] += p.x * p.x;
        accu[1] += p.x * p.y;
        accu[2] += p.x * p.z;
        accu[3] += p.y * p.y;
        accu[4] += p.y * p.z;
        accu[5] += p.z * p.z;
    }
    Mat(accu) /= float(indices.size());

    Mat_<float> covariance_matrix(3,3);
    covariance_matrix <<
        accu[0], accu[1], accu[2],
        accu[1], accu[3], accu[4],
        accu[2], accu[4], accu[5];

    Mat_<float> evec, eval;
    eigen(covariance_matrix, eval, evec);
    // the evec corresponding to the *smallest* eval
    Mat_<float> ev = evec.row(2);
    std::vector<double> coeffs = {
        ev(0), ev(1), ev(2),
        -1 * (ev(0)*centroid.x + ev(1)*centroid.y + ev(2)*centroid.z)
    };

    return coeffs;
}

struct SphereSolverCallback : LMSolver::Callback {
    Mat_<Point3f> P;
    SphereSolverCallback(const Mat &pts) : P(pts) {}

    bool compute (InputArray param, OutputArray err, OutputArray J) const CV_OVERRIDE {
        Mat_<double> in = param.getMat();

        Point3f c(float(in(0)),float(in(1)),float(in(2)));
        double r = in(3);

        // the levmarquard solver needs an error metrics for each callback
        int count = (int)P.total();
        err.create(count,1,CV_64F);
        Mat_<double> e = err.getMat();

        // but the jacobian matrix needs only to be (re)generated, if the error falls beyond some threshold
        Mat_<double> j;
        if (J.needed()) {
            J.create(count, (int)in.total(), CV_64F);
            j = J.getMat();
        }

        for (int i=0; i<count; i++) {
            Point3f dir = P(i) - c;
            Point3f u = c + r * dir / norm(dir);
            e(i) = norm(u-P(i)) + 0.000001;
            if (j.data) {
                j(i,0) = (u.x - P(i).x) / e(i);
                j(i,1) = (u.y - P(i).y) / e(i);
                j(i,2) = (u.z - P(i).z) / e(i);
                j(i,3) = -1;
            }
        }
        return true;
    }
};

static std::vector<double> optimizeSphere(const Mat &pointcloud,
                                   const std::vector<int> &indices,
                                   const std::vector<double> &old_coeffs = {}) {
    std::vector<double> new_coeffs = old_coeffs;
    if (new_coeffs.empty())
        new_coeffs = std::vector<double>(4,1.0);

    Mat_<Point3f> pts;
    for(auto i:indices)
        pts.push_back(pointcloud.at<Point3f>(i));

    Ptr<LMSolver::Callback> cb = makePtr<SphereSolverCallback>(pts);
    Ptr<LMSolver> lm = LMSolver::create(cb, 100);
    int n = lm->run(new_coeffs);
    CV_Check(n, n<100, "failed to converge");
    CV_Assert(!std::isnan(new_coeffs[4]));
    return new_coeffs;
}

// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4890955/
struct CylinderSolverCallback : LMSolver::Callback {
    Mat_<Point3f> P;
    CylinderSolverCallback(const Mat &pts) : P(pts) {}

    bool compute (InputArray param, OutputArray err, OutputArray J) const CV_OVERRIDE {
        Mat_<double> in = param.getMat();

        double x=in(0), y=in(1), z=in(2); // point on axis
        double a=in(3), b=in(4), c=in(5); // axis dir
        double r=in(6); // radius

        // normalize dir vec
        double len = sqrt(a*a+b*b+c*c) + 0.000001;
        a /= len; b /= len; c /= len;

        int count = (int)P.total();
        err.create(count, 1, CV_64F);
        Mat_<double> e = err.getMat();

        Mat_<double> j;
        if (J.needed()) {
            J.create(count, (int)in.total(), CV_64F);
            j = J.getMat();
        }

        for (int i=0; i<count; i++) {
            double xi = P(i).x, yi = P(i).y, zi=P(i).z;
            double u = c*(yi-y) - b*(zi-z); // dir.cross(pi-p)
            double v = a*(zi-z) - c*(xi-x);
            double w = b*(xi-x) - a*(yi-y);
            double f = sqrt(u*u + v*v + w*w) / len;  // ch 3.1
            double g = a*(xi-x) + b*(yi-y) + c*(zi-z);
            CV_Assert(f!=0);
            e(i) = f - r;
            if (j.data) {
                j(i,0) = (a*g - (xi - x)) / f;
                j(i,1) = (b*g - (yi - y)) / f;
                j(i,2) = (c*g - (zi - z)) / f;
                j(i,3) = g * j(i,0);
                j(i,4) = g * j(i,1);
                j(i,5) = g * j(i,2);
                j(i,6) = -1;
            }
        }
        return true;
    }
};

static std::vector<double> optimizeCylinder(const Mat &pointcloud,
                                     const std::vector<int> &indices,
                                     const std::vector<double> &old_coeffs = {}) {
    std::vector<double> new_coeffs = old_coeffs;
    if (new_coeffs.empty())
        new_coeffs = std::vector<double>(7,1.0);

    Mat_<Point3f> pts;
    for (auto i:indices)
        pts.push_back(pointcloud.at<Point3f>(i));

    Ptr<LMSolver::Callback> cb = makePtr<CylinderSolverCallback>(pts);
    Ptr<LMSolver> lm = LMSolver::create(cb, 100);
    int n = lm->run(new_coeffs);
    CV_Check(n, n<100, "failed to converge");

    // normalize direction vector:
    double len = norm(Point3d(new_coeffs[3],new_coeffs[4],new_coeffs[5]));
    new_coeffs[3] /= len; new_coeffs[4] /= len; new_coeffs[5] /= len;

    return new_coeffs;
}

std::vector<double> optimizeModel(int model_type, const Mat &cloud, const std::vector<int> &inliers_indices, const std::vector<double> &old_coeff) {
    switch (model_type) {
        case PLANE_MODEL:
            return optimizePlane(cloud, inliers_indices, old_coeff);
        case SPHERE_MODEL:
            return optimizeSphere(cloud, inliers_indices, old_coeff);
        case CYLINDER_MODEL:
            return optimizeCylinder(cloud, inliers_indices, old_coeff);
    //    default:
    //        CV_Error(215, format("invalid model_type %d", model_type).c_str());
    }
    return std::vector<double>();
}


// how many points to sample for a minimal hypothesis
static size_t rnd_model_points(int model_type, const Mat &normals) {
    size_t num_rnd_model_points =
        model_type==PLANE_MODEL  ? 3 :
        model_type==SPHERE_MODEL ? 4 :
        model_type==CYLINDER_MODEL ?
            size_t(normals.empty()
              ? 3  // from 3 points
              : 2  // 2 points and 2 normals
            ) : 0; // invalid
    CV_Assert(num_rnd_model_points != 0); // invalid model enum
    return num_rnd_model_points;
}

static std::vector<int> sampleHypothesis(RNG &rng, const Mat &cloud, const std::vector<int> &indices, size_t num_rnd_model_points, double max_napsac_radius) {
    std::vector<int> current_model_inliers;
    if (max_napsac_radius > 0) {
        // NAPSAC like sampling strategy,
        // assume good hypothesis inliers are locally close to each other
        // (enforce spatial proximity)
        int start = rng.uniform(0, (int)indices.size());
        current_model_inliers.emplace_back(start);
        Point3f a = cloud.at<Point3f>(indices[start]);
        for (int n=0; n<100; n++) {
            int next = rng.uniform(0, (int)indices.size());
            Point3f b = cloud.at<Point3f>(indices[next]);
            if (norm(a-b) > max_napsac_radius)
                continue;

            current_model_inliers.emplace_back(next);
            if (current_model_inliers.size() == num_rnd_model_points)
                break;
        }
    } else {
        // uniform sample from indices
        for (size_t j = 0; j < num_rnd_model_points; j++)
            current_model_inliers.emplace_back(rng.uniform(0, (int)indices.size()));
    }

    return current_model_inliers;
}

bool generateHypothesis(int model_type, const Mat &cloud, const std::vector<int> &current_model_inliers, const Mat &normals, double max_radius, double normal_distance_weight_, std::vector<double> &coefficients) {
    if (model_type == PLANE_MODEL) {
        return getPlaneFromPoints(cloud, current_model_inliers, coefficients);
    }

    if (model_type == SPHERE_MODEL) {
        bool valid = getSphereFromPoints(cloud, current_model_inliers, coefficients);
        if (!valid) return false;
        if (coefficients[3] > max_radius) return false;
        return true;
    }
    if (model_type == CYLINDER_MODEL) {
        if ((normals.total() > 0) && (normal_distance_weight_ > 0))
            return getCylinderFromPoints(cloud, normals, current_model_inliers, coefficients);
        else
            return getCylinderFromPoints(cloud, current_model_inliers, coefficients);
    }

    return false;
}

Mat generateNormals(const Mat &cloud) {
    // Reshape the cloud for Compute Normals Function
    Mat _cld_reshaped = cloud;
    if (_cld_reshaped.rows < _cld_reshaped.cols)
        _cld_reshaped = _cld_reshaped.t();
    _cld_reshaped = _cld_reshaped.reshape(1);

    // calculate normals from Knn cloud neighbours
    Mat _pointsAndNormals;
    ppf_match_3d::computeNormalsPC3d(_cld_reshaped, _pointsAndNormals, 8, false, Vec3d(0,0,0));

    Mat normals;
    Mat(_pointsAndNormals.colRange(3,6)).copyTo(normals);
    normals = normals.reshape(3, (int)cloud.total());
    return normals.t();
}


class SACModelFittingImpl : public SACModelFitting {
private:
    Mat cloud;
    Mat normals;
    int model_type;
    int method_type;
    int preemptive_count = 0;
    int min_inliers = 10;
    int max_iters;
    double threshold;
    double normal_distance_weight_ = 0;
    double max_sphere_radius = 60;
    double max_napsac_radius = 0;

    bool use_sprt = false;
    Ptr<SPRT> sprt;

    bool fit_ransac(const std::vector<int>& indices, std::vector<SACModel> &model_instances);
    bool fit_preemptive(const std::vector<int> &indices, std::vector<SACModel> &model_instances);

public:
    SACModelFittingImpl (InputArray cloud, int model_type = PLANE_MODEL, int method_type = SAC_METHOD_RANSAC, double threshold = 20,int max_iters = 1000);

    void segment(std::vector<SACModel> &model_instances, OutputArray new_cloud=noArray()) CV_OVERRIDE;

    void set_cloud(InputArray cloud, bool with_normals=false) CV_OVERRIDE;
    void set_model_type(int type) CV_OVERRIDE;
    void set_method_type(int type) CV_OVERRIDE;
    void set_threshold (double threshold) CV_OVERRIDE;
    void set_iterations (int iterations) CV_OVERRIDE;
    void set_normal_distance_weight(double weight) CV_OVERRIDE;
    void set_max_sphere_radius(double max_radius) CV_OVERRIDE;
    void set_max_napsac_radius(double max_radius) CV_OVERRIDE;
    void set_preemptive_count(int preemptive) CV_OVERRIDE;
    void set_min_inliers(int min_inliers) CV_OVERRIDE;
    void set_use_sprt(bool sprt) CV_OVERRIDE;
};


Ptr<SACModelFitting> SACModelFitting::create(InputArray cloud, int model_type, int method_type, double threshold, int max_iters) {
    return makePtr<SACModelFittingImpl>(cloud, model_type, method_type, threshold, max_iters);
}

SACModelFittingImpl::SACModelFittingImpl (InputArray inp_cloud, int set_model_type, int set_method_type, double set_threshold, int set_max_iters)
    : model_type(set_model_type), method_type(set_method_type), max_iters(set_max_iters), threshold(set_threshold) {
    // if there are normals, keep them
    Mat input_cloud = inp_cloud.getMat();
    bool with_normals = input_cloud.channels() == 1 && (input_cloud.cols == 6 || input_cloud.rows == 6);
    set_cloud(input_cloud, with_normals);
}

bool SACModelFittingImpl::fit_ransac(const std::vector<int> &indices, std::vector<SACModel> &model_instances) {
    RNG rng((uint64)-1);
    // Initialize the best future model.
    SACModel bestModel, curModel;

    size_t num_rnd_model_points = rnd_model_points(model_type, normals);
    CV_Assert(indices.size() >= num_rnd_model_points);

    int stop = (int)stopping(min_inliers, indices.size(), num_rnd_model_points);

    if (use_sprt) {
        sprt = SPRT::create(9999, (int)indices.size(),
              threshold, 0.01, 0.008,
              200, 3, (int)num_rnd_model_points, max_iters, (ScoreMethod)method_type);
    }

    for (int i = 0; i < std::min(stop, max_iters); ++i) {
        // generate model hypothesis
        curModel = SACModel(SacModelType(model_type));

        std::vector<int> hypothesis_inliers = sampleHypothesis(rng, cloud, indices, num_rnd_model_points, max_napsac_radius);
        if (hypothesis_inliers.size() != num_rnd_model_points)
            continue; // bad starting point for napsac

        bool valid_model = generateHypothesis(model_type, cloud, hypothesis_inliers, normals, max_sphere_radius, normal_distance_weight_, curModel.coefficients);
        if (!valid_model)
            continue;

        // check inliers for the current hypothesis
        curModel.score = getInliers(model_type, curModel.coefficients, cloud, normals, indices, threshold, bestModel.score.first, curModel.indices, normal_distance_weight_, sprt);

        if (curModel.indices.size() < size_t(min_inliers))
            continue;

        bool better = false;
        if (method_type == SAC_METHOD_RANSAC)
            better = bestModel.score.first < curModel.score.first;
        if (method_type == SAC_METHOD_MSAC)
            better = bestModel.score.second > curModel.score.second;

        if (better) {
            // apply local optimization
            // (this could run on a fraction of the indices, but right now it's fast enough to run on all of them)
            std::vector<double> new_coeff = optimizeModel(model_type, cloud, curModel.indices, curModel.coefficients);

            // check again if it improved
            std::vector<int> new_inliers;
            SACScore new_result = getInliers(model_type, new_coeff, cloud, normals, indices, threshold, curModel.score.first, new_inliers, normal_distance_weight_, sprt);

            if (method_type == SAC_METHOD_RANSAC)
                better = new_result.first > curModel.score.first;
            if (method_type == SAC_METHOD_MSAC)
                better = new_result.second < curModel.score.second;

            if (better) {
                curModel.score = new_result;
                curModel.coefficients = new_coeff;
                curModel.indices = new_inliers;
            }

            if (!sprt.empty())
                stop = i + (int)sprt->update((int)curModel.score.first);
            else
                stop = i + (int)stopping((size_t)curModel.score.first, indices.size(), num_rnd_model_points);

            bestModel = curModel;
        }
    }

    if (bestModel.coefficients.size() && (bestModel.indices.size() > size_t(min_inliers))) {
        model_instances.push_back(bestModel);
        return true;
    }
    return false;
}

// how many models to retain after each data block
// RaguramECCV08.pdf (5)
inline
double preemptive_func(size_t i, size_t M, size_t B) {
    return M * std::pow(2.0, -(double(i)/B));
}

// D. Nister, Preemptive RANSAC for live structure and motion estimation, in International Conference on Computer Vision (ICCV), 2003.
bool SACModelFittingImpl::fit_preemptive(const std::vector<int> &indices, std::vector<SACModel> &model_instances) {
    RNG rng((uint64)-1);

    size_t num_rnd_model_points = rnd_model_points(model_type, normals);
    if (indices.size() < num_rnd_model_points)
        return false;

    // generate model hypotheses in parallel
    std::vector<SACModel> preemptive;
    while (preemptive.size() < (size_t)preemptive_count) {
        SACModel model = SACModel(SacModelType(model_type));
        std::vector<int> hypothesis_inliers = sampleHypothesis(rng, cloud, indices, num_rnd_model_points, max_napsac_radius);
        if (hypothesis_inliers.size() != num_rnd_model_points)
            continue; // bad starting point for napsac

        bool valid_model = generateHypothesis(model_type, cloud, hypothesis_inliers, normals, max_sphere_radius, normal_distance_weight_, model.coefficients);
        if (!valid_model)
            continue;

        preemptive.push_back(model);
    }

    size_t preemptive_pos  = 0;   // "i", the current data position
    size_t preemptive_step = 100; // "B", the block size
    while ((preemptive.size() > 1) && (preemptive_pos < indices.size())) {

        // slice a data block
        std::vector<int> partial_set(
            indices.begin() + preemptive_pos,
            indices.begin() + std::min((preemptive_pos + preemptive_step), indices.size()));
        preemptive_pos += preemptive_step;

        // parallel evaluation of each data block
        for (size_t j=0; j<preemptive.size(); j++) {
            SACModel &model = preemptive[j];
            std::vector<int> current_model_inliers;
            SACScore score = getInliers(model_type, model.coefficients, cloud, normals, partial_set, threshold, 0, current_model_inliers, normal_distance_weight_);
            model.score.first += score.first;
            model.score.second += score.second;
            model.indices.insert(model.indices.end(), current_model_inliers.begin(), current_model_inliers.end());
        }

        // sort descending
        if (method_type == SAC_METHOD_RANSAC) {
            std::sort(preemptive.begin(), preemptive.end(), [](const SACModel &a,const SACModel &b) -> bool {
                return b.score.first < a.score.first;    // RANSAC
            });
        } else {
            std::sort(preemptive.begin(), preemptive.end(), [](const SACModel &a,const SACModel &b) -> bool {
                return b.score.second > a.score.second;  // MSAC
            });
        }

        // prune models
        int retain = (int)preemptive_func(preemptive_pos, (size_t)preemptive_count, preemptive_step);
        preemptive.erase(preemptive.begin() + retain + 1, preemptive.end());
    }

    SACModel &model = preemptive[0];
    if (model.score.first < min_inliers)
        return false;

    // "polish" the current best model, it might not have seen all data available.
    std::vector<double> new_coeff = optimizeModel(model_type, cloud, model.indices, model.coefficients);
    std::vector<int> new_inliers;
    SACScore new_result = getInliers(model_type, new_coeff, cloud, normals, indices, threshold, 0, new_inliers, normal_distance_weight_);
    if (new_inliers.size() >= model.indices.size()) {
        model.score = new_result;
        model.indices = new_inliers;
        model.coefficients = new_coeff;
    }

    model_instances.push_back(model);
    return true;
}


void SACModelFittingImpl::segment(std::vector<SACModel> &model_instances, OutputArray new_cloud) {
    sprt.release(); // make sure it's only used in fit_ransac() and only if use_sprt is on.

    size_t num_points = cloud.total();

    // optionally generate normals for the Cylinder model
    if ((model_type == CYLINDER_MODEL) && (normals.empty()) && (normal_distance_weight_ > 0)) {
        normals = generateNormals(cloud);
    }

    // a "mask" for already segmented points (0 == not_segmented)
    std::vector<int> point_labels(cloud.total(), 0);
    long num_segmented_points = 0;
    int label = 0; // to mark segmented points
    while (true) {
        label++;

        // filter unused point indices
        std::vector<int> indices;
        for (size_t i = 0; i < num_points; i++) {
            if (point_labels[i] == 0) indices.push_back(int(i));
        }
        if (indices.empty())
            break;

        // randomize indices, so we can assume, possible inliers are distributed uniformly.
        knuth_shuffle(indices);

        bool successful_fitting = (preemptive_count == 0)
            ? fit_ransac(indices, model_instances)
            : fit_preemptive(indices, model_instances);
        if (!successful_fitting)
            break;

        SACModel &latest = model_instances.back();
        num_segmented_points += (long)latest.indices.size();

        if (num_segmented_points == 0)
            break;

        // This loop is for implementation purposes only, and maps each point to a label.
        // All the points still labelled with 0 are non-segmented.
        // This way, complexity of the finding non-segmented is decreased to O(n).
        for(size_t i = 0; i < latest.indices.size(); i++) {
            int idx = latest.indices[i];
            point_labels[idx] = label;
            latest.points.push_back(cloud.at<Point3f>(idx));
        }
    }

    // optionally, copy remaining non-segmented points
    if (new_cloud.needed()) {
        Mat _cloud;
        for (size_t i=0; i<point_labels.size(); i++) {
            int p = point_labels[i];
            if (p == 0) _cloud.push_back(cloud.at<Point3f>(int(i)));
        }
        new_cloud.assign(_cloud);
    }
}


void SACModelFittingImpl::set_cloud(InputArray inp_cloud, bool with_normals) {
    Mat input_cloud = inp_cloud.getMat();

    if (! with_normals) {
        // normals are not required.
        normals.release();
        // the cloud should have three channels.
        CV_Assert(input_cloud.channels() == 3 || (input_cloud.channels() == 1 && (input_cloud.cols == 3 || input_cloud.rows == 3)));
        if (input_cloud.rows == 1 && input_cloud.channels() == 3) {
            cloud = input_cloud.clone();
            return;
        }

        if (input_cloud.channels() != 3 && input_cloud.rows == 3) {
            cloud = input_cloud.t();
        } else {
            cloud = input_cloud.clone();
        }

        cloud = cloud.reshape(3, input_cloud.rows);

    } else { // with_normals
        Mat _cld, _normals;
        CV_Assert(input_cloud.channels() == 1 && (input_cloud.cols == 6 || input_cloud.rows == 6));
        if (input_cloud.rows == 6) {
            input_cloud.rowRange(0, 3).copyTo(_cld);
            input_cloud.rowRange(3, 6).copyTo(_normals);
        } else {
            input_cloud.colRange(0, 3).copyTo(_cld);
            input_cloud.colRange(3, 6).copyTo(_normals);
        }
        cloud = Mat(_cld).reshape(3, 1);
        normals = Mat(_normals).reshape(3, 1);
    }
}

void SACModelFittingImpl::set_model_type(int type) {
    model_type = type;
}

void SACModelFittingImpl::set_method_type(int type) {
    method_type = type;
}

void SACModelFittingImpl::set_use_sprt(bool sprt_) {
    use_sprt = sprt_;
}

void SACModelFittingImpl::set_threshold (double threshold_value) {
    threshold = threshold_value;
}

void SACModelFittingImpl::set_preemptive_count(int preemptive) {
    preemptive_count = preemptive;
}

void SACModelFittingImpl::set_max_sphere_radius(double max_radius) {
    max_sphere_radius = max_radius;
}

void SACModelFittingImpl::set_max_napsac_radius(double max_radius) {
    max_napsac_radius = max_radius;
}

void SACModelFittingImpl::set_iterations (int iterations) {
    max_iters = iterations;
}

void SACModelFittingImpl::set_min_inliers (int inliers) {
    min_inliers = inliers;
}

void SACModelFittingImpl::set_normal_distance_weight(double weight) {
    if (weight > 1) {
        normal_distance_weight_ = 1;
    } else if (weight < 0) {
        normal_distance_weight_ = 0;
    } else {
        normal_distance_weight_ = weight;
    }
}


void cluster(InputArray _cloud, double distance, int min_inliers, std::vector<SACModel> &models, OutputArray _new_cloud) {
    Mat cloud = _cloud.getMat();
    std::vector<Point3f> pts(cloud.begin<Point3f>(), cloud.end<Point3f>());

    std::vector<int> cluster_indices;
    int n = cv::partition(pts, cluster_indices, [distance](const Point3f &a, const Point3f &b) -> bool{
        return norm(a-b) < distance;
    });
    if (n==0) return;

    std::vector<SACModel> mdl(n);
    for (size_t i=0; i<pts.size(); i++) {
        mdl[cluster_indices[i]].indices.push_back(int(i));
        mdl[cluster_indices[i]].points.push_back(pts[i]);
    }

    Mat_<Point3f> new_cloud;
    for (size_t i=0; i<mdl.size(); i++) {
        if (mdl[i].points.size() < (size_t)min_inliers) {
            if (_new_cloud.needed()) {
                for (auto p : mdl[i].points)
                    new_cloud.push_back(p);
            }
            continue; // rejected
        }
        box b = bbox(Mat(mdl[i].points));
        Point3f c = b.m + (b.M - b.m) / 2;

        mdl[i].coefficients = std::vector<double> {c.x, c.y, c.z, b.r};
        mdl[i].type = BLOB_MODEL;
        models.push_back(mdl[i]);
    }
    _new_cloud.assign(new_cloud);
}

} // ptcloud
} // cv
