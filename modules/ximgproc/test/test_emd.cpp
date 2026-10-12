// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

using namespace cv;
using namespace std;

namespace opencv_test { namespace {

//==============================================================================
// Utility

template <typename T>
inline T sqr(T val)
{
    return val * val;
}

inline static float calcEMD(Mat w1, Mat w2, Mat& flow, int dist, int dims)
{
    float mass1 = 0.f, mass2 = 0.f, work = 0.f;
    for (int i = 0; i < flow.rows; ++i)
    {
        mass1 += w1.at<float>(i, 0);
        for (int j = 0; j < flow.cols; ++j)
        {
            if (i == 0)
                mass2 += w2.at<float>(j, 0);
            float dist_ = 0.f;
            switch (dist)
            {
                case DIST_L1:
                {
                    for (int k = 1; k <= dims; ++k)
                    {
                        dist_ += abs(w1.at<float>(i, k) - w2.at<float>(j, k));
                    }
                    break;
                }
                case DIST_L2:
                {
                    for (int k = 1; k <= dims; ++k)
                    {
                        dist_ += sqr(w1.at<float>(i, k) - w2.at<float>(j, k));
                    }
                    dist_ = sqrt(dist_);
                    break;
                }
                case DIST_C:
                {
                    for (int k = 1; k <= dims; ++k)
                    {
                        const float val = abs(w1.at<float>(i, k) - w2.at<float>(j, k));
                        if (val > dist_)
                            dist_ = val;
                    }
                    break;
                }
            }
            const float weight = flow.at<float>(i, j);
            work += dist_ * weight;
        }
    }
    return work / max(mass1, mass2);
}

// Independent reference for the lower bound: the distance between the weighted
// mass centers of the two signatures, normalized by the larger total weight.
// Mirrors EMDSolver::checkLowerBound so it can pin that code path.
inline static float massCenterLowerBound(Mat s1, Mat s2, int dist, int dims)
{
    vector<float> xs(dims, 0.f), xd(dims, 0.f);
    float sum1 = 0.f, sum2 = 0.f;
    for (int j = 0; j < s1.rows; ++j)
    {
        const float w = s1.at<float>(j, 0);
        sum1 += w;
        for (int k = 0; k < dims; ++k)
            xs[k] += s1.at<float>(j, k + 1) * w;
    }
    for (int j = 0; j < s2.rows; ++j)
    {
        const float w = s2.at<float>(j, 0);
        sum2 += w;
        for (int k = 0; k < dims; ++k)
            xd[k] += s2.at<float>(j, k + 1) * w;
    }
    float d = 0.f;
    switch (dist)
    {
        case DIST_L1:
            for (int k = 0; k < dims; ++k)
                d += abs(xs[k] - xd[k]);
            break;
        case DIST_L2:
            for (int k = 0; k < dims; ++k)
                d += sqr(xs[k] - xd[k]);
            d = sqrt(d);
            break;
        case DIST_C:
            for (int k = 0; k < dims; ++k)
                d = max(d, abs(xs[k] - xd[k]));
            break;
    }
    return d / max(sum1, sum2);
}

//==============================================================================

TEST(Ximgproc_EMD, regression)
{
    // input data
    const float M = 10000;
    Matx<float, 4, 1> w1 {50, 60, 50, 50};
    Matx<float, 5, 1> w2 {30, 20, 70, 30, 60};
    Matx<float, 4, 5> cost {16, 16, 13, 22, 17, 14, 14, 13, 19, 15,
                            19, 19, 20, 23, M,  M,  0,  M,  0,  0};

    // expected results
    const double emd0 = 2460. / 210;
    Matx<float, 4, 5> flow0 {0, 0, 50, 0, 0, 0, 0, 20, 0, 40, 30, 20, 0, 0, 0, 0, 0, 0, 30, 20};

    // basic call with cost
    {
        float emd = 0.f;
        ASSERT_NO_THROW(emd = cv::ximgproc::EMD(w1, w2, DIST_USER, cost));
        EXPECT_NEAR(emd, emd0, 1e-6 * emd0);
    }

    // basic call with cost and flow output
    {
        Mat flow;
        float emd = 0.f;
        ASSERT_NO_THROW(emd = cv::ximgproc::EMD(w1, w2, DIST_USER, cost, nullptr, flow));
        EXPECT_NEAR(emd, emd0, 1e-6 * emd0);
        EXPECT_LE(cvtest::norm(Mat(flow0), flow, NORM_INF), 1e-6);
    }
    // no cost and DIST_USER - error
    {
        Mat flow;
        EXPECT_THROW(cv::ximgproc::EMD(w1, w2, DIST_USER, noArray(), nullptr, flow), cv::Exception);
        EXPECT_THROW(cv::ximgproc::EMD(w1, w2, DIST_USER), cv::Exception);
    }
}

TEST(Ximgproc_EMD, distance_types)
{
    // 1D (sum = 210)
    Matx<float, 4, 2> w1 {50, 1, 60, 2, 50, 3, 50, 4};
    Matx<float, 5, 2> w2 {30, 1, 20, 2, 70, 3, 30, 4, 60, 5};

    // 2D (sum = 210)
    Matx<float, 4, 3> w3 {50, 0, 0, 60, 0, 1, 50, 1, 0, 50, 1, 1};
    Matx<float, 5, 3> w4 {20, 0, 1, 70, 1, 0, 30, 1, 1, 60, 2, 2, 30, 3, 3};

    // basic call with all distance types
    {
        const vector<DistanceTypes> good_types {DIST_L1, DIST_L2, DIST_C};
        for (const auto& dt : good_types)
        {
            SCOPED_TRACE(cv::format("dt=%d", dt));
            float emd = 0.f;
            Mat flow;
            // 1D
            {
                ASSERT_NO_THROW(emd = cv::ximgproc::EMD(w1, w2, dt, noArray(), nullptr, flow));
                const float emd0 = calcEMD(Mat(w1), Mat(w2), flow, dt, 1);
                EXPECT_NEAR(emd0, emd, 1e-6);
            }
            // 2D
            {
                ASSERT_NO_THROW(emd = cv::ximgproc::EMD(w3, w4, dt, noArray(), nullptr, flow));
                const float emd0 = calcEMD(Mat(w3), Mat(w4), flow, dt, 2);
                EXPECT_NEAR(emd0, emd, 1e-6);
            }
        }
    }
}

TEST(Ximgproc_EMD, lower_bound)
{
    // Equal total weights (210 each) so the lower-bound path is active
    // (checkLowerBound is only reached when the signature sums match).
    Matx<float, 4, 2> w1 {50, 1, 60, 2, 50, 3, 50, 4};
    Matx<float, 5, 2> w2 {30, 1, 20, 2, 70, 3, 30, 4, 60, 5};
    const int dims = 1;

    const vector<DistanceTypes> good_types {DIST_L1, DIST_L2, DIST_C};
    for (const auto& dt : good_types)
    {
        SCOPED_TRACE(cv::format("dt=%d", dt));
        const float lb0 = massCenterLowerBound(Mat(w1), Mat(w2), dt, dims);
        const float tol = 1e-4f * max(lb0, 1.f);

        // Small initial bound: the mass-center distance already meets it, so the
        // signatures are "far enough", the solver is skipped, and the lower
        // bound is returned. bound is updated to the mass-center distance.
        {
            Mat flow;
            float bound = 0.f, emd = 0.f;
            ASSERT_NO_THROW(emd = cv::ximgproc::EMD(w1, w2, dt, noArray(), &bound, flow));
            EXPECT_NEAR(bound, lb0, tol);
            EXPECT_NEAR(emd, lb0, tol);
        }

        // Large initial bound: not far enough, so the full EMD is computed.
        // bound is still updated to the mass-center distance on return.
        {
            Mat flow;
            float bound = 1e9f, emd = 0.f;
            ASSERT_NO_THROW(emd = cv::ximgproc::EMD(w1, w2, dt, noArray(), &bound, flow));
            const float emd0 = calcEMD(Mat(w1), Mat(w2), flow, dt, dims);
            EXPECT_NEAR(emd, emd0, 1e-4f * max(emd0, 1.f));
            EXPECT_NEAR(bound, lb0, tol);
            EXPECT_LE(lb0, emd + tol);  // a lower bound never exceeds the EMD
        }
    }
}

typedef testing::TestWithParam<int> Ximgproc_EMD_dist;

TEST_P(Ximgproc_EMD_dist, random_flow_verify)
{
    const int dist = GetParam();
    for (size_t iter = 0; iter < 100; ++iter)
    {
        SCOPED_TRACE(cv::format("iter=%zu", iter));
        RNG& rng = TS::ptr()->get_rng();
        const int dims = rng.uniform(1, 10);
        Mat w1(rng.uniform(1, 10), dims + 1, CV_32FC1);
        Mat w2(rng.uniform(1, 10), dims + 1, CV_32FC1);

        // weights > 0
        {
            Mat w1_weights = w1.col(0);
            Mat w2_weights = w2.col(0);
            cvtest::randUni(rng, w1_weights, 0, 100);
            cvtest::randUni(rng, w2_weights, 0, 100);
        }

        // coord
        {
            Mat w1_coord = w1.colRange(1, dims + 1);
            Mat w2_coord = w2.colRange(1, dims + 1);
            cvtest::randUni(rng, w1_coord, -10, +10);
            cvtest::randUni(rng, w2_coord, -10, +10);
        }

        float emd1 = 0.f, emd2 = 0.f;
        const float eps = 1e-5f;
        Mat flow;
        {
            ASSERT_NO_THROW(emd1 = cv::ximgproc::EMD(w1, w2, dist, noArray(), nullptr, flow));
            const float emd0 = calcEMD(w1, w2, flow, dist, dims);
            EXPECT_NEAR(emd0, emd1, eps);
        }
        {
            ASSERT_NO_THROW(emd2 = cv::ximgproc::EMD(w2, w1, dist, noArray(), nullptr, flow));
            const float emd0 = calcEMD(w2, w1, flow, dist, dims);
            EXPECT_NEAR(emd0, emd2, eps);
        }
        EXPECT_NEAR(emd1, emd2, eps);
    }
}

INSTANTIATE_TEST_CASE_P(, Ximgproc_EMD_dist, testing::Values(DIST_L1, DIST_L2, DIST_C));


TEST(Ximgproc_EMD, invalid)
{
    Matx<float, 4, 2> w1 {50, 1, 60, 2, 50, 3, 50, 4};
    Matx<float, 5, 2> w2 {30, 1, 20, 2, 70, 3, 30, 4, 60, 5};

    // empty signature
    {
        Mat empty;
        EXPECT_THROW(cv::ximgproc::EMD(empty, w2, DIST_USER), cv::Exception);
        EXPECT_THROW(cv::ximgproc::EMD(w1, empty, DIST_USER), cv::Exception);
    }

    // zero total weight, negative weight
    {
        Matx<float, 3, 1> wz {0, 0, 0};
        Matx<float, 3, 2> wz1 {0, 1, 0, 2, 0, 3};
        Matx<float, 3, 1> wn {0, 3, -2};
        Matx<float, 3, 2> wn1 {0, 1, 3, 2, -2, 3};
        EXPECT_THROW(cv::ximgproc::EMD(wz, w2, DIST_USER), cv::Exception);
        EXPECT_THROW(cv::ximgproc::EMD(wz1, w2, DIST_USER), cv::Exception);
        EXPECT_THROW(cv::ximgproc::EMD(wn, w2, DIST_USER), cv::Exception);
        EXPECT_THROW(cv::ximgproc::EMD(wn1, w2, DIST_USER), cv::Exception);
    }

    // user distance type, but no cost matrix provided or is wrong
    {
        Mat cost(3, 3, CV_32FC1, Scalar::all(0)), cost8u(4, 5, CV_8UC1, Scalar::all(0)), empty;
        EXPECT_THROW(cv::ximgproc::EMD(w1, w2, DIST_USER, noArray()), cv::Exception);
        EXPECT_THROW(cv::ximgproc::EMD(w1, w2, DIST_USER, empty), cv::Exception);
        EXPECT_THROW(cv::ximgproc::EMD(w1, w2, DIST_USER, cost8u), cv::Exception);
        EXPECT_THROW(cv::ximgproc::EMD(w1, w2, DIST_USER, cost), cv::Exception);
    }

    // lower_bound is set together with cost
    {
        Mat cost(4, 5, CV_32FC1, Scalar::all(0));
        float bound = 0.f;
        EXPECT_THROW(cv::ximgproc::EMD(w1, w2, DIST_USER, cost, &bound), cv::Exception);
    }

    // zero dimensions with non-user distance type
    const vector<DistanceTypes> good_types {DIST_L1, DIST_L2, DIST_C};
    for (const auto& dt : good_types)
    {
        SCOPED_TRACE(cv::format("dt=%d", dt));
        Matx<float, 4, 1> w01 {20, 30, 40, 50};
        Matx<float, 5, 1> w02 {20, 30, 40, 50, 10};
        EXPECT_THROW(cv::ximgproc::EMD(w01, w02, dt), cv::Exception);
    }

    // wrong distance type
    const vector<DistanceTypes> bad_types {DIST_L12, DIST_FAIR, DIST_WELSCH, DIST_HUBER};
    for (const auto& dt : bad_types)
    {
        SCOPED_TRACE(cv::format("dt=%d", dt));
        EXPECT_THROW(cv::ximgproc::EMD(w1, w2, dt), cv::Exception);
    }
}

}}  // namespace opencv_test
