// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/signal.hpp"

#include <vector>
#include <numeric>
#include <cmath>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>


namespace opencv_test { namespace {

using namespace cv;
using namespace cv::signal;

float MSE(const Mat1f &outSignal, const Mat1f &refSignal)
{
    float mse = 0.f;
    for (int i = 0; i < refSignal.cols; ++i)
    {
        mse += powf(outSignal.at<float>(0,i) - refSignal.at<float>(0,i), 2.f);
    }
    mse /= refSignal.cols;
    return mse;
}

// RRMSE = sqrt( MSE / SUM(sqr(refSignal(i))) ) * 100%
float RRMSE(const Mat1f &outSignal, const Mat1f &refSignal)
{
    float rrmse = 0.f;
    float div = 0.f;
    rrmse = MSE(outSignal, refSignal);
    for (int i = 0; i < refSignal.cols; ++i)
    {
        div += powf(refSignal.at<float>(0,i), 2.f);
    }
    rrmse /= div;
    rrmse = sqrt(rrmse) * 100;
    return rrmse;
}

TEST(ResampleTest, simple_resample_test_up)
{
    Mat1f sample_signal(Size(1000U,1U));
    Mat1f outSignal;
    std::iota(sample_signal.begin(), sample_signal.end(), 1.f);
    resampleSignal(sample_signal, outSignal, 16000U, 32000U);
    vector<float> ref(outSignal.cols, 0.f);
    for (uint32_t i = 0U; i < 2000U; ++i)
    {
        ref[i] = static_cast<float>(i) / 2.f;
    }
    EXPECT_NEAR(cvtest::norm(ref, NORM_L2) / cvtest::norm(outSignal, NORM_L2), 1.0f, 0.05f)
                << "\nL2_norm(refSignal) = " << cvtest::norm(ref, NORM_L2)
                << "\nL2_norm(outSignal) = " << cvtest::norm(outSignal, NORM_L2);
}

TEST(ResampleTest, resample_sin_signal_up_2)
{
    Mat1f sample_signal(Size(1000U,1U));
    Mat1f outSignal;
    for (uint32_t i = 0U; i < (uint32_t)sample_signal.cols; ++i)
    {
        sample_signal.at<float>(0, i) = sinf(float(i));
    }
    resampleSignal(sample_signal, outSignal, 16000U, 32000U);
    vector<float> ref(outSignal.cols, 0.f);
    for (uint32_t i = 0U; i < 2000U; ++i)
    {
        ref[i] = sin(static_cast<float>(i) / 2.f);
    }
    EXPECT_NEAR(cvtest::norm(ref, NORM_L2) / cvtest::norm(outSignal, NORM_L2), 1.0f, 0.05f)
                << "\nL2_norm(refSignal) = " << cvtest::norm(ref, NORM_L2)
                << "\nL2_norm(outSignal) = " << cvtest::norm(outSignal, NORM_L2);
}

TEST(ResampleTest, simple_resample_test_dn)
{
    Mat1f sample_signal(Size(1000U,1U));
    Mat1f outSignal;
    std::iota(sample_signal.begin(), sample_signal.end(), 1.f);
    resampleSignal(sample_signal, outSignal, 32000U, 16000U);
    vector<float> ref(outSignal.cols, 0.f);
    for (uint32_t i = 0U; i < 500U; ++i)
    {
        ref[i] = static_cast<float>(i) * 2.f;
    }
    EXPECT_NEAR(cvtest::norm(ref, NORM_L2) / cvtest::norm(outSignal, NORM_L2), 1.0f, 0.05f)
                << "\nL2_norm(refSignal) = " << cvtest::norm(ref, NORM_L2)
                << "\nL2_norm(outSignal) = " << cvtest::norm(outSignal, NORM_L2);
}

TEST(ResampleTest, resample_sin_signal_dn_2)
{
    Mat1f sample_signal(Size(1000U,1U));
    Mat1f outSignal;
    for (uint32_t i = 0U; i < (uint32_t)sample_signal.cols; ++i)
    {
        sample_signal.at<float>(0, i) = sinf(float(i));
    }
    resampleSignal(sample_signal, outSignal, 32000U, 16000U);
    std::vector<float> ref(outSignal.cols, 0.f);
    for (uint32_t i = 0U; i < 500U; ++i)
    {
        ref[i] = sin(static_cast<float>(i) * 2.f);
    }
    EXPECT_NEAR(cvtest::norm(ref, NORM_L2) / cvtest::norm(outSignal, NORM_L2), 1.0f, 0.05f)
                << "\nL2_norm(refSignal) = " << cvtest::norm(ref, NORM_L2)
                << "\nL2_norm(outSignal) = " << cvtest::norm(outSignal, NORM_L2);
}

// produce 1s of signal @ freq hz
void fillSignal(uint32_t freq, Mat1f &inSignal)
{
    static std::default_random_engine e((unsigned int)(time(NULL)));
    static std::uniform_real_distribution<> dis(0, 1); // range [0, 1)
    static auto a = dis(e), b = dis(e), c = dis(e);
    uint32_t idx = 0;
    std::generate(inSignal.begin(), inSignal.end(), [&]()
    {
        float ret = static_cast<float>(sin(idx/(float)freq + a) + 3 * sin(CV_PI / 4 * (idx/(float)freq + b))
                                    + 5 * sin(CV_PI/12 * idx/(float)freq + c) + 20*cos(idx/(float)freq*4000));
        idx++;
        return ret;
    });
}

class ResampleTestClass : public testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(ResampleTestClass, func_test) {
    auto params1 = GetParam();
    uint32_t inFreq = std::get<0>(params1);
    uint32_t outFreq = std::get<1>(params1);
    // 1 second @ inFreq hz
    Mat1f inSignal(Size(inFreq, 1U));
    Mat1f outSignal;
    // generating testing function as a sum of different sinusoids
    fillSignal(inFreq, inSignal);
    resampleSignal(inSignal, outSignal, inFreq, outFreq);
    // reference signal
    // 1 second @ outFreq hz
    Mat1f refSignal(Size(outFreq, 1U));
    fillSignal(outFreq, refSignal);
    // calculating maxDiff
    float maxDiff = 0.f;
    // exclude 2 elements and last 2 elements from testing
    for (uint32_t i = 2; i < (uint32_t)refSignal.cols - 2; ++i)
    {
        if(maxDiff < abs(refSignal.at<float>(0,i) - outSignal.at<float>(0,i)))
        {
            maxDiff = abs(refSignal.at<float>(0,i) - outSignal.at<float>(0,i));
        }
    }
    auto max = std::max_element(outSignal.begin(), outSignal.end());
    float maxDiffRel = maxDiff / (*max);
    EXPECT_LE(maxDiffRel, 0.35f);
    // calculating relative error of L2 norms
    EXPECT_NEAR(abs(cvtest::norm(outSignal, NORM_L2) - cvtest::norm(refSignal, NORM_L2)) /
                    cvtest::norm(refSignal, NORM_L2), 0.0f, 0.05f);
    // calculating relative mean squared error
    float rrmse = RRMSE(outSignal, refSignal);
    // 1% error
    EXPECT_LE(rrmse, 1.f);
}

INSTANTIATE_TEST_CASE_P(RefSignalTestingCase,
                         ResampleTestClass,
                         ::testing::Combine(testing::Values(16000, 32000, 44100, 48000),
                                            testing::Values(16000, 32000, 44100, 48000)));

}} // namespace
