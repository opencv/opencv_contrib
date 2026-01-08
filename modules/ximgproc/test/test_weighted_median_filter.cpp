// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

static string getDataDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

typedef tuple<Size, WMFWeightType> WMFParams;
typedef TestWithParam<WMFParams> WeightedMedianFilterTest;

TEST_P(WeightedMedianFilterTest, SplatSurfaceAccuracy)
{

    WMFParams params = GetParam();
    Size size = get<0>(params);
    WMFWeightType weightType = get<1>(params);

    RNG rnd(0);

    int guideCn = rnd.uniform(1, 2);
    if(guideCn==2) guideCn++; //1 or 3 channels
    Mat guide(size, CV_MAKE_TYPE(CV_8U, guideCn));
    randu(guide, 0, 255);

    Scalar surfaceValue;
    int srcCn = rnd.uniform(1, 4);
    rnd.fill(surfaceValue, RNG::UNIFORM, 0, 255);
    Mat src(size, CV_MAKE_TYPE(CV_8U, srcCn), surfaceValue);

    int r = int(rnd.uniform(3, 11));
    double sigma  = rnd.uniform(9.0, 100.0);

    Mat res;
    weightedMedianFilter(guide, src, res, r, sigma, weightType);

    double normL1 = cvtest::norm(src, res, NORM_L1)/src.total()/src.channels();
    EXPECT_LE(normL1, 1.0/64);
}

TEST(WeightedMedianFilterTest, ReferenceAccuracy)
{
    string dir = getDataDir() + "cv/edgefilter";

    Mat src = imread(dir + "/kodim23.png");
    Mat ref = imread(dir + "/fgs/kodim23_lambda=1000_sigma=10.png");

    ASSERT_FALSE(src.empty());
    ASSERT_FALSE(ref.empty());

    Mat res;
    weightedMedianFilter(src, src, res, 7);

    double totalMaxError = 1.0/32.0*src.total()*src.channels();

    EXPECT_LE(cvtest::norm(res, ref, NORM_L2), totalMaxError);
}

TEST(WeightedMedianFilterTest, mask_zeros_no_crash)
{
    Mat img = imread(getDataDir() + "cv/ximgproc/sources/01.png");
    Mat mask = Mat::zeros(img.size(), CV_8U);
    Mat filtered;
    weightedMedianFilter(img, img, filtered, 3, 20, WMF_EXP, mask);

    EXPECT_EQ(cv::norm(img, filtered, NORM_INF), 0.0);
}

INSTANTIATE_TEST_CASE_P(TypicalSET, WeightedMedianFilterTest, Combine(Values(szODD, szQVGA),  Values(WMF_EXP, WMF_IV2, WMF_OFF)));


}} // namespace
