// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

static std::string getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

static void checkSimilarity(InputArray src, InputArray ref)
{
    // Doesn't work with bilateral filter: EXPECT_LE(cvtest::norm(src, ref, NORM_INF), 1.0);
    EXPECT_LE(cvtest::norm(src, ref, NORM_L2 | NORM_RELATIVE), 1e-3);
}

static Mat convertTypeAndSize(Mat src, int dstType, Size dstSize)
{
    Mat dst;
    int srcCnNum = src.channels();
    int dstCnNum = CV_MAT_CN(dstType);

    if (srcCnNum == dstCnNum)
    {
        src.copyTo(dst);
    }
    else if (srcCnNum == 3 && dstCnNum == 1)
    {
        cvtColor(src, dst, COLOR_BGR2GRAY);
    }
    else if (srcCnNum == 1 && dstCnNum == 3)
    {
        cvtColor(src, dst, COLOR_GRAY2BGR);
    }
    else
    {
        CV_Error(Error::BadNumChannels, "Bad num channels in src");
    }

    dst.convertTo(dst, dstType);
    resize(dst, dst, dstSize, 0, 0, INTER_LINEAR_EXACT);

    return dst;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

typedef tuple<double, MatType, int> RGFParams;
typedef TestWithParam<RGFParams> RollingGuidanceFilterTest;

TEST_P(RollingGuidanceFilterTest, SplatSurfaceAccuracy)
{
    RGFParams params = GetParam();
    double sigmaS   = get<0>(params);
    int depth       = get<1>(params);
    int srcCn       = get<2>(params);

    RNG rnd(0);

    Size sz(rnd.uniform(512,1024), rnd.uniform(512,1024));

    for (int i = 0; i < 5; i++)
    {
        Scalar surfaceValue;
        rnd.fill(surfaceValue, RNG::UNIFORM, 0, 255);
        Mat src(sz, CV_MAKE_TYPE(depth, srcCn), surfaceValue);

        double sigmaC = rnd.uniform(1.0, 255.0);
	int iterNum = int(rnd.uniform(1.0, 5.0));

        Mat res;
        rollingGuidanceFilter(src, res, -1, sigmaC, sigmaS, iterNum);

        double normL1 = cvtest::norm(src, res, NORM_L1)/src.total()/src.channels();
        EXPECT_LE(normL1, 1.0/64);
    }
}

TEST_P(RollingGuidanceFilterTest, MultiThreadReproducibility)
{
    if (cv::getNumberOfCPUs() == 1)
        return;

    RGFParams params = GetParam();
    double sigmaS   = get<0>(params);
    int depth       = get<1>(params);
    int srcCn       = get<2>(params);

    double MAX_DIF = 1.0;
    double MAX_MEAN_DIF = 1.0 / 64.0;
    int loopsCount = 2;
    RNG rnd(1);

    Size sz(rnd.uniform(512,1024), rnd.uniform(512,1024));

    Mat src(sz,CV_MAKE_TYPE(depth, srcCn));
    if(src.depth()==CV_8U)
        randu(src, 0, 255);
    else if(src.depth()==CV_16S)
        randu(src, -32767, 32767);
    else
        randu(src, -100000.0f, 100000.0f);

    int nThreads = cv::getNumThreads();
    if (nThreads == 1)
        throw SkipTestException("Single thread environment");
    for (int iter = 0; iter <= loopsCount; iter++)
    {
        int iterNum = int(rnd.uniform(1.0, 5.0));
        double sigmaC = rnd.uniform(1.0, 255.0);

        cv::setNumThreads(nThreads);
        Mat resMultiThread;
        rollingGuidanceFilter(src, resMultiThread, -1, sigmaC, sigmaS, iterNum);

        cv::setNumThreads(1);
        Mat resSingleThread;
        rollingGuidanceFilter(src, resSingleThread, -1, sigmaC, sigmaS, iterNum);

        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_INF), MAX_DIF);
        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_L1), MAX_MEAN_DIF*src.total()*src.channels());
    }
}

INSTANTIATE_TEST_CASE_P(TypicalSet1, RollingGuidanceFilterTest,
    Combine(
    Values(2.0, 5.0),
    Values(CV_8U, CV_32F),
    Values(1, 3)
    )
);

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

typedef tuple<double, string, int> RGFBFParam;
typedef TestWithParam<RGFBFParam> RollingGuidanceFilterTest_BilateralRef;

TEST_P(RollingGuidanceFilterTest_BilateralRef, Accuracy)
{
    RGFBFParam params = GetParam();
    double sigmaS       = get<0>(params);
    string srcPath      = get<1>(params);
    int srcType         = get<2>(params);

    Mat src = imread(getOpenCVExtraDir() + srcPath);
    ASSERT_TRUE(!src.empty());
    src = convertTypeAndSize(src, srcType, src.size());

    RNG rnd(0);
    double sigmaC = rnd.uniform(0.0, 255.0);

    Mat resRef;
    bilateralFilter(src, resRef, 0, sigmaC, sigmaS);

    Mat res, joint = src.clone();
    rollingGuidanceFilter(src, res, 0, sigmaC, sigmaS, 1);

    checkSimilarity(res, resRef);
}

INSTANTIATE_TEST_CASE_P(TypicalSet2, RollingGuidanceFilterTest_BilateralRef,
    Combine(
    Values(4.0, 6.0, 8.0),
    Values("/cv/shared/pic2.png", "/cv/shared/lena.png", "cv/shared/box_in_scene.png"),
    Values(CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3)
    )
);


}} // namespace
