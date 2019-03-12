// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef tuple<int, double, double, MatType, int> BTFParams;
typedef TestWithParam<BTFParams> BilateralTextureFilterTest;

TEST_P(BilateralTextureFilterTest, SplatSurfaceAccuracy)
{
  BTFParams params = GetParam();
  int fr            = get<0>(params);
  double sigmaAlpha = get<1>(params);
  double sigmaAvg   = get<2>(params);
  int depth         = get<3>(params);
  int srcCn         = get<4>(params);

  RNG rnd(0);

  Size sz(rnd.uniform(256,512), rnd.uniform(256,512));

  for (int i = 0; i < 5; i++)
  {
    Scalar surfaceValue;
    if(depth == CV_8U)
        rnd.fill(surfaceValue, RNG::UNIFORM, 0, 255);
    else
        rnd.fill(surfaceValue, RNG::UNIFORM, 0.0f, 1.0f);

    Mat src(sz, CV_MAKE_TYPE(depth, srcCn), surfaceValue);

    Mat res;
    bilateralTextureFilter(src, res, fr, 1, sigmaAlpha, sigmaAvg);

    double normL1 = cvtest::norm(src, res, NORM_L1)/src.total()/src.channels();
    EXPECT_LE(normL1, 1.0/64.0);
  }
}

TEST_P(BilateralTextureFilterTest, MultiThreadReproducibility)
{
    if (cv::getNumberOfCPUs() == 1)
      return;

    BTFParams params = GetParam();
    int fr            = get<0>(params);
    double sigmaAlpha = get<1>(params);
    double sigmaAvg   = get<2>(params);
    int depth         = get<3>(params);
    int srcCn         = get<4>(params);

    double MAX_DIF = 1.0;
    double MAX_MEAN_DIF = 1.0 / 64.0;
    int loopsCount = 2;
    RNG rnd(1);

    Size sz(rnd.uniform(256,512), rnd.uniform(256,512));

    Mat src(sz,CV_MAKE_TYPE(depth, srcCn));
    if(src.depth()==CV_8U)
        randu(src, 0, 255);
    else if(src.depth()==CV_16S)
        randu(src, -32767, 32767);
    else
        randu(src, 0.0f, 1.0f);

    int nThreads = cv::getNumThreads();
    if (nThreads == 1)
        throw SkipTestException("Single thread environment");
    for (int iter = 0; iter <= loopsCount; iter++)
    {
        cv::setNumThreads(nThreads);
        Mat resMultiThread;
        bilateralTextureFilter(src, resMultiThread, fr, 1, sigmaAlpha, sigmaAvg);

        cv::setNumThreads(1);
        Mat resSingleThread;
        bilateralTextureFilter(src, resSingleThread, fr, 1, sigmaAlpha, sigmaAvg);

        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_INF), MAX_DIF);
        EXPECT_LE(
          cv::norm(resSingleThread, resMultiThread, NORM_L1),
          MAX_MEAN_DIF * src.total() * src.channels());
    }
}

INSTANTIATE_TEST_CASE_P(
  TypicalSet1,
  BilateralTextureFilterTest,
  Combine(
    Values(2),
    Values(0.5),
    Values(0.5),
    Values(CV_8U, CV_32F),
    Values(1, 3)
    )
);


}} // namespace
