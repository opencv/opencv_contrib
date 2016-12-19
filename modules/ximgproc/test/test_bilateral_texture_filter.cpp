/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "test_precomp.hpp"

namespace cvtest
{

using namespace std;
using namespace std::tr1;
using namespace testing;
using namespace perf;
using namespace cv;
using namespace cv::ximgproc;

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

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

    for (int iter = 0; iter <= loopsCount; iter++)
    {
        cv::setNumThreads(cv::getNumberOfCPUs());
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
}
