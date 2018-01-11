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

CV_ENUM(SrcTypes, CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3);
typedef tuple<Size, SrcTypes> L0SmoothParams;
typedef TestWithParam<L0SmoothParams> L0SmoothTest;

TEST(L0SmoothTest, SplatSurfaceAccuracy)
{
    RNG rnd(0);

    for (int i = 0; i < 3; i++)
    {
        Size sz(rnd.uniform(512, 1024), rnd.uniform(512, 1024));

        Scalar surfaceValue;
        int srcCn = 3;
        rnd.fill(surfaceValue, RNG::UNIFORM, 0, 255);
        Mat src(sz, CV_MAKE_TYPE(CV_8U, srcCn), surfaceValue);

        double lambda = rnd.uniform(0.01, 0.05);
        double kappa  = rnd.uniform(1.5, 5.0);

        Mat res;
        l0Smooth(src, res, lambda, kappa);

        // When filtering a constant image we should get the same image:
        double normL1 = cvtest::norm(src, res, NORM_L1)/src.total()/src.channels();
        EXPECT_LE(normL1, 1.0/64);
    }
}

TEST_P(L0SmoothTest, MultiThreadReproducibility)
{
    if (cv::getNumberOfCPUs() == 1)
        return;

    double MAX_DIF = 10.0;
    double MAX_MEAN_DIF = 1.0 / 8.0;
    int loopsCount = 2;
    RNG rng(0);

    L0SmoothParams params = GetParam();
    Size size     = get<0>(params);
    int srcType   = get<1>(params);

    Mat src(size,srcType);
    if(src.depth()==CV_8U)
        randu(src, 0, 255);
    else if(src.depth()==CV_16U)
        randu(src, 0, 65535);
    else
        randu(src, -100000.0f, 100000.0f);


    for (int iter = 0; iter <= loopsCount; iter++)
    {
        double lambda = rng.uniform(0.01, 0.05);
        double kappa  = rng.uniform(1.5, 5.0);

        cv::setNumThreads(cv::getNumberOfCPUs());
        Mat resMultiThread;
        l0Smooth(src, resMultiThread, lambda, kappa);

        cv::setNumThreads(1);
        Mat resSingleThread;
        l0Smooth(src, resSingleThread, lambda, kappa);

        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_INF), MAX_DIF);
        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_L1), MAX_MEAN_DIF*src.total()*src.channels());
    }
}
INSTANTIATE_TEST_CASE_P(FullSet, L0SmoothTest,Combine(Values(szODD, szQVGA), SrcTypes::all()));

}
