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

static string getDataDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

CV_ENUM(SrcTypes, CV_8UC1, CV_8UC3, CV_8UC4, CV_16SC1, CV_16SC3, CV_32FC1);
CV_ENUM(GuideTypes, CV_8UC1, CV_8UC3)
typedef tuple<Size, SrcTypes, GuideTypes> FGSParams;
typedef TestWithParam<FGSParams> FastGlobalSmootherTest;

TEST(FastGlobalSmootherTest, SplatSurfaceAccuracy)
{
    RNG rnd(0);

    for (int i = 0; i < 5; i++)
    {
        Size sz(rnd.uniform(512, 1024), rnd.uniform(512, 1024));

        int guideCn = rnd.uniform(1, 2);
        if(guideCn==2) guideCn++; //1 or 3 channels
        Mat guide(sz, CV_MAKE_TYPE(CV_8U, guideCn));
        randu(guide, 0, 255);

        Scalar surfaceValue;
        int srcCn = rnd.uniform(1, 4);
        rnd.fill(surfaceValue, RNG::UNIFORM, 0, 255);
        Mat src(sz, CV_MAKE_TYPE(CV_16S, srcCn), surfaceValue);

        double lambda = rnd.uniform(100, 10000);
        double sigma  = rnd.uniform(1.0, 100.0);

        Mat res;
        fastGlobalSmootherFilter(guide, src, res, lambda, sigma);

        // When filtering a constant image we should get the same image:
        double normL1 = cvtest::norm(src, res, NORM_L1)/src.total()/src.channels();
        EXPECT_LE(normL1, 1.0/64);
    }
}

TEST(FastGlobalSmootherTest, ReferenceAccuracy)
{
    string dir = getDataDir() + "cv/edgefilter";

    Mat src = imread(dir + "/kodim23.png");
    Mat ref = imread(dir + "/fgs/kodim23_lambda=1000_sigma=10.png");

    ASSERT_FALSE(src.empty());
    ASSERT_FALSE(ref.empty());

    cv::setNumThreads(cv::getNumberOfCPUs());
    Mat res;
    fastGlobalSmootherFilter(src,src,res,1000.0,10.0);

    double totalMaxError = 1.0/64.0*src.total()*src.channels();

    EXPECT_LE(cvtest::norm(res, ref, NORM_L2), totalMaxError);
    EXPECT_LE(cvtest::norm(res, ref, NORM_INF), 1);
}

TEST_P(FastGlobalSmootherTest, MultiThreadReproducibility)
{
    if (cv::getNumberOfCPUs() == 1)
        return;

    double MAX_DIF = 1.0;
    double MAX_MEAN_DIF = 1.0 / 64.0;
    int loopsCount = 2;
    RNG rng(0);

    FGSParams params = GetParam();
    Size size     = get<0>(params);
    int srcType   = get<1>(params);
    int guideType = get<2>(params);

    Mat guide(size, guideType);
    randu(guide, 0, 255);
    Mat src(size,srcType);
    if(src.depth()==CV_8U)
        randu(src, 0, 255);
    else if(src.depth()==CV_16S)
        randu(src, -32767, 32767);
    else
        randu(src, -100000.0f, 100000.0f);

    for (int iter = 0; iter <= loopsCount; iter++)
    {
        double lambda = rng.uniform(100.0, 10000.0);
        double sigma  = rng.uniform(1.0, 100.0);

        cv::setNumThreads(cv::getNumberOfCPUs());
        Mat resMultiThread;
        fastGlobalSmootherFilter(guide, src, resMultiThread, lambda, sigma);

        cv::setNumThreads(1);
        Mat resSingleThread;
        fastGlobalSmootherFilter(guide, src, resSingleThread, lambda, sigma);

        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_INF), MAX_DIF);
        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_L1), MAX_MEAN_DIF*src.total()*src.channels());
    }
}
INSTANTIATE_TEST_CASE_P(FullSet, FastGlobalSmootherTest,Combine(Values(szODD, szQVGA), SrcTypes::all(), GuideTypes::all()));

}
