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

    cv::setNumThreads(cv::getNumberOfCPUs());
    Mat res;
    weightedMedianFilter(src, src, res, 7);

    double totalMaxError = 1.0/32.0*src.total()*src.channels();

    EXPECT_LE(cvtest::norm(res, ref, NORM_L2), totalMaxError);
}

INSTANTIATE_TEST_CASE_P(TypicalSET, WeightedMedianFilterTest, Combine(Values(szODD, szQVGA),  Values(WMF_EXP, WMF_IV2, WMF_OFF)));

}
