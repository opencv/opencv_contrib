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

#include "perf_precomp.hpp"

namespace cvtest
{

using std::tr1::tuple;
using std::tr1::get;
using namespace perf;
using namespace testing;
using namespace cv;
using namespace cv::ximgproc;

typedef tuple<double, Size, MatType, int> RGFTestParam;
typedef TestBaseWithParam<RGFTestParam> RollingGuidanceFilterTest;

PERF_TEST_P(RollingGuidanceFilterTest, perf,
    Combine(
    Values(2.0, 4.0, 6.0, 10.0),
    SZ_TYPICAL,
    Values(CV_8U, CV_32F),
    Values(1, 3))
)
{
    RGFTestParam params = GetParam();
    double sigmaS   = get<0>(params);
    Size sz         = get<1>(params);
    int depth       = get<2>(params);
    int srcCn       = get<3>(params);

    Mat src(sz, CV_MAKE_TYPE(depth, srcCn));
    Mat dst(sz, src.type());

    cv::setNumThreads(cv::getNumberOfCPUs());
    declare.in(src, WARMUP_RNG).out(dst).tbb_threads(cv::getNumberOfCPUs());

    RNG rnd(cvRound(10*sigmaS) + sz.height + depth + srcCn);
    double sigmaC = rnd.uniform(1.0, 255.0);
    int iterNum = int(rnd.uniform(1.0, 5.0));

    TEST_CYCLE_N(1)
    {
        rollingGuidanceFilter(src, dst, -1, sigmaC, sigmaS, iterNum);
    }

    SANITY_CHECK_NOTHING();
}
}
