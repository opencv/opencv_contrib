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


typedef tuple<bool, Size, int, int, MatType> AMPerfTestParam;
typedef TestBaseWithParam<AMPerfTestParam> AdaptiveManifoldPerfTest;

PERF_TEST_P( AdaptiveManifoldPerfTest, perf,
    Combine(
    Values(true, false),            //adjust_outliers flag
    Values(sz1080p, sz720p),        //size
    Values(1, 3, 8),                //joint channels num
    Values(1, 3),                   //source channels num
    Values(CV_8U, CV_32F)           //source and joint depth
    )
)
{
    AMPerfTestParam params = GetParam();
    bool adjustOutliers = get<0>(params);
    Size sz             = get<1>(params);
    int jointCnNum      = get<2>(params);
    int srcCnNum        = get<3>(params);
    int depth           = get<4>(params);

    Mat joint(sz, CV_MAKE_TYPE(depth, jointCnNum));
    Mat src(sz, CV_MAKE_TYPE(depth, srcCnNum));
    Mat dst(sz, CV_MAKE_TYPE(depth, srcCnNum));

    cv::setNumThreads(cv::getNumberOfCPUs());

    declare.in(joint, src, WARMUP_RNG).out(dst).tbb_threads(cv::getNumberOfCPUs());

    double sigma_s = 16;
    double sigma_r = 0.5;
    TEST_CYCLE_N(3)
    {
        Mat res;
        amFilter(joint, src, res, sigma_s, sigma_r, adjustOutliers);

        //at 5th cycle sigma_s will be five times more and tree depth will be 5
        sigma_s *= 1.38;
        sigma_r /= 1.38;
    }

    SANITY_CHECK_NOTHING();
}

}
