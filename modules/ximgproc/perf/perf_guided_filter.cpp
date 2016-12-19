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

CV_ENUM(GuideTypes, CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3);
CV_ENUM(SrcTypes, CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3);
typedef tuple<GuideTypes, SrcTypes, Size> GFParams;

typedef TestBaseWithParam<GFParams> GuidedFilterPerfTest;

PERF_TEST_P( GuidedFilterPerfTest, perf, Combine(GuideTypes::all(), SrcTypes::all(), Values(sz1080p, sz2K)) )
{
    RNG rng(0);

    GFParams params = GetParam();
    int guideType   = get<0>(params);
    int srcType     = get<1>(params);
    Size sz         = get<2>(params);

    Mat guide(sz, guideType);
    Mat src(sz, srcType);
    Mat dst(sz, srcType);

    declare.in(guide, src, WARMUP_RNG).out(dst).tbb_threads(cv::getNumberOfCPUs());

    cv::setNumThreads(cv::getNumberOfCPUs());
    TEST_CYCLE_N(3)
    {
        int radius = rng.uniform(5, 30);
        double eps = rng.uniform(0.1, 1e5);
        guidedFilter(guide, src, dst, radius, eps);
    }

    SANITY_CHECK_NOTHING();
}

}
