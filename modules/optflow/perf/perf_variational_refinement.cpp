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
*  *Redistributions of source code must retain the above copyright notice,
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

using std::tr1::tuple;
using std::tr1::get;
using namespace perf;
using namespace testing;
using namespace cv;
using namespace cv::optflow;

typedef tuple<Size, int, int> VarRefParams;
typedef TestBaseWithParam<VarRefParams> DenseOpticalFlow_VariationalRefinement;

PERF_TEST_P(DenseOpticalFlow_VariationalRefinement, perf, Combine(Values(szQVGA, szVGA), Values(5, 10), Values(5, 10)))
{
    VarRefParams params = GetParam();
    Size sz = get<0>(params);
    int sorIter = get<1>(params);
    int fixedPointIter = get<2>(params);

    Mat frame1(sz, CV_8U);
    Mat frame2(sz, CV_8U);
    Mat flow(sz, CV_32FC2);

    randu(frame1, 0, 255);
    randu(frame2, 0, 255);
    flow.setTo(0.0f);

    cv::setNumThreads(cv::getNumberOfCPUs());
    TEST_CYCLE_N(10)
    {
        Ptr<VariationalRefinement> var = createVariationalFlowRefinement();
        var->setAlpha(20.0f);
        var->setGamma(10.0f);
        var->setDelta(5.0f);
        var->setSorIterations(sorIter);
        var->setFixedPointIterations(fixedPointIter);
        var->calc(frame1, frame2, flow);
    }

    SANITY_CHECK_NOTHING();
}
