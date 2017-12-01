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
#include "opencv2/imgproc.hpp"

using std::tr1::tuple;
using std::tr1::get;
using namespace std;
using namespace cv;
using namespace perf;
using namespace testing;

typedef tuple<Size, MatType> learningBasedWBParams;
typedef TestBaseWithParam<learningBasedWBParams> learningBasedWBPerfTest;

PERF_TEST_P(learningBasedWBPerfTest, perf, Combine(SZ_ALL_HD, Values(CV_8UC3, CV_16UC3)))
{
    Size size = get<0>(GetParam());
    MatType t = get<1>(GetParam());
    Mat src(size, t);
    Mat dst(size, t);

    int range_max_val = 255, hist_bin_num = 64;
    if (t == CV_16UC3)
    {
        range_max_val = 65535;
        hist_bin_num = 256;
    }

    Mat src_dscl(Size(size.width / 16, size.height / 16), t);
    RNG rng(1234);
    rng.fill(src_dscl, RNG::UNIFORM, 0, range_max_val);
    resize(src_dscl, src, src.size(), 0, 0, INTER_LINEAR_EXACT);
    Ptr<xphoto::LearningBasedWB> wb = xphoto::createLearningBasedWB();
    wb->setRangeMaxVal(range_max_val);
    wb->setSaturationThreshold(0.98f);
    wb->setHistBinNum(hist_bin_num);

    TEST_CYCLE() wb->balanceWhite(src, dst);

    SANITY_CHECK_NOTHING();
}
