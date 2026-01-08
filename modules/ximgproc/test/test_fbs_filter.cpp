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

#ifdef HAVE_EIGEN

namespace opencv_test { namespace {

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

static string getDataDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

CV_ENUM(SrcTypes, CV_8UC1, CV_8UC3, CV_8UC4, CV_16SC1, CV_16SC3, CV_32FC1);
CV_ENUM(GuideTypes, CV_8UC1, CV_8UC3)
typedef tuple<Size, SrcTypes, GuideTypes> FBSParams;
typedef TestWithParam<FBSParams> FastBilateralSolverTest;

TEST(FastBilateralSolverTest, SplatSurfaceAccuracy)
{
    RNG rnd(0);
    int chanLut[] = {1,3,4};

    for (int i = 0; i < 5; i++)
    {
        Size sz(rnd.uniform(512, 1024), rnd.uniform(512, 1024));

        int guideCn = rnd.uniform(0, 2); // 1 or 3 channels
        Mat guide(sz, CV_MAKE_TYPE(CV_8U, chanLut[guideCn]));
        randu(guide, 0, 255);

        Scalar surfaceValue;
        int srcCn = rnd.uniform(0, 3); // 1, 3 or 4 channels
        rnd.fill(surfaceValue, RNG::UNIFORM, 0, 255);
        Mat src(sz, CV_MAKE_TYPE(CV_16S, chanLut[srcCn]), surfaceValue);
        Mat confidence(sz, CV_MAKE_TYPE(CV_8U, 1), 255);

        double sigma_spatial = rnd.uniform(4.0, 40.0);
        double sigma_luma = rnd.uniform(4.0, 40.0);
        double sigma_chroma  = rnd.uniform(4.0, 40.0);

        Mat res;
        fastBilateralSolverFilter(guide, src, confidence, res, sigma_spatial, sigma_luma, sigma_chroma);

        // When filtering a constant image we should get the same image:
        double normL1 = cvtest::norm(src, res, NORM_L1)/src.total()/src.channels();
        EXPECT_LE(normL1, 1.0/64);
    }
}

#define COUNT_EXCEED(MAT1, MAT2, THRESHOLD, PIXEL_COUNT) \
{                                                        \
    Mat diff, count;                                     \
    absdiff(MAT1.reshape(1), MAT2.reshape(1), diff);     \
    cvtest::compare(diff, THRESHOLD, count, CMP_GT);     \
    PIXEL_COUNT = countNonZero(count.reshape(1));        \
    PIXEL_COUNT /= MAT1.channels();                      \
}

TEST(FastBilateralSolverTest, ReferenceAccuracy)
{
    string dir = getDataDir() + "cv/edgefilter";

    Mat src = imread(dir + "/kodim23.png");
    Mat ref = imread(dir + "/fbs/kodim23_spatial=16_luma=16_chroma=16.png");

    Mat confidence(src.size(), CV_MAKE_TYPE(CV_8U, 1), 255);

    ASSERT_FALSE(src.empty());
    ASSERT_FALSE(ref.empty());

    Mat res;
    fastBilateralSolverFilter(src,src,confidence,res, 16.0, 16.0, 16.0);

    double totalMaxError = 1.0/64.0*src.total()*src.channels();

    EXPECT_LE(cvtest::norm(res, ref, NORM_L2), totalMaxError);
#if defined (__x86_64__) || defined (_M_X64)
    EXPECT_LE(cvtest::norm(res, ref, NORM_INF), 1);
#else
    // fastBilateralSolverFilter is not bit-exact
    int pixelCount = 0;
    COUNT_EXCEED(res, ref, 2, pixelCount);
    EXPECT_LE(pixelCount, (int)(res.cols*res.rows*1/100));
#endif
}

INSTANTIATE_TEST_CASE_P(FullSet, FastBilateralSolverTest,Combine(Values(szODD, szQVGA), SrcTypes::all(), GuideTypes::all()));

}
}

#endif //HAVE_EIGEN
