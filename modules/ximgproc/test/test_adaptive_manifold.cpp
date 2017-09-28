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
using namespace cv;
using namespace cv::ximgproc;

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

static string getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

static void checkSimilarity(InputArray res, InputArray ref, double maxNormInf = 1, double maxNormL2 = 1.0 / 64)
{
    double normInf = cvtest::norm(res, ref, NORM_INF);
    double normL2 = cvtest::norm(res, ref, NORM_L2) / res.total();

    if (maxNormInf >= 0) EXPECT_LE(normInf, maxNormInf);
    if (maxNormL2 >= 0) EXPECT_LE(normL2, maxNormL2);
}

TEST(AdaptiveManifoldTest, SplatSurfaceAccuracy)
{
    RNG rnd(0);

    cv::setNumThreads(cv::getNumberOfCPUs());

    for (int i = 0; i < 5; i++)
    {
        Size sz(rnd.uniform(512, 1024), rnd.uniform(512, 1024));

        int guideCn = rnd.uniform(1, 8);
        Mat guide(sz, CV_MAKE_TYPE(CV_32F, guideCn));
        randu(guide, 0, 1);

        Scalar surfaceValue;
        int srcCn = rnd.uniform(1, 4);
        rnd.fill(surfaceValue, RNG::UNIFORM, 0, 255);
        Mat src(sz, CV_MAKE_TYPE(CV_8U, srcCn), surfaceValue);

        double sigma_s = rnd.uniform(1.0, 50.0);
        double sigma_r = rnd.uniform(0.1, 0.9);

        Mat res;
        amFilter(guide, src, res, sigma_s, sigma_r, false);

        double normInf = cvtest::norm(src, res, NORM_INF);
        EXPECT_EQ(normInf, 0);
    }
}

TEST(AdaptiveManifoldTest, AuthorsReferenceAccuracy)
{
    String srcImgPath = "cv/edgefilter/kodim23.png";

    String refPaths[] =
    {
        "cv/edgefilter/amf/kodim23_amf_ss5_sr0.3_ref.png",
        "cv/edgefilter/amf/kodim23_amf_ss30_sr0.1_ref.png",
        "cv/edgefilter/amf/kodim23_amf_ss50_sr0.3_ref.png"
    };

    pair<double, double> refParams[] =
    {
        make_pair(5.0, 0.3),
        make_pair(30.0, 0.1),
        make_pair(50.0, 0.3)
    };

    String refOutliersPaths[] =
    {
        "cv/edgefilter/amf/kodim23_amf_ss5_sr0.1_outliers_ref.png",
        "cv/edgefilter/amf/kodim23_amf_ss15_sr0.3_outliers_ref.png",
        "cv/edgefilter/amf/kodim23_amf_ss50_sr0.5_outliers_ref.png"
    };

    pair<double, double> refOutliersParams[] =
    {
        make_pair(5.0, 0.1),
        make_pair(15.0, 0.3),
        make_pair(50.0, 0.5),
    };

    Mat srcImg = imread(getOpenCVExtraDir() + srcImgPath);
    ASSERT_TRUE(!srcImg.empty());

    cv::setNumThreads(cv::getNumberOfCPUs());

    for (int i = 0; i < 3; i++)
    {
        Mat refRes = imread(getOpenCVExtraDir() + refPaths[i]);
        double sigma_s = refParams[i].first;
        double sigma_r = refParams[i].second;
        ASSERT_TRUE(!refRes.empty());

        Mat res;
        Ptr<AdaptiveManifoldFilter> amf = createAMFilter(sigma_s, sigma_r, false);
        amf->setUseRNG(false);
        amf->filter(srcImg, res, srcImg);
        amf->collectGarbage();

        checkSimilarity(res, refRes);
    }

    for (int i = 0; i < 3; i++)
    {
        Mat refRes = imread(getOpenCVExtraDir() + refOutliersPaths[i]);
        double sigma_s = refOutliersParams[i].first;
        double sigma_r = refOutliersParams[i].second;
        ASSERT_TRUE(!refRes.empty());

        Mat res;
        Ptr<AdaptiveManifoldFilter> amf = createAMFilter(sigma_s, sigma_r, true);
        amf->setUseRNG(false);
        amf->filter(srcImg, res, srcImg);
        amf->collectGarbage();

        checkSimilarity(res, refRes);
    }
}

typedef tuple<string, string> AMRefTestParams;
typedef TestWithParam<AMRefTestParams> AdaptiveManifoldRefImplTest;

Ptr<AdaptiveManifoldFilter> createAMFilterRefImpl(double sigma_s, double sigma_r, bool adjust_outliers = false);

TEST_P(AdaptiveManifoldRefImplTest, RefImplAccuracy)
{
    AMRefTestParams params = GetParam();

    string guideFileName = get<0>(params);
    string srcFileName = get<1>(params);

    Mat guide = imread(getOpenCVExtraDir() + guideFileName);
    Mat src = imread(getOpenCVExtraDir() + srcFileName);
    ASSERT_TRUE(!guide.empty() && !src.empty());

    int seed = 10 * (int)guideFileName.length() + (int)srcFileName.length();
    RNG rnd(seed);

    //inconsistent downsample/upsample operations in reference implementation
    Size dstSize((guide.cols + 15) & ~15, (guide.rows + 15) & ~15);

    resize(guide, guide, dstSize);
    resize(src, src, dstSize);

    for (int iter = 0; iter < 4; iter++)
    {
        double sigma_s = rnd.uniform(1.0, 50.0);
        double sigma_r = rnd.uniform(0.1, 0.9);
        bool adjust_outliers = (iter % 2 == 0);

        cv::setNumThreads(cv::getNumberOfCPUs());
        Mat res;
        amFilter(guide, src, res, sigma_s, sigma_r, adjust_outliers);

        cv::setNumThreads(1);
        Mat resRef;
        Ptr<AdaptiveManifoldFilter> amf = createAMFilterRefImpl(sigma_s, sigma_r, adjust_outliers);
        amf->filter(src, resRef, guide);

        //results of reference implementation may differ on small sigma_s into small isolated region
        //due to low single-precision floating point numbers accuracy
        //therefore the threshold of inf norm was increased
        checkSimilarity(res, resRef, 25);
    }
}

INSTANTIATE_TEST_CASE_P(TypicalSet, AdaptiveManifoldRefImplTest,
    Combine(
    Values("cv/edgefilter/kodim23.png", "cv/npr/test4.png"),
    Values("cv/edgefilter/kodim23.png", "cv/npr/test4.png")
));

}
