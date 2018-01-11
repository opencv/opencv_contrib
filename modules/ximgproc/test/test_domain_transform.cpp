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

static string getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

CV_ENUM(SupportedTypes, CV_8UC1, CV_8UC3, CV_32FC1); // reduced set
CV_ENUM(ModeType, DTF_NC, DTF_IC, DTF_RF)
typedef tuple<Size, ModeType, SupportedTypes, SupportedTypes> DTParams;

Mat convertTypeAndSize(Mat src, int dstType, Size dstSize)
{
    Mat dst;
    CV_Assert(src.channels() == 3);

    int dstChannels = CV_MAT_CN(dstType);

    if (dstChannels == 1)
    {
        cvtColor(src, dst, COLOR_BGR2GRAY);
    }
    else if (dstChannels == 2)
    {
        Mat srcCn[3];
        split(src, srcCn);
        merge(srcCn, 2, dst);
    }
    else if (dstChannels == 3)
    {
        dst = src.clone();
    }
    else if (dstChannels == 4)
    {
        Mat srcCn[4];
        split(src, srcCn);
        srcCn[3] = srcCn[0].clone();
        merge(srcCn, 4, dst);
    }

    dst.convertTo(dst, dstType);
    resize(dst, dst, dstSize, 0, 0, dstType == CV_32FC1 ? INTER_LINEAR : INTER_LINEAR_EXACT);

    return dst;
}

TEST(DomainTransformTest, SplatSurfaceAccuracy)
{
    static int dtModes[] = {DTF_NC, DTF_RF, DTF_IC};
    RNG rnd(0);

    for (int i = 0; i < 15; i++)
    {
        Size sz(rnd.uniform(512, 1024), rnd.uniform(512, 1024));

        int guideCn = rnd.uniform(1, 4);
        Mat guide(sz, CV_MAKE_TYPE(CV_32F, guideCn));
        randu(guide, 0, 255);

        Scalar surfaceValue;
        int srcCn = rnd.uniform(1, 4);
        rnd.fill(surfaceValue, RNG::UNIFORM, 0, 255);
        Mat src(sz, CV_MAKE_TYPE(CV_8U, srcCn), surfaceValue);

        double sigma_s = rnd.uniform(1.0, 100.0);
        double sigma_r = rnd.uniform(1.0, 100.0);
        int mode = dtModes[i%3];

        Mat res;
        dtFilter(guide, src, res, sigma_s, sigma_r, mode, 1);

        double normL1 = cvtest::norm(src, res, NORM_L1)/src.total()/src.channels();
        EXPECT_LE(normL1, 1.0/64);
    }
}

typedef TestWithParam<DTParams> DomainTransformTest;
TEST_P(DomainTransformTest, MultiThreadReproducibility)
{
    if (cv::getNumberOfCPUs() == 1)
        return;

    double MAX_DIF = 1.0;
    double MAX_MEAN_DIF = 1.0 / 256.0;
    int loopsCount = 2;
    RNG rng(0);

    DTParams params = GetParam();
    Size size = get<0>(params);
    int mode = get<1>(params);
    int guideType = get<2>(params);
    int srcType = get<3>(params);

    Mat original = imread(getOpenCVExtraDir() + "cv/edgefilter/statue.png");
    Mat guide = convertTypeAndSize(original, guideType, size);
    Mat src = convertTypeAndSize(original, srcType, size);

    for (int iter = 0; iter <= loopsCount; iter++)
    {
        double ss = rng.uniform(0.0, 100.0);
        double sc = rng.uniform(0.0, 100.0);

        cv::setNumThreads(cv::getNumberOfCPUs());
        Mat resMultithread;
        dtFilter(guide, src, resMultithread, ss, sc, mode);

        cv::setNumThreads(1);
        Mat resSingleThread;
        dtFilter(guide, src, resSingleThread, ss, sc, mode);

        EXPECT_LE(cv::norm(resSingleThread, resMultithread, NORM_INF), MAX_DIF);
        EXPECT_LE(cv::norm(resSingleThread, resMultithread, NORM_L1), MAX_MEAN_DIF*src.total());
    }
}

INSTANTIATE_TEST_CASE_P(FullSet, DomainTransformTest,
    Combine(Values(szODD, szQVGA), ModeType::all(), SupportedTypes::all(), SupportedTypes::all())
);

template<typename SrcVec>
Mat getChessMat1px(Size sz, double whiteIntensity = 255)
{
    typedef typename DataType<SrcVec>::channel_type SrcType;

    Mat dst(sz, traits::Type<SrcVec>::value);

    SrcVec black = SrcVec::all(0);
    SrcVec white = SrcVec::all((SrcType)whiteIntensity);

    for (int i = 0; i < dst.rows; i++)
        for (int j = 0; j < dst.cols; j++)
            dst.at<SrcVec>(i, j) = ((i + j) % 2) ? white : black;

    return dst;
}

TEST(DomainTransformTest, ChessBoard_NC_accuracy)
{
    RNG rng(0);
    double MAX_DIF = 1;
    Size sz = szVGA;
    double ss = 80;
    double sc = 60;

    Mat srcb = randomMat(rng, sz, CV_8UC4, 0, 255, true);
    Mat srcf = randomMat(rng, sz, CV_32FC4, 0, 255, true);
    Mat chessb = getChessMat1px<Vec3b>(sz);

    Mat dstb, dstf;
    dtFilter(chessb, srcb.clone(), dstb, ss, sc, DTF_NC);
    dtFilter(chessb, srcf.clone(), dstf, ss, sc, DTF_NC);

    EXPECT_LE(cv::norm(srcb, dstb, NORM_INF), MAX_DIF);
    EXPECT_LE(cv::norm(srcf, dstf, NORM_INF), MAX_DIF);
}

TEST(DomainTransformTest, BoxFilter_NC_accuracy)
{
    double MAX_DIF = 1;
    int radius = 5;
    double sc = 1.0;
    double ss = 1.01*radius / sqrt(3.0);

    Mat src = imread(getOpenCVExtraDir() + "cv/edgefilter/statue.png");
    ASSERT_TRUE(!src.empty());

    Mat1b guide(src.size(), 200);
    Mat res_dt, res_box;

    blur(src, res_box, Size(2 * radius + 1, 2 * radius + 1));
    dtFilter(guide, src, res_dt, ss, sc, DTF_NC, 1);

    EXPECT_LE(cv::norm(res_dt, res_box, NORM_L2), MAX_DIF*src.total());
}

TEST(DomainTransformTest, AuthorReferenceAccuracy)
{
    string dir = getOpenCVExtraDir() + "cv/edgefilter";
    double ss = 30;
    double sc = 0.2 * 255;

    Mat src = imread(dir + "/statue.png");
    Mat ref_NC = imread(dir + "/dt/authors_statue_NC_ss30_sc0.2.png");
    Mat ref_IC = imread(dir + "/dt/authors_statue_IC_ss30_sc0.2.png");
    Mat ref_RF = imread(dir + "/dt/authors_statue_RF_ss30_sc0.2.png");

    ASSERT_FALSE(src.empty());
    ASSERT_FALSE(ref_NC.empty());
    ASSERT_FALSE(ref_IC.empty());
    ASSERT_FALSE(ref_RF.empty());

    cv::setNumThreads(cv::getNumberOfCPUs());
    Mat res_NC, res_IC, res_RF;
    dtFilter(src, src, res_NC, ss, sc, DTF_NC);
    dtFilter(src, src, res_IC, ss, sc, DTF_IC);
    dtFilter(src, src, res_RF, ss, sc, DTF_RF);

    double totalMaxError = 1.0/64.0*src.total();

    EXPECT_LE(cvtest::norm(res_NC, ref_NC, NORM_L2), totalMaxError);
    EXPECT_LE(cvtest::norm(res_NC, ref_NC, NORM_INF), 1);

    EXPECT_LE(cvtest::norm(res_IC, ref_IC, NORM_L2), totalMaxError);
    EXPECT_LE(cvtest::norm(res_IC, ref_IC, NORM_INF), 1);

    EXPECT_LE(cvtest::norm(res_RF, ref_RF, NORM_L2), totalMaxError);
    EXPECT_LE(cvtest::norm(res_IC, ref_IC, NORM_INF), 1);
}

}
