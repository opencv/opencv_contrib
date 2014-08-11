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

static void checkSimilarity(InputArray res, InputArray ref)
{
    double normInf = cvtest::norm(res, ref, NORM_INF);
    double normL2 = cvtest::norm(res, ref, NORM_L2) / res.total();

    EXPECT_LE(normInf, 1);
    EXPECT_LE(normL2, 1.0 / 64);
}

TEST(AdaptiveManifoldTest, SplatSurfaceAccuracy)
{
    RNG rnd(0);

    for (int i = 0; i < 10; i++)
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

    for (int i = 0; i < 3; i++)
    {
        Mat refRes = imread(getOpenCVExtraDir() + refPaths[i]);
        double sigma_s = refParams[i].first;
        double sigma_r = refParams[i].second;
        ASSERT_TRUE(!refRes.empty());

        Mat res;
        Ptr<AdaptiveManifoldFilter> amf = createAMFilter(sigma_s, sigma_r, false);
        amf->setBool("use_RNG", false);
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
        amf->setBool("use_RNG", false);
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

    int seed = 10 * guideFileName.length() + srcFileName.length();
    RNG rnd(seed);

    //inconsistent downsample/upsample operations in reference implementation
    Size dstSize((guide.cols + 15) & ~15, (guide.rows + 15) & ~15);

    resize(guide, guide, dstSize);
    resize(src, src, dstSize);

    for (int iter = 0; iter < 6; iter++)
    {
        double sigma_s = rnd.uniform(1.0, 50.0);
        double sigma_r = rnd.uniform(0.1, 0.9);
        bool adjust_outliers = (iter % 2 == 0);

        Mat res;
        amFilter(guide, src, res, sigma_s, sigma_r, adjust_outliers);

        Mat resRef;
        Ptr<AdaptiveManifoldFilter> amf = createAMFilterRefImpl(sigma_s, sigma_r, adjust_outliers);
        amf->filter(src, resRef, guide);

        checkSimilarity(res, resRef);
    }
}

INSTANTIATE_TEST_CASE_P(TypicalSet, AdaptiveManifoldRefImplTest, 
    Combine(
    Values("cv/shared/lena.png", "cv/edgefilter/kodim23.png", "cv/npr/test4.png"),
    Values("cv/shared/lena.png", "cv/edgefilter/kodim23.png", "cv/npr/test4.png")
));

}