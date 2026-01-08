// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"

namespace opencv_test { namespace {

static string getDataDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

CV_ENUM(SrcTypes, CV_16S);
CV_ENUM(GuideTypes, CV_8UC1, CV_8UC3)
typedef tuple<Size, SrcTypes, GuideTypes, bool, bool> DisparityWLSParams;
typedef TestWithParam<DisparityWLSParams> DisparityWLSFilterTest;

TEST(DisparityWLSFilterTest, ReferenceAccuracy)
{
    string dir = getDataDir() + "cv/disparityfilter";

    Mat left = imread(dir + "/left_view.png",IMREAD_COLOR);
    ASSERT_FALSE(left.empty());
    Mat left_disp  = imread(dir + "/disparity_left_raw.png",IMREAD_GRAYSCALE);
    ASSERT_FALSE(left_disp.empty());
    left_disp.convertTo(left_disp,CV_16S,16);
    Mat right_disp = imread(dir + "/disparity_right_raw.png",IMREAD_GRAYSCALE);
    ASSERT_FALSE(right_disp.empty());
    right_disp.convertTo(right_disp,CV_16S,-16);

    Mat GT;
    ASSERT_FALSE(readGT(dir + "/GT.png",GT));

    FileStorage ROI_storage( dir + "/ROI.xml", FileStorage::READ );
    Rect ROI((int)ROI_storage["x"],(int)ROI_storage["y"],(int)ROI_storage["width"],(int)ROI_storage["height"]);

    FileStorage reference_res( dir + "/reference_accuracy.xml", FileStorage::READ );
    double ref_MSE = (double)reference_res["MSE_after"];
    double ref_BadPercent = (double)reference_res["BadPercent_after"];

    Mat res;

    Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilterGeneric(true);
    wls_filter->setLambda(8000.0);
    wls_filter->setSigmaColor(0.5);
    wls_filter->filter(left_disp,left,res,right_disp,ROI);

    double MSE = computeMSE(GT,res,ROI);
    double BadPercent = computeBadPixelPercent(GT,res,ROI);
    double eps = 0.01;

    EXPECT_LE(MSE,ref_MSE+eps*ref_MSE);
    EXPECT_LE(BadPercent,ref_BadPercent+eps*ref_BadPercent);
}

TEST_P(DisparityWLSFilterTest, MultiThreadReproducibility)
{
    if (cv::getNumberOfCPUs() == 1)
        return;

    double MAX_DIF = 1.0;
    double MAX_MEAN_DIF = 1.0 / 256.0;
    int loopsCount = 2;
    RNG rng(0);

    DisparityWLSParams params = GetParam();
    Size size          = get<0>(params);
    int srcType        = get<1>(params);
    int guideType      = get<2>(params);
    bool use_conf      = get<3>(params);
    bool use_downscale = get<4>(params);

    Mat left(size, guideType);
    randu(left, 0, 255);
    Mat left_disp(size,srcType);
    int max_disp = (int)(size.width*0.1);
    randu(left_disp, 0, max_disp-1);
    Mat right_disp(size,srcType);
    randu(left_disp, -max_disp+1, 0);
    Rect ROI(max_disp,0,size.width-max_disp,size.height);

    if(use_downscale)
    {
        resize(left_disp,left_disp,Size(),0.5,0.5, INTER_LINEAR_EXACT);
        resize(right_disp,right_disp,Size(),0.5,0.5, INTER_LINEAR_EXACT);
        ROI = Rect(ROI.x/2,ROI.y/2,ROI.width/2,ROI.height/2);
    }

    int nThreads = cv::getNumThreads();
    if (nThreads == 1)
        throw SkipTestException("Single thread environment");
    for (int iter = 0; iter <= loopsCount; iter++)
    {
        double lambda = rng.uniform(100.0, 10000.0);
        double sigma  = rng.uniform(1.0, 100.0);

        Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilterGeneric(use_conf);
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);

        cv::setNumThreads(nThreads);
        Mat resMultiThread;
        wls_filter->filter(left_disp,left,resMultiThread,right_disp,ROI);

        cv::setNumThreads(1);
        Mat resSingleThread;
        wls_filter->filter(left_disp,left,resSingleThread,right_disp,ROI);

        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_INF), MAX_DIF);
        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_L1), MAX_MEAN_DIF*left.total());
    }
}
INSTANTIATE_TEST_CASE_P(FullSet,DisparityWLSFilterTest,Combine(Values(szODD, szQVGA), SrcTypes::all(), GuideTypes::all(),Values(true,false),Values(true,false)));


}} // namespace
