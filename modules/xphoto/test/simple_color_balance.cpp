// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

    TEST(xphoto_simplecolorbalance, regression)
    {
        cv::String dir = cvtest::TS::ptr()->get_data_path() + "cv/xphoto/simple_white_balance/";
        int nTests = 8;
        cv::Ptr<cv::xphoto::WhiteBalancer> wb = cv::xphoto::createSimpleWB();

        for (int i = 0; i < nTests; ++i)
        {
            cv::String srcName = dir + cv::format( "sources/%02d.png", i + 1);
            cv::Mat src = cv::imread( srcName, 1 );
            ASSERT_TRUE(!src.empty());

            cv::String previousResultName = dir + cv::format( "results/%02d.jpg", i + 1 );
            cv::Mat previousResult = cv::imread( previousResultName, 1 );

            cv::Mat currentResult;
            wb->balanceWhite(src, currentResult);

            double psnr = cv::PSNR(currentResult, previousResult);

            EXPECT_GE( psnr, 30 );
        }
    }

    TEST(xphoto_simplecolorbalance, max_value)
    {
        const float oldMax = 24000., newMax = 65536.;

        Mat test = Mat::zeros(3,3,CV_32FC1);
        test.at<float>(0, 0) = oldMax;
        test.at<float>(0, 1) = oldMax / 2;
        test.at<float>(0, 2) = oldMax / 4;

        double minSrc, maxSrc;
        cv::minMaxIdx(test, &minSrc, &maxSrc);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin((float)minSrc);
        wb->setInputMax((float)maxSrc);
        wb->setOutputMin(0);
        wb->setOutputMax(newMax);

        wb->balanceWhite(test, test);

        double minDst, maxDst;
        cv::minMaxIdx(test, &minDst, &maxDst);

        ASSERT_NEAR(maxDst, newMax, newMax*1e-4);
    }


}} // namespace
