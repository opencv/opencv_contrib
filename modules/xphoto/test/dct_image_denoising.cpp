// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

    TEST(xphoto_dctimagedenoising, regression)
    {
        cv::String subfolder = "cv/xphoto/";
        cv::String dir = cvtest::TS::ptr()->get_data_path() + subfolder + "dct_image_denoising/";
        int nTests = 1;

        double thresholds[] = {0.2};

        int psize[] = {8};
        double sigma[] = {9.0};

        for (int i = 0; i < nTests; ++i)
        {
            cv::String srcName = dir + cv::format( "sources/%02d.png", i + 1);
            cv::Mat src = cv::imread( srcName, 1 );
            ASSERT_TRUE(!src.empty());

            cv::String previousResultName = dir + cv::format( "results/%02d.png", i + 1 );
            cv::Mat previousResult = cv::imread( previousResultName, 1 );
            ASSERT_TRUE(!src.empty());

            cv::Mat currentResult;

            cv::xphoto::dctDenoising(src, currentResult, sigma[i], psize[i]);

            cv::Mat sqrError = ( currentResult - previousResult )
                .mul( currentResult - previousResult );
            cv::Scalar mse = cv::sum(sqrError) / cv::Scalar::all( double(sqrError.total()*sqrError.channels()) );

            EXPECT_LE( mse[0] + mse[1] + mse[2] + mse[3], thresholds[i] );
        }
    }

    TEST(xphoto_dctimagedenoising, synthetic_gray_and_bgr_in_place_parity)
    {
        RNG& rng = theRNG();
        Mat cleanGray(72, 96, CV_8UC1);
        for (int r = 0; r < cleanGray.rows; ++r)
        {
            for (int c = 0; c < cleanGray.cols; ++c)
                cleanGray.at<uchar>(r, c) = saturate_cast<uchar>(40 + ((r / 12 + c / 12) % 2) * 140);
        }

        Mat noise(cleanGray.size(), CV_16SC1);
        rng.fill(noise, RNG::NORMAL, 0, 8);
        Mat noisyGray;
        cv::add(cleanGray, noise, noisyGray, noArray(), CV_8UC1);

        Mat denoisedGray, denoisedGrayInPlace = noisyGray.clone();
        cv::xphoto::dctDenoising(noisyGray, denoisedGray, 9.0, 8);
        cv::xphoto::dctDenoising(denoisedGrayInPlace, denoisedGrayInPlace, 9.0, 8);
        EXPECT_EQ(0.0, cv::norm(denoisedGray, denoisedGrayInPlace, NORM_INF));

        Rect inner(8, 8, cleanGray.cols - 16, cleanGray.rows - 16);
        EXPECT_LT(cv::norm(denoisedGray(inner), cleanGray(inner), NORM_L2),
                  cv::norm(noisyGray(inner), cleanGray(inner), NORM_L2));

        Mat noisyBgr;
        cvtColor(noisyGray, noisyBgr, COLOR_GRAY2BGR);
        Mat denoisedBgr, denoisedBgr32f, noisyBgr32f;
        cv::xphoto::dctDenoising(noisyBgr, denoisedBgr, 9.0, 8);
        EXPECT_EQ(noisyBgr.size(), denoisedBgr.size());
        EXPECT_EQ(CV_8UC3, denoisedBgr.type());

        noisyBgr.convertTo(noisyBgr32f, CV_32FC3);
        cv::xphoto::dctDenoising(noisyBgr32f, denoisedBgr32f, 9.0, 8);
        EXPECT_TRUE(cv::checkRange(denoisedBgr32f));
    }

}} // namespace
