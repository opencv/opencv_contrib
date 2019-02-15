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


}} // namespace
