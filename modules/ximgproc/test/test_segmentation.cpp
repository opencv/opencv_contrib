// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

using namespace cv::ximgproc::segmentation;

namespace opencv_test { namespace {

// See https://github.com/opencv/opencv_contrib/issues/3544
TEST(ximgproc_ImageSegmentation, createGraphSegmentation_issueC3544)
{
    cv::String testImagePath = cvtest::TS::ptr()->get_data_path() + "cv/ximgproc/" + "pascal_voc_bird.png";
    Mat testImg = imread(testImagePath);
    ASSERT_FALSE(testImg.empty()) << "Could not load input image " << testImagePath;
    Ptr<GraphSegmentation> gs = createGraphSegmentation();
    Mat segImg;
    double min,max;

    // Try CV_8UC3
    Mat testImg_8UC3;
    testImg_8UC3 = testImg.clone();
    ASSERT_EQ(testImg_8UC3.type(), CV_8UC3 ) << "Input image is not CV_8UC3";
    ASSERT_NO_THROW( gs->processImage(testImg_8UC3, segImg) );
    ASSERT_NO_THROW( minMaxLoc(segImg, &min, &max) );
    EXPECT_EQ( min, 0 );
    EXPECT_EQ( max, 17 );

    // Try CV_8UC1
    Mat testImg_8UC1;
    cvtColor(testImg, testImg_8UC1, COLOR_BGR2GRAY);
    ASSERT_EQ(testImg_8UC1.type(), CV_8UC1 ) << "Input image is not CV_8UC1";
    ASSERT_NO_THROW( gs->processImage(testImg_8UC1, segImg) );
    ASSERT_NO_THROW( minMaxLoc(segImg, &min, &max) );
    EXPECT_EQ( min, 0 );
    EXPECT_EQ( max, 14 ); // Gray image

    // Try CV_8UC4
    Mat testImg_8UC4;
    cvtColor(testImg, testImg_8UC4, COLOR_BGR2BGRA);
    ASSERT_EQ(testImg_8UC4.type(), CV_8UC4 ) << "Input image is not CV_8UC4";
    ASSERT_NO_THROW( gs->processImage(testImg_8UC4, segImg) );
    ASSERT_NO_THROW( minMaxLoc(segImg, &min, &max) );
    EXPECT_EQ( min, 0 );
    EXPECT_EQ( max, 17 );

    // Try CV_16UC3
    Mat testImg_16UC3;
    testImg.convertTo(testImg_16UC3, CV_16U, 256. , 0.0);
    ASSERT_EQ(testImg_16UC3.type(), CV_16UC3 ) << "Input image is not CV_16UC3";
    ASSERT_NO_THROW( gs->processImage(testImg_16UC3, segImg) );
    ASSERT_NO_THROW( minMaxLoc(segImg, &min, &max) );
    EXPECT_EQ( min, 0 );
    EXPECT_EQ( max, 17 );

    // Try CV_32FC3
    Mat testImg_32FC3;
    testImg.convertTo(testImg_32FC3, CV_32F, 1./256. , 0.0);
    ASSERT_EQ(testImg_32FC3.type(), CV_32FC3 ) << "Input image is not CV_32FC3";
    ASSERT_NO_THROW( gs->processImage(testImg_32FC3, segImg) );
    ASSERT_NO_THROW( minMaxLoc(segImg, &min, &max) );
    EXPECT_EQ( min, 0 );
    EXPECT_EQ( max, 17 );

    // Try CV_64FC3
    Mat testImg_64FC3;
    testImg.convertTo(testImg_64FC3, CV_64F, 1./256. , 0.0);
    ASSERT_EQ(testImg_64FC3.type(), CV_64FC3 ) << "Input image is not CV_64FC3";
    ASSERT_NO_THROW( gs->processImage(testImg_64FC3, segImg) );
    ASSERT_NO_THROW( minMaxLoc(segImg, &min, &max) );
    EXPECT_EQ( min, 0 );
    EXPECT_EQ( max, 17 );

    // Unsupported Images
    ASSERT_ANY_THROW( gs->processImage( cv::Mat::zeros(320,240, CV_8SC1), segImg) );
    ASSERT_ANY_THROW( gs->processImage( cv::Mat::zeros(320,240, CV_8SC3), segImg) );
    ASSERT_ANY_THROW( gs->processImage( cv::Mat::zeros(320,240, CV_8SC4), segImg) );
    ASSERT_ANY_THROW( gs->processImage( cv::Mat::zeros(320,240, CV_16SC1), segImg) );
    ASSERT_ANY_THROW( gs->processImage( cv::Mat::zeros(320,240, CV_16SC3), segImg) );
    ASSERT_ANY_THROW( gs->processImage( cv::Mat::zeros(320,240, CV_16SC4), segImg) );
    ASSERT_ANY_THROW( gs->processImage( cv::Mat::zeros(320,240, CV_32SC1), segImg) );
    ASSERT_ANY_THROW( gs->processImage( cv::Mat::zeros(320,240, CV_32SC3), segImg) );
    ASSERT_ANY_THROW( gs->processImage( cv::Mat::zeros(320,240, CV_32SC4), segImg) );
}

}} // namespace
