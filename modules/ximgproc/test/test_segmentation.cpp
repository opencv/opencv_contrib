// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

using namespace cv::ximgproc::segmentation;

namespace opencv_test { namespace {

// See https://github.com/opencv/opencv_contrib/issues/3544

typedef tuple< int, double, int> GraphSegmentationTestParam;
typedef testing::TestWithParam<GraphSegmentationTestParam> createGraphSegmentation_issueC3544;

TEST_P( createGraphSegmentation_issueC3544, regression )
{
    const int mat_type     = get<0>(GetParam());
    const double alpha     = get<1>(GetParam());
    const int expected_max = get<2>(GetParam());

    cv::String testImagePath = cvtest::TS::ptr()->get_data_path() + "cv/ximgproc/pascal_voc_bird.png";
    Mat testImg = imread(testImagePath, cv::IMREAD_COLOR);
    ASSERT_FALSE(testImg.empty()) << "Could not load input image " << testImagePath;

    if( CV_MAT_CN( mat_type ) == 1 ) {
        cvtColor( testImg, testImg, COLOR_BGR2GRAY );
    }else if( CV_MAT_CN(mat_type) == 2 ){
        // Do nothing.
    }else if( CV_MAT_CN(mat_type) == 3 ){
        // Do nothing.
    }else if( CV_MAT_CN(mat_type) == 4 ){
        cvtColor( testImg, testImg, COLOR_BGR2BGRA );
    }

    if ( alpha != 1.0 )
    {
        testImg.convertTo( testImg, mat_type, alpha );
    }

    // Force to convert from 3ch to 2ch.
    if ( CV_MAT_CN(mat_type) == 2 )
    {
        std::vector<cv::Mat> planes;
        cv::split( testImg, planes );
        planes.erase( planes.begin() );
        cv::merge( planes, testImg );
    }

    Mat segImg;
    Ptr<GraphSegmentation> gs = createGraphSegmentation();

    if( expected_max > 0 )
    {
        ASSERT_NO_THROW( gs->processImage(testImg, segImg) );

        double minValue = DBL_MIN;
        double maxValue = DBL_MAX;
        ASSERT_NO_THROW( minMaxLoc(segImg, &minValue, &maxValue) );
        EXPECT_EQ( static_cast<int>(minValue), 0 );
        EXPECT_EQ( static_cast<int>(maxValue), expected_max );
    }
    else
    {
        ASSERT_ANY_THROW( gs->processImage(testImg, segImg) );
    }
}

/*
 * CV_8S results were different because some information are lost.    : [37,15,16,16]
 *
 * CV_16S results were slightly different from others.
 *
 * [CV_16S] -(convertTo(alpha)-> [CV_32F]                             : [13,18,15,15]
 * [CV_16S] -(convertTo(alpha)-> [CV_8U ] -(convertTo(1.0)-> [CV_32F] : [14,17,17,17]
 *
 * [CV_any] -(convertTo(alpha)-> [CV_32F]                             : [14,17,17,17]
 */

const GraphSegmentationTestParam gstest_list[] =
{
    make_tuple<int, double, int>( CV_8UC1,  1.0,                14 ),
    make_tuple<int, double, int>( CV_8UC2,  1.0,                17 ),
    make_tuple<int, double, int>( CV_8UC3,  1.0,                17 ),
    make_tuple<int, double, int>( CV_8UC4,  1.0,                17 ),

    make_tuple<int, double, int>( CV_8SC1,  127. / 255.,        37 ),
    make_tuple<int, double, int>( CV_8SC2,  127. / 255.,        15 ),
    make_tuple<int, double, int>( CV_8SC3,  127. / 255.,        16 ),
    make_tuple<int, double, int>( CV_8SC4,  127. / 255.,        16 ),

    make_tuple<int, double, int>( CV_16UC1, 65535. / 255.,      14 ),
    make_tuple<int, double, int>( CV_16UC2, 65535. / 255.,      17 ),
    make_tuple<int, double, int>( CV_16UC3, 65535. / 255.,      17 ),
    make_tuple<int, double, int>( CV_16UC4, 65535. / 255.,      17 ),

    make_tuple<int, double, int>( CV_16SC1, 32767. / 255.,      13 ),
    make_tuple<int, double, int>( CV_16SC2, 32767. / 255.,      18 ),
    make_tuple<int, double, int>( CV_16SC3, 32767. / 255.,      15 ),
    make_tuple<int, double, int>( CV_16SC4, 32767. / 255.,      15 ),

    make_tuple<int, double, int>( CV_32SC1, 2147483647. / 255., -1 ),
    make_tuple<int, double, int>( CV_32SC2, 2147483647. / 255., -1 ),
    make_tuple<int, double, int>( CV_32SC3, 2147483647. / 255., -1 ),
    make_tuple<int, double, int>( CV_32SC4, 2147483647. / 255., -1 ),

    make_tuple<int, double, int>( CV_32FC1, 1. / 255.,          14 ),
    make_tuple<int, double, int>( CV_32FC2, 1. / 255.,          17 ),
    make_tuple<int, double, int>( CV_32FC3, 1. / 255.,          17 ),
    make_tuple<int, double, int>( CV_32FC4, 1. / 255.,          17 ),

    make_tuple<int, double, int>( CV_64FC1, 1. / 255.,          14 ),
    make_tuple<int, double, int>( CV_64FC2, 1. / 255.,          17 ),
    make_tuple<int, double, int>( CV_64FC3, 1. / 255.,          17 ),
    make_tuple<int, double, int>( CV_64FC4, 1. / 255.,          17 ),

    make_tuple<int, double, int>( CV_16FC1, 1. / 255.,          -1 ),
    make_tuple<int, double, int>( CV_16FC2, 1. / 255.,          -1 ),
    make_tuple<int, double, int>( CV_16FC3, 1. / 255.,          -1 ),
    make_tuple<int, double, int>( CV_16FC4, 1. / 255.,          -1 )

#ifdef CV_16BF
    make_tuple<int, double, int>( CV_16BFC1, 1. / 255.,         -1 ),
    make_tuple<int, double, int>( CV_16BFC2, 1. / 255.,         -1 ),
    make_tuple<int, double, int>( CV_16BFC3, 1. / 255.,         -1 ),
    make_tuple<int, double, int>( CV_16BFC4, 1. / 255.,         -1 )
#endif // CV_16BF

#ifdef CV_Bool
    ,
    make_tuple<int, double, int>( CV_BoolC1, 1.,                -1 ),
    make_tuple<int, double, int>( CV_BoolC2, 1.,                -1 ),
    make_tuple<int, double, int>( CV_BoolC3, 1.,                -1 ),
    make_tuple<int, double, int>( CV_BoolC4, 1.,                -1 )
#endif // CV_Bool

#ifdef CV_64U
    ,
    make_tuple<int, double, int>( CV_64UC1,  1.,                -1 ),
    make_tuple<int, double, int>( CV_64UC2,  1.,                -1 ),
    make_tuple<int, double, int>( CV_64UC3,  1.,                -1 ),
    make_tuple<int, double, int>( CV_64UC4,  1.,                -1 )
#endif // CV_64U

#ifdef CV_64S
    ,
    make_tuple<int, double, int>( CV_64SC1,  1.,                -1 ),
    make_tuple<int, double, int>( CV_64SC2,  1.,                -1 ),
    make_tuple<int, double, int>( CV_64SC3,  1.,                -1 ),
    make_tuple<int, double, int>( CV_64SC4,  1.,                -1 )
#endif // CV_64S

#ifdef CV_32U
    ,
    make_tuple<int, double, int>( CV_32UC1,  1.,                -1 ),
    make_tuple<int, double, int>( CV_32UC2,  1.,                -1 ),
    make_tuple<int, double, int>( CV_32UC3,  1.,                -1 ),
    make_tuple<int, double, int>( CV_32UC4,  1.,                -1 )
#endif // CV_32U

};

INSTANTIATE_TEST_CASE_P(GraphSegmentation,
                        createGraphSegmentation_issueC3544,
                        testing::ValuesIn(gstest_list));


TEST(ximgproc_ImageSegmentation, createGraphSegmentation_negativeValue)
{
    Ptr<GraphSegmentation> gs = createGraphSegmentation();
    Mat src ;
    Mat segImg;

    src = cv::Mat(320,240,CV_8SC3);
    src.at<int8_t>(0,0) = -1;
    ASSERT_ANY_THROW( gs->processImage(src, segImg) );

    src = cv::Mat(320,240,CV_16SC3);
    src.at<int16_t>(0,0) = -1;
    ASSERT_ANY_THROW( gs->processImage(src, segImg) );

    src = cv::Mat(320,240,CV_32FC3);
    src.at<float>(0,0) = -1.0;
    ASSERT_ANY_THROW( gs->processImage(src, segImg) );

    src = cv::Mat(320,240,CV_64FC3);
    src.at<double>(0,0) = -1.0;
    ASSERT_ANY_THROW( gs->processImage(src, segImg) );

}

}} // namespace
