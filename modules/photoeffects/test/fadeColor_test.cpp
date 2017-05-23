#include "test_precomp.hpp"

using namespace cv;
using namespace cv::photoeffects;

using namespace std;

TEST(photoeffects_fadeColor, invalid_image_format)
{
    Mat src(10, 20, CV_8UC2);
    Mat dst;

    EXPECT_ERROR(CV_StsAssert, fadeColor(src, dst, Point(5, 5), Point(5, 10)));
}

TEST(photoeffects_fadeColor, invalid_argument)
{
    Mat src(10, 20, CV_8UC1);
    Mat dst;

    EXPECT_ERROR(CV_StsAssert, fadeColor(src, dst, Point(50,5), Point(5,10)));
    EXPECT_ERROR(CV_StsAssert, fadeColor(src, dst, Point(5,5), Point(5,-10)));
    EXPECT_ERROR(CV_StsAssert, fadeColor(src, dst, Point(5,5), Point(5,5)));
}

TEST(photoeffects_fadeColor, regression)
{
    string input = cvtest::TS::ptr()->get_data_path() + "photoeffects/fadeColor_test.png";
    string expectedOutput = cvtest::TS::ptr()->get_data_path() + "photoeffects/fadeColor_result.png";

    Mat image, dst, rightDst;
    image = imread(input, CV_LOAD_IMAGE_COLOR);
    rightDst = imread(expectedOutput, CV_LOAD_IMAGE_COLOR);

    if (image.empty())
        FAIL() << "Can't read " + input + " image";
    if (rightDst.empty())
        FAIL() << "Can't read " + expectedOutput + " image";

    fadeColor(image, dst, Point(100, 100), Point(250, 250));

    Mat diff = abs(rightDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
