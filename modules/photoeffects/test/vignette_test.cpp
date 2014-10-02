#include "test_precomp.hpp"

using namespace cv;
using namespace cv::photoeffects;

using namespace std;

TEST(photoeffects_vignette, invalid_arguments)
{
    Mat image(100, 100, CV_8UC1);
    Mat dst;
    Size rectangle;
    rectangle.height = 0;
    rectangle.width = 0;

    EXPECT_ERROR(CV_StsAssert, vignette(image, dst, rectangle));
}

TEST(photoeffects_vignette, test)
{
    Mat image(100, 100, CV_8UC3);
    Mat dst;
    Size rectangle;
    rectangle.height = image.rows / 1.5f;
    rectangle.width = image.cols / 2.0f;

    EXPECT_EQ(0, vignette(image, dst, rectangle));
}

TEST(photoeffects_vignette, regression)
{
    string input = cvtest::TS::ptr()->get_data_path() + "photoeffects/vignette_test.png";
    string expectedOutput = cvtest::TS::ptr()->get_data_path() + "photoeffects/vignette_test_result.png";

    Mat image, dst, rightDst;
    image = imread(input, CV_LOAD_IMAGE_COLOR);
    rightDst = imread(expectedOutput, CV_LOAD_IMAGE_COLOR);

    if (image.empty())
    {
        FAIL() << "Can't read " + input + " image";
    }
    if (rightDst.empty())
    {
        FAIL() << "Can't read " + expectedOutput + " image";
    }

    Size rect;
    rect.height = image.rows / 1.5f;
    rect.width = image.cols / 2.0f;

    EXPECT_EQ(0, vignette(image, dst, rect));

    Mat diff = abs(rightDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
