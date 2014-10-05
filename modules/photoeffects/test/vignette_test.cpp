#include "test_precomp.hpp"

using namespace cv;
using namespace cv::photoeffects;

using namespace std;

TEST(photoeffects_vignette, incorrect_image)
{
    Mat image(100, 100, CV_8UC1);
    Mat dst;
    Size rectangle;
    rectangle.height = image.rows / 1.5f;
    rectangle.width = image.cols / 2.0f;

    EXPECT_ERROR(CV_StsAssert, vignette(image, dst, rectangle));
}

TEST(photoeffects_vignette, incorrect_ellipse_size)
{
    Mat image(100, 100, CV_8UC3);
    Mat dst;
    Size rectangle;
    rectangle.height = 0.0f;
    rectangle.width = 0.0f;

    EXPECT_ERROR(CV_StsAssert, vignette(image, dst, rectangle));
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

    Size rectangle;
    rectangle.height = image.rows / 1.5f;
    rectangle.width = image.cols / 2.0f;

    vignette(image, dst, rectangle);

    Mat diff = abs(rightDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
