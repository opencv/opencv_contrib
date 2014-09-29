#include "precomp.hpp"

using namespace cv;

TEST(photoeffects_warmify, invalid_image_format)
{
    Mat image(100, 100, CV_8UC1);
    Mat dst;

    EXPECT_ERROR(CV_StsAssert, warmify(image, dst, 30));
}

TEST(photoeffects_warmify, test)
{
    Mat image(100, 100, CV_8UC3);
    Mat dst;
    Vec3b intensity_src;
    Vec3b intensity_dst;

    EXPECT_EQ(0, warmify(image, dst, 30));
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            // blue_dst = blue_src
            intensity_src = image.at<Vec3b>(i, j);
            intensity_dst = dst.at<Vec3b>(i, j);
            EXPECT_LE (intensity_dst[0] - 1, intensity_src[0]);
            EXPECT_GE (intensity_dst[0] + 1, intensity_src[0]);
        }
    }
}

TEST(photoeffects_warmify, regression)
{
    string input = cvtest::TS::ptr()->get_data_path() + "testdata/warmify_test.png";
    string expectedOutput = cvtest::TS::ptr()->get_data_path() + "testdata/warmify_test_result.png";

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

    EXPECT_EQ(0, warmify(image, dst, 30));

    Mat diff = abs(rightDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
