#include "precomp.hpp"

using namespace cv;

TEST(photoeffects_glow, test) {
    Mat image(10, 10, CV_32FC3), dst;
    image = Mat::zeros(10, 10, CV_32FC3);

    EXPECT_EQ(0, glow(image, dst, 1.0f, 0.5f));
}

TEST(photoeffects_glow, regression) {
    string input = "./testdata/glow_test.png";
    string expectedOutput = "./testdata/glow_test_result.png";

    Mat image, rightDst;

    image = imread(input, CV_LOAD_IMAGE_COLOR);
    rightDst = imread(expectedOutput, CV_LOAD_IMAGE_COLOR);

    if (image.empty())
        FAIL() << "Can't read " + input + " image";
    if (rightDst.empty())
        FAIL() << "Can't read " + expectedOutput + " image";

    Mat dst;
    EXPECT_EQ(0, glow(image, dst, 33, 0.9f));
    Mat diff = abs(rightDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}

TEST(photoeffects_glow, bad_radius) {
    Mat image(10, 10, CV_32FC3), dst;

    EXPECT_ERROR(CV_StsAssert, glow(image, dst, -1, 0.5f));
}

TEST(photoeffects_glow, bad_intensity) {
    Mat image(10, 10, CV_32FC3), dst;

    EXPECT_ERROR(CV_StsAssert, glow(image, dst, 5.0f, 5.0f));
    EXPECT_ERROR(CV_StsAssert, glow(image, dst, 5.0f, -5.0f));
}

TEST(photoeffects_glow, bad_image) {
    Mat image(10, 10, CV_8UC1), dst;

    EXPECT_ERROR(CV_StsAssert, glow(image, dst, 5.0f, 0.5f));
}