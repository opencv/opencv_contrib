#include "test_precomp.hpp"

using namespace cv;
using namespace cv::photoeffects;

using namespace std;

TEST(photoeffects_glow, regression) {
    string input = cvtest::TS::ptr()->get_data_path() + "photoeffects/glow_test.png";
    string expectedOutput = cvtest::TS::ptr()->get_data_path() + "photoeffects/glow_test_result.png";

    Mat image, rightDst;

    image = imread(input, CV_LOAD_IMAGE_COLOR);
    rightDst = imread(expectedOutput, CV_LOAD_IMAGE_COLOR);

    if (image.empty())
        FAIL() << "Can't read " + input + " image";
    if (rightDst.empty())
        FAIL() << "Can't read " + expectedOutput + " image";

    Mat dst;
    glow(image, dst, 33, 0.9f);
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

    EXPECT_ERROR(CV_StsAssert, glow(image, dst, 5, 5.0f));
    EXPECT_ERROR(CV_StsAssert, glow(image, dst, 5, -5.0f));
}

TEST(photoeffects_glow, bad_image) {
    Mat image(10, 10, CV_8UC1), dst;

    EXPECT_ERROR(CV_StsAssert, glow(image, dst, 5, 0.5f));
}