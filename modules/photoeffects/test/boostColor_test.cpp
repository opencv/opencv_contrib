#include "test_precomp.hpp"

using namespace cv;
using namespace cv::photoeffects;

using namespace std;

TEST(photoeffects_boostColor, regression)
{
    string input = cvtest::TS::ptr()->get_data_path() + "photoeffects/boostColor_test.png";
    string expectedOutput = cvtest::TS::ptr()->get_data_path() + "photoeffects/boostColor_test_result.png";

    Mat image = imread(input, CV_LOAD_IMAGE_COLOR);
    Mat rightDst = imread(expectedOutput, CV_LOAD_IMAGE_COLOR);

    if (image.empty())
        FAIL() << "Can't read " + input + " image";
    if (rightDst.empty())
        FAIL() << "Can't read " + expectedOutput + " image";

    Mat dst;
    boostColor(image, dst, 0.5f);
    Mat diff = abs(rightDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}

TEST(photoeffects_boostColor, bad_intensity)
{
    Mat image(10, 10, CV_32FC3), dst;

    EXPECT_ERROR(CV_StsAssert, boostColor(image, dst, -1.0f));
    EXPECT_ERROR(CV_StsAssert, boostColor(image, dst, 2.0f));
}

TEST(photoeffects_boostColor, bad_image)
{
    Mat image(10, 10, CV_8UC1), dst;

    EXPECT_ERROR(CV_StsAssert, boostColor(image, dst, 0.5f));
}