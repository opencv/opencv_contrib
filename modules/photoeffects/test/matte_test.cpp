#include "test_precomp.hpp"

using namespace cv;
using namespace cv::photoeffects;

using namespace std;

TEST(photoeffects_matte, bad_intensity)
{
   Mat image(10, 10, CV_32FC3),dst;
   EXPECT_ERROR(CV_StsAssert, matte(image, dst, -15.0f));
}

TEST(photoeffects_matte, bad_image)
{
    Mat src(10, 10, CV_8UC1);
    Mat dst;

    EXPECT_ERROR(CV_StsAssert, matte(src, dst,25));
}

TEST(photoeffects_matte, regression)
{
    string input = cvtest::TS::ptr()->get_data_path() + "photoeffects/matte_test.png";
    string expectedOutput = cvtest::TS::ptr()->get_data_path() + "photoeffects/matte_test_result.png";
    Mat src = imread(input, CV_LOAD_IMAGE_COLOR);
    Mat expectedDst = imread(expectedOutput, CV_LOAD_IMAGE_COLOR);
    if(src.empty())
    {
        FAIL() << "Can't read " + input + "image";
    }
    if(expectedDst.empty())
    {
        FAIL() << "Can't read " + expectedOutput + "image";
    }
    Mat dst;
    matte(src, dst, 25);
    dst.convertTo(dst, CV_8UC3, 255);
    Mat diff = abs(expectedDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
