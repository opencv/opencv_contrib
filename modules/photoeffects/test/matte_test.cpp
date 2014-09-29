#include "precomp.hpp"

using namespace cv;

TEST(photoeffects_matte, test)
{
    Mat srcUCThreeChannels(10, 10, CV_8UC3);
    srcUCThreeChannels = Mat::zeros(10, 10, CV_8UC3);
    Point fisrtpoint(0, 0);
    Point secondpoint(10, 10);
    Mat srcFCThreeChannels(10, 10, CV_32FC3);
    srcFCThreeChannels = Mat::zeros(10, 10, CV_32FC3);
    EXPECT_EQ(0, matte(srcUCThreeChannels, srcUCThreeChannels, fisrtpoint, secondpoint, 1.0f, 1.0f));
    EXPECT_EQ(0, matte(srcFCThreeChannels, srcFCThreeChannels, fisrtpoint, secondpoint, 1.0f, 1.0f));
}

TEST(photoeffects_matte, invalid_image_format)
{
    Mat src(10, 10, CV_8UC1);
    Mat dst;
    Point firstpoint(0, 0);
    Point secondpoint(10, 10);
    EXPECT_ERROR(CV_StsAssert, matte(src, dst, firstpoint, secondpoint, 1.0f, 1.0f));
}

TEST(photoeffects_matte, regression)
{
    string input = "./testdata/matte_test.png";
    string expectedOutput = "./testdata/matte_test_result.png";
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
    EXPECT_EQ(0, matte(src, dst, Point(100, 100), Point(300, 300), 50.0f, 50.0f));
    dst.convertTo(dst, CV_8UC3, 255);
    Mat diff = abs(expectedDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
