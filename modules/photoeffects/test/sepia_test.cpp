#include "precomp.hpp"

using namespace cv;

TEST(photoeffects_sepia, invalid_image_format)
{
    Mat src(10, 10, CV_8UC3);
    Mat dst;

    EXPECT_ERROR(CV_StsAssert, sepia(src, dst));
}


TEST(photoeffects_sepia, test)
{
    Mat src(10, 10, CV_8UC1, Scalar(0)), dst, hsvDst;
    vector<Mat> channels(3);

    EXPECT_EQ(0, sepia(src, dst));
    cvtColor(dst, hsvDst, CV_BGR2HSV);
    split(hsvDst, channels);
    EXPECT_LE(19 - 1, channels[0].at<uchar>(0, 0)); // hue = 19
    EXPECT_GE(19 + 1, channels[0].at<uchar>(0, 0));
    EXPECT_LE(78 - 1, channels[1].at<uchar>(0, 0)); // saturation = 78
    EXPECT_GE(78 + 1, channels[1].at<uchar>(0, 0));
    EXPECT_LE(src.at<uchar>(0, 0) + 20 - 1, channels[2].at<uchar>(0, 0));
    EXPECT_GE(src.at<uchar>(0, 0) + 20 + 1, channels[2].at<uchar>(0, 0));
}

TEST(photoeffects_sepia, regression)
{
    string input = "./testdata/sepia_test.png";
    string expectedOutput = "./testdata/sepia_test_result.png";

    Mat image, rightDst;

    image = imread(input, CV_LOAD_IMAGE_GRAYSCALE);
    rightDst = imread(expectedOutput, CV_LOAD_IMAGE_COLOR);

    if (image.empty())
        FAIL() << "Can't read " + input + " image";
    if (rightDst.empty())
        FAIL() << "Can't read " + expectedOutput + " image";

    Mat dst;
    EXPECT_EQ(0, sepia(image, dst));

    Mat diff = abs(rightDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
