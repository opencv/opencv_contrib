#include "precomp.hpp"

using namespace cv;

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

TEST(photoeffects_fadeColor, test) {
    Mat imageWithOneChannel(100, 200, CV_8UC1);
    Mat imageWithThreeChannel(100, 200, CV_8UC3);
    Mat dst;
    EXPECT_EQ(0, fadeColor(imageWithOneChannel, dst, Point(5,5), Point(5,8)));
    EXPECT_EQ(0, fadeColor(imageWithThreeChannel, dst, Point(5,5), Point(5,8)));
}

TEST(photoeffects_fadeColor, regression)
{
    string input ="./testdata/fadeColor_test.png";
    string expectedOutput ="./testdata/fadeColor_result.png";

    Mat image, dst, rightDst;
    image = imread(input, CV_LOAD_IMAGE_COLOR);
    rightDst = imread(expectedOutput, CV_LOAD_IMAGE_COLOR);

    if (image.empty())
        FAIL() << "Can't read " + input + " image";
    if (rightDst.empty())
        FAIL() << "Can't read " + expectedOutput + " image";

    EXPECT_EQ(0, fadeColor(image, dst, Point(100, 100), Point(250, 250)));

    Mat diff = abs(rightDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
