#include "test_precomp.hpp"

using namespace cv;
using namespace cv::photoeffects;

using namespace std;

TEST(photoeffects_filmGrain, invalid_image_format)
{
    Mat src(10, 20, CV_8UC2);
    Mat dst;

    EXPECT_ERROR(CV_StsAssert, filmGrain(src,dst,5));
}

TEST(photoeffects_filmGrain, test) {
    Mat imageWithOneChannel(10, 20, CV_8UC1);
    Mat imageWithThreeChannel(10, 20, CV_8UC3);
    Mat dst;
    EXPECT_EQ(0, filmGrain(imageWithOneChannel, dst, 5));
    EXPECT_EQ(0, filmGrain(imageWithThreeChannel, dst, 5));
}

TEST(photoeffects_filmGrain, regression)
{
    string input = cvtest::TS::ptr()->get_data_path() + "photoeffects/filmGrain_test.png";
    string expectedOutput = cvtest::TS::ptr()->get_data_path() + "photoeffects/filmGrain_test_result.png";

    Mat image = imread(input, CV_LOAD_IMAGE_COLOR);
    Mat rightDst = imread(expectedOutput, CV_LOAD_IMAGE_COLOR);

    if (image.empty())
        FAIL() << "Can't read " + input + " image";
    if (rightDst.empty())
        FAIL() << "Can't read " + input + " image";

    Mat dst;
    theRNG()=RNG(0);
    EXPECT_EQ(0, filmGrain(image, dst, 25));

    Mat diff = abs(rightDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
