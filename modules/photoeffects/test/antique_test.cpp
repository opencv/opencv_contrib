#include "test_precomp.hpp"

using namespace cv;
using namespace cv::photoeffects;

using namespace std;

TEST(photoeffects_antique, test)
{
    Mat srcUCThreeChannels(10, 10, CV_8UC3);
    srcUCThreeChannels = Mat::zeros(10, 10, CV_8UC3);
    Mat textureUCThreeChannels(10, 10, CV_8UC3);
    textureUCThreeChannels = Mat::zeros(10, 10, CV_8UC3);
    Mat dst;
    EXPECT_ERROR(CV_StsAssert, antique(srcUCThreeChannels, dst, textureUCThreeChannels, -0.5f));
    Mat srcFCThreeChannels(10, 10, CV_32FC3);
    srcFCThreeChannels = Mat::zeros(10, 10, CV_32FC3);
    Mat textureFCThreeChannels(10, 10, CV_32FC3);
    textureFCThreeChannels = Mat::zeros(10, 10, CV_32FC3);
    EXPECT_ERROR(CV_StsAssert, antique(srcFCThreeChannels, dst, textureFCThreeChannels, -0.5f));
}

TEST(photoeffects_antique, invalid_image_format)
{
    Mat src(10, 10, CV_8UC1);
    Mat textureNormal(10, 10, CV_8UC3);
    Mat dst;
    EXPECT_ERROR(CV_StsAssert, antique(src, dst, textureNormal, 0.5f));
    Mat srcNormal(10, 10, CV_8UC3);
    EXPECT_ERROR(CV_StsAssert, antique(srcNormal, dst, textureNormal, 0.0f));
    Mat texture(10, 10, CV_8UC1);
    EXPECT_ERROR(CV_StsAssert, antique(srcNormal, dst, texture, 0.4f));
}

TEST(photoeffects_antique, regression)
{
    string input = cvtest::TS::ptr()->get_data_path() + "photoeffects/antique_test.png";
    string texture = cvtest::TS::ptr()->get_data_path() + "photoeffects/antique_texture_test.png";
    string expectedOut = cvtest::TS::ptr()->get_data_path() + "photoeffects/antique_test_result.png";
    Mat src = imread(input, CV_LOAD_IMAGE_COLOR);
    if (src.empty())
    {
        FAIL() << "Can't read " + input + " image";
    }
    Mat txtre = imread(texture, CV_LOAD_IMAGE_COLOR);
    if (txtre.empty())
    {
        FAIL() << "Can't read " + texture + " image";
    }
    Mat expectedDst = imread(expectedOut, CV_LOAD_IMAGE_COLOR);
    if (expectedDst.empty())
    {
        FAIL() << "Can't read" + expectedOut + " image";
    }
    Mat dst;
    antique(src, dst, txtre, 0.9f);
    Mat diff = abs(expectedDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}
