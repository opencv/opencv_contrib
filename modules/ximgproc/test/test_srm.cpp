// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<std::string, int, double> SRMParams;

class SRMTest : public ::testing::TestWithParam<SRMParams>
{
};

// Use ::testing:: scope for generators
INSTANTIATE_TEST_CASE_P(Set1,
                        SRMTest,
                        ::testing::Combine(::testing::Values("cv/ximgproc/image.png"),
                                           ::testing::Values(16, 32, 64),
                                           ::testing::Values(0.5, 1.0)));
TEST_P(SRMTest, BGROutputSizeAndType)
{
    string img_path = cvtest::findDataFile(get<0>(GetParam()));
    int Q = get<1>(GetParam());
    double sigma = get<2>(GetParam());

    Mat img = imread(img_path);
    ASSERT_FALSE(img.empty()) << "Could not load: " << img_path;

    Mat result;
    ximgproc::segmentation::SRMSegmentation(img, result, Q, sigma);

    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), img.size());
    EXPECT_EQ(result.type(), CV_32F);
}

TEST_P(SRMTest, GrayscaleOutputSizeAndType)
{
    string img_path = cvtest::findDataFile(get<0>(GetParam()));
    int Q = get<1>(GetParam());
    double sigma = get<2>(GetParam());

    Mat img, gray;
    img = imread(img_path);
    ASSERT_FALSE(img.empty()) << "Could not load: " << img_path;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    Mat result;
    ximgproc::segmentation::SRMSegmentation(gray, result, Q, sigma);

    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), gray.size());
    EXPECT_EQ(result.type(), CV_32F);
}

// Edge cases

TEST(SRMEdgeCases, UniformImageDoesNotCrash)
{
    Mat uniform(50, 50, CV_8UC1, Scalar(128));
    Mat result;
    ASSERT_NO_THROW(ximgproc::segmentation::SRMSegmentation(uniform, result));
    EXPECT_FALSE(result.empty());
}

TEST(SRMEdgeCases, SinglePixelDoesNotCrash)
{
    Mat single(1, 1, CV_8UC1, Scalar(200));
    Mat result;
    ASSERT_NO_THROW(ximgproc::segmentation::SRMSegmentation(single, result));
    EXPECT_FALSE(result.empty());
}

TEST(SRMEdgeCases, ZeroQIsClampedSafely)
{
    Mat img(50, 50, CV_8UC1, Scalar(128));
    Mat result;
    ASSERT_NO_THROW(ximgproc::segmentation::SRMSegmentation(img, result, 0));
    EXPECT_FALSE(result.empty());
}

TEST(SRMEdgeCases, ZeroSigmaIsClampedSafely)
{
    Mat img(50, 50, CV_8UC1, Scalar(128));
    Mat result;
    ASSERT_NO_THROW(ximgproc::segmentation::SRMSegmentation(img, result, 32, 0.0));
    EXPECT_FALSE(result.empty());
}

}}  // namespace opencv_test