// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static Mat makeSyntheticImage(const Size& size)
{
    Mat img(size, CV_8UC1);
    for (int y = 0; y < img.rows; ++y)
    {
        uchar* row = img.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x)
        {
            const int v = (x * 255) / std::max(1, img.cols - 1);
            row[x] = static_cast<uchar>(v);
        }
    }
    return img;
}

TEST(ximgproc_ChanVese, createAndConfigure)
{
    Ptr<cv::ximgproc::segmentation::ChanVese> algo =  cv::ximgproc::segmentation::createChanVese();
    ASSERT_NE(algo, nullptr);

    algo->set_Lambda(1.5f);
    algo->set_mu(0.2f);
    algo->set_v(0.1f);
    algo->set_iterations(7);
    algo->set_dt(0.4f);

    Mat src = makeSyntheticImage(Size(32, 32));
    Mat dst;
    algo->ProcessImage(src, dst);

    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(dst.size(), src.size());
    EXPECT_EQ(dst.type(), CV_32F);
    EXPECT_EQ(dst.channels(), 1);
}

TEST(ximgproc_ChanVese, helperInitProducesValidOutput)
{
    Mat src = makeSyntheticImage(Size(48, 48));
    Mat dst;

    cv::ximgproc::segmentation::ChanVeseInit(src, dst, 1.0f, 0.0f, 0.3f, 10, 0.5f);

    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(dst.size(), src.size());
    EXPECT_EQ(dst.type(), CV_32F);
    EXPECT_EQ(dst.channels(), 1);
}

TEST(ximgproc_ChanVese, colorInputIsAccepted)
{
    Mat src(64, 64, CV_8UC3);
    cv::randu(src, Scalar(0, 0, 0), Scalar(255, 255, 255));

    Mat dst;
    cv::ximgproc::segmentation::ChanVeseInit(src, dst, 1.0f, 0.0f, 0.3f, 8, 0.5f);

    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(dst.size(), src.size());
    EXPECT_EQ(dst.type(), CV_32F);
    EXPECT_EQ(dst.channels(), 1);
}

TEST(ximgproc_ChanVese, outputIsBinaryForSyntheticInput)
{
    Mat src = makeSyntheticImage(Size(40, 40));
    Mat dst;
    cv::ximgproc::segmentation::ChanVeseInit(src, dst, 1.0f, 0.0f, 0.3f, 12, 0.5f);

    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(dst.size(), src.size());
    EXPECT_EQ(dst.type(), CV_32F);
    EXPECT_EQ(dst.channels(), 1);

    Mat valid = (dst == 0) | (dst == 255);
    EXPECT_EQ(cv::countNonZero(valid), static_cast<int>(dst.total()));
}

TEST(ximgproc_ChanVese, referenceImageCheck)
{
    Mat src = imread(cvtest::findDataFile("cv/ximgproc/chanvese_input.png"), IMREAD_GRAYSCALE);
    Mat expected = imread(cvtest::findDataFile("cv/ximgproc/chanvese_output.png"), IMREAD_GRAYSCALE);

    ASSERT_FALSE(src.empty());
    ASSERT_FALSE(expected.empty());

    Mat dst;
    cv::ximgproc::segmentation::ChanVeseInit(src, dst);

    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(dst.size(), src.size());
    EXPECT_EQ(dst.type(), CV_32F);
    EXPECT_EQ(dst.channels(), 1);

    Mat checker;
    dst.convertTo(checker, CV_8U);

    EXPECT_EQ(checker.size(), expected.size());
    // chanvese.cpp returns a binary image, so allow a small relative difference
    const int checkerCount = cv::countNonZero(checker);
    const int expectedCount = cv::countNonZero(expected);
    const int diff = std::abs(checkerCount - expectedCount);
    const double relDiff = expectedCount > 0
        ? static_cast<double>(diff) / expectedCount
        : (checkerCount == 0 ? 0.0 : 1.0);
    EXPECT_LE(relDiff, 0.01);
}

}} // namespace
