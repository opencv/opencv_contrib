/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "test_precomp.hpp"

#include <cmath>

namespace opencv_test { namespace {

using namespace cv;
using namespace cv::structured_light;

static size_t requiredBits(int value)
{
    return value > 1 ? static_cast<size_t>(std::ceil(std::log(static_cast<double>(value)) / std::log(2.0))) : 1;
}

static void circularShiftCols(const Mat& src, Mat& dst, int shift)
{
    CV_Assert(src.channels() == 1);
    if (src.empty())
    {
        dst.release();
        return;
    }

    const int width = src.cols;
    if (width == 0)
    {
        dst.release();
        return;
    }

    shift %= width;
    if (shift < 0)
        shift += width;

    if (shift == 0)
    {
        src.copyTo(dst);
        return;
    }

    dst.create(src.size(), src.type());
    src.colRange(0, width - shift).copyTo(dst.colRange(shift, width));
    src.colRange(width - shift, width).copyTo(dst.colRange(0, shift));
}

static void applyLowContrastColumns(std::vector<Mat>& images, int colStart, int colEnd, int value)
{
    if (images.empty())
        return;

    const int width = images[0].cols;
    colStart = std::max(0, std::min(colStart, width));
    colEnd = std::max(0, std::min(colEnd, width));
    if (colStart >= colEnd)
        return;

    for (Mat& img : images)
    {
        img.colRange(colStart, colEnd).setTo(Scalar(value));
    }
}

static void buildSyntheticCapture(const Ptr<GrayCodePattern>& graycode,
                                  int columnShift,
                                  int lowContrastStart,
                                  int lowContrastEnd,
                                  std::vector<std::vector<Mat>>& captured,
                                  std::vector<Mat>& blackImages,
                                  std::vector<Mat>& whiteImages)
{
    std::vector<Mat> basePattern;
    ASSERT_TRUE(graycode->generate(basePattern));

    const size_t numImages = basePattern.size();
    captured.assign(2, std::vector<Mat>(numImages));
    for (size_t idx = 0; idx < numImages; ++idx)
    {
        basePattern[idx].copyTo(captured[0][idx]);
        circularShiftCols(basePattern[idx], captured[1][idx], columnShift);
    }

    applyLowContrastColumns(captured[0], lowContrastStart, lowContrastEnd, 128);
    applyLowContrastColumns(captured[1], lowContrastStart, lowContrastEnd, 128);

    const int height = basePattern[0].rows;
    const int width = basePattern[0].cols;
    blackImages.assign(2, Mat(height, width, basePattern[0].type(), Scalar(0)));
    whiteImages.assign(2, Mat(height, width, basePattern[0].type(), Scalar(255)));
}

static void convertTo16U(std::vector<std::vector<Mat>>& captured,
                         std::vector<Mat>& blackImages,
                         std::vector<Mat>& whiteImages)
{
    const double scale = 257.0; // 255 * 257 = 65535
    for (std::vector<Mat>& cameraImages : captured)
    {
        for (Mat& img : cameraImages)
        {
            Mat tmp;
            img.convertTo(tmp, CV_16U, scale);
            img = tmp;
        }
    }
    for (Mat& img : blackImages)
    {
        Mat tmp;
        img.convertTo(tmp, CV_16U, scale);
        img = tmp;
    }
    for (Mat& img : whiteImages)
    {
        Mat tmp;
        img.convertTo(tmp, CV_16U, scale);
        img = tmp;
    }
}

static Mat computeLegacyDisparity(const Ptr<GrayCodePattern>& graycode,
                                  const std::vector<std::vector<Mat>>& captured)
{
    CV_Assert(captured.size() == 2);
    CV_Assert(!captured[0].empty());

    const int projWidth = captured[0][0].cols;
    const int projHeight = captured[0][0].rows;

    std::vector<Mat> sumX(2), counts(2);
    for (int k = 0; k < 2; ++k)
    {
        sumX[k] = Mat::zeros(projHeight, projWidth, CV_64F);
        counts[k] = Mat::zeros(projHeight, projWidth, CV_32S);
    }

    Mat projectorCoordinateMap(projHeight, projWidth, CV_32SC2, Vec2i(-1, -1));

    for (int cam = 0; cam < 2; ++cam)
    {
        for (int y = 0; y < projHeight; ++y)
        {
            for (int x = 0; x < projWidth; ++x)
            {
                Point projPix;
                bool error = graycode->getProjPixel(captured[cam], x, y, projPix);
                if (!error && projPix.x >= 0 && projPix.x < projWidth && projPix.y >= 0 && projPix.y < projHeight)
                {
                    sumX[cam].at<double>(projPix.y, projPix.x) += x;
                    counts[cam].at<int>(projPix.y, projPix.x) += 1;
                    if (cam == 0)
                    {
                        projectorCoordinateMap.at<Vec2i>(y, x) = Vec2i(projPix.x, projPix.y);
                    }
                }
            }
        }
    }

    Mat avgX[2];
    for (int cam = 0; cam < 2; ++cam)
    {
        Mat cnt64;
        counts[cam].convertTo(cnt64, CV_64F);
        Mat zeroMask = cnt64 == 0;
        cnt64.setTo(1.0, zeroMask);
        cv::divide(sumX[cam], cnt64, avgX[cam]);
        avgX[cam].setTo(0.0, zeroMask);
    }

    Mat projectorDisparity = avgX[1] - avgX[0];
    Mat invalidMask = (counts[0] == 0) | (counts[1] == 0);
    projectorDisparity.setTo(0.0, invalidMask);

    Mat expected(projHeight, projWidth, CV_64F, Scalar(0));
    for (int y = 0; y < projHeight; ++y)
    {
        for (int x = 0; x < projWidth; ++x)
        {
            const Vec2i proj = projectorCoordinateMap.at<Vec2i>(y, x);
            if (proj[0] >= 0)
            {
                expected.at<double>(y, x) = projectorDisparity.at<double>(proj[1], proj[0]);
            }
        }
    }

    return expected;
}

/****************************************************************************************\
*                              Pattern generation test                                   *
\****************************************************************************************/
class CV_GrayPatternEncodingTest : public cvtest::BaseTest
{
public:
    void run(int) CV_OVERRIDE
    {
        GrayCodePattern::Params params;
        params.width = 8;
        params.height = 6;
        Ptr<GrayCodePattern> graycode = GrayCodePattern::create(params);

        std::vector<Mat> pattern;
        ASSERT_TRUE(graycode->generate(pattern));

        const size_t numColImgs = requiredBits(params.width);
        const size_t numRowImgs = requiredBits(params.height);
        EXPECT_EQ(pattern.size(), 2 * (numColImgs + numRowImgs));

        Mat diff;
        for (size_t bit = 0; bit < numColImgs; ++bit)
        {
            size_t idx1 = 2 * numColImgs - 2 * bit - 2;
            size_t idx2 = idx1 + 1;
            for (int col = 0; col < params.width; ++col)
            {
                uchar expected = static_cast<uchar>((((col >> bit) & 1) ^ ((col >> (bit + 1)) & 1)) * 255);
                cv::compare(pattern[idx1].col(col), Scalar(expected), diff, CMP_NE);
                EXPECT_EQ(0, countNonZero(diff));
                cv::compare(pattern[idx2].col(col), Scalar(255 - expected), diff, CMP_NE);
                EXPECT_EQ(0, countNonZero(diff));
            }
        }

        const size_t baseIdx = 2 * numColImgs;
        for (size_t bit = 0; bit < numRowImgs; ++bit)
        {
            size_t idx1 = baseIdx + 2 * numRowImgs - 2 * bit - 2;
            size_t idx2 = idx1 + 1;
            for (int row = 0; row < params.height; ++row)
            {
                uchar expected = static_cast<uchar>((((row >> bit) & 1) ^ ((row >> (bit + 1)) & 1)) * 255);
                cv::compare(pattern[idx1].row(row), Scalar(expected), diff, CMP_NE);
                EXPECT_EQ(0, countNonZero(diff));
                cv::compare(pattern[idx2].row(row), Scalar(255 - expected), diff, CMP_NE);
                EXPECT_EQ(0, countNonZero(diff));
            }
        }
    }
};

/****************************************************************************************\
*                        Fast decode parity (8-bit) test                                 *
\****************************************************************************************/
class CV_GrayPatternFastDecode8uTest : public cvtest::BaseTest
{
public:
    void run(int) CV_OVERRIDE
    {
        GrayCodePattern::Params params;
        params.width = 16;
        params.height = 8;
        Ptr<GrayCodePattern> graycode = GrayCodePattern::create(params);

        std::vector<std::vector<Mat>> captured;
        std::vector<Mat> blackImages, whiteImages;
        buildSyntheticCapture(graycode, 3, 2, 5, captured, blackImages, whiteImages);

        Mat disparity;
        ASSERT_TRUE(graycode->decode(captured, disparity, blackImages, whiteImages, DECODE_3D_UNDERWORLD));
        ASSERT_FALSE(disparity.empty());

        Mat expected = computeLegacyDisparity(graycode, captured);
        EXPECT_EQ(disparity.type(), CV_64F);
        EXPECT_EQ(expected.type(), CV_64F);
        EXPECT_EQ(disparity.size(), expected.size());
        double err = cv::norm(disparity, expected, NORM_INF);
        EXPECT_LT(err, 1e-9);
    }
};

/****************************************************************************************\
*                        Fast decode parity (16-bit) test                                *
\****************************************************************************************/
class CV_GrayPatternFastDecode16uTest : public cvtest::BaseTest
{
public:
    void run(int) CV_OVERRIDE
    {
        GrayCodePattern::Params params;
        params.width = 16;
        params.height = 8;
        Ptr<GrayCodePattern> graycode = GrayCodePattern::create(params);

        std::vector<std::vector<Mat>> captured;
        std::vector<Mat> blackImages, whiteImages;
        buildSyntheticCapture(graycode, 4, 1, 4, captured, blackImages, whiteImages);
        convertTo16U(captured, blackImages, whiteImages);

        Mat disparity;
        ASSERT_TRUE(graycode->decode(captured, disparity, blackImages, whiteImages, DECODE_3D_UNDERWORLD));
        ASSERT_FALSE(disparity.empty());

        Mat expected = computeLegacyDisparity(graycode, captured);
        double err = cv::norm(disparity, expected, NORM_INF);
        EXPECT_LT(err, 1e-9);
    }
};

/****************************************************************************************\
*                                Test registration                                      *
\****************************************************************************************/

TEST(GrayCodePattern, generates_gray_bitplanes)
{
    CV_GrayPatternEncodingTest test;
    test.safe_run();
}

TEST(GrayCodePattern, fast_decode_matches_legacy_8u)
{
    CV_GrayPatternFastDecode8uTest test;
    test.safe_run();
}

TEST(GrayCodePattern, fast_decode_matches_legacy_16u)
{
    CV_GrayPatternFastDecode16uTest test;
    test.safe_run();
}

TEST(GrayCodePattern, shadow_mask_respects_16u_thresholds)
{
    GrayCodePattern::Params params;
    params.width = 16;
    params.height = 8;
    Ptr<GrayCodePattern> graycode = GrayCodePattern::create(params);

    std::vector<std::vector<Mat>> captured;
    std::vector<Mat> unusedBlack, unusedWhite;
    buildSyntheticCapture(graycode, 2, 0, 0, captured, unusedBlack, unusedWhite);

    std::vector<Mat> blackImages(2), whiteImages(2);
    for (int cam = 0; cam < 2; ++cam)
    {
        blackImages[cam] = Mat(params.height, params.width, CV_8U, Scalar(100));
        whiteImages[cam] = Mat(params.height, params.width, CV_8U, Scalar(140)); // 40 levels above black
    }

    convertTo16U(captured, blackImages, whiteImages);

    graycode->setWhiteThreshold(static_cast<size_t>(10 * 257));

    graycode->setBlackThreshold(static_cast<size_t>(50 * 257));
    Mat disparityMasked;
    ASSERT_TRUE(graycode->decode(captured, disparityMasked, blackImages, whiteImages, DECODE_3D_UNDERWORLD));
    EXPECT_EQ(CV_64F, disparityMasked.type());
    EXPECT_EQ(0, countNonZero(disparityMasked)) << "Scaled threshold should mark all pixels as shadowed";

    graycode->setBlackThreshold(50); // forgetting the scale makes the mask pass
    Mat disparityUnmasked;
    ASSERT_TRUE(graycode->decode(captured, disparityUnmasked, blackImages, whiteImages, DECODE_3D_UNDERWORLD));
    EXPECT_GT(countNonZero(disparityUnmasked), 0);
}

}} // namespace
