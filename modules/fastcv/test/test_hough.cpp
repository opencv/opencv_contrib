/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<std::string /* file name */, double /* threshold */ > HoughLinesTestParams;
class HoughLinesTest : public ::testing::TestWithParam<HoughLinesTestParams> {};

TEST_P(HoughLinesTest, accuracy)
{
    auto p = GetParam();
    std::string fname = std::get<0>(p);
    double threshold  = std::get<1>(p);

    cv::Mat src = imread(cvtest::findDataFile(fname), cv::IMREAD_GRAYSCALE);

    // make it aligned by 8
    cv::Mat withBorder;
    int bpix = ((src.cols & 0xfffffff8) + 8) - src.cols;
    cv::copyMakeBorder(src, withBorder, 0, 0, 0, bpix, BORDER_REFLECT101);
    src = withBorder;

    cv::Mat contoured;
    cv::Canny(src, contoured, 100, 200);

    std::vector<cv::Vec4f> lines;
    cv::fastcv::houghLines(contoured, lines, threshold);

    std::vector<cv::Vec4f> refLines;
    double rho = 1.0, theta = 1.0 * CV_PI / 180.0;
    // cloned since image may be modified by the function
    cv::HoughLinesP(contoured.clone(), refLines, rho, theta, threshold);

    for (const cv::Vec4f& l : lines)
    {
        cv::Point2f from(l[0], l[1]), to(l[2], l[3]);
        EXPECT_GE(from.x, 0);
        EXPECT_GE(from.y, 0);
        EXPECT_LE(from.x, src.cols);
        EXPECT_LE(from.y, src.rows);
        EXPECT_GE(to.x, 0);
        EXPECT_GE(to.y, 0);
        EXPECT_LE(to.x, src.cols);
        EXPECT_LE(to.y, src.rows);
    }

    auto makeDistTrans = [src](const std::vector<Vec4f>& ls) -> cv::Mat
    {
        Mat lineMap(src.size(), CV_8U, Scalar(255));
        for (const cv::Vec4f& l : ls)
        {
            cv::Point from(l[0], l[1]), to(l[2], l[3]);
            cv::line(lineMap, from, to, Scalar::all(0));
        }
        Mat distTrans(src.size(), CV_8U);
        cv::distanceTransform(lineMap, distTrans, DIST_L2, DIST_MASK_PRECISE);
        return distTrans;
    };

    cv::Mat distTrans = makeDistTrans(lines);
    cv::Mat refDistTrans = makeDistTrans(refLines);

    double normInf = cvtest::norm(refDistTrans, distTrans, cv::NORM_INF);
    double normL2  = cvtest::norm(refDistTrans, distTrans, cv::NORM_L2)  / src.size().area();

    EXPECT_LT(normInf, 120.0);
    EXPECT_LT(normL2, 0.0361);

    if (cvtest::debugLevel > 0)
    {
        cv::Mat draw;
        cvtColor(src, draw, COLOR_GRAY2BGR);
        cv::Mat refDraw = draw.clone();

        for (const cv::Vec4f& l : lines)
        {
            cv::Point from(l[0], l[1]), to(l[2], l[3]);
            cv::line(draw, from, to, Scalar(0, 255, 0));
        }
        size_t idx = fname.find_last_of("/\\");
        std::string fout = fname.substr(idx+1, fname.length() - idx - 5);
        cv::imwrite(cv::format("line_%s_t%5f_fcv.png", fout.c_str(), threshold), draw);

        for (const cv::Vec4f& l : refLines)
        {
            cv::Point from(l[0], l[1]), to(l[2], l[3]);
            cv::line(refDraw, from, to, Scalar(0, 255, 0));
        }
        cv::imwrite(cv::format("line_%s_t%5f_ref.png", fout.c_str(), threshold), refDraw);
    }
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, HoughLinesTest,
                        ::testing::Combine(::testing::Values("cv/shared/pic5.png",
                                                             "stitching/a1.png",
                                                             "cv/shared/pic5.png",
                                                             "cv/shared/pic1.png"), // images
                                           ::testing::Values(0.05, 0.25, 0.5, 0.75) // threshold
                                           ));


typedef std::tuple<std::string /* file name */, uint32_t /* minDist */,   uint32_t /* cannyThreshold */,
                   uint32_t /* accThreshold */, uint32_t /* minRadius */, uint32_t /* maxRadius */> HoughCirclesTestParams;
class HoughCirclesTest : public ::testing::TestWithParam<HoughCirclesTestParams> {};

TEST_P(HoughCirclesTest, accuracy)
{
    auto p = GetParam();
    std::string fname       = std::get<0>(p);
    uint32_t minDist        = std::get<1>(p);
    uint32_t cannyThreshold = std::get<2>(p);
    uint32_t accThreshold   = std::get<3>(p);
    uint32_t minRadius      = std::get<4>(p);
    uint32_t maxRadius      = std::get<5>(p);

    cv::Mat src = imread(cvtest::findDataFile(fname), cv::IMREAD_GRAYSCALE);
    // make it aligned by 8
    cv::Mat withBorder;
    int bpix = ((src.cols & 0xfffffff8) + 8) - src.cols;
    cv::copyMakeBorder(src, withBorder, 0, 0, 0, bpix, BORDER_REFLECT101);
    src = withBorder;

    std::vector<cv::Vec3f> refCircles;
    cv::HoughCircles(src, refCircles, HOUGH_GRADIENT, 1.5, minDist,
                     cannyThreshold, accThreshold,
                     minRadius, maxRadius);

    Mat icircles;
    cv::fastcv::houghCircles(src, icircles, minDist,
                             cannyThreshold, accThreshold,
                             minRadius, maxRadius);

    std::vector<cv::Vec3f> circles;
    icircles.convertTo(circles, CV_32FC3);

    // usually the number of detected circles is small, brute force is OK
    float totalDist = 0;
    for (size_t i = 0; i < circles.size(); i++)
    {
        cv::Vec3f c = circles[i];
        float dist = std::numeric_limits<float>::max();
        for (size_t j = 0; j < refCircles.size(); j++)
        {
            cv::Vec3f rc = refCircles[i];
            float d = (rc - c).ddot(rc - c);
            if (d < dist)
            {
                dist = d;
            }
        }
        totalDist += dist;
    }
    totalDist = std::sqrt(totalDist);

    EXPECT_LT(totalDist, 811.0);

    if (cvtest::debugLevel > 0)
    {
        cv::Mat draw;
        cvtColor(src, draw, COLOR_GRAY2BGR);
        cv::Mat refDraw = draw.clone();
        for (const cv::Vec3f& c : refCircles)
        {
            cv::Point center(c[0], c[1]);
            cv::circle(refDraw, center, c[2], Scalar(0, 255, 0));
        }
        for (const cv::Vec3f& c : circles)
        {
            cv::Point center(c[0], c[1]);
            cv::circle(draw, center, c[2], Scalar(0, 255, 0));
        }
        std::cout << "circles: " << circles.size() << std::endl;
        size_t idx = fname.find_last_of("/\\");
        std::string fout = fname.substr(idx+1, fname.length() - idx - 5);
        cv::imwrite(cv::format("circle_%s_mdt%d_can%d_acc%d_rf%d_rt%d_ref.png", fout.c_str(),
                               minDist, cannyThreshold, accThreshold, minRadius, maxRadius), refDraw);
        cv::imwrite(cv::format("circle_%s_mdt%d_can%d_acc%d_rf%d_rt%d_fcv.png", fout.c_str(),
                               minDist, cannyThreshold, accThreshold, minRadius, maxRadius), draw);
    }
}

// NOTE: test files should be manually loaded to folder on a device, for example like this:
// adb push fastcv/misc/hough/ /sdcard/testdata/fastcv/hough/
INSTANTIATE_TEST_CASE_P(FastCV_Extension, HoughCirclesTest,
                        ::testing::Values(
                            // gpu/connectedcomponents/concentric_circles.png
                            HoughCirclesTestParams {"cv/cameracalibration/circles/circles4.png", 100, 100, 50, 10, 100 },
                            HoughCirclesTestParams {"cv/cameracalibration/circles/circles4.png", 100, 100, 50, 30, 100 },
                            HoughCirclesTestParams {"cv/cameracalibration/circles/circles4.png", 100, 100, 50, 50, 100 },
                            HoughCirclesTestParams {"cv/cameracalibration/circles/circles4.png",  10, 100, 50, 10, 100 },
                            HoughCirclesTestParams {"cv/cameracalibration/circles/circles4.png",  10, 100, 50, 30, 100 },
                            HoughCirclesTestParams {"cv/cameracalibration/circles/circles4.png",  10, 100, 50, 50, 100 }
                         ));

}} // namespaces opencv_test, ::
