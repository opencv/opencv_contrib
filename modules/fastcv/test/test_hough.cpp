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

}} // namespaces opencv_test, ::
