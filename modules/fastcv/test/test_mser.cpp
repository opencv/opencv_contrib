/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {
// we use such nested structure to combine test values
typedef std::tuple< std::tuple<bool /* useBboxes */, bool /* useContourData */>,
                    int  /* numNeighbors */, std::string /*file path*/> MSERTestParams;
class MSERTest : public ::testing::TestWithParam<MSERTestParams> {};

// compare results to OpenCV's MSER detector
// by comparing resulting contours
TEST_P(MSERTest, accuracy)
{
    auto p = GetParam();
    bool useBboxes      = std::get<0>(std::get<0>(p));
    bool useContourData = std::get<1>(std::get<0>(p));
    int  numNeighbors   =             std::get<1>(p); // 4 or 8
    std::string imgPath =             std::get<2>(p);

    cv::Mat src = imread(cvtest::findDataFile(imgPath), cv::IMREAD_GRAYSCALE);

    uint32_t delta = 2;
    uint32_t minArea = 256;
    uint32_t maxArea = (int)src.total()/4;
    float        maxVariation = 0.15f;
    float        minDiversity = 0.2f;

    std::vector<std::vector<Point>> contours;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::fastcv::FCVMSER::ContourData> contourData;
    cv::Ptr<cv::fastcv::FCVMSER> mser;
    mser = cv::fastcv::FCVMSER::create(src.size(), numNeighbors, delta, minArea, maxArea,
                                    maxVariation, minDiversity);
    if (useBboxes)
    {
        if (useContourData)
        {
            mser->detect(src, contours, bboxes, contourData);
        }
        else
        {
            mser->detect(src, contours, bboxes);
        }
    }
    else
    {
        mser->detect(src, contours);
    }

    Rect imgRect(0, 0, src.cols, src.rows);
    if (useBboxes)
    {
        ASSERT_EQ(contours.size(), bboxes.size());
        for (size_t i = 0; i < contours.size(); i++)
        {
            ASSERT_TRUE(imgRect.contains(bboxes[i].tl()));
            ASSERT_TRUE(imgRect.contains(bboxes[i].br()));

            for (size_t j = 0; j < contours[i].size(); j++)
            {
                ASSERT_TRUE(bboxes[i].contains(contours[i][j]));
            }
        }
    }

    if (useContourData)
    {
        ASSERT_EQ(contours.size(), contourData.size());
        for (size_t i = 0; i < contours.size(); i++)
        {
            int polarity = contourData[i].polarity;
            EXPECT_TRUE(polarity == -1 || polarity == 1);
        }
    }

    // compare each pair of contours using dist transform of their points
    // find pair of contours by similar moments
    typedef cv::Matx<double, 10, 1> MomentVec;

    auto calcEstimate = [](const std::vector<std::vector<Point>>& ctrs, Size srcSize) -> std::vector<std::pair<Mat, MomentVec>>
    {
        std::vector<std::pair<Mat, MomentVec>> res;
        for (size_t i = 0; i < ctrs.size(); i++)
        {
            const std::vector<Point>& contour = ctrs[i];
            Mat ptsMap(srcSize, CV_8U, Scalar(255));
            for(size_t j = 0; j < contour.size(); ++j)
            {
                ptsMap.at<uchar>(contour[j].y, contour[j].x) = 0;
            }
            Mat distTrans(srcSize, CV_8U);
            cv::distanceTransform(ptsMap, distTrans, DIST_L2, DIST_MASK_PRECISE);

            cv::Moments m = cv::moments(contour);
            double invRows = 1.0 / srcSize.height,       invCols = 1.0 / srcSize.width;
            double invRows2 = invRows  / srcSize.height, invCols2 = invCols  / srcSize.width;
            double invRows3 = invRows2 / srcSize.height, invCols3 = invCols2 / srcSize.width;
            MomentVec mx  = { m.m00, m.m10 * invCols, m.m01 * invRows,
                              m.m20 * invCols2, m.m11 * invCols * invRows, m.m02 * invRows2,
                              m.m30 * invCols3,
                              m.m21 * invCols2 * invRows,
                              m.m12 * invCols * invRows2,
                              m.m03 * invRows3};
            res.push_back({distTrans, mx});
        }

        return res;
    };

    std::vector<std::pair<Mat, MomentVec>> contourEstimate = calcEstimate(contours, src.size());

    std::vector<std::vector<Point>> ocvContours;
    std::vector<cv::Rect> ocvBboxes;

    cv::Ptr<MSER> ocvMser = cv::MSER::create(delta, minArea, maxArea, maxVariation, minDiversity);
    ocvMser->detectRegions(src, ocvContours, ocvBboxes);

    std::vector<std::pair<Mat, MomentVec>> ocvContourEstimate = calcEstimate(ocvContours, src.size());

    // brute force match by moments comparison
    double overallL2Sqr = 0;
    int nInliers = 0;
    for (size_t i = 0; i < contourEstimate.size(); i++)
    {
        double minDist = std::numeric_limits<double>::max();
        size_t minIdx = -1;
        for (size_t j = 0; j < ocvContourEstimate.size(); j++)
        {
            double d = cv::norm(contourEstimate[i].second - ocvContourEstimate[j].second);
            if (d < minDist)
            {
                minDist = d; minIdx = j;
            }
        }
        // compare dist transforms of contours
        Mat ref = ocvContourEstimate[minIdx].first;
        Mat fcv = contourEstimate[i].first;
        double normL2Sqr  = cvtest::norm(ref, fcv, cv::NORM_L2SQR);
        double normInf    = cvtest::norm(ref, fcv, cv::NORM_INF);
        normL2Sqr = normL2Sqr / src.size().area();

        if (cvtest::debugLevel > 0)
        {
            Mat draw(src.rows, src.cols*2, CV_8U);
            ref.copyTo(draw(Range::all(), Range(0, src.cols)));
            fcv.copyTo(draw(Range::all(), Range(src.cols, src.cols*2)));
            cv::putText(draw, cv::format("dM: %f L2^2: %f Inf: %f",minDist, normL2Sqr, normInf), Point(0, src.rows),
                        cv::FONT_HERSHEY_COMPLEX, 1, Scalar::all(128));
            cv::imwrite(cv::format("dist_n%d_c%03d_r%03d.png", numNeighbors, (int)i, (int)minIdx), draw);
        }

        if (normInf < 50.0)
        {
            overallL2Sqr += normL2Sqr;
            nInliers++;
        }
    }

    double overallL2 = std::sqrt(overallL2Sqr);
    EXPECT_LT(std::sqrt(overallL2), 11.45);
    double ratioInliers = double(nInliers) / contourEstimate.size();
    EXPECT_GT(ratioInliers, 0.363);
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, MSERTest,
    ::testing::Combine(::testing::Values( // useBboxes useContourData
                                         std::tuple<bool, bool> { true, false},
                                         std::tuple<bool, bool> {false, false},
                                         std::tuple<bool, bool> { true,  true}),
                       ::testing::Values(4, 8), // numNeighbors
                       ::testing::Values("cv/shared/baboon.png", "cv/mser/puzzle.png")
                      )
    );
}} // namespaces opencv_test, ::
