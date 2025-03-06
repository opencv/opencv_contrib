/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

typedef std::tuple<bool /*useFloat*/, int /*nLevels*/, bool /*scaleBy2*/> PyramidTestParams;
class PyramidTest : public ::testing::TestWithParam<PyramidTestParams> { };

TEST_P(PyramidTest, accuracy)
{
    auto par = GetParam();

    bool useFloat = std::get<0>(par);
    int  nLevels  = std::get<1>(par);
    bool scaleBy2 = std::get<2>(par);

    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    if (useFloat)
    {
        cv::Mat f;
        src.convertTo(f, CV_32F);
        src = f;
    }

    std::vector<cv::Mat> pyr;
    cv::fastcv::buildPyramid(src, pyr, nLevels, scaleBy2);

    ASSERT_EQ(pyr.size(), (size_t)nLevels);

    std::vector<cv::Mat> refPyr;
    if (scaleBy2)
    {
        cv::buildPyramid(src, refPyr, nLevels - 1);
    }
    else // ORB downscaling
    {
        for (int i = 0; i < nLevels; i++)
        {
            // we don't know how exactly the bit-accurate size is calculated
            cv::Mat level;
            cv::resize(src, level, pyr[i].size(), 0, 0, cv::INTER_AREA);
            refPyr.push_back(level);
        }
    }

    for (int i = 0; i < nLevels; i++)
    {
        cv::Mat ref = refPyr[i];
        cv::Mat m = pyr[i];
        ASSERT_EQ(m.size(), ref.size());
        double l2diff   = cv::norm(m, ref, cv::NORM_L2);
        double linfdiff = cv::norm(m, ref, cv::NORM_INF);

        double l2Thresh   = scaleBy2 ? 178.0 : 5216.0;
        double linfThresh = scaleBy2 ?  16.0 :  116.0;
        EXPECT_LE(l2diff,   l2Thresh);
        EXPECT_LE(linfdiff, linfThresh);
    }

    if (cvtest::debugLevel > 0)
    {
        for (int i = 0; i < nLevels; i++)
        {
            char tchar = useFloat ? 'f' : 'i';
            std::string scaleStr = scaleBy2 ? "x2" : "xORB";
            cv::imwrite(cv::format("pyr_diff_%c_%d_%s_l%d.png", tchar, nLevels, scaleStr.c_str(), i), cv::abs(pyr[i] - refPyr[i]));
        }
    }
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, PyramidTest,
                        // useFloat, nLevels, scaleBy2
                        ::testing::Values(
                            PyramidTestParams { true, 2,  true}, PyramidTestParams { true, 3,  true}, PyramidTestParams { true, 4,  true},
                            PyramidTestParams {false, 2,  true}, PyramidTestParams {false, 3,  true}, PyramidTestParams {false, 4,  true},
                            PyramidTestParams {false, 2, false}, PyramidTestParams {false, 3, false}, PyramidTestParams {false, 4, false}
                            ));

typedef std::tuple<MatType, size_t> SobelPyramidTestParams;
class SobelPyramidTest : public ::testing::TestWithParam<SobelPyramidTestParams> {};

TEST_P(SobelPyramidTest, accuracy)
{
    auto p = GetParam();
    int    type    = std::get<0>(p);
    size_t nLevels = std::get<1>(p);

    // NOTE: test files should be manually loaded to folder on a device, for example like this:
    // adb push fastcv/misc/bilateral_recursive/ /sdcard/testdata/fastcv/bilateral/
    cv::Mat src = imread(cvtest::findDataFile("cv/shared/baboon.png"), cv::IMREAD_GRAYSCALE);

    std::vector<cv::Mat> pyr;
    cv::fastcv::buildPyramid(src, pyr, nLevels);

    std::vector<cv::Mat> pyrDx, pyrDy;
    cv::fastcv::sobelPyramid(pyr, pyrDx, pyrDy, type);

    ASSERT_EQ(pyrDx.size(), nLevels);
    ASSERT_EQ(pyrDy.size(), nLevels);

    for (size_t i = 0; i < nLevels; i++)
    {
        ASSERT_EQ(pyrDx[i].type(), type);
        ASSERT_EQ(pyrDx[i].size(), pyr[i].size());
        ASSERT_EQ(pyrDy[i].type(), type);
        ASSERT_EQ(pyrDy[i].size(), pyr[i].size());
    }

    std::vector<cv::Mat> refPyrDx(nLevels), refPyrDy(nLevels);
    for (size_t i = 0; i < nLevels; i++)
    {
        int stype = (type == CV_8S) ? CV_16S : type;
        cv::Mat dx, dy;
        cv::Sobel(pyr[i], dx, stype, 1, 0);
        cv::Sobel(pyr[i], dy, stype, 0, 1);
        dx.convertTo(refPyrDx[i], type, 1.0/8.0, 0.0);
        dy.convertTo(refPyrDy[i], type, 1.0/8.0, 0.0);
    }

    for (size_t i = 0; i < nLevels; i++)
    {
        cv::Mat ref, dst;
        double normInf, normL2;
        cv::Rect roi(1, 1, pyr[i].cols - 2, pyr[i].rows - 2);
        ref = refPyrDx[i](roi);
        dst = pyrDx[i](roi);
        normInf = cvtest::norm(dst, ref, cv::NORM_INF);
        normL2  = cvtest::norm(dst, ref, cv::NORM_L2) / dst.total();

        EXPECT_LE(normInf, 76.1);
        EXPECT_LT(normL2,   0.4);

        ref = refPyrDy[i](roi);
        dst = pyrDy[i](roi);
        normInf = cvtest::norm(dst, ref, cv::NORM_INF);
        normL2  = cvtest::norm(dst, ref, cv::NORM_L2) / dst.total();

        EXPECT_LE(normInf, 66.6);
        EXPECT_LT(normL2,   0.4);
    }

    if (cvtest::debugLevel > 0)
    {
        std::map<int, std::string> typeToString =
        {
            {CV_8U,   "8u"}, {CV_8S,   "8s"}, {CV_16U, "16u"}, {CV_16S, "16s"},
            {CV_32S, "32s"}, {CV_32F, "32f"}, {CV_64F, "64f"}, {CV_16F, "16f"},
        };

        for (size_t i = 0; i < nLevels; i++)
        {
            cv::imwrite(cv::format("pyr_l%zu.png", i), pyr[i]);
            cv::imwrite(cv::format("pyr_sobel_x_t%s_l%zu.png", typeToString.at(type).c_str(), i), pyrDx[i] + 128);
            cv::imwrite(cv::format("pyr_sobel_y_t%s_l%zu.png", typeToString.at(type).c_str(), i), pyrDy[i] + 128);

            cv::imwrite(cv::format("ref_pyr_sobel_x_t%s_l%zu.png", typeToString.at(type).c_str(), i), refPyrDx[i] + 128);
            cv::imwrite(cv::format("ref_pyr_sobel_y_t%s_l%zu.png", typeToString.at(type).c_str(), i), refPyrDy[i] + 128);
        }
    }
}

INSTANTIATE_TEST_CASE_P(FastCV_Extension, SobelPyramidTest, ::testing::Combine(
    ::testing::Values(CV_8S, CV_16S, CV_32F), // depth
    ::testing::Values(3, 6))); // nLevels


}} // namespaces opencv_test, ::
