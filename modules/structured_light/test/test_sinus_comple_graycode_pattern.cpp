/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this
 license.
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
 // Redistribution and use in source and binary forms, with or without
 modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright
 notice,
 //     this list of conditions and the following disclaimer in the
 documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote
 products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is"
 and
 // any express or implied warranties, including, but not limited to, the
 implied
 // warranties of merchantability and fitness for a particular purpose are
 disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any
 direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/
#include "test_precomp.hpp"

#include <opencv2/structured_light/sinus_comple_graycode_pattern.hpp>

namespace opencv_test {
namespace {

const string STRUCTURED_LIGHT_DIR = "structured_light";
const string CALIBRATION_PATH =
    "data/sinus_complementary_graycode/test_decode/cali.yml";
const string LEFT_CAM_FOLDER_DATA =
    "data/sinus_complementary_graycode/test_decode/left";
const string RIGHT_CAM_FOLDER_DATA =
    "data/sinus_complementary_graycode/test_decode/right";
const string TEST_GENERATE_FOLDER_DATA =
    "data/sinus_complementary_graycode/test_generate";
const string TEST_UNWRAP_FOLDER_DATA =
    "data/sinus_complementary_graycode/test_unwrap";

class SinusoidalComplementaryGrayCodeFixSuit : public testing::Test {
  public:
    void SetUp() override {
        params.height = 720;
        params.width = 1280;
        params.horizontal = false;
        params.nbrOfPeriods = 32;
        params.shiftTime = 4;
        params.confidenceThreshold = 5;

        pattern =
            cv::structured_light::SinusCompleGrayCodePattern::create(params);
    }

    void TearDown() override {}

    cv::structured_light::SinusCompleGrayCodePattern::Params params;
    cv::Ptr<cv::structured_light::SinusCompleGrayCodePattern> pattern;
};

TEST_F(SinusoidalComplementaryGrayCodeFixSuit, test_generate) {
    std::vector<cv::Mat> imgs;
    pattern->generate(imgs);

    EXPECT_EQ(static_cast<int>(imgs.size()), 10);

    for (int i = 0; i < 10; ++i) {
        cv::Mat groundTruth = cv::imread(cvtest::TS::ptr()->get_data_path() +
                                             "/" + STRUCTURED_LIGHT_DIR + "/" +
                                             TEST_GENERATE_FOLDER_DATA + "/im" +
                                             std::to_string(i) + ".bmp",
                                         0);
        double error = cv::norm(groundTruth, imgs[i], cv::NormTypes::NORM_L1) /
                       (params.width * params.height);
        EXPECT_LE(error, 1.0);
    }
}

TEST_F(SinusoidalComplementaryGrayCodeFixSuit, test_unwrapPhaseMap) {
    std::vector<cv::Mat> imgs;
    for (int i = 0; i < 10; ++i) {
        cv::Mat tempImg = cv::imread(cvtest::TS::ptr()->get_data_path() + "/" +
                                         STRUCTURED_LIGHT_DIR + "/" +
                                         TEST_GENERATE_FOLDER_DATA + "/im" +
                                         std::to_string(i) + ".bmp",
                                     0);
        imgs.push_back(tempImg);
    }

    cv::Mat confidenceMap, wrappedPhaseMap, floorMap, unwrappedPhaseMap;
    pattern->computeConfidenceMap(imgs, confidenceMap);
    pattern->computePhaseMap(imgs, wrappedPhaseMap);
    pattern->computeFloorMap(imgs, confidenceMap, wrappedPhaseMap, floorMap);
    pattern->unwrapPhaseMap(wrappedPhaseMap, floorMap, unwrappedPhaseMap,
                            confidenceMap > params.confidenceThreshold);

    cv::Mat groundTruth = cv::imread(
        cvtest::TS::ptr()->get_data_path() + "/" + STRUCTURED_LIGHT_DIR + "/" +
            TEST_UNWRAP_FOLDER_DATA + "/unwrap.tiff",
        cv::IMREAD_UNCHANGED);

    EXPECT_LE(abs(unwrappedPhaseMap.ptr<float>(120)[120] - groundTruth.ptr<float>(120)[120]), 0.1f);
}

TEST_F(SinusoidalComplementaryGrayCodeFixSuit, test_decode) {
    cv::structured_light::SinusCompleGrayCodePattern::Params tempParams;
    tempParams.height = 720;
    tempParams.width = 1280;
    tempParams.horizontal = false;
    tempParams.nbrOfPeriods = 64;
    tempParams.shiftTime = 12;
    tempParams.confidenceThreshold = 20;
    tempParams.minDisparity = -500;
    tempParams.maxDisparity = 500;
    cv::Ptr<cv::structured_light::SinusCompleGrayCodePattern> tempPattern =
        cv::structured_light::SinusCompleGrayCodePattern::create(tempParams);

    cv::FileStorage caliReader(cvtest::TS::ptr()->get_data_path() + "/" +
                                   STRUCTURED_LIGHT_DIR + "/" +
                                   CALIBRATION_PATH,
                               cv::FileStorage::READ);

    cv::Mat M1, M2, D1, D2, R, T, R1, R2, P1, P2, Q, S;
    caliReader["M1"] >> M1;
    caliReader["M2"] >> M2;
    caliReader["D1"] >> D1;
    caliReader["D2"] >> D2;
    caliReader["R"] >> R;
    caliReader["T"] >> T;
    caliReader["R1"] >> R1;
    caliReader["R2"] >> R2;
    caliReader["P1"] >> P1;
    caliReader["P2"] >> P2;
    caliReader["Q"] >> Q;
    caliReader["S"] >> S;

    cv::Size imgSize(S.at<int>(0, 0), S.at<int>(0, 1));
    cv::Mat mapLX, mapLY, mapRX, mapRY;
    cv::initUndistortRectifyMap(M1, D1, R1, P1, imgSize, CV_32FC1, mapLX,
                                mapLY);
    cv::initUndistortRectifyMap(M2, D2, R2, P2, imgSize, CV_32FC1, mapRX,
                                mapRY);

    std::vector<std::vector<cv::Mat>> imgsOfCameras(2);
    for (int i = 1; i < 20; ++i) {
        cv::Mat tempImg = cv::imread(
            cvtest::TS::ptr()->get_data_path() + "/" + STRUCTURED_LIGHT_DIR +
                "/" + LEFT_CAM_FOLDER_DATA + "/im" + std::to_string(i) + ".bmp",
            0);
        cv::remap(tempImg, tempImg, mapLX, mapLY, cv::INTER_LINEAR);
        imgsOfCameras[0].push_back(tempImg);
        tempImg =
            cv::imread(cvtest::TS::ptr()->get_data_path() + "/" +
                           STRUCTURED_LIGHT_DIR + "/" + RIGHT_CAM_FOLDER_DATA +
                           "/im" + std::to_string(i) + ".bmp",
                       0);
        cv::remap(tempImg, tempImg, mapRX, mapRY, cv::INTER_LINEAR);
        imgsOfCameras[1].push_back(tempImg);
    }

    cv::Mat disparityMap;
    tempPattern->decode(
        imgsOfCameras, disparityMap, cv::noArray(), cv::noArray(),
        cv::structured_light::SINUSOIDAL_COMPLEMENTARY_GRAY_CODE);

    EXPECT_LE(abs(disparityMap.ptr<float>(500)[740] - 219.f), 1.f);
}

} // namespace
} // namespace opencv_test
