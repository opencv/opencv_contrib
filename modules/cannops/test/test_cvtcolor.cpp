// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test
{
namespace
{

void cvtColorTest(int code, int cn, int dcn = 3, float diff = 0.0f)
{
    cv::cann::setDevice(DEVICE_ID);
    Mat cpuRet, npuRet;

    Mat img8U = randomMat(512, 512, CV_MAKETYPE(CV_8U, cn), 0.0f, 255.0f);
    Mat img16U = randomMat(512, 512, CV_MAKETYPE(CV_16U, cn), 0.0f, 65535.0f);
    Mat img32F = randomMat(512, 512, CV_MAKETYPE(CV_32F, cn), 0.0f, 65535.0f);

    cv::cvtColor(img8U, cpuRet, code, dcn);
    cv::cann::cvtColor(img8U, npuRet, code, dcn);
    EXPECT_MAT_NEAR(cpuRet, npuRet, diff);

    cv::cvtColor(img16U, cpuRet, code, dcn);
    cv::cann::cvtColor(img16U, npuRet, code, dcn);
    EXPECT_MAT_NEAR(cpuRet, npuRet, diff);

    cv::cvtColor(img32F, cpuRet, code, dcn);
    cv::cann::cvtColor(img32F, npuRet, code, dcn);
    EXPECT_MAT_NEAR(cpuRet, npuRet, diff);
    cv::cann::resetDevice();
}

TEST(CVT_COLOR, BGR2BGRA) { cvtColorTest(COLOR_BGR2BGRA, 3, 4); }
TEST(CVT_COLOR, BGRA2BGR) { cvtColorTest(COLOR_BGRA2BGR, 4); }
TEST(CVT_COLOR, BGR2RGBA) { cvtColorTest(COLOR_BGR2RGBA, 3, 4); }
TEST(CVT_COLOR, RGBA2BGR) { cvtColorTest(COLOR_RGBA2BGR, 4); }
TEST(CVT_COLOR, BGR2RGB) { cvtColorTest(COLOR_BGR2RGB, 3); }
TEST(CVT_COLOR, BGRA2RGBA) { cvtColorTest(COLOR_BGRA2RGBA, 4, 4); }

// Due to parameter accuracy issues, the calculation results have certain accuracy differences.
TEST(CVT_COLOR, BGR2GRAY) { cvtColorTest(COLOR_BGR2GRAY, 3, 1, 10.0f); }
TEST(CVT_COLOR, RGB2GRAY) { cvtColorTest(COLOR_BGR2GRAY, 3, 1, 10.0f); }
TEST(CVT_COLOR, GRAY2BGR) { cvtColorTest(COLOR_GRAY2BGR, 1); }
TEST(CVT_COLOR, GRAY2BGRA) { cvtColorTest(COLOR_GRAY2BGRA, 1, 4); }
TEST(CVT_COLOR, BGRA2GRAY) { cvtColorTest(COLOR_BGRA2GRAY, 4, 1, 10.0f); }
TEST(CVT_COLOR, RGBA2GRAY) { cvtColorTest(COLOR_RGBA2GRAY, 4, 1, 10.0f); }

TEST(CVT_COLOR, RGB2XYZ) { cvtColorTest(COLOR_RGB2XYZ, 3, 3, 50.0f); }
TEST(CVT_COLOR, BGR2XYZ) { cvtColorTest(COLOR_BGR2XYZ, 3, 3, 50.0f); }
TEST(CVT_COLOR, XYZ2BGR) { cvtColorTest(COLOR_XYZ2BGR, 3, 3, 150.0f); }
TEST(CVT_COLOR, XYZ2RGB) { cvtColorTest(COLOR_XYZ2RGB, 3, 3, 150.0f); }
TEST(CVT_COLOR, XYZ2BGR_DC4) { cvtColorTest(COLOR_XYZ2BGR, 3, 4, 150.0f); }
TEST(CVT_COLOR, XYZ2RGB_DC4) { cvtColorTest(COLOR_XYZ2RGB, 3, 4, 150.0f); }

TEST(CVT_COLOR, BGR2YCrCb) { cvtColorTest(COLOR_BGR2YCrCb, 3, 3, 10.0f); }
TEST(CVT_COLOR, RGB2YCrCb) { cvtColorTest(COLOR_RGB2YCrCb, 3, 3, 10.0f); }
TEST(CVT_COLOR, YCrCb2BGR) { cvtColorTest(COLOR_YCrCb2BGR, 3, 3, 10.0f); }
TEST(CVT_COLOR, YCrCb2RGB) { cvtColorTest(COLOR_YCrCb2RGB, 3, 3, 10.0f); }
TEST(CVT_COLOR, YCrCb2BGR_DC4) { cvtColorTest(COLOR_YCrCb2BGR, 3, 4, 10.0f); }
TEST(CVT_COLOR, YCrCb2RGB_DC4) { cvtColorTest(COLOR_YCrCb2RGB, 3, 4, 10.0f); }

TEST(CVT_COLOR, BGR2YUV) { cvtColorTest(COLOR_BGR2YUV, 3, 3, 10.0f); }
TEST(CVT_COLOR, RGB2YUV) { cvtColorTest(COLOR_RGB2YUV, 3, 3, 10.0f); }
TEST(CVT_COLOR, YUV2BGR) { cvtColorTest(COLOR_YUV2BGR, 3, 3, 10.0f); }
TEST(CVT_COLOR, YUV2RGB) { cvtColorTest(COLOR_YUV2RGB, 3, 3, 10.0f); }
TEST(CVT_COLOR, YUV2BGR_DC4) { cvtColorTest(COLOR_YUV2BGR, 3, 4, 10.0f); }
TEST(CVT_COLOR, YUV2RGB_DC4) { cvtColorTest(COLOR_YUV2RGB, 3, 4, 10.0f); }

// Test of AscendMat. Since the logic is the same, only interface test is needed.
TEST(CVT_COLOR, COLOR_BGR2BGRA_ASCENDMAT)
{
    cv::cann::setDevice(DEVICE_ID);
    Mat cpuRet, npuRet;

    Mat img8U = randomMat(512, 512, CV_8UC3, 0.0f, 255.0f);
    cv::cvtColor(img8U, cpuRet, COLOR_BGR2BGRA, 4);

    AscendMat npuImg8U, npuChecker;
    npuImg8U.upload(img8U);
    cv::cann::cvtColor(npuImg8U, npuChecker, COLOR_BGR2BGRA, 4);
    npuChecker.download(npuRet);
    EXPECT_MAT_NEAR(cpuRet, npuRet, 10.0f);
    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
