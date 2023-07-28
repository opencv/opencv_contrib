// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <vector>

namespace opencv_test
{
namespace
{
TEST(CORE, MERGE)
{
    Mat m1 = (Mat_<uchar>(2, 2) << 1, 4, 7, 10);
    Mat m2 = (Mat_<uchar>(2, 2) << 2, 5, 8, 11);
    Mat m3 = (Mat_<uchar>(2, 2) << 3, 6, 9, 12);
    Mat channels[3] = {m1, m2, m3};
    Mat m;
    cv::merge(channels, 3, m);

    cv::cann::setDevice(0);

    AscendMat a1, a2, a3;
    a1.upload(m1);
    a2.upload(m2);
    a3.upload(m3);
    AscendMat aclChannels[3] = {a1, a2, a3};
    std::vector<AscendMat> aclChannelsVector;
    aclChannelsVector.push_back(a1);
    aclChannelsVector.push_back(a2);
    aclChannelsVector.push_back(a3);

    Mat checker1, checker2;
    cv::cann::merge(aclChannels, 3, checker1);
    cv::cann::merge(aclChannelsVector, checker2);

    EXPECT_MAT_NEAR(m, checker1, 0.0);
    EXPECT_MAT_NEAR(m, checker2, 0.0);

    cv::cann::resetDevice();
}

TEST(CORE, SPLIT)
{
    char d[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Mat m(2, 2, CV_8UC3, d);
    Mat channels[3];
    cv::split(m, channels);

    cv::cann::setDevice(0);

    AscendMat aclChannels[3];
    std::vector<AscendMat> aclChannelsVector;

    cv::cann::split(m, aclChannels);
    cv::cann::split(m, aclChannelsVector);

    Mat checker1[3], checker2[3];
    aclChannels[0].download(checker1[0]);
    aclChannels[1].download(checker1[1]);
    aclChannels[2].download(checker1[2]);

    aclChannelsVector[0].download(checker2[0]);
    aclChannelsVector[1].download(checker2[1]);
    aclChannelsVector[2].download(checker2[2]);

    EXPECT_MAT_NEAR(channels[0], checker1[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker1[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker1[2], 0.0);

    EXPECT_MAT_NEAR(channels[0], checker2[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker2[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker2[2], 0.0);

    AscendMat npuM;
    npuM.upload(m);
    cv::cann::split(npuM, aclChannels);
    cv::cann::split(npuM, aclChannelsVector);

    aclChannels[0].download(checker1[0]);
    aclChannels[1].download(checker1[1]);
    aclChannels[2].download(checker1[2]);

    aclChannelsVector[0].download(checker2[0]);
    aclChannelsVector[1].download(checker2[1]);
    aclChannelsVector[2].download(checker2[2]);

    EXPECT_MAT_NEAR(channels[0], checker1[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker1[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker1[2], 0.0);

    EXPECT_MAT_NEAR(channels[0], checker2[0], 0.0);
    EXPECT_MAT_NEAR(channels[1], checker2[1], 0.0);
    EXPECT_MAT_NEAR(channels[2], checker2[2], 0.0);
    cv::cann::resetDevice();
}

TEST(CORE, TRANSPOSE)
{
    Mat cpuMat = randomMat(10, 10, CV_32SC3), cpuRetMat, checker;
    cv::transpose(cpuMat, cpuRetMat);
    cv::cann::transpose(cpuMat, checker);
    EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    cv::cann::transpose(npuMat, npuChecker);
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
}

TEST(CORE, FLIP)
{
    Mat cpuMat = randomMat(10, 10, CV_32SC3), cpuRetMat, checker;

    int flipMode;

    for (flipMode = -1; flipMode < 2; flipMode++)
    {
        cv::flip(cpuMat, cpuRetMat, flipMode);
        cv::cann::flip(cpuMat, checker, flipMode);
        EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
    }

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    for (flipMode = -1; flipMode < 2; flipMode++)
    {
        cv::flip(cpuMat, cpuRetMat, flipMode);
        cv::cann::flip(npuMat, npuChecker, flipMode);
        npuChecker.download(checker);
        EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
    }
}

TEST(CORE, ROTATE)
{
    Mat cpuRetMat, checker, cpuMat = randomMat(3, 5, CV_16S, 0.0, 255.0);

    int rotateMode;
    for (rotateMode = 0; rotateMode < 3; rotateMode++)
    {
        cv::rotate(cpuMat, cpuRetMat, rotateMode);
        cv::cann::rotate(cpuMat, checker, rotateMode);
        EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
    }

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    for (rotateMode = 0; rotateMode < 3; rotateMode++)
    {
        cv::rotate(cpuMat, cpuRetMat, rotateMode);
        cv::cann::rotate(npuMat, npuChecker, rotateMode);
        npuChecker.download(checker);
        EXPECT_MAT_NEAR(cpuRetMat, checker, 0.0);
    }
}

TEST(CORE, CROP)
{
    Mat cpuOpRet, checker, cpuMat = randomMat(6, 6, CV_32SC3, 0.0, 255.0);
    Rect b(1, 2, 4, 4);
    Mat cropped_cv(cpuMat, b);
    AscendMat cropped_cann(cpuMat, b);
    cropped_cann.download(checker);
    EXPECT_MAT_NEAR(cropped_cv, checker, 1e-10);
}

TEST(CORE, CROP_OVERLOAD)
{
    Mat cpuOpRet, checker, cpuMat = randomMat(6, 6, CV_16SC3, 0.0, 255.0);
    const Rect b(1, 2, 4, 4);
    Mat cropped_cv = cpuMat(b);
    AscendMat cropped_cann = cv::cann::crop(cpuMat, b);
    cropped_cann.download(checker);
    EXPECT_MAT_NEAR(cropped_cv, checker, 1e-10);

    AscendMat npuMat;
    npuMat.upload(cpuMat);
    cropped_cann = cv::cann::crop(npuMat, b);
    cropped_cann.download(checker);
    EXPECT_MAT_NEAR(cropped_cv, checker, 1e-10);
}

TEST(CORE, RESIZE)
{
    Mat resized_cv, checker, cpuMat = randomMat(10, 10, CV_32F, 100.0, 255.0);
    Size dsize = Size(6, 6);
    // only support {2 INTER_CUBIC} and {3 INTER_AREA}
    // only the resize result of INTER_AREA is close to CV's.
    int flags = 3;
    cv::cann::setDevice(0);
    cv::resize(cpuMat, resized_cv, dsize, 0, 0, flags);
    cv::cann::resize(cpuMat, checker, dsize, 0, 0, flags);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-4);

    cv::resize(cpuMat, resized_cv, Size(), 0.5, 0.5, flags);
    cv::cann::resize(cpuMat, checker, Size(), 0.5, 0.5, flags);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-4);

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    cv::resize(cpuMat, resized_cv, dsize, 0, 0, flags);
    cv::cann::resize(npuMat, npuChecker, dsize, 0, 0, flags);
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-4);

    cv::resize(cpuMat, resized_cv, Size(), 0.5, 0.5, flags);
    cv::cann::resize(npuMat, npuChecker, Size(), 0.5, 0.5, flags);
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(resized_cv, checker, 1e-4);
    cv::cann::resetDevice();
}


} // namespace
} // namespace opencv_test
