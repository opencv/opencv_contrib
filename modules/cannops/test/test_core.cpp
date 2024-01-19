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

TEST(CORE, RESIZE_NEW)
{
    Mat resized_cv, checker;
    Mat cpuMat = randomMat(1280, 1706, CV_8UC3, 100.0, 255.0);
    Size dsize = Size(768, 832);
    // add support for {0 INTER_NEAREST} and {1 INTER_LINEAR}
    // only the resize result of INTER_LINEAR is close to CV's.
    int interpolation = 1;
    cv::resize(cpuMat, resized_cv, dsize, 0, 0, interpolation);
    cv::cann::resize(cpuMat, checker, dsize, 0, 0, interpolation);
    EXPECT_MAT_NEAR(resized_cv, checker, 1);

    cv::resize(cpuMat, resized_cv, Size(), 0.5, 0.5, interpolation);
    cv::cann::resize(cpuMat, checker, Size(), 0.5, 0.5, interpolation);
    EXPECT_MAT_NEAR(resized_cv, checker, 1);

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    cv::resize(cpuMat, resized_cv, dsize, 0, 0, interpolation);
    cv::cann::resize(npuMat, npuChecker, dsize, 0, 0, interpolation);
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(resized_cv, checker, 1);

    cv::resize(cpuMat, resized_cv, Size(), 0.5, 0.5, interpolation);
    cv::cann::resize(npuMat, npuChecker, Size(), 0.5, 0.5, interpolation);
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(resized_cv, checker, 1);
}

TEST(CORE, CROP_RESIZE)
{
    Mat cpuMat = randomMat(1280, 1706, CV_8UC1, 100.0, 255.0);
    Mat resized_cv, checker, cpuOpRet;
    Size dsize = Size(496, 512);
    const Rect b(300, 500, 224, 256);

    cv::cann::cropResize(cpuMat, checker, b, dsize, 0, 0, 1);
    Mat cropped_cv(cpuMat, b);
    cv::resize(cropped_cv, cpuOpRet, dsize, 0, 0, 1);
    EXPECT_MAT_NEAR(checker, cpuOpRet, 1);

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    cv::cann::cropResize(npuMat, npuChecker, b, dsize, 0, 0, 1);
    npuChecker.download(checker);
    EXPECT_MAT_NEAR(cpuOpRet, checker, 1);
}
TEST(CORE, CROP_RESIZE_MAKE_BORDER)
{
    Mat cpuMat = randomMat(1024, 896, CV_8UC1, 100.0, 255.0);

    Mat resized_cv, checker, cpuOpRet;
    Size dsize = Size(320, 256);
    const Rect b(300, 500, 496, 512);
    RNG rng(12345);
    float scalarV[3] = {0, 0, 255};
    int top, bottom, left, right;
    top = 54;
    bottom = 0;
    left = 32;
    right = 0;
    int interpolation = 1;

    Scalar value = {scalarV[0], scalarV[1], scalarV[2], 0};
    for (int borderType = 0; borderType < 2; borderType++)
    {
        cv::cann::cropResizeMakeBorder(cpuMat, checker, b, dsize, 0, 0, interpolation, top, left,
                                       borderType, value);
        Mat cropped_cv(cpuMat, b);
        cv::resize(cropped_cv, resized_cv, dsize, 0, 0, interpolation);
        cv::copyMakeBorder(resized_cv, cpuOpRet, top, bottom, left, right, borderType, value);
        EXPECT_MAT_NEAR(checker, cpuOpRet, 1e-10);
    }
    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    for (int borderType = 0; borderType < 2; borderType++)
    {
        cv::cann::cropResizeMakeBorder(npuMat, npuChecker, b, dsize, 0, 0, interpolation, top, left,
                                       borderType, value);
        npuChecker.download(checker);
        Mat cropped_cv(cpuMat, b);
        cv::resize(cropped_cv, resized_cv, dsize, 0, 0, interpolation);
        cv::copyMakeBorder(resized_cv, cpuOpRet, top, bottom, left, right, borderType, value);
        EXPECT_MAT_NEAR(checker, cpuOpRet, 1e-10);
    }
}

TEST(CORE, COPY_MAKE_BORDER)
{
    Mat cpuMat = randomMat(1280, 1706, CV_8UC3, 100, 255);

    Mat cpuOpRet, checker;
    RNG rng(12345);
    Scalar value = {static_cast<double>(rng.uniform(0, 255)),
                    static_cast<double>(rng.uniform(0, 255)),
                    static_cast<double>(rng.uniform(0, 255))};
    int top, bottom, left, right;
    top = 20;
    bottom = 30;
    left = 30;
    right = 20;

    int borderType = 0;
    for (borderType = 0; borderType < 2; borderType++)
    {
        cv::cann::copyMakeBorder(cpuMat, checker, top, bottom, left, right, borderType, value);

        cv::copyMakeBorder(cpuMat, cpuOpRet, top, bottom, left, right, borderType, value);
        EXPECT_MAT_NEAR(checker, cpuOpRet, 1e-10);
    }

    AscendMat npuMat, npuChecker;
    npuMat.upload(cpuMat);
    for (borderType = 0; borderType < 2; borderType++)
    {
        cv::cann::copyMakeBorder(npuMat, npuChecker, top, bottom, left, right, borderType, value);
        npuChecker.download(checker);

        cv::copyMakeBorder(cpuMat, cpuOpRet, top, bottom, left, right, borderType, value);
        EXPECT_MAT_NEAR(checker, cpuOpRet, 1e-10);
    }
}

} // namespace
} // namespace opencv_test
