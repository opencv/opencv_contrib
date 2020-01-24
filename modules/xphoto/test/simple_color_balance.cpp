// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

    TEST(xphoto_simplecolorbalance, uchar_max_value)
    {
        const uchar oldMax = 120, newMax = 255;

        Mat test = Mat::zeros(3,3,CV_8UC1);
        test.at<uchar>(0, 0) = oldMax;
        test.at<uchar>(0, 1) = oldMax / 2;
        test.at<uchar>(0, 2) = oldMax / 4;

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(0);
        wb->setInputMax(oldMax);
        wb->setOutputMin(0);
        wb->setOutputMax(newMax);

        wb->balanceWhite(test, test);

        double minDst, maxDst;
        cv::minMaxIdx(test, &minDst, &maxDst);

        ASSERT_NEAR(maxDst, newMax, 1e-4);
    }

    TEST(xphoto_simplecolorbalance, uchar_min_value)
    {
        const uchar oldMin = 120, newMin = 0;

        Mat test = Mat::zeros(1,3,CV_8UC1);
        test.at<uchar>(0, 0) = oldMin;
        test.at<uchar>(0, 1) = (256 + oldMin) / 2;
        test.at<uchar>(0, 2) = 255;

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(oldMin);
        wb->setInputMax(255);
        wb->setOutputMin(newMin);
        wb->setOutputMax(255);

        wb->balanceWhite(test, test);

        double minDst, maxDst;
        cv::minMaxIdx(test, &minDst, &maxDst);

        ASSERT_NEAR(minDst, newMin, 1e-4);
    }

    TEST(xphoto_simplecolorbalance, uchar_equal_range)
    {
        const int N = 4;
        uchar data[N] = {0, 1, 16, 255};
        Mat test = Mat(1, N, CV_8UC1, data);
        Mat result = Mat(1, N, CV_8UC1, data);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(0);
        wb->setInputMax(255);
        wb->setOutputMin(0);
        wb->setOutputMax(255);

        wb->balanceWhite(test, test);

        double err;
        cv::minMaxIdx(cv::abs(test - result), NULL, &err);
        ASSERT_LE(err, 1e-4);
    }

    TEST(xphoto_simplecolorbalance, uchar_single_value)
    {
        const int N = 4;
        uchar data0[N] = {51, 51, 51, 51};
        uchar data1[N] = {33, 33, 33, 33};
        Mat test = Mat(1, N, CV_8UC1, data0);
        Mat result = Mat(1, N, CV_8UC1, data1);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(51);
        wb->setInputMax(51);
        wb->setOutputMin(33);
        wb->setOutputMax(200);

        wb->balanceWhite(test, test);

        double err;
        cv::minMaxIdx(cv::abs(test - result), NULL, &err);
        ASSERT_LE(err, 1e-4);
    }

    TEST(xphoto_simplecolorbalance, uchar_p)
    {
        const int N = 5;
        uchar data0[N] = {10, 55, 102, 188, 233};
        uchar data1[N] = {0, 1, 90, 254, 255};
        Mat test = Mat(1, N, CV_8UC1, data0);
        Mat result = Mat(1, N, CV_8UC1, data1);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(10);
        wb->setInputMax(233);
        wb->setOutputMin(0);
        wb->setOutputMax(255);
        wb->setP(21);

        wb->balanceWhite(test, test);

        double err;
        cv::minMaxIdx(cv::abs(test - result), NULL, &err);
        ASSERT_LE(err, 1e-4);
    }

    TEST(xphoto_simplecolorbalance, uchar_c3)
    {
        const int N = 15;
        uchar data0[N] = {10, 55, 102, 55, 102, 188, 102, 188, 233, 188, 233, 10, 233, 10, 55};
        uchar data1[N] = {0, 1, 90, 1, 90, 254, 90, 254, 255, 254, 255, 0, 255, 0, 1};
        Mat test = Mat(1, N / 3, CV_8UC3, data0);
        Mat result = Mat(1, N / 3, CV_8UC3, data1);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(10);
        wb->setInputMax(233);
        wb->setOutputMin(0);
        wb->setOutputMax(255);
        wb->setP(21);

        wb->balanceWhite(test, test);

        double err;
        cv::minMaxIdx(cv::abs(test - result), NULL, &err);
        ASSERT_LE(err, 1e-4);
    }

    TEST(xphoto_simplecolorbalance, float_max_value)
    {
        const float oldMax = 24000.f, newMax = 65536.f;

        Mat test = Mat::zeros(3,3,CV_32FC1);
        test.at<float>(0, 0) = oldMax;
        test.at<float>(0, 1) = oldMax / 2;
        test.at<float>(0, 2) = oldMax / 4;

        double minSrc, maxSrc;
        cv::minMaxIdx(test, &minSrc, &maxSrc);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin((float)minSrc);
        wb->setInputMax((float)maxSrc);
        wb->setOutputMin(0);
        wb->setOutputMax(newMax);

        wb->balanceWhite(test, test);

        double minDst, maxDst;
        cv::minMaxIdx(test, &minDst, &maxDst);

        ASSERT_NEAR(maxDst, newMax, newMax*1e-4);
    }

    TEST(xphoto_simplecolorbalance, float_min_value)
    {
        const float oldMin = 24000.f, newMin = 0.f;

        Mat test = Mat::zeros(1,3,CV_32FC1);
        test.at<float>(0, 0) = oldMin;
        test.at<float>(0, 1) = (65536.f + oldMin) / 2;
        test.at<float>(0, 2) = 65536.f;

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(oldMin);
        wb->setInputMax(65536.f);
        wb->setOutputMin(newMin);
        wb->setOutputMax(65536.f);

        wb->balanceWhite(test, test);

        double minDst, maxDst;
        cv::minMaxIdx(test, &minDst, &maxDst);

        ASSERT_NEAR(minDst, newMin, 65536*1e-4);
    }

    TEST(xphoto_simplecolorbalance, float_equal_range)
    {
        const int N = 5;
        float data[N] = {0.f, 1.f, 16.2f, 256.3f, 4096.f};
        Mat test = Mat(1, N, CV_32FC1, data);
        Mat result = Mat(1, N, CV_32FC1, data);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(0);
        wb->setInputMax(4096);
        wb->setOutputMin(0);
        wb->setOutputMax(4096);

        wb->balanceWhite(test, test);

        double err;
        cv::minMaxIdx(cv::abs(test - result), NULL, &err);
        ASSERT_LE(err, 1e-4);
    }

    TEST(xphoto_simplecolorbalance, float_single_value)
    {
        const int N = 4;
        float data0[N] = {24000.5f, 24000.5f, 24000.5f, 24000.5f};
        float data1[N] = {52000.25f, 52000.25f, 52000.25f, 52000.25f};
        Mat test = Mat(1, N, CV_32FC1, data0);
        Mat result = Mat(1, N, CV_32FC1, data1);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(24000.5f);
        wb->setInputMax(24000.5f);
        wb->setOutputMin(52000.25f);
        wb->setOutputMax(65536.f);

        wb->balanceWhite(test, test);

        double err;
        cv::minMaxIdx(cv::abs(test - result), NULL, &err);
        ASSERT_LE(err, 65536*1e-4);
    }

    TEST(xphoto_simplecolorbalance, float_p)
    {
        const int N = 5;
        float data0[N] = {16000.f, 20000.5f, 24000.f, 36000.5f, 48000.f};
        float data1[N] = {-16381.952f, 0.f, 16381.952f, 65536.f, 114685.952f};
        Mat test = Mat(1, N, CV_32FC1, data0);
        Mat result = Mat(1, N, CV_32FC1, data1);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(16000.f);
        wb->setInputMax(48000.f);
        wb->setOutputMin(0.f);
        wb->setOutputMax(65536.f);
        wb->setP(21);

        wb->balanceWhite(test, test);

        double err;
        cv::minMaxIdx(cv::abs(test - result), NULL, &err);
        ASSERT_LE(err, 65536*1e-4);
    }

    TEST(xphoto_simplecolorbalance, float_c3)
    {
        const int N = 15;
        float data0[N] = {16000.f, 20000.5f, 24000.f, 20000.5f, 24000.f, 36000.5f, 24000.f, 36000.5f, 48000.f, 36000.5f, 48000.f, 16000.f, 48000.f, 16000.f, 20000.5f};
        float data1[N] = {-16381.952f, 0.f, 16381.952f, 0.f, 16381.952f, 65536.f, 16381.952f, 65536.f, 114685.952f, 65536.f, 114685.952f, -16381.952f, 114685.952f, -16381.952f, 0.f};
        Mat test = Mat(1, N / 3, CV_32FC3, data0);
        Mat result = Mat(1, N / 3, CV_32FC3, data1);

        cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(16000.f);
        wb->setInputMax(48000.f);
        wb->setOutputMin(0.f);
        wb->setOutputMax(65536.f);
        wb->setP(21);

        wb->balanceWhite(test, test);

        double err;
        cv::minMaxIdx(cv::abs(test - result), NULL, &err);
        ASSERT_LE(err, 65536*1e-4);
    }

}} // namespace
