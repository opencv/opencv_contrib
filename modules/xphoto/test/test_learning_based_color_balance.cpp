#include "test_precomp.hpp"
using namespace cv;
namespace cvtest
{
TEST(xphoto_simplefeatures, regression)
{
    float acc_thresh = 0.01f;

    // Generate a test image:
    Mat test_im(1000, 1000, CV_8UC3);
    RNG rng(1234);
    rng.fill(test_im, RNG::NORMAL, Scalar(64, 100, 128), Scalar(10, 10, 10));
    threshold(test_im, test_im, 200.0, 255.0, THRESH_TRUNC);
    test_im.at<Vec3b>(0, 0) = Vec3b(240, 220, 200);

    // Which should have the following features:
    Vec2f ref1(128.0f / (64 + 100 + 128), 100.0f / (64 + 100 + 128));
    Vec2f ref2(200.0f / (240 + 220 + 200), 220.0f / (240 + 220 + 200));

    vector<Vec2f> dst_features;
    Ptr<xphoto::LearningBasedWB> wb = xphoto::createLearningBasedWB();
    wb->setRangeMaxVal(255);
    wb->setSaturationThreshold(0.98f);
    wb->setHistBinNum(64);
    wb->extractSimpleFeatures(test_im, dst_features);
    ASSERT_LE(cv::norm(dst_features[0], ref1, NORM_INF), acc_thresh);
    ASSERT_LE(cv::norm(dst_features[1], ref2, NORM_INF), acc_thresh);
    ASSERT_LE(cv::norm(dst_features[2], ref1, NORM_INF), acc_thresh);
    ASSERT_LE(cv::norm(dst_features[3], ref1, NORM_INF), acc_thresh);

    // check 16 bit depth:
    test_im.convertTo(test_im, CV_16U, 256.0);
    wb->setRangeMaxVal(65535);
    wb->setSaturationThreshold(0.98f);
    wb->setHistBinNum(128);
    wb->extractSimpleFeatures(test_im, dst_features);
    ASSERT_LE(cv::norm(dst_features[0], ref1, NORM_INF), acc_thresh);
    ASSERT_LE(cv::norm(dst_features[1], ref2, NORM_INF), acc_thresh);
    ASSERT_LE(cv::norm(dst_features[2], ref1, NORM_INF), acc_thresh);
    ASSERT_LE(cv::norm(dst_features[3], ref1, NORM_INF), acc_thresh);
}
}
