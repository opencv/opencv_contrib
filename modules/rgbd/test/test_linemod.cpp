#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(RGBD_Linemod, MatchUnaligned)
{
    cv::Mat color_full(480, 641, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat depth_full(480, 641, CV_16UC1, cv::Scalar(0));

    cv::Mat color_unaligned = color_full(cv::Rect(1, 0, 640, 480));
    cv::Mat depth_unaligned = depth_full(cv::Rect(1, 0, 640, 480));

    cv::Ptr<cv::linemod::Detector> detector = cv::linemod::getDefaultLINEMOD();

    std::vector<cv::Mat> sources;
    sources.push_back(color_unaligned);
    sources.push_back(depth_unaligned);

    std::vector<cv::linemod::Match> matches;

    EXPECT_NO_THROW(detector->match(sources, 80.0f, matches));
}

}}