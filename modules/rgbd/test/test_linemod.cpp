#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(RGBD_Linemod, MatchUnaligned)
{
    cv::Mat color(128, 128, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat depth(128, 128, CV_16UC1, cv::Scalar(0));

    cv::rectangle(color, cv::Rect(32, 32, 64, 64), cv::Scalar(255, 255, 255), -1);
    cv::rectangle(depth, cv::Rect(32, 32, 64, 64), cv::Scalar(10000), -1);

    cv::Ptr<cv::linemod::Detector> detector = cv::linemod::getDefaultLINEMOD();

    std::vector<cv::Mat> sources;
    sources.push_back(color);
    sources.push_back(depth);

    detector->addTemplate(sources, "object", cv::Mat());

    std::vector<cv::linemod::Match> matches;

    // التنفيذ للتأكد من عدم حدوث Segfault عند المعالجة
    EXPECT_NO_THROW(detector->match(sources, 80.0f, matches));
}

}}