#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(RGBD_Linemod, MatchUnaligned)
{
    cv::Mat unaligned_mat(100, 17, CV_8UC1);
    unaligned_mat.setTo(0);

    cv::linemod::Detector detector;
    
    std::vector<cv::Mat> sources;
    sources.push_back(unaligned_mat);

    std::vector<cv::linemod::Match> matches;
    
    EXPECT_NO_THROW(detector.match(sources, 80.0f, matches));
}

}}