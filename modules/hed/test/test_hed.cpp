#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(HED, LoadModel)
{
    std::string model = cvtest::findDataFile("hed_pretrained_bsds.caffemodel", false);
    std::string proto = cvtest::findDataFile("deploy.prototxt", false);
    if (model.empty() || proto.empty())
        throw SkipTestException("HED model files not found");

    auto detector = cv::hed::HEDDetector::create(model, proto);
    ASSERT_FALSE(detector.empty());
}

TEST(HED, OutputShape)
{
    std::string model = cvtest::findDataFile("hed_pretrained_bsds.caffemodel", false);
    std::string proto = cvtest::findDataFile("deploy.prototxt", false);
    if (model.empty() || proto.empty())
        throw SkipTestException("HED model files not found");

    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
    auto detector = cv::hed::HEDDetector::create(model, proto);
    cv::Mat edges = detector->detectEdges(img);

    EXPECT_EQ(edges.rows, 100);
    EXPECT_EQ(edges.cols, 100);
    EXPECT_EQ(edges.type(), CV_32F);
}

}} // namespace

CV_TEST_MAIN(".")