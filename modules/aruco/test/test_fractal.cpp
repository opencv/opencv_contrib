#include "opencv2/aruco/fractal_markers.hpp"
#include "opencv2/ts.hpp"

namespace opencv_test {
namespace {

TEST(FractalMarkers, BasicAPI) {
    std::vector<std::vector<cv::Point2f>> corners;
    cv::Mat testImage(480, 640, CV_8UC1, cv::Scalar(0));
    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::aruco::DetectorParameters params;
    EXPECT_NO_THROW(detectFractalMarkers(testImage, dict, corners, params));
}
}
}
