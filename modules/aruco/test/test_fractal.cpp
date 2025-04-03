#include "opencv2/aruco/fractal_markers.hpp"
#include "opencv2/ts.hpp"

namespace opencv_test {
namespace {

TEST(FractalMarkers, API_Exists) {
    EXPECT_NO_THROW({
        cv::Ptr<cv::aruco::FractalDictionary> dict;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::Mat testImage(480, 640, CV_8UC1, cv::Scalar(0));
        cv::aruco::detectFractalMarkers(testImage, dict, corners, noArray());
    });
}

} // namespace
} // namespace opencv_test