#ifndef __OPENCV_RGBD_TWIST_HPP__
#define __OPENCV_RGBD_TWIST_HPP__

#include "opencv2/video/tracking.hpp"

namespace cv
{
namespace rgbd
{
class CV_EXPORTS_W Twist
{
public:
    Twist();

    cv::Vec6d compute(const cv::Mat& im0, const cv::Mat& im1, const cv::Mat depths0,
                      const cv::Mat& K, const double dt);

private:
    void interactionMatrix(const cv::Mat& uv, const cv::Mat& depth, const cv::Mat& K, cv::Mat& J);

private:
    Ptr<DenseOpticalFlow> _optflow;
    Ptr<cv::Mat> _flow;
};
} // namespace rgbd
} // namespace cv

#endif
