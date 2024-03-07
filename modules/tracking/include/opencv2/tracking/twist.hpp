#ifndef OPENCV_TWIST_HPP
#define OPENCV_TWIST_HPP

#include "opencv2/video/tracking.hpp"
#include "opencv2/core.hpp"

namespace cv
{
class CV_EXPORTS Twist
{};

namespace detail
{
inline namespace tracking
{
//! @addtogroup tracking_detail
//! @{

class CV_EXPORTS Twist
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

//! @}

} // namespace tracking
} // namespace detail
} // namespace cv

#endif
