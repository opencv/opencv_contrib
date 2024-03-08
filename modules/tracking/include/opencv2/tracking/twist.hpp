#ifndef OPENCV_TWIST_HPP
#define OPENCV_TWIST_HPP

#include "opencv2/core.hpp"

namespace cv
{
class CV_EXPORTS Twist
{
};

namespace detail
{
inline namespace tracking
{
//! @addtogroup tracking_detail
//! @{

class CV_EXPORTS Twist
{
public:
    Twist() = default;

    // TODO(ernie): docs
    cv::Vec6d compute(const cv::Mat& uv, const cv::Mat& duv, const cv::Mat depths,
                      const cv::Mat& K);

    // TODO(ernie): docs
    void interactionMatrix(const cv::Mat& uv, const cv::Mat& depth, const cv::Mat& K, cv::Mat& J);
};

//! @}

} // namespace tracking
} // namespace detail
} // namespace cv

#endif
