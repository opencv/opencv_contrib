
#include "precomp.hpp"
#include "opencv2/tracking/twist.hpp"

namespace cv
{
namespace detail
{
inline namespace tracking
{

void computeInteractionMatrix(const cv::Mat& uv, const cv::Mat& depths, const cv::Mat& K_, cv::Mat& J)
{
    CV_Assert(uv.cols == depths.cols);
    CV_Assert(depths.type() == CV_32F);
    CV_Assert(K_.cols == 3 && K_.rows == 3 && K_.type() == CV_32F);

    J.create(depths.cols * 2, 6, CV_32F);
    J.setTo(0);

    Matx33f K, Kinv;
    K_.copyTo(K);
    Kinv = K.inv();

    for (int i = 0; i < uv.cols; i++)
    {
        const float z = depths.at<float>(i);
        // skip points with zero depth
        if (cv::abs(z) < 0.001f)
            continue;

        const cv::Matx31f p(uv.at<float>(0, i), uv.at<float>(1, i), 1.0);

        // convert to normalized image-plane coordinates
        Matx31f xy = Kinv * p;
        float x = xy(0,0);
        float y = xy(1,0);

        Matx<float, 2, 6> Jp;

        // 2x6 Jacobian for this point
        Jp(0, 0) = -1 / z;
        Jp(0, 1) = 0.0;
        Jp(0, 2) = x / z;
        Jp(0, 3) = x * y;
        Jp(0, 4) = -(1 + x * x);
        Jp(0, 5) = y;
        Jp(1, 0) = 0.0;
        Jp(1, 1) = -1 / z;
        Jp(1, 2) = y / z;
        Jp(1, 3) = 1 + y * y;
        Jp(1, 4) = -x * y;
        Jp(1, 5) = -x;

        Jp = Matx22f(K(0,0), K(0,1), K(1,0), K(1,1)) * Jp;

        // push into Jacobian
        Mat(2, 6, CV_32F, Jp.val).copyTo(J(cv::Rect(0, 2 * i, 6, 2)));
    }
}

cv::Vec6d computeTwist(const cv::Mat& uv, const cv::Mat& duv, const cv::Mat& depths,
                       const cv::Mat& K)
{
    CV_Assert(uv.cols * 2 == duv.rows);

    cv::Mat J;
    computeInteractionMatrix(uv, depths, K, J);
    cv::Mat Jinv;
    cv::invert(J, Jinv, cv::DECOMP_SVD);
    cv::Mat twist = Jinv * duv;
    return twist;
}

} // namespace tracking
} // namespace detail
} // namespace cv
