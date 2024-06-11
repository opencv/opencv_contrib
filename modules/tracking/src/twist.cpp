
#include "precomp.hpp"
#include "opencv2/tracking/twist.hpp"

namespace cv
{
namespace detail
{
inline namespace tracking
{

void computeInteractionMatrix(const cv::Mat& uv, const cv::Mat& depths, const cv::Mat& K, cv::Mat& J)
{
    CV_Assert(uv.cols == depths.cols);
    CV_Assert(depths.type() == CV_32F);
    CV_Assert(K.cols == 3 && K.rows == 3);

    J.create(depths.cols * 2, 6, CV_32F);
    J.setTo(0);

    cv::Mat Kinv;
    cv::invert(K, Kinv);

    cv::Mat xy(3, 1, CV_32F);
    cv::Mat Jp(2, 6, CV_32F);
    for (int i = 0; i < uv.cols; i++)
    {
        const float z = depths.at<float>(i);
        // skip points with zero depth
        if (cv::abs(z) < 0.001f)
            continue;

        const cv::Point3f p(uv.at<float>(0, i), uv.at<float>(1, i), 1.0);

        // convert to normalized image-plane coordinates
        xy = Kinv * cv::Mat(p);
        float x = xy.at<float>(0);
        float y = xy.at<float>(1);

        // 2x6 Jacobian for this point
        Jp.at<float>(0, 0) = -1 / z;
        Jp.at<float>(0, 1) = 0.0;
        Jp.at<float>(0, 2) = x / z;
        Jp.at<float>(0, 3) = x * y;
        Jp.at<float>(0, 4) = -(1 + x * x);
        Jp.at<float>(0, 5) = y;
        Jp.at<float>(1, 0) = 0.0;
        Jp.at<float>(1, 1) = -1 / z;
        Jp.at<float>(1, 2) = y / z;
        Jp.at<float>(1, 3) = 1 + y * y;
        Jp.at<float>(1, 4) = -x * y;
        Jp.at<float>(1, 5) = -x;

        Jp = K(cv::Rect(0, 0, 2, 2)) * Jp;

        // push into Jacobian
        Jp.copyTo(J(cv::Rect(0, 2 * i, 6, 2)));
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
