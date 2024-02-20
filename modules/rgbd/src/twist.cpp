
#include "precomp.hpp"
#include "opencv2/rgbd/twist.hpp"

namespace cv
{
namespace rgbd
{

Twist::Twist()
{
    _optflow = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
    _flow = new cv::Mat();
}

void Twist::interactionMatrix(const cv::Mat& uv, const cv::Mat& depth, const cv::Mat& K, cv::Mat& J)
{
    CV_Assert(uv.cols == depth.cols);
    CV_Assert(depth.type() == CV_32F); // Validate depth input type

    J.create(depth.cols * 2, 6, CV_32F);
    J.setTo(0);

    cv::Mat Kinv;
    cv::invert(K, Kinv);

    cv::Mat xy(3, 1, CV_32F);
    cv::Mat Jp(2, 6, CV_32F);
    for (int i = 0; i < uv.cols; i++)
    {
        const float z = depth.at<float>(i);
        if (cv::abs(z) < 0.001f)
            continue;

        const cv::Point3f p(uv.at<float>(0, i), uv.at<float>(1, i), 1.0);

        // convert to normalized image-plane coordinates
        xy = Kinv * cv::Mat(p);
        float x = xy.at<float>(0);
        float y = xy.at<float>(1);

        // 2x6 Jacobian for this point
        Jp = (cv::Mat_<float>(2, 6) << -1 / z, 0.0, x / z, x * y, -(1 + x * x), y, 0.0, -1 / z,
              y / z, 1 + y * y, -x * y, -x);

        Jp = K(cv::Rect(0, 0, 2, 2)) * Jp;

        // push into Jacobian
        Jp.copyTo(J(cv::Rect(0, 2 * i, 6, 2)));
    }
}

cv::Vec6d Twist::compute(const cv::Mat& im0, const cv::Mat& im1, const cv::Mat depths0, const cv::Mat& K,
               const double dt)
{
    _optflow->calc(im0, im1, *_flow);

    const int N = (im0.cols * im0.rows) * 0.1;
    cv::Mat uv(2, N, CV_32F);
    cv::Mat depth(1, N, CV_32F);
    cv::Mat flow(1, 2 * N, CV_32F);
    for (int i = 0; i < N; i++)
    {
        int x = rand() % im0.cols;
        int y = rand() % im0.rows;
        uv.at<float>(0, i) = x;
        uv.at<float>(1, i) = y;
        depth.at<float>(i) = depths0.at<float>(y, x);
        flow.at<float>(0, 2 * i) = _flow->at<cv::Point2f>(y, x).x;
        flow.at<float>(0, 2 * i + 1) = _flow->at<cv::Point2f>(y, x).y;
    }
    cv::Mat J;
    interactionMatrix(uv, depth, K, J);

    cv::Mat Jinv;
    cv::invert(J, Jinv, cv::DECOMP_SVD);
    cv::Mat duv = flow / dt;
    cv::Mat twist = Jinv * duv.t();
    return twist;
}

} // namespace rgbd
} // namespace cv
