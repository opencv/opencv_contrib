#include "test_precomp.hpp"

#ifdef HAVE_CUDA

namespace opencv_test { namespace {

///////////////////////////////////////////////////////////////////
// Gold implementation

namespace
{
    template <typename T>
    void resizeLanczosImpl(const cv::Mat& src, cv::Mat& dst, double fx, double fy)
    {
        const int cn = src.channels();

        cv::Size dsize(cv::saturate_cast<int>(src.cols * fx), cv::saturate_cast<int>(src.rows * fy));

        dst.create(dsize, src.type());

        float ifx = static_cast<float>(1.0 / fx);
        float ify = static_cast<float>(1.0 / fy);

        // OpenCV CPU resize uses center-aligned coordinate mapping: (x + 0.5) * fx - 0.5
        // Since fx and fy here are scale factors, and ifx = 1.0 / fx, ify = 1.0 / fy,
        // the center-aligned mapping becomes: (x + 0.5) / fx - 0.5 = (x + 0.5) * ifx - 0.5
        for (int y = 0; y < dsize.height; ++y)
        {
            for (int x = 0; x < dsize.width; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    float src_x = (static_cast<float>(x) + 0.5f) * ifx - 0.5f;
                    float src_y = (static_cast<float>(y) + 0.5f) * ify - 0.5f;
                    dst.at<T>(y, x * cn + c) = LanczosInterpolator<T>::getValue(src, src_y, src_x, c, cv::BORDER_REPLICATE);
                }
            }
        }
    }

    void resizeLanczosGold(const cv::Mat& src, cv::Mat& dst, double fx, double fy)
    {
        typedef void (*func_t)(const cv::Mat& src, cv::Mat& dst, double fx, double fy);

        static const func_t lanczos_funcs[] =
        {
            resizeLanczosImpl<unsigned char>,
            resizeLanczosImpl<signed char>,
            resizeLanczosImpl<unsigned short>,
            resizeLanczosImpl<short>,
            resizeLanczosImpl<int>,
            resizeLanczosImpl<float>
        };

        lanczos_funcs[src.depth()](src, dst, fx, fy);
    }
}

///////////////////////////////////////////////////////////////////
// Test

PARAM_TEST_CASE(ResizeLanczos, cv::cuda::DeviceInfo, cv::Size, MatType, double, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    double coeff;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        coeff = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }

    virtual void TearDown()
    {
        // GpuMat destructors will automatically clean up GPU memory
    }
};

CUDA_TEST_P(ResizeLanczos, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::cuda::GpuMat dst = createMat(cv::Size(cv::saturate_cast<int>(src.cols * coeff), cv::saturate_cast<int>(src.rows * coeff)), type, useRoi);
    cv::cuda::resize(loadMat(src, useRoi), dst, cv::Size(), coeff, coeff, cv::INTER_LANCZOS4);

    cv::Mat dst_gold;
    resizeLanczosGold(src, dst_gold, coeff, coeff);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-2 : 1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Warping, ResizeLanczos, testing::Combine(
    testing::Values(cv::cuda::DeviceInfo()),
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(0.3, 0.5, 1.5, 2.0),
    WHOLE_SUBMAT));

/////////////////

PARAM_TEST_CASE(ResizeLanczosSameAsHost, cv::cuda::DeviceInfo, cv::Size, MatType, double, UseRoi)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    double coeff;
    int type;
    bool useRoi;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        coeff = GET_PARAM(3);
        useRoi = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }

    virtual void TearDown()
    {
        // GpuMat destructors will automatically clean up GPU memory
    }
};

CUDA_TEST_P(ResizeLanczosSameAsHost, Accuracy)
{
    cv::Mat src = randomMat(size, type);

    cv::cuda::GpuMat dst = createMat(cv::Size(cv::saturate_cast<int>(src.cols * coeff), cv::saturate_cast<int>(src.rows * coeff)), type, useRoi);
    cv::cuda::resize(loadMat(src, useRoi), dst, cv::Size(), coeff, coeff, cv::INTER_LANCZOS4);

    cv::Mat dst_gold;
    cv::resize(src, dst_gold, cv::Size(), coeff, coeff, cv::INTER_LANCZOS4);

    EXPECT_MAT_NEAR(dst_gold, dst, src.depth() == CV_32F ? 1e-2 : 1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_Warping, ResizeLanczosSameAsHost, testing::Combine(
    testing::Values(cv::cuda::DeviceInfo()),
    DIFFERENT_SIZES,
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_8UC4), MatType(CV_16UC1), MatType(CV_16UC3), MatType(CV_16UC4), MatType(CV_32FC1), MatType(CV_32FC3), MatType(CV_32FC4)),
    testing::Values(0.3, 0.5, 1.5, 2.0),
    WHOLE_SUBMAT));

}} // namespace
#endif // HAVE_CUDA
