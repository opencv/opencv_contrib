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

        for (int y = 0; y < dsize.height; ++y)
        {
            for (int x = 0; x < dsize.width; ++x)
            {
                for (int c = 0; c < cn; ++c)
                    dst.at<T>(y, x * cn + c) = LanczosInterpolator<T>::getValue(src, y * ify, x * ifx, c, cv::BORDER_REPLICATE);
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

/////////////////
// Performance Test

PARAM_TEST_CASE(ResizeLanczosPerformance, cv::cuda::DeviceInfo, cv::Size, MatType, double)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    double coeff;
    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        type = GET_PARAM(2);
        coeff = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(ResizeLanczosPerformance, Performance)
{
    cv::Mat src = randomMat(size, type);
    cv::Size dsize(cv::saturate_cast<int>(src.cols * coeff), cv::saturate_cast<int>(src.rows * coeff));

    // Warm up
    {
        cv::cuda::GpuMat gpuSrc, gpuDst;
        gpuSrc.upload(src);
        gpuDst.create(dsize, type);
        cv::cuda::resize(gpuSrc, gpuDst, cv::Size(), coeff, coeff, cv::INTER_LANCZOS4);
        cv::Mat dummy;
        gpuDst.download(dummy); // Synchronize
    }

    // GPU performance
    const int iterations = 100;
    cv::TickMeter tm_gpu;
    cv::cuda::GpuMat gpuSrc, gpuDst;
    gpuSrc.upload(src);
    gpuDst.create(dsize, type);

    tm_gpu.start();
    for (int i = 0; i < iterations; ++i)
    {
        cv::cuda::resize(gpuSrc, gpuDst, cv::Size(), coeff, coeff, cv::INTER_LANCZOS4);
    }
    cv::Mat dummy;
    gpuDst.download(dummy); // Synchronize
    tm_gpu.stop();

    // CPU performance
    cv::TickMeter tm_cpu;
    cv::Mat dst_cpu;

    tm_cpu.start();
    for (int i = 0; i < iterations; ++i)
    {
        cv::resize(src, dst_cpu, cv::Size(), coeff, coeff, cv::INTER_LANCZOS4);
    }
    tm_cpu.stop();

    double gpu_time = tm_gpu.getTimeMilli() / iterations;
    double cpu_time = tm_cpu.getTimeMilli() / iterations;
    double speedup = cpu_time / gpu_time;

    std::cout << "Size: " << size << " -> " << dsize 
              << ", Type: " << type 
              << ", Coeff: " << coeff << std::endl;
    std::cout << "  CPU: " << cpu_time << " ms" << std::endl;
    std::cout << "  GPU: " << gpu_time << " ms" << std::endl;
    std::cout << "  Speedup: " << speedup << "x" << std::endl;
}

INSTANTIATE_TEST_CASE_P(CUDA_Warping, ResizeLanczosPerformance, testing::Combine(
    testing::Values(cv::cuda::DeviceInfo()),
    testing::Values(cv::Size(512, 512), cv::Size(1024, 1024), cv::Size(2048, 2048)),
    testing::Values(MatType(CV_8UC1), MatType(CV_8UC3), MatType(CV_32FC1), MatType(CV_32FC3)),
    testing::Values(0.5, 1.5, 2.0)));

}} // namespace
#endif // HAVE_CUDA

