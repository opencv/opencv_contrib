// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "cuda/moments.cuh"

using namespace cv;
using namespace cv::cuda;

int cv::cuda::numMoments(const MomentsOrder order) {
    return order == MomentsOrder::FIRST_ORDER_MOMENTS ? device::imgproc::n1 : order == MomentsOrder::SECOND_ORDER_MOMENTS ? device::imgproc::n12 : device::imgproc::n123;
}

template<typename T>
cv::Moments convertSpatialMomentsT(Mat spatialMoments, const MomentsOrder order) {
    switch (order) {
    case MomentsOrder::FIRST_ORDER_MOMENTS:
        return Moments(spatialMoments.at<T>(0), spatialMoments.at<T>(1), spatialMoments.at<T>(2), 0, 0, 0, 0, 0, 0, 0);
    case MomentsOrder::SECOND_ORDER_MOMENTS:
        return Moments(spatialMoments.at<T>(0), spatialMoments.at<T>(1), spatialMoments.at<T>(2), spatialMoments.at<T>(3), spatialMoments.at<T>(4), spatialMoments.at<T>(5), 0, 0, 0, 0);
    default:
        return Moments(spatialMoments.at<T>(0), spatialMoments.at<T>(1), spatialMoments.at<T>(2), spatialMoments.at<T>(3), spatialMoments.at<T>(4), spatialMoments.at<T>(5), spatialMoments.at<T>(6), spatialMoments.at<T>(7), spatialMoments.at<T>(8), spatialMoments.at<T>(9));
    }
}

cv::Moments cv::cuda::convertSpatialMoments(Mat spatialMoments, const MomentsOrder order, const int momentsType) {
    if (momentsType == CV_32F)
        return convertSpatialMomentsT<float>(spatialMoments, order);
    else
        return convertSpatialMomentsT<double>(spatialMoments, order);
}

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)
    Moments cv::cuda::moments(InputArray src, const bool binary, const MomentsOrder order, const int momentsType) { throw_no_cuda(); }
    void spatialMoments(InputArray src, OutputArray moments, const bool binary, const MomentsOrder order, const int momentsType, Stream& stream) { throw_no_cuda(); }
#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device { namespace imgproc {
        template <typename TSrc, typename TMoments>
        void moments(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
}}}}

void cv::cuda::spatialMoments(InputArray src, OutputArray moments, const bool binary, const MomentsOrder order, const int momentsType, Stream& stream) {
    CV_Assert(src.depth() <= CV_64F);
    const GpuMat srcDevice = getInputMat(src, stream);

    CV_Assert(momentsType == CV_32F || momentsType == CV_64F);
    const int nMoments = numMoments(order);
    const int momentsCols = nMoments < moments.cols() ? moments.cols() : nMoments;
    GpuMat momentsDevice = getOutputMat(moments, 1, momentsCols, momentsType, stream);
    momentsDevice.setTo(0);

    Point ofs; Size wholeSize;
    srcDevice.locateROI(wholeSize, ofs);

    typedef void (*func_t)(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
    static const func_t funcs[7][2] =
    {
        {device::imgproc::moments<uchar, float>,  device::imgproc::moments<uchar, double> },
        {device::imgproc::moments<schar, float>,  device::imgproc::moments<schar, double> },
        {device::imgproc::moments<ushort, float>, device::imgproc::moments<ushort, double>},
        {device::imgproc::moments<short, float>,  device::imgproc::moments<short, double> },
        {device::imgproc::moments<int, float>,    device::imgproc::moments<int, double> },
        {device::imgproc::moments<float, float>,  device::imgproc::moments<float, double> },
        {device::imgproc::moments<double, float>, device::imgproc::moments<double, double> }
    };

    const func_t func = funcs[srcDevice.depth()][momentsType == CV_64F];
    func(srcDevice, momentsDevice, binary, static_cast<int>(order), ofs.x, StreamAccessor::getStream(stream));
    syncOutput(momentsDevice, moments, stream);
}

Moments cv::cuda::moments(InputArray src, const bool binary, const MomentsOrder order, const int momentsType) {
    Stream stream;
    HostMem dst;
    spatialMoments(src, dst, binary, order, momentsType, stream);
    stream.waitForCompletion();
    Mat moments = dst.createMatHeader();
    return convertSpatialMoments(moments, order, momentsType);
}

#endif /* !defined (HAVE_CUDA) */
