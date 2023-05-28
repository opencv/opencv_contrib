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
    Stream& stream = Stream::Null();
    HostMem dst;
    spatialMoments(src, dst, binary, order, momentsType, stream);
    stream.waitForCompletion();
    Mat moments = dst.createMatHeader();
    if(momentsType == CV_32F)
        return Moments(moments.at<float>(0), moments.at<float>(1), moments.at<float>(2), moments.at<float>(3), moments.at<float>(4), moments.at<float>(5), moments.at<float>(6), moments.at<float>(7), moments.at<float>(8), moments.at<float>(9));
    else
        return Moments(moments.at<double>(0), moments.at<double>(1), moments.at<double>(2), moments.at<double>(3), moments.at<double>(4), moments.at<double>(5), moments.at<double>(6), moments.at<double>(7), moments.at<double>(8), moments.at<double>(9));
}

#endif /* !defined (HAVE_CUDA) */
