// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "opencv2/core/private.cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

namespace {

template <typename T, int cn>
void inRangeImpl(const GpuMat& src,
                 const Scalar& lowerb,
                 const Scalar& upperb,
                 GpuMat& dst,
                 Stream& stream) {
    gridTransformUnary(globPtr<typename MakeVec<T, cn>::type>(src),
                       globPtr<uchar>(dst),
                       InRangeFunc<T, cn>(lowerb, upperb),
                       stream);
}

}  // namespace

void cv::cuda::inRange(InputArray _src,
                       const Scalar& _lowerb,
                       const Scalar& _upperb,
                       OutputArray _dst,
                       Stream& stream) {
    const GpuMat src = getInputMat(_src, stream);

    typedef void (*func_t)(const GpuMat& src,
                           const Scalar& lowerb,
                           const Scalar& upperb,
                           GpuMat& dst,
                           Stream& stream);

    // Note: We cannot support 16F with the current implementation because we
    // use a CUDA vector (e.g. int3) to store the bounds, and there is no CUDA
    // vector type for float16
    static constexpr const int MAX_CHANNELS = 4;
    static constexpr const int NUM_DEPTHS = CV_64F + 1;

    static const std::array<std::array<func_t, NUM_DEPTHS>, MAX_CHANNELS>
            funcs = {std::array<func_t, NUM_DEPTHS>{inRangeImpl<uchar, 1>,
                                                    inRangeImpl<schar, 1>,
                                                    inRangeImpl<ushort, 1>,
                                                    inRangeImpl<short, 1>,
                                                    inRangeImpl<int, 1>,
                                                    inRangeImpl<float, 1>,
                                                    inRangeImpl<double, 1>},
                     std::array<func_t, NUM_DEPTHS>{inRangeImpl<uchar, 2>,
                                                    inRangeImpl<schar, 2>,
                                                    inRangeImpl<ushort, 2>,
                                                    inRangeImpl<short, 2>,
                                                    inRangeImpl<int, 2>,
                                                    inRangeImpl<float, 2>,
                                                    inRangeImpl<double, 2>},
                     std::array<func_t, NUM_DEPTHS>{inRangeImpl<uchar, 3>,
                                                    inRangeImpl<schar, 3>,
                                                    inRangeImpl<ushort, 3>,
                                                    inRangeImpl<short, 3>,
                                                    inRangeImpl<int, 3>,
                                                    inRangeImpl<float, 3>,
                                                    inRangeImpl<double, 3>},
                     std::array<func_t, NUM_DEPTHS>{inRangeImpl<uchar, 4>,
                                                    inRangeImpl<schar, 4>,
                                                    inRangeImpl<ushort, 4>,
                                                    inRangeImpl<short, 4>,
                                                    inRangeImpl<int, 4>,
                                                    inRangeImpl<float, 4>,
                                                    inRangeImpl<double, 4>}};

    CV_CheckLE(src.channels(), MAX_CHANNELS, "Src must have <= 4 channels");
    CV_CheckLE(src.depth(),
               CV_64F,
               "Src must have depth 8U, 8S, 16U, 16S, 32S, 32F, or 64F");

    GpuMat dst = getOutputMat(_dst, src.size(), CV_8UC1, stream);

    const func_t func = funcs.at(src.channels() - 1).at(src.depth());
    func(src, _lowerb, _upperb, dst, stream);

    syncOutput(dst, _dst, stream);
}

#endif
