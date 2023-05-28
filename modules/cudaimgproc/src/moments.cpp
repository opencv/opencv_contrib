// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

cv::Moments cv::cuda::moments(InputArray _src, bool binary) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace cuda { namespace device { namespace imgproc {
        Moments Moments(const cv::cuda::GpuMat& img, bool binaryImage);
}}}}


cv::Moments cv::cuda::moments(InputArray _src, bool binary) {
    const cv::cuda::GpuMat src = _src.getGpuMat();
    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    CV_Assert(cn == 1 && depth == CV_8U);

    return cv::cuda::device::imgproc::Moments(src, binary);
}

#endif /* !defined (HAVE_CUDA) */
