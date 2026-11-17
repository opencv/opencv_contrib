// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_TRITON_ENTRY_HPP
#define OPENCV_TRITON_ENTRY_HPP
#include "runtime/runtime.hpp"
namespace cv
{
namespace triton
{
namespace detail
{
// Adapter-facing metadata only. Allocation and native ownership stay with the adapter.
struct OutputInfo
{
    Size size;
    int type;
};
CV_EXPORTS void requireStatus(const cv_triton_runtime::Status &status, const char *operation,
                              bool submittedOnly = false);
CV_EXPORTS OutputInfo resizeOutput(const cv_triton_runtime::DeviceView &src, Size dsize, double fx,
                                   double fy, int interpolation);
CV_EXPORTS OutputInfo cvtColorOutput(const cv_triton_runtime::DeviceView &src, int code);
CV_EXPORTS cv_triton_warp::Request warpRequest(const cv_triton_runtime::DeviceView &src,
                                               InputArray matrix, int flags, int border,
                                               const Scalar &value);
CV_EXPORTS OutputInfo warpOutput(const cv_triton_warp::Request &request, Size dsize);
}
}
} // namespace cv::triton::detail
#endif
