// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "entry.hpp"
namespace cv
{
namespace triton
{
namespace detail
{
void requireStatus(const cv_triton_runtime::Status &s, const char *operation, bool submittedOnly)
{
    if (submittedOnly ? s.code == cv_triton_runtime::Code::Submitted : s.ok())
        return;
    using cv_triton_runtime::Reason;
    const String what = cv::format("cv::triton::%s: %s (%s)", operation, s.message.c_str(),
                                   cv_triton_runtime::reason_name(s.reason));
    // Unsupported types/modes map to StsUnsupportedFormat; malformed request
    // geometry, layout and ownership map to StsBadArg. Execution failures use StsError.
    if (s.code == cv_triton_runtime::Code::Unsupported)
    {
        switch (s.reason)
        {
        case Reason::DepthOrChannels:
        case Reason::Interpolation:
        case Reason::TypeMismatch:
        case Reason::RouteNotEnabled:
            CV_Error(Error::StsUnsupportedFormat, what);
        default:
            CV_Error(Error::StsBadArg, what);
        }
    }
    CV_Error(Error::StsError,
             what + (s.submitted ? " -- work had already been submitted; dst is undefined" : ""));
}

OutputInfo resizeOutput(const cv_triton_runtime::DeviceView &src, Size dsize, double fx, double fy,
                        int interpolation)
{
    cv_triton_runtime::ResizeRequest request;
    request.src = src;
    request.dsize_w = dsize.width;
    request.dsize_h = dsize.height;
    request.fx = fx;
    request.fy = fy;
    request.interpolation = interpolation;
    cv_triton_resize::Size2 size;
    requireStatus(cv_triton_runtime::resolve_output(request, size), "resize");
    const int depth = src.type == cv_triton_runtime::ElemType::F32 ? CV_32F : CV_8U;
    return {Size(size.dst_w, size.dst_h), CV_MAKETYPE(depth, src.channels)};
}

OutputInfo cvtColorOutput(const cv_triton_runtime::DeviceView &src, int code)
{
    cv_triton_color::Output out;
    requireStatus(cv_triton_color::resolve_output(src, code, out), "cvtColor");
    const int depth = src.type == cv_triton_runtime::ElemType::F32
                          ? CV_32F
                          : (src.type == cv_triton_runtime::ElemType::U16 ? CV_16U : CV_8U);
    return {Size(out.cols, out.rows), CV_MAKETYPE(depth, out.spec->dcn)};
}

cv_triton_warp::Request warpRequest(const cv_triton_runtime::DeviceView &src, InputArray matrix,
                                    int flags, int border, const Scalar &value)
{
    if (flags & ~(INTER_MAX | WARP_INVERSE_MAP))
        CV_Error(Error::StsUnsupportedFormat, "cv::triton::warpAffine: unsupported flags");
    const Mat m = matrix.getMat();
    CV_Assert(m.rows == 2 && m.cols == 3 && (m.type() == CV_32F || m.type() == CV_64F));
    Mat m64;
    m.convertTo(m64, CV_64F);
    cv_triton_warp::Request request;
    request.src = src;
    for (int i = 0; i < 6; ++i)
        request.M[i] = m64.at<double>(i / 3, i % 3);
    for (int i = 0; i < 4; ++i)
        request.border[i] = value[i];
    request.interpolation = flags & INTER_MAX;
    request.inverse_map = (flags & WARP_INVERSE_MAP) != 0;
    request.border_mode = border;
    return request;
}

OutputInfo warpOutput(const cv_triton_warp::Request &request, Size dsize)
{
    requireStatus(cv_triton_warp::validate_source(request), "warpAffine");
    if (dsize.empty())
        dsize = Size(request.src.cols, request.src.rows);
    requireStatus(cv_triton_warp::validate_output(request, dsize.height, dsize.width),
                  "warpAffine");
    const int depth = request.src.type == cv_triton_runtime::ElemType::F32
                          ? CV_32F
                          : (request.src.type == cv_triton_runtime::ElemType::U16 ? CV_16U : CV_8U);
    return {dsize, CV_MAKETYPE(depth, request.src.channels)};
}
}
}
} // namespace cv::triton::detail
