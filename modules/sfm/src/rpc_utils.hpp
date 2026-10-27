// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SFM_RPC_UTILS_HPP
#define OPENCV_SFM_RPC_UTILS_HPP

#include <opencv2/sfm/rpc.hpp>
#include <cmath>

namespace cv { namespace sfm { namespace rpc_detail {

// A small shared adapter preserves matrix strides and accepts the usual OpenCV point vectors.
inline Mat pointRows(InputArray input, int dimensions)
{
    Mat points = input.getMat();
    if (points.empty())
        return Mat(0, dimensions, CV_64F);
    CV_CheckLE(points.dims, 2, "Expected an RPC point vector or two-dimensional matrix");
    CV_CheckEQ(points.depth(), CV_64F, "RPC coordinates must be double precision");
    if (points.channels() == dimensions && (points.rows == 1 || points.cols == 1))
    {
        if (points.rows == 1)
            return points.reshape(1, points.cols);
        return points.reshape(1);
    }
    CV_CheckEQ(points.dims, 2, "Expected a point vector or N-by-D matrix");
    CV_CheckEQ(points.channels(), 1, "Expected a point vector or N-by-D matrix");
    CV_CheckEQ(points.cols, dimensions, "Expected N-by-D coordinates");
    return points;
}

inline bool finite(const Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

// Geographic = linear * local + offset. All frame validation is performed here.
void localTransform(const RPCLocalFrame& frame, Matx33d& linear, Vec3d& offset,
                    Matx33d* inverse = 0);

}}} // namespace cv::sfm::rpc_detail
#endif
