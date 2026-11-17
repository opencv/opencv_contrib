// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_TRITON_VALIDATION_GUARDS_HPP
#define OPENCV_TRITON_VALIDATION_GUARDS_HPP
#include "opencv2/core.hpp"
#include <cstring>
namespace opencv_test
{
namespace triton_test
{
/** Compare pixels outside changed, preserving each input's pitch and raw representation. */
inline bool guardsEqual(const cv::Mat &before, const cv::Mat &after, cv::Rect changed)
{
    if (before.dims != 2 || after.dims != 2 || before.size() != after.size() ||
        before.type() != after.type())
        return false;
    for (int y = 0; y < before.rows; ++y)
        for (int x = 0; x < before.cols; ++x)
            if (!changed.contains(cv::Point(x, y)) &&
                std::memcmp(before.ptr(y) + x * before.elemSize(),
                            after.ptr(y) + x * after.elemSize(), before.elemSize()) != 0)
                return false;
    return true;
}
}
}
#endif
