// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_TRITON_TEST_RESIZE_REFERENCE_HPP
#define OPENCV_TRITON_TEST_RESIZE_REFERENCE_HPP
#include "../src/runtime/resize_plan.hpp"
#include <cstddef>
#include <cstdint>
// Test-only CPU evaluation of resize, with the generic OpenCV coefficient formulas
// and explicit 128-bit SIMD/tail rounding order. Planning is shared with the runtime;
// test_resize_reference.cpp adds independent mathematical gold cases. It does not dispatch
// through optional IPP or HAL implementations. Compile with -ffp-contract=off.
namespace cv_triton_resize
{

// An image in host memory: rows of `cols * channels` elements, `step` bytes apart.
struct ImageRef
{
    void *data;
    int rows, cols, channels;
    std::size_t step;
    bool is_f32;
};

// cv::resize on the host. dst.rows/cols must already be the resolved size (resolve_size).
// inv_y / inv_x are the resolved inverse scales. Throws std::invalid_argument on misuse.
void resize(const ImageRef &src, const ImageRef &dst, double inv_y, double inv_x, int interp);

} // namespace cv_triton_resize
#endif  // OPENCV_TRITON_TEST_RESIZE_REFERENCE_HPP
