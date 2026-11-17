// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Private default resize launch configurations, indexed by route, layout and depth.
#ifndef OPENCV_TRITON_RUNTIME_DISPATCH_TABLE_HPP
#define OPENCV_TRITON_RUNTIME_DISPATCH_TABLE_HPP
namespace cv_triton_runtime
{

struct DispatchRow
{
    const char *route; // route_name(): copy, nearest, linear, cubic, area_fast, area
    int cn;
    bool f32;
    const char *ydir; // "up" when dst_h > src_h, else "down"
    bool two_by_two;  // area_fast only: ix == 2 and iy == 2
    int block_h, block_w, num_warps;
};

// Fallback configuration for copies and unlisted layout classes.
constexpr int DISPATCH_FALLBACK_BLOCK_H = 1;
constexpr int DISPATCH_FALLBACK_BLOCK_W = 256;
constexpr int DISPATCH_FALLBACK_NUM_WARPS = 4;
constexpr int DISPATCH_DEFAULT_NUM_STAGES = 3;
// Minimum program count used when shrinking tiles for small outputs.
constexpr int DISPATCH_MIN_PROGRAMS = 512;

// Columns: route, channels, float32, vertical direction, exact-2x2 AREA,
//          block height, block width, warp count. Block width units follow the route.
inline constexpr DispatchRow DISPATCH_TABLE[] = {
    {"nearest", 1, false, "down", false, 2, 512, 4},
    {"nearest", 1, false, "up", false, 8, 512, 2},
    {"nearest", 1, true, "down", false, 2, 512, 4},
    {"nearest", 1, true, "up", false, 4, 512, 4},
    {"nearest", 3, false, "down", false, 2, 256, 2},
    {"nearest", 3, false, "up", false, 8, 512, 2},
    {"nearest", 3, true, "down", false, 2, 256, 2},
    {"nearest", 3, true, "up", false, 2, 256, 2},
    {"linear", 1, false, "down", false, 2, 128, 1},
    {"linear", 1, false, "up", false, 8, 128, 1},
    {"linear", 1, true, "down", false, 2, 256, 2},
    {"linear", 1, true, "up", false, 4, 256, 2},
    {"linear", 3, false, "down", false, 2, 256, 2},
    {"linear", 3, false, "up", false, 16, 128, 1},
    {"linear", 3, true, "down", false, 2, 256, 2},
    {"linear", 3, true, "up", false, 8, 256, 2},
    {"cubic", 1, false, "down", false, 2, 128, 1},
    {"cubic", 1, false, "up", false, 8, 128, 1},
    {"cubic", 1, true, "down", false, 2, 128, 1},
    {"cubic", 1, true, "up", false, 8, 128, 1},
    {"cubic", 3, false, "down", false, 2, 256, 2},
    {"cubic", 3, false, "up", false, 16, 256, 2},
    {"cubic", 3, true, "down", false, 2, 256, 2},
    {"cubic", 3, true, "up", false, 16, 256, 2},
    {"area", 1, false, "down", false, 1, 64, 1},
    {"area", 1, true, "down", false, 1, 256, 4},
    {"area", 3, false, "down", false, 1, 32, 1},
    {"area", 3, true, "down", false, 1, 32, 1},
    {"area_fast", 1, false, "down", false, 1, 32, 1},
    {"area_fast", 1, true, "down", false, 1, 32, 1},
    {"area_fast", 3, false, "down", false, 1, 64, 2},
    {"area_fast", 3, true, "down", false, 1, 64, 2},
    {"area_fast", 1, false, "down", true, 1, 128, 1},
    {"area_fast", 1, true, "down", true, 1, 128, 2},
    {"area_fast", 3, false, "down", true, 1, 64, 1},
    {"area_fast", 3, true, "down", true, 1, 64, 1},
};

} // namespace cv_triton_runtime
#endif // OPENCV_TRITON_RUNTIME_DISPATCH_TABLE_HPP
