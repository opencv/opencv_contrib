// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
#ifndef OPENCV_TRITON_RESIZE_PLAN_HPP
#define OPENCV_TRITON_RESIZE_PLAN_HPP
#include <vector>
// Private resize parameter resolution and device-table construction.
// Coefficients use separate float32 operations; compile with -ffp-contract=off.
namespace cv_triton_resize
{

enum Interp
{
    INTER_NEAREST = 0,
    INTER_LINEAR = 1,
    INTER_CUBIC = 2,
    INTER_AREA = 3
};
enum Route
{
    ROUTE_COPY,
    ROUTE_NEAREST,
    ROUTE_LINEAR,
    ROUTE_CUBIC,
    ROUTE_AREA_FAST,
    ROUTE_AREA
};

const char *route_name(Route r);

// cv::resize parameter resolution. dsize wins when both sides are > 0; otherwise fx/fy
// (both > 0) give dst = cvRound(src * f). Size2 retains the original fx/fy values.
struct Size2
{
    int dst_h, dst_w;
    double inv_y, inv_x;
};
Size2 resolve_size(int src_h, int src_w, int dst_w, int dst_h, double fx, double fy);

struct RouteInfo
{
    Route kind;
    bool area_mode = false; // linear path with AREA's coefficient formula (AREA upscale)
    int ix = 0, iy = 0;     // integer ratio of ROUTE_AREA_FAST
};
RouteInfo route(int src_h, int src_w, int dst_h, int dst_w, double inv_y, double inv_x, int interp);

// One axis of the separable path: source offsets, float32 coefficients [dst x ksize],
// and horizontal destination-index boundaries [xmin, xmax). Outside this range
// the sampling window reaches a source border and requires edge handling.
struct Axis
{
    std::vector<int> ofs;
    std::vector<float> coeffs;
    int ksize = 2, xmin = 0, xmax = 0;
};
Axis generic_axis(int dst, int src, double inv_scale, int ksize, bool area_mode, bool horizontal);
// saturate_cast<short>(c * INTER_RESIZE_COEF_SCALE)
std::vector<int> fixed_coeffs(const std::vector<float> &coeffs);

// Rectangular storage for variable-length AREA tap lists. Each destination uses
// count[d] entries; remaining src_index/alpha slots are zero-filled.
struct AreaTab
{
    int taps = 0;               // K
    std::vector<int> src_index; // [dst x K]
    std::vector<float> alpha;   // [dst x K]
    std::vector<int> count;     // [dst]
};
AreaTab area_tab(int src, int dst, double scale);

} // namespace cv_triton_resize
#endif  // OPENCV_TRITON_RESIZE_PLAN_HPP
