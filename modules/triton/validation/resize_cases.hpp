// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Private case specification shared by all device adapters; not installed.
#ifndef OPENCV_TRITON_VALIDATION_RESIZE_CASES_HPP
#define OPENCV_TRITON_VALIDATION_RESIZE_CASES_HPP
#include "opencv2/imgproc.hpp"
namespace opencv_test
{
namespace triton_test
{
namespace resize_cases
{
struct RouteCase
{
    Size src, dst;
    double fx, fy;
    int interp;
    const char *route;
};

const RouteCase kCases[] = {
    {Size(97, 61), Size(131, 53), 0, 0, INTER_CUBIC, "cubic"},
    {Size(100, 80), Size(37, 29), 0, 0, INTER_AREA, "area"},
    // copy: identity, whatever the mode
    {Size(97, 53), Size(97, 53), 0, 0, INTER_NEAREST, "copy"},
    {Size(97, 53), Size(97, 53), 0, 0, INTER_LINEAR, "copy"},
    {Size(97, 53), Size(97, 53), 0, 0, INTER_AREA, "copy"},
    {Size(97, 53), Size(97, 53), 0, 0, INTER_CUBIC, "copy"},
    // Four-mode CPU semantics: cubic and fractional AREA use the existing generic kernels.
    {Size(131, 97), Size(61, 53), 0, 0, INTER_CUBIC, "cubic"},
    {Size(64, 48), Size(200, 150), 0, 0, INTER_CUBIC, "cubic"},
    {Size(1, 40), Size(5, 20), 0, 0, INTER_CUBIC, "cubic"},
    {Size(40, 1), Size(), 1.7, 3.0, INTER_CUBIC, "cubic"},
    {Size(131, 97), Size(61, 53), 0, 0, INTER_AREA, "area"},
    {Size(64, 48), Size(20, 21), 0, 0, INTER_AREA, "area"},
    {Size(64, 48), Size(), 0.7, 0.6, INTER_AREA, "area"},
    // nearest: down, up, exact ratios, one-pixel edges, fx/fy
    {Size(131, 97), Size(61, 53), 0, 0, INTER_NEAREST, "nearest"},
    {Size(64, 48), Size(200, 150), 0, 0, INTER_NEAREST, "nearest"},
    {Size(128, 96), Size(64, 48), 0, 0, INTER_NEAREST, "nearest"},
    {Size(640, 480), Size(1920, 1080), 0, 0, INTER_NEAREST, "nearest"},
    {Size(1, 1), Size(3, 2), 0, 0, INTER_NEAREST, "nearest"},
    {Size(1, 40), Size(5, 20), 0, 0, INTER_NEAREST, "nearest"},
    {Size(40, 1), Size(20, 5), 0, 0, INTER_NEAREST, "nearest"},
    {Size(64, 48), Size(), 1.5, 0.4, INTER_NEAREST, "nearest"},
    // linear: non-integer down, up, exact 2x up, mixed axes, one-pixel edges, fx/fy
    {Size(131, 97), Size(61, 53), 0, 0, INTER_LINEAR, "linear"},
    {Size(64, 48), Size(200, 150), 0, 0, INTER_LINEAR, "linear"},
    {Size(64, 48), Size(128, 96), 0, 0, INTER_LINEAR, "linear"},
    {Size(640, 480), Size(1920, 1080), 0, 0, INTER_LINEAR, "linear"},
    {Size(60, 90), Size(90, 60), 0, 0, INTER_LINEAR, "linear"},
    {Size(1, 1), Size(7, 5), 0, 0, INTER_LINEAR, "linear"},
    {Size(40, 1), Size(60, 3), 0, 0, INTER_LINEAR, "linear"},
    {Size(64, 48), Size(), 0.7, 1.3, INTER_LINEAR, "linear"},
    // AREA with an upscaled axis is the linear path with AREA's coefficient formula
    {Size(64, 48), Size(200, 150), 0, 0, INTER_AREA, "linear"},
    {Size(64, 48), Size(128, 24), 0, 0, INTER_AREA, "linear"},
    // area_fast: exact 2x LINEAR is routed here, integer ratios, anisotropic, odd sources via fx/fy
    {Size(128, 96), Size(64, 48), 0, 0, INTER_LINEAR, "area_fast"},
    {Size(1920, 1080), Size(960, 540), 0, 0, INTER_LINEAR, "area_fast"},
    {Size(128, 96), Size(64, 48), 0, 0, INTER_AREA, "area_fast"},
    {Size(129, 96), Size(43, 32), 0, 0, INTER_AREA, "area_fast"},
    {Size(160, 120), Size(40, 30), 0, 0, INTER_AREA, "area_fast"},
    {Size(640, 480), Size(80, 60), 0, 0, INTER_AREA, "area_fast"},
    {Size(120, 60), Size(40, 30), 0, 0, INTER_AREA, "area_fast"},
    {Size(129, 97), Size(), 0.5, 0.5, INTER_AREA, "area_fast"},
    {Size(129, 97), Size(), 0.5, 0.5, INTER_LINEAR, "area_fast"},
    {Size(131, 134), Size(), 0.25, 0.25, INTER_AREA, "area_fast"},
};

const int kTypes[] = {CV_8UC1, CV_8UC3, CV_8UC4, CV_32FC1, CV_32FC3, CV_32FC4};

}
}
}
#endif
