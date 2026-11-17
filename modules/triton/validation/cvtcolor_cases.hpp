// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
#ifndef OPENCV_TRITON_VALIDATION_CVTCOLOR_CASES_HPP
#define OPENCV_TRITON_VALIDATION_CVTCOLOR_CASES_HPP
#include "opencv2/imgproc.hpp"
namespace opencv_test
{
namespace triton_test
{
namespace cvt_cases
{
struct CvtCase
{
    int code;
    const char *name;
    int scn, dcn;
    // CPU conversions preserve depth. Additional 16U and 32F combinations are
    // listed in their respective tables below.
    int depth = CV_8U;
    // 4:2:0 decode takes one CV_8UC1 of (3H/2) x W and produces H x W. The fixture has to build a
    // source of that shape and expect a shorter destination, so the row says so instead of the
    // test guessing from the code number.
    bool yuv420 = false;
};

// Every code the module serves, one row each. A code added to the table without a row here is
// caught by test_cvtcolor.cpp::CoversEverySupportedCode.
const CvtCase kCases[] = {
    {cv::COLOR_BGR2BGRA, "BGR2BGRA", 3, 4},
    {cv::COLOR_BGRA2BGR, "BGRA2BGR", 4, 3},
    {cv::COLOR_BGR2RGBA, "BGR2RGBA", 3, 4},
    {cv::COLOR_RGBA2BGR, "RGBA2BGR", 4, 3},
    {cv::COLOR_BGR2RGB, "BGR2RGB", 3, 3},
    {cv::COLOR_BGRA2RGBA, "BGRA2RGBA", 4, 4},
    {cv::COLOR_BGR2GRAY, "BGR2GRAY", 3, 1},
    {cv::COLOR_RGB2GRAY, "RGB2GRAY", 3, 1},
    {cv::COLOR_BGRA2GRAY, "BGRA2GRAY", 4, 1},
    {cv::COLOR_RGBA2GRAY, "RGBA2GRAY", 4, 1},
    {cv::COLOR_BGR2YCrCb, "BGR2YCrCb", 3, 3},
    {cv::COLOR_RGB2YCrCb, "RGB2YCrCb", 3, 3},
    {cv::COLOR_YCrCb2BGR, "YCrCb2BGR", 3, 3},
    {cv::COLOR_YCrCb2RGB, "YCrCb2RGB", 3, 3},
    {cv::COLOR_BGR2HSV, "BGR2HSV", 3, 3},
    {cv::COLOR_RGB2HSV, "RGB2HSV", 3, 3},
    {cv::COLOR_BGR2HSV_FULL, "BGR2HSV_FULL", 3, 3},
    {cv::COLOR_RGB2HSV_FULL, "RGB2HSV_FULL", 3, 3},
    {cv::COLOR_BGR2YUV, "BGR2YUV", 3, 3},
    {cv::COLOR_RGB2YUV, "RGB2YUV", 3, 3},
    {cv::COLOR_YUV2BGR, "YUV2BGR", 3, 3},
    {cv::COLOR_YUV2RGB, "YUV2RGB", 3, 3},
    // 4:2:2 interleaved: CV_8UC2 in, same size out; only width % 2 == 0 is required.
    {cv::COLOR_YUV2BGR_UYVY, "YUV2BGR_UYVY", 2, 3},
    {cv::COLOR_YUV2RGB_UYVY, "YUV2RGB_UYVY", 2, 3},
    {cv::COLOR_YUV2BGR_YUY2, "YUV2BGR_YUY2", 2, 3},
    {cv::COLOR_YUV2RGB_YUY2, "YUV2RGB_YUY2", 2, 3},
    {cv::COLOR_YUV2BGR_YVYU, "YUV2BGR_YVYU", 2, 3},
    {cv::COLOR_YUV2RGB_YVYU, "YUV2RGB_YVYU", 2, 3},
    // YUV2GRAY_UYVY/YUY2 are extractChannel, not a conversion: they accept ODD widths.
    {cv::COLOR_YUV2GRAY_UYVY, "YUV2GRAY_UYVY", 2, 1},
    {cv::COLOR_YUV2GRAY_YUY2, "YUV2GRAY_YUY2", 2, 1},
    // 4:2:0 semi-planar decode: (3H/2) x W in, H x W out.
    {cv::COLOR_YUV2BGR_NV12, "YUV2BGR_NV12", 1, 3, CV_8U, true},
    {cv::COLOR_YUV2RGB_NV12, "YUV2RGB_NV12", 1, 3, CV_8U, true},
    {cv::COLOR_YUV2BGRA_NV12, "YUV2BGRA_NV12", 1, 4, CV_8U, true},
    {cv::COLOR_YUV2RGBA_NV12, "YUV2RGBA_NV12", 1, 4, CV_8U, true},
    {cv::COLOR_YUV2BGR_NV21, "YUV2BGR_NV21", 1, 3, CV_8U, true},
    {cv::COLOR_YUV2RGB_NV21, "YUV2RGB_NV21", 1, 3, CV_8U, true},
    {cv::COLOR_YUV2BGRA_NV21, "YUV2BGRA_NV21", 1, 4, CV_8U, true},
    {cv::COLOR_YUV2RGBA_NV21, "YUV2RGBA_NV21", 1, 4, CV_8U, true},
    // gray -> color: cv::COLOR_GRAY2RGB is an alias of cv::COLOR_GRAY2BGR (both 8) and
    // cv::COLOR_GRAY2RGBA of cv::COLOR_GRAY2BGRA (both 9), so two rows cover the four names.
    {cv::COLOR_GRAY2BGR, "GRAY2BGR", 1, 3},
    {cv::COLOR_GRAY2BGRA, "GRAY2RGBA", 1, 4},
    // packed 5x5. A "2-channel" source or destination here is one 16-bit word per pixel held in
    // two bytes (CV_8UC2) -- that is what OpenCV 5.x means by a packed 565/555 image, and a
    // 16UC1 Mat of the same pixel count is rejected by the CPU. Both formats, both directions,
    // all four channel combinations and the two gray directions: 20 codes, 20 rows.
    {cv::COLOR_BGR2BGR565, "BGR2BGR565", 3, 2},
    {cv::COLOR_RGB2BGR565, "RGB2BGR565", 3, 2},
    {cv::COLOR_BGR5652BGR, "BGR5652BGR", 2, 3},
    {cv::COLOR_BGR5652RGB, "BGR5652RGB", 2, 3},
    {cv::COLOR_BGRA2BGR565, "BGRA2BGR565", 4, 2},
    {cv::COLOR_RGBA2BGR565, "RGBA2BGR565", 4, 2},
    {cv::COLOR_BGR5652BGRA, "BGR5652BGRA", 2, 4},
    {cv::COLOR_BGR5652RGBA, "BGR5652RGBA", 2, 4},
    {cv::COLOR_GRAY2BGR565, "GRAY2BGR565", 1, 2},
    {cv::COLOR_BGR5652GRAY, "BGR5652GRAY", 2, 1},
    {cv::COLOR_BGR2BGR555, "BGR2BGR555", 3, 2},
    {cv::COLOR_RGB2BGR555, "RGB2BGR555", 3, 2},
    {cv::COLOR_BGR5552BGR, "BGR5552BGR", 2, 3},
    {cv::COLOR_BGR5552RGB, "BGR5552RGB", 2, 3},
    {cv::COLOR_BGRA2BGR555, "BGRA2BGR555", 4, 2},
    {cv::COLOR_RGBA2BGR555, "RGBA2BGR555", 4, 2},
    {cv::COLOR_BGR5552BGRA, "BGR5552BGRA", 2, 4},
    {cv::COLOR_BGR5552RGBA, "BGR5552RGBA", 2, 4},
    {cv::COLOR_GRAY2BGR555, "GRAY2BGR555", 1, 2},
    {cv::COLOR_BGR5552GRAY, "BGR5552GRAY", 2, 1},
    // Bayer bilinear demosaic. One 8U row per served code; the 16U rows live in kCases16U
    // below, because the harness builds its Mat at this row's depth and the depth walk is a
    // separate exercise. Both depths are required by the table, so CoversEverySupportedCode
    // is satisfied by these rows and the 16U test asserts the 16U ones exist.
    {cv::COLOR_BayerBG2BGR, "BayerBG2BGR", 1, 3},
    {cv::COLOR_BayerGB2BGR, "BayerGB2BGR", 1, 3},
    {cv::COLOR_BayerRG2BGR, "BayerRG2BGR", 1, 3},
    {cv::COLOR_BayerGR2BGR, "BayerGR2BGR", 1, 3},
    {cv::COLOR_BayerBG2BGRA, "BayerBG2BGRA", 1, 4},
    {cv::COLOR_BayerBG2GRAY, "BayerBG2GRAY", 1, 1},
    // XYZ and the Lab/Luv families: 18 codes, every one the module serves.
    //
    // Forward sRGB Luv follows the CPU trilinear lookup table, rather than an
    // analytic CIE formula. Forward linear-Luv codes are unsupported because
    // SIMD-body and scalar-tail rounding can differ. Refusal tests cover them.
    {cv::COLOR_BGR2XYZ, "BGR2XYZ", 3, 3},
    {cv::COLOR_RGB2XYZ, "RGB2XYZ", 3, 3},
    {cv::COLOR_XYZ2BGR, "XYZ2BGR", 3, 3},
    {cv::COLOR_XYZ2RGB, "XYZ2RGB", 3, 3},
    {cv::COLOR_BGR2Lab, "BGR2Lab", 3, 3},
    {cv::COLOR_RGB2Lab, "RGB2Lab", 3, 3},
    {cv::COLOR_Lab2BGR, "Lab2BGR", 3, 3},
    {cv::COLOR_Lab2RGB, "Lab2RGB", 3, 3},
    {cv::COLOR_BGR2Luv, "BGR2Luv", 3, 3},
    {cv::COLOR_RGB2Luv, "RGB2Luv", 3, 3},
    {cv::COLOR_Luv2BGR, "Luv2BGR", 3, 3},
    {cv::COLOR_Luv2RGB, "Luv2RGB", 3, 3},
    {cv::COLOR_LBGR2Lab, "LBGR2Lab", 3, 3},
    {cv::COLOR_LRGB2Lab, "LRGB2Lab", 3, 3},
    {cv::COLOR_Lab2LBGR, "Lab2LBGR", 3, 3},
    {cv::COLOR_Lab2LRGB, "Lab2LRGB", 3, 3},
    {cv::COLOR_Luv2LBGR, "Luv2LBGR", 3, 3},
    {cv::COLOR_Luv2LRGB, "Luv2LRGB", 3, 3},
};

// Supported 16U conversions. Keep depth-specific coverage separate from the
// 8U code table; depth coverage tests compare against the planner's support.
const CvtCase kCases16U[] = {
    {cv::COLOR_BayerBG2BGR, "BayerBG2BGR", 1, 3, CV_16U},
    {cv::COLOR_BayerGB2BGR, "BayerGB2BGR", 1, 3, CV_16U},
    {cv::COLOR_BayerRG2BGR, "BayerRG2BGR", 1, 3, CV_16U},
    {cv::COLOR_BayerGR2BGR, "BayerGR2BGR", 1, 3, CV_16U},
    {cv::COLOR_BayerBG2BGRA, "BayerBG2BGRA", 1, 4, CV_16U},
    {cv::COLOR_BayerBG2GRAY, "BayerBG2GRAY", 1, 1, CV_16U},
    // The three families whose CPU path is the same integer table at 16U as at 8U.
    {cv::COLOR_BGR2GRAY, "BGR2GRAY", 3, 1, CV_16U},
    {cv::COLOR_RGB2GRAY, "RGB2GRAY", 3, 1, CV_16U},
    {cv::COLOR_BGRA2GRAY, "BGRA2GRAY", 4, 1, CV_16U},
    {cv::COLOR_RGBA2GRAY, "RGBA2GRAY", 4, 1, CV_16U},
    {cv::COLOR_GRAY2BGR, "GRAY2BGR", 1, 3, CV_16U},
    // cv::COLOR_GRAY2BGRA is an alias of cv::COLOR_GRAY2RGBA (both 9), and the table's canonical name
    // for the code is "GRAY2RGBA" -- the route assertion compares against it, so the label here
    // must be that one. The 8U row above already labels it the same way.
    {cv::COLOR_GRAY2BGRA, "GRAY2RGBA", 1, 4, CV_16U},
    {cv::COLOR_BGR2YCrCb, "BGR2YCrCb", 3, 3, CV_16U},
    {cv::COLOR_RGB2YCrCb, "RGB2YCrCb", 3, 3, CV_16U},
    {cv::COLOR_YCrCb2BGR, "YCrCb2BGR", 3, 3, CV_16U},
    {cv::COLOR_YCrCb2RGB, "YCrCb2RGB", 3, 3, CV_16U},
    {cv::COLOR_BGR2YUV, "BGR2YUV", 3, 3, CV_16U},
    {cv::COLOR_RGB2YUV, "RGB2YUV", 3, 3, CV_16U},
    {cv::COLOR_YUV2BGR, "YUV2BGR", 3, 3, CV_16U},
    {cv::COLOR_YUV2RGB, "YUV2RGB", 3, 3, CV_16U},
};

// Every admitted 32F numeric code. Copies compare exactly; arithmetic uses the
// per-pixel input-channel scale bound. Coverage is checked against the planner table.
const CvtCase kCases32F[] = {
    {cv::COLOR_BGR2BGRA, "BGR2BGRA", 3, 4, CV_32F},
    {cv::COLOR_BGRA2BGR, "BGRA2BGR", 4, 3, CV_32F},
    {cv::COLOR_BGR2RGBA, "BGR2RGBA", 3, 4, CV_32F},
    {cv::COLOR_RGBA2BGR, "RGBA2BGR", 4, 3, CV_32F},
    {cv::COLOR_BGR2RGB, "BGR2RGB", 3, 3, CV_32F},
    {cv::COLOR_BGRA2RGBA, "BGRA2RGBA", 4, 4, CV_32F},
    {cv::COLOR_GRAY2BGR, "GRAY2BGR", 1, 3, CV_32F},
    {cv::COLOR_GRAY2BGRA, "GRAY2BGRA", 1, 4, CV_32F},
    // Float arithmetic permits CPU dispatch and fused-operation rounding
    // differences under the per-pixel bound
    // |got-ref| <= 8*2^-24*max(1,|ref|,max|input channels|).
    {cv::COLOR_BGR2GRAY, "BGR2GRAY", 3, 1, CV_32F},
    {cv::COLOR_RGB2GRAY, "RGB2GRAY", 3, 1, CV_32F},
    {cv::COLOR_BGRA2GRAY, "BGRA2GRAY", 4, 1, CV_32F},
    {cv::COLOR_RGBA2GRAY, "RGBA2GRAY", 4, 1, CV_32F},
    {cv::COLOR_BGR2YCrCb, "BGR2YCrCb", 3, 3, CV_32F},
    {cv::COLOR_RGB2YCrCb, "RGB2YCrCb", 3, 3, CV_32F},
    {cv::COLOR_YCrCb2BGR, "YCrCb2BGR", 3, 3, CV_32F},
    {cv::COLOR_YCrCb2RGB, "YCrCb2RGB", 3, 3, CV_32F},
    {cv::COLOR_BGR2YUV, "BGR2YUV", 3, 3, CV_32F},
    {cv::COLOR_RGB2YUV, "RGB2YUV", 3, 3, CV_32F},
    {cv::COLOR_YUV2BGR, "YUV2BGR", 3, 3, CV_32F},
    {cv::COLOR_YUV2RGB, "YUV2RGB", 3, 3, CV_32F},
};

// Device/perf fixtures use this table as their shape/depth owner. A missing row
// must fail fixture setup rather than guessing a channel count or video layout.
inline const CvtCase *find(int code, int depth)
{
    if (depth == CV_8U)
        for (const auto &row : kCases)
        {
            if (row.code == code)
                return &row;
        }
    if (depth == CV_16U)
        for (const auto &row : kCases16U)
        {
            if (row.code == code)
                return &row;
        }
    if (depth == CV_32F)
        for (const auto &row : kCases32F)
        {
            if (row.code == code)
                return &row;
        }
    return nullptr;
}

inline bool floatArithmetic(int code)
{
    return code == cv::COLOR_BGR2GRAY || code == cv::COLOR_RGB2GRAY ||
           code == cv::COLOR_BGRA2GRAY || code == cv::COLOR_RGBA2GRAY ||
           code == cv::COLOR_BGR2YCrCb || code == cv::COLOR_RGB2YCrCb ||
           code == cv::COLOR_YCrCb2BGR || code == cv::COLOR_YCrCb2RGB ||
           code == cv::COLOR_BGR2YUV || code == cv::COLOR_RGB2YUV || code == cv::COLOR_YUV2BGR ||
           code == cv::COLOR_YUV2RGB;
}

}
}
}
#endif
