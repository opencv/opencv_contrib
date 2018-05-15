// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE file found in this module's directory

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "opencv2/imgproc.hpp"
#include "opencv2/kinect_fusion/kinfu.hpp"
#include "opencv2/core/hal/intrin.hpp"

// Print execution times of each block marked with ScopeTime
#define PRINT_TIME 0

#if PRINT_TIME
#include <iostream>
#endif

namespace cv {
namespace kinfu {

typedef float depthType;

const float qnan = std::numeric_limits<float>::quiet_NaN();
const cv::Vec3f nan3(qnan, qnan, qnan);
#if CV_SIMD128
const cv::v_float32x4 nanv(qnan, qnan, qnan, qnan);
#endif

inline bool isNaN(cv::Point3f p)
{
    return (cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z));
}

#if CV_SIMD128
static inline bool isNaN(const cv::v_float32x4& p)
{
    return cv::v_check_any(p != p);
}
#endif

struct ScopeTime
{
    ScopeTime(std::string name_, bool _enablePrint = true);
    ~ScopeTime();

#if PRINT_TIME
    static int nested;
    const std::string name;
    const bool enablePrint;
    double start;
#endif
};

} // namespace kinfu
} // namespace cv
#endif
