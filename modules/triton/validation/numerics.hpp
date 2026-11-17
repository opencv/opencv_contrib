// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_TRITON_VALIDATION_NUMERICS_HPP
#define OPENCV_TRITON_VALIDATION_NUMERICS_HPP
#include <algorithm>
#include <cmath>
namespace triton_validation
{
namespace numeric
{
// Resize absolute bounds for inputs with |x| <= 2, using the generic OpenCV
// 5.0.0 CPU algorithm and its matching float64 evaluation. A fixed 766-case
// comparison measured per-mode maxima E and selected these approximately 2E
// limits. E was about 2.474e-7 (LINEAR), 7.004e-7 (CUBIC), and 4.210e-7
// (AREA); NEAREST copies exactly. Printed E values are rounded.
// This finite-domain calibration is not a universal error proof. Changes to the
// reference arithmetic or input domain require revalidation, not larger limits.
inline double resizeF32Atol(int interp)
{
    return interp == 0 ? 0.0 : (interp == 1 ? 4.948e-7 : (interp == 2 ? 1.401e-6 : 8.419e-7));
}
// Warp uses a per-element relative allowance with a hard 1e-4 absolute ceiling.
inline double warpF32Tolerance(double value)
{
    return std::min(5.0e-5 * std::max(std::abs(value), 1.0), 1.0e-4);
}
// inputScale is the largest absolute input channel at the corresponding pixel.
// The optional absolute ceiling applies only to the identified CPU arithmetic
// profile described in doc/cvtcolor.markdown.
inline double cvtF32Tolerance(double reference, double inputScale, bool absoluteCap = false)
{
    const double scaled =
        8 * std::ldexp(1.0, -24) * std::max(1.0, std::max(std::abs(reference), inputScale));
    return absoluteCap ? std::min(scaled, 1.0e-4) : scaled;
}
}
}
#endif
