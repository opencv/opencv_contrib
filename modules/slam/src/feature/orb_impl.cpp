/*******************************************************************************

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2009, Willow Garage Inc., all rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*******************************************************************************/

#include "feature/orb_impl.hpp"
#include "feature/orb_point_pairs.hpp"
#include "util/trigonometric.hpp"

#ifdef USE_SSE_ORB
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif // USE_SSE_ORB

namespace cv::slam {
namespace feature {

orb_impl::orb_impl() {
    // Preparate  for computation of orientation
    u_max_.resize(fast_half_patch_size_ + 1);
    const unsigned int vmax = std::floor(fast_half_patch_size_ * std::sqrt(2.0) / 2 + 1);
    const unsigned int vmin = std::ceil(fast_half_patch_size_ * std::sqrt(2.0) / 2);
    for (unsigned int v = 0; v <= vmax; ++v) {
        u_max_.at(v) = std::round(std::sqrt(fast_half_patch_size_ * fast_half_patch_size_ - v * v));
    }
    for (unsigned int v = fast_half_patch_size_, v0 = 0; vmin <= v; --v) {
        while (u_max_.at(v0) == u_max_.at(v0 + 1)) {
            ++v0;
        }
        u_max_.at(v) = v0;
        ++v0;
    }
}

float orb_impl::ic_angle(const cv::Mat& image, const cv::Point2f& point) const {
    int m_01 = 0, m_10 = 0;

    const uchar* const center = &image.at<uchar>(cvRound(point.y), cvRound(point.x));

    for (int u = -fast_half_patch_size_; u <= fast_half_patch_size_; ++u) {
        m_10 += u * center[u];
    }

    const auto step = static_cast<int>(image.step1());
    for (int v = 1; v <= fast_half_patch_size_; ++v) {
        int v_sum = 0;
        const int d = u_max_.at(v);
        for (int u = -d; u <= d; ++u) {
            const int val_plus = center[u + v * step];
            const int val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return cv::fastAtan2(m_01, m_10);
}

void orb_impl::compute_orb_descriptor(const cv::KeyPoint& keypt, const cv::Mat& image, uchar* desc) const {
    const float angle = keypt.angle * M_PI / 180.0;
    const float cos_angle = util::cos(angle);
    const float sin_angle = util::sin(angle);

    const uchar* const center = &image.at<uchar>(cvRound(keypt.pt.y), cvRound(keypt.pt.x));
    const auto step = static_cast<int>(image.step);

#ifdef USE_SSE_ORB
#if !((defined _MSC_VER && defined _M_X64)                            \
      || (defined __GNUC__ && defined __x86_64__ && defined __SSE3__) \
      || CV_SSE3)
#error "The processor is not compatible with SSE. Please configure the CMake with -DUSE_SSE_ORB=OFF."
#endif

    const __m128 _trig1 = _mm_set_ps(cos_angle, sin_angle, cos_angle, sin_angle);
    const __m128 _trig2 = _mm_set_ps(-sin_angle, cos_angle, -sin_angle, cos_angle);
    __m128 _point_pairs;
    __m128 _mul1;
    __m128 _mul2;
    __m128 _vs;
    __m128i _vi;
    alignas(16) int32_t ii[4];

#define COMPARE_ORB_POINTS(shift)                          \
    (_point_pairs = _mm_load_ps(orb_point_pairs + shift),  \
     _mul1 = _mm_mul_ps(_point_pairs, _trig1),             \
     _mul2 = _mm_mul_ps(_point_pairs, _trig2),             \
     _vs = _mm_hadd_ps(_mul1, _mul2),                      \
     _vi = _mm_cvtps_epi32(_vs),                           \
     _mm_store_si128(reinterpret_cast<__m128i*>(ii), _vi), \
     center[ii[0] * step + ii[2]] < center[ii[1] * step + ii[3]])

#else

#define GET_VALUE(shift)                                                                                        \
    (center[cvRound(*(orb_point_pairs + shift) * sin_angle + *(orb_point_pairs + shift + 1) * cos_angle) * step \
            + cvRound(*(orb_point_pairs + shift) * cos_angle - *(orb_point_pairs + shift + 1) * sin_angle)])

#define COMPARE_ORB_POINTS(shift) \
    (GET_VALUE(shift) < GET_VALUE(shift + 2))

#endif

    // interval: (X, Y) x 2 points x 8 pairs = 32
    static constexpr unsigned interval = 32;

    for (unsigned int i = 0; i < orb_point_pairs_size / interval; ++i) {
        int32_t val = COMPARE_ORB_POINTS(i * interval);
        val |= COMPARE_ORB_POINTS(i * interval + 4) << 1;
        val |= COMPARE_ORB_POINTS(i * interval + 8) << 2;
        val |= COMPARE_ORB_POINTS(i * interval + 12) << 3;
        val |= COMPARE_ORB_POINTS(i * interval + 16) << 4;
        val |= COMPARE_ORB_POINTS(i * interval + 20) << 5;
        val |= COMPARE_ORB_POINTS(i * interval + 24) << 6;
        val |= COMPARE_ORB_POINTS(i * interval + 28) << 7;
        desc[i] = static_cast<uchar>(val);
    }

#undef GET_VALUE
#undef COMPARE_ORB_POINTS
}
} // namespace feature
} // namespace cv::slam
