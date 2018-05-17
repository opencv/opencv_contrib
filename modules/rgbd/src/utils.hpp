/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __OPENCV_RGBD_UTILS_HPP__
#define __OPENCV_RGBD_UTILS_HPP__

#include "precomp.hpp"

namespace cv
{
namespace rgbd
{

/** If the input image is of type CV_16UC1 (like the Kinect one), the image is converted to floats, divided
 * by 1000 to get a depth in meters, and the values 0 are converted to std::numeric_limits<float>::quiet_NaN()
 * Otherwise, the image is simply converted to floats
 * @param in the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
 *              (as done with the Microsoft Kinect), it is assumed in meters)
 * @param the desired output depth (floats or double)
 * @param out The rescaled float depth image
 */
template<typename T>
void
rescaleDepthTemplated(const Mat& in, Mat& out);

template<>
inline void
rescaleDepthTemplated<float>(const Mat& in, Mat& out)
{
  rescaleDepth(in, CV_32F, out);
}

template<>
inline void
rescaleDepthTemplated<double>(const Mat& in, Mat& out)
{
  rescaleDepth(in, CV_64F, out);
}

} // namespace rgbd


namespace kinfu {

// Print execution times of each block marked with ScopeTime
#define PRINT_TIME 0

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

/** @brief Camera intrinsics */
struct Intr
{
    /** Reprojects screen point to camera space given z coord. */
    struct Reprojector
    {
        Reprojector() {}
        inline Reprojector(Intr intr)
        {
            fxinv = 1.f/intr.fx, fyinv = 1.f/intr.fy;
            cx = intr.cx, cy = intr.cy;
        }
        template<typename T>
        inline cv::Point3_<T> operator()(cv::Point3_<T> p) const
        {
            T x = p.z * (p.x - cx) * fxinv;
            T y = p.z * (p.y - cy) * fyinv;
            return cv::Point3_<T>(x, y, p.z);
        }

        float fxinv, fyinv, cx, cy;
    };
    /** Projects camera space vector onto screen */
    struct Projector
    {
        inline Projector(Intr intr) : fx(intr.fx), fy(intr.fy), cx(intr.cx), cy(intr.cy) { }
        template<typename T>
        inline cv::Point_<T> operator()(cv::Point3_<T> p) const
        {
            T invz = T(1)/p.z;
            T x = fx*(p.x*invz) + cx;
            T y = fy*(p.y*invz) + cy;
            return cv::Point_<T>(x, y);
        }
        template<typename T>
        inline cv::Point_<T> operator()(cv::Point3_<T> p, cv::Point3_<T>& pixVec) const
        {
            T invz = T(1)/p.z;
            pixVec = cv::Point3_<T>(p.x*invz, p.y*invz, 1);
            T x = fx*pixVec.x + cx;
            T y = fy*pixVec.y + cy;
            return cv::Point_<T>(x, y);
        }
        float fx, fy, cx, cy;
    };
    Intr() : fx(), fy(), cx(), cy() { }
    Intr(float _fx, float _fy, float _cx, float _cy) : fx(_fx), fy(_fy), cx(_cx), cy(_cy) { }
    Intr(cv::Matx33f m) : fx(m(0, 0)), fy(m(1, 1)), cx(m(0, 2)), cy(m(1, 2)) { }
    // scale intrinsics to pyramid level
    inline Intr scale(int pyr) const
    {
        float factor = (1.f /(1 << pyr));
        return Intr(fx*factor, fy*factor, cx*factor, cy*factor);
    }
    inline Reprojector makeReprojector() const { return Reprojector(*this); }
    inline Projector   makeProjector()   const { return Projector(*this);   }

    float fx, fy, cx, cy;
};

} // namespace kinfu

} // namespace cv


#endif

/* End of file. */
