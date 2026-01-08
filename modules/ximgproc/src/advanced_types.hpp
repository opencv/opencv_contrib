/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __ADVANCED_TYPES_HPP__
#define __ADVANCED_TYPES_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>

/********************* Defines *********************/

#ifndef CV_SQR
#  define CV_SQR(x)  ((x)*(x))
#endif

#ifndef CV_CUBE
#  define CV_CUBE(x)  ((x)*(x)*(x))
#endif

#ifndef CV_INIT_VECTOR
#  define CV_INIT_VECTOR(vname, type, ...) \
    static const type vname##_a[] = __VA_ARGS__; \
    std::vector <type> vname(vname##_a, \
    vname##_a + sizeof(vname##_a) / sizeof(*vname##_a))
#endif

/********************* Types *********************/

/*! fictitious type to highlight that function
 *  can process n-channels arguments */
typedef cv::Mat NChannelsMat;

/********************* Functions *********************/

namespace cv
{
namespace ximgproc
{

template <typename _Tp, typename _Tp2> inline
    cv::Size_<_Tp> operator * (const _Tp2 &x, const cv::Size_<_Tp> &sz)
{
    return cv::Size_<_Tp>(cv::saturate_cast<_Tp>(x*sz.width), cv::saturate_cast<_Tp>(x*sz.height));
}

template <typename _Tp, typename _Tp2> inline
    cv::Size_<_Tp> operator / (const cv::Size_<_Tp> &sz, const _Tp2 &x)
{
    return cv::Size_<_Tp>(cv::saturate_cast<_Tp>(sz.width/x), cv::saturate_cast<_Tp>(sz.height/x));
}

} // cv
}
#endif
#endif /* __ADVANCED_TYPES_HPP__ */