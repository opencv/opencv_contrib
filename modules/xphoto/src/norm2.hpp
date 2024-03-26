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

#ifndef __NORM2_HPP__
#define __NORM2_HPP__

template<bool B, class T = void> struct iftype {};
template<class T> struct iftype<true, T> { typedef T type; }; // enable_if

template<class T, T v> struct int_const { // integral_constant
    static const T value = v;
    typedef T value_type;
    typedef int_const type;
    operator value_type() const { return value; }
    value_type operator()() const { return value; }
};

typedef int_const<bool,true> ttype; // true_type
typedef int_const<bool,false> ftype; // false_type

template <class T, class U> struct same_as : ftype {};
template <class T> struct same_as<T, T> : ttype {};   // is_same


template <typename _Tp> struct is_norm2_type :
    int_const<bool, !same_as<_Tp,   int8_t>::value
                 && !same_as<_Tp,  uint8_t>::value
                 && !same_as<_Tp, uint16_t>::value
                 && !same_as<_Tp, uint32_t>::value>{};

template <typename _Tp, int cn> static inline typename iftype< is_norm2_type<_Tp>::value, _Tp >::
    type norm2(cv::Vec<_Tp, cn> a, cv::Vec<_Tp, cn> b) { return (a - b).dot(a - b); }

template <typename _Tp> static inline typename iftype< is_norm2_type<_Tp>::value, _Tp >::
    type norm2(const _Tp &a, const _Tp &b) { return (a - b)*(a - b); }

#endif /* __NORM2_HPP__ */
