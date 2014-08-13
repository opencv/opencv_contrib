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

/************************ General template *************************/

template <typename Tp> static inline Tp sqr(Tp x) { return x*x; }

template <typename Tp, int cn> static inline Tp sqr( cv::Vec<Tp, cn> x) { return x.dot(x); }

template <typename Tp> static inline Tp norm2(const Tp &a, const Tp &b) { return sqr(a - b); }

template <typename Tp, int cn> static inline
Tp norm2(const cv::Vec <Tp, cn> &a, const cv::Vec<Tp, cn> &b) { return sqr(a - b); }



/******************* uchar, char, ushort, uint *********************/

static inline int norm2(const uchar &a, const uchar &b) { return sqr(int(a) - int(b)); }

template <int cn> static inline
    int norm2(const cv::Vec <uchar, cn> &a, const cv::Vec<uchar, cn> &b)
{
    return sqr( cv::Vec<int, cn>(a) - cv::Vec<int, cn>(b) );
}

static inline int norm2(const char &a, const char &b) { return sqr(int(a) - int(b)); }

template <int cn> static inline
    int norm2(const cv::Vec <char, cn> &a, const cv::Vec<char, cn> &b)
{
    return sqr( cv::Vec<int, cn>(a) - cv::Vec<int, cn>(b) );
}

static inline short norm2(const ushort &a, const ushort &b) { return sqr <short>(short(a) - short(b)); }

template <int cn> static inline
    short norm2(const cv::Vec <ushort, cn> &a, const cv::Vec<ushort, cn> &b)
{
    return sqr( cv::Vec<short, cn>(a) - cv::Vec<short, cn>(b) );
}

static inline int norm2(const uint &a, const uint &b) { return sqr(int(a) - int(b)); }

template <int cn> static inline
    int norm2(const cv::Vec <uint, cn> &a, const cv::Vec<uint, cn> &b)
{
    return sqr( cv::Vec<int, cn>(a) - cv::Vec<int, cn>(b) );
}


#endif /* __NORM2_HPP__ */