/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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

#ifndef __OPENCV_BM3D_DENOISING_TRANSFORMS_HPP__
#define __OPENCV_BM3D_DENOISING_TRANSFORMS_HPP__

#include "bm3d_denoising_transforms_haar.hpp"

namespace cv
{
namespace xphoto
{

// Following class contains interface of the tranform domain functions.
template <typename T, typename TT>
class Transform
{
  public:
    // 2D transforms
    typedef void(*Forward2D)(const T *ptr, TT *dst, const int &step, const int blockSize);
    typedef void(*Inverse2D)(TT *src, const int blockSize);

    // 1D transforms
    typedef void(*Forward1D)(BlockMatch<TT, int, TT> *z, const int &n, const unsigned &N);
    typedef void(*Inverse1D)(BlockMatch<TT, int, TT> *z, const int &n, const unsigned &N);

    // Specialized 1D transforms
    typedef void(*Forward1Ds)(BlockMatch<TT, int, TT> *z, const int &n);
    typedef void(*Inverse1Ds)(BlockMatch<TT, int, TT> *z, const int &n);
};

}  // namespace xphoto
}  // namespace cv

#endif