/*M///////////////////////////////////////////////////////////////////////////////////////
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                              License Agreement
//                    For Open Source Computer Vision Library
//                           (3 - clause BSD License)
//
// Copyright(C) 2000 - 2016, Intel Corporation, all rights reserved.
// Copyright(C) 2009 - 2011, Willow Garage Inc., all rights reserved.
// Copyright(C) 2009 - 2016, NVIDIA Corporation, all rights reserved.
// Copyright(C) 2010 - 2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright(C) 2015 - 2016, OpenCV Foundation, all rights reserved.
// Copyright(C) 2015 - 2016, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met :
//
//      * Redistributions of source code must retain the above copyright notice,
//        this list of conditions and the following disclaimer.
//
//      * Redistributions in binary form must reproduce the above copyright notice,
//        this list of conditions and the following disclaimer in the documentation
//        and / or other materials provided with the distribution.
//
//      * Neither the names of the copyright holders nor the names of the contributors
//        may be used to endorse or promote products derived from this software
//        without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.

// Copyright 2014 Isis Innovation Limited and the authors of InfiniTAM
//
//M*/

#ifndef _OPENCV_OR_VECTOR_HPP_
#define _OPENCV_OR_VECTOR_HPP_
#ifdef __cplusplus

#include "opencv2/core/cuda/common.hpp"

namespace cv { namespace hfs { namespace orutils {


template <class T> struct Vector2_ {
    union {
        struct { T x, y; };
        struct { T width, height; };
        T v[2];
    };
};

template <class T> struct Vector4_ {
    union {
        struct { T x, y, z, w; };
        struct { T r, g, b, a; };
        T v[4];
    };
};

template <class T> class Vector2 : public Vector2_ < T >
{
public:
    __CV_CUDA_HOST_DEVICE__ Vector2() {}

    __CV_CUDA_HOST_DEVICE__ Vector2(const T v0, const T v1)
    { 
        this->x = v0; 
        this->y = v1; 
    }
    
    __CV_CUDA_HOST_DEVICE__ Vector2(const Vector2_<T> &v) 
    { 
        this->x = v.x; 
        this->y = v.y; 
    }

    __CV_CUDA_HOST_DEVICE__ friend Vector2<T> &operator /= (Vector2<T> &lhs, T d) 
    {
        if (d == 0) return lhs; lhs.x /= d; lhs.y /= d; return lhs;
    }

    __CV_CUDA_HOST_DEVICE__ friend Vector2<T>& 
        operator += (Vector2<T> &lhs, const Vector2<T> &rhs) 
    {
        lhs.x += rhs.x; lhs.y += rhs.y;    return lhs;
    }
};
            
template <class T> class Vector4 : public Vector4_ < T >
{
public:
    __CV_CUDA_HOST_DEVICE__ Vector4() {}

    __CV_CUDA_HOST_DEVICE__ Vector4(const T v0, const T v1, const T v2, const T v3) 
    { 
        this->x = v0; 
        this->y = v1; 
        this->z = v2; 
        this->w = v3; 
    }

    __CV_CUDA_HOST_DEVICE__ friend Vector4<T> &operator /= (Vector4<T> &lhs, T d) 
    {
        lhs.x /= d; 
        lhs.y /= d; 
        lhs.z /= d; 
        lhs.w /= d; 
        return lhs;
    }

    __CV_CUDA_HOST_DEVICE__ friend Vector4<T>& 
        operator += (Vector4<T> &lhs, const Vector4<T> &rhs) 
    {
        lhs.x += rhs.x; 
        lhs.y += rhs.y; 
        lhs.z += rhs.z; 
        lhs.w += rhs.w; 
        return lhs;
    }
};


}}}

#endif
#endif
