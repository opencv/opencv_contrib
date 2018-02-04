// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_OR_VECTOR_HPP_
#define _OPENCV_OR_VECTOR_HPP_

#ifdef __CUDACC__
#define __CV_CUDA_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
#define __CV_CUDA_HOST_DEVICE__
#endif

namespace cv { namespace hfs { namespace orutils {


template <class T>
struct Vector2_ {
    T x, y;
};

template <class T>
struct Vector4_ {
    T x, y, z, w;
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
        if (d == 0) {
            return lhs;
        }
        lhs.x /= d;
        lhs.y /= d;
        return lhs;
    }

    __CV_CUDA_HOST_DEVICE__ friend Vector2<T>&
        operator += (Vector2<T> &lhs, const Vector2<T> &rhs)
    {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        return lhs;
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
