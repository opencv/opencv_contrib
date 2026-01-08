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

#ifndef __OPENCV_BM3D_DENOISING_TRANSFORMS_2D_HPP__
#define __OPENCV_BM3D_DENOISING_TRANSFORMS_2D_HPP__

namespace cv
{
namespace xphoto
{

class HaarTransform2D
{
    template <int X>
    static void CalculateIndices(unsigned *diffIndices, const unsigned &size)
    {
        unsigned diffIdx = 1;
        unsigned diffAllIdx = 0;
        for (unsigned i = 1; i <= X; i <<= 1)
        {
            diffAllIdx += (i >> 1);
            for (unsigned j = 0; j < (i >> 1); ++j)
                diffIndices[diffIdx++] = size - (--diffAllIdx);
            diffAllIdx += i;
        }
    }

public:
    /// Transforms for 2D block of arbitrary size
    template <typename T, typename TT, int X, int N>
    inline static void ForwardTransformX(const T *src, TT *dst, const int &step)
    {
        const unsigned size = X + (X << 1) - 2;
        TT dstX[size];

        // Fill dstX with source values
        for (unsigned i = 0; i < X; ++i)
            dstX[i] = *(src + i * step);

        unsigned idx = 0, dstIdx = X;
        for (unsigned i = X; i > 1; i >>= 1)
        {
            // Get sums
            for (unsigned j = 0; j < (i >> 1); ++j)
                dstX[dstIdx++] = (dstX[idx + 2 * j] + dstX[idx + j * 2 + 1] + 1) >> 1;

            // Get diffs
            for (unsigned j = 0; j < (i >> 1); ++j)
                dstX[dstIdx++] = dstX[idx + 2 * j] - dstX[idx + j * 2 + 1];

            idx = dstIdx - i;
        }

        // Calculate indices in the destination matrix.
        unsigned diffIndices[X];
        CalculateIndices<X>(diffIndices, size);

        // Fill in destination matrix
        dst[0] = dstX[size - 2];
        for (int i = 1; i < X; ++i)
            dst[i * N] = dstX[diffIndices[i]];
    }

    template <typename T, typename TT, int X>
    inline static void ForwardTransformXxX(const T *ptr, TT *dst, const int &step, const int /*blockSize*/)
    {
        TT temp[X * X];

        // Transform columns first
        for (unsigned i = 0; i < X; ++i)
            ForwardTransformX<T, TT, X, X>(ptr + i, temp + i, step);

        // Then transform rows
        for (unsigned i = 0; i < X; ++i)
            ForwardTransformX<TT, TT, X, 1>(temp + i * X, dst + i * X, 1);
    }

    template <typename T, int X, int N>
    inline static void InverseTransformX(T *src, T *dst)
    {
        const unsigned dstSize = (X << 1) - 2;
        T dstX[dstSize];
        T srcX[X];

        // Fill srcX with source values
        srcX[0] = src[0] * 2;
        for (int i = 1; i < X; ++i)
            srcX[i] = src[i * N];

        // Take care of first two elements
        dstX[0] = srcX[0] + srcX[1];
        dstX[1] = srcX[0] - srcX[1];

        unsigned idx = 0, dstIdx = 2;
        for (int i = 4; i < X; i <<= 1)
        {
            for (int j = 0; j < (i >> 1); ++j)
            {
                dstX[dstIdx++] = dstX[idx + j] + srcX[idx + 2 + j];
                dstX[dstIdx++] = dstX[idx + j] - srcX[idx + 2 + j];
            }
            idx += (i >> 1);
        }

        // Handle the last X elements
        dstIdx = 0;
        for (int j = 0; j < (X >> 1); ++j)
        {
            dst[dstIdx++ * N] = (dstX[idx + j] + srcX[idx + 2 + j]) >> 1;
            dst[dstIdx++ * N] = (dstX[idx + j] - srcX[idx + 2 + j]) >> 1;
        }
    }

    template <typename T, int X>
    inline static void InverseTransformXxX(T *src, const int /*blockSize*/)
    {
        T temp[X * X];

        // Invert columns first
        for (int i = 0; i < X; ++i)
            InverseTransformX<T, X, X>(src + i, temp + i);

        // Then invert rows
        for (int i = 0; i < X; ++i)
            InverseTransformX<T, X, 1>(temp + i * X, src + i * X);
    }

    // Same as above but X and N are arguments, not template parameters.

    static void CalculateIndices(int *diffIndices, const int &size, const int &X)
    {
        int diffIdx = 1;
        int diffAllIdx = 0;
        for (int i = 1; i <= X; i <<= 1)
        {
            diffAllIdx += (i >> 1);
            for (int j = 0; j < (i >> 1); ++j)
                diffIndices[diffIdx++] = size - (--diffAllIdx);
            diffAllIdx += i;
        }
    }

    template <typename T, typename TT>
    inline static void ForwardTransformX(const T *src, TT *dst, const int &step, const int &X, const int &N)
    {
        const int size = X + (X << 1) - 2;
        TT *dstX = new TT[size];

        // Fill dstX with source values
        for (int i = 0; i < X; ++i)
            dstX[i] = *(src + i * step);

        int idx = 0, dstIdx = X;
        for (int i = X; i > 1; i >>= 1)
        {
            // Get sums
            for (int j = 0; j < (i >> 1); ++j)
                dstX[dstIdx++] = (dstX[idx + 2 * j] + dstX[idx + j * 2 + 1] + 1) >> 1;

            // Get diffs
            for (int j = 0; j < (i >> 1); ++j)
                dstX[dstIdx++] = dstX[idx + 2 * j] - dstX[idx + j * 2 + 1];

            idx = dstIdx - i;
        }

        // Calculate indices in the destination matrix.
        int *diffIndices = new int[X];
        CalculateIndices(diffIndices, size, X);

        // Fill in destination matrix
        dst[0] = dstX[size - 2];
        for (int i = 1; i < X; ++i)
            dst[i * N] = dstX[diffIndices[i]];

        delete[] diffIndices;
        delete[] dstX;
    }

    template <typename T, typename TT>
    inline static void ForwardTransformXxX(const T *ptr, TT *dst, const int &step, const int X)
    {
        TT *temp = new TT[X * X];

        // Transform columns first
        for (int i = 0; i < X; ++i)
            ForwardTransformX<T, TT>(ptr + i, temp + i, step, X, X);

        // Then transform rows
        for (int i = 0; i < X; ++i)
            ForwardTransformX<TT, TT>(temp + i * X, dst + i * X, 1, X, 1);

        delete[] temp;
    }

    template <typename T>
    inline static void InverseTransformX(T *src, T *dst, const int &X, const int &N)
    {
        const unsigned dstSize = (X << 1) - 2;
        T *dstX = new T[dstSize];
        T *srcX = new T[X];

        // Fill srcX with source values
        srcX[0] = src[0] * 2;
        for (int i = 1; i < X; ++i)
            srcX[i] = src[i * N];

        // Take care of first two elements
        dstX[0] = srcX[0] + srcX[1];
        dstX[1] = srcX[0] - srcX[1];

        unsigned idx = 0, dstIdx = 2;
        for (int i = 4; i < X; i <<= 1)
        {
            for (int j = 0; j < (i >> 1); ++j)
            {
                dstX[dstIdx++] = dstX[idx + j] + srcX[idx + 2 + j];
                dstX[dstIdx++] = dstX[idx + j] - srcX[idx + 2 + j];
            }
            idx += (i >> 1);
        }

        // Handle the last X elements
        dstIdx = 0;
        for (int j = 0; j < (X >> 1); ++j)
        {
            dst[dstIdx++ * N] = (dstX[idx + j] + srcX[idx + 2 + j]) >> 1;
            dst[dstIdx++ * N] = (dstX[idx + j] - srcX[idx + 2 + j]) >> 1;
        }

        delete[] dstX;
        delete[] srcX;
    }

    template <typename T>
    inline static void InverseTransformXxX(T *src, const int X)
    {
        T *temp = new T[X * X];

        // Invert columns first
        for (int i = 0; i < X; ++i)
            InverseTransformX<T>(src + i, temp + i, X, X);

        // Then invert rows
        for (int i = 0; i < X; ++i)
            InverseTransformX<T>(temp + i * X, src + i * X, X, 1);

        delete[] temp;
    }

    /// Transforms for 2x2 2D block

    template <typename T, typename TT, int N>
    inline static void ForwardTransform2(const T *src, TT *dst, const int &step)
    {
        const T *src0 = src;
        const T *src1 = src + 1 * step;

        dst[0 * N] = (*src0 + *src1 + 1) >> 1;
        dst[1 * N] = *src0 - *src1;
    }

    template <typename T, typename TT>
    inline static void ForwardTransform2x2(const T *ptr, TT *dst, const int &step, const int /*blockSize*/)
    {
        TT temp[4];

        // Transform columns first
        for (int i = 0; i < 2; ++i)
            ForwardTransform2<T, TT, 2>(ptr + i, temp + i, step);

        // Then transform rows
        for (int i = 0; i < 2; ++i)
            ForwardTransform2<TT, TT, 1>(temp + i * 2, dst + i * 2, 1);
    }

    template <typename TT, int N>
    inline static void InverseTransform2(TT *src, TT *dst)
    {
        TT src0 = src[0 * N] * 2;
        TT src1 = src[1 * N];

        dst[0 * N] = (src0 + src1) >> 1;
        dst[1 * N] = (src0 - src1) >> 1;
    }

    template <typename T>
    inline static void InverseTransform2x2(T *src, const int /*blockSize*/)
    {
        T temp[4];

        // Invert columns first
        for (int i = 0; i < 2; ++i)
            InverseTransform2<T, 2>(src + i, temp + i);

        // Then invert rows
        for (int i = 0; i < 2; ++i)
            InverseTransform2<T, 1>(temp + i * 2, src + i * 2);
    }

    /// Transforms for 4x4 2D block

    template <typename T, typename TT, int N>
    inline static void ForwardTransform4(const T *src, TT *dst, const int &step)
    {
        const T *src0 = src;
        const T *src1 = src + 1 * step;
        const T *src2 = src + 2 * step;
        const T *src3 = src + 3 * step;

        TT sum0 = (*src0 + *src1 + 1) >> 1;
        TT sum1 = (*src2 + *src3 + 1) >> 1;
        TT dif0 = *src0 - *src1;
        TT dif1 = *src2 - *src3;

        TT sum00 = (sum0 + sum1 + 1) >> 1;
        TT dif00 = sum0 - sum1;

        dst[0 * N] = sum00;
        dst[1 * N] = dif00;
        dst[2 * N] = dif0;
        dst[3 * N] = dif1;
    }

    template <typename T, typename TT>
    inline static void ForwardTransform4x4(const T *ptr, TT *dst, const int &step, const int /*blockSize*/)
    {
        TT temp[16];

        // Transform columns first
        for (int i = 0; i < 4; ++i)
            ForwardTransform4<T, TT, 4>(ptr + i, temp + i, step);

        // Then transform rows
        for (int i = 0; i < 4; ++i)
            ForwardTransform4<TT, TT, 1>(temp + i * 4, dst + i * 4, 1);
    }

    template <typename TT, int N>
    inline static void InverseTransform4(TT *src, TT *dst)
    {
        TT src0 = src[0 * N] * 2;
        TT src1 = src[1 * N];
        TT src2 = src[2 * N];
        TT src3 = src[3 * N];

        TT sum0 = src0 + src1;
        TT dif0 = src0 - src1;

        dst[0 * N] = (sum0 + src2) >> 1;
        dst[1 * N] = (sum0 - src2) >> 1;
        dst[2 * N] = (dif0 + src3) >> 1;
        dst[3 * N] = (dif0 - src3) >> 1;
    }

    template <typename T>
    inline static void InverseTransform4x4(T *src, const int /*blockSize*/)
    {
        T temp[16];

        // Invert columns first
        for (int i = 0; i < 4; ++i)
            InverseTransform4<T, 4>(src + i, temp + i);

        // Then invert rows
        for (int i = 0; i < 4; ++i)
            InverseTransform4<T, 1>(temp + i * 4, src + i * 4);
    }

    /// Transforms for 8x8 2D block

    template <typename T, typename TT, int N>
    inline static void ForwardTransform8(const T *src, TT *dst, const int &step)
    {
        const T *src0 = src;
        const T *src1 = src + 1 * step;
        const T *src2 = src + 2 * step;
        const T *src3 = src + 3 * step;
        const T *src4 = src + 4 * step;
        const T *src5 = src + 5 * step;
        const T *src6 = src + 6 * step;
        const T *src7 = src + 7 * step;

        TT sum0 = (*src0 + *src1 + 1) >> 1;
        TT sum1 = (*src2 + *src3 + 1) >> 1;
        TT sum2 = (*src4 + *src5 + 1) >> 1;
        TT sum3 = (*src6 + *src7 + 1) >> 1;
        TT dif0 = *src0 - *src1;
        TT dif1 = *src2 - *src3;
        TT dif2 = *src4 - *src5;
        TT dif3 = *src6 - *src7;

        TT sum00 = (sum0 + sum1 + 1) >> 1;
        TT sum11 = (sum2 + sum3 + 1) >> 1;
        TT dif00 = sum0 - sum1;
        TT dif11 = sum2 - sum3;

        TT sum000 = (sum00 + sum11 + 1) >> 1;
        TT dif000 = sum00 - sum11;

        dst[0 * N] = sum000;
        dst[1 * N] = dif000;
        dst[2 * N] = dif00;
        dst[3 * N] = dif11;
        dst[4 * N] = dif0;
        dst[5 * N] = dif1;
        dst[6 * N] = dif2;
        dst[7 * N] = dif3;
    }

    template <typename T, typename TT>
    inline static void ForwardTransform8x8(const T *ptr, TT *dst, const int &step, const int /*blockSize*/)
    {
        TT temp[64];

        // Transform columns first
        for (int i = 0; i < 8; ++i)
            ForwardTransform8<T, TT, 8>(ptr + i, temp + i, step);

        // Then transform rows
        for (int i = 0; i < 8; ++i)
            ForwardTransform8<TT, TT, 1>(temp + i * 8, dst + i * 8, 1);
    }

    template <typename T, int N>
    inline static void InverseTransform8(T *src, T *dst)
    {
        T src0 = src[0] * 2;
        T src1 = src[1 * N];
        T src2 = src[2 * N];
        T src3 = src[3 * N];
        T src4 = src[4 * N];
        T src5 = src[5 * N];
        T src6 = src[6 * N];
        T src7 = src[7 * N];

        T sum0 = src0 + src1;
        T dif0 = src0 - src1;

        T sum00 = sum0 + src2;
        T dif00 = sum0 - src2;
        T sum11 = dif0 + src3;
        T dif11 = dif0 - src3;

        dst[0 * N] = (sum00 + src4) >> 1;
        dst[1 * N] = (sum00 - src4) >> 1;
        dst[2 * N] = (dif00 + src5) >> 1;
        dst[3 * N] = (dif00 - src5) >> 1;
        dst[4 * N] = (sum11 + src6) >> 1;
        dst[5 * N] = (sum11 - src6) >> 1;
        dst[6 * N] = (dif11 + src7) >> 1;
        dst[7 * N] = (dif11 - src7) >> 1;
    }

    template <typename T>
    inline static void InverseTransform8x8(T *src, const int /*blockSize*/)
    {
        T temp[64];

        // Invert columns first
        for (int i = 0; i < 8; ++i)
            InverseTransform8<T, 8>(src + i, temp + i);

        // Then invert rows
        for (int i = 0; i < 8; ++i)
            InverseTransform8<T, 1>(temp + i * 8, src + i * 8);
    }
};

}  // namespace xphoto
}  // namespace cv

#endif