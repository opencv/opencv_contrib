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

#ifndef __OPENCV_BM3D_DENOISING_TRANSFORMS_1D_HPP__
#define __OPENCV_BM3D_DENOISING_TRANSFORMS_1D_HPP__

namespace cv
{
namespace xphoto
{

class HaarTransform1D
{
    static void CalculateIndicesN(unsigned *diffIndices, const unsigned &size, const unsigned &N)
    {
        unsigned diffIdx = 1;
        unsigned diffAllIdx = 0;
        for (unsigned i = 1; i <= N; i <<= 1)
        {
            diffAllIdx += (i >> 1);
            for (unsigned j = 0; j < (i >> 1); ++j)
                diffIndices[diffIdx++] = size - (--diffAllIdx);
            diffAllIdx += i;
        }
    }

public:
    /// 1D forward transformations of array of arbitrary size
    template <typename T, typename DT, typename CT>
    inline static void ForwardTransformN(BlockMatch<T, DT, CT> *src, const int &n, const unsigned &N)
    {
        const unsigned size = N + (N << 1) - 2;
        T *dstX = new T[size];

        // Fill dstX with source values
        for (unsigned i = 0; i < N; ++i)
            dstX[i] = src[i][n];

        unsigned idx = 0, dstIdx = N;
        for (unsigned i = N; i > 1; i >>= 1)
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
        unsigned *diffIndices = new unsigned[N];
        CalculateIndicesN(diffIndices, size, N);

        // Fill in destination matrix
        src[0][n] = dstX[size - 2];
        for (unsigned i = 1; i < N; ++i)
            src[i][n] = dstX[diffIndices[i]];

        delete[] dstX;
        delete[] diffIndices;
    }

    /// 1D inverse transformation of array of arbitrary size
    template <typename T, typename DT, typename CT>
    inline static void InverseTransformN(BlockMatch<T, DT, CT> *src, const int &n, const unsigned &N)
    {
        const unsigned dstSize = (N << 1) - 2;
        T *dstX = new T[dstSize];
        T *srcX = new T[N];

        // Fill srcX with source values
        srcX[0] = src[0][n] * 2;
        for (unsigned i = 1; i < N; ++i)
            srcX[i] = src[i][n];

        // Take care of first two elements
        dstX[0] = srcX[0] + srcX[1];
        dstX[1] = srcX[0] - srcX[1];

        unsigned idx = 0, dstIdx = 2;
        for (unsigned i = 4; i < N; i <<= 1)
        {
            for (unsigned j = 0; j < (i >> 1); ++j)
            {
                dstX[dstIdx++] = dstX[idx + j] + srcX[idx + 2 + j];
                dstX[dstIdx++] = dstX[idx + j] - srcX[idx + 2 + j];
            }
            idx += (i >> 1);
        }

        // Handle the last X elements
        dstIdx = 0;
        for (unsigned j = 0; j < (N >> 1); ++j)
        {
            src[dstIdx++][n] = (dstX[idx + j] + srcX[idx + 2 + j]) >> 1;
            src[dstIdx++][n] = (dstX[idx + j] - srcX[idx + 2 + j]) >> 1;
        }

        delete[] srcX;
        delete[] dstX;
    }

    /// 1D forward transformations of fixed array size: 2, 4, 8 and 16

    template <typename T, typename DT, typename CT>
    inline static void ForwardTransform2(BlockMatch<T, DT, CT> *z, const int &n)
    {
        T sum = (z[0][n] + z[1][n] + 1) >> 1;
        T dif = z[0][n] - z[1][n];

        z[0][n] = sum;
        z[1][n] = dif;
    }

    template <typename T, typename DT, typename CT>
    inline static void ForwardTransform4(BlockMatch<T, DT, CT> *z, const int &n)
    {
        T sum0 = (z[0][n] + z[1][n] + 1) >> 1;
        T sum1 = (z[2][n] + z[3][n] + 1) >> 1;
        T dif0 = z[0][n] - z[1][n];
        T dif1 = z[2][n] - z[3][n];

        T sum00 = (sum0 + sum1 + 1) >> 1;
        T dif00 = sum0 - sum1;

        z[0][n] = sum00;
        z[1][n] = dif00;
        z[2][n] = dif0;
        z[3][n] = dif1;
    }

    template <typename T, typename DT, typename CT>
    inline static void ForwardTransform8(BlockMatch<T, DT, CT> *z, const int &n)
    {
        T sum0 = (z[0][n] + z[1][n] + 1) >> 1;
        T sum1 = (z[2][n] + z[3][n] + 1) >> 1;
        T sum2 = (z[4][n] + z[5][n] + 1) >> 1;
        T sum3 = (z[6][n] + z[7][n] + 1) >> 1;
        T dif0 = z[0][n] - z[1][n];
        T dif1 = z[2][n] - z[3][n];
        T dif2 = z[4][n] - z[5][n];
        T dif3 = z[6][n] - z[7][n];

        T sum00 = (sum0 + sum1 + 1) >> 1;
        T sum11 = (sum2 + sum3 + 1) >> 1;
        T dif00 = sum0 - sum1;
        T dif11 = sum2 - sum3;

        T sum000 = (sum00 + sum11 + 1) >> 1;
        T dif000 = sum00 - sum11;

        z[0][n] = sum000;
        z[1][n] = dif000;
        z[2][n] = dif00;
        z[3][n] = dif11;
        z[4][n] = dif0;
        z[5][n] = dif1;
        z[6][n] = dif2;
        z[7][n] = dif3;
    }

    template <typename T, typename DT, typename CT>
    inline static void ForwardTransform16(BlockMatch<T, DT, CT> *z, const int &n)
    {
        T sum0 = (z[0][n] + z[1][n] + 1) >> 1;
        T sum1 = (z[2][n] + z[3][n] + 1) >> 1;
        T sum2 = (z[4][n] + z[5][n] + 1) >> 1;
        T sum3 = (z[6][n] + z[7][n] + 1) >> 1;
        T sum4 = (z[8][n] + z[9][n] + 1) >> 1;
        T sum5 = (z[10][n] + z[11][n] + 1) >> 1;
        T sum6 = (z[12][n] + z[13][n] + 1) >> 1;
        T sum7 = (z[14][n] + z[15][n] + 1) >> 1;
        T dif0 = z[0][n] - z[1][n];
        T dif1 = z[2][n] - z[3][n];
        T dif2 = z[4][n] - z[5][n];
        T dif3 = z[6][n] - z[7][n];
        T dif4 = z[8][n] - z[9][n];
        T dif5 = z[10][n] - z[11][n];
        T dif6 = z[12][n] - z[13][n];
        T dif7 = z[14][n] - z[15][n];

        T sum00 = (sum0 + sum1 + 1) >> 1;
        T sum11 = (sum2 + sum3 + 1) >> 1;
        T sum22 = (sum4 + sum5 + 1) >> 1;
        T sum33 = (sum6 + sum7 + 1) >> 1;
        T dif00 = sum0 - sum1;
        T dif11 = sum2 - sum3;
        T dif22 = sum4 - sum5;
        T dif33 = sum6 - sum7;

        T sum000 = (sum00 + sum11 + 1) >> 1;
        T sum111 = (sum22 + sum33 + 1) >> 1;
        T dif000 = sum00 - sum11;
        T dif111 = sum22 - sum33;

        T sum0000 = (sum000 + sum111 + 1) >> 1;
        T dif0000 = dif000 - dif111;

        z[0][n] = sum0000;
        z[1][n] = dif0000;
        z[2][n] = dif000;
        z[3][n] = dif111;
        z[4][n] = dif00;
        z[5][n] = dif11;
        z[6][n] = dif22;
        z[7][n] = dif33;
        z[8][n] = dif0;
        z[9][n] = dif1;
        z[10][n] = dif2;
        z[11][n] = dif3;
        z[12][n] = dif4;
        z[13][n] = dif5;
        z[14][n] = dif6;
        z[15][n] = dif7;
    }

    /// 1D inverse transformations of fixed array size: 2, 4, 8 and 16

    template <typename T, typename DT, typename CT>
    inline static void InverseTransform2(BlockMatch<T, DT, CT> *src, const int &n)
    {
        T src0 = src[0][n] * 2;
        T src1 = src[1][n];

        src[0][n] = (src0 + src1) >> 1;
        src[1][n] = (src0 - src1) >> 1;
    }

    template <typename T, typename DT, typename CT>
    inline static void InverseTransform4(BlockMatch<T, DT, CT> *src, const int &n)
    {
        T src0 = src[0][n] * 2;
        T src1 = src[1][n];
        T src2 = src[2][n];
        T src3 = src[3][n];

        T sum0 = src0 + src1;
        T dif0 = src0 - src1;

        src[0][n] = (sum0 + src2) >> 1;
        src[1][n] = (sum0 - src2) >> 1;
        src[2][n] = (dif0 + src3) >> 1;
        src[3][n] = (dif0 - src3) >> 1;
    }

    template <typename T, typename DT, typename CT>
    inline static void InverseTransform8(BlockMatch<T, DT, CT> *src, const int &n)
    {
        T src0 = src[0][n] * 2;
        T src1 = src[1][n];
        T src2 = src[2][n];
        T src3 = src[3][n];
        T src4 = src[4][n];
        T src5 = src[5][n];
        T src6 = src[6][n];
        T src7 = src[7][n];

        T sum0 = src0 + src1;
        T dif0 = src0 - src1;

        T sum00 = sum0 + src2;
        T dif00 = sum0 - src2;
        T sum11 = dif0 + src3;
        T dif11 = dif0 - src3;

        src[0][n] = (sum00 + src4) >> 1;
        src[1][n] = (sum00 - src4) >> 1;
        src[2][n] = (dif00 + src5) >> 1;
        src[3][n] = (dif00 - src5) >> 1;
        src[4][n] = (sum11 + src6) >> 1;
        src[5][n] = (sum11 - src6) >> 1;
        src[6][n] = (dif11 + src7) >> 1;
        src[7][n] = (dif11 - src7) >> 1;
    }

    template <typename T, typename DT, typename CT>
    inline static void InverseTransform16(BlockMatch<T, DT, CT> *src, const int &n)
    {
        T src0 = src[0][n] * 2;
        T src1 = src[1][n];
        T src2 = src[2][n];
        T src3 = src[3][n];
        T src4 = src[4][n];
        T src5 = src[5][n];
        T src6 = src[6][n];
        T src7 = src[7][n];
        T src8 = src[8][n];
        T src9 = src[9][n];
        T src10 = src[10][n];
        T src11 = src[11][n];
        T src12 = src[12][n];
        T src13 = src[13][n];
        T src14 = src[14][n];
        T src15 = src[15][n];

        T sum0 = src0 + src1;
        T dif0 = src0 - src1;

        T sum00 = sum0 + src2;
        T dif00 = sum0 - src2;
        T sum11 = dif0 + src3;
        T dif11 = dif0 - src3;

        T sum000 = sum00 + src4;
        T dif000 = sum00 - src4;
        T sum111 = dif00 + src5;
        T dif111 = dif00 - src5;
        T sum222 = sum11 + src6;
        T dif222 = sum11 - src6;
        T sum333 = dif11 + src7;
        T dif333 = dif11 - src7;

        src[0][n] = (sum000 + src8) >> 1;
        src[1][n] = (sum000 - src8) >> 1;
        src[2][n] = (dif000 + src9) >> 1;
        src[3][n] = (dif000 - src9) >> 1;
        src[4][n] = (sum111 + src10) >> 1;
        src[5][n] = (sum111 - src10) >> 1;
        src[6][n] = (dif111 + src11) >> 1;
        src[7][n] = (dif111 - src11) >> 1;
        src[8][n] = (sum222 + src12) >> 1;
        src[9][n] = (sum222 - src12) >> 1;
        src[10][n] = (dif222 + src13) >> 1;
        src[11][n] = (dif222 - src13) >> 1;
        src[12][n] = (sum333 + src14) >> 1;
        src[13][n] = (sum333 - src14) >> 1;
        src[14][n] = (dif333 + src15) >> 1;
        src[15][n] = (dif333 - src15) >> 1;
    }
};

}  // namespace xphoto
}  // namespace cv

#endif