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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#if !defined CUDA_DISABLER
// #define __CUDACC__ 110700
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"

namespace cv { namespace cuda { namespace device {

    __device__ __forceinline__ int clamp(int x, int lo, int hi)
    {
        return x < lo ? lo : hi < x ? hi : x;
    }

    template <typename T>
    __device__ __forceinline__ T* ptr(PtrStepb const& src, int y)
    { return reinterpret_cast<T*>(src.data + y * src.step); }

    template <typename T>
    __device__ __forceinline__ T& at(PtrStepb const& src, int y, int x)
    { return ptr<T>(src, y)[x]; }

    struct LinearCoeff
    {
        enum { ksize = 2 };

        LinearCoeff(float) {}

        __device__ __forceinline__ float at(float x) const
        { return __saturatef(1.f - ::fabsf(x)); }
    };

    struct CubicCoeff
    {
        enum { ksize = 4 };

        float A, A2, A3;

        CubicCoeff(float a) : A(a), A2(a + 2), A3(a + 3) {}

        __device__ __forceinline__ float at(float x) const
        {
            x = ::fabsf(x);
            if (x <= 1)
                x = (A2 * x - A3) * x * x + 1;
            else if (x <= 2)
                x = A * (((x - 5) * x + 8) * x - 4);
            else
                x = 0;
            return x;
        }
    };

    //==================== sampler ====================//

    struct SamplerBase
    {
        PtrStepb src;
        PtrStepSzb dst;
        int row1, col1;

        SamplerBase(PtrStepSzb const& S, PtrStepSzb const& D)
            : src(S), dst(D), row1(S.rows - 1), col1(S.cols - 1)
        {}
    };

    template <typename Coeff>
    struct AntiBase : public SamplerBase
    {
        static_assert(Coeff::ksize % 2 == 0, "");

        float xscale, yscale;
        int xstart, xend, ystart, yend;
        Coeff coeff;

        AntiBase(PtrStepSzb const& S, PtrStepSzb const& D,
            Point2f const& scale, float A)
            : SamplerBase(S, D), coeff(A)
        {
            int const khalf = Coeff::ksize / 2;
            xscale = std::min(scale.x, 1.f);
            yscale = std::min(scale.y, 1.f);
            xstart = cvFloor(-khalf / xscale) + 1;
            xend = 2 - xstart;
            ystart = cvFloor(-khalf / yscale) + 1;
            yend = 2 - ystart;
        }
    };

    ////////// nearest neighbor //////////

    template <typename T>
    struct NearestVec : public SamplerBase
    {
        using SamplerBase::SamplerBase;

        __device__ void to(int sx, int sy, int dx, int dy) const
        {
            sx = clamp(sx, 0, col1);
            sy = clamp(sy, 0, row1);
            at<T>(dst, dy, dx) = at<T>(src, sy, sx);
        }
    };

    struct NearestSize : public SamplerBase
    {
        size_t esz;

        NearestSize(PtrStepSzb const& S, PtrStepSzb const& D, size_t sz)
            : SamplerBase(S, D), esz(sz)
        {}

        __device__ void to(int sx, int sy, int dx, int dy) const
        {
            sx = clamp(sx, 0, col1);
            sy = clamp(sy, 0, row1);
            uchar const* S = ptr<uchar>(src, sy) + sx * esz;
            uchar      * D = ptr<uchar>(dst, dy) + dx * esz;
            for (size_t i = 0; i < esz; ++i)
                D[i] = S[i];
        }
    };

    ////////// anti-alias brute force //////////
    // because we can not allocate temporary memory to store coeffs and offsets

    template <typename T1, typename W1, int cn, typename Coeff>
    struct AntiVec : public AntiBase<Coeff>
    {
        using AntiBase<Coeff>::AntiBase;
        using T = typename TypeVec<T1, cn>::vec_type;
        using W = typename TypeVec<W1, cn>::vec_type;

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int ix = __float2int_rd(fx), iy = __float2int_rd(fy);
            float rx = fx - ix, ry = fy - iy;
            W1 weight = 0;
            W sumval = VecTraits<W>::all(0);
            for (int h = this->ystart; h < this->yend; ++h)
            {
                W1 wline = 0;
                W sline = VecTraits<W>::all(0);
                int sy = clamp(iy + h, 0, this->row1);
                T const* S = ptr<T>(this->src, sy);
                for (int w = this->xstart; w < this->xend; ++w)
                {
                    int sx = clamp(ix + w, 0, this->col1);
                    W1 t = this->coeff.at((w - rx) * this->xscale);
                    wline += t;
                    sline += t * saturate_cast<W>(S[sx]);
                }
                W1 u = this->coeff.at((h - ry) * this->yscale);
                weight += u * wline;
                sumval += u * sline;
            }
            at<T>(this->dst, dy, dx) = saturate_cast<T>(sumval / weight);
        }
    };

    template <typename T, typename W, typename Coeff>
    struct AntiCn : public AntiBase<Coeff>
    {
        int cn;

        AntiCn(PtrStepSzb const& S, PtrStepSzb const& D,
            Point2f const& scale, float A, int _cn)
            : AntiBase<Coeff>(S, D, scale, A), cn(_cn)
        {}

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int ix = __float2int_rd(fx), iy = __float2int_rd(fy);
            float rx = fx - ix, ry = fy - iy;
            W weight = 0, sumval = 0;
            T* D = ptr<T>(this->dst, dy) + dx * cn;
            for (int h = this->ystart; h < this->yend; ++h)
            {
                W wline = 0, sline = 0;
                int sy = clamp(iy + h, 0, this->row1);
                T const* S = ptr<T>(this->src, sy);
                for (int w = this->xstart; w < this->xend; ++w)
                {
                    int sx = clamp(ix + w, 0, this->col1) * cn;
                    W t = this->coeff.at((w - rx) * this->xscale);
                    wline += t;
                    sline += t * S[sx];
                }
                W u = this->coeff.at((h - ry) * this->yscale);
                weight += u * wline;
                sumval += u * sline;
            }
            D[0] = saturate_cast<T>(sumval / weight);
            for (int i = 1; i < cn; ++i)
            {
                sumval = 0;
                for (int h = this->ystart; h < this->yend; ++h)
                {
                    W sline = 0;
                    int sy = clamp(iy + h, 0, this->row1);
                    T const* S = ptr<T>(this->src, sy) + i;
                    for (int w = this->xstart; w < this->xend; ++w)
                    {
                        int sx = clamp(ix + w, 0, this->col1) * cn;
                        W t = this->coeff.at((w - rx) * this->xscale);
                        sline += t * S[sx];
                    }
                    W u = this->coeff.at((h - ry) * this->yscale);
                    sumval += u * sline;
                }
                D[i] = saturate_cast<T>(sumval / weight);
            }
        }
    };

    template <typename T1, typename W1, int cn, typename Coeff>
    struct AntiVecExOut : public AntiBase<Coeff>
    {
        using AntiBase<Coeff>::AntiBase;
        using T = typename TypeVec<T1, cn>::vec_type;
        using W = typename TypeVec<W1, cn>::vec_type;

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int ix = __float2int_rd(fx), iy = __float2int_rd(fy);
            float rx = fx - ix, ry = fy - iy;
            W1 weight = 0;
            W sumval = VecTraits<W>::all(0);
            for (int h = this->ystart; h < this->yend; ++h)
            {
                int sy = iy + h;
                if (static_cast<unsigned>(sy) > static_cast<unsigned>(this->row1))
                    continue;
                W1 wline = 0;
                W sline = VecTraits<W>::all(0);
                T const* S = ptr<T>(this->src, sy);
                for (int w = this->xstart; w < this->xend; ++w)
                {
                    int sx = ix + w;
                    if (static_cast<unsigned>(sx) > static_cast<unsigned>(this->col1))
                        continue;
                    W1 t = this->coeff.at((w - rx) * this->xscale);
                    wline += t;
                    sline += t * saturate_cast<W>(S[sx]);
                }
                W1 u = this->coeff.at((h - ry) * this->yscale);
                weight += u * wline;
                sumval += u * sline;
            }
            at<T>(this->dst, dy, dx) = saturate_cast<T>(sumval / weight);
        }
    };

    template <typename T, typename W, typename Coeff>
    struct AntiCnExOut : public AntiBase<Coeff>
    {
        int cn;

        AntiCnExOut(PtrStepSzb const& S, PtrStepSzb const& D,
            Point2f const& scale, float A, int _cn)
            : AntiBase<Coeff>(S, D, scale, A), cn(_cn)
        {}

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int ix = __float2int_rd(fx), iy = __float2int_rd(fy);
            float rx = fx - ix, ry = fy - iy;
            W weight = 0, sumval = 0;
            T* D = ptr<T>(this->dst, dy) + dx * cn;
            for (int h = this->ystart; h < this->yend; ++h)
            {
                int sy = iy + h;
                if (static_cast<unsigned>(sy) > static_cast<unsigned>(this->row1))
                    continue;
                W wline = 0, sline = 0;
                T const* S = ptr<T>(this->src, sy);
                for (int w = this->xstart; w < this->xend; ++w)
                {
                    int sx = ix + w;
                    if (static_cast<unsigned>(sx) > static_cast<unsigned>(this->col1))
                        continue;
                    sx = sx * cn;
                    W t = this->coeff.at((w - rx) * this->xscale);
                    wline += t;
                    sline += t * S[sx];
                }
                W u = this->coeff.at((h - ry) * this->yscale);
                weight += u * wline;
                sumval += u * sline;
            }
            D[0] = saturate_cast<T>(sumval / weight);
            for (int i = 1; i < cn; ++i)
            {
                sumval = 0;
                for (int h = this->ystart; h < this->yend; ++h)
                {
                    int sy = iy + h;
                    if (static_cast<unsigned>(sy) > static_cast<unsigned>(this->row1))
                        continue;
                    W sline = 0;
                    T const* S = ptr<T>(this->src, sy) + i;
                    for (int w = this->xstart; w < this->xend; ++w)
                    {
                        int sx = ix + w;
                        if (static_cast<unsigned>(sx) > static_cast<unsigned>(this->col1))
                            continue;
                        sx = sx * cn;
                        W t = this->coeff.at((w - rx) * this->xscale);
                        sline += t * S[sx];
                    }
                    W u = this->coeff.at((h - ry) * this->yscale);
                    sumval += u * sline;
                }
                D[i] = saturate_cast<T>(sumval / weight);
            }
        }
    };

    ////////// bi-linear //////////

    template <typename T1, typename W1, int cn>
    struct LinearVec : public SamplerBase
    {
        using SamplerBase::SamplerBase;
        using T = typename TypeVec<T1, cn>::vec_type;
        using W = typename TypeVec<W1, cn>::vec_type;

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int ix = __float2int_rd(fx), iy = __float2int_rd(fy);
            float u1 = fx - ix, v1 = fy - iy;
            float u0 = 1.f - u1, v0 = 1.f - v1;
            int x0 = ::max(ix, 0);
            int y0 = ::max(iy, 0);
            int x1 = ::min(ix + 1, col1);
            int y1 = ::min(iy + 1, row1);
            W s0 = saturate_cast<W>(at<T>(src, y0, x0));
            W s1 = saturate_cast<W>(at<T>(src, y0, x1));
            W s2 = saturate_cast<W>(at<T>(src, y1, x0));
            W s3 = saturate_cast<W>(at<T>(src, y1, x1));
            W val = (u0 * v0) * s0 + (u1 * v0) * s1 + (u0 * v1) * s2 + (u1 * v1) * s3;
            at<T>(dst, dy, dx) = saturate_cast<T>(val);
        }
    };

    template <typename T, typename W>
    struct LinearCn : public SamplerBase
    {
        int cn;

        LinearCn(PtrStepSzb const& S, PtrStepSzb const& D, int _cn)
            : SamplerBase(S, D), cn(_cn)
        {}

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int ix = __float2int_rd(fx), iy = __float2int_rd(fy);
            float u1 = fx - ix, v1 = fy - iy;
            float u0 = 1.f - u1, v0 = 1.f - v1;
            int x0 = ::max(ix, 0);
            int y0 = ::max(iy, 0);
            int x1 = ::min(ix + 1, col1);
            int y1 = ::min(iy + 1, row1);
            W coeff[4] = {u0 * v0, u1 * v0, u0 * v1, u1 * v1};
            T const* S0 = ptr<T>(src, y0) + x0 * cn;
            T const* S1 = ptr<T>(src, y0) + x1 * cn;
            T const* S2 = ptr<T>(src, y1) + x0 * cn;
            T const* S3 = ptr<T>(src, y1) + x1 * cn;
            T      * D  = ptr<T>(dst, dy) + dx * cn;
            for (int i = 0; i < cn; ++i)
            {
                D[i] = saturate_cast<T>(coeff[0] * S0[i]
                    + coeff[1] * S1[i] + coeff[2] * S2[i] + coeff[3] * S3[i]);
            }
        }
    };

    template <typename T1, typename W1, int cn>
    using LinearAntiVec = AntiVec<T1, W1, cn, LinearCoeff>;

    template <typename T, typename W>
    using LinearAntiCn = AntiCn<T, W, LinearCoeff>;

    template <typename T1, typename W1, int cn>
    using LinearAntiVecExOut = AntiVecExOut<T1, W1, cn, LinearCoeff>;

    template <typename T, typename W>
    using LinearAntiCnExOut = AntiCnExOut<T, W, LinearCoeff>;

    ////////// bi-cubic //////////

    template <typename T1, typename W1, int cn>
    struct CubicVec : public SamplerBase
    {
        CubicCoeff cubic;
        using T = typename TypeVec<T1, cn>::vec_type;
        using W = typename TypeVec<W1, cn>::vec_type;

        CubicVec(PtrStepSzb const& S, PtrStepSzb const& D, float A)
            : SamplerBase(S, D), cubic(A)
        {}

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int xstart = __float2int_rd(fx) - 1;
            int ystart = __float2int_rd(fy) - 1;
            int xoffset[4];
            W1 xcoeff[4];
            for (int x = 0; x < 4; ++x, ++xstart)
            {
                xoffset[x] = clamp(xstart, 0, col1);
                xcoeff [x] = cubic.at(xstart - fx);
            }
            W sumval = VecTraits<W>::all(0);
            for (int y = 0; y < 4; ++y, ++ystart)
            {
                int yoffest = clamp(ystart, 0, row1);
                T const* S = ptr<T>(src, yoffest);
                W sline = VecTraits<W>::all(0);
                for (int x = 0; x < 4; ++x)
                    sline += xcoeff[x] * saturate_cast<W>(S[xoffset[x]]);
                sumval += sline * cubic.at(ystart - fy);
            }
            at<T>(dst, dy, dx) = saturate_cast<T>(sumval);
        }
    };

    template <typename T, typename W>
    struct CubicCn : public SamplerBase
    {
        CubicCoeff cubic;
        int cn;

        CubicCn(PtrStepSzb const& S, PtrStepSzb const& D, float A, int _cn)
            : SamplerBase(S, D), cubic(A), cn(_cn)
        {}

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int xstart = __float2int_rd(fx) - 1;
            int ystart = __float2int_rd(fy) - 1;
            int xoffset[4], yoffset[4];
            W xcoeff[4], ycoeff[4];
            for (int x = 0; x < 4; ++x, ++xstart)
            {
                xoffset[x] = clamp(xstart, 0, col1) * cn;
                xcoeff [x] = cubic.at(xstart - fx);
            }
            for (int y = 0; y < 4; ++y, ++ystart)
            {
                yoffset[y] = clamp(ystart, 0, row1);
                ycoeff [y] = cubic.at(ystart - fy);
            }
            T* D = ptr<T>(dst, dy) + dx * cn;
            for (int i = 0; i < cn; ++i)
            {
                W sumval = 0;
                for (int y = 0; y < 4; ++y)
                {
                    T const* S = ptr<T>(src, yoffset[y]) + i;
                    W sline = 0;
                    for (int x = 0; x < 4; ++x)
                        sline += xcoeff[x] * S[xoffset[x]];
                    sumval += sline * ycoeff[y];
                }
                D[i] = saturate_cast<T>(sumval);
            }
        }
    };

    template <typename T1, typename W1, int cn>
    struct CubicVecExOut : public SamplerBase
    {
        CubicCoeff cubic;
        using T = typename TypeVec<T1, cn>::vec_type;
        using W = typename TypeVec<W1, cn>::vec_type;

        CubicVecExOut(PtrStepSzb const& S, PtrStepSzb const& D, float A)
            : SamplerBase(S, D), cubic(A)
        {}

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int xstart = __float2int_rd(fx) - 1;
            int ystart = __float2int_rd(fy) - 1;
            int xoffset[4];
            W1 xcoeff[4], xcoeffsum = 0, ycoeffsum = 0;
            for (int x = 0; x < 4; ++x, ++xstart)
            {
                xoffset[x] = clamp(xstart, 0, col1);
                xcoeff [x] = cubic.at(xstart - fx);
                if (static_cast<unsigned>(xstart) > static_cast<unsigned>(col1))
                    xcoeff[x] = 0;
                xcoeffsum += xcoeff[x];
            }
            W sumval = VecTraits<W>::all(0);
            for (int y = 0; y < 4; ++y, ++ystart)
            {
                if (static_cast<unsigned>(ystart) > static_cast<unsigned>(row1))
                    continue;
                int yoffest = ystart;
                T const* S = ptr<T>(src, yoffest);
                W sline = VecTraits<W>::all(0);
                for (int x = 0; x < 4; ++x)
                    sline += xcoeff[x] * saturate_cast<W>(S[xoffset[x]]);
                W1 u = cubic.at(ystart - fy);
                ycoeffsum += u;
                sumval += sline * u;
            }
            at<T>(dst, dy, dx) = saturate_cast<T>(sumval / (ycoeffsum * xcoeffsum));
        }
    };

    template <typename T, typename W>
    struct CubicCnExOut : public SamplerBase
    {
        CubicCoeff cubic;
        int cn;

        CubicCnExOut(PtrStepSzb const& S, PtrStepSzb const& D, float A, int _cn)
            : SamplerBase(S, D), cubic(A), cn(_cn)
        {}

        __device__ void to(float fx, float fy, int dx, int dy) const
        {
            int xstart = __float2int_rd(fx) - 1;
            int ystart = __float2int_rd(fy) - 1;
            int xoffset[4], yoffset[4];
            W xcoeff[4], ycoeff[4], xcoeffsum = 0, ycoeffsum = 0;
            for (int x = 0; x < 4; ++x, ++xstart)
            {
                xoffset[x] = clamp(xstart, 0, col1) * cn;
                xcoeff [x] = cubic.at(xstart - fx);
                if (static_cast<unsigned>(xstart) > static_cast<unsigned>(col1))
                    xcoeff[x] = 0;
                xcoeffsum += xcoeff[x];
            }
            for (int y = 0; y < 4; ++y, ++ystart)
            {
                yoffset[y] = clamp(ystart, 0, row1);
                ycoeff [y] = cubic.at(ystart - fy);
                if (static_cast<unsigned>(ystart) > static_cast<unsigned>(row1))
                    ycoeff[y] = 0;
                ycoeffsum += ycoeff[y];
            }
            W weight = xcoeffsum * ycoeffsum;
            T* D = ptr<T>(dst, dy) + dx * cn;
            for (int i = 0; i < cn; ++i)
            {
                W sumval = 0;
                for (int y = 0; y < 4; ++y)
                {
                    T const* S = ptr<T>(src, yoffset[y]) + i;
                    W sline = 0;
                    for (int x = 0; x < 4; ++x)
                        sline += xcoeff[x] * S[xoffset[x]];
                    sumval += sline * ycoeff[y];
                }
                D[i] = saturate_cast<T>(sumval / weight);
            }
        }
    };

    template <typename T1, typename W1, int cn>
    using CubicAntiVec = AntiVec<T1, W1, cn, CubicCoeff>;

    template <typename T, typename W>
    using CubicAntiCn = AntiCn<T, W, CubicCoeff>;

    template <typename T1, typename W1, int cn>
    using CubicAntiVecExOut = AntiVecExOut<T1, W1, cn, CubicCoeff>;

    template <typename T, typename W>
    using CubicAntiCnExOut = AntiCnExOut<T, W, CubicCoeff>;

    ////////// generic //////////

    template <typename Sampler>
    __global__ void sampleKernel(Matx22f const M, Sampler const sampler)
    {
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;
        if (dx < sampler.dst.cols && dy < sampler.dst.rows)
        {
            float fx = ::fmaf(static_cast<float>(dx), M.val[0], M.val[1]);
            float fy = ::fmaf(static_cast<float>(dy), M.val[2], M.val[3]);
            sampler.to(fx, fy, dx, dy);
        }
    }

    //==================== nearest neighbor  ====================//

    struct RoundUp
    {
        __device__ __forceinline__ int operator()(float x) const
        { return __float2int_ru(x); }
    };

    struct RoundDown
    {
        __device__ __forceinline__ int operator()(float x) const
        { return __float2int_rd(x); }
    };

    template <typename RoundOp, typename Sampler>
    __global__ void nnBySampler(
        RoundOp const R, Sampler const sampler, Matx22f const M, float const offset)
    {
        int dx = blockDim.x * blockIdx.x + threadIdx.x;
        int dy = blockDim.y * blockIdx.y + threadIdx.y;
        if (dx < sampler.dst.cols && dy < sampler.dst.rows)
        {
            int sx = R(::fmaf(static_cast<float>(dx), M.val[0], M.val[1]) + offset);
            int sy = R(::fmaf(static_cast<float>(dy), M.val[2], M.val[3]) + offset);
            sampler.to(sx, sy, dx, dy);
        }
    }

    template <typename RoundOp>
    void nnByRound(size_t esz, PtrStepSzb const& src, PtrStepSzb dst,
        Matx22f const& M, float offset, cudaStream_t stream)
    {
        RoundOp R;
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (esz == 1)
            nnBySampler<<<grid, block, 0, stream>>>(R, NearestVec<uchar>(src, dst), M, offset);
        else if (esz == 2)
            nnBySampler<<<grid, block, 0, stream>>>(R, NearestVec<ushort>(src, dst), M, offset);
        else if (esz == 3)
            nnBySampler<<<grid, block, 0, stream>>>(R, NearestVec<uchar3>(src, dst), M, offset);
        else if (esz == 4)
            nnBySampler<<<grid, block, 0, stream>>>(R, NearestVec<uint>(src, dst), M, offset);
        else if (esz == 6)
            nnBySampler<<<grid, block, 0, stream>>>(R, NearestVec<ushort3>(src, dst), M, offset);
        else if (esz == 8)
            nnBySampler<<<grid, block, 0, stream>>>(R, NearestVec<uint2>(src, dst), M, offset);
        else if (esz == 12)
            nnBySampler<<<grid, block, 0, stream>>>(R, NearestVec<uint3>(src, dst), M, offset);
        else if (esz == 16)
            nnBySampler<<<grid, block, 0, stream>>>(R, NearestVec<uint4>(src, dst), M, offset);
        else
            nnBySampler<<<grid, block, 0, stream>>>(R, NearestSize(src, dst, esz), M, offset);
    }

    void resizeOnnxNN(size_t elemSize, PtrStepSzb const& src, PtrStepSzb const& dst,
        Matx22f const& M, int mode, cudaStream_t stream)
    {
        float offset = 0.f;
        if (mode == INTER_NEAREST_PREFER_FLOOR)
            offset = -0.5f;
        if (mode == INTER_NEAREST_PREFER_CEIL)
            offset = +0.5f;

        if (mode == INTER_NEAREST_PREFER_FLOOR || mode == INTER_NEAREST_CEIL)
            nnByRound<RoundUp>(elemSize, src, dst, M, offset, stream);
        else
            nnByRound<RoundDown>(elemSize, src, dst, M, offset, stream);
        if (!stream)
            cudaSafeCall(cudaDeviceSynchronize());
    }

    //==================== linear ====================//

    template <typename T, typename W>
    void linear(int cn, PtrStepSzb const& src, PtrStepSzb const& dst,
        Matx22f const& M, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (cn == 1)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearVec<T, W, 1>(src, dst));
        else if (cn == 2)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearVec<T, W, 2>(src, dst));
        else if (cn == 3)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearVec<T, W, 3>(src, dst));
        else if (cn == 4)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearVec<T, W, 4>(src, dst));
        else
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearCn<T, W>(src, dst, cn));
    }

    template <typename T, typename W>
    void linearAnti(int cn, PtrStepSzb const& src, PtrStepSzb const& dst,
        Matx22f const& M, Point2f const& scale, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (cn == 1)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiVec<T, W, 1>(src, dst, scale, 0));
        else if (cn == 2)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiVec<T, W, 2>(src, dst, scale, 0));
        else if (cn == 3)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiVec<T, W, 3>(src, dst, scale, 0));
        else if (cn == 4)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiVec<T, W, 4>(src, dst, scale, 0));
        else
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiCn<T, W>(src, dst, scale, 0, cn));
    }

    template <typename T, typename W>
    void linearAntiExOut(int cn, PtrStepSzb const& src, PtrStepSzb const& dst,
        Matx22f const& M, Point2f const& scale, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (cn == 1)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiVecExOut<T, W, 1>(src, dst, scale, 0));
        else if (cn == 2)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiVecExOut<T, W, 2>(src, dst, scale, 0));
        else if (cn == 3)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiVecExOut<T, W, 3>(src, dst, scale, 0));
        else if (cn == 4)
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiVecExOut<T, W, 4>(src, dst, scale, 0));
        else
            sampleKernel<<<grid, block, 0, stream>>>(M, LinearAntiCnExOut<T, W>(src, dst, scale, 0, cn));
    }

    //==================== cubic  ====================//

    template <typename T, typename W>
    void cubic(int cn, float A, PtrStepSzb const& src,
        PtrStepSzb const& dst, Matx22f const& M, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (cn == 1)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicVec<T, W, 1>(src, dst, A));
        else if (cn == 2)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicVec<T, W, 2>(src, dst, A));
        else if (cn == 3)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicVec<T, W, 3>(src, dst, A));
        else if (cn == 4)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicVec<T, W, 4>(src, dst, A));
        else
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicCn<T, W>(src, dst, A, cn));
    }

    template <typename T, typename W>
    void cubicExOut(int cn, float A, PtrStepSzb const& src,
        PtrStepSzb const& dst, Matx22f const& M, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (cn == 1)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicVecExOut<T, W, 1>(src, dst, A));
        else if (cn == 2)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicVecExOut<T, W, 2>(src, dst, A));
        else if (cn == 3)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicVecExOut<T, W, 3>(src, dst, A));
        else if (cn == 4)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicVecExOut<T, W, 4>(src, dst, A));
        else
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicCnExOut<T, W>(src, dst, A, cn));
    }

    template <typename T, typename W>
    void cubicAnti(int cn, float A, PtrStepSzb const& src, PtrStepSzb const& dst,
        Matx22f const& M, Point2f const& scale, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (cn == 1)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiVec<T, W, 1>(src, dst, scale, A));
        else if (cn == 2)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiVec<T, W, 2>(src, dst, scale, A));
        else if (cn == 3)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiVec<T, W, 3>(src, dst, scale, A));
        else if (cn == 4)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiVec<T, W, 4>(src, dst, scale, A));
        else
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiCn<T, W>(src, dst, scale, A, cn));
    }

    template <typename T, typename W>
    void cubicAntiExOut(int cn, float A, PtrStepSzb const& src, PtrStepSzb const& dst,
        Matx22f const& M, Point2f const& scale, cudaStream_t stream)
    {
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
        if (cn == 1)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiVecExOut<T, W, 1>(src, dst, scale, A));
        else if (cn == 2)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiVecExOut<T, W, 2>(src, dst, scale, A));
        else if (cn == 3)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiVecExOut<T, W, 3>(src, dst, scale, A));
        else if (cn == 4)
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiVecExOut<T, W, 4>(src, dst, scale, A));
        else
            sampleKernel<<<grid, block, 0, stream>>>(M, CubicAntiCnExOut<T, W>(src, dst, scale, A, cn));
    }

template <typename T, typename W>
void resizeOnnx(int cn, float A, PtrStepSzb const& src, PtrStepSzb const& dst,
    Matx22f const& M, Point2f const& scale, int interpolation, cudaStream_t stream)
{
    int sampler = interpolation & INTER_SAMPLER_MASK;
    int antialias = interpolation & INTER_ANTIALIAS_MASK;
    int exclude_outside = interpolation & INTER_EXCLUDE_OUTSIDE_MASK;
    if (exclude_outside)
    {
        if (sampler == INTER_LINEAR && !antialias)
            linear<T, W>(cn, src, dst, M, stream);
        else if (sampler == INTER_LINEAR && antialias)
            linearAntiExOut<T, W>(cn, src, dst, M, scale, stream);
        else if (sampler == INTER_CUBIC && !antialias)
            cubicExOut<T, W>(cn, A, src, dst, M, stream);
        else if (sampler == INTER_CUBIC && antialias)
            cubicAntiExOut<T, W>(cn, A, src, dst, M, scale, stream);
        else
            CV_Error(cv::Error::StsBadArg, "unsupported interpolation");
    }
    else
    {
        if (sampler == INTER_LINEAR && !antialias)
            linear<T, W>(cn, src, dst, M, stream);
        else if (sampler == INTER_LINEAR && antialias)
            linearAnti<T, W>(cn, src, dst, M, scale, stream);
        else if (sampler == INTER_CUBIC && !antialias)
            cubic<T, W>(cn, A, src, dst, M, stream);
        else if (sampler == INTER_CUBIC && antialias)
            cubicAnti<T, W>(cn, A, src, dst, M, scale, stream);
        else
            CV_Error(cv::Error::StsBadArg, "unsupported interpolation");
    }

    if (!stream)
        cudaSafeCall(cudaDeviceSynchronize());
}

template void resizeOnnx<uchar, float>(int cn, float A,
    PtrStepSzb const& src, PtrStepSzb const& dst, Matx22f const& M,
    Point2f const& scale, int interpolation, cudaStream_t stream);

template void resizeOnnx<schar, float>(int cn, float A,
    PtrStepSzb const& src, PtrStepSzb const& dst, Matx22f const& M,
    Point2f const& scale, int interpolation, cudaStream_t stream);

template void resizeOnnx<ushort, float>(int cn, float A,
    PtrStepSzb const& src, PtrStepSzb const& dst, Matx22f const& M,
    Point2f const& scale, int interpolation, cudaStream_t stream);

template void resizeOnnx<short, float>(int cn, float A,
    PtrStepSzb const& src, PtrStepSzb const& dst, Matx22f const& M,
    Point2f const& scale, int interpolation, cudaStream_t stream);

template void resizeOnnx<int, double>(int cn, float A,
    PtrStepSzb const& src, PtrStepSzb const& dst, Matx22f const& M,
    Point2f const& scale, int interpolation, cudaStream_t stream);

template void resizeOnnx<float, float>(int cn, float A,
    PtrStepSzb const& src, PtrStepSzb const& dst, Matx22f const& M,
    Point2f const& scale, int interpolation, cudaStream_t stream);

template void resizeOnnx<double, double>(int cn, float A,
    PtrStepSzb const& src, PtrStepSzb const& dst, Matx22f const& M,
    Point2f const& scale, int interpolation, cudaStream_t stream);

/*template void resizeOnnx<__half, float>(int cn, float A,
    PtrStepSzb const& src, PtrStepSzb const& dst, Matx22f const& M,
    Point2f const& scale, int interpolation, cudaStream_t stream);*/
}}}

#endif /* CUDA_DISABLER */
