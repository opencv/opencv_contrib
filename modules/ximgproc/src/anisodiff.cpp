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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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

/* the reference code has been contributed by Chris Sav */

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencl_kernels_ximgproc.hpp"

namespace cv {
namespace ximgproc {

#if CV_SIMD128
inline void v_expand_s(const v_uint8x16& a, v_int16x8& b, v_int16x8& c)
{
    v_uint16x8 t0, t1;
    v_expand(a, t0, t1);
    b = v_reinterpret_as_s16(t0);
    c = v_reinterpret_as_s16(t1);
}

inline void v_expand_f32(const v_int16x8& a, v_float32x4& b, v_float32x4& c)
{
    v_int32x4 t0, t1;
    v_expand(a, t0, t1);
    b = v_cvt_f32(t0);
    c = v_cvt_f32(t1);
}

inline v_uint8x16 v_finalize_pix_ch(const v_int16x8& c0, const v_int16x8& c1,
                                    const v_float32x4& s0, const v_float32x4& s1,
                                    const v_float32x4& s2, const v_float32x4& s3,
                                    const v_float32x4& alpha)
{
    v_float32x4 f0, f1, f2, f3;
    v_expand_f32(c0, f0, f1);
    v_expand_f32(c1, f2, f3);

    v_int16x8 d0 = v_pack(v_round(s0*alpha + f0), v_round(s1*alpha + f1));
    v_int16x8 d1 = v_pack(v_round(s2*alpha + f2), v_round(s3*alpha + f3));

    return v_pack_u(d0, d1);
}
#endif

class ADBody : public ParallelLoopBody
{
public:
    ADBody(const Mat* src_, Mat* dst_, const float* exptab, float alpha)
    {
        src = src_;
        dst = dst_;
        exptab_ = exptab;
        alpha_ = alpha;
    }

    void operator()(const Range& range) const CV_OVERRIDE
    {
        const int cn = 3;
        int cols = src->cols;
        int step = (int)src->step;
        int tab[] = { -cn, cn, -step-cn, -step, -step+cn, step-cn, step, step+cn };
        float alpha = alpha_;
        const float* exptab = exptab_;

        for( int i = range.start; i < range.end; i++ )
        {
            const uchar* psrc0 = src->ptr<uchar>(i);
            uchar* pdst = dst->ptr<uchar>(i);
            int j = 0;

#if CV_SIMD128
            v_float32x4 v_alpha = v_setall_f32(alpha);
            for( ; j <= cols - 16; j += 16 )
            {
                v_uint8x16 c0, c1, c2;
                v_load_deinterleave(psrc0 + j*3, c0, c1, c2);
                v_int16x8 c00, c01, c10, c11, c20, c21;

                v_expand_s(c0, c00, c01);
                v_expand_s(c1, c10, c11);
                v_expand_s(c2, c20, c21);

                v_float32x4 s00 = v_setzero_f32(), s01 = s00, s02 = s00, s03 = s00;
                v_float32x4 s10 = v_setzero_f32(), s11 = s00, s12 = s00, s13 = s00;
                v_float32x4 s20 = v_setzero_f32(), s21 = s00, s22 = s00, s23 = s00;
                v_float32x4 fd0, fd1, fd2, fd3;

                for( int k = 0; k < 8; k++ )
                {
                    const uchar* psrc1 = psrc0 + j*3 + tab[k];
                    v_uint8x16 p0, p1, p2;
                    v_int16x8 p00, p01, p10, p11, p20, p21;
                    v_load_deinterleave(psrc1, p0, p1, p2);

                    v_expand_s(p0, p00, p01);
                    v_expand_s(p1, p10, p11);
                    v_expand_s(p2, p20, p21);

                    v_int16x8 d00 = p00 - c00, d01 = p01 - c01;
                    v_int16x8 d10 = p10 - c10, d11 = p11 - c11;
                    v_int16x8 d20 = p20 - c20, d21 = p21 - c21;

                    v_uint16x8 n0 = v_abs(d00) + v_abs(d10) + v_abs(d20);
                    v_uint16x8 n1 = v_abs(d01) + v_abs(d11) + v_abs(d21);

                    ushort CV_DECL_ALIGNED(16) nbuf[16];
                    v_store(nbuf, n0);
                    v_store(nbuf + 8, n1);

                    v_float32x4 w0(exptab[nbuf[0]], exptab[nbuf[1]], exptab[nbuf[2]], exptab[nbuf[3]]);
                    v_float32x4 w1(exptab[nbuf[4]], exptab[nbuf[5]], exptab[nbuf[6]], exptab[nbuf[7]]);
                    v_float32x4 w2(exptab[nbuf[8]], exptab[nbuf[9]], exptab[nbuf[10]], exptab[nbuf[11]]);
                    v_float32x4 w3(exptab[nbuf[12]], exptab[nbuf[13]], exptab[nbuf[14]], exptab[nbuf[15]]);

                    v_expand_f32(d00, fd0, fd1);
                    v_expand_f32(d01, fd2, fd3);
                    s00 += fd0*w0; s01 += fd1*w1; s02 += fd2*w2; s03 += fd3*w3;
                    v_expand_f32(d10, fd0, fd1);
                    v_expand_f32(d11, fd2, fd3);
                    s10 += fd0*w0; s11 += fd1*w1; s12 += fd2*w2; s13 += fd3*w3;
                    v_expand_f32(d20, fd0, fd1);
                    v_expand_f32(d21, fd2, fd3);
                    s20 += fd0*w0; s21 += fd1*w1; s22 += fd2*w2; s23 += fd3*w3;
                }

                c0 = v_finalize_pix_ch(c00, c01, s00, s01, s02, s03, v_alpha);
                c1 = v_finalize_pix_ch(c10, c11, s10, s11, s12, s13, v_alpha);
                c2 = v_finalize_pix_ch(c20, c21, s20, s21, s22, s23, v_alpha);
                v_store_interleave(pdst + j*3, c0, c1, c2);
            }
            j *= 3;
#endif

            for( ; j < cols*cn; j += cn )
            {
                int c0  = psrc0[j], c1 = psrc0[j+1], c2 = psrc0[j+2];
                float s0 = 0.f, s1 = 0.f, s2 = 0.f;
                for( int k = 0; k < 8; k++ )
                {
                    const uchar* psrc1 = psrc0 + j + tab[k];
                    int delta0 = psrc1[0] - c0;
                    int delta1 = psrc1[1] - c1;
                    int delta2 = psrc1[2] - c2;
                    int nabla = std::abs(delta0) + std::abs(delta1) + std::abs(delta2);
                    float w = exptab[nabla];
                    s0 += delta0*w;
                    s1 += delta1*w;
                    s2 += delta2*w;
                }
                pdst[j] = saturate_cast<uchar>(c0 + alpha*s0);
                pdst[j+1] = saturate_cast<uchar>(c1 + alpha*s1);
                pdst[j+2] = saturate_cast<uchar>(c2 + alpha*s2);
            }
        }
    }

    const Mat* src;
    Mat* dst;
    const float* exptab_;
    float alpha_;
};

#ifdef HAVE_OPENCL
static bool ocl_anisotropicDiffusion(InputArray src_, OutputArray dst_,
                                     float alpha, int niters,
                                     const std::vector<float>& exptab)
{
    UMat src0 = src_.getUMat(), dst0 = dst_.getUMat();
    int type = src0.type();
    int rows = src0.rows, cols = src0.cols;

    ocl::Kernel k("anisodiff", ocl::ximgproc::anisodiff_oclsrc, "");
    if (k.empty())
        return false;

    UMat temp0x(rows + 2, cols + 2, type);
    UMat temp1x(rows + 2, cols + 2, type);
    UMat temp0(temp0x, Rect(1, 1, cols, rows));
    UMat temp1(temp1x, Rect(1, 1, cols, rows));

    int tabsz = (int)exptab.size();
    UMat uexptab = Mat(1, tabsz, CV_32F, (void*)&exptab[0]).getUMat(ACCESS_READ);

    for (int t = 0; t < niters; t++)
    {
        UMat src = temp0, dst = t == niters-1 ? dst0 : temp1;
        copyMakeBorder(t == 0 ? src0 : src, temp0x, 1, 1, 1, 1, BORDER_REPLICATE);

        k.args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnly(dst),
               ocl::KernelArg::PtrReadOnly(uexptab), alpha);

        size_t globalsize[] = { (size_t)cols, (size_t)rows };
        if(!k.run(2, globalsize, NULL, true))
            return false;

        std::swap(temp0, temp1);
        std::swap(temp0x, temp1x);
    }
    return true;
}
#endif

void anisotropicDiffusion(InputArray src_, OutputArray dst_, float alpha, float K, int niters )
{
    if( niters == 0 )
    {
        src_.copyTo(dst_);
        return;
    }

    int type = src_.type();
    CV_Assert(src_.dims() == 2 && type == CV_8UC3);
    CV_Assert(K != 0);
    CV_Assert(alpha > 0);
    CV_Assert(niters >= 0);

    const int cn = 3;
    float sigma = K * cn * 255.f;
    float isigma2 = 1 / (sigma * sigma);
    std::vector<float> exptab_(255*3);
    float* exptab = &exptab_[0];

    for( int k = 0; k < 255*3; k++ )
        exptab[k] = std::exp(-k*k*isigma2);

    dst_.create(src_.size(), type);

    CV_OCL_RUN(dst_.isUMat(),
               ocl_anisotropicDiffusion(src_, dst_, alpha, niters, exptab_));

    Mat src0 = src_.getMat();
    int rows = src0.rows, cols = src0.cols;

    Mat dst0 = dst_.getMat();
    Mat temp0x(rows + 2, cols + 2, type);
    Mat temp1x(rows + 2, cols + 2, type);
    Mat temp0(temp0x, Rect(1, 1, cols, rows));
    Mat temp1(temp1x, Rect(1, 1, cols, rows));

    for (int t = 0; t < niters; t++)
    {
        Mat src = temp0, dst = t == niters-1 ? dst0 : temp1;
        copyMakeBorder(t == 0 ? src0 : src, temp0x, 1, 1, 1, 1, BORDER_REPLICATE);

        ADBody body(&src, &dst, exptab, alpha);
        parallel_for_(Range(0, rows), body, 8);

        std::swap(temp0, temp1);
        std::swap(temp0x, temp1x);
    }
}

}
}
