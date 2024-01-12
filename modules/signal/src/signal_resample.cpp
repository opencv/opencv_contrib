// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"

#include <opencv2/signal/signal_resample.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/utils/trace.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace cv {
namespace signal {

#if (CV_SIMD || CV_SIMD_SCALABLE)
#define v_float32_width (uint32_t)VTraits<v_float32>::vlanes()
const uint32_t v_float32_max_width = (uint32_t)VTraits<v_float32>::max_nlanes;
#endif

// Modified Bessel function 1st kind 0th order
static float Bessel(float x)
{
    int k = 12; // approximation parameter
    float defmul = x * x * 0.25f;
    float mul = defmul;
    float acc = 0.f;
    for(int i = 0 ; i < k; ++i)
    {
        mul = powf(defmul, static_cast<float>(i));
        mul = mul / powf(tgammaf(static_cast<float>(i + 1)), 2.f); // tgamma(i+1) equals i!
        acc +=mul;
    }
    return acc;
}

static void init_filter(float beta, int ntabs, float* tabs)
{
    float fc = 0.25f;
    // build sinc filter
    for (int i = 0; i < ntabs; ++i)
    {
        tabs[i] = 2 * fc * (i - (ntabs - 1) / 2);
    }
    std::vector<float> tmparr(ntabs);
    for (int i = 0 ; i < ntabs; ++i)
    {
        if (tabs[i] == 0.f)
        {
            tmparr[i] = 1.f;
            continue;
        }
        tmparr[i] = (float)(CV_PI * tabs[i]);
    }
    float mult = 2.f / (float)(ntabs - 1);
    // multiply by Kaiser window
    for (int i = 0; i < ntabs; ++i)
    {
        tabs[i] = std::sin(tmparr[i]) / tmparr[i];
        tabs[i] *= Bessel(beta * sqrtf((float)1 - powf((i * mult - 1), 2))) / Bessel(beta);
    }
    float sum = 0.f;
    for (int i = 0 ; i < ntabs; ++i)
    {
        sum += tabs[i];
    }
    sum = 1.f/sum;
    // normalize tabs to get unity gain
    for (int i = 0; i < ntabs; ++i)
    {
        tabs[i] *= sum;
    }
}

/////////////// cubic Hermite spline (tail of execIntrinLoop or scalar version) ///////////////
static float scal_cubicHermite(float A, float B, float C, float D, float t)
{
    float a = (-A + (3.0f * B) - (3.0f * C) + D) * 0.5f;
    float b = A + C + C - (5.0f * B + D) * 0.5f;
    float c = (-A + C) * 0.5f;
    return a * t * t * t + b * t * t + c * t + B;
}

/////////////// cubic Hermite spline (OpenCV's Universal Intrinsics) ///////////////
#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline v_float32 simd_cubicHermite(const v_float32 &v_A, const v_float32 &v_B, const v_float32 &v_C,
                                    const v_float32 &v_D, const v_float32 &v_t)
{
    v_float32 v_zero = vx_setzero_f32();
    v_float32 v_three= vx_setall_f32(3.0f);
    v_float32 v_half = vx_setall_f32(0.5f);
    v_float32 v_five = vx_setall_f32(5.0f);

    v_float32 v_inv_A = v_sub(v_zero, v_A);

    v_float32 v_a = v_mul(v_sub(v_fma(v_three, v_B, v_add(v_inv_A, v_D)), v_mul(v_three, v_C)), v_half);
    v_float32 v_b = v_sub(v_add(v_A, v_C, v_C), v_mul(v_fma(v_five, v_B, v_D), v_half));
    v_float32 v_c = v_mul(v_add(v_inv_A, v_C), v_half);

    return v_add(v_mul(v_a, v_t, v_t, v_t), v_mul(v_b, v_t, v_t), v_fma(v_c, v_t, v_B));
}
#endif

static void cubicInterpolate(const Mat1f &src, uint32_t dstlen, Mat1f &dst, uint32_t srclen)
{
    Mat1f tmp(Size(srclen + 3U, 1U));
    tmp.at<float>(0) = src.at<float>(0);

#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_float32 v_reg = vx_setall_f32(src.at<float>(srclen - 1U));
    vx_store(tmp.ptr<float>(0) + (srclen - 1U), v_reg);
#else // scalar version
    tmp.at<float>(srclen + 1U) = src.at<float>(srclen - 1U);
    tmp.at<float>(srclen + 2U) = src.at<float>(srclen - 1U);
#endif

    uint32_t i = 0U;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    uint32_t len_sub_vfloatStep = (uint32_t)std::max((int64_t)srclen - (int64_t)v_float32_width, (int64_t)0);
    for (; i < len_sub_vfloatStep; i+= v_float32_width)
    {
        v_float32 v_copy = vx_load(src.ptr<float>(0) + i);
        vx_store(tmp.ptr<float>(0) + (i + 1U), v_copy);
    }
#endif

    // if the tail exists or scalar version
    for (; i < srclen; ++i)
    {
        tmp.at<float>(i + 1U) = src.at<float>(i);
    }

    i = 0U;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int ptr_x_int[v_float32_max_width];
    uint32_t j;

    v_float32 v_dstlen_sub_1 = vx_setall_f32((float)(dstlen - 1U));
    v_float32 v_one = vx_setall_f32(1.0f);
    v_float32 v_x_start = v_div(v_one, v_dstlen_sub_1);
    v_float32 v_u = vx_setall_f32((float)srclen);
    v_float32 v_half = vx_setall_f32(0.5f);

    len_sub_vfloatStep = (uint32_t)std::max((int64_t)dstlen - (int64_t)v_float32_width, (int64_t)0);
    for (; i < v_float32_width; ++i)
    {
        ptr_x_int[i] = (int)i;
    }

    float ptr_for_cubicHermite[v_float32_max_width];
    v_float32 v_sequence = v_cvt_f32(vx_load(ptr_x_int));
    for (i = 0U; i < len_sub_vfloatStep; i+= v_float32_width)
    {
        v_float32 v_reg_i = v_add(vx_setall_f32((float)i), v_sequence);

        v_float32 v_x = v_sub(v_mul(v_x_start, v_reg_i, v_u), v_half);

        v_int32 v_x_int = v_trunc(v_x);
        v_float32 v_x_fract = v_sub(v_x, v_cvt_f32(v_floor(v_x)));

        vx_store(ptr_x_int, v_x_int);

        for(j = 0U; j < v_float32_width; ++j)
            ptr_for_cubicHermite[j] = *(tmp.ptr<float>(0) + (ptr_x_int[j] - 1));
        v_float32 v_x_int_add_A = vx_load(ptr_for_cubicHermite);

        for(j = 0U; j < v_float32_width; ++j)
            ptr_for_cubicHermite[j] = *(tmp.ptr<float>(0) + (ptr_x_int[j]));
        v_float32 v_x_int_add_B = vx_load(ptr_for_cubicHermite);

        for(j = 0U; j < v_float32_width; ++j)
            ptr_for_cubicHermite[j] = *(tmp.ptr<float>(0) + (ptr_x_int[j] + 1));
        v_float32 v_x_int_add_C = vx_load(ptr_for_cubicHermite);

        for(j = 0U; j < v_float32_width; ++j)
            ptr_for_cubicHermite[j] = *(tmp.ptr<float>(0) + (ptr_x_int[j] + 2));
        v_float32 v_x_int_add_D = vx_load(ptr_for_cubicHermite);


        vx_store(dst.ptr<float>(0) + i, simd_cubicHermite(v_x_int_add_A, v_x_int_add_B, v_x_int_add_C, v_x_int_add_D, v_x_fract));
    }
#endif

    // if the tail exists or scalar version
    float *ptr = tmp.ptr<float>(0) + 1U;
    float lenScale = 1.0f / (float)(dstlen - 1U);
    float U, X, xfract;
    int xint;
    for(; i < dstlen; ++i)
    {
        U = (float)i * lenScale;
        X = (U * (float)srclen) - 0.5f;
        xfract = X - floor(X);
        xint = (int)X;
        dst.at<float>(i) = scal_cubicHermite(ptr[xint - 1], ptr[xint], ptr[xint + 1], ptr[xint + 2], xfract);
    }

}

static void fir_f32(const float *pSrc,       float *pDst,
                    const float *pCoeffs,    float *pBuffer,
                       uint32_t  numTaps, uint32_t  blockSize)
{
    uint32_t copyLen = std::min(blockSize, numTaps);

    /////////////// delay line to the left ///////////////
    uint32_t i = numTaps - 1U, k = 0U, j = 0U;
    uint32_t value_i;
    const float* ptr = pSrc + 1U - numTaps;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    v_float32 v_pDst;
    value_i = (uint32_t)std::max((int64_t)(numTaps + numTaps - 2U) - (int64_t)v_float32_width, (int64_t)0);
    uint32_t value_k = (uint32_t)std::max((int64_t)copyLen - (int64_t)v_float32_width, (int64_t)0);


    uint32_t value_j = (uint32_t)std::max((int64_t)(numTaps) - (int64_t)v_float32_width, (int64_t)0);

    for (; i < value_i && k < value_k; i += v_float32_width, k += v_float32_width)
    {
        v_float32 pSrc_data = vx_load(ptr + i); //vx_load(pSrc + (i + 1U - numTaps));
        vx_store(pBuffer + i, pSrc_data);
    }
#endif

    // if the tail exists or scalar version
    value_i = numTaps + numTaps - 2U;
    for (; i < value_i && k < copyLen; ++i, ++k)
    {
        *(pBuffer + i) = *(ptr + i); // pBuffer[i] = pSrc[i + 1U - numTaps]
    }


    /////////////// process delay line ///////////////
    i = 0U; k = 0U;
    value_i = numTaps - 1U;
    float *ptr_Buf;

    for(; i < value_i && k < copyLen; ++i, ++k)
    {
        ptr_Buf = pBuffer + i;
        j = 0U;

#if (CV_SIMD || CV_SIMD_SCALABLE)

        v_pDst = vx_setzero_f32();
        for (; j < value_j; j += v_float32_width)
        {
            v_float32 v_pBuffer = vx_load(ptr_Buf + j); //vx_load(pBuffer[i + j])
            v_float32 v_pCoeffs = vx_load(pCoeffs + j); //vx_load(pCoeffs[j])

            v_pDst = v_fma(v_pBuffer, v_pCoeffs, v_pDst); // v_pDst = v_pBuffer * v_pCoeffs + v_pDst
        }
        pDst[i] = v_reduce_sum(v_pDst);
#endif

        // if the tail exists or scalar version
        for (; j < numTaps; ++j)
            pDst[i] += pCoeffs[j] * *(ptr_Buf + j); // pDst[i] += pCoeffs[j] * pBuffer[i + j];
    }


    /////////////// process main block ///////////////
    i = numTaps - 1U;

    for(; i < blockSize; ++i)
    {
        const float *ptr_Src = pSrc + (i + 1U - numTaps);
        j = 0U;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_pDst = vx_setzero_f32();
        for (; j < value_j; j += v_float32_width)
        {
            v_float32 v_pSrc = vx_load(ptr_Src + j); // vx_load(pSrc[i + j - (numTaps - 1)])
            v_float32 v_pCoeffs = vx_load(pCoeffs + j); //vx_load(pCoeffs[j])
            v_pDst = v_fma(v_pSrc, v_pCoeffs, v_pDst);
        }
        pDst[i] = v_reduce_sum(v_pDst);
#endif

        // if the tail exists or scalar version
        for (; j < numTaps; ++j)
            pDst[i] += pCoeffs[j] * *(ptr_Src + j); // pDst[i] += pCoeffs[j] * pSrc[i + j + 1U - numTaps];
    }


    /////////////// move delay line left by copyLen elements ///////////////
#if (CV_SIMD || CV_SIMD_SCALABLE)
    value_i = (uint32_t)std::max((int64_t)(numTaps - 1U) - (int64_t)v_float32_width, (int64_t)0);
    ptr_Buf = pBuffer + copyLen;

    for(i = 0U; i < value_i; i += v_float32_width)
    {
        v_float32 v_pBuffer = vx_load(ptr_Buf + i); //vx_load(pBuffer[copyLen + i])
        vx_store(pBuffer + i, v_pBuffer);
    }
#endif

    // if the tail exists or scalar version
    value_i = numTaps - 1U;
    for (; i < value_i; ++i)
    {
        pBuffer[i] = pBuffer[i + copyLen];
    }


    /////////////// copy new elements       ///////////////
    /////////////// post-process delay line ///////////////
    int l = (int)(numTaps - 2U); k = 0U;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    int value_l = (int)v_float32_width;
    const float* ptr_part = pSrc + (blockSize + 1U - numTaps - v_float32_width);
    for(; l >= value_l && k < value_k; l -= value_l, k += v_float32_width)
    {
        v_float32 v_pSrc = vx_load(ptr_part + l); // vx_load(pSrc[blockSize - (numTaps - 1) + l - v_float32_width])
        vx_store(pBuffer + (l - value_l), v_pSrc);
    }
#endif
    const float* ptr_Src = pSrc + (blockSize + 1U - numTaps);
    for(; l >= 0 && k < copyLen; --l, ++k)
    {
        pBuffer[l] = *(ptr_Src + l); // pBuffer[l] = pSrc[blockSize + 1U - numTaps + l];
    }
}

void resampleSignal(InputArray inputSignal, OutputArray outputSignal,
                    const int inFreq, const int  outFreq)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!inputSignal.empty());
    CV_CheckGE(inFreq, 1000, "");
    CV_CheckGE(outFreq, 1000, "");
    if (inFreq == outFreq)
    {
        inputSignal.copyTo(outputSignal);
        return;
    }
    uint32_t filtLen = 33U;
    float beta = 3.395f;
    std::vector<float> filt_window(filtLen, 0.f);
    init_filter(beta, filtLen, filt_window.data());
    float ratio = (float)outFreq / float(inFreq);
    Mat1f inMat = inputSignal.getMat();
    Mat1f outMat = Mat1f(Size(cvFloor(inMat.cols * ratio), 1));
    cubicInterpolate(inMat, outMat.cols, outMat, inMat.cols);
    if (inFreq < 2 * outFreq)
    {
        std::vector<float> dlyl(filtLen * 2 - 1, 0.f);
        std::vector<float> ptmp(outMat.cols + 2 * filtLen, 0.);

        for (auto i = filtLen; i < outMat.cols + filtLen; ++i)
        {
            ptmp[i] = outMat.at<float>(i - filtLen);
        }
        std::vector<float> ptmp2(outMat.cols + 2 * filtLen, 0.f);
        fir_f32(ptmp.data(), ptmp2.data(), filt_window.data(), dlyl.data(), filtLen, (uint32_t)(ptmp.size()));
        for (auto i = filtLen; i < outMat.cols + filtLen; ++i)
        {
            outMat.at<float>(i - filtLen) = ptmp2[i + cvFloor((float)filtLen / 2.f)];
        }
    }
    outputSignal.assign(std::move(outMat));
}


}
}
