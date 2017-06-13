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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "precomp.hpp"
#include "layers_common.hpp"
#include <immintrin.h>
#include "opencv2/core/hal/intrin.hpp"

#if CV_DNN_TRY_AVX

#define AVX2_TARGET __attribute__((target("avx, fma")))

namespace cv {
namespace dnn {

void AVX2_TARGET fastConv_avx( const float* weights, size_t wstep, const float* bias,
                               const float* rowbuf, float* output, const int* outShape,
                               int blockSize, int vecsize, int vecsize_aligned, bool initOutput )
{
    int outCn = outShape[1];
    size_t outPlaneSize = outShape[2]*outShape[3];

    // now compute dot product of the weights
    // and im2row-transformed part of the tensor
    for( int i = 0; i < outCn; i += 2 )
    {
        const float* wptr0 = weights + i*wstep;
        const float* wptr1 = wptr0 + wstep;
        float* outptr0 = output + i*outPlaneSize;
        float* outptr1 = outptr0 + outPlaneSize;
        float bias0 = bias[i], bias1 = bias[i+1];

        if( i+1 >= outCn )
        {
            wptr1 = wptr0;
            outptr1 = outptr0;
            bias1 = bias0;
        }

        int j = 0;
        for( ; j <= blockSize - 4; j += 4 )
        {
            const float* rptr = rowbuf + j*vecsize_aligned;
            __m256 s0, s1;

            if( initOutput )
            {
                s0 = _mm256_set1_ps(bias0);
                s1 = _mm256_set1_ps(bias1);
            }
            else
            {
                s0 = _mm256_castps128_ps256(_mm_loadu_ps(outptr0 + j));
                s1 = _mm256_castps128_ps256(_mm_loadu_ps(outptr1 + j));
            }

            __m256 vs00 = _mm256_setzero_ps(), vs01 = _mm256_setzero_ps(),
                   vs02 = _mm256_setzero_ps(), vs03 = _mm256_setzero_ps(),
                   vs10 = _mm256_setzero_ps(), vs11 = _mm256_setzero_ps(),
                   vs12 = _mm256_setzero_ps(), vs13 = _mm256_setzero_ps();

            for( int k = 0; k < vecsize; k += 8, rptr += 8 )
            {
                __m256 w0 = _mm256_load_ps(wptr0 + k), w1 = _mm256_load_ps(wptr1 + k);
                __m256 r0 = _mm256_load_ps(rptr),
                       r1 = _mm256_load_ps(rptr + vecsize_aligned),
                       r2 = _mm256_load_ps(rptr + vecsize_aligned*2),
                       r3 = _mm256_load_ps(rptr + vecsize_aligned*3);

                vs00 = _mm256_fmadd_ps(w0, r0, vs00);
                vs01 = _mm256_fmadd_ps(w0, r1, vs01);
                vs02 = _mm256_fmadd_ps(w0, r2, vs02);
                vs03 = _mm256_fmadd_ps(w0, r3, vs03);
                vs10 = _mm256_fmadd_ps(w1, r0, vs10);
                vs11 = _mm256_fmadd_ps(w1, r1, vs11);
                vs12 = _mm256_fmadd_ps(w1, r2, vs12);
                vs13 = _mm256_fmadd_ps(w1, r3, vs13);
            }

            __m256 t0 = _mm256_hadd_ps(_mm256_hadd_ps(vs00, vs01), _mm256_hadd_ps(vs02, vs03));
            __m256 t1 = _mm256_hadd_ps(_mm256_hadd_ps(vs10, vs11), _mm256_hadd_ps(vs12, vs13));

            t0 = _mm256_add_ps(t0, _mm256_permute2f128_ps(t0, t0, 1));
            t1 = _mm256_add_ps(t1, _mm256_permute2f128_ps(t1, t1, 1));

            s0 = _mm256_add_ps(s0, t0);
            s1 = _mm256_add_ps(s1, t1);

            _mm_storeu_ps(outptr0 + j, _mm256_castps256_ps128(s0));
            _mm_storeu_ps(outptr1 + j, _mm256_castps256_ps128(s1));
        }

        for( ; j < blockSize; j++ )
        {
            const float* rptr = rowbuf + j*vecsize_aligned;
            float s00, s10;

            if( initOutput )
            {
                s00 = bias0;
                s10 = bias1;
            }
            else
            {
                s00 = outptr0[j];
                s10 = outptr1[j];
            }

            for( int k = 0; k < vecsize; k++ )
            {
                float r0 = rptr[k];
                s00 += wptr0[k]*r0;
                s10 += wptr1[k]*r0;
            }

            outptr0[j] = s00;
            outptr1[j] = s10;
        }
    }
    _mm256_zeroupper();
}

}
}

#endif
