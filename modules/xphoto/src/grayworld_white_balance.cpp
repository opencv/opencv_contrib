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

#include "opencv2/xphoto.hpp"

#include "opencv2/core.hpp"
#include "opencv2/hal/intrin.hpp"

namespace cv { namespace xphoto {

    /*!
     */
    void autowbGrayworld(InputArray _src, OutputArray _dst)
    {

        Mat src = _src.getMat();
        CV_Assert(!src.empty());
        CV_Assert(src.isContinuous());

        // TODO: Handle CV_8UC1
        // TODO: Handle types other than CV_8U
        CV_Assert(src.type() == CV_8UC3);

        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();

        int width  = src.cols,
            height = src.rows,
            N      = width*height,
            N3     = N*3;

        // Calculate sum of pixel values of each channel
        const uchar* src_data = src.ptr<uchar>(0);
        ulong sum1 = 0, sum2 = 0, sum3 = 0;
        int i = 0;
#if CV_SIMD128
        v_uint8x16 v_in;
        v_uint16x8 v_s1, v_s2;
        v_uint32x4 v_i1, v_i2, v_i3, v_i4,
                   v_S1 = v_setzero_u32(),
                   v_S2 = v_setzero_u32(),
                   v_S3 = v_setzero_u32(),
                   v_S4 = v_setzero_u32();

        for (; i < N3 - 14; i += 15)
        {
            // Load 16 x 8bit uchars
            v_in = v_load(&src_data[i]);

            // Split into two vectors of 8 ushorts
            v_expand(v_in, v_s1, v_s2);

            // Split into four vectors of 4 uints
            v_expand(v_s1, v_i1, v_i2);
            v_expand(v_s2, v_i3, v_i4);

            // Add to accumulators
            v_S1 += v_i1;
            v_S2 += v_i2;
            v_S3 += v_i3;
            v_S4 += v_i4;
        }

        // Store accumulated values into memory
        uint sums[16];
        v_store(&sums[0],  v_S1);
        v_store(&sums[4],  v_S2);
        v_store(&sums[8],  v_S3);
        v_store(&sums[12], v_S4);

        // Perform final reduction
        sum1 = sums[0] + sums[3] + sums[6] + sums[9]  + sums[12],
        sum2 = sums[1] + sums[4] + sums[7] + sums[10] + sums[13],
        sum3 = sums[2] + sums[5] + sums[8] + sums[11] + sums[14];
#endif
        for (; i < N3; i += 3)
        {
            sum1 += src_data[i + 0];
            sum2 += src_data[i + 1];
            sum3 += src_data[i + 2];
        }

        // Find inverse of averages
        double dinv1 = sum1 == 0 ? 0.f : (double)N / (double)sum1,
               dinv2 = sum2 == 0 ? 0.f : (double)N / (double)sum2,
               dinv3 = sum3 == 0 ? 0.f : (double)N / (double)sum3;

        // Find maximum
        double inv_max = max(dinv1, max(dinv2, dinv3));

        // Convert to floats
        float inv1 = (float) dinv1,
              inv2 = (float) dinv2,
              inv3 = (float) dinv3;

        // Scale by maximum
        if (inv_max > 0)
        {
            inv1 /= inv_max;
            inv2 /= inv_max;
            inv3 /= inv_max;
        }

        // Scale input pixel values
        uchar* dst_data = dst.ptr<uchar>(0);
        i = 0;
#if CV_SIMD128
        v_uint8x16  v_out;
        v_float32x4 v_f1, v_f2, v_f3, v_f4,
                    scal1(inv1, inv2, inv3, inv1),
                    scal2(inv2, inv3, inv1, inv2),
                    scal3(inv3, inv1, inv2, inv3),
                    scal4(inv1, inv2, inv3, 0.f);

        for (; i < N3 - 14; i += 15)
        {
            // Load 16 x 8bit uchars
            v_in = v_load(&src_data[i]);

            // Split into two vectors of 8 ushorts
            v_expand(v_in, v_s1, v_s2);

            // Split into four vectors of 4 uints
            v_expand(v_s1, v_i1, v_i2);
            v_expand(v_s2, v_i3, v_i4);

            // Convert into four vectors of 4 floats
            v_f1 = v_cvt_f32(v_reinterpret_as_s32(v_i1));
            v_f2 = v_cvt_f32(v_reinterpret_as_s32(v_i2));
            v_f3 = v_cvt_f32(v_reinterpret_as_s32(v_i3));
            v_f4 = v_cvt_f32(v_reinterpret_as_s32(v_i4));

            // Multiply by scaling factors
            v_f1 *= scal1;
            v_f2 *= scal2;
            v_f3 *= scal3;
            v_f4 *= scal4;

            // Convert back into four vectors of 4 uints
            v_i1 = v_reinterpret_as_u32(v_round(v_f1));
            v_i2 = v_reinterpret_as_u32(v_round(v_f2));
            v_i3 = v_reinterpret_as_u32(v_round(v_f3));
            v_i4 = v_reinterpret_as_u32(v_round(v_f4));

            // Pack into two vectors of 8 ushorts
            v_s1 = v_pack(v_i1, v_i2);
            v_s2 = v_pack(v_i3, v_i4);

            // Pack into vector of 16 uchars
            v_out = v_pack(v_s1, v_s2);

            // Store
            v_store(&dst_data[i], v_out);
        }
#endif
        for (; i < N3; i += 3)
        {
            dst_data[i + 0] = src_data[i + 0] * inv1;
            dst_data[i + 1] = src_data[i + 1] * inv2;
            dst_data[i + 2] = src_data[i + 2] * inv3;
        }
    }

}}
