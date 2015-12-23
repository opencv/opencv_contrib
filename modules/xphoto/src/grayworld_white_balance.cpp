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
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace xphoto {

    void autowbGrayworld(InputArray _src, OutputArray _dst, float thresh)
    {

        Mat src = _src.getMat();
        CV_Assert(!src.empty());
        CV_Assert(src.isContinuous());

        // TODO: Handle CV_8UC1
        // TODO: Handle types other than CV_8U
        CV_Assert(src.type() == CV_8UC3);

        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();
        CV_Assert(dst.isContinuous());

        int width  = src.cols,
            height = src.rows,
            N      = width*height,
            N3     = N*3;

        // Calculate sum of pixel values of each channel
        const uchar* src_data = src.ptr<uchar>(0);
        unsigned long sum1 = 0, sum2 = 0, sum3 = 0;
        unsigned int thresh255 = cvRound(thresh * 255);
        int i = 0;
#if CV_SIMD128
        v_uint8x16 v_inB, v_inG, v_inR;
        v_uint16x8 v_s1, v_s2;
        v_uint32x4 v_iB1, v_iB2, v_iB3, v_iB4,
                   v_iG1, v_iG2, v_iG3, v_iG4,
                   v_iR1, v_iR2, v_iR3, v_iR4,
                   v_255 = v_setall_u32(255),
                   v_thresh = v_setall_u32(thresh255),
                   v_min1, v_min2, v_min3, v_min4,
                   v_max1, v_max2, v_max3, v_max4,
                   v_m1, v_m2, v_m3, v_m4,
                   v_SB = v_setzero_u32(),
                   v_SG = v_setzero_u32(),
                   v_SR = v_setzero_u32();

        for ( ; i < N3 - 47; i += 48 )
        {
            // NOTE: This block assumes BGR channels in naming variables

            // Load 3x uint8x16 and deinterleave into vectors of each channel
            v_load_deinterleave(&src_data[i], v_inB, v_inG, v_inR);

            // Split into four int vectors per channel
            v_expand(v_inB, v_s1, v_s2);
            v_expand(v_s1, v_iB1, v_iB2);
            v_expand(v_s2, v_iB3, v_iB4);

            v_expand(v_inG, v_s1, v_s2);
            v_expand(v_s1, v_iG1, v_iG2);
            v_expand(v_s2, v_iG3, v_iG4);

            v_expand(v_inR, v_s1, v_s2);
            v_expand(v_s1, v_iR1, v_iR2);
            v_expand(v_s2, v_iR3, v_iR4);

            // Get mins and maxs
            v_min1 = v_min(v_iB1, v_min(v_iG1, v_iR1));
            v_min2 = v_min(v_iB2, v_min(v_iG2, v_iR2));
            v_min3 = v_min(v_iB3, v_min(v_iG3, v_iR3));
            v_min4 = v_min(v_iB4, v_min(v_iG4, v_iR4));

            v_max1 = v_max(v_iB1, v_max(v_iG1, v_iR1));
            v_max2 = v_max(v_iB2, v_max(v_iG2, v_iR2));
            v_max3 = v_max(v_iB3, v_max(v_iG3, v_iR3));
            v_max4 = v_max(v_iB4, v_max(v_iG4, v_iR4));

            // Calculate masks
            v_m1 = ~((v_max1 - v_min1) * v_255 > v_thresh * v_max1);
            v_m2 = ~((v_max2 - v_min2) * v_255 > v_thresh * v_max2);
            v_m3 = ~((v_max3 - v_min3) * v_255 > v_thresh * v_max3);
            v_m4 = ~((v_max4 - v_min4) * v_255 > v_thresh * v_max4);

            // Apply mask
            v_SB += (v_iB1 & v_m1) + (v_iB2 & v_m2) + (v_iB3 & v_m3) + (v_iB4 & v_m4);
            v_SG += (v_iG1 & v_m1) + (v_iG2 & v_m2) + (v_iG3 & v_m3) + (v_iG4 & v_m4);
            v_SR += (v_iR1 & v_m1) + (v_iR2 & v_m2) + (v_iR3 & v_m3) + (v_iR4 & v_m4);
        }

        // Perform final reduction
        sum1 = v_reduce_sum(v_SB);
        sum2 = v_reduce_sum(v_SG);
        sum3 = v_reduce_sum(v_SR);
#endif
        unsigned int minRGB, maxRGB;
        for ( ; i < N3; i += 3 )
        {
            minRGB = min(src_data[i], min(src_data[i + 1], src_data[i + 2]));
            maxRGB = max(src_data[i], max(src_data[i + 1], src_data[i + 2]));
            if ( (maxRGB - minRGB) * 255 > thresh255 * maxRGB ) continue;
            sum1 += src_data[i];
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
        if ( inv_max > 0 )
        {
            inv1 = (float)((double)inv1 / inv_max);
            inv2 = (float)((double)inv2 / inv_max);
            inv3 = (float)((double)inv3 / inv_max);
        }

        // Fixed point arithmetic, mul by 2^8 then shift back 8 bits
        int i_inv1 = cvRound(inv1 * (1 << 8)),
            i_inv2 = cvRound(inv2 * (1 << 8)),
            i_inv3 = cvRound(inv3 * (1 << 8));

        // Scale input pixel values
        uchar* dst_data = dst.ptr<uchar>(0);
        i = 0;
#if CV_SIMD128
        v_uint8x16 v_outB, v_outG, v_outR;
        v_uint16x8 v_sB1, v_sB2, v_sG1, v_sG2, v_sR1, v_sR2,
                   v_invB = v_setall_u16((unsigned short) i_inv1),
                   v_invG = v_setall_u16((unsigned short) i_inv2),
                   v_invR = v_setall_u16((unsigned short) i_inv3);

        for ( ; i < N3 - 47; i += 48 )
        {
            // Load 16 x 8bit uchars
            v_load_deinterleave(&src_data[i], v_inB, v_inG, v_inR);

            // Split into four int vectors per channel
            v_expand(v_inB, v_sB1, v_sB2);
            v_expand(v_inG, v_sG1, v_sG2);
            v_expand(v_inR, v_sR1, v_sR2);

            // Multiply by scaling factors
            v_sB1 = (v_sB1 * v_invB) >> 8;
            v_sB2 = (v_sB2 * v_invB) >> 8;
            v_sG1 = (v_sG1 * v_invG) >> 8;
            v_sG2 = (v_sG2 * v_invG) >> 8;
            v_sR1 = (v_sR1 * v_invR) >> 8;
            v_sR2 = (v_sR2 * v_invR) >> 8;

            // Pack into vectors of v_uint8x16
            v_store_interleave(&dst_data[i], v_pack(v_sB1, v_sB2),
                v_pack(v_sG1, v_sG2), v_pack(v_sR1, v_sR2));
        }
#endif
        for ( ; i < N3; i += 3 )
        {
            dst_data[i]     = (uchar)((src_data[i]     * i_inv1) >> 8);
            dst_data[i + 1] = (uchar)((src_data[i + 1] * i_inv2) >> 8);
            dst_data[i + 2] = (uchar)((src_data[i + 2] * i_inv3) >> 8);
        }
    }

}}
