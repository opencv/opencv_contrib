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

#include "opencv2/core.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/xphoto.hpp"

namespace cv
{
namespace xphoto
{

void calculateChannelSums(uint &sumB, uint &sumG, uint &sumR, uchar *src_data, int src_len, float thresh);
void calculateChannelSums(uint64 &sumB, uint64 &sumG, uint64 &sumR, ushort *src_data, int src_len, float thresh);

class GrayworldWBImpl CV_FINAL : public GrayworldWB
{
  private:
    float thresh;

  public:
    GrayworldWBImpl() { thresh = 0.9f; }
    float getSaturationThreshold() const CV_OVERRIDE { return thresh; }
    void setSaturationThreshold(float val) CV_OVERRIDE { thresh = val; }
    void balanceWhite(InputArray _src, OutputArray _dst) CV_OVERRIDE
    {
        CV_Assert(!_src.empty());
        CV_Assert(_src.isContinuous());
        CV_Assert(_src.type() == CV_8UC3 || _src.type() == CV_16UC3);
        Mat src = _src.getMat();

        int N = src.cols * src.rows, N3 = N * 3;

        double dsumB = 0.0, dsumG = 0.0, dsumR = 0.0;
        if (src.type() == CV_8UC3)
        {
            uint sumB = 0, sumG = 0, sumR = 0;
            calculateChannelSums(sumB, sumG, sumR, src.ptr<uchar>(), N3, thresh);
            dsumB = (double)sumB;
            dsumG = (double)sumG;
            dsumR = (double)sumR;
        }
        else if (src.type() == CV_16UC3)
        {
            uint64 sumB = 0, sumG = 0, sumR = 0;
            calculateChannelSums(sumB, sumG, sumR, src.ptr<ushort>(), N3, thresh);
            dsumB = (double)sumB;
            dsumG = (double)sumG;
            dsumR = (double)sumR;
        }

        // Find inverse of averages
        double max_sum = max(dsumB, max(dsumR, dsumG));
        const double eps = 0.1;
        float dinvB = dsumB < eps ? 0.f : (float)(max_sum / dsumB),
              dinvG = dsumG < eps ? 0.f : (float)(max_sum / dsumG),
              dinvR = dsumR < eps ? 0.f : (float)(max_sum / dsumR);

        // Use the inverse of averages as channel gains:
        applyChannelGains(src, _dst, dinvB, dinvG, dinvR);
    }
};

/* Computes sums for each channel, while ignoring saturated pixels which are determined by thresh
 * (version for CV_8UC3)
 */
void calculateChannelSums(uint &sumB, uint &sumG, uint &sumR, uchar *src_data, int src_len, float thresh)
{
    sumB = sumG = sumR = 0;
    ushort thresh255 = (ushort)cvRound(thresh * 255);
    int i = 0;
#if CV_SIMD128
    v_uint8x16 v_inB, v_inG, v_inR, v_min_val, v_max_val;
    v_uint16x8 v_iB1, v_iB2, v_iG1, v_iG2, v_iR1, v_iR2;
    v_uint16x8 v_min1, v_min2, v_max1, v_max2, v_m1, v_m2;
    v_uint16x8 v_255 = v_setall_u16(255), v_thresh = v_setall_u16(thresh255);
    v_uint32x4 v_uint1, v_uint2;
    v_uint32x4 v_SB = v_setzero_u32(), v_SG = v_setzero_u32(), v_SR = v_setzero_u32();

    for (; i < src_len - 47; i += 48)
    {
        // Load 3x uint8x16 and deinterleave into vectors of each channel
        v_load_deinterleave(&src_data[i], v_inB, v_inG, v_inR);

        // Get min and max
        v_min_val = v_min(v_inB, v_min(v_inG, v_inR));
        v_max_val = v_max(v_inB, v_max(v_inG, v_inR));

        // Split into two ushort vectors per channel
        v_expand(v_inB, v_iB1, v_iB2);
        v_expand(v_inG, v_iG1, v_iG2);
        v_expand(v_inR, v_iR1, v_iR2);
        v_expand(v_min_val, v_min1, v_min2);
        v_expand(v_max_val, v_max1, v_max2);

        // Calculate masks
        v_m1 = ~((v_max1 - v_min1) * v_255 > v_thresh * v_max1);
        v_m2 = ~((v_max2 - v_min2) * v_255 > v_thresh * v_max2);

        // Apply masks
        v_iB1 = (v_iB1 & v_m1) + (v_iB2 & v_m2);
        v_iG1 = (v_iG1 & v_m1) + (v_iG2 & v_m2);
        v_iR1 = (v_iR1 & v_m1) + (v_iR2 & v_m2);

        // Split and add to the sums:
        v_expand(v_iB1, v_uint1, v_uint2);
        v_SB += v_uint1 + v_uint2;
        v_expand(v_iG1, v_uint1, v_uint2);
        v_SG += v_uint1 + v_uint2;
        v_expand(v_iR1, v_uint1, v_uint2);
        v_SR += v_uint1 + v_uint2;
    }

    sumB = v_reduce_sum(v_SB);
    sumG = v_reduce_sum(v_SG);
    sumR = v_reduce_sum(v_SR);
#endif
    unsigned int minRGB, maxRGB;
    for (; i < src_len; i += 3)
    {
        minRGB = min(src_data[i], min(src_data[i + 1], src_data[i + 2]));
        maxRGB = max(src_data[i], max(src_data[i + 1], src_data[i + 2]));
        if ((maxRGB - minRGB) * 255 > thresh255 * maxRGB)
            continue;
        sumB += src_data[i];
        sumG += src_data[i + 1];
        sumR += src_data[i + 2];
    }
}

/* Computes sums for each channel, while ignoring saturated pixels which are determined by thresh
 * (version for CV_16UC3)
 */
void calculateChannelSums(uint64 &sumB, uint64 &sumG, uint64 &sumR, ushort *src_data, int src_len, float thresh)
{
    sumB = sumG = sumR = 0;
    uint thresh65535 = cvRound(thresh * 65535);
    int i = 0;
#if CV_SIMD128
    v_uint16x8 v_inB, v_inG, v_inR, v_min_val, v_max_val;
    v_uint32x4 v_iB1, v_iB2, v_iG1, v_iG2, v_iR1, v_iR2;
    v_uint32x4 v_min1, v_min2, v_max1, v_max2, v_m1, v_m2;
    v_uint32x4 v_65535 = v_setall_u32(65535), v_thresh = v_setall_u32(thresh65535);
    v_uint64x2 v_u64_1, v_u64_2;
    v_uint64x2 v_SB = v_setzero_u64(), v_SG = v_setzero_u64(), v_SR = v_setzero_u64();

    for (; i < src_len - 23; i += 24)
    {
        // Load 3x uint16x8 and deinterleave into vectors of each channel
        v_load_deinterleave(&src_data[i], v_inB, v_inG, v_inR);

        // Get min and max
        v_min_val = v_min(v_inB, v_min(v_inG, v_inR));
        v_max_val = v_max(v_inB, v_max(v_inG, v_inR));

        // Split into two uint vectors per channel
        v_expand(v_inB, v_iB1, v_iB2);
        v_expand(v_inG, v_iG1, v_iG2);
        v_expand(v_inR, v_iR1, v_iR2);
        v_expand(v_min_val, v_min1, v_min2);
        v_expand(v_max_val, v_max1, v_max2);

        // Calculate masks
        v_m1 = ~((v_max1 - v_min1) * v_65535 > v_thresh * v_max1);
        v_m2 = ~((v_max2 - v_min2) * v_65535 > v_thresh * v_max2);

        // Apply masks
        v_iB1 = (v_iB1 & v_m1) + (v_iB2 & v_m2);
        v_iG1 = (v_iG1 & v_m1) + (v_iG2 & v_m2);
        v_iR1 = (v_iR1 & v_m1) + (v_iR2 & v_m2);

        // Split and add to the sums:
        v_expand(v_iB1, v_u64_1, v_u64_2);
        v_SB += v_u64_1 + v_u64_2;
        v_expand(v_iG1, v_u64_1, v_u64_2);
        v_SG += v_u64_1 + v_u64_2;
        v_expand(v_iR1, v_u64_1, v_u64_2);
        v_SR += v_u64_1 + v_u64_2;
    }

    // Perform final reduction
    uint64 sum_arr[2];
    v_store(sum_arr, v_SB);
    sumB = sum_arr[0] + sum_arr[1];
    v_store(sum_arr, v_SG);
    sumG = sum_arr[0] + sum_arr[1];
    v_store(sum_arr, v_SR);
    sumR = sum_arr[0] + sum_arr[1];
#endif
    unsigned int minRGB, maxRGB;
    for (; i < src_len; i += 3)
    {
        minRGB = min(src_data[i], min(src_data[i + 1], src_data[i + 2]));
        maxRGB = max(src_data[i], max(src_data[i + 1], src_data[i + 2]));
        if ((maxRGB - minRGB) * 65535 > thresh65535 * maxRGB)
            continue;
        sumB += src_data[i];
        sumG += src_data[i + 1];
        sumR += src_data[i + 2];
    }
}

void applyChannelGains(InputArray _src, OutputArray _dst, float gainB, float gainG, float gainR)
{
    Mat src = _src.getMat();
    CV_Assert(!src.empty());
    CV_Assert(src.isContinuous());
    CV_Assert(src.type() == CV_8UC3 || src.type() == CV_16UC3);

    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();
    int N3 = 3 * src.cols * src.rows;
    int i = 0;

    // Scale gains by their maximum (fixed point approximation works only when all gains are <=1)
    float gain_max = max(gainB, max(gainG, gainR));
    if (gain_max > 0)
    {
        gainB /= gain_max;
        gainG /= gain_max;
        gainR /= gain_max;
    }

    if (src.type() == CV_8UC3)
    {
        // Fixed point arithmetic, mul by 2^8 then shift back 8 bits
        int i_gainB = cvRound(gainB * (1 << 8)), i_gainG = cvRound(gainG * (1 << 8)),
            i_gainR = cvRound(gainR * (1 << 8));
        const uchar *src_data = src.ptr<uchar>();
        uchar *dst_data = dst.ptr<uchar>();
#if CV_SIMD128
        v_uint8x16 v_inB, v_inG, v_inR;
        v_uint8x16 v_outB, v_outG, v_outR;
        v_uint16x8 v_sB1, v_sB2, v_sG1, v_sG2, v_sR1, v_sR2;
        v_uint16x8 v_gainB = v_setall_u16((ushort)i_gainB), v_gainG = v_setall_u16((ushort)i_gainG),
                   v_gainR = v_setall_u16((ushort)i_gainR);

        for (; i < N3 - 47; i += 48)
        {
            // Load 3x uint8x16 and deinterleave into vectors of each channel
            v_load_deinterleave(&src_data[i], v_inB, v_inG, v_inR);

            // Split into two ushort vectors per channel
            v_expand(v_inB, v_sB1, v_sB2);
            v_expand(v_inG, v_sG1, v_sG2);
            v_expand(v_inR, v_sR1, v_sR2);

            // Multiply by gains
            v_sB1 = (v_sB1 * v_gainB) >> 8;
            v_sB2 = (v_sB2 * v_gainB) >> 8;
            v_sG1 = (v_sG1 * v_gainG) >> 8;
            v_sG2 = (v_sG2 * v_gainG) >> 8;
            v_sR1 = (v_sR1 * v_gainR) >> 8;
            v_sR2 = (v_sR2 * v_gainR) >> 8;

            // Pack into vectors of v_uint8x16
            v_store_interleave(&dst_data[i], v_pack(v_sB1, v_sB2), v_pack(v_sG1, v_sG2), v_pack(v_sR1, v_sR2));
        }
#endif
        for (; i < N3; i += 3)
        {
            dst_data[i] = (uchar)((src_data[i] * i_gainB) >> 8);
            dst_data[i + 1] = (uchar)((src_data[i + 1] * i_gainG) >> 8);
            dst_data[i + 2] = (uchar)((src_data[i + 2] * i_gainR) >> 8);
        }
    }
    else if (src.type() == CV_16UC3)
    {
        // Fixed point arithmetic, mul by 2^16 then shift back 16 bits
        int i_gainB = cvRound(gainB * (1 << 16)), i_gainG = cvRound(gainG * (1 << 16)),
            i_gainR = cvRound(gainR * (1 << 16));
        const ushort *src_data = src.ptr<ushort>();
        ushort *dst_data = dst.ptr<ushort>();
#if CV_SIMD128
        v_uint16x8 v_inB, v_inG, v_inR;
        v_uint16x8 v_outB, v_outG, v_outR;
        v_uint32x4 v_sB1, v_sB2, v_sG1, v_sG2, v_sR1, v_sR2;
        v_uint32x4 v_gainB = v_setall_u32((uint)i_gainB), v_gainG = v_setall_u32((uint)i_gainG),
                   v_gainR = v_setall_u32((uint)i_gainR);

        for (; i < N3 - 23; i += 24)
        {
            // Load 3x uint16x8 and deinterleave into vectors of each channel
            v_load_deinterleave(&src_data[i], v_inB, v_inG, v_inR);

            // Split into two uint vectors per channel
            v_expand(v_inB, v_sB1, v_sB2);
            v_expand(v_inG, v_sG1, v_sG2);
            v_expand(v_inR, v_sR1, v_sR2);

            // Multiply by scaling factors
            v_sB1 = (v_sB1 * v_gainB) >> 16;
            v_sB2 = (v_sB2 * v_gainB) >> 16;
            v_sG1 = (v_sG1 * v_gainG) >> 16;
            v_sG2 = (v_sG2 * v_gainG) >> 16;
            v_sR1 = (v_sR1 * v_gainR) >> 16;
            v_sR2 = (v_sR2 * v_gainR) >> 16;

            // Pack into vectors of v_uint16x8
            v_store_interleave(&dst_data[i], v_pack(v_sB1, v_sB2), v_pack(v_sG1, v_sG2), v_pack(v_sR1, v_sR2));
        }
#endif
        for (; i < N3; i += 3)
        {
            dst_data[i] = (ushort)((src_data[i] * i_gainB) >> 16);
            dst_data[i + 1] = (ushort)((src_data[i + 1] * i_gainG) >> 16);
            dst_data[i + 2] = (ushort)((src_data[i + 2] * i_gainR) >> 16);
        }
    }
}

Ptr<GrayworldWB> createGrayworldWB() { return makePtr<GrayworldWBImpl>(); }
}
}
