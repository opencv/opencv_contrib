// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _RLOF_INVOKERBASE_HPP_
#define _RLOF_INVOKERBASE_HPP_

#ifndef CV_DESCALE
#define CV_DESCALE(x, n)     (((x) + (1 << ((n)-1))) >> (n))
#endif
#define FLT_RESCALE 1

#include "rlof_localflow.h"
#include <unordered_map>
#include "opencv2/core/hal/intrin.hpp"
using namespace std;
using namespace cv;
namespace cv {
namespace optflow {


typedef short deriv_type;
#if CV_SIMD128

static inline void getVBitMask(const int & width, v_int32x4 & mask0, v_int32x4 & mask1)
{
    int noBits = width - static_cast<int>(floor(width / 8.f) * 8.f);
    unsigned int val[8];
    for (int n = 0; n < 8; n++)
    {
        val[n] = (noBits > n) ? (std::numeric_limits<unsigned int>::max()) : 0;
    }
    mask0 = v_int32x4(val[0], val[1], val[2], val[3]);
    mask1 = v_int32x4(val[4], val[5], val[6], val[7]);
}
#endif
typedef uchar tMaskType;
#define tCVMaskType CV_8UC1
#define MaskSet 0xffffffff

static inline void copyWinBuffers(int iw00, int iw01, int iw10, int iw11,
    Size winSize,
    const Mat & I, const Mat & derivI, const Mat & winMaskMat,
    Mat & IWinBuf, Mat & derivIWinBuf,
    Point iprevPt)
{
    int cn = I.channels(), cn2 = cn * 2;
    const int W_BITS = 14;
#if CV_SIMD128
    v_int16x8 vqw0((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
    v_int16x8 vqw1((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
    v_int32x4 vdelta_d = v_setall_s32(1 << (W_BITS - 1));
    v_int32x4 vdelta = v_setall_s32(1 << (W_BITS - 5 - 1));
    v_int32x4 vmax_val_32 = v_setall_s32(std::numeric_limits<unsigned int>::max());
    v_int32x4 vmask_border_0, vmask_border_1;
    getVBitMask(winSize.width, vmask_border_0, vmask_border_1);
#endif

    // extract the patch from the first image, compute covariation matrix of derivatives
    int x, y;
    for (y = 0; y < winSize.height; y++)
    {
        const uchar* src = I.ptr<uchar>(y + iprevPt.y, 0) + iprevPt.x*cn;
        const uchar* src1 = I.ptr<uchar>(y + iprevPt.y + 1, 0) + iprevPt.x*cn;
        const short* dsrc = derivI.ptr<short>(y + iprevPt.y, 0) + iprevPt.x*cn2;
        const short* dsrc1 = derivI.ptr<short>(y + iprevPt.y + 1, 0) + iprevPt.x*cn2;
        short* Iptr = IWinBuf.ptr<short>(y, 0);
        short* dIptr = derivIWinBuf.ptr<short>(y, 0);
        const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
        x = 0;
#if CV_SIMD128

        for (; x <= winSize.width*cn; x += 8, dsrc += 8 * 2, dsrc1 += 8 * 2, dIptr += 8 * 2)
        {
            v_int32x4 vmask0 = v_reinterpret_as_s32(v_load_expand_q(maskPtr + x)) * vmax_val_32;
            v_int32x4 vmask1 = v_reinterpret_as_s32(v_load_expand_q(maskPtr + x + 4)) * vmax_val_32;
            if (x + 4 > winSize.width)
            {
                vmask0 = vmask0 & vmask_border_0;
            }
            if (x + 8 > winSize.width)
            {
                vmask1 = vmask1 & vmask_border_1;
            }

            v_int32x4 t0, t1;
            v_int16x8 v00, v01, v10, v11, t00, t01, t10, t11;
            v00 = v_reinterpret_as_s16(v_load_expand(src + x));
            v01 = v_reinterpret_as_s16(v_load_expand(src + x + cn));
            v10 = v_reinterpret_as_s16(v_load_expand(src1 + x));
            v11 = v_reinterpret_as_s16(v_load_expand(src1 + x + cn));

            v_zip(v00, v01, t00, t01);
            v_zip(v10, v11, t10, t11);
            t0 = v_dotprod(t00, vqw0, vdelta) + v_dotprod(t10, vqw1);
            t1 = v_dotprod(t01, vqw0, vdelta) + v_dotprod(t11, vqw1);
            t0 = t0 >> (W_BITS - 5)  & vmask0;
            t1 = t1 >> (W_BITS - 5)  & vmask1;
            v_store(Iptr + x, v_pack(t0, t1));

            v00 = v_reinterpret_as_s16(v_load(dsrc));
            v01 = v_reinterpret_as_s16(v_load(dsrc + cn2));
            v10 = v_reinterpret_as_s16(v_load(dsrc1));
            v11 = v_reinterpret_as_s16(v_load(dsrc1 + cn2));

            v_zip(v00, v01, t00, t01);
            v_zip(v10, v11, t10, t11);

            t0 = v_dotprod(t00, vqw0, vdelta_d) + v_dotprod(t10, vqw1);
            t1 = v_dotprod(t01, vqw0, vdelta_d) + v_dotprod(t11, vqw1);
            t0 = t0 >> W_BITS;
            t1 = t1 >> W_BITS;
            v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
            v00 = v00 & v_reinterpret_as_s16(vmask0);
            v_store(dIptr, v00);

            v00 = v_reinterpret_as_s16(v_load(dsrc + 4 * 2));
            v01 = v_reinterpret_as_s16(v_load(dsrc + 4 * 2 + cn2));
            v10 = v_reinterpret_as_s16(v_load(dsrc1 + 4 * 2));
            v11 = v_reinterpret_as_s16(v_load(dsrc1 + 4 * 2 + cn2));

            v_zip(v00, v01, t00, t01);
            v_zip(v10, v11, t10, t11);

            t0 = v_dotprod(t00, vqw0, vdelta_d) + v_dotprod(t10, vqw1);
            t1 = v_dotprod(t01, vqw0, vdelta_d) + v_dotprod(t11, vqw1);
            t0 = t0 >> W_BITS;
            t1 = t1 >> W_BITS;
            v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
            v00 = v00 & v_reinterpret_as_s16(vmask1);
            v_store(dIptr + 4 * 2, v00);
        }
#else
        for (; x < winSize.width*cn; x++, dsrc += 2, dsrc1 += 2, dIptr += 2)
        {
            if (maskPtr[x] == 0)
            {
                dIptr[0] = 0;
                dIptr[1] = 0;
                continue;
            }
            int ival = CV_DESCALE(src[x] * iw00 + src[x + cn] * iw01 +
                src1[x] * iw10 + src1[x + cn] * iw11, W_BITS - 5);
            int ixval = CV_DESCALE(dsrc[0] * iw00 + dsrc[cn2] * iw01 +
                dsrc1[0] * iw10 + dsrc1[cn2] * iw11, W_BITS);
            int iyval = CV_DESCALE(dsrc[1] * iw00 + dsrc[cn2 + 1] * iw01 + dsrc1[1] * iw10 +
                dsrc1[cn2 + 1] * iw11, W_BITS);

            Iptr[x] = (short)ival;
            dIptr[0] = (short)ixval;
            dIptr[1] = (short)iyval;
        }
#endif
    }
}

static inline void copyWinBuffers(int iw00, int iw01, int iw10, int iw11,
    Size winSize,
    const Mat & I, const Mat & derivI, const Mat & winMaskMat,
    Mat & IWinBuf, Mat & derivIWinBuf,
    float & A11, float & A22, float & A12,
    Point iprevPt)
{
    const float FLT_SCALE = (1.f / (1 << 20));
    int cn = I.channels(), cn2 = cn * 2;
    const int W_BITS = 14;
#if CV_SIMD128
    v_int16x8 vqw0((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
    v_int16x8 vqw1((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
    v_int32x4 vdelta_d = v_setall_s32(1 << (W_BITS - 1));
    v_int32x4 vdelta = v_setall_s32(1 << (W_BITS - 5 - 1));
    v_int32x4 vmax_val_32 = v_setall_s32(std::numeric_limits<unsigned int>::max());
    v_int32x4 vmask_border0, vmask_border1;
    v_float32x4 vA11 = v_setzero_f32(), vA22 = v_setzero_f32(), vA12 = v_setzero_f32();
    getVBitMask(winSize.width, vmask_border0, vmask_border1);
#endif

    // extract the patch from the first image, compute covariation matrix of derivatives
    for (int y = 0; y < winSize.height; y++)
    {
        const uchar* src = I.ptr<uchar>(y + iprevPt.y, 0) + iprevPt.x*cn;
        const uchar* src1 = I.ptr<uchar>(y + iprevPt.y + 1, 0) + iprevPt.x*cn;
        const short* dsrc = derivI.ptr<short>(y + iprevPt.y, 0) + iprevPt.x*cn2;
        const short* dsrc1 = derivI.ptr<short>(y + iprevPt.y + 1, 0) + iprevPt.x*cn2;
        short* Iptr = IWinBuf.ptr<short>(y, 0);
        short* dIptr = derivIWinBuf.ptr<short>(y, 0);
        const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
#if CV_SIMD128
        for (int x = 0; x <= winSize.width*cn; x += 8, dsrc += 8 * 2, dsrc1 += 8 * 2, dIptr += 8 * 2)
        {
            v_int32x4 vmask0 = v_reinterpret_as_s32(v_load_expand_q(maskPtr + x)) * vmax_val_32;
            v_int32x4 vmask1 = v_reinterpret_as_s32(v_load_expand_q(maskPtr + x + 4)) * vmax_val_32;
            if (x + 4 > winSize.width)
            {
                vmask0 = vmask0 & vmask_border0;
            }
            if (x + 8 > winSize.width)
            {
                vmask1 = vmask1 & vmask_border1;
            }

            v_int32x4 t0, t1;
            v_int16x8 v00, v01, v10, v11, t00, t01, t10, t11;
            v00 = v_reinterpret_as_s16(v_load_expand(src + x));
            v01 = v_reinterpret_as_s16(v_load_expand(src + x + cn));
            v10 = v_reinterpret_as_s16(v_load_expand(src1 + x));
            v11 = v_reinterpret_as_s16(v_load_expand(src1 + x + cn));

            v_zip(v00, v01, t00, t01);
            v_zip(v10, v11, t10, t11);
            t0 = v_dotprod(t00, vqw0, vdelta) + v_dotprod(t10, vqw1);
            t1 = v_dotprod(t01, vqw0, vdelta) + v_dotprod(t11, vqw1);
            t0 = t0 >> (W_BITS - 5);
            t1 = t1 >> (W_BITS - 5);
            t0 = t0 & vmask0;
            t1 = t1 & vmask1;
            v_store(Iptr + x, v_pack(t0, t1));

            v00 = v_reinterpret_as_s16(v_load(dsrc));
            v01 = v_reinterpret_as_s16(v_load(dsrc + cn2));
            v10 = v_reinterpret_as_s16(v_load(dsrc1));
            v11 = v_reinterpret_as_s16(v_load(dsrc1 + cn2));

            v_zip(v00, v01, t00, t01);
            v_zip(v10, v11, t10, t11);

            t0 = v_dotprod(t00, vqw0, vdelta_d) + v_dotprod(t10, vqw1);
            t1 = v_dotprod(t01, vqw0, vdelta_d) + v_dotprod(t11, vqw1);
            t0 = t0 >> W_BITS;
            t1 = t1 >> W_BITS;
            v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
            v00 = v00 & v_reinterpret_as_s16(vmask0);
            v_store(dIptr, v00);

            v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(v00))));
            v_expand(v00, t1, t0);

            v_float32x4 fy = v_cvt_f32(t0);
            v_float32x4 fx = v_cvt_f32(t1);

            vA22 = v_muladd(fy, fy, vA22);
            vA12 = v_muladd(fx, fy, vA12);
            vA11 = v_muladd(fx, fx, vA11);

            v00 = v_reinterpret_as_s16(v_load(dsrc + 4 * 2));
            v01 = v_reinterpret_as_s16(v_load(dsrc + 4 * 2 + cn2));
            v10 = v_reinterpret_as_s16(v_load(dsrc1 + 4 * 2));
            v11 = v_reinterpret_as_s16(v_load(dsrc1 + 4 * 2 + cn2));

            v_zip(v00, v01, t00, t01);
            v_zip(v10, v11, t10, t11);

            t0 = v_dotprod(t00, vqw0, vdelta_d) + v_dotprod(t10, vqw1);
            t1 = v_dotprod(t01, vqw0, vdelta_d) + v_dotprod(t11, vqw1);
            t0 = t0 >> W_BITS;
            t1 = t1 >> W_BITS;
            v00 = v_pack(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
            v00 = v00 & v_reinterpret_as_s16(vmask1);
            v_store(dIptr + 4 * 2, v00);

            v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(v00))));
            v_expand(v00, t1, t0);

            fy = v_cvt_f32(t0);
            fx = v_cvt_f32(t1);

            vA22 = v_muladd(fy, fy, vA22);
            vA12 = v_muladd(fx, fy, vA12);
            vA11 = v_muladd(fx, fx, vA11);
        }
#else
        for (int x = 0; x < winSize.width*cn; x++, dsrc += 2, dsrc1 += 2, dIptr += 2)
        {
            if (maskPtr[x] == 0)
            {
                dIptr[0] = 0;
                dIptr[1] = 0;
                continue;
            }
            int ival = CV_DESCALE(src[x] * iw00 + src[x + cn] * iw01 +
                src1[x] * iw10 + src1[x + cn] * iw11, W_BITS - 5);
            int ixval = CV_DESCALE(dsrc[0] * iw00 + dsrc[cn2] * iw01 +
                dsrc1[0] * iw10 + dsrc1[cn2] * iw11, W_BITS);
            int iyval = CV_DESCALE(dsrc[1] * iw00 + dsrc[cn2 + 1] * iw01 + dsrc1[1] * iw10 +
                dsrc1[cn2 + 1] * iw11, W_BITS);

            Iptr[x] = (short)ival;
            dIptr[0] = (short)ixval;
            dIptr[1] = (short)iyval;
            A11 += (float)(ixval*ixval);
            A12 += (float)(ixval*iyval);
            A22 += (float)(iyval*iyval);
        }
#endif
    }
#if CV_SIMD128
    A11 += v_reduce_sum(vA11);
    A12 += v_reduce_sum(vA12);
    A22 += v_reduce_sum(vA22);
#endif

    A11 *= FLT_SCALE;
    A12 *= FLT_SCALE;
    A22 *= FLT_SCALE;
}

static void getLocalPatch(
        const cv::Mat & src,
        const cv::Point2i & prevPoint, // feature points
        cv::Mat & winPointMask,
        int & noPoints,
        cv::Rect & winRoi,
        int minWinSize)
{
    int maxWinSizeH = (winPointMask.cols - 1) / 2;
    winRoi.x = prevPoint.x;// - maxWinSizeH;
    winRoi.y = prevPoint.y;// - maxWinSizeH;
    winRoi.width  =  winPointMask.cols;
    winRoi.height =  winPointMask.rows;

    if( minWinSize == winPointMask.cols || prevPoint.x < 0 || prevPoint.y < 0
        || prevPoint.x + 2*maxWinSizeH >= src.cols || prevPoint.y + 2*maxWinSizeH >= src.rows)
    {
        winRoi.x = prevPoint.x - maxWinSizeH;
        winRoi.y = prevPoint.y - maxWinSizeH;
        winPointMask.setTo(1);
        noPoints = winPointMask.size().area();
        return;
    }
    winPointMask.setTo(0);
    noPoints = 0;
    int c            = prevPoint.x + maxWinSizeH;
    int r            = prevPoint.y + maxWinSizeH;
    int min_c = c;
    int max_c = c;
    int border_left    = c - maxWinSizeH;
    int border_top    = r - maxWinSizeH;
    cv::Vec4i bounds = src.at<cv::Vec4i>(r,c);
    int min_r = bounds.val[2];
    int max_r = bounds.val[3];

    for( int _r = min_r; _r <= max_r; _r++)
    {
        cv::Rect roi(maxWinSizeH, _r - border_top,  winPointMask.cols, 1);
        if( _r >= 0 && _r < src.cols)
        {
            bounds = src.at<cv::Vec4i>(_r,c);
            roi.x      = bounds.val[0] - border_left;
            roi.width = bounds.val[1] - bounds.val[0];
            cv::Mat(winPointMask, roi).setTo(1);
        }
        else
        {
            bounds.val[0] = border_left;
            bounds.val[1] = border_left + roi.width;
        }
        min_c = MIN(min_c, bounds.val[0]);
        max_c = MAX(max_c, bounds.val[1]);
        noPoints += roi.width;
    }

    if( noPoints < minWinSize * minWinSize)
    {
        cv::Rect roi(    winPointMask.cols / 2 - (minWinSize-1)/2,
                        winPointMask.rows / 2 - (minWinSize-1)/2,
                        minWinSize, minWinSize);
        cv::Mat(winPointMask, roi).setTo(1);
        roi.x += border_left;
        roi.y += border_top;
        min_c = MIN(MIN(min_c, roi.tl().x),roi.br().x);
        max_c = MAX(MAX(max_c, roi.tl().x),roi.br().x);
        min_r = MIN(MIN(min_r, roi.tl().y),roi.br().y);
        max_r = MAX(MAX(max_r, roi.tl().y),roi.br().y);
        noPoints += minWinSize * minWinSize;
    }
    winRoi.x = min_c - maxWinSizeH;
    winRoi.y = min_r - maxWinSizeH;
    winRoi.width  =  max_c - min_c;
    winRoi.height =  max_r - min_r;
    winPointMask = winPointMask(cv::Rect(min_c - border_left, min_r - border_top, winRoi.width, winRoi.height));
}

static inline
bool calcWinMaskMat(
        const cv::Mat & BI,
        const int windowType,
        const cv::Point2i & iprevPt,
        cv::Mat & winMaskMat,
        cv::Size & winSize,
        cv::Point2f & halfWin,
        int & winArea,
        const int minWinSize,
        const int maxWinSize)
{
    if (windowType == SR_CROSS && maxWinSize != minWinSize)
    {
        // patch generation
        cv::Rect winRoi;
        getLocalPatch(BI, iprevPt, winMaskMat, winArea, winRoi, minWinSize);
        if (winArea == 0)
            return false;
        winSize = winRoi.size();
        halfWin = Point2f(static_cast<float>(iprevPt.x - winRoi.tl().x),
                          static_cast<float>(iprevPt.y - winRoi.tl().y));
    }
    else
    {
        winSize = cv::Size(maxWinSize, maxWinSize);
        halfWin = Point2f((winSize.width - 1) / 2.f, (winSize.height - 1) / 2.f);
        winMaskMat.setTo(1);
    }
    return true;
}

static inline
short estimateScale(cv::Mat & residuals)
{
    cv::Mat absMat = cv::abs(residuals);
    return quickselect<short>(absMat, absMat.rows / 2);
}

}} // namespace
#endif
