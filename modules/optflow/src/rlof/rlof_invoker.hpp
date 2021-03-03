// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _RLOF_INVOKER_HPP_
#define _RLOF_INVOKER_HPP_
#include "rlof_invokerbase.hpp"
namespace cv {
namespace optflow {
namespace rlof {
namespace ica {

class TrackerInvoker : public cv::ParallelLoopBody
{
public:
    TrackerInvoker(
        const Mat&      _prevImg,
        const Mat&      _prevDeriv,
        const Mat&      _nextImg,
        const Mat&      _rgbPrevImg,
        const Mat&      _rgbNextImg,
        const Point2f*  _prevPts,
        Point2f*        _nextPts,
        uchar*          _status,
        float*          _err,
        int             _level,
        int             _maxLevel,
        int             _winSize[2],
        int             _maxIteration,
        bool            _useInitialFlow,
        int             _supportRegionType,
        const std::vector<float>& _normSigmas,
        float           _minEigenValue,
        int             _crossSegmentationThreshold
    ) :
        normSigma0(_normSigmas[0]),
        normSigma1(_normSigmas[1]),
        normSigma2(_normSigmas[2])
    {
        prevImg = &_prevImg;
        prevDeriv = &_prevDeriv;
        nextImg = &_nextImg;
        rgbPrevImg = &_rgbPrevImg;
        rgbNextImg = &_rgbNextImg;
        prevPts = _prevPts;
        nextPts = _nextPts;
        status = _status;
        err = _err;
        minWinSize = _winSize[0];
        maxWinSize = _winSize[1];
        criteria.maxCount = _maxIteration;
        criteria.epsilon = 0.01;
        level = _level;
        maxLevel = _maxLevel;
        windowType = _supportRegionType;
        minEigThreshold = _minEigenValue;
        useInitialFlow = _useInitialFlow;
        crossSegmentationThreshold = _crossSegmentationThreshold;
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        cv::Size    winSize;
        cv::Point2f halfWin;

        const Mat& I = *prevImg;
        const Mat& J = *nextImg;
        const Mat& derivI = *prevDeriv;
        const Mat& BI = *rgbPrevImg;

        winSize = cv::Size(maxWinSize, maxWinSize);
        int winMaskwidth = roundUp(winSize.width, 8) * 2;
        cv::Mat winMaskMatBuf(winMaskwidth, winMaskwidth, tCVMaskType);
        winMaskMatBuf.setTo(1);

        const float FLT_SCALE = (1.f / (1 << 20)); // 20

        int cn = I.channels(), cn2 = cn * 2;
        int winbufwidth = roundUp(winSize.width, 8);
        cv::Size winBufSize(winbufwidth, winbufwidth);

        std::vector<short> _buf(winBufSize.area()*(cn + cn2));
        Mat IWinBuf(winBufSize, CV_MAKETYPE(CV_16S, cn), &_buf[0]);
        Mat derivIWinBuf(winBufSize, CV_MAKETYPE(CV_16S, cn2), &_buf[winBufSize.area()*cn]);

        for (int ptidx = range.start; ptidx < range.end; ptidx++)
        {
            Point2f prevPt = prevPts[ptidx] * (float)(1. / (1 << level));
            Point2f nextPt;
            if (level == maxLevel)
            {
                if ( useInitialFlow)
                    nextPt = nextPts[ptidx] * (float)(1. / (1 << level));
                else
                    nextPt = prevPt;
            }
            else
                nextPt = nextPts[ptidx] * 2.f;
            nextPts[ptidx] = nextPt;

            Point2i iprevPt, inextPt;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);
            int winArea = maxWinSize * maxWinSize;
            cv::Mat winMaskMat(winMaskMatBuf, cv::Rect(0, 0, maxWinSize, maxWinSize));

            if (calcWinMaskMat(BI, windowType, iprevPt,
                winMaskMat, winSize, halfWin, winArea,
                minWinSize, maxWinSize) == false)
                continue;
            halfWin = Point2f(static_cast<float>(maxWinSize) ,static_cast<float>(maxWinSize) ) - halfWin;
            prevPt += halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);
            if( iprevPt.x < 0 || iprevPt.x >= derivI.cols - winSize.width ||
                iprevPt.y < 0 || iprevPt.y >= derivI.rows - winSize.height - 1)
            {
                if (level == 0)
                {
                    if (status)
                        status[ptidx] = 3;
                    if (err)
                        err[ptidx] = 0;
                }
                continue;
            }

            float a = prevPt.x - iprevPt.x;
            float b = prevPt.y - iprevPt.y;
            const int W_BITS = 14;

            int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            float A11 = 0, A12 = 0, A22 = 0;
            float b1 = 0, b2 = 0;
            float D = 0;
            float minEig;

            copyWinBuffers(iw00, iw01, iw10, iw11, winSize, I, derivI, winMaskMat, IWinBuf, derivIWinBuf, iprevPt);

            cv::Mat residualMat = cv::Mat::zeros(winSize.height * (winSize.width + 8) * cn, 1, CV_16SC1);

            cv::Point2f backUpNextPt = nextPt;
            nextPt += halfWin;
            int j;
            float MEstimatorScale = 1;
            int buffIdx = 0;
            cv::Point2f prevDelta(0, 0);

            for (j = 0; j < criteria.maxCount; j++)
            {
                inextPt.x = cvFloor(nextPt.x);
                inextPt.y = cvFloor(nextPt.y);

                if( inextPt.x < 0 || inextPt.x >= J.cols - winSize.width ||
                    inextPt.y < 0 || inextPt.y >= J.rows - winSize.height - 1)
                {
                    if (level == 0 && status)
                        status[ptidx] = 3;
                    if (level > 0)
                        nextPts[ptidx] = backUpNextPt;
                    break;
                }


                a = nextPt.x - inextPt.x;
                b = nextPt.y - inextPt.y;
                iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                b1 = 0;
                b2 = 0;

                if (j == 0 )
                {
                    A11 = 0;
                    A12 = 0;
                    A22 = 0;
                }

                if (j == 0 )
                {
                    buffIdx = 0;
                    for (int y = 0; y < winSize.height; y++)
                    {
                        const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                        const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                        const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                        const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                        const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                        for (int x = 0; x < winSize.width*cn; x++, dIptr += 2)
                        {
                            if (maskPtr[x] == 0)
                                continue;
                            int diff = CV_DESCALE(Jptr[x] * iw00 + Jptr[x + cn] * iw01 + Jptr1[x] * iw10 + Jptr1[x + cn] * iw11, W_BITS - 5) - Iptr[x];
                            residualMat.at<short>(buffIdx++) = static_cast<short>(diff);
                        }
                    }
                    /*! Estimation for the residual */
                    cv::Mat residualMatRoi(residualMat, cv::Rect(0, 0, 1, buffIdx));
                    MEstimatorScale = (buffIdx == 0) ? 1.f : estimateScale(residualMatRoi);
                }

                float eta = 1.f / winArea;
                float fParam0 = normSigma0 * 32.f;
                float fParam1 = normSigma1 * 32.f;
                fParam0 = normSigma0 * MEstimatorScale;
                fParam1 = normSigma1 * MEstimatorScale;
#if CV_SIMD128
                v_int16x8 vqw0 = v_int16x8((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
                v_int16x8 vqw1 = v_int16x8((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
                v_float32x4 vqb0 = v_setzero_f32(), vqb1 = v_setzero_f32();
                v_float32x4 vAxx = v_setzero_f32(), vAxy = v_setzero_f32(), vAyy = v_setzero_f32();
                v_int32x4 vdelta = v_setall_s32(1 << (W_BITS - 5 - 1));
                v_int16x8 vscale = v_setall_s16(static_cast<short>(MEstimatorScale));
                v_int16x8 veta = v_setzero_s16();
                v_int16x8 vzero = v_setzero_s16();
                v_int16x8 vparam0 = v_setall_s16(MIN(std::numeric_limits<short>::max() - 1, static_cast<short>(fParam0)));
                v_int16x8 vparam1 = v_setall_s16(MIN(std::numeric_limits<short>::max() - 1, static_cast<short>(fParam1)));
                v_int16x8 vneg_param1 = v_setall_s16(-MIN(std::numeric_limits<short>::max() - 1, static_cast<short>(fParam1)));
                int s2bitShift = normSigma2 == 0 ? 1 : cvCeil(log(200.f / std::fabs(normSigma2)) / log(2.f));
                v_int16x8 vparam2 = v_setall_s16(static_cast<short>(normSigma2 * (float)(1 << s2bitShift)));
                v_int16x8 voness = v_setall_s16(1 << s2bitShift);
                v_float32x4 vparam2s = v_setall_f32(0.01f * normSigma2);
                v_float32x4 vparam2s2 = v_setall_f32(normSigma2 * normSigma2);
                v_float32x4 vones = v_setall_f32(1.f);
                v_float32x4 vzeros = v_setzero_f32();
                v_int16x8 vmax_val_16 = v_setall_s16(std::numeric_limits<unsigned short>::max());
#endif

                buffIdx = 0;
                for (int y = 0; y < winSize.height; y++)
                {
                    const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                    const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                    const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                    const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                    const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
#if CV_SIMD128
                    for (int x = 0; x <= winSize.width*cn; x += 8, dIptr += 8 * 2)
                    {
                        v_int16x8 diff0 = v_reinterpret_as_s16(v_load(Iptr + x)), diff1, diff2;
                        v_int16x8 v00 = v_reinterpret_as_s16(v_load_expand(Jptr + x));
                        v_int16x8 v01 = v_reinterpret_as_s16(v_load_expand(Jptr + x + cn));
                        v_int16x8 v10 = v_reinterpret_as_s16(v_load_expand(Jptr1 + x));
                        v_int16x8 v11 = v_reinterpret_as_s16(v_load_expand(Jptr1 + x + cn));
                        v_int16x8 vmask = v_reinterpret_as_s16(v_load_expand(maskPtr + x)) * vmax_val_16;

                        v_int32x4 t0, t1;
                        v_int16x8 t00, t01, t10, t11;
                        v_zip(v00, v01, t00, t01);
                        v_zip(v10, v11, t10, t11);

                        t0 = v_dotprod(t00, vqw0, vdelta) + v_dotprod(t10, vqw1);
                        t1 = v_dotprod(t01, vqw0, vdelta) + v_dotprod(t11, vqw1);
                        t0 = t0 >> (W_BITS - 5);
                        t1 = t1 >> (W_BITS - 5);
                        diff0 = v_pack(t0, t1) - diff0;
                        diff0 = diff0 & vmask;

                        v_int16x8 vscale_diff_is_pos = diff0 > vscale;
                        veta = veta + (vscale_diff_is_pos & v_setall_s16(2)) + v_setall_s16(-1);
                        // since there is no abs vor int16x8 we have to do this hack
                        v_int16x8 vabs_diff = v_reinterpret_as_s16(v_abs(diff0));
                        v_int16x8 vset2, vset1;
                        // |It| < sigma1 ?
                        vset2 = vabs_diff < vparam1;
                        // It > 0 ?
                        v_int16x8 vdiff_is_pos = diff0 > vzero;
                        // sigma0 < |It| < sigma1 ?
                        vset1 = vset2 & (vabs_diff > vparam0);
                        // val = |It| -/+ sigma1
                        v_int16x8 vtmp_param1 = diff0 + v_select(vdiff_is_pos, vneg_param1, vparam1);
                        // It == 0     ? |It| > sigma13
                        diff0 = vset2 & diff0;
                        // It == val ? sigma0 < |It| < sigma1
                        diff0 = v_select(vset1, vtmp_param1, diff0);

                        v_int16x8 tale_ = v_select(vset1, vparam2, voness); // mask for 0 - 3
                        // diff = diff * sigma2
                        v_int32x4 diff_int_0, diff_int_1;
                        v_mul_expand(diff0, tale_, diff_int_0, diff_int_1);
                        diff0 = v_pack(diff_int_0 >> s2bitShift, diff_int_1 >> s2bitShift);
                        v_zip(diff0, diff0, diff2, diff1); // It0 It0 It1 It1 ...

                        v_int16x8 vIxy_0 = v_reinterpret_as_s16(v_load(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                        v_int16x8 vIxy_1 = v_reinterpret_as_s16(v_load(dIptr + 8));
                        v_zip(vIxy_0, vIxy_1, v10, v11);
                        v_zip(diff2, diff1, v00, v01);

                        vqb0 += v_cvt_f32(v_dotprod(v00, v10));
                        vqb1 += v_cvt_f32(v_dotprod(v01, v11));
                        if (j == 0)
                        {
                            v_int32x4 vset1_0, vset1_1, vset2_0, vset2_1;
                            v_int32x4 vmask_0, vmask_1;

                            v_expand(vset1, vset1_0, vset1_1);
                            v_expand(vset2, vset2_0, vset2_1);
                            v_expand(vmask, vmask_0, vmask_1);

                            v_float32x4 vtale_0 = v_select(v_reinterpret_as_f32(vset1_0), vparam2s2, vones);
                            v_float32x4 vtale_1 = v_select(v_reinterpret_as_f32(vset1_1), vparam2s2, vones);
                            vtale_0 = v_select(v_reinterpret_as_f32(vset2_0), vtale_0, vparam2s);
                            vtale_1 = v_select(v_reinterpret_as_f32(vset2_1), vtale_1, vparam2s);

                            vtale_0 = v_select(v_reinterpret_as_f32(vmask_0), vtale_0, vzeros);
                            vtale_1 = v_select(v_reinterpret_as_f32(vmask_1), vtale_1, vzeros);

                            v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(vIxy_0))));
                            v_expand(v00, t1, t0);

                            v_float32x4 fy = v_cvt_f32(t0);
                            v_float32x4 fx = v_cvt_f32(t1);

                            // A11 - A22
                            v_float32x4 fxtale = fx * vtale_0;
                            v_float32x4 fytale = fy * vtale_0;

                            vAyy = v_muladd(fy, fytale, vAyy);
                            vAxy = v_muladd(fx, fytale, vAxy);
                            vAxx = v_muladd(fx, fxtale, vAxx);

                            v01 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(vIxy_1))));
                            v_expand(v01, t1, t0);

                            fy = v_cvt_f32(t0);
                            fx = v_cvt_f32(t1);

                            // A11 - A22
                            fxtale = fx * vtale_1;
                            fytale = fy * vtale_1;

                            vAyy = v_muladd(fy, fytale, vAyy);
                            vAxy = v_muladd(fx, fytale, vAxy);
                            vAxx = v_muladd(fx, fxtale, vAxx);
                        }
                    }
#else
                    for (int x = 0; x < winSize.width*cn; x++, dIptr += 2)
                    {
                        if (maskPtr[x] == 0)
                            continue;
                        int diff = CV_DESCALE(Jptr[x] * iw00 + Jptr[x + cn] * iw01 +
                            Jptr1[x] * iw10 + Jptr1[x + cn] * iw11,
                            W_BITS - 5) - Iptr[x];

                        if (diff > MEstimatorScale)
                            MEstimatorScale += eta;
                        if (diff < MEstimatorScale)
                            MEstimatorScale -= eta;

                        int abss = (diff < 0) ? -diff : diff;
                        if (abss > fParam1)
                        {
                            diff = 0;
                        }
                        else if (abss > fParam0 && diff >= 0)
                        {
                            //diff = fParam1 * (diff - fParam1);
                            diff = static_cast<int>(normSigma2 * (diff - fParam1));
                        }
                        else if (abss > fParam0 && diff < 0)
                        {
                            //diff = fParam1 * (diff + fParam1);
                            diff = static_cast<int>(normSigma2 * (diff + fParam1));

                        }

                        float ixval = (float)(dIptr[0]);
                        float iyval = (float)(dIptr[1]);
                        b1 += (float)(diff*ixval);
                        b2 += (float)(diff*iyval);

                        if ( j == 0)
                        {
                            float tale = normSigma2 * FLT_RESCALE;
                            if (abss < fParam0)
                            {
                                tale = FLT_RESCALE;
                            }
                            else if (abss > fParam1)
                            {
                                tale *= 0.01f;
                            }
                            else
                            {
                                tale *= normSigma2;
                            }
                            A11 += (float)(ixval*ixval*tale);
                            A12 += (float)(ixval*iyval*tale);
                            A22 += (float)(iyval*iyval*tale);
                        }
                    }
#endif
                }

#if CV_SIMD128
                MEstimatorScale += eta * v_reduce_sum(veta);
#endif
                if (j == 0)
                {
#if CV_SIMD128
                    A11 = v_reduce_sum(vAxx);
                    A12 = v_reduce_sum(vAxy);
                    A22 = v_reduce_sum(vAyy);
#endif

                    A11 *= FLT_SCALE;
                    A12 *= FLT_SCALE;
                    A22 *= FLT_SCALE;

                    D = A11 * A22 - A12 * A12;
                    minEig = (A22 + A11 - std::sqrt((A11 - A22)*(A11 - A22) +
                        4.f*A12*A12)) / (2 * winArea);
                    D = 1.f / D;

                    if (minEig < minEigThreshold || std::abs(D) < FLT_EPSILON)
                    {
                        if (level == 0 && status)
                            status[ptidx] = 0;
                        if (level > 0)
                        {
                            nextPts[ptidx] = backUpNextPt;
                        }
                        break;
                    }
                }

#if CV_SIMD128
                float CV_DECL_ALIGNED(16) bbuf[4];
                v_store_aligned(bbuf, vqb0 + vqb1);
                b1 += bbuf[0] + bbuf[2];
                b2 += bbuf[1] + bbuf[3];
#endif
                b1 *= FLT_SCALE;
                b2 *= FLT_SCALE;

                Point2f delta((float)((A12*b2 - A22 * b1) * D), (float)((A12*b1 - A11 * b2) * D));

                delta.x = (delta.x != delta.x) ? 0 : delta.x;
                delta.y = (delta.y != delta.y) ? 0 : delta.y;

                nextPt += delta * 0.7;
                nextPts[ptidx] = nextPt - halfWin;

                if (j > 0 && std::abs(delta.x - prevDelta.x) < 0.01  &&
                    std::abs(delta.y - prevDelta.y) < 0.01)
                {
                    nextPts[ptidx] -= delta * 0.5f;
                    break;
                }
                prevDelta = delta;
            }
        }
    }

    const Mat*          prevImg;
    const Mat*          nextImg;
    const Mat*          prevDeriv;
    const Mat*          rgbPrevImg;
    const Mat*          rgbNextImg;
    const Point2f*      prevPts;
    Point2f*            nextPts;
    uchar*              status;
    float*              err;
    int                 maxWinSize;
    int                 minWinSize;
    TermCriteria        criteria;
    int                 level;
    int                 maxLevel;
    int                 windowType;
    float               minEigThreshold;
    bool                useInitialFlow;
    const float         normSigma0, normSigma1, normSigma2;
    int                 crossSegmentationThreshold;
};

} // namespace
namespace radial {

class TrackerInvoker : public cv::ParallelLoopBody
{
public:
    TrackerInvoker(
        const Mat&      _prevImg,
        const Mat&      _prevDeriv,
        const Mat&      _nextImg,
        const Mat&      _rgbPrevImg,
        const Mat&      _rgbNextImg,
        const Point2f*  _prevPts,
        Point2f*        _nextPts,
        uchar*          _status,
        float*          _err,
        Point2f*        _gainVecs,
        int             _level,
        int             _maxLevel,
        int             _winSize[2],
        int             _maxIteration,
        bool            _useInitialFlow,
        int             _supportRegionType,
        const std::vector<float>& _normSigmas,
        float           _minEigenValue,
        int             _crossSegmentationThreshold
    ) :
        normSigma0(_normSigmas[0]),
        normSigma1(_normSigmas[1]),
        normSigma2(_normSigmas[2])
    {
        prevImg = &_prevImg;
        prevDeriv = &_prevDeriv;
        nextImg = &_nextImg;
        rgbPrevImg = &_rgbPrevImg;
        rgbNextImg = &_rgbNextImg;
        prevPts = _prevPts;
        nextPts = _nextPts;
        status = _status;
        err = _err;
        gainVecs = _gainVecs;
        minWinSize = _winSize[0];
        maxWinSize = _winSize[1];
        criteria.maxCount = _maxIteration;
        criteria.epsilon = 0.01;
        level = _level;
        maxLevel = _maxLevel;
        windowType = _supportRegionType;
        minEigThreshold = _minEigenValue;
        useInitialFlow = _useInitialFlow;
        crossSegmentationThreshold = _crossSegmentationThreshold;
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        Point2f halfWin;
        cv::Size winSize;
        const Mat& I = *prevImg;
        const Mat& J = *nextImg;
        const Mat& derivI = *prevDeriv;
        const Mat& BI = *rgbPrevImg;


        const float FLT_SCALE = (1.f / (1 << 16));

        winSize = cv::Size(maxWinSize, maxWinSize);
        int winMaskwidth = roundUp(winSize.width, 16);
        cv::Mat winMaskMatBuf(winMaskwidth, winMaskwidth, tCVMaskType);
        winMaskMatBuf.setTo(1);

        int cn = I.channels(), cn2 = cn * 2;
        int winbufwidth = roundUp(winSize.width, 16);
        cv::Size winBufSize(winbufwidth, winbufwidth);

        cv::Matx44f invTensorMat;
        cv::Vec4f mismatchMat;
        cv::Vec4f resultMat;


        std::vector<short> _buf(winBufSize.area()*(cn + cn2));
        Mat IWinBuf(winBufSize, CV_MAKETYPE(CV_16S, cn), &_buf[0]);
        Mat derivIWinBuf(winBufSize, CV_MAKETYPE(CV_16S, cn2), &_buf[winBufSize.area()*cn]);

        for (int ptidx = range.start; ptidx < range.end; ptidx++)
        {

            Point2f prevPt = prevPts[ptidx] * (float)(1. / (1 << level));
            Point2f nextPt;
            if (level == maxLevel)
            {
                if (useInitialFlow)
                {
                    nextPt = nextPts[ptidx] * (float)(1. / (1 << level));
                }
                else
                    nextPt = prevPt;
            }
            else
            {
                nextPt = nextPts[ptidx] * 2.f;

            }
            nextPts[ptidx] = nextPt;

            Point2i iprevPt, inextPt;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);
            int winArea = maxWinSize * maxWinSize;
            cv::Mat winMaskMat(winMaskMatBuf, cv::Rect(0, 0, maxWinSize, maxWinSize));
            winMaskMatBuf.setTo(0);
            if (calcWinMaskMat(BI, windowType, iprevPt,
                winMaskMat, winSize, halfWin, winArea,
                minWinSize, maxWinSize) == false)
                continue;
            halfWin = Point2f(static_cast<float>(maxWinSize) ,static_cast<float>(maxWinSize) ) - halfWin;
            prevPt += halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);
            if( iprevPt.x < 0 || iprevPt.x >= derivI.cols - winSize.width ||
                iprevPt.y < 0 || iprevPt.y >= derivI.rows - winSize.height - 1)
            {
                if (level == 0)
                {
                    if (status)
                        status[ptidx] = 3;
                    if (err)
                        err[ptidx] = 0;
                }
                continue;
            }

            float a = prevPt.x - iprevPt.x;
            float b = prevPt.y - iprevPt.y;
            const int W_BITS = 14;

            int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            float A11 = 0, A12 = 0, A22 = 0;
            float b1 = 0, b2 = 0, b3 = 0, b4 = 0;
            // tensor
            float sumIx = 0;
            float sumIy = 0;
            float sumI = 0;
            float sumW = 0;
            float w1 = 0, w2 = 0; // -IyI
            float dI = 0; // I^2
            float D = 0;

            copyWinBuffers(iw00, iw01, iw10, iw11, winSize, I, derivI, winMaskMat, IWinBuf, derivIWinBuf, iprevPt);

            cv::Mat residualMat = cv::Mat::zeros(winSize.height * (winSize.width + 8) * cn, 1, CV_16SC1);

            cv::Point2f backUpNextPt = nextPt;
            nextPt += halfWin;
            Point2f prevDelta(0, 0);    //related to h(t-1)
            Point2f prevGain(0, 0);
            cv::Point2f gainVec = gainVecs[ptidx];
            cv::Point2f backUpGain = gainVec;
            cv::Size _winSize = winSize;
            int j;
            float MEstimatorScale = 1;
            int buffIdx = 0;
            float minEigValue;

            for (j = 0; j < criteria.maxCount; j++)
            {

                inextPt.x = cvFloor(nextPt.x);
                inextPt.y = cvFloor(nextPt.y);

                if( inextPt.x < 0 || inextPt.x >= J.cols - winSize.width ||
                    inextPt.y < 0 || inextPt.y >= J.rows - winSize.height - 1)
                {
                    if (level == 0 && status)
                        status[ptidx] = 3;
                    if (level > 0)
                    {
                        nextPts[ptidx] = backUpNextPt;
                        gainVecs[ptidx] = backUpGain;
                    }
                    break;
                }

                a = nextPt.x - inextPt.x;
                b = nextPt.y - inextPt.y;
                iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

                // mismatch
                b1 = 0;
                b2 = 0;
                b3 = 0;
                b4 = 0;
                if (j == 0)
                {
                    // tensor
                    w1 = 0; // -IxI
                    w2 = 0; // -IyI
                    dI = 0; // I^2
                    sumIx = 0;
                    sumIy = 0;
                    sumI = 0;
                    sumW = 0;
                    A11 = 0;
                    A12 = 0;
                    A22 = 0;
                }

                if (j == 0 )
                {
                    buffIdx = 0;
                    for (int y = 0; y < winSize.height; y++)
                    {
                        const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                        const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                        const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                        const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                        for (int x = 0; x < winSize.width*cn; x++, dIptr += 2)
                        {
                            if (dIptr[0] == 0 && dIptr[1] == 0)
                                continue;
                            int diff = static_cast<int>(CV_DESCALE(    Jptr[x] * iw00 +
                                                    Jptr[x + cn] * iw01 +
                                                    Jptr1[x] * iw10 +
                                                    Jptr1[x + cn] * iw11, W_BITS - 5)
                                - Iptr[x] + Iptr[x] * gainVec.x + gainVec.y);
                            residualMat.at<short>(buffIdx++) = static_cast<short>(diff);
                        }
                    }
                    /*! Estimation for the residual */
                    cv::Mat residualMatRoi(residualMat, cv::Rect(0, 0, 1, buffIdx));
                    MEstimatorScale = (buffIdx == 0) ? 1.f : estimateScale(residualMatRoi);
                }

                float eta = 1.f / winArea;
                float fParam0 = normSigma0 * 32.f;
                float fParam1 = normSigma1 * 32.f;
                fParam0 = normSigma0 * MEstimatorScale;
                fParam1 = normSigma1 * MEstimatorScale;

#if CV_SIMD128
                v_int16x8 vqw0 = v_int16x8((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
                v_int16x8 vqw1 = v_int16x8((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
                v_float32x4 vqb0 = v_setzero_f32(), vqb1 = v_setzero_f32(), vqb2 = v_setzero_f32(), vqb3 = v_setzero_f32();
                v_float32x4 vsumW1 = v_setzero_f32(), vsumW2 = v_setzero_f32(), vsumW = v_setzero_f32();
                v_float32x4 vsumIy = v_setzero_f32(), vsumIx = v_setzero_f32(), vsumI = v_setzero_f32(), vsumDI = v_setzero_f32();
                v_float32x4 vAxx = v_setzero_f32(), vAxy = v_setzero_f32(), vAyy = v_setzero_f32();

                int s2bitShift = normSigma2 == 0 ? 1 : cvCeil(log(200.f / std::fabs(normSigma2)) / log(2.f));
                v_int32x4 vdelta = v_setall_s32(1 << (W_BITS - 5 - 1));
                v_int16x8 vzero = v_setzero_s16();
                v_int16x8 voness = v_setall_s16(1 << s2bitShift);
                v_float32x4 vones = v_setall_f32(1.f);
                v_float32x4 vzeros = v_setzero_f32();
                v_int16x8 vmax_val_16 = v_setall_s16(std::numeric_limits<unsigned short>::max());

                v_int16x8 vscale = v_setall_s16(static_cast<short>(MEstimatorScale));
                v_int16x8 veta = v_setzero_s16();
                v_int16x8 vparam0 = v_setall_s16(MIN(std::numeric_limits<short>::max() - 1, static_cast<short>(fParam0)));
                v_int16x8 vparam1 = v_setall_s16(MIN(std::numeric_limits<short>::max() - 1, static_cast<short>(fParam1)));
                v_int16x8 vneg_param1 = v_setall_s16(-MIN(std::numeric_limits<short>::max() - 1, static_cast<short>(fParam1)));
                v_int16x8 vparam2 = v_setall_s16(static_cast<short>(normSigma2 * (float)(1 << s2bitShift)));
                v_float32x4 vparam2s = v_setall_f32(0.01f * normSigma2);
                v_float32x4 vparam2s2 = v_setall_f32(normSigma2 * normSigma2);

                float gainVal = gainVec.x > 0 ? gainVec.x : -gainVec.x;
                int bitShift = gainVec.x == 0 ? 1 : cvCeil(log(200.f / gainVal) / log(2.f));
                v_int16x8 vgain_value = v_setall_s16(static_cast<short>(gainVec.x * (float)(1 << bitShift)));
                v_int16x8 vconst_value = v_setall_s16(static_cast<short>(gainVec.y));
#endif
                buffIdx = 0;
                for (int y = 0; y < _winSize.height; y++)
                {
                    const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                    const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                    const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                    const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                    const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
#if CV_SIMD128
                    for (int x = 0; x <= _winSize.width*cn; x += 8, dIptr += 8 * 2)
                    {
                        v_int16x8 vI = v_reinterpret_as_s16(v_load(Iptr + x)), diff0, diff1, diff2;
                        v_int16x8 v00 = v_reinterpret_as_s16(v_load_expand(Jptr + x));
                        v_int16x8 v01 = v_reinterpret_as_s16(v_load_expand(Jptr + x + cn));
                        v_int16x8 v10 = v_reinterpret_as_s16(v_load_expand(Jptr1 + x));
                        v_int16x8 v11 = v_reinterpret_as_s16(v_load_expand(Jptr1 + x + cn));
                        v_int16x8 vmask = v_reinterpret_as_s16(v_load_expand(maskPtr + x)) * vmax_val_16;

                        v_int32x4 t0, t1;
                        v_int16x8 t00, t01, t10, t11;
                        v_zip(v00, v01, t00, t01);
                        v_zip(v10, v11, t10, t11);

                        //subpixel interpolation
                        t0 = v_dotprod(t00, vqw0, vdelta) + v_dotprod(t10, vqw1);
                        t1 = v_dotprod(t01, vqw0, vdelta) + v_dotprod(t11, vqw1);
                        t0 = t0 >> (W_BITS - 5);
                        t1 = t1 >> (W_BITS - 5);

                        // diff = J - I
                        diff0 = v_pack(t0, t1) - vI;
                        // I*gain.x + gain.x
                        v_mul_expand(vI, vgain_value, t0, t1);
                        diff0 = diff0 + v_pack(t0 >> bitShift, t1 >> bitShift) + vconst_value;
                        diff0 = diff0 & vmask;

                        v_int16x8 vscale_diff_is_pos = diff0 > vscale;
                        veta = veta + (vscale_diff_is_pos & v_setall_s16(2)) + v_setall_s16(-1);
                        // since there is no abs vor int16x8 we have to do this hack
                        v_int16x8 vabs_diff = v_reinterpret_as_s16(v_abs(diff0));
                        v_int16x8 vset2, vset1;
                        // |It| < sigma1 ?
                        vset2 = vabs_diff < vparam1;
                        // It > 0 ?
                        v_int16x8 vdiff_is_pos = diff0 > vzero;
                        // sigma0 < |It| < sigma1 ?
                        vset1 = vset2 & (vabs_diff > vparam0);
                        // val = |It| -/+ sigma1
                        v_int16x8 vtmp_param1 = diff0 + v_select(vdiff_is_pos, vneg_param1, vparam1);
                        // It == 0     ? |It| > sigma13
                        diff0 = vset2 & diff0;
                        // It == val ? sigma0 < |It| < sigma1
                        diff0 = v_select(vset1, vtmp_param1, diff0);

                        v_int16x8 tale_ = v_select(vset1, vparam2, voness); // mask for 0 - 3
                        // diff = diff * sigma2
                        v_int32x4 diff_int_0, diff_int_1;
                        v_mul_expand(diff0, tale_, diff_int_0, diff_int_1);
                        v_int32x4 diff0_0 = diff_int_0 >> s2bitShift;
                        v_int32x4 diff0_1 = diff_int_1 >> s2bitShift;
                        diff0 = v_pack(diff0_0, diff0_1);
                        v_zip(diff0, diff0, diff2, diff1); // It0 It0 It1 It1 ...

                        v_int16x8 vIxy_0 = v_reinterpret_as_s16(v_load(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                        v_int16x8 vIxy_1 = v_reinterpret_as_s16(v_load(dIptr + 8));
                        v_zip(vIxy_0, vIxy_1, v10, v11);
                        v_zip(diff2, diff1, v00, v01);

                        vqb0 += v_cvt_f32(v_dotprod(v00, v10));
                        vqb1 += v_cvt_f32(v_dotprod(v01, v11));

                        v_int32x4 vI0, vI1;
                        v_expand(vI, vI0, vI1);
                        vqb2 += v_cvt_f32(diff0_0 * vI0);
                        vqb2 += v_cvt_f32(diff0_1 * vI1);

                        vqb3 += v_cvt_f32(diff0_0);
                        vqb3 += v_cvt_f32(diff0_1);

                        if (j == 0)
                        {
                            v_int32x4 vset1_0, vset1_1, vset2_0, vset2_1;
                            v_int32x4 vmask_0, vmask_1;

                            v_expand(vset1, vset1_0, vset1_1);
                            v_expand(vset2, vset2_0, vset2_1);
                            v_expand(vmask, vmask_0, vmask_1);

                            v_float32x4 vtale_0 = v_select(v_reinterpret_as_f32(vset1_0), vparam2s2, vones);
                            v_float32x4 vtale_1 = v_select(v_reinterpret_as_f32(vset1_1), vparam2s2, vones);
                            vtale_0 = v_select(v_reinterpret_as_f32(vset2_0), vtale_0, vparam2s);
                            vtale_1 = v_select(v_reinterpret_as_f32(vset2_1), vtale_1, vparam2s);

                            vtale_0 = v_select(v_reinterpret_as_f32(vmask_0), vtale_0, vzeros);
                            vtale_1 = v_select(v_reinterpret_as_f32(vmask_1), vtale_1, vzeros);

                            v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(vIxy_0))));
                            v_expand(v00, t1, t0);

                            v_float32x4 vI_ps = v_cvt_f32(vI0);

                            v_float32x4 fy = v_cvt_f32(t0);
                            v_float32x4 fx = v_cvt_f32(t1);

                            // A11 - A22
                            v_float32x4 fxtale = fx * vtale_0;
                            v_float32x4 fytale = fy * vtale_0;

                            vAyy = v_muladd(fy, fytale, vAyy);
                            vAxy = v_muladd(fx, fytale, vAxy);
                            vAxx = v_muladd(fx, fxtale, vAxx);

                            // sumIx und sumIy
                            vsumIx += fxtale;
                            vsumIy += fytale;

                            vsumW1 += vI_ps * fxtale;
                            vsumW2 += vI_ps * fytale;

                            // sumI
                            v_float32x4 vI_tale = vI_ps * vtale_0;
                            vsumI += vI_tale;

                            // sumW
                            vsumW += vtale_0;

                            // sumDI
                            vsumDI += vI_ps * vI_tale;

                            v01 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(vIxy_1))));
                            v_expand(v01, t1, t0);
                            vI_ps = v_cvt_f32(vI1);

                            fy = v_cvt_f32(t0);
                            fx = v_cvt_f32(t1);

                            // A11 - A22
                            fxtale = fx * vtale_1;
                            fytale = fy * vtale_1;

                            vAyy = v_muladd(fy, fytale, vAyy);
                            vAxy = v_muladd(fx, fytale, vAxy);
                            vAxx = v_muladd(fx, fxtale, vAxx);

                            // sumIx und sumIy
                            vsumIx += fxtale;
                            vsumIy += fytale;

                            vsumW1 += vI_ps * fxtale;
                            vsumW2 += vI_ps * fytale;

                            // sumI
                            vI_tale = vI_ps * vtale_1;
                            vsumI += vI_tale;

                            // sumW
                            vsumW += vtale_1;

                            // sumDI
                            vsumDI += vI_ps * vI_tale;
                        }
                    }
#else
                    for (int x = 0; x < _winSize.width*cn; x++, dIptr += 2)
                    {
                        if (maskPtr[x] == 0)
                            continue;
                        int J_val = CV_DESCALE(Jptr[x] * iw00 + Jptr[x + cn] * iw01 + Jptr1[x] * iw10 + Jptr1[x + cn] * iw11,
                            W_BITS - 5);
                        short ixval = static_cast<short>(dIptr[0]);
                        short iyval = static_cast<short>(dIptr[1]);
                        int diff = static_cast<int>(J_val - Iptr[x] + Iptr[x] * gainVec.x + gainVec.y);
                        int abss = (diff < 0) ? -diff : diff;
                        if (diff > MEstimatorScale)
                            MEstimatorScale += eta;
                        if (diff < MEstimatorScale)
                            MEstimatorScale -= eta;
                        if (abss > static_cast<int>(fParam1))
                        {
                            diff = 0;
                        }
                        else if (abss > static_cast<int>(fParam0) && diff >= 0)
                        {
                            diff = static_cast<int>(normSigma2 * (diff - fParam1));
                        }
                        else if (abss > static_cast<int>(fParam0) && diff < 0)
                        {
                            diff = static_cast<int>(normSigma2 * (diff + fParam1));
                        }
                        b1 += (float)(diff*ixval);
                        b2 += (float)(diff*iyval); ;
                        b3 += (float)(diff)* Iptr[x];
                        b4 += (float)(diff);


                        // compute the Gradient Matrice
                        if (j == 0)
                        {
                            float tale = normSigma2 * FLT_RESCALE;
                            if (abss < fParam0 || j < 0)
                            {
                                tale = FLT_RESCALE;
                            }
                            else if (abss > fParam1)
                            {
                                tale *= 0.01f;
                            }
                            else
                            {
                                tale *= normSigma2;
                            }
                            if (j == 0)
                            {
                                A11 += (float)(ixval*ixval)*tale;
                                A12 += (float)(ixval*iyval)*tale;
                                A22 += (float)(iyval*iyval)*tale;
                            }

                            dI += Iptr[x] * Iptr[x] * tale;
                            float dx = static_cast<float>(dIptr[0]) * tale;
                            float dy = static_cast<float>(dIptr[1]) * tale;
                            sumIx += dx;
                            sumIy += dy;
                            w1 += dx * Iptr[x];
                            w2 += dy * Iptr[x];
                            sumI += Iptr[x] * tale;
                            sumW += tale;
                        }

                    }
#endif
                }
#if CV_SIMD128
                MEstimatorScale += eta * v_reduce_sum(veta);
#endif
                if (j == 0)
                {
#if CV_SIMD128
                    w1 = v_reduce_sum(vsumW1);
                    w2 = v_reduce_sum(vsumW2);
                    dI = v_reduce_sum(vsumDI);
                    sumI = v_reduce_sum(vsumI);
                    sumIx = v_reduce_sum(vsumIx);
                    sumIy = v_reduce_sum(vsumIy);
                    sumW = v_reduce_sum(vsumW);
                    A11 = v_reduce_sum(vAxx);
                    A12 = v_reduce_sum(vAxy);
                    A22 = v_reduce_sum(vAyy);
#endif
                    sumIx *= -FLT_SCALE;
                    sumIy *= -FLT_SCALE;
                    sumI *= FLT_SCALE;
                    sumW *= FLT_SCALE;
                    w1 *= -FLT_SCALE;
                    w2 *= -FLT_SCALE;
                    dI *= FLT_SCALE;
#if CV_SIMD128
#endif
                    A11 *= FLT_SCALE;
                    A12 *= FLT_SCALE;
                    A22 *= FLT_SCALE;
                }
#if CV_SIMD128
                float CV_DECL_ALIGNED(16) bbuf[4];
                v_store_aligned(bbuf, vqb0 + vqb1);
                b1 = bbuf[0] + bbuf[2];
                b2 = bbuf[1] + bbuf[3];
                b3 = v_reduce_sum(vqb2);
                b4 = v_reduce_sum(vqb3);
#endif
                mismatchMat(0) = b1 * FLT_SCALE;
                mismatchMat(1) = b2 * FLT_SCALE;
                mismatchMat(2) = -b3 * FLT_SCALE;
                mismatchMat(3) = -b4 * FLT_SCALE;

                D = -A12 * A12*sumI*sumI + dI * sumW*A12*A12 + 2 * A12*sumI*sumIx*w2 + 2 * A12*sumI*sumIy*w1
                    - 2 * dI*A12*sumIx*sumIy - 2 * sumW*A12*w1*w2 + A11 * A22*sumI*sumI - 2 * A22*sumI*sumIx*w1
                    - 2 * A11*sumI*sumIy*w2 - sumIx * sumIx*w2*w2 + A22 * dI*sumIx*sumIx + 2 * sumIx*sumIy*w1*w2
                    - sumIy * sumIy*w1*w1 + A11 * dI*sumIy*sumIy + A22 * sumW*w1*w1 + A11 * sumW*w2*w2 - A11 * A22*dI*sumW;

                float sqrtVal = std::sqrt((A11 - A22)*(A11 - A22) + 4.f*A12*A12);
                minEigValue = (A22 + A11 - sqrtVal) / (2.0f*winArea);
                if (minEigValue < minEigThreshold || std::abs(D) < FLT_EPSILON)
                {
                    if (level == 0 && status)
                        status[ptidx] = 0;
                    if (level > 0)
                    {
                        nextPts[ptidx] = backUpNextPt;
                        gainVecs[ptidx] = backUpGain;
                    }
                    break;
                }

                D = (1.f / D);

                invTensorMat(0, 0) = (A22*sumI*sumI - 2 * sumI*sumIy*w2 + dI * sumIy*sumIy + sumW * w2*w2 - A22 * dI*sumW)* D;
                invTensorMat(0, 1) = (A12*dI*sumW - A12 * sumI * sumI - dI * sumIx*sumIy + sumI * sumIx*w2 + sumI * sumIy*w1 - sumW * w1*w2)* D;
                invTensorMat(0, 2) = (A12*sumI*sumIy - sumIy * sumIy*w1 - A22 * sumI*sumIx - A12 * sumW*w2 + A22 * sumW*w1 + sumIx * sumIy*w2)* D;
                invTensorMat(0, 3) = (A22*dI*sumIx - A12 * dI*sumIy - sumIx * w2*w2 + A12 * sumI*w2 - A22 * sumI*w1 + sumIy * w1*w2) * D;
                invTensorMat(1, 0) = invTensorMat(0, 1);
                invTensorMat(1, 1) = (A11*sumI * sumI - 2 * sumI*sumIx*w1 + dI * sumIx * sumIx + sumW * w1*w1 - A11 * dI*sumW) * D;
                invTensorMat(1, 2) = (A12*sumI*sumIx - A11 * sumI*sumIy - sumIx * sumIx*w2 + A11 * sumW*w2 - A12 * sumW*w1 + sumIx * sumIy*w1) * D;
                invTensorMat(1, 3) = (A11*dI*sumIy - sumIy * w1*w1 - A12 * dI*sumIx - A11 * sumI*w2 + A12 * sumI*w1 + sumIx * w1*w2)* D;
                invTensorMat(2, 0) = invTensorMat(0, 2);
                invTensorMat(2, 1) = invTensorMat(1, 2);
                invTensorMat(2, 2) = (sumW*A12*A12 - 2 * A12*sumIx*sumIy + A22 * sumIx*sumIx + A11 * sumIy*sumIy - A11 * A22*sumW)* D;
                invTensorMat(2, 3) = (A11*A22*sumI - A12 * A12*sumI - A11 * sumIy*w2 + A12 * sumIx*w2 + A12 * sumIy*w1 - A22 * sumIx*w1)* D;
                invTensorMat(3, 0) = invTensorMat(0, 3);
                invTensorMat(3, 1) = invTensorMat(1, 3);
                invTensorMat(3, 2) = invTensorMat(2, 3);
                invTensorMat(3, 3) = (dI*A12*A12 - 2 * A12*w1*w2 + A22 * w1*w1 + A11 * w2*w2 - A11 * A22*dI)* D;

                resultMat = invTensorMat * mismatchMat;

                Point2f delta(-resultMat(0), -resultMat(1));
                Point2f deltaGain(resultMat(2), resultMat(3));




                if (j == 0)
                    prevGain = deltaGain;
                gainVec += deltaGain * 0.8;
                nextPt += delta * 0.8;
                nextPts[ptidx] = nextPt - halfWin;
                gainVecs[ptidx] = gainVec;

                if (
                    (std::abs(delta.x - prevDelta.x) < 0.01  &&    std::abs(delta.y - prevDelta.y) < 0.01)
                    || ((delta.ddot(delta) <= 0.001) && std::abs(prevGain.x - deltaGain.x) < 0.01)
                    )
                {
                    nextPts[ptidx] -= delta * 0.5f;
                    gainVecs[ptidx] -= deltaGain * 0.5f;
                    break;
                }

                prevDelta = delta;
                prevGain = deltaGain;
            }

        }

    }

    const Mat*          prevImg;
    const Mat*          nextImg;
    const Mat*          prevDeriv;
    const Mat*          rgbPrevImg;
    const Mat*          rgbNextImg;
    const Point2f*      prevPts;
    Point2f*            nextPts;
    uchar*              status;
    cv::Point2f*        gainVecs;
    float*              err;
    int                 maxWinSize;
    int                 minWinSize;
    TermCriteria        criteria;
    int                 level;
    int                 maxLevel;
    int                 windowType;
    float               minEigThreshold;
    bool                useInitialFlow;
    const float         normSigma0, normSigma1, normSigma2;
    int                 crossSegmentationThreshold;
};

}}}} // namespace
#endif
