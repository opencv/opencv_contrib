// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _BERLOF_INVOKER_HPP_
#define _BERLOF_INVOKER_HPP_
#include "rlof_invokerbase.hpp"


namespace cv{
namespace optflow{


static inline bool checkSolution(float x, float y, float * c )
{
    float _a = x - 0.002f;
    float _b = y - 0.002f;
    cv::Point2f tl( c[0] * _a * _b + c[1] * _a + c[2] * _b + c[3],  c[4] * _a * _b + c[5] * _a + c[6] * _b + c[7]);
    _a = x + 0.002f;
    cv::Point2f tr( c[0] * _a * _b + c[1] * _a + c[2] * _b + c[3],  c[4] * _a * _b + c[5] * _a + c[6] * _b + c[7]);
    _b = y + 0.002f;
    cv::Point2f br( c[0] * _a * _b + c[1] * _a + c[2] * _b + c[3],  c[4] * _a * _b + c[5] * _a + c[6] * _b + c[7]);
    _a = x - 0.002f;
    cv::Point2f bl( c[0] * _a * _b + c[1] * _a + c[2] * _b + c[3],  c[4] * _a * _b + c[5] * _a + c[6] * _b + c[7]);
    return (tl.x >= 0 && tl.y >= 0) && (tr.x <= 0 && tr.y >= 0)
        && (bl.x >= 0 && bl.y <= 0) && (br.x <= 0 && br.y <= 0);
}

static inline cv::Point2f est_DeltaGain(const cv::Matx44f& src, const cv::Vec4f& val)
{
    return cv::Point2f(
        src(2,0) * val[0] + src(2,1) * val[1] + src(2,2) * val[2] + src(2,3) * val[3],
        src(3,0) * val[0] + src(3,1) * val[1] + src(3,2) * val[2] + src(3,3) * val[3]);
}
static inline void est_Result(const cv::Matx44f& src, const cv::Vec4f & val, cv::Point2f & delta, cv::Point2f & gain)
{

    delta = cv::Point2f(
        -(src(0,0) * val[0] + src(0,1) * val[1] + src(0,2) * val[2] + src(0,3) * val[3]),
        -(src(1,0) * val[0] + src(1,1) * val[1] + src(1,2) * val[2] + src(1,3) * val[3]));

    gain = cv::Point2f(
        src(2,0) * val[0] + src(2,1) * val[1] + src(2,2) * val[2] + src(2,3) * val[3],
        src(3,0) * val[0] + src(3,1) * val[1] + src(3,2) * val[2] + src(3,3) * val[3]);
}

namespace berlof {

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
        int             _crossSegmentationThreshold,
        const std::vector<float>& _normSigmas,
        float           _minEigenValue
    ) :
        normSigma0(_normSigmas[0]),
        normSigma1(_normSigmas[1]),
        normSigma2(_normSigmas[2])
    {
        prevImg =       &_prevImg;
        prevDeriv =     &_prevDeriv;
        nextImg =       &_nextImg;
        rgbPrevImg =    &_rgbPrevImg;
        rgbNextImg =    &_rgbNextImg;
        prevPts =       _prevPts;
        nextPts =       _nextPts;
        status =        _status;
        err =           _err;
        minWinSize =    _winSize[0];
        maxWinSize =    _winSize[1];
        criteria.maxCount = _maxIteration;
        criteria.epsilon = 0.01;
        level =         _level;
        maxLevel =      _maxLevel;
        windowType =    _supportRegionType;
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


        const float FLT_SCALE = (1.f/(1 << 16));

        winSize = cv::Size(maxWinSize,maxWinSize);
        int winMaskwidth = roundUp(winSize.width, 16);
        cv::Mat winMaskMatBuf(winMaskwidth, winMaskwidth, tCVMaskType);
        winMaskMatBuf.setTo(1);

        int j, cn = I.channels(), cn2 = cn*2;
        int winbufwidth = roundUp(winSize.width, 16);
        cv::Size winBufSize(winbufwidth,winbufwidth);

        cv::AutoBuffer<deriv_type> _buf(winBufSize.area()*(cn + cn2));
        int derivDepth = DataType<deriv_type>::depth;

        Mat IWinBuf(winBufSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf.data());
        Mat derivIWinBuf(winBufSize, CV_MAKETYPE(derivDepth, cn2), (deriv_type*)_buf.data() + winBufSize.area()*cn);


        for( int ptidx = range.start; ptidx < range.end; ptidx++ )
        {
            Point2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
            Point2f nextPt;
            if( level == maxLevel )
            {
                if( useInitialFlow )
                    nextPt = nextPts[ptidx]*(float)(1./(1 << level));
                else
                    nextPt = prevPt;
            }
            else
                nextPt = nextPts[ptidx]*2.f;
            nextPts[ptidx] = nextPt;

            Point2i iprevPt, inextPt;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);
            int winArea = maxWinSize * maxWinSize;
            cv::Mat winMaskMat(winMaskMatBuf, cv::Rect(0,0, maxWinSize,maxWinSize));
            winMaskMatBuf.setTo(0);
            if( calcWinMaskMat(BI, windowType, iprevPt,
                                winMaskMat,winSize,halfWin,winArea,
                                minWinSize,maxWinSize) == false)
            {
                continue;
            }
            halfWin = Point2f(static_cast<float>(maxWinSize), static_cast<float>(maxWinSize) ) - halfWin;
            prevPt += halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);
            if( iprevPt.x < 0 || iprevPt.x >= derivI.cols - winSize.width ||
                iprevPt.y < 0 || iprevPt.y >= derivI.rows - winSize.height - 1)
            {
                if( level == 0 )
                {
                    if( status )
                        status[ptidx] = 3;
                    if( err )
                        err[ptidx] = 0;
                }
                continue;
            }

            float a = prevPt.x - iprevPt.x;
            float b = prevPt.y - iprevPt.y;
            const int W_BITS = 14;// , W_BITS = 14;

            int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            float A11 = 0, A12 = 0, A22 = 0;
            float D = 0;

            copyWinBuffers(iw00, iw01, iw10, iw11, winSize, I, derivI, winMaskMat, IWinBuf, derivIWinBuf, iprevPt);

            cv::Mat residualMat = cv::Mat::zeros(winSize.height * (winSize.width + 8) * cn, 1, CV_16SC1);
            cv::Point2f backUpNextPt = nextPt;
            nextPt += halfWin;
            Point2f prevDelta(0,0);    //denotes h(t-1)
            cv::Size _winSize = winSize;
            float MEstimatorScale = 1;
            int buffIdx = 0;
            cv::Mat GMc0, GMc1, GMc2, GMc3;
            cv::Vec2f Mc0, Mc1, Mc2, Mc3;
            for( j = 0; j < criteria.maxCount; j++ )
            {
                cv::Point2f delta(0,0);
                cv::Point2f deltaGain(0,0);
                bool hasSolved = false;
                a = nextPt.x - inextPt.x;
                b = nextPt.y - inextPt.y;
                float ab = a * b;
                if( j == 0
                    || ( inextPt.x != cvFloor(nextPt.x) || inextPt.y != cvFloor(nextPt.y) || j % 2 != 0 ))
                {
                    inextPt.x = cvFloor(nextPt.x);
                    inextPt.y = cvFloor(nextPt.y);
                    if( inextPt.x < 0 || inextPt.x >= J.cols - winSize.width ||
                       inextPt.y < 0 || inextPt.y >= J.rows - winSize.height - 1)
                    {
                        if( level == 0 && status )
                            status[ptidx] = 3;
                        break;
                    }


                    a = nextPt.x - inextPt.x;
                    b = nextPt.y - inextPt.y;
                    ab = a * b;
                    iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                    iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                    iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                    iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                    // mismatch
                    if( j == 0 )
                    {
                        A11 = 0;
                        A12 = 0;
                        A22 = 0;
                    }

                    if ( j == 0 )
                    {
                        buffIdx = 0;
                        for(int y = 0; y < winSize.height; y++ )
                        {
                            const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                            const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                            const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                            const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                            const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                            for(int x = 0 ; x < winSize.width*cn; x++, dIptr += 2)
                            {
                                if( maskPtr[x] == 0)
                                    continue;
                                int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 + Jptr1[x]*iw10 + Jptr1[x+cn]*iw11, W_BITS-5)
                                    - Iptr[x];
                                residualMat.at<short>(buffIdx++) = static_cast<short>(diff);
                            }
                        }
                        /*! Estimation for the residual */
                        cv::Mat residualMatRoi(residualMat, cv::Rect(0,0,1, buffIdx));
                        MEstimatorScale = (buffIdx == 0) ? 1.f : estimateScale(residualMatRoi);
                    }

                    float eta = 1.f / winArea;
                    float fParam0 = normSigma0 * 32.f;
                    float fParam1 = normSigma1 * 32.f;
                    fParam0 = normSigma0 * MEstimatorScale;
                    fParam1 = normSigma1 * MEstimatorScale;
                    buffIdx = 0;
                    float _b0[4] = {0,0,0,0};
                    float _b1[4] = {0,0,0,0};
#if CV_SIMD128
                    v_int16x8 vqw0 = v_int16x8((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
                    v_int16x8 vqw1 = v_int16x8((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
                    v_float32x4 vqb0[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
                    v_float32x4 vqb1[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
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
#endif
                    for(int y = 0; y < _winSize.height; y++ )
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
                            diff0 = v_pack(t0, t1);
                            // I*gain.x + gain.x
                            v_int16x8 diff[4] =
                            {
                                ((v11 << 5) - vI) & vmask,
                                ((v01 << 5) - vI) & vmask,
                                ((v10 << 5) - vI) & vmask,
                                ((v00 << 5) - vI) & vmask
                            };
                            diff0 = diff0 - vI;
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

                            v_int16x8 vIxy_0 = v_reinterpret_as_s16(v_load(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                            v_int16x8 vIxy_1 = v_reinterpret_as_s16(v_load(dIptr + 8));
                            v_int32x4 vI0, vI1;
                            v_expand(vI, vI0, vI1);

                            for (unsigned int mmi = 0; mmi < 4; mmi++)
                            {
                                // It == 0     ? |It| > sigma13
                                diff0 = vset2 & diff[mmi];
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

                                v_zip(vIxy_0, vIxy_1, v10, v11);
                                v_zip(diff2, diff1, v00, v01);

                                vqb0[mmi] += v_cvt_f32(v_dotprod(v00, v10));
                                vqb1[mmi] += v_cvt_f32(v_dotprod(v01, v11));
                            }
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
                            }
                        }

#else
                        for(int x = 0 ; x < _winSize.width*cn; x++, dIptr += 2 )
                        {
                            if( maskPtr[x] == 0)
                                continue;
                            int illValue =   - Iptr[x];
                            float It[4] = {static_cast<float>((Jptr1[x+cn]<< 5)    + illValue),
                                         static_cast<float>((Jptr[x+cn]<< 5)        + illValue),
                                         static_cast<float>((Jptr1[x]<< 5)        + illValue),
                                         static_cast<float>((Jptr[x] << 5)            + illValue)};



                            int J_val  =  CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                                  Jptr1[x]*iw10 + Jptr1[x+cn]*iw11,
                                                  W_BITS-5);


                            int diff = J_val + illValue;


                            MEstimatorScale += (diff < MEstimatorScale) ? -eta : eta;

                            int abss = (diff < 0) ? -diff : diff;

                            // compute the missmatch vector
                            if( j >= 0)
                            {
                                if( abss > fParam1)
                                {
                                    It[0] = 0;
                                    It[1] = 0;
                                    It[2] = 0;
                                    It[3] = 0;
                                }
                                else if( abss > fParam0 && diff >= 0 )
                                {
                                    It[0] = normSigma2 * (It[0] - fParam1);
                                    It[1] = normSigma2 * (It[1] - fParam1);
                                    It[2] = normSigma2 * (It[2] - fParam1);
                                    It[3] = normSigma2 * (It[3] - fParam1);
                                }
                                else if( abss > fParam0 && diff < 0 )
                                {
                                    It[0] = normSigma2 * (It[0] + fParam1);
                                    It[1] = normSigma2 * (It[1] + fParam1);
                                    It[2] = normSigma2 * (It[2] + fParam1);
                                    It[3] = normSigma2 * (It[3] + fParam1);
                                }
                            }

                            float It0 = It[0];
                            float It1 = It[1];
                            float It2 = It[2];
                            float It3 = It[3];

                            float ixval = static_cast<float>(dIptr[0]);
                            float iyval = static_cast<float>(dIptr[1]);
                            _b0[0] += It0 * ixval;
                            _b0[1] += It1 * ixval;
                            _b0[2] += It2 * ixval;
                            _b0[3] += It3 * ixval;

                            _b1[0] += It0*iyval;
                            _b1[1] += It1*iyval;
                            _b1[2] += It2*iyval;
                            _b1[3] += It3*iyval;

                            // compute the Gradient Matrice
                            if( j == 0)
                            {
                                float tale = normSigma2 * FLT_RESCALE;
                                if( abss < fParam0 || j < 0)
                                {
                                    tale = FLT_RESCALE;
                                }
                                else if( abss > fParam1)
                                {
                                    tale *= 0.01f;
                                }
                                else
                                {
                                    tale *= normSigma2;
                                }

                                A11 += (float)(ixval*ixval)*tale;
                                A12 += (float)(ixval*iyval)*tale;
                                A22 += (float)(iyval*iyval)*tale;
                            }

                        }
#endif
                    }

#if CV_SIMD128
                    MEstimatorScale += eta * v_reduce_sum(veta);
#endif


                    if( j == 0 )
                    {
#if CV_SIMD128
                        A11 = v_reduce_sum(vAxx);
                        A12 = v_reduce_sum(vAxy);
                        A22 = v_reduce_sum(vAyy);
#endif
                        A11 *= FLT_SCALE; // 54866744.
                        A12 *= FLT_SCALE; // -628764.00
                        A22 *= FLT_SCALE; // 19730.000

                        D = A11 * A22 - A12 * A12;
                        float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                                4.f*A12*A12))/(2*winArea);

                        if(  minEig < minEigThreshold || std::abs(D) < FLT_EPSILON)
                        {
                            if( level == 0 && status )
                                status[ptidx] = 0;
                            if( level > 0)
                            {
                                nextPts[ptidx] = backUpNextPt;
                            }
                            break;
                        }

                        D = (1.f / D);

                    }

#if CV_SIMD128
                    float CV_DECL_ALIGNED(16) bbuf[4];
                    for (int mmi = 0; mmi < 4; mmi++)
                    {
                        v_store_aligned(bbuf, vqb0[mmi] + vqb1[mmi]);
                        _b0[mmi] = bbuf[0] + bbuf[2];
                        _b1[mmi] = bbuf[1] + bbuf[3];
                    }
#endif
                    _b0[0] *= FLT_SCALE;_b0[1] *= FLT_SCALE;_b0[2] *= FLT_SCALE;_b0[3] *= FLT_SCALE;
                    _b1[0] *= FLT_SCALE;_b1[1] *= FLT_SCALE;_b1[2] *= FLT_SCALE;_b1[3] *= FLT_SCALE;


                    Mc0[0] =   _b0[0] - _b0[1] - _b0[2] + _b0[3];
                    Mc0[1] =   _b1[0] - _b1[1] - _b1[2] + _b1[3];

                    Mc1[0] =   _b0[1] - _b0[3];
                    Mc1[1] =   _b1[1] - _b1[3];


                    Mc2[0] =   _b0[2] - _b0[3];
                    Mc2[1] =   _b1[2] - _b1[3];


                    Mc3[0] =  _b0[3];
                    Mc3[1] =  _b1[3];

                    float c[8] = {};
                    c[0] = -Mc0[0];
                    c[1] = -Mc1[0];
                    c[2] = -Mc2[0];
                    c[3] = -Mc3[0];
                    c[4] = -Mc0[1];
                    c[5] = -Mc1[1];
                    c[6] = -Mc2[1];
                    c[7] = -Mc3[1];

                    float e0 = 1.f / (c[6] * c[0] - c[4] * c[2]);
                    float e1 = e0 * 0.5f * (c[6] * c[1] + c[7] * c[0] - c[5] * c[2] - c[4] * c[3]);
                    float e2 = e0 * (c[1] * c[7] -c[3] * c[5]);
                    e0 = e1 * e1 - e2;
                    hasSolved = false;
                    if ( e0 > 0)
                    {
                        e0 = sqrt(e0);
                        float _y[2] = {-e1 - e0, e0 - e1};
                        float c0yc1[2] = {c[0] * _y[0] + c[1],
                                            c[0] * _y[1] + c[1]};
                        float _x[2] = {-(c[2] * _y[0] + c[3]) / c0yc1[0],
                                        -(c[2] * _y[1] + c[3]) / c0yc1[1]};
                        bool isIn1 = (_x[0] >=0 && _x[0] <=1 && _y[0] >= 0 && _y[0] <= 1);
                        bool isIn2 = (_x[1] >=0 && _x[1] <=1 && _y[1] >= 0 && _y[1] <= 1);

                        bool isSolution1 = checkSolution(_x[0], _y[0], c );
                        bool isSolution2 = checkSolution(_x[1], _y[1], c );
                        bool isSink1 = isIn1 && isSolution1;
                        bool isSink2 = isIn2 && isSolution2;

                        if ( isSink1 != isSink2)
                        {
                            a = isSink1 ? _x[0] : _x[1];
                            b = isSink1 ? _y[0] : _y[1];
                            ab = a * b;
                            hasSolved = true;
                            delta.x = inextPt.x + a - nextPt.x;
                            delta.y = inextPt.y + b - nextPt.y;
                        } // isIn1 != isIn2
                    }
                }
                else
                {
                    hasSolved = false;
                }
                if( hasSolved == false )
                {

                    cv::Vec2f mismatchVec = ab * Mc0  + Mc1 *a + Mc2 * b + Mc3;
                    delta.x = (A12 * mismatchVec.val[1] - A22 * mismatchVec.val[0]) * D;
                    delta.y = (A12 * mismatchVec.val[0] - A11 * mismatchVec.val[1]) * D;
                    delta.x = MAX(-1.f, MIN(1.f, delta.x));
                    delta.y = MAX(-1.f, MIN(1.f, delta.y));
                    nextPt  += delta;
                    nextPts[ptidx] = nextPt - halfWin;
                }
                else
                {
                    nextPt += delta;
                    nextPts[ptidx] = nextPt - halfWin;
                    break;
                }

                delta.x = ( delta.x != delta.x) ? 0 : delta.x;
                delta.y = ( delta.y != delta.y) ? 0 : delta.y;

                if(j > 0 && (
                    (std::abs(delta.x - prevDelta.x) < 0.01  &&    std::abs(delta.y - prevDelta.y) < 0.01)))
                {
                    nextPts[ptidx]  -= delta*0.5f;
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
}  // namespace
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
        int             _crossSegmentationThreshold,
        const std::vector<float>& _normSigmas,
        float           _minEigenValue
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
        const float FLT_SCALE = (1.f/(1 << 16));//(1.f/(1 << 20)); // 20
        winSize = cv::Size(maxWinSize,maxWinSize);
        int winMaskwidth = roundUp(winSize.width, 16);
        cv::Mat winMaskMatBuf(winMaskwidth, winMaskwidth, tCVMaskType);
        winMaskMatBuf.setTo(1);

        int cn = I.channels(), cn2 = cn*2;
        int winbufwidth = roundUp(winSize.width, 16);
        cv::Size winBufSize(winbufwidth,winbufwidth);


        cv::Matx44f invTensorMat;

        cv::AutoBuffer<deriv_type> _buf(winBufSize.area()*(cn + cn2));
        int derivDepth = DataType<deriv_type>::depth;

        Mat IWinBuf(winBufSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf.data());
        Mat derivIWinBuf(winBufSize, CV_MAKETYPE(derivDepth, cn2), (deriv_type*)_buf.data() + winBufSize.area()*cn);

        for( int ptidx = range.start; ptidx < range.end; ptidx++ )
        {
            Point2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
            Point2f nextPt;
            if( level == maxLevel )
            {
                if( useInitialFlow )
                    nextPt = nextPts[ptidx]*(float)(1./(1 << level));
                else
                    nextPt = prevPt;
            }
            else
                nextPt = nextPts[ptidx]*2.f;
            nextPts[ptidx] = nextPt;

            Point2i iprevPt, inextPt;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);
            int winArea = maxWinSize * maxWinSize;
            cv::Mat winMaskMat(winMaskMatBuf, cv::Rect(0,0, maxWinSize,maxWinSize));
            winMaskMatBuf.setTo(0);
            if( calcWinMaskMat(BI,  windowType, iprevPt,
                                winMaskMat,winSize,halfWin,winArea,
                                minWinSize,maxWinSize) == false)
            continue;
            halfWin = Point2f(static_cast<float>(maxWinSize), static_cast<float>(maxWinSize) ) - halfWin;
            prevPt += halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);
            if( iprevPt.x < 0 || iprevPt.x >= derivI.cols - winSize.width ||
                iprevPt.y < 0 || iprevPt.y >= derivI.rows - winSize.height - 1)
            {
                if( level == 0 )
                {
                    if( status )
                        status[ptidx] = 3;
                    if( err )
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

            // tensor
            float sumIx = 0;
            float sumIy = 0;
            float sumI  = 0;
            float sumW = 0;
            float w1 = 0, w2 = 0; // -IyI
            float dI = 0; // I^2
            float D = 0;

            copyWinBuffers(iw00, iw01, iw10, iw11, winSize, I, derivI, winMaskMat, IWinBuf, derivIWinBuf, iprevPt);

            cv::Mat residualMat = cv::Mat::zeros(winSize.height * (winSize.width + 8) * cn, 1, CV_16SC1);
            cv::Point2f backUpNextPt = nextPt;
                    nextPt += halfWin;
            Point2f prevDelta(0,0);    //relates to h(t-1)
            Point2f prevGain(1,0);
            cv::Point2f gainVec = gainVecs[ptidx];
            cv::Point2f backUpGain = gainVec;
            cv::Size _winSize = winSize;
            int j;
            float MEstimatorScale = 1;
            int buffIdx = 0;
            cv::Mat GMc0, GMc1, GMc2, GMc3;
            cv::Vec4f Mc0, Mc1, Mc2, Mc3;
            for( j = 0; j < criteria.maxCount; j++ )
            {
                cv::Point2f delta(0,0);
                cv::Point2f deltaGain(0,0);
                bool hasSolved = false;
                a = nextPt.x - inextPt.x;
                b = nextPt.y - inextPt.y;
                float ab = a * b;
                if (j == 0
                    || (inextPt.x != cvFloor(nextPt.x) || inextPt.y != cvFloor(nextPt.y) || j % 2 != 0) )
                {
                    inextPt.x = cvFloor(nextPt.x);
                    inextPt.y = cvFloor(nextPt.y);

                    if( inextPt.x < 0 || inextPt.x >= J.cols - winSize.width ||
                        inextPt.y < 0 || inextPt.y >= J.rows - winSize.height - 1)
                    {
                        if( level == 0 && status )
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
                    ab = a * b;
                    iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                    iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                    iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                    iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                    // mismatch

                    if( j == 0)
                    {
                        // tensor
                        w1 = 0; // -IxI
                        w2 = 0; // -IyI
                        dI = 0; // I^2
                        sumIx = 0;
                        sumIy = 0;
                        sumI  = 0;
                        sumW = 0;
                        A11 = 0;
                        A12 = 0;
                        A22 = 0;
                    }

                    if ( j == 0 )
                    {
                        buffIdx = 0;
                        for(int y = 0; y < winSize.height; y++ )
                        {
                            const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                            const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                            const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                            const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                            const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                            for(int x = 0 ; x < winSize.width*cn; x++, dIptr += 2)
                            {
                                if( maskPtr[x] == 0)
                                    continue;
                                int diff = static_cast<int>(CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 + Jptr1[x]*iw10 + Jptr1[x+cn]*iw11, W_BITS-5)
                                    - Iptr[x] + Iptr[x] * gainVec.x +gainVec.y);
                                residualMat.at<short>(buffIdx++) = static_cast<short>(diff);
                            }
                        }
                        /*! Estimation for the residual */
                        cv::Mat residualMatRoi(residualMat, cv::Rect(0,0,1, buffIdx));
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
                    v_float32x4 vqb0[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
                    v_float32x4 vqb1[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
                    v_float32x4 vqb2[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
                    v_float32x4 vqb3[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
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
                    float _b0[4] = {0,0,0,0};
                    float _b1[4] = {0,0,0,0};
                    float _b2[4] = {0,0,0,0};
                    float _b3[4] = {0,0,0,0};
                    for(int y = 0; y < _winSize.height; y++ )
                    {
                        const uchar* Jptr =  J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                        const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                        const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                        const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                        const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
#if CV_SIMD128
                        for(int x = 0 ; x <= _winSize.width*cn; x += 8, dIptr += 8*2 )
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
                            diff0 = v_pack(t0, t1);
                            // I*gain.x + gain.x
                            v_mul_expand(vI, vgain_value, t0, t1);
                            v_int16x8 diff_value = v_pack(t0 >> bitShift, t1 >> bitShift) + vconst_value - vI;

                            v_int16x8 diff[4] =
                            {
                                ((v11 << 5) + diff_value) & vmask,
                                ((v01 << 5) + diff_value) & vmask,
                                ((v10 << 5) + diff_value) & vmask,
                                ((v00 << 5) + diff_value) & vmask
                            };
                            diff0 = diff0 + diff_value;
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

                            v_int16x8 vIxy_0 = v_reinterpret_as_s16(v_load(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                            v_int16x8 vIxy_1 = v_reinterpret_as_s16(v_load(dIptr + 8));
                            v_int32x4 vI0, vI1;
                            v_expand(vI, vI0, vI1);

                            for (unsigned int mmi = 0; mmi < 4; mmi++)
                            {
                                // It == 0     ? |It| > sigma13
                                diff0 = vset2 & diff[mmi];
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

                                v_zip(vIxy_0, vIxy_1, v10, v11);
                                v_zip(diff2, diff1, v00, v01);

                                vqb0[mmi] += v_cvt_f32(v_dotprod(v00, v10));
                                vqb1[mmi] += v_cvt_f32(v_dotprod(v01, v11));

                                vqb2[mmi] += v_cvt_f32(diff0_0 * vI0);
                                vqb2[mmi] += v_cvt_f32(diff0_1 * vI1);

                                vqb3[mmi] += v_cvt_f32(diff0_0);
                                vqb3[mmi] += v_cvt_f32(diff0_1);
                            }
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
                    for(int x = 0 ; x < _winSize.width*cn; x++, dIptr += 2 )
                    {
                        if( maskPtr[x] == 0)
                            continue;

                        short ixval = dIptr[0];
                        short iyval = dIptr[1];
                        int illValue =  static_cast<int>(Iptr[x] * gainVec.x + gainVec.y  - Iptr[x]);

                        int It[4] = {(Jptr1[x+cn]<< 5)    + illValue,
                                        (Jptr[x+cn]<< 5)        + illValue,
                                        (Jptr1[x]<< 5)        + illValue,
                                        (Jptr[x] << 5)            + illValue};


                        int J_val  =  CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                                Jptr1[x]*iw10 + Jptr1[x+cn]*iw11,
                                                W_BITS-5);

                        int diff =  J_val + illValue;



                        MEstimatorScale += (diff < MEstimatorScale) ? -eta : eta;

                        int abss = (diff < 0) ? -diff : diff;


                        // compute the missmatch vector
                        if( j >= 0)
                        {
                            if( abss > fParam1)
                            {
                                It[0] = 0;
                                It[1] = 0;
                                It[2] = 0;
                                It[3] = 0;
                            }
                            else if( abss > static_cast<int>(fParam0) && diff >= 0 )
                            {
                                It[0] = static_cast<int>(normSigma2 * (It[0] - fParam1));
                                It[1] = static_cast<int>(normSigma2 * (It[1] - fParam1));
                                It[2] = static_cast<int>(normSigma2 * (It[2] - fParam1));
                                It[3] = static_cast<int>(normSigma2 * (It[3] - fParam1));
                            }
                            else if( abss > static_cast<int>(fParam0) && diff < 0 )
                            {
                                It[0] = static_cast<int>(normSigma2 * (It[0] + fParam1));
                                It[1] = static_cast<int>(normSigma2 * (It[1] + fParam1));
                                It[2] = static_cast<int>(normSigma2 * (It[2] + fParam1));
                                It[3] = static_cast<int>(normSigma2 * (It[3] + fParam1));
                            }
                        }
                        _b0[0] += (float)(It[0]*dIptr[0]) ;
                        _b0[1] += (float)(It[1]*dIptr[0]) ;
                        _b0[2] += (float)(It[2]*dIptr[0]) ;
                        _b0[3] += (float)(It[3]*dIptr[0]) ;


                        _b1[0] += (float)(It[0]*dIptr[1]) ;
                        _b1[1] += (float)(It[1]*dIptr[1]) ;
                        _b1[2] += (float)(It[2]*dIptr[1]) ;
                        _b1[3] += (float)(It[3]*dIptr[1]) ;

                        _b2[0] += (float)(It[0])*Iptr[x] ;
                        _b2[1] += (float)(It[1])*Iptr[x] ;
                        _b2[2] += (float)(It[2])*Iptr[x] ;
                        _b2[3] += (float)(It[3])*Iptr[x] ;

                        _b3[0] += (float)(It[0]);
                        _b3[1] += (float)(It[1]);
                        _b3[2] += (float)(It[2]);
                        _b3[3] += (float)(It[3]);

                        // compute the Gradient Matrice
                        if( j == 0)
                        {
                            float tale = normSigma2 * FLT_RESCALE;
                            if( abss < fParam0 || j < 0)
                            {
                                tale = FLT_RESCALE;
                            }
                            else if( abss > fParam1)
                            {
                                tale *= 0.01f;
                            }
                            else
                            {
                                tale *= normSigma2;
                            }
                            if( j == 0 )
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
                            sumI += Iptr[x]  * tale;
                            sumW += tale;
                        }

                    }
#endif
                }

#if CV_SIMD128
                MEstimatorScale += eta * v_reduce_sum(veta);
#endif
                if( j == 0 )
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
                    sumI *=FLT_SCALE;
                    sumW *= FLT_SCALE;
                    w1 *= -FLT_SCALE;
                    w2 *= -FLT_SCALE;
                    dI *= FLT_SCALE;
                    A11 *= FLT_SCALE;
                    A12 *= FLT_SCALE;
                    A22 *= FLT_SCALE;

                    D = - A12*A12*sumI*sumI + dI*sumW*A12*A12 + 2*A12*sumI*sumIx*w2 + 2*A12*sumI*sumIy*w1
                        - 2*dI*A12*sumIx*sumIy - 2*sumW*A12*w1*w2 + A11*A22*sumI*sumI - 2*A22*sumI*sumIx*w1
                        - 2*A11*sumI*sumIy*w2 - sumIx*sumIx*w2*w2 + A22*dI*sumIx*sumIx + 2*sumIx*sumIy*w1*w2
                        - sumIy*sumIy*w1*w1 + A11*dI*sumIy*sumIy + A22*sumW*w1*w1 + A11*sumW*w2*w2 - A11*A22*dI*sumW;

                    float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                            4.f*A12*A12))/(2*winArea);
                    if(  minEig < minEigThreshold || std::abs(D) < FLT_EPSILON )
                    {
                        if( level == 0 && status )
                            status[ptidx] = 0;
                        if( level > 0)
                        {
                            nextPts[ptidx] = backUpNextPt;
                            gainVecs[ptidx] = backUpGain;
                        }
                        break;
                    }


                    D = (1.f / D);

                    invTensorMat(0,0) = (A22*sumI*sumI - 2*sumI*sumIy*w2 + dI*sumIy*sumIy + sumW*w2*w2 - A22*dI*sumW)* D;
                    invTensorMat(0,1) = (A12*dI*sumW - A12*sumI * sumI - dI*sumIx*sumIy + sumI*sumIx*w2 + sumI*sumIy*w1 - sumW*w1*w2)* D;
                    invTensorMat(0,2) = (A12*sumI*sumIy - sumIy*sumIy*w1 - A22*sumI*sumIx - A12*sumW*w2 + A22*sumW*w1 + sumIx*sumIy*w2)* D;
                    invTensorMat(0,3) = (A22*dI*sumIx - A12*dI*sumIy - sumIx*w2*w2 + A12*sumI*w2 - A22*sumI*w1 + sumIy*w1*w2) * D;
                    invTensorMat(1,0) = invTensorMat(0,1);
                    invTensorMat(1,1) = (A11*sumI * sumI - 2*sumI*sumIx*w1 + dI*sumIx * sumIx + sumW*w1*w1 - A11*dI*sumW) * D;
                    invTensorMat(1,2) = (A12*sumI*sumIx - A11*sumI*sumIy - sumIx * sumIx*w2 + A11*sumW*w2 - A12*sumW*w1 + sumIx*sumIy*w1) * D;
                    invTensorMat(1,3) = (A11*dI*sumIy - sumIy*w1*w1 - A12*dI*sumIx - A11*sumI*w2 + A12*sumI*w1 + sumIx*w1*w2)* D;
                    invTensorMat(2,0) = invTensorMat(0,2);
                    invTensorMat(2,1) = invTensorMat(1,2);
                    invTensorMat(2,2) = (sumW*A12*A12 - 2*A12*sumIx*sumIy + A22*sumIx*sumIx + A11*sumIy*sumIy - A11*A22*sumW)* D;
                    invTensorMat(2,3) = (A11*A22*sumI - A12*A12*sumI - A11*sumIy*w2 + A12*sumIx*w2 + A12*sumIy*w1 - A22*sumIx*w1)* D;
                    invTensorMat(3,0) = invTensorMat(0,3);
                    invTensorMat(3,1) = invTensorMat(1,3);
                    invTensorMat(3,2) = invTensorMat(2,3);
                    invTensorMat(3,3) = (dI*A12*A12 - 2*A12*w1*w2 + A22*w1*w1 + A11*w2*w2 - A11*A22*dI)* D;
                }


#if CV_SIMD128
                float CV_DECL_ALIGNED(16) bbuf[4];
                for(int mmi = 0; mmi < 4; mmi++)
                {
                    v_store_aligned(bbuf, vqb0[mmi] + vqb1[mmi]);
                    _b0[mmi] = bbuf[0] + bbuf[2];
                    _b1[mmi] = bbuf[1] + bbuf[3];
                    _b2[mmi] = v_reduce_sum(vqb2[mmi]);
                    _b3[mmi] = v_reduce_sum(vqb3[mmi]);
                }
#endif

                _b0[0] *= FLT_SCALE;_b0[1] *= FLT_SCALE;_b0[2] *= FLT_SCALE;_b0[3] *= FLT_SCALE;
                _b1[0] *= FLT_SCALE;_b1[1] *= FLT_SCALE;_b1[2] *= FLT_SCALE;_b1[3] *= FLT_SCALE;
                _b2[0] *= FLT_SCALE;_b2[1] *= FLT_SCALE;_b2[2] *= FLT_SCALE;_b2[3] *= FLT_SCALE;
                _b3[0] *= FLT_SCALE;_b3[1] *= FLT_SCALE;_b3[2] *= FLT_SCALE;_b3[3] *= FLT_SCALE;


                Mc0[0] =   _b0[0] - _b0[1] - _b0[2] + _b0[3];
                Mc0[1] =   _b1[0] - _b1[1] - _b1[2] + _b1[3];
                Mc0[2] = -(_b2[0] - _b2[1] - _b2[2] + _b2[3]);
                Mc0[3] = -(_b3[0] - _b3[1] - _b3[2] + _b3[3]);

                Mc1[0] =   _b0[1] - _b0[3];
                Mc1[1] =   _b1[1] - _b1[3];
                Mc1[2] = -(_b2[1] - _b2[3]);
                Mc1[3] = -(_b3[1] - _b3[3]);


                Mc2[0] =   _b0[2] - _b0[3];
                Mc2[1] =   _b1[2] - _b1[3];
                Mc2[2] = -(_b2[2] - _b2[3]);
                Mc2[3] = -(_b3[2] - _b3[3]);


                Mc3[0] =  _b0[3];
                Mc3[1] =  _b1[3];
                Mc3[2] = -_b2[3];
                Mc3[3] = -_b3[3];

                //
                float c[8] = {};
                c[0] = -Mc0[0];
                c[1] = -Mc1[0];
                c[2] = -Mc2[0];
                c[3] = -Mc3[0];
                c[4] = -Mc0[1];
                c[5] = -Mc1[1];
                c[6] = -Mc2[1];
                c[7] = -Mc3[1];

                float e0 = 1.f / (c[6] * c[0] - c[4] * c[2]);
                float e1 = e0 * 0.5f * (c[6] * c[1] + c[7] * c[0] - c[5] * c[2] - c[4] * c[3]);
                float e2 = e0 * (c[1] * c[7] -c[3] * c[5]);
                e0 = e1 * e1 - e2;
                hasSolved = false;
                if ( e0 > 0)
                {
                    e0 = sqrt(e0);
                    float _y[2] = {-e1 - e0, e0 - e1};
                    float c0yc1[2] = {c[0] * _y[0] + c[1],
                                        c[0] * _y[1] + c[1]};
                    float _x[2] = {-(c[2] * _y[0] + c[3]) / c0yc1[0],
                                    -(c[2] * _y[1] + c[3]) / c0yc1[1]};
                    bool isIn1 = (_x[0] >=0 && _x[0] <=1 && _y[0] >= 0 && _y[0] <= 1);
                    bool isIn2 = (_x[1] >=0 && _x[1] <=1 && _y[1] >= 0 && _y[1] <= 1);

                    bool isSolution1 = checkSolution(_x[0], _y[0], c );
                    bool isSolution2 = checkSolution(_x[1], _y[1], c );
                    bool isSink1 = isIn1 && isSolution1;
                    bool isSink2 = isIn2 && isSolution2;

                    if ( isSink1 != isSink2)
                    {
                        a = isSink1 ? _x[0] : _x[1];
                        b = isSink1 ? _y[0] : _y[1];
                        ab = a * b;
                        hasSolved = true;
                        delta.x = inextPt.x + a - nextPt.x;
                        delta.y = inextPt.y + b - nextPt.y;

                        cv::Vec4f mismatchVec = ab * Mc0  + Mc1 *a + Mc2 * b + Mc3;
                        deltaGain = est_DeltaGain(invTensorMat, mismatchVec);

                    } // isIn1 != isIn2
                }
                }
                else
                {
                    hasSolved = false;
                }
                if( hasSolved == false )
                {

                    cv::Vec4f mismatchVec = ab * Mc0  + Mc1 *a + Mc2 * b + Mc3;
                    est_Result(invTensorMat, mismatchVec, delta, deltaGain);

                    delta.x  = MAX(-1.f, MIN( 1.f , delta.x));
                    delta.y  = MAX(-1.f, MIN( 1.f , delta.y));


                    if( j == 0)
                        prevGain = deltaGain;
                    gainVec += deltaGain;
                    nextPt  += delta    ;
                    nextPts[ptidx] = nextPt - halfWin;
                    gainVecs[ptidx]= gainVec;

                }
                else
                {
                    nextPt += delta;
                    nextPts[ptidx] = nextPt - halfWin;
                    gainVecs[ptidx]= gainVec + deltaGain;
                    break;
                }

                if(j > 0 && (
                    (std::abs(delta.x - prevDelta.x) < 0.01  &&    std::abs(delta.y - prevDelta.y) < 0.01)
                 || ((delta.ddot(delta) <= 0.001) && std::abs(prevGain.x - deltaGain.x) < 0.01)))
                {
                            nextPts[ptidx]  += delta*0.5f;
                    gainVecs[ptidx] -= deltaGain* 0.5f;
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
    cv::Point2f*        gainVecs;        // gain vector x -> multiplier y -> offset
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
}} // namespace
namespace beplk {
namespace ica {
class TrackerInvoker  : public cv::ParallelLoopBody
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
        int             _crossSegmentationThreshold,
        float           _minEigenValue)
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

        const Mat& I    = *prevImg;
        const Mat& J    = *nextImg;
        const Mat& derivI = *prevDeriv;
        const Mat& BI = *rgbPrevImg;

        winSize = cv::Size(maxWinSize,maxWinSize);
        int winMaskwidth = roundUp(winSize.width, 8) * 2;
        cv::Mat winMaskMatBuf(winMaskwidth, winMaskwidth, tCVMaskType);
        winMaskMatBuf.setTo(1);

        const float FLT_SCALE = (1.f/(1 << 20)); // 20

        int j, cn = I.channels(), cn2 = cn*2;
        int winbufwidth = roundUp(winSize.width, 8);
        cv::Size winBufSize(winbufwidth,winbufwidth);

        std::vector<short> _buf(winBufSize.area()*(cn + cn2));
        Mat IWinBuf(winBufSize, CV_MAKETYPE(CV_16S, cn), &_buf[0]);
        Mat derivIWinBuf(winBufSize, CV_MAKETYPE(CV_16S, cn2), &_buf[winBufSize.area()*cn]);

        for( int ptidx = range.start; ptidx < range.end; ptidx++ )
        {
            Point2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
            Point2f nextPt;
            if( level == maxLevel )
            {
                if( useInitialFlow )
                    nextPt = nextPts[ptidx]*(float)(1./(1 << level));
                else
                    nextPt = prevPt;
            }
            else
                nextPt = nextPts[ptidx]*2.f;
            nextPts[ptidx] = nextPt;

            Point2i iprevPt, inextPt;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);
            int winArea = maxWinSize * maxWinSize;
            cv::Mat winMaskMat(winMaskMatBuf, cv::Rect(0,0, maxWinSize,maxWinSize));

            if( calcWinMaskMat(BI, windowType, iprevPt,
                    winMaskMat,winSize,halfWin,winArea,
                    minWinSize,maxWinSize) == false)
                continue;

            halfWin = Point2f(static_cast<float>(maxWinSize), static_cast<float>(maxWinSize)) - halfWin;
            prevPt += halfWin;
            iprevPt.x = cvFloor(prevPt.x);
            iprevPt.y = cvFloor(prevPt.y);

            if( iprevPt.x < 0 || iprevPt.x >= derivI.cols - winSize.width ||
                iprevPt.y < 0 || iprevPt.y >= derivI.rows - winSize.height - 1)
            {
                if( level == 0 )
                {
                    if( status )
                        status[ptidx] = 3;
                    if( err )
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

            copyWinBuffers(iw00, iw01, iw10, iw11, winSize, I, derivI, winMaskMat, IWinBuf, derivIWinBuf, A11, A22, A12, iprevPt);

            float D = A11*A22 - A12*A12;
            float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                             4.f*A12*A12))/(2 * winArea);

            if( err )
                err[ptidx] = (float)minEig;

            if( D < FLT_EPSILON )
            {
                if( level == 0 && status )
                    status[ptidx] = 0;
                continue;
            }

            D = 1.f/D;

            cv::Point2f backUpNextPt = nextPt;
            nextPt += halfWin;
            Point2f prevDelta(0,0);

            for( j = 0; j < criteria.maxCount; j++ )
            {
                cv::Point2f delta;
                bool hasSolved = false;
                a = nextPt.x - cvFloor(nextPt.x);
                b = nextPt.y - cvFloor(nextPt.y);
                float ab = a * b;

                float c[8] = {};

                if( (inextPt.x != cvFloor(nextPt.x) || inextPt.y != cvFloor(nextPt.y) || j == 0))
                {
                    inextPt.x = cvFloor(nextPt.x);
                    inextPt.y = cvFloor(nextPt.y);
                    if( inextPt.x < 0 || inextPt.x >= J.cols - winSize.width ||
                        inextPt.y < 0 || inextPt.y >= J.rows - winSize.height - 1)
                    {
                        if( level == 0 && status )
                            status[ptidx] = 3;
                        if (level > 0)
                            nextPts[ptidx] = backUpNextPt;
                        break;
                    }


                    iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                    iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                    iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                    iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

                    float _b1[4] = {0,0,0,0};
                    float _b2[4] = {0,0,0,0};
#if CV_SIMD128
                    v_float32x4 vqb0[4] = {v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32()};
                    v_float32x4 vqb1[4] = {v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32()};
                    v_int16x8 vmax_val_16 = v_setall_s16(std::numeric_limits<unsigned short>::max());
#endif
                    for(int y = 0; y < winSize.height; y++ )
                    {
                        const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                        const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                        const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                        const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
 #if CV_SIMD128
                        const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                        for(int x = 0 ; x <= winSize.width*cn; x += 8, dIptr += 8*2 )
                        {
                            v_int16x8 vI = v_reinterpret_as_s16(v_load(Iptr + x)), diff0, diff1;
                            v_int16x8 v00 = v_reinterpret_as_s16(v_load_expand(Jptr + x));
                            v_int16x8 v01 = v_reinterpret_as_s16(v_load_expand(Jptr + x + cn));
                            v_int16x8 v10 = v_reinterpret_as_s16(v_load_expand(Jptr1 + x));
                            v_int16x8 v11 = v_reinterpret_as_s16(v_load_expand(Jptr1 + x + cn));
                            v_int16x8 vmask = v_reinterpret_as_s16(v_load_expand(maskPtr + x)) * vmax_val_16;

                            v_int16x8 diff[4] =
                            {
                                ((v00 << 5) - vI) & vmask,
                                ((v01 << 5) - vI) & vmask,
                                ((v10 << 5) - vI) & vmask,
                                ((v11 << 5) - vI) & vmask,
                            };

                            v_int16x8 vIxy_0 = v_reinterpret_as_s16(v_load(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                            v_int16x8 vIxy_1 = v_reinterpret_as_s16(v_load(dIptr + 8));
                            for (unsigned int mmi = 0; mmi < 4; mmi++)
                            {
                                v_zip(diff[mmi], diff[mmi], diff1, diff0);
                                v_zip(vIxy_0, vIxy_1, v10, v11);
                                v_zip(diff1, diff0, v00, v01);
                                vqb0[mmi] += v_cvt_f32(v_dotprod(v00, v10));
                                vqb1[mmi] += v_cvt_f32(v_dotprod(v01, v11));
                            }
                         }
#else
                        for(int x = 0 ; x < winSize.width*cn; x++, dIptr += 2 )
                        {
                            if( dIptr[0] == 0 && dIptr[1] == 0)
                                continue;
                            short It[4] = {
                                (short)((Jptr [x]    << 5)        - Iptr[x]),
                                (short)((Jptr [x+cn] << 5)        - Iptr[x]),
                                (short)((Jptr1[x]    << 5)        - Iptr[x]),
                                (short)((Jptr1[x+cn] << 5)        - Iptr[x])
                            };
                            _b1[0] += (float)(It[0]*dIptr[0]);
                            _b1[1] += (float)(It[1]*dIptr[0]);
                            _b1[2] += (float)(It[2]*dIptr[0]);
                            _b1[3] += (float)(It[3]*dIptr[0]);

                            _b2[0] += (float)(It[0]*dIptr[1]);
                            _b2[1] += (float)(It[1]*dIptr[1]);
                            _b2[2] += (float)(It[2]*dIptr[1]);
                            _b2[3] += (float)(It[3]*dIptr[1]);
                        }
#endif
                    }

#if CV_SIMD128
                    float CV_DECL_ALIGNED(16) bbuf[4];
                    for (int mmi = 0; mmi < 4; mmi++)
                    {
                        v_store_aligned(bbuf, vqb0[mmi] + vqb1[mmi]);
                        _b1[mmi] = bbuf[0] + bbuf[2];
                        _b2[mmi] = bbuf[1] + bbuf[3];
                    }
#endif
                    _b1[0] *= FLT_SCALE;
                    _b1[1] *= FLT_SCALE;
                    _b1[2] *= FLT_SCALE;
                    _b1[3] *= FLT_SCALE;
                    _b2[0] *= FLT_SCALE;
                    _b2[1] *= FLT_SCALE;
                    _b2[2] *= FLT_SCALE;
                    _b2[3] *= FLT_SCALE;

                    float c0[4] = {    _b1[3] + _b1[0] - _b1[2] - _b1[1],
                                    _b1[1] - _b1[0],
                                    _b1[2] - _b1[0],
                                    _b1[0]};
                    float c1[4] = {    _b2[3] + _b2[0] - _b2[2] - _b2[1],
                                    _b2[1] - _b2[0],
                                    _b2[2] - _b2[0],
                                    _b2[0]};

                    float DA12 = A12 * D ;
                    float DA22 = A22 * D ;
                    float DA11 = A11 * D ;
                    c[0]    = DA12 * c1[0] - DA22 * c0[0];
                    c[1]    = DA12 * c1[1] - DA22 * c0[1];
                    c[2]    = DA12 * c1[2] - DA22 * c0[2];
                    c[3]    = DA12 * c1[3] - DA22 * c0[3];
                    c[4]    = DA12 * c0[0] - DA11 * c1[0];
                    c[5]    = DA12 * c0[1] - DA11 * c1[1];
                    c[6]    = DA12 * c0[2] - DA11 * c1[2];
                    c[7]    = DA12 * c0[3] - DA11 * c1[3];

                    float e0 = 1.f / (c[6] * c[0] - c[4] * c[2]);
                    float e1 = e0 * 0.5f * (c[6] * c[1] + c[7] * c[0] - c[5] * c[2] - c[4] * c[3]);
                    float e2 = e0 * (c[1] * c[7] -c[3] * c[5]);
                    e0 = e1 * e1 - e2;
                    if ( e0 >= 0)
                    {
                        e0 = sqrt(e0);
                        float _y[2] = {-e1 - e0, e0 - e1};
                        float c0yc1[2] = {c[0] * _y[0] + c[1],
                                          c[0] * _y[1] + c[1]};

                        float _x[2] = {-(c[2] * _y[0] + c[3]) / c0yc1[0],
                                       -(c[2] * _y[1] + c[3]) / c0yc1[1]};

                        bool isIn1 = (_x[0] >=0 && _x[0] <=1 && _y[0] >= 0 && _y[0] <= 1);
                        bool isIn2 = (_x[1] >=0 && _x[1] <=1 && _y[1] >= 0 && _y[1] <= 1);


                        bool isSink1 = isIn1 && checkSolution(_x[0], _y[0], c );
                        bool isSink2 = isIn2 && checkSolution(_x[1], _y[1], c );


                        //if( isSink1 != isSink2 )
                        {
                            if( isSink1 )
                            {
                                hasSolved = true;
                                delta.x = inextPt.x + _x[0] - nextPt.x;
                                delta.y = inextPt.y + _y[0] - nextPt.y;
                            }
                            if (isSink2 )
                            {
                                hasSolved = true;
                                delta.x = inextPt.x + _x[1] - nextPt.x;
                                delta.y = inextPt.y + _y[1] - nextPt.y;
                            }
                        } // isIn1 != isIn2
                    } // e0 >= 0
                }
                else
                    hasSolved = false;
                if(hasSolved == false)
                {
                    delta = Point2f( c[0] * ab + c[1] * a + c[2] * b + c[3],
                                     c[4] * ab + c[5] * a + c[6] * b + c[7]);
                    nextPt += 0.7 * delta;
                    nextPts[ptidx] = nextPt - halfWin;
                }
                else
                {
                    nextPt += delta;
                    nextPts[ptidx] = nextPt - halfWin;
                    break;
                }


                if( delta.ddot(delta) <= criteria.epsilon)
                    break;

                if(j > 0 && std::abs(delta.x - prevDelta.x) < 0.01  &&
                            std::abs(delta.y - prevDelta.y) < 0.01)
                {
                    nextPts[ptidx]  -= delta*0.35f;
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
    int                 crossSegmentationThreshold;
};
}
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
            int             _crossSegmentationThreshold,
            float           _minEigenValue
        )
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
            const float FLT_SCALE = (1.f / (1 << 16));//(1.f/(1 << 20)); // 20
            winSize = cv::Size(maxWinSize, maxWinSize);
            int winMaskwidth = roundUp(winSize.width, 16);
            cv::Mat winMaskMatBuf(winMaskwidth, winMaskwidth, tCVMaskType);
            winMaskMatBuf.setTo(1);

            int cn = I.channels(), cn2 = cn * 2;
            int winbufwidth = roundUp(winSize.width, 16);
            cv::Size winBufSize(winbufwidth, winbufwidth);


            cv::Matx44f invTensorMat;

            cv::AutoBuffer<deriv_type> _buf(winBufSize.area()*(cn + cn2));
            int derivDepth = DataType<deriv_type>::depth;

            Mat IWinBuf(winBufSize, CV_MAKETYPE(derivDepth, cn), (deriv_type*)_buf.data());
            Mat derivIWinBuf(winBufSize, CV_MAKETYPE(derivDepth, cn2), (deriv_type*)_buf.data() + winBufSize.area()*cn);

            for (int ptidx = range.start; ptidx < range.end; ptidx++)
            {
                Point2f prevPt = prevPts[ptidx] * (float)(1. / (1 << level));
                Point2f nextPt;
                if (level == maxLevel)
                {
                    if (useInitialFlow)
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
                winMaskMatBuf.setTo(0);
                if (calcWinMaskMat(BI, windowType, iprevPt,
                    winMaskMat, winSize, halfWin, winArea,
                    minWinSize, maxWinSize) == false)
                    continue;
                halfWin = Point2f(static_cast<float>(maxWinSize), static_cast<float>(maxWinSize)) - halfWin;
                prevPt += halfWin;
                iprevPt.x = cvFloor(prevPt.x);
                iprevPt.y = cvFloor(prevPt.y);
                if (iprevPt.x < 0 || iprevPt.x >= derivI.cols - winSize.width ||
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
                Point2f prevDelta(0, 0);    //relates to h(t-1)
                Point2f prevGain(1, 0);
                cv::Point2f gainVec = gainVecs[ptidx];
                cv::Point2f backUpGain = gainVec;
                cv::Size _winSize = winSize;
                int j;
                cv::Mat GMc0, GMc1, GMc2, GMc3;
                cv::Vec4f Mc0, Mc1, Mc2, Mc3;
                for (j = 0; j < criteria.maxCount; j++)
                {
                    cv::Point2f delta(0, 0);
                    cv::Point2f deltaGain(0, 0);
                    bool hasSolved = false;
                    a = nextPt.x - inextPt.x;
                    b = nextPt.y - inextPt.y;
                    float ab = a * b;
                    if (j == 0
                        || (inextPt.x != cvFloor(nextPt.x) || inextPt.y != cvFloor(nextPt.y) || j % 2 != 0))
                    {
                        inextPt.x = cvFloor(nextPt.x);
                        inextPt.y = cvFloor(nextPt.y);

                        if (inextPt.x < 0 || inextPt.x >= J.cols - winSize.width ||
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
                        ab = a * b;
                        iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                        iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                        iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                        iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                        // mismatch

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

#if CV_SIMD128
                        v_int16x8 vqw0 = v_int16x8((short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01), (short)(iw00), (short)(iw01));
                        v_int16x8 vqw1 = v_int16x8((short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11), (short)(iw10), (short)(iw11));
                        v_float32x4 vqb0[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
                        v_float32x4 vqb1[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
                        v_float32x4 vqb2[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
                        v_float32x4 vqb3[4] = { v_setzero_f32(), v_setzero_f32(), v_setzero_f32(), v_setzero_f32() };
                        v_float32x4 vsumW1 = v_setzero_f32(), vsumW2 = v_setzero_f32();
                        v_float32x4 vsumIy = v_setzero_f32(), vsumIx = v_setzero_f32(), vsumI = v_setzero_f32(), vsumDI = v_setzero_f32();
                        v_float32x4 vAxx = v_setzero_f32(), vAxy = v_setzero_f32(), vAyy = v_setzero_f32();

                        v_int32x4 vdelta = v_setall_s32(1 << (W_BITS - 5 - 1));
                        v_int16x8 vmax_val_16 = v_setall_s16(std::numeric_limits<unsigned short>::max());

                        float gainVal = gainVec.x > 0 ? gainVec.x : -gainVec.x;
                        int bitShift = gainVec.x == 0 ? 1 : cvCeil(log(200.f / gainVal) / log(2.f));
                        v_int16x8 vgain_value = v_setall_s16(static_cast<short>(gainVec.x * (float)(1 << bitShift)));
                        v_int16x8 vconst_value = v_setall_s16(static_cast<short>(gainVec.y));
#endif
                        float _b0[4] = { 0,0,0,0 };
                        float _b1[4] = { 0,0,0,0 };
                        float _b2[4] = { 0,0,0,0 };
                        float _b3[4] = { 0,0,0,0 };
                        for (int y = 0; y < _winSize.height; y++)
                        {
                            const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                            const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                            const short* Iptr = IWinBuf.ptr<short>(y, 0);
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
                                diff0 = v_pack(t0, t1);
                                // I*gain.x + gain.x
                                v_mul_expand(vI, vgain_value, t0, t1);
                                v_int16x8 diff_value = v_pack(t0 >> bitShift, t1 >> bitShift) + vconst_value - vI;

                                v_int16x8 diff[4] =
                                {
                                    ((v11 << 5) + diff_value) & vmask,
                                    ((v01 << 5) + diff_value) & vmask,
                                    ((v10 << 5) + diff_value) & vmask,
                                    ((v00 << 5) + diff_value) & vmask
                                };
                                v_int16x8 vIxy_0 = v_reinterpret_as_s16(v_load(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                                v_int16x8 vIxy_1 = v_reinterpret_as_s16(v_load(dIptr + 8));
                                v_zip(vIxy_0, vIxy_1, v10, v11);
                                v_int32x4 vI0, vI1;
                                v_expand(vI, vI0, vI1);

                                for (unsigned int mmi = 0; mmi < 4; mmi++)
                                {
                                    v_int32x4 diff0_0;
                                    v_int32x4 diff0_1;
                                    v_expand(diff[mmi], diff0_0, diff0_1);
                                    v_zip(diff[mmi], diff[mmi], diff2, diff1);

                                    v_zip(diff2, diff1, v00, v01);
                                    vqb0[mmi] += v_cvt_f32(v_dotprod(v00, v10));
                                    vqb1[mmi] += v_cvt_f32(v_dotprod(v01, v11));

                                    vqb2[mmi] += v_cvt_f32(diff0_0 * vI0);
                                    vqb2[mmi] += v_cvt_f32(diff0_1 * vI1);

                                    vqb3[mmi] += v_cvt_f32(diff0_0);
                                    vqb3[mmi] += v_cvt_f32(diff0_1);
                                }
                                if (j == 0)
                                {
                                    v00 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(vIxy_0))));
                                    v_expand(v00, t1, t0);

                                    v_float32x4 vI_ps = v_cvt_f32(vI0);

                                    v_float32x4 fy = v_cvt_f32(t0);
                                    v_float32x4 fx = v_cvt_f32(t1);

                                    vAyy = v_muladd(fy, fy, vAyy);
                                    vAxy = v_muladd(fx, fy, vAxy);
                                    vAxx = v_muladd(fx, fx, vAxx);

                                    // sumIx und sumIy
                                    vsumIx += fx;
                                    vsumIy += fy;

                                    vsumW1 += vI_ps * fx;
                                    vsumW2 += vI_ps * fy;

                                    // sumI
                                    vsumI += vI_ps;

                                    // sumDI
                                    vsumDI += vI_ps * vI_ps;

                                    v01 = v_reinterpret_as_s16(v_interleave_pairs(v_reinterpret_as_s32(v_interleave_pairs(vIxy_1))));
                                    v_expand(v01, t1, t0);
                                    vI_ps = v_cvt_f32(vI1);

                                    fy = v_cvt_f32(t0);
                                    fx = v_cvt_f32(t1);

                                    // A11 - A22
                                    vAyy = v_muladd(fy, fy, vAyy);
                                    vAxy = v_muladd(fx, fy, vAxy);
                                    vAxx = v_muladd(fx, fx, vAxx);

                                    // sumIx und sumIy
                                    vsumIx += fx;
                                    vsumIy += fy;

                                    vsumW1 += vI_ps * fx;
                                    vsumW2 += vI_ps * fy;

                                    // sumI
                                    vsumI += vI_ps;

                                    // sumDI
                                    vsumDI += vI_ps * vI_ps;
                                }
                            }

#else
                            for (int x = 0; x < _winSize.width*cn; x++, dIptr += 2)
                            {
                                if (maskPtr[x] == 0)
                                    continue;

                                short ixval = dIptr[0];
                                short iyval = dIptr[1];
                                int illValue = static_cast<int>(Iptr[x] * gainVec.x + gainVec.y - Iptr[x]);

                                int It[4] = { (Jptr1[x + cn] << 5) + illValue,
                                                (Jptr[x + cn] << 5) + illValue,
                                                (Jptr1[x] << 5) + illValue,
                                                (Jptr[x] << 5) + illValue };

                                // compute the missmatch vector
                                _b0[0] += (float)(It[0] * dIptr[0]);
                                _b0[1] += (float)(It[1] * dIptr[0]);
                                _b0[2] += (float)(It[2] * dIptr[0]);
                                _b0[3] += (float)(It[3] * dIptr[0]);

                                _b1[0] += (float)(It[0] * dIptr[1]);
                                _b1[1] += (float)(It[1] * dIptr[1]);
                                _b1[2] += (float)(It[2] * dIptr[1]);
                                _b1[3] += (float)(It[3] * dIptr[1]);

                                _b2[0] += (float)(It[0])*Iptr[x];
                                _b2[1] += (float)(It[1])*Iptr[x];
                                _b2[2] += (float)(It[2])*Iptr[x];
                                _b2[3] += (float)(It[3])*Iptr[x];

                                _b3[0] += (float)(It[0]);
                                _b3[1] += (float)(It[1]);
                                _b3[2] += (float)(It[2]);
                                _b3[3] += (float)(It[3]);

                                // compute the Gradient Matrice
                                if (j == 0)
                                {
                                    A11 += (float)(ixval*ixval);
                                    A12 += (float)(ixval*iyval);
                                    A22 += (float)(iyval*iyval);
                                    dI += Iptr[x] * Iptr[x];
                                    float dx = static_cast<float>(dIptr[0]);
                                    float dy = static_cast<float>(dIptr[1]);
                                    sumIx += dx;
                                    sumIy += dy;
                                    w1 += dx * Iptr[x];
                                    w2 += dy * Iptr[x];
                                    sumI += Iptr[x];
                                }

                            }
#endif
                        }

                        if (j == 0)
                        {
#if CV_SIMD128
                            w1 = v_reduce_sum(vsumW1);
                            w2 = v_reduce_sum(vsumW2);
                            dI = v_reduce_sum(vsumDI);
                            sumI = v_reduce_sum(vsumI);
                            sumIx = v_reduce_sum(vsumIx);
                            sumIy = v_reduce_sum(vsumIy);
                            A11 = v_reduce_sum(vAxx);
                            A12 = v_reduce_sum(vAxy);
                            A22 = v_reduce_sum(vAyy);
#endif
                            sumIx *= -FLT_SCALE;
                            sumIy *= -FLT_SCALE;
                            sumI *= FLT_SCALE;
                            sumW = winArea * FLT_SCALE;
                            w1 *= -FLT_SCALE;
                            w2 *= -FLT_SCALE;
                            dI *= FLT_SCALE;


                            A11 *= FLT_SCALE;
                            A12 *= FLT_SCALE;
                            A22 *= FLT_SCALE;

                            D = -A12 * A12*sumI*sumI + dI * sumW*A12*A12 + 2 * A12*sumI*sumIx*w2 + 2 * A12*sumI*sumIy*w1
                                - 2 * dI*A12*sumIx*sumIy - 2 * sumW*A12*w1*w2 + A11 * A22*sumI*sumI - 2 * A22*sumI*sumIx*w1
                                - 2 * A11*sumI*sumIy*w2 - sumIx * sumIx*w2*w2 + A22 * dI*sumIx*sumIx + 2 * sumIx*sumIy*w1*w2
                                - sumIy * sumIy*w1*w1 + A11 * dI*sumIy*sumIy + A22 * sumW*w1*w1 + A11 * sumW*w2*w2 - A11 * A22*dI*sumW;

                            float minEig = (A22 + A11 - std::sqrt((A11 - A22)*(A11 - A22) +
                                4.f*A12*A12)) / (2 * winArea);
                            if (minEig < minEigThreshold || std::abs(D) < FLT_EPSILON)
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
                        }


#if CV_SIMD128
                        float CV_DECL_ALIGNED(16) bbuf[4];
                        for (int mmi = 0; mmi < 4; mmi++)
                        {
                            v_store_aligned(bbuf, vqb0[mmi] + vqb1[mmi]);
                            _b0[mmi] = bbuf[0] + bbuf[2];
                            _b1[mmi] = bbuf[1] + bbuf[3];
                            _b2[mmi] = v_reduce_sum(vqb2[mmi]);
                            _b3[mmi] = v_reduce_sum(vqb3[mmi]);
                        }
#endif

                        _b0[0] *= FLT_SCALE; _b0[1] *= FLT_SCALE; _b0[2] *= FLT_SCALE; _b0[3] *= FLT_SCALE;
                        _b1[0] *= FLT_SCALE; _b1[1] *= FLT_SCALE; _b1[2] *= FLT_SCALE; _b1[3] *= FLT_SCALE;
                        _b2[0] *= FLT_SCALE; _b2[1] *= FLT_SCALE; _b2[2] *= FLT_SCALE; _b2[3] *= FLT_SCALE;
                        _b3[0] *= FLT_SCALE; _b3[1] *= FLT_SCALE; _b3[2] *= FLT_SCALE; _b3[3] *= FLT_SCALE;


                        Mc0[0] = _b0[0] - _b0[1] - _b0[2] + _b0[3];
                        Mc0[1] = _b1[0] - _b1[1] - _b1[2] + _b1[3];
                        Mc0[2] = -(_b2[0] - _b2[1] - _b2[2] + _b2[3]);
                        Mc0[3] = -(_b3[0] - _b3[1] - _b3[2] + _b3[3]);

                        Mc1[0] = _b0[1] - _b0[3];
                        Mc1[1] = _b1[1] - _b1[3];
                        Mc1[2] = -(_b2[1] - _b2[3]);
                        Mc1[3] = -(_b3[1] - _b3[3]);


                        Mc2[0] = _b0[2] - _b0[3];
                        Mc2[1] = _b1[2] - _b1[3];
                        Mc2[2] = -(_b2[2] - _b2[3]);
                        Mc2[3] = -(_b3[2] - _b3[3]);


                        Mc3[0] = _b0[3];
                        Mc3[1] = _b1[3];
                        Mc3[2] = -_b2[3];
                        Mc3[3] = -_b3[3];

                        //
                        float c[8] = {};
                        c[0] = -Mc0[0];
                        c[1] = -Mc1[0];
                        c[2] = -Mc2[0];
                        c[3] = -Mc3[0];
                        c[4] = -Mc0[1];
                        c[5] = -Mc1[1];
                        c[6] = -Mc2[1];
                        c[7] = -Mc3[1];

                        float e0 = 1.f / (c[6] * c[0] - c[4] * c[2]);
                        float e1 = e0 * 0.5f * (c[6] * c[1] + c[7] * c[0] - c[5] * c[2] - c[4] * c[3]);
                        float e2 = e0 * (c[1] * c[7] - c[3] * c[5]);
                        e0 = e1 * e1 - e2;
                        hasSolved = false;
                        if (e0 > 0)
                        {
                            e0 = sqrt(e0);
                            float _y[2] = { -e1 - e0, e0 - e1 };
                            float c0yc1[2] = { c[0] * _y[0] + c[1],
                                                c[0] * _y[1] + c[1] };
                            float _x[2] = { -(c[2] * _y[0] + c[3]) / c0yc1[0],
                                            -(c[2] * _y[1] + c[3]) / c0yc1[1] };
                            bool isIn1 = (_x[0] >= 0 && _x[0] <= 1 && _y[0] >= 0 && _y[0] <= 1);
                            bool isIn2 = (_x[1] >= 0 && _x[1] <= 1 && _y[1] >= 0 && _y[1] <= 1);

                            bool isSolution1 = checkSolution(_x[0], _y[0], c);
                            bool isSolution2 = checkSolution(_x[1], _y[1], c);
                            bool isSink1 = isIn1 && isSolution1;
                            bool isSink2 = isIn2 && isSolution2;

                            if (isSink1 != isSink2)
                            {
                                a = isSink1 ? _x[0] : _x[1];
                                b = isSink1 ? _y[0] : _y[1];
                                ab = a * b;
                                hasSolved = true;
                                delta.x = inextPt.x + a - nextPt.x;
                                delta.y = inextPt.y + b - nextPt.y;

                                cv::Vec4f mismatchVec = ab * Mc0 + Mc1 * a + Mc2 * b + Mc3;
                                deltaGain = est_DeltaGain(invTensorMat, mismatchVec);

                            } // isIn1 != isIn2
                        }
                    }
                    else
                    {
                        hasSolved = false;
                    }
                    if (hasSolved == false)
                    {

                        cv::Vec4f mismatchVec = ab * Mc0 + Mc1 * a + Mc2 * b + Mc3;
                        est_Result(invTensorMat, mismatchVec, delta, deltaGain);

                        delta.x = MAX(-1.f, MIN(1.f, delta.x));
                        delta.y = MAX(-1.f, MIN(1.f, delta.y));


                        if (j == 0)
                            prevGain = deltaGain;
                        gainVec += deltaGain;
                        nextPt += delta;
                        nextPts[ptidx] = nextPt - halfWin;
                        gainVecs[ptidx] = gainVec;

                    }
                    else
                    {
                        nextPt += delta;
                        nextPts[ptidx] = nextPt - halfWin;
                        gainVecs[ptidx] = gainVec + deltaGain;
                        break;
                    }

                    if (j > 0 && (
                        (std::abs(delta.x - prevDelta.x) < 0.01  &&    std::abs(delta.y - prevDelta.y) < 0.01)
                        || ((delta.ddot(delta) <= 0.001) && std::abs(prevGain.x - deltaGain.x) < 0.01)))
                    {
                        nextPts[ptidx] += delta * 0.5f;
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
        cv::Point2f*        gainVecs;        // gain vector x -> multiplier y -> offset
        float*              err;
        int                 maxWinSize;
        int                 minWinSize;
        TermCriteria        criteria;
        int                 level;
        int                 maxLevel;
        int                 windowType;
        float               minEigThreshold;
        bool                useInitialFlow;
        int                 crossSegmentationThreshold;
    };
}}}}  // namespace
#endif
