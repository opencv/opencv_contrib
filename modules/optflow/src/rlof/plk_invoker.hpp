// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _PLK_INVOKER_HPP_
#define _PLK_INVOKER_HPP_
#include "rlof_invokerbase.hpp"


namespace cv {
namespace optflow {
namespace plk {

// implementierung ohne SSE
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
        float           _minEigenValue,
        int             _crossSegmentationThreshold)
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
        cv::Size    winSize;
        cv::Point2f halfWin;

        const Mat& I    = *prevImg;
        const Mat& J    = *nextImg;
        const Mat& derivI = *prevDeriv;
        const Mat& BI = *rgbPrevImg;

        winSize = cv::Size(maxWinSize,maxWinSize);
        int winMaskwidth = roundUp(winSize.width, 16);
        cv::Mat winMaskMatBuf(winMaskwidth, winMaskwidth, tCVMaskType);
        winMaskMatBuf.setTo(1);
        const float FLT_SCALE = (1.f/(1 << 16));//(1.f/(1 << 20)); // 20

        int cn = I.channels(), cn2 = cn*2;
        int winbufwidth = roundUp(winSize.width, 16);
        cv::Size winBufSize(winbufwidth,winbufwidth);

        cv::Matx44f invTensorMat;
        Vec4f mismatchMat;
        Vec4f resultMat;

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
                {
                    nextPt = nextPts[ptidx]*(float)(1./(1 << level));
                }
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
                        continue;
            halfWin = Point2f(static_cast<float>(maxWinSize) ,static_cast<float>(maxWinSize) ) - halfWin;
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
            const int W_BITS = 14, W_BITS1 = 14;
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
 #ifdef RLOF_SSE

            __m128i qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
            __m128i qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
            __m128i z = _mm_setzero_si128();
            __m128i qdelta_d = _mm_set1_epi32(1 << (W_BITS1-1));
            __m128i qdelta = _mm_set1_epi32(1 << (W_BITS1-5-1));
            __m128i mmMask4_epi32;
            __m128i mmMaskSet_epi16   = _mm_set1_epi16(std::numeric_limits<unsigned short>::max());
            get4BitMask(winSize.width, mmMask4_epi32);
#endif
            // extract the patch from the first image, compute covariation matrix of derivatives
            int x, y;
            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* src = I.ptr<uchar>(y + iprevPt.y, 0) + iprevPt.x*cn;
                const uchar* src1 = I.ptr<uchar>(y + iprevPt.y + 1, 0) + iprevPt.x*cn;
                const short* dsrc = derivI.ptr<short>(y + iprevPt.y, 0) + iprevPt.x*cn2;
                const short* dsrc1 = derivI.ptr<short>(y + iprevPt.y + 1, 0) + iprevPt.x*cn2;
                short* Iptr  = IWinBuf.ptr<short>(y, 0);
                short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                x = 0;
#ifdef RLOF_SSE
               for( ; x <= winBufSize.width*cn - 4; x += 4, dsrc += 4*2, dsrc1 += 8, dIptr += 4*2 )
                {
                    __m128i mask_0_7_epi16 = _mm_mullo_epi16(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(maskPtr+x))), mmMaskSet_epi16);
                    __m128i mask_0_3_epi16 = _mm_unpacklo_epi16(mask_0_7_epi16, mask_0_7_epi16);
                    __m128i v00, v01, v10, v11, t0, t1;
                    v00 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x)), z);
                    v01 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + cn)), z);
                    v10 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src1 + x)), z);
                    v11 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src1 + x + cn)), z);

                    t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                       _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
                    t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1-5);
                    if( x + 4 > winSize.width)
                    {
                        t0 = _mm_and_si128(t0, mmMask4_epi32);
                    }
                    t0 = _mm_and_si128(t0, mask_0_3_epi16);
                    _mm_storel_epi64((__m128i*)(Iptr + x), _mm_packs_epi32(t0,t0));


                    v00 = _mm_loadu_si128((const __m128i*)(dsrc));
                    v01 = _mm_loadu_si128((const __m128i*)(dsrc + cn2));
                    v10 = _mm_loadu_si128((const __m128i*)(dsrc1));
                    v11 = _mm_loadu_si128((const __m128i*)(dsrc1 + cn2));

                    t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                       _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
                    t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                                       _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
                    t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta_d), W_BITS1);
                    t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta_d), W_BITS1);
                    v00 = _mm_packs_epi32(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...
                    if( x + 4 > winSize.width)
                    {
                        v00 = _mm_and_si128(v00, mmMask4_epi32);
                    }
                    v00 = _mm_and_si128(v00, mask_0_3_epi16);
                    _mm_storeu_si128((__m128i*)dIptr, v00);
                }
#else
                for( ; x < winSize.width*cn; x++, dsrc += 2, dsrc1 += 2, dIptr += 2 )
                {
                    if( maskPtr[x] == 0)
                    {
                        dIptr[0] = 0;
                        dIptr[1] = 0;
                        continue;
                    }
                    int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                          src1[x]*iw10 + src1[x+cn]*iw11, W_BITS1-5);
                    int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                           dsrc1[0]*iw10 + dsrc1[cn2]*iw11, W_BITS1);
                    int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc1[1]*iw10 +
                                           dsrc1[cn2+1]*iw11, W_BITS1);

                    Iptr[x] = (short)ival;
                    dIptr[0] = (short)ixval;
                    dIptr[1] = (short)iyval;
                }
#endif

            }

            cv::Point2f backUpNextPt = nextPt;
            nextPt += halfWin;
            Point2f prevDelta(0,0);    //relates to h(t-1)
            Point2f prevGain(1,0);
            cv::Point2f gainVec = gainVecs[ptidx];
            cv::Point2f backUpGain = gainVec;
            int j;
            for( j = 0; j < criteria.maxCount; j++ )
            {
                status[ptidx] = static_cast<uchar>(j);
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
                iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                float b1,b2,b3,b4;
                b1 = 0;
                b2 = 0;
                b3 = 0;
                b4 = 0;

                if( j == 0 )
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
#ifdef RLOF_SSE
                qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
                qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
                __m128 qb0 = _mm_setzero_ps(), qb1 = _mm_setzero_ps(), qb2 = _mm_setzero_ps();
                __m128 qb3 = _mm_setzero_ps();
                __m128 mmSumW1 = _mm_setzero_ps(), mmSumW2 = _mm_setzero_ps();
                __m128 mmSumI = _mm_setzero_ps(), mmSumW = _mm_setzero_ps(), mmSumDI = _mm_setzero_ps();
                __m128 mmSumIy = _mm_setzero_ps(),  mmSumIx = _mm_setzero_ps();
                __m128 mmAxx = _mm_setzero_ps(), mmAxy = _mm_setzero_ps(), mmAyy = _mm_setzero_ps();
                float gainVal = gainVec.x > 0 ? gainVec.x : -gainVec.x;
                int bitShift = gainVec.x == 0 ? 1 : cvCeil(log(200.f / gainVal) / log(2.f));
                __m128i mmGainValue_epi16 = _mm_set1_epi16(static_cast<short>(gainVec.x * (float)(1 << bitShift)));
                __m128i mmConstValue_epi16 = _mm_set1_epi16(static_cast<short>(gainVec.y));
#endif
                for( y = 0; y < winSize.height; y++ )
                {
                    const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                    const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                    const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                    const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                    const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                    x = 0;
#ifdef RLOF_SSE

                    for( ; x <= winSize.width*cn; x += 8, dIptr += 8*2 )
                    {
                        // iw = iw << 14 (16384 short / 2)
                        // iq0 =  iw01 |iw00, iw01 |iw00, iw01 |iw00, iw01 |iw00    16 bit
                        __m128i mask_0_7_epi16 = _mm_mullo_epi16(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(maskPtr+x))), mmMaskSet_epi16);

                        // I [0...8]
                        __m128i diff0, diff1;
                        __m128i I_0_7_epi16 = _mm_loadu_si128((const __m128i*)(Iptr + x)); //  element 0 to 7

                        __m128i v00 = _mm_unpacklo_epi8(
                            _mm_loadl_epi64((const __m128i*)(Jptr + x)) // J0, J1, J2, ..., J7, 64 bit nullen
                                , z); //J0 , 0, J1, 0, J2, 0 ... J7,0
                        __m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + cn)), z);
                        __m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr1 + x)), z);
                        __m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr1 + x + cn)), z);


                        __m128i t0 = _mm_add_epi32
                            (_mm_madd_epi16(
                                _mm_unpacklo_epi16(v00, v01), //J0, , J1, J1, J2, J2,  ... J3 , J4
                                qw0),
                            _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));

                        __m128i t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                                                   _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));

                        t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1-5);
                        t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta), W_BITS1-5);


                        __m128i Ilo = _mm_mullo_epi16(mmGainValue_epi16, I_0_7_epi16);
                        __m128i Ihi = _mm_mulhi_epi16(mmGainValue_epi16, I_0_7_epi16);

                        __m128i Igain_0_3_epi32 = _mm_srai_epi32(_mm_unpacklo_epi16(Ilo, Ihi), bitShift);
                        __m128i Igain_4_7_epi32 = _mm_srai_epi32(_mm_unpackhi_epi16(Ilo, Ihi), bitShift);
                        __m128i Igain_epi16 =  _mm_packs_epi32(Igain_0_3_epi32, Igain_4_7_epi32);

                        // diff = J - I + I*gain.x + gain.y
                        __m128i mmDiff_epi16 = _mm_add_epi16(
                            _mm_subs_epi16(_mm_packs_epi32(t0, t1), I_0_7_epi16),    // J - I
                            _mm_add_epi16(Igain_epi16, mmConstValue_epi16));


                        mmDiff_epi16 = _mm_and_si128(mmDiff_epi16, mask_0_7_epi16);


                        __m128i Ixy_0 = _mm_loadu_si128((const __m128i*)(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                        __m128i Ixy_1 = _mm_loadu_si128((const __m128i*)(dIptr + 8));

                        diff1 = _mm_unpackhi_epi16(mmDiff_epi16, mmDiff_epi16); // It4 It4 It5 It5 It6 It6 It7 It7   | It12 It12 It13 It13...
                        diff0 = _mm_unpacklo_epi16(mmDiff_epi16, mmDiff_epi16); // It0 It0 It1 It1 It2 It2 It3 It3   | It8 It8 It9 It9...

                        v10 = _mm_mullo_epi16(Ixy_0, diff0);
                        v11 = _mm_mulhi_epi16(Ixy_0, diff0);
                        v00 = _mm_unpacklo_epi16(v10, v11);
                        v10 = _mm_unpackhi_epi16(v10, v11);
                        qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
                        qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));

                        v10 = _mm_mullo_epi16(Ixy_1, diff1);
                        v11 = _mm_mulhi_epi16(Ixy_1, diff1);
                        v00 = _mm_unpacklo_epi16(v10, v11);
                        v10 = _mm_unpackhi_epi16(v10, v11);
                        qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
                        qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
                        // for set 1  sigma 1
                        // b3 += diff * Iptr[x]
                        Ilo = _mm_mullo_epi16(mmDiff_epi16, I_0_7_epi16);
                        Ihi = _mm_mulhi_epi16(mmDiff_epi16, I_0_7_epi16);
                        v00 = _mm_unpacklo_epi16(Ilo, Ihi);
                        v10 = _mm_unpackhi_epi16(Ilo, Ihi);
                        qb2 = _mm_add_ps(qb2, _mm_cvtepi32_ps(v00));
                        qb2 = _mm_add_ps(qb2, _mm_cvtepi32_ps(v10));

                        // b4+= diff
                        __m128 mmDiff_0_3_ps  = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(mmDiff_epi16)); // (_mm_srai_epi32(_mm_unpacklo_epi16(mmDiff_epi16, mmDiff_epi16),16));
                        __m128 mmDiff_4_7_ps  = _mm_cvtepi32_ps((_mm_srai_epi32(_mm_unpackhi_epi16(mmDiff_epi16, mmDiff_epi16),16)));
                        qb3 = _mm_add_ps(qb3, mmDiff_0_3_ps);
                        qb3 = _mm_add_ps(qb3, mmDiff_4_7_ps);
                        if( j == 0 )
                        {
                            __m128 mask_0_4_ps = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(mask_0_7_epi16));
                            __m128 mask_4_7_ps = _mm_cvtepi32_ps((_mm_srai_epi32(_mm_unpackhi_epi16(mask_0_7_epi16, mask_0_7_epi16), 16)));

                            t0 = _mm_srai_epi32(Ixy_0, 16); // Iy0 Iy1 Iy2 Iy3
                            t1 = _mm_srai_epi32(_mm_slli_epi32(Ixy_0, 16), 16); // Ix0 Ix1 Ix2 Ix3

                            __m128 fy = _mm_blendv_ps(_mm_set1_ps(0), _mm_cvtepi32_ps(t0), mask_0_4_ps);
                            __m128 fx = _mm_blendv_ps(_mm_set1_ps(0), _mm_cvtepi32_ps(t1), mask_0_4_ps);

                            // 0 ... 3
                            __m128 I_ps = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(I_0_7_epi16, I_0_7_epi16), 16));

                            // A11 - A22
                            mmAyy = _mm_add_ps(mmAyy, _mm_mul_ps(fy, fy));
                            mmAxy = _mm_add_ps(mmAxy, _mm_mul_ps(fx, fy));
                            mmAxx = _mm_add_ps(mmAxx, _mm_mul_ps(fx, fx));

                            // sumIx und sumIy
                            mmSumIx = _mm_add_ps(mmSumIx, fx);
                            mmSumIy = _mm_add_ps(mmSumIy, fy);

                            mmSumW1 = _mm_add_ps(mmSumW1, _mm_mul_ps(I_ps, fx));
                            mmSumW2 = _mm_add_ps(mmSumW2, _mm_mul_ps(I_ps, fy));

                            // sumI
                            I_ps  = _mm_blendv_ps(_mm_set1_ps(0), I_ps, mask_0_4_ps);
                            mmSumI = _mm_add_ps(mmSumI,I_ps);

                            // sumDI
                            mmSumDI = _mm_add_ps(mmSumDI, _mm_mul_ps( I_ps, I_ps));

                            t0 = _mm_srai_epi32(Ixy_1, 16); // Iy8 Iy9 Iy10 Iy11
                            t1 = _mm_srai_epi32(_mm_slli_epi32(Ixy_1, 16), 16); // Ix0 Ix1 Ix2 Ix3

                            fy = _mm_blendv_ps(_mm_set1_ps(0), _mm_cvtepi32_ps(t0), mask_4_7_ps);
                            fx = _mm_blendv_ps(_mm_set1_ps(0), _mm_cvtepi32_ps(t1), mask_4_7_ps);

                            // 4 ... 7
                            I_ps = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(I_0_7_epi16, I_0_7_epi16), 16));



                            mmAyy = _mm_add_ps(mmAyy, _mm_mul_ps(fy, fy));
                            mmAxy = _mm_add_ps(mmAxy, _mm_mul_ps(fx, fy));
                            mmAxx = _mm_add_ps(mmAxx, _mm_mul_ps(fx, fx));

                            // sumIx und sumIy
                            mmSumIx = _mm_add_ps(mmSumIx, fx);
                            mmSumIy = _mm_add_ps(mmSumIy, fy);

                            mmSumW1 = _mm_add_ps(mmSumW1, _mm_mul_ps(I_ps, fx));
                            mmSumW2 = _mm_add_ps(mmSumW2, _mm_mul_ps(I_ps, fy));

                            // sumI
                            I_ps  = _mm_blendv_ps(_mm_set1_ps(0), I_ps, mask_4_7_ps);
                            mmSumI = _mm_add_ps(mmSumI, I_ps);

                            // sumW
                            // sumDI
                            mmSumDI = _mm_add_ps(mmSumDI, _mm_mul_ps( I_ps, I_ps));
                        }

                    }
#else
                    for( ; x < winSize.width*cn; x++, dIptr += 2 )
                    {
                        if( maskPtr[x] == 0)
                            continue;
                        int J_val = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                              Jptr1[x]*iw10 + Jptr1[x+cn]*iw11,
                                              W_BITS1-5);
                        int diff = static_cast<int>(J_val - Iptr[x] + Iptr[x] * gainVec.x + gainVec.y);

                        b1 += (float)(diff*dIptr[0])  * FLT_RESCALE;
                        b2 += (float)(diff*dIptr[1])  * FLT_RESCALE;
                        b3 += (float)(diff) * Iptr[x] * FLT_RESCALE;
                        b4 += (float)(diff);

                        // compute the Gradient Matrice
                        if( j == 0 )
                        {
                            A11 += (float)(dIptr[0]*dIptr[0]);
                            A12 += (float)(dIptr[0]*dIptr[1]);
                            A22 += (float)(dIptr[1]*dIptr[1]);


                            dI += Iptr[x] * Iptr[x] * FLT_RESCALE;
                            float dx = static_cast<float>(dIptr[0]) * FLT_RESCALE;
                            float dy = static_cast<float>(dIptr[1]) * FLT_RESCALE;
                            sumIx += dx;
                            sumIy += dy;
                            w1 += dx * Iptr[x];
                            w2 += dy * Iptr[x];
                            sumI += Iptr[x]  * FLT_RESCALE;
                            //sumW += FLT_RESCALE;

                        }
                    }
#endif
                }
#ifdef RLOF_SSE
                float CV_DECL_ALIGNED(16) wbuf[4];
#endif
                if( j == 0 )
                {
#ifdef RLOF_SSE
                        _mm_store_ps(wbuf, mmSumW1);
                        w1  = wbuf[0] + wbuf[1] + wbuf[2] + wbuf[3];
                        _mm_store_ps(wbuf, mmSumW2);
                        w2  = wbuf[0] + wbuf[1] + wbuf[2] + wbuf[3];
                        _mm_store_ps(wbuf, mmSumDI);
                        dI  = wbuf[0] + wbuf[1] + wbuf[2] + wbuf[3];
                        _mm_store_ps(wbuf, mmSumI);
                        sumI  = wbuf[0] + wbuf[1] + wbuf[2] + wbuf[3];
                        _mm_store_ps(wbuf, mmSumIx);
                        sumIx  = wbuf[0] + wbuf[1] + wbuf[2] + wbuf[3];
                        _mm_store_ps(wbuf, mmSumIy);
                        sumIy  = wbuf[0] + wbuf[1] + wbuf[2] + wbuf[3];
                        _mm_store_ps(wbuf, mmSumW);
                        sumW  = wbuf[0] + wbuf[1] + wbuf[2] + wbuf[3];
#endif
                    sumIx *= -FLT_SCALE;
                    sumIy *= -FLT_SCALE;
                    sumI *=FLT_SCALE;
                    sumW = winArea * FLT_SCALE;
                    w1 *= -FLT_SCALE;
                    w2 *= -FLT_SCALE;
                    dI *= FLT_SCALE;
#ifdef RLOF_SSE
                    float CV_DECL_ALIGNED(32) A11buf[4], A12buf[4], A22buf[4];//

                    _mm_store_ps(A11buf, mmAxx);
                    _mm_store_ps(A12buf, mmAxy);
                    _mm_store_ps(A22buf, mmAyy);

                    A11 = A11buf[0] + A11buf[1] + A11buf[2] + A11buf[3];
                    A12 = A12buf[0] + A12buf[1] + A12buf[2] + A12buf[3];
                    A22 = A22buf[0] + A22buf[1] + A22buf[2] + A22buf[3];
#endif
                    A11 *= FLT_SCALE;
                    A12 *= FLT_SCALE;
                    A22 *= FLT_SCALE;
                }

#ifdef RLOF_SSE
                float CV_DECL_ALIGNED(16) bbuf[4];
                _mm_store_ps(bbuf, _mm_add_ps(qb0, qb1));
                b1 = bbuf[0] + bbuf[2];
                b2 = bbuf[1] + bbuf[3];
                _mm_store_ps(bbuf, qb2);
                b3 = bbuf[0] + bbuf[1] + bbuf[2] + bbuf[3];
                _mm_store_ps(bbuf, qb3);
                b4 = bbuf[0] + bbuf[1] + bbuf[2] + bbuf[3];
#endif

                mismatchMat(0) =  b1 * FLT_SCALE;
                mismatchMat(1) =  b2 * FLT_SCALE;
                mismatchMat(2) = -b3 * FLT_SCALE;
                mismatchMat(3) = -b4 * FLT_SCALE;

                D = - A12*A12*sumI*sumI + dI*sumW*A12*A12 + 2*A12*sumI*sumIx*w2 + 2*A12*sumI*sumIy*w1
                        - 2*dI*A12*sumIx*sumIy - 2*sumW*A12*w1*w2 + A11*A22*sumI*sumI - 2*A22*sumI*sumIx*w1
                        - 2*A11*sumI*sumIy*w2 - sumIx*sumIx*w2*w2 + A22*dI*sumIx*sumIx + 2*sumIx*sumIy*w1*w2
                        - sumIy*sumIy*w1*w1 + A11*dI*sumIy*sumIy + A22*sumW*w1*w1 + A11*sumW*w2*w2 - A11*A22*dI*sumW;


                float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                        4.f*A12*A12))/(2*winArea);
                if(  minEig < minEigThreshold )
                {
                    if( (level == 0 && status) || std::abs(D) < FLT_EPSILON)
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

                resultMat = invTensorMat * mismatchMat;

                // 0.08 -12.10
                Point2f delta(-resultMat(0), -resultMat(1));
                Point2f deltaGain(-resultMat(2), -resultMat(3));


                if( j == 0)
                    prevGain = deltaGain;
                nextPt += delta;
                nextPts[ptidx] = nextPt - halfWin;
                gainVecs[ptidx]= gainVec;
                if( delta.ddot(delta) <= criteria.epsilon)
                    break;
                if ((
                        std::abs(delta.x - prevDelta.x) < 0.01 &&
                        std::abs(delta.y - prevDelta.y) < 0.01
                    ) || (
                        delta.ddot(delta) <= 0.001 &&
                        std::abs(prevGain.x - deltaGain.x) < 0.01
                    ))
                {
                    nextPts[ptidx]  -= delta*0.5f;
                    gainVecs[ptidx] -= deltaGain* 0.5f;
                    break;
                }

                if(j > 0 && std::abs(delta.x - prevDelta.x) < 0.01  &&
                            std::abs(delta.y - prevDelta.y) < 0.01)
                {
                    nextPts[ptidx]  -= delta*0.5f;
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

}  // namespace
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

            if( calcWinMaskMat(BI,  windowType, iprevPt,
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
            const int W_BITS = 14, W_BITS1 = 14;

            int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            float A11 = 0, A12 = 0, A22 = 0;

#ifdef RLOF_SSE
            __m128i qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
            __m128i qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
            __m128i z = _mm_setzero_si128();
            __m128i qdelta_d = _mm_set1_epi32(1 << (W_BITS1-1));
            __m128i qdelta = _mm_set1_epi32(1 << (W_BITS1-5-1));
            __m128 qA11 = _mm_setzero_ps(), qA12 = _mm_setzero_ps(), qA22 = _mm_setzero_ps();
            __m128i mmMask4_epi32;
            get4BitMask(winSize.width, mmMask4_epi32);
#endif

            // extract the patch from the first image, compute covariation matrix of derivatives
            int x, y;
            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* src = I.ptr<uchar>(y + iprevPt.y, 0) + iprevPt.x*cn;
                const uchar* src1 = I.ptr<uchar>(y + iprevPt.y + 1, 0) + iprevPt.x*cn;
                const short* dsrc = derivI.ptr<short>(y + iprevPt.y, 0) + iprevPt.x*cn2;
                const short* dsrc1 = derivI.ptr<short>(y + iprevPt.y + 1, 0) + iprevPt.x*cn2;
                short* Iptr  = IWinBuf.ptr<short>(y, 0);
                short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                x = 0;
#ifdef RLOF_SSE
                for( ; x <= winSize.width*cn; x += 4, dsrc += 4*2, dsrc1 += 8, dIptr += 4*2 )
                {
                    __m128i wMask = _mm_set_epi32(MaskSet * maskPtr[x+3],
                                                  MaskSet * maskPtr[x+2],
                                                  MaskSet * maskPtr[x+1],
                                                  MaskSet * maskPtr[x]);
                    __m128i v00, v01, v10, v11, t0, t1;
                    v00 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x)), z);
                    v01 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src + x + cn)), z);
                    v10 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src1 + x)), z);
                    v11 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src1 + x + cn)), z);

                    t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                       _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));




                    t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1-5);
                    _mm_storel_epi64((__m128i*)(Iptr + x), _mm_packs_epi32(t0,t0));



                    v00 = _mm_loadu_si128((const __m128i*)(dsrc));
                    v01 = _mm_loadu_si128((const __m128i*)(dsrc + cn2));
                    v10 = _mm_loadu_si128((const __m128i*)(dsrc1));
                    v11 = _mm_loadu_si128((const __m128i*)(dsrc1 + cn2));

                    t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                       _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
                    t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                                       _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
                    t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta_d), W_BITS1);
                    t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta_d), W_BITS1);
                    v00 = _mm_packs_epi32(t0, t1); // Ix0 Iy0 Ix1 Iy1 ...

                    if( x + 4 > winSize.width)
                    {
                        v00 = _mm_and_si128(v00, mmMask4_epi32);
                    }
                    v00 = _mm_and_si128(v00, wMask);

                    _mm_storeu_si128((__m128i*)dIptr, v00);
                    t0 = _mm_srai_epi32(v00, 16); // Iy0 Iy1 Iy2 Iy3
                    t1 = _mm_srai_epi32(_mm_slli_epi32(v00, 16), 16); // Ix0 Ix1 Ix2 Ix3

                    __m128 fy = _mm_cvtepi32_ps(t0);
                    __m128 fx = _mm_cvtepi32_ps(t1);


                    qA22 = _mm_add_ps(qA22, _mm_mul_ps(fy, fy));
                    qA12 = _mm_add_ps(qA12, _mm_mul_ps(fx, fy));
                    qA11 = _mm_add_ps(qA11, _mm_mul_ps(fx, fx));
                }
#else

                for( ; x < winSize.width*cn; x++, dsrc += 2, dsrc1 += 2, dIptr += 2 )
                {
                    if( maskPtr[x] == 0)
                    {
                        dIptr[0] = 0;
                        dIptr[1] = 0;
                        continue;
                    }
                    int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                          src1[x]*iw10 + src1[x+cn]*iw11, W_BITS1-5);
                    int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                           dsrc1[0]*iw10 + dsrc1[cn2]*iw11, W_BITS1);
                    int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc1[1]*iw10 +
                                           dsrc1[cn2+1]*iw11, W_BITS1);

                    Iptr[x] = (short)ival;
                    dIptr[0] = (short)ixval;
                    dIptr[1] = (short)iyval;
                    A11 += (float)(ixval*ixval);
                    A12 += (float)(ixval*iyval);
                    A22 += (float)(iyval*iyval);
                }
#endif
            }

#ifdef RLOF_SSE
            float CV_DECL_ALIGNED(16) A11buf[4], A12buf[4], A22buf[4];
            _mm_store_ps(A11buf, qA11);
            _mm_store_ps(A12buf, qA12);
            _mm_store_ps(A22buf, qA22);
            A11 += A11buf[0] + A11buf[1] + A11buf[2] + A11buf[3];
            A12 += A12buf[0] + A12buf[1] + A12buf[2] + A12buf[3];
            A22 += A22buf[0] + A22buf[1] + A22buf[2] + A22buf[3];
#endif

            A11 *= FLT_SCALE;
            A12 *= FLT_SCALE;
            A22 *= FLT_SCALE;

            float D = A11*A22 - A12*A12;
            float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                             4.f*A12*A12))/(2 * winArea);

            if( err )
                err[ptidx] = (float)minEig;

            if( minEig < minEigThreshold ||  D < FLT_EPSILON )
            {
                if( level == 0 && status )
                    status[ptidx] = 0;
                continue;
            }

            D = 1.f/D;

            nextPt += halfWin;
            Point2f prevDelta(0,0);    //relates to h(t-1)
#ifdef RLOF_SSE
            __m128i mmMask0, mmMask1, mmMask;
            getWBitMask(winSize.width, mmMask0, mmMask1, mmMask);
#endif
            for( j = 0; j < criteria.maxCount; j++ )
            {
                status[ptidx] = static_cast<uchar>(j);
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
                iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
                iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
                iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
                float b1 = 0, b2 = 0;
#ifdef RLOF_SSE
                qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
                qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
                __m128 qb0 = _mm_setzero_ps(), qb1 = _mm_setzero_ps();


#endif
                for( y = 0; y < winSize.height; y++ )
                {
                    const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                    const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                    const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                    const short* dIptr = derivIWinBuf.ptr<short>(y, 0);

                    x = 0;
#ifdef RLOF_SSE
                    const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                    for( ; x <= winSize.width*cn; x += 8, dIptr += 8*2 )
                    {
                        if( maskPtr[x  ] == 0 && maskPtr[x+1] == 0 && maskPtr[x+2] == 0 && maskPtr[x+3] == 0
                        &&    maskPtr[x+4] == 0 && maskPtr[x+5] == 0 && maskPtr[x+6] == 0 && maskPtr[x+7] == 0)
                            continue;
                        __m128i diff0 = _mm_loadu_si128((const __m128i*)(Iptr + x)), diff1;
                        __m128i v00 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x)), z);
                        __m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + cn)), z);
                        __m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr1 + x)), z);
                        __m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr1 + x + cn)), z);

                        __m128i t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                                   _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
                        __m128i t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                                                   _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));
                        t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1-5);
                        t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta), W_BITS1-5);

                        __m128i mmDiff_epi16 = _mm_subs_epi16(_mm_packs_epi32(t0, t1), diff0);

                        __m128i Ixy_0 = _mm_loadu_si128((const __m128i*)(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                        __m128i Ixy_1 = _mm_loadu_si128((const __m128i*)(dIptr + 8));

                        if(  x > winSize.width*cn - 8)
                        {
                            Ixy_0 = _mm_and_si128(Ixy_0, mmMask0);
                            Ixy_1 = _mm_and_si128(Ixy_1, mmMask1);
                        }


                        diff1 = _mm_unpackhi_epi16(mmDiff_epi16, mmDiff_epi16); // It4 It4 It5 It5 It6 It6 It7 It7   | It12 It12 It13 It13...
                        diff0 = _mm_unpacklo_epi16(mmDiff_epi16, mmDiff_epi16); // It0 It0 It1 It1 It2 It2 It3 It3   | It8 It8 It9 It9...


                        v10 = _mm_mullo_epi16(Ixy_0, diff0);
                        v11 = _mm_mulhi_epi16(Ixy_0, diff0);
                        v00 = _mm_unpacklo_epi16(v10, v11);
                        v10 = _mm_unpackhi_epi16(v10, v11);
                        qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
                        qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));

                        // It * Ix It * Iy [4 ... 7]
                        // for set 1 hi sigma 1

                        v10 = _mm_mullo_epi16(Ixy_1, diff1);
                        v11 = _mm_mulhi_epi16(Ixy_1, diff1);
                        v00 = _mm_unpacklo_epi16(v10, v11);
                        v10 = _mm_unpackhi_epi16(v10, v11);
                        qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
                        qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
                    }
#else
                    for( ; x < winSize.width*cn; x++, dIptr += 2 )
                    {
                        if( dIptr[0] == 0 && dIptr[1] == 0)
                            continue;
                        int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                              Jptr1[x]*iw10 + Jptr1[x+cn]*iw11,
                                              W_BITS1-5) - Iptr[x];

                        b1 += (float)(diff*dIptr[0]) * FLT_RESCALE;
                        b2 += (float)(diff*dIptr[1]) * FLT_RESCALE;
                    }
#endif
                }

#ifdef RLOF_SSE
                float CV_DECL_ALIGNED(16) bbuf[4];
                _mm_store_ps(bbuf, _mm_add_ps(qb0, qb1));
                b1 += bbuf[0] + bbuf[2];
                b2 += bbuf[1] + bbuf[3];
#endif

                b1 *= FLT_SCALE;
                b2 *= FLT_SCALE;

                Point2f delta( (float)((A12*b2 - A22*b1) * D),
                              (float)((A12*b1 - A11*b2) * D));



                nextPt += delta;
                nextPts[ptidx] = nextPt - halfWin;

                if( delta.ddot(delta) <= criteria.epsilon)
                    break;
                if(j > 0 && std::abs(delta.x - prevDelta.x) < 0.01  &&
                            std::abs(delta.y - prevDelta.y) < 0.01)
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
    int                 crossSegmentationThreshold;
};

}}}}  // namespace
#endif
