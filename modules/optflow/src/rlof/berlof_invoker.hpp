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
            const int W_BITS = 14, W_BITS1 = 14;

            int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            float A11 = 0, A12 = 0, A22 = 0;
            float D = 0;

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
                x = 0;
                for( ; x < winSize.width*cn; x++, dsrc += 2, dsrc1 += 2, dIptr += 2 )
                {
                    if( winMaskMat.at<uchar>(y,x) == 0)
                    {
                        dIptr[0] = 0;
                        dIptr[1] = 0;
                        continue;
                    }
                    int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                          src1[x]*iw10 + src1[x+cn]*iw11, W_BITS1-5);
                    int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                           dsrc1[0]*iw10 + dsrc1[cn2]*iw11, W_BITS1);
                    int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 +
                                           dsrc1[1]*iw10 + dsrc1[cn2+1]*iw11, W_BITS1);
                    Iptr[x] = (short)ival;
                    dIptr[0] = (short)ixval;
                    dIptr[1] = (short)iyval;
                }
            }

            cv::Mat residualMat = cv::Mat::zeros(winSize.height * (winSize.width + 8) * cn, 1, CV_16SC1);
            cv::Point2f backUpNextPt = nextPt;
            nextPt += halfWin;
            Point2f prevDelta(0,0);    //denotes h(t-1)
            cv::Size _winSize = winSize;
#ifdef RLOF_SSE
            __m128i mmMask0, mmMask1, mmMask;
            getWBitMask(_winSize.width, mmMask0, mmMask1, mmMask);
#endif
            float MEstimatorScale = 1;
            int buffIdx = 0;
            float c[8];
            cv::Mat GMc0, GMc1, GMc2, GMc3;
            cv::Vec2f Mc0, Mc1, Mc2, Mc3;
            int noIteration = 0;
            int noReusedIteration = 0;
            int noSolvedIteration = 0;
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
                        for( y = 0; y < winSize.height; y++ )
                        {
                            const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                            const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                            const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                            const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                            const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                            x = 0;
                            for( ; x < winSize.width*cn; x++, dIptr += 2)
                            {
                                if( maskPtr[x] == 0)
                                    continue;
                                int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 + Jptr1[x]*iw10 + Jptr1[x+cn]*iw11, W_BITS1-5)
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

                    /*
                    */
                    for( y = 0; y < _winSize.height; y++ )
                    {
                        const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                        const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                        const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                        const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                        const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                        x = 0;
                        for( ; x < _winSize.width*cn; x++, dIptr += 2 )
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
                                                  W_BITS1-5);


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
                    }

                    if( j == 0 )
                    {

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
                            noIteration++;
                            break;
                        }

                        D = (1.f / D);

                    }

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
                    if( hasSolved == false)
                        noIteration++;
                }
                else
                {
                    hasSolved = false;
                    noReusedIteration++;
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
                    noSolvedIteration++;
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
                x = 0;
                const uchar* src  = I.ptr<uchar>(y + iprevPt.y, 0) + iprevPt.x*cn;
                const uchar* src1 = I.ptr<uchar>(y + iprevPt.y + 1, 0) + iprevPt.x*cn;
                const short* dsrc  = derivI.ptr<short>(y + iprevPt.y, 0) + iprevPt.x*cn2;
                const short* dsrc1 = derivI.ptr<short>(y + iprevPt.y + 1, 0) + iprevPt.x*cn2;
                short* Iptr  = IWinBuf.ptr<short>(y, 0);
                short* dIptr = derivIWinBuf.ptr<short>(y, 0);
#ifdef RLOF_SSE
                const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
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
                    if( winMaskMat.at<uchar>(y,x) == 0)
                    {
                        dIptr[0] = 0;
                        dIptr[1] = 0;
                        continue;
                    }
                    int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                          src1[x]*iw10 + src1[x+cn]*iw11, W_BITS1-5);
                    int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                           dsrc1[0]*iw10 + dsrc1[cn2]*iw11, W_BITS1);
                    int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 +
                                            dsrc1[1]*iw10 + dsrc1[cn2+1]*iw11, W_BITS1);

                    Iptr[x] = (short)ival;
                    dIptr[0] = (short)ixval;
                    dIptr[1] = (short)iyval;

                }
#endif
            }

            cv::Mat residualMat = cv::Mat::zeros(winSize.height * (winSize.width + 8) * cn, 1, CV_16SC1);
            cv::Point2f backUpNextPt = nextPt;
                    nextPt += halfWin;
            Point2f prevDelta(0,0);    //relates to h(t-1)
            Point2f prevGain(1,0);
            cv::Point2f gainVec = gainVecs[ptidx];
            cv::Point2f backUpGain = gainVec;
            cv::Size _winSize = winSize;
            int j;
#ifdef RLOF_SSE
            __m128i mmMask0, mmMask1, mmMask;
            getWBitMask(_winSize.width, mmMask0, mmMask1, mmMask);
            __m128  mmOnes   = _mm_set1_ps(1.f );
#endif
            float MEstimatorScale = 1;
            int buffIdx = 0;
            float c[8];
            cv::Mat GMc0, GMc1, GMc2, GMc3;
            cv::Vec4f Mc0, Mc1, Mc2, Mc3;
            int noIteration = 0;
            int noReusedIteration = 0;
            int noSolvedIteration = 0;
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
                        noIteration++;
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
                        for( y = 0; y < winSize.height; y++ )
                        {
                            const uchar* Jptr = J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                            const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                            const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                            const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                            const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                            x = 0;
                            for( ; x < winSize.width*cn; x++, dIptr += 2)
                            {
                                if( maskPtr[x] == 0)
                                    continue;
                                int diff = static_cast<int>(CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 + Jptr1[x]*iw10 + Jptr1[x+cn]*iw11, W_BITS1-5)
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

    #ifdef RLOF_SSE

                    qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
                    qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
                    __m128 qb0[4] = {_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps()};
                    __m128 qb1[4] = {_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps()};
                    __m128 qb2[4] = {_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps()};
                    __m128 qb3[4] = {_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps()};
                    __m128 mmSumW1 = _mm_setzero_ps(), mmSumW2 = _mm_setzero_ps();
                    __m128 mmSumI = _mm_setzero_ps(), mmSumW = _mm_setzero_ps(), mmSumDI = _mm_setzero_ps();
                    __m128 mmSumIy = _mm_setzero_ps(),  mmSumIx = _mm_setzero_ps();
                    __m128 mmAxx = _mm_setzero_ps(), mmAxy = _mm_setzero_ps(), mmAyy = _mm_setzero_ps();
                    __m128i mmParam0 = _mm_set1_epi16(MIN(std::numeric_limits<short>::max() -1, static_cast<short>(fParam0)));
                    __m128i mmParam1 = _mm_set1_epi16(MIN(std::numeric_limits<short>::max()- 1, static_cast<short>(fParam1)));


                    float s2Val = std::fabs(normSigma2);
                    int s2bitShift = normSigma2 == 0 ? 1 : cvCeil(log(200.f / s2Val) / log(2.f));
                    __m128i mmParam2_epi16 = _mm_set1_epi16(static_cast<short>(normSigma2 * (float)(1 << s2bitShift)));
                    __m128i mmOness_epi16 = _mm_set1_epi16(1 << s2bitShift);
                    __m128  mmParam2s = _mm_set1_ps(0.01f * normSigma2);
                    __m128  mmParam2s2 = _mm_set1_ps(normSigma2 * normSigma2);
                    float gainVal = gainVec.x > 0 ? gainVec.x : -gainVec.x;
                    int bitShift = gainVec.x == 0 ? 1 : cvCeil(log(200.f / gainVal) / log(2.f));
                    __m128i mmGainValue_epi16 = _mm_set1_epi16(static_cast<short>(gainVec.x * (float)(1 << bitShift)));
                    __m128i mmConstValue_epi16 = _mm_set1_epi16(static_cast<short>(gainVec.y));
                    __m128i mmEta     = _mm_setzero_si128();
                    __m128i mmScale      = _mm_set1_epi16(static_cast<short>(MEstimatorScale));

    #endif

                    buffIdx = 0;
                    float _b0[4] = {0,0,0,0};
                    float _b1[4] = {0,0,0,0};
                    float _b2[4] = {0,0,0,0};
                    float _b3[4] = {0,0,0,0};
                    /*
                    */
                    for( y = 0; y < _winSize.height; y++ )
                    {
                        const uchar* Jptr =  J.ptr<uchar>(y + inextPt.y, inextPt.x*cn);
                        const uchar* Jptr1 = J.ptr<uchar>(y + inextPt.y + 1, inextPt.x*cn);
                        const short* Iptr  = IWinBuf.ptr<short>(y, 0);
                        const short* dIptr = derivIWinBuf.ptr<short>(y, 0);
                        const tMaskType* maskPtr = winMaskMat.ptr<tMaskType>(y, 0);
                        x = 0;
    #ifdef RLOF_SSE
                        for( ; x <= _winSize.width*cn; x += 8, dIptr += 8*2 )
                        {
                            __m128i mask_0_7_epi16 = _mm_mullo_epi16(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)(maskPtr+x))), mmMaskSet_epi16);
                            __m128i I_0_7_epi16 = _mm_loadu_si128((const __m128i*)(Iptr + x));

                            __m128i v00 = _mm_unpacklo_epi8(
                                _mm_loadl_epi64((const __m128i*)(Jptr + x)), z);
                            __m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + cn)), z);
                            __m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr1 + x)), z);
                            __m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr1 + x + cn)), z);

                            __m128i t0 = _mm_add_epi32
                                (_mm_madd_epi16(
                                    _mm_unpacklo_epi16(v00, v01),
                                    qw0),
                                _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
                            __m128i t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                                                       _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));

                            t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS1-5);
                            t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta), W_BITS1-5);

                            __m128i lo = _mm_mullo_epi16(mmGainValue_epi16, I_0_7_epi16);
                            __m128i hi = _mm_mulhi_epi16(mmGainValue_epi16, I_0_7_epi16);

                            __m128i Igain_0_3_epi32 = _mm_srai_epi32(_mm_unpacklo_epi16(lo, hi), bitShift);
                            __m128i Igain_4_7_epi32 = _mm_srai_epi32(_mm_unpackhi_epi16(lo, hi), bitShift);
                            __m128i Igain_epi16 =  _mm_packs_epi32(Igain_0_3_epi32, Igain_4_7_epi32);


                            __m128i diffValue = _mm_subs_epi16(_mm_add_epi16(Igain_epi16, mmConstValue_epi16), I_0_7_epi16);
                            __m128i mmDiffc_epi16[4] =
                            {
                               _mm_add_epi16(_mm_slli_epi16(v11, 5), diffValue),
                               _mm_add_epi16(_mm_slli_epi16(v01, 5), diffValue),
                               _mm_add_epi16(_mm_slli_epi16(v10, 5), diffValue),
                               _mm_add_epi16(_mm_slli_epi16(v00, 5), diffValue)
                            };

                            __m128i mmDiff_epi16 = _mm_add_epi16( _mm_packs_epi32(t0, t1), diffValue);


                            mmDiff_epi16 = _mm_and_si128(mmDiff_epi16, mask_0_7_epi16);

                            __m128i scalediffIsPos_epi16    = _mm_cmpgt_epi16(mmDiff_epi16, mmScale);
                            mmEta = _mm_add_epi16(mmEta, _mm_add_epi16(_mm_and_si128(scalediffIsPos_epi16, _mm_set1_epi16(2)), _mm_set1_epi16(-1)));


                            __m128i Ixy_0 = _mm_loadu_si128((const __m128i*)(dIptr));
                            __m128i Ixy_1 = _mm_loadu_si128((const __m128i*)(dIptr + 8));


                            __m128i abs_epi16 = _mm_abs_epi16(mmDiff_epi16);
                            __m128i bSet2_epi16, bSet1_epi16;
                            // |It| < sigma1 ?
                            bSet2_epi16        = _mm_cmplt_epi16(abs_epi16, mmParam1);
                            // It > 0 ?
                            __m128i diffIsPos_epi16    = _mm_cmpgt_epi16(mmDiff_epi16, _mm_setzero_si128());
                            // sigma0 < |It| < sigma1 ?
                            bSet1_epi16        = _mm_and_si128(bSet2_epi16, _mm_cmpgt_epi16(abs_epi16, mmParam0));
                                                        // val = |It| -/+ sigma1
                            __m128i tmpParam1_epi16 = _mm_add_epi16(_mm_and_si128(diffIsPos_epi16, _mm_sub_epi16(mmDiff_epi16, mmParam1)),
                                                                 _mm_andnot_si128(diffIsPos_epi16, _mm_add_epi16(mmDiff_epi16, mmParam1)));
                            // It == 0     ? |It| > sigma13
                            mmDiff_epi16 = _mm_and_si128(bSet2_epi16, mmDiff_epi16);

                            for( unsigned int mmi = 0; mmi < 4; mmi++)
                            {
                                __m128i mmDiffc_epi16_t = _mm_and_si128(mmDiffc_epi16[mmi], mask_0_7_epi16);
                                mmDiffc_epi16_t = _mm_and_si128(bSet2_epi16, mmDiffc_epi16_t);

                                // It == val ? sigma0 < |It| < sigma1
                                mmDiffc_epi16_t = _mm_blendv_epi8(mmDiffc_epi16_t, tmpParam1_epi16, bSet1_epi16);
                                __m128i tale_epi16_ = _mm_blendv_epi8(mmOness_epi16, mmParam2_epi16, bSet1_epi16); // mask for 0 - 3
                                // diff = diff * sigma2
                                lo = _mm_mullo_epi16(tale_epi16_, mmDiffc_epi16_t);
                                hi = _mm_mulhi_epi16(tale_epi16_, mmDiffc_epi16_t);
                                __m128i diff_0_3_epi32 = _mm_srai_epi32(_mm_unpacklo_epi16(lo, hi), s2bitShift);
                                __m128i diff_4_7_epi32 = _mm_srai_epi32(_mm_unpackhi_epi16(lo, hi), s2bitShift);

                                mmDiffc_epi16_t = _mm_packs_epi32(diff_0_3_epi32, diff_4_7_epi32);
                                __m128i diff1 = _mm_unpackhi_epi16(mmDiffc_epi16_t, mmDiffc_epi16_t); // It4 It4 It5 It5 It6 It6 It7 It7   | It12 It12 It13 It13...
                                __m128i diff0 = _mm_unpacklo_epi16(mmDiffc_epi16_t, mmDiffc_epi16_t); // It0 It0 It1 It1 It2 It2 It3 It3   | It8 It8 It9 It9...

                                // Ix * diff / Iy * diff
                                v10 = _mm_mullo_epi16(Ixy_0, diff0);
                                v11 = _mm_mulhi_epi16(Ixy_0, diff0);
                                v00 = _mm_unpacklo_epi16(v10, v11);
                                v10 = _mm_unpackhi_epi16(v10, v11);

                                qb0[mmi] = _mm_add_ps(qb0[mmi], _mm_cvtepi32_ps(v00));
                                qb1[mmi] = _mm_add_ps(qb1[mmi], _mm_cvtepi32_ps(v10));
                                // It * Ix It * Iy [4 ... 7]
                                // for set 1 hi sigma 1
                                v10 = _mm_mullo_epi16(Ixy_1, diff1);
                                v11 = _mm_mulhi_epi16(Ixy_1, diff1);
                                v00 = _mm_unpacklo_epi16(v10, v11);
                                v10 = _mm_unpackhi_epi16(v10, v11);
                                qb0[mmi] = _mm_add_ps(qb0[mmi], _mm_cvtepi32_ps(v00));
                                qb1[mmi] = _mm_add_ps(qb1[mmi], _mm_cvtepi32_ps(v10));
                                // diff * J [0 ... 7]
                                // for set 1  sigma 1
                                // b3 += diff * Iptr[x]
                                v10 = _mm_mullo_epi16(mmDiffc_epi16_t, I_0_7_epi16);
                                v11 = _mm_mulhi_epi16(mmDiffc_epi16_t, I_0_7_epi16);
                                v00 = _mm_unpacklo_epi16(v10, v11);

                                v10 = _mm_unpackhi_epi16(v10, v11);
                                qb2[mmi] = _mm_add_ps(qb2[mmi], _mm_cvtepi32_ps(v00));
                                qb2[mmi] = _mm_add_ps(qb2[mmi], _mm_cvtepi32_ps(v10));
                                qb3[mmi] = _mm_add_ps(qb3[mmi], _mm_cvtepi32_ps(diff_0_3_epi32));
                                qb3[mmi] = _mm_add_ps(qb3[mmi], _mm_cvtepi32_ps(diff_4_7_epi32));
                            }

                            if( j == 0 )
                            {
                                __m128 bSet1_0_3_ps = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(bSet1_epi16));
                                __m128 bSet1_4_7_ps = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(bSet1_epi16,bSet1_epi16), 16));
                                __m128 mask_0_4_ps = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(mask_0_7_epi16));
                                __m128 mask_4_7_ps = _mm_cvtepi32_ps((_mm_srai_epi32(_mm_unpackhi_epi16(mask_0_7_epi16, mask_0_7_epi16),16)));

                                __m128 bSet2_0_3_ps = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(bSet2_epi16));
                                __m128 bSet2_4_7_ps = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(bSet2_epi16, bSet2_epi16),16));

                                __m128 tale_0_3_ps = _mm_blendv_ps(mmOnes, mmParam2s2, bSet1_0_3_ps);
                                __m128 tale_4_7_ps = _mm_blendv_ps(mmOnes, mmParam2s2, bSet1_4_7_ps);
                                tale_0_3_ps = _mm_blendv_ps(mmParam2s, tale_0_3_ps, bSet2_0_3_ps);
                                tale_4_7_ps = _mm_blendv_ps(mmParam2s, tale_4_7_ps, bSet2_4_7_ps);

                                tale_0_3_ps = _mm_blendv_ps(_mm_set1_ps(0), tale_0_3_ps, mask_0_4_ps);
                                tale_4_7_ps = _mm_blendv_ps(_mm_set1_ps(0), tale_4_7_ps, mask_4_7_ps);

                                t0 = _mm_srai_epi32(Ixy_0, 16); // Iy0 Iy1 Iy2 Iy3
                                t1 = _mm_srai_epi32(_mm_slli_epi32(Ixy_0, 16), 16); // Ix0 Ix1 Ix2 Ix3

                                __m128 fy = _mm_cvtepi32_ps(t0);
                                __m128 fx = _mm_cvtepi32_ps(t1);

                                // 0 ... 3
                                __m128 I_ps = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(I_0_7_epi16, I_0_7_epi16), 16));

                                // A11 - A22
                                __m128 fxtale = _mm_mul_ps(fx, tale_0_3_ps);
                                __m128 fytale = _mm_mul_ps(fy, tale_0_3_ps);

                                mmAyy = _mm_add_ps(mmAyy, _mm_mul_ps(fy, fytale));
                                mmAxy = _mm_add_ps(mmAxy, _mm_mul_ps(fx, fytale));
                                mmAxx = _mm_add_ps(mmAxx, _mm_mul_ps(fx, fxtale));

                                // sumIx und sumIy
                                mmSumIx = _mm_add_ps(mmSumIx, fxtale);
                                mmSumIy = _mm_add_ps(mmSumIy, fytale);

                                mmSumW1 = _mm_add_ps(mmSumW1, _mm_mul_ps(I_ps, fxtale));
                                mmSumW2 = _mm_add_ps(mmSumW2, _mm_mul_ps(I_ps, fytale));

                                // sumI
                                __m128 I_tale_ps = _mm_mul_ps(I_ps, tale_0_3_ps);
                                mmSumI = _mm_add_ps(mmSumI,I_tale_ps);

                                // sumW
                                mmSumW = _mm_add_ps(mmSumW, tale_0_3_ps);

                                // sumDI
                                mmSumDI = _mm_add_ps(mmSumDI, _mm_mul_ps( I_ps, I_tale_ps));


                                t0 = _mm_srai_epi32(Ixy_1, 16); // Iy8 Iy9 Iy10 Iy11
                                t1 = _mm_srai_epi32(_mm_slli_epi32(Ixy_1, 16), 16); // Ix0 Ix1 Ix2 Ix3

                                fy =  _mm_cvtepi32_ps(t0);
                                fx =  _mm_cvtepi32_ps(t1);

                                // 4 ... 7
                                I_ps = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(I_0_7_epi16, I_0_7_epi16), 16));

                                // A11 - A22
                                fxtale = _mm_mul_ps(fx, tale_4_7_ps);
                                fytale = _mm_mul_ps(fy, tale_4_7_ps);

                                mmAyy = _mm_add_ps(mmAyy, _mm_mul_ps(fy, fytale));
                                mmAxy = _mm_add_ps(mmAxy, _mm_mul_ps(fx, fytale));
                                mmAxx = _mm_add_ps(mmAxx, _mm_mul_ps(fx, fxtale));

                                // sumIx und sumIy
                                mmSumIx = _mm_add_ps(mmSumIx, fxtale);
                                mmSumIy = _mm_add_ps(mmSumIy, fytale);

                                mmSumW1 = _mm_add_ps(mmSumW1, _mm_mul_ps(I_ps, fxtale));
                                mmSumW2 = _mm_add_ps(mmSumW2, _mm_mul_ps(I_ps, fytale));

                                // sumI
                                I_tale_ps = _mm_mul_ps(I_ps, tale_4_7_ps);
                                mmSumI = _mm_add_ps(mmSumI, I_tale_ps);

                                // sumW
                                mmSumW = _mm_add_ps(mmSumW, tale_4_7_ps);

                                // sumDI
                                mmSumDI = _mm_add_ps(mmSumDI, _mm_mul_ps( I_ps, I_tale_ps));
                            }

                        }
    #else
                        for( ; x < _winSize.width*cn; x++, dIptr += 2 )
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
                                                  W_BITS1-5);

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

    #ifdef RLOF_SSE
                    short etaValues[8];
                    _mm_storeu_si128((__m128i*)(etaValues), mmEta);
                    MEstimatorScale += eta * (etaValues[0] + etaValues[1] + etaValues[2] + etaValues[3]
                                               + etaValues[4] + etaValues[5] + etaValues[6] + etaValues[7]);
                    float CV_DECL_ALIGNED(32) wbuf[4];
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
                            sumW *= FLT_SCALE;
                            w1 *= -FLT_SCALE;
                            w2 *= -FLT_SCALE;
                            dI *= FLT_SCALE;


    #ifdef RLOF_SSE
                        float CV_DECL_ALIGNED(16) A11buf[4], A12buf[4], A22buf[4];//

                        _mm_store_ps(A11buf, mmAxx);
                        _mm_store_ps(A12buf, mmAxy);
                        _mm_store_ps(A22buf, mmAyy);


                        A11 = A11buf[0] + A11buf[1] + A11buf[2] + A11buf[3];
                        A12 = A12buf[0] + A12buf[1] + A12buf[2] + A12buf[3];
                        A22 = A22buf[0] + A22buf[1] + A22buf[2] + A22buf[3];
    #endif
                        A11 *= FLT_SCALE; // 54866744.
                        A12 *= FLT_SCALE; // -628764.00
                        A22 *= FLT_SCALE; // 19730.000

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
                            noIteration++;
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


    #ifdef RLOF_SSE
                    float CV_DECL_ALIGNED(16) bbuf[4];
                    for(int mmi = 0; mmi < 4; mmi++)
                    {

                        _mm_store_ps(bbuf, _mm_add_ps(qb0[mmi], qb1[mmi]));
                        _b0[mmi] = bbuf[0] + bbuf[2];
                        _b1[mmi] = bbuf[1] + bbuf[3];
                        _mm_store_ps(bbuf, qb2[mmi]);
                        _b2[mmi] = bbuf[0] + bbuf[1] + bbuf[2] + bbuf[3];
                        _mm_store_ps(bbuf, qb3[mmi]);
                        _b3[mmi] = bbuf[0] + bbuf[1] + bbuf[2] + bbuf[3];

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
                    if( hasSolved == false)
                        noIteration++;
                }
                else
                {
                    hasSolved = false;
                    noReusedIteration++;
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
                    noSolvedIteration++;
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
                for( ; x < winSize.width*cn; x += 4, dsrc += 4*2, dsrc1 += 8,dIptr += 4*2 )
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

                for( ; x < winSize.width*cn; x++, dsrc += 2, dsrc1 += 2, dIptr += 2)
                {
                    if( maskPtr[x] == 0)
                    {
                        dIptr[0] = (short)0;
                        dIptr[1] = (short)0;

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

            float c[8];
#ifdef RLOF_SSE
            __m128i mmMask0, mmMask1, mmMask;
            getWBitMask(winSize.width, mmMask0, mmMask1, mmMask);
#endif
            for( j = 0; j < criteria.maxCount; j++ )
            {
                cv::Point2f delta;
                bool hasSolved = false;
                a = nextPt.x - cvFloor(nextPt.x);
                b = nextPt.y - cvFloor(nextPt.y);
                float ab = a * b;


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
    #ifdef RLOF_SSE
                    __m128 qbc0[4] = {_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps()};
                    __m128 qbc1[4] = {_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps(),_mm_setzero_ps()};
                    qw0 = _mm_set1_epi32(iw00 + (iw01 << 16));
                    qw1 = _mm_set1_epi32(iw10 + (iw11 << 16));
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
                            __m128i diff0 = _mm_loadu_si128((const __m128i*)(Iptr + x));
                            __m128i v00 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x)), z);
                            __m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr + x + cn)), z);
                            __m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr1 + x)), z);
                            __m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)(Jptr1 + x + cn)), z);

                            __m128i mmDiffc_epi16[4] =
                            { _mm_subs_epi16(_mm_slli_epi16(v00, 5), diff0),
                              _mm_subs_epi16(_mm_slli_epi16(v01, 5), diff0),
                              _mm_subs_epi16(_mm_slli_epi16(v10, 5), diff0),
                              _mm_subs_epi16(_mm_slli_epi16(v11, 5), diff0)
                            };

                            __m128i Ixy_0 = _mm_loadu_si128((const __m128i*)(dIptr)); // Ix0 Iy0 Ix1 Iy1 ...
                            __m128i Ixy_1 = _mm_loadu_si128((const __m128i*)(dIptr + 8));

                            if(  x > winSize.width*cn - 8)
                            {
                                Ixy_0 = _mm_and_si128(Ixy_0, mmMask0);
                                Ixy_1 = _mm_and_si128(Ixy_1, mmMask1);
                            }

                            __m128i diffc1[4] =
                            {_mm_unpackhi_epi16(mmDiffc_epi16[0],mmDiffc_epi16[0]),
                             _mm_unpackhi_epi16(mmDiffc_epi16[1],mmDiffc_epi16[1]),
                             _mm_unpackhi_epi16(mmDiffc_epi16[2],mmDiffc_epi16[2]),
                             _mm_unpackhi_epi16(mmDiffc_epi16[3],mmDiffc_epi16[3])
                            };

                            __m128i diffc0[4] =
                            {_mm_unpacklo_epi16(mmDiffc_epi16[0],mmDiffc_epi16[0]),
                             _mm_unpacklo_epi16(mmDiffc_epi16[1],mmDiffc_epi16[1]),
                             _mm_unpacklo_epi16(mmDiffc_epi16[2],mmDiffc_epi16[2]),
                             _mm_unpacklo_epi16(mmDiffc_epi16[3],mmDiffc_epi16[3])
                            };


                            // It * Ix It * Iy [0 ... 3]
                            //mask split for multiplication
                            // for set 1 lo sigma 1


                            v10 = _mm_mullo_epi16(Ixy_0, diffc0[0]);
                            v11 = _mm_mulhi_epi16(Ixy_0, diffc0[0]);
                            v00 = _mm_unpacklo_epi16(v10, v11);
                            v10 = _mm_unpackhi_epi16(v10, v11);
                            qbc0[0] = _mm_add_ps(qbc0[0], _mm_cvtepi32_ps(v00));
                            qbc1[0] = _mm_add_ps(qbc1[0], _mm_cvtepi32_ps(v10));

                            v10 = _mm_mullo_epi16(Ixy_0, diffc0[1]);
                            v11 = _mm_mulhi_epi16(Ixy_0, diffc0[1]);
                            v00 = _mm_unpacklo_epi16(v10, v11);
                            v10 = _mm_unpackhi_epi16(v10, v11);
                            qbc0[1] = _mm_add_ps(qbc0[1], _mm_cvtepi32_ps(v00));
                            qbc1[1] = _mm_add_ps(qbc1[1], _mm_cvtepi32_ps(v10));

                            v10 = _mm_mullo_epi16(Ixy_0, diffc0[2]);
                            v11 = _mm_mulhi_epi16(Ixy_0, diffc0[2]);
                            v00 = _mm_unpacklo_epi16(v10, v11);
                            v10 = _mm_unpackhi_epi16(v10, v11);
                            qbc0[2] = _mm_add_ps(qbc0[2], _mm_cvtepi32_ps(v00));
                            qbc1[2] = _mm_add_ps(qbc1[2], _mm_cvtepi32_ps(v10));

                            v10 = _mm_mullo_epi16(Ixy_0, diffc0[3]);
                            v11 = _mm_mulhi_epi16(Ixy_0, diffc0[3]);
                            v00 = _mm_unpacklo_epi16(v10, v11);
                            v10 = _mm_unpackhi_epi16(v10, v11);
                            qbc0[3] = _mm_add_ps(qbc0[3], _mm_cvtepi32_ps(v00));
                            qbc1[3] = _mm_add_ps(qbc1[3], _mm_cvtepi32_ps(v10));
                            // It * Ix It * Iy [4 ... 7]
                            // for set 1 hi sigma 1

                            v10 = _mm_mullo_epi16(Ixy_1, diffc1[0]);
                            v11 = _mm_mulhi_epi16(Ixy_1, diffc1[0]);
                            v00 = _mm_unpacklo_epi16(v10, v11);
                            v10 = _mm_unpackhi_epi16(v10, v11);
                            qbc0[0] = _mm_add_ps(qbc0[0], _mm_cvtepi32_ps(v00));
                            qbc1[0] = _mm_add_ps(qbc1[0], _mm_cvtepi32_ps(v10));

                            v10 = _mm_mullo_epi16(Ixy_1, diffc1[1]);
                            v11 = _mm_mulhi_epi16(Ixy_1, diffc1[1]);
                            v00 = _mm_unpacklo_epi16(v10, v11);
                            v10 = _mm_unpackhi_epi16(v10, v11);
                            qbc0[1] = _mm_add_ps(qbc0[1], _mm_cvtepi32_ps(v00));
                            qbc1[1] = _mm_add_ps(qbc1[1], _mm_cvtepi32_ps(v10));

                            v10 = _mm_mullo_epi16(Ixy_1, diffc1[2]);
                            v11 = _mm_mulhi_epi16(Ixy_1, diffc1[2]);
                            v00 = _mm_unpacklo_epi16(v10, v11);
                            v10 = _mm_unpackhi_epi16(v10, v11);
                            qbc0[2] = _mm_add_ps(qbc0[2], _mm_cvtepi32_ps(v00));
                            qbc1[2] = _mm_add_ps(qbc1[2], _mm_cvtepi32_ps(v10));

                            v10 = _mm_mullo_epi16(Ixy_1, diffc1[3]);
                            v11 = _mm_mulhi_epi16(Ixy_1, diffc1[3]);
                            v00 = _mm_unpacklo_epi16(v10, v11);
                            v10 = _mm_unpackhi_epi16(v10, v11);
                            qbc0[3] = _mm_add_ps(qbc0[3], _mm_cvtepi32_ps(v00));
                            qbc1[3] = _mm_add_ps(qbc1[3], _mm_cvtepi32_ps(v10));

                         }
    #else
                        for( ; x < winSize.width*cn; x++, dIptr += 2 )
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

    #ifdef RLOF_SSE
                    float CV_DECL_ALIGNED(16) bbuf[4];
                    _mm_store_ps(bbuf, _mm_add_ps(qbc0[0], qbc1[0]));
                    _b1[0] += bbuf[0] + bbuf[2];
                    _b2[0] += bbuf[1] + bbuf[3];

                    _mm_store_ps(bbuf, _mm_add_ps(qbc0[1], qbc1[1]));
                    _b1[1] += bbuf[0] + bbuf[2];
                    _b2[1] += bbuf[1] + bbuf[3];

                    _mm_store_ps(bbuf, _mm_add_ps(qbc0[2], qbc1[2]));
                    _b1[2] += bbuf[0] + bbuf[2];
                    _b2[2] += bbuf[1] + bbuf[3];

                    _mm_store_ps(bbuf, _mm_add_ps(qbc0[3], qbc1[3]));
                    _b1[3] += bbuf[0] + bbuf[2];
                    _b2[3] += bbuf[1] + bbuf[3];

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

}}}}  // namespace
#endif
