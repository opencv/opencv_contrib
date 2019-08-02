// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef _RLOF_INVOKERBASE_HPP_
#define _RLOF_INVOKERBASE_HPP_


#if CV_SSE4_1
#define RLOF_SSE
#endif

//#define DEBUG_INVOKER

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
#ifdef RLOF_SSE
static inline void get4BitMask(const int & width, __m128i & mask)
{
    int noBits = width - static_cast<int>(floor(width / 4.f) * 4.f);
    unsigned int val[4];
    for (int n = 0; n < 4; n++)
    {
        val[n] = (noBits > n) ? (std::numeric_limits<unsigned int>::max()) : 0;
    }
    mask = _mm_set_epi32(val[3], val[2], val[1], val[0]);

}
static inline void getWBitMask(const int & width, __m128i & t0, __m128i & t1, __m128i & t2)
{
    int noBits = width - static_cast<int>(floor(width / 8.f) * 8.f);
    unsigned short val[8];
    for (int n = 0; n < 8; n++)
    {
        val[n] = (noBits > n) ? (0xffff) : 0;
    }
    t1 = _mm_set_epi16(val[7], val[7], val[6], val[6], val[5], val[5], val[4], val[4]);
    t0 = _mm_set_epi16(val[3], val[3], val[2], val[2], val[1], val[1], val[0], val[0]);
    t2 = _mm_set_epi16(val[7], val[6], val[5], val[4], val[3], val[2], val[1], val[0]);
}
#endif
typedef uchar tMaskType;
#define tCVMaskType CV_8UC1
#define MaskSet 0xffffffff

static
void getLocalPatch(
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
