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

static inline bool notSameColor(const cv::Point3_<uchar> & ref, int r, int c, const cv::Mat & img, int winSize, int threshold)
{
    if (r >= img.rows + winSize || c >= img.cols + winSize || r < -winSize || c < -winSize) return true;
    int step = static_cast<int>(img.step1());
    const cv::Point3_<uchar> * tval = img.ptr<cv::Point3_<uchar>>();
    tval += c;
    tval += r * step / 3;
    int R = std::abs((int)ref.x - (int)tval->x);
    int G = std::abs((int)ref.y - (int)tval->y);
    int B = std::abs((int)ref.z - (int)tval->z);
    int diff = MAX(R, MAX(G, B));
    return diff > threshold;
}

/*! Estimate the mask with points of the same color as the middle point
*\param prevGray input CV_8UC1
*\param prevImg input CV_8UC3
*\param nextImg input CV_8UC3
*\param prevPoint global position of the middle point of the mask window
*\param nextPoint global position of the middle point of the mask window
*\param winPointMask mask matrice with 1 labeling points contained and 0 point is not contained by the segment
*\param noPoints number of points contained by the segment
*\param winRoi rectangle of the region of interesset in global coordinates
*\param minWinSize,
*\param threshold,
*\param useBothImages
*/
static
void getLocalPatch(
    const cv::Mat & prevImg,
    const cv::Point2i & prevPoint, // feature points
    cv::Mat & winPointMask,
    int & noPoints,
    cv::Rect & winRoi,
    int minWinSize,
    int threshold)
{
    int maxWinSizeH = (winPointMask.cols - 1) / 2;
    winRoi.x = prevPoint.x;
    winRoi.y = prevPoint.y;
    winRoi.width = winPointMask.cols;
    winRoi.height = winPointMask.rows;

    if (minWinSize == winPointMask.cols)
    {
        winRoi.x = prevPoint.x - maxWinSizeH;
        winRoi.y = prevPoint.y - maxWinSizeH;
        winPointMask.setTo(1);
        noPoints = winPointMask.size().area();
        return;
    }
    noPoints = 0;
    int c = prevPoint.x;
    int r = prevPoint.y;
    int c_left = c - 1;
    int c_right = c + 1;
    int r_top = r - 1;
    int r_bottom = r;
    int border_left = c - maxWinSizeH;
    int border_right = c + maxWinSizeH;
    int border_top = r - maxWinSizeH;
    int border_bottom = r + maxWinSizeH;
    int c_local_diff = prevPoint.x - maxWinSizeH;
    int r_local_diff = prevPoint.y - maxWinSizeH;
    int _c = c - c_local_diff;
    int _r = r - r_local_diff;
    int min_r = _r;
    int max_r = _r;
    int min_c = _c;
    int max_c = _c;
    // horizontal line
    if (r < 0 || r >= prevImg.rows || c < 0 || c >= prevImg.cols)
    {
        noPoints = 0;
        return;
    }
    cv::Point3_<uchar> val1 = prevImg.at<cv::Point3_<uchar>>(r, c); // middle grayvalue
    cv::Point3_<uchar> tval;

    //vertical line
    for (int dr = r_top; dr >= border_top; dr--)
    {
        if (notSameColor(val1, dr, c, prevImg, maxWinSizeH, threshold))
            break;

        int _dr = dr - r_local_diff;
        min_r = MIN(min_r, _dr);
        max_r = MAX(max_r, _dr);
        winPointMask.at<uchar>(_dr, _c) = 1;
        noPoints++;
    }
    for (int dr = r_bottom; dr < border_bottom; dr++)
    {
        if (notSameColor(val1, dr, c, prevImg, maxWinSizeH, threshold)
            )
            break;

        int _dr = dr - r_local_diff;
        min_r = MIN(min_r, _dr);
        max_r = MAX(max_r, _dr);
        winPointMask.at<uchar>(_dr, _c) = 1;
        noPoints++;
    }
    // accumulate along the vertical line and the search line that was still labled
    for (int dr = min_r + r_local_diff; dr <= max_r + r_local_diff; dr++)
    {
        int _dr = dr - r_local_diff;
        if (winPointMask.at<uchar>(_dr, _c) == 0)
        {
            winPointMask.row(_dr).setTo(0);
            continue;
        }
        bool skip = false;
        int _dc = c_right - c_local_diff;
        for (int dc = c_right; dc < border_right; dc++, _dc++)
        {
            if (skip == false)
            {
                if (notSameColor(val1, dr, dc, prevImg, maxWinSizeH, threshold))
                    skip = true;
            }
            if (skip == false)
            {
                min_c = MIN(min_c, _dc);
                max_c = MAX(max_c, _dc);
                winPointMask.at<uchar>(_dr, _dc) = 1;
                noPoints++;
            }
            else
                winPointMask.at<uchar>(_dr, _dc) = 0;
        }

        skip = false;
        _dc = c_left - c_local_diff;
        for (int dc = c_left; dc >= border_left; dc--, _dc--)
        {
            if (skip == false)
            {
                if (notSameColor(val1, dr, dc, prevImg, maxWinSizeH, threshold))
                    skip = true;
            }
            if (skip == false)
            {
                min_c = MIN(min_c, _dc);
                max_c = MAX(max_c, _dc);
                winPointMask.at<uchar>(_dr, _dc) = 1;
                noPoints++;
            }
            else
                winPointMask.at<uchar>(_dr, _dc) = 0;
        }
    }


    // get the initial small window
    if (noPoints < minWinSize * minWinSize)
    {
        cv::Rect roi(winPointMask.cols / 2 - (minWinSize - 1) / 2,
            winPointMask.rows / 2 - (minWinSize - 1) / 2,
            minWinSize, minWinSize);
        cv::Mat(winPointMask, roi).setTo(1);
        min_c = MIN(MIN(min_c, roi.tl().x), roi.br().x - 1);
        max_c = MAX(MAX(max_c, roi.tl().x), roi.br().x - 1);
        min_r = MIN(MIN(min_r, roi.tl().y), roi.br().y - 1);
        max_r = MAX(MAX(max_r, roi.tl().y), roi.br().y - 1);
        noPoints += minWinSize * minWinSize;
    }
    winRoi.x = c_local_diff + min_c;
    winRoi.y = r_local_diff + min_r;
    winRoi.width = max_c - min_c + 1;
    winRoi.height = max_r - min_r + 1;
    winPointMask = winPointMask(cv::Rect(min_c, min_r, winRoi.width, winRoi.height));
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
        const int maxWinSize,
        const int crossSegmentationThreshold)
{
    if (windowType == SR_CROSS && maxWinSize != minWinSize)
    {
        // patch generation
        cv::Rect winRoi;
        getLocalPatch(BI, iprevPt, winMaskMat, winArea, winRoi, minWinSize, crossSegmentationThreshold);
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
