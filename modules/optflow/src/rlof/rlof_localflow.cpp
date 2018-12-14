// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../precomp.hpp"

#include "opencv2/calib3d.hpp"  // findHomography

#include "rlof_localflow.h"
#include "berlof_invoker.hpp"
#include "rlof_invoker.hpp"
#include "plk_invoker.hpp"
using namespace std;
using namespace cv;

namespace cv {
namespace detail {
typedef short deriv_type;
} // namespace

namespace {
static void calcSharrDeriv(const cv::Mat& src, cv::Mat& dst)
{
    using namespace cv;
    using cv::detail::deriv_type;
    int rows = src.rows, cols = src.cols, cn = src.channels(), colsn = cols * cn, depth = src.depth();
    CV_Assert(depth == CV_8U);
    dst.create(rows, cols, CV_MAKETYPE(DataType<deriv_type>::depth, cn * 2));

    int x, y, delta = (int)alignSize((cols + 2)*cn, 16);
    AutoBuffer<deriv_type> _tempBuf(delta * 2 + 64);
    deriv_type *trow0 = alignPtr(_tempBuf.data() + cn, 16), *trow1 = alignPtr(trow0 + delta, 16);

#if CV_SIMD128
    v_int16x8 c3 = v_setall_s16(3), c10 = v_setall_s16(10);
    bool haveSIMD = checkHardwareSupport(CV_CPU_SSE2) || checkHardwareSupport(CV_CPU_NEON);
#endif

    for (y = 0; y < rows; y++)
    {
        const uchar* srow0 = src.ptr<uchar>(y > 0 ? y - 1 : rows > 1 ? 1 : 0);
        const uchar* srow1 = src.ptr<uchar>(y);
        const uchar* srow2 = src.ptr<uchar>(y < rows - 1 ? y + 1 : rows > 1 ? rows - 2 : 0);
        deriv_type* drow = dst.ptr<deriv_type>(y);

        // do vertical convolution
        x = 0;
#if CV_SIMD128
        if (haveSIMD)
        {
            for (; x <= colsn - 8; x += 8)
            {
                v_int16x8 s0 = v_reinterpret_as_s16(v_load_expand(srow0 + x));
                v_int16x8 s1 = v_reinterpret_as_s16(v_load_expand(srow1 + x));
                v_int16x8 s2 = v_reinterpret_as_s16(v_load_expand(srow2 + x));

                v_int16x8 t1 = s2 - s0;
                v_int16x8 t0 = v_mul_wrap(s0 + s2, c3) + v_mul_wrap(s1, c10);

                v_store(trow0 + x, t0);
                v_store(trow1 + x, t1);
            }
        }
#endif

        for (; x < colsn; x++)
        {
            int t0 = (srow0[x] + srow2[x]) * 3 + srow1[x] * 10;
            int t1 = srow2[x] - srow0[x];
            trow0[x] = (deriv_type)t0;
            trow1[x] = (deriv_type)t1;
        }

        // make border
        int x0 = (cols > 1 ? 1 : 0)*cn, x1 = (cols > 1 ? cols - 2 : 0)*cn;
        for (int k = 0; k < cn; k++)
        {
            trow0[-cn + k] = trow0[x0 + k]; trow0[colsn + k] = trow0[x1 + k];
            trow1[-cn + k] = trow1[x0 + k]; trow1[colsn + k] = trow1[x1 + k];
        }

        // do horizontal convolution, interleave the results and store them to dst
        x = 0;
#if CV_SIMD128
        if (haveSIMD)
        {
            for (; x <= colsn - 8; x += 8)
            {
                v_int16x8 s0 = v_load(trow0 + x - cn);
                v_int16x8 s1 = v_load(trow0 + x + cn);
                v_int16x8 s2 = v_load(trow1 + x - cn);
                v_int16x8 s3 = v_load(trow1 + x);
                v_int16x8 s4 = v_load(trow1 + x + cn);

                v_int16x8 t0 = s1 - s0;
                v_int16x8 t1 = v_mul_wrap(s2 + s4, c3) + v_mul_wrap(s3, c10);

                v_store_interleave((drow + x * 2), t0, t1);
            }
        }
#endif
        for (; x < colsn; x++)
        {
            deriv_type t0 = (deriv_type)(trow0[x + cn] - trow0[x - cn]);
            deriv_type t1 = (deriv_type)((trow1[x + cn] + trow1[x - cn]) * 3 + trow1[x] * 10);
            drow[x * 2] = t0; drow[x * 2 + 1] = t1;
        }
    }
}

} // namespace
namespace optflow {

int buildOpticalFlowPyramidScale(InputArray _img, OutputArrayOfArrays pyramid, Size winSize, int maxLevel, bool withDerivatives,
    int pyrBorder, int derivBorder, bool tryReuseInputImage, float levelScale[2]);
void calcLocalOpticalFlowCore(Ptr<CImageBuffer>  prevPyramids[2], Ptr<CImageBuffer> currPyramids[2], InputArray _prevPts,
    InputOutputArray _nextPts, const RLOFOpticalFlowParameter & param);

void preprocess(Ptr<CImageBuffer> prevPyramids[2], Ptr<CImageBuffer> currPyramids[2], const std::vector<cv::Point2f> & prevPoints,
    std::vector<cv::Point2f> & currPoints,  const RLOFOpticalFlowParameter & param);

inline bool isrobust(const RLOFOpticalFlowParameter & param)
{
    return (param.normSigma0 != std::numeric_limits<float>::max())
        && (param.normSigma1 != std::numeric_limits<float>::max());
}
inline std::vector<float> get_norm(float sigma0, float sigma1)
{
    std::vector<float> result = { sigma0, sigma1, sigma0 / (sigma0 - sigma1), sigma0 * sigma1 };
    return result;
}


int buildOpticalFlowPyramidScale(InputArray _img, OutputArrayOfArrays pyramid, Size winSize, int maxLevel, bool withDerivatives,
    int pyrBorder, int derivBorder, bool tryReuseInputImage, float levelScale[2])
{
    Mat img = _img.getMat();
    CV_Assert(img.depth() == CV_8U && winSize.width > 2 && winSize.height > 2);
    int pyrstep = withDerivatives ? 2 : 1;

    pyramid.create(1, (maxLevel + 1) * pyrstep, 0 /*type*/, -1, true);

    int derivType = CV_MAKETYPE(DataType<short>::depth, img.channels() * 2);

    //level 0
    bool lvl0IsSet = false;
    if (tryReuseInputImage && img.isSubmatrix() && (pyrBorder & BORDER_ISOLATED) == 0)
    {
        Size wholeSize;
        Point ofs;
        img.locateROI(wholeSize, ofs);
        if (ofs.x >= winSize.width && ofs.y >= winSize.height
            && ofs.x + img.cols + winSize.width <= wholeSize.width
            && ofs.y + img.rows + winSize.height <= wholeSize.height)
        {
            pyramid.getMatRef(0) = img;
            lvl0IsSet = true;
        }
    }

    if (!lvl0IsSet)
    {
        Mat& temp = pyramid.getMatRef(0);

        if (!temp.empty())
            temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
        if (temp.type() != img.type() || temp.cols != winSize.width * 2 + img.cols || temp.rows != winSize.height * 2 + img.rows)
            temp.create(img.rows + winSize.height * 2, img.cols + winSize.width * 2, img.type());

        if (pyrBorder == BORDER_TRANSPARENT)
            img.copyTo(temp(Rect(winSize.width, winSize.height, img.cols, img.rows)));
        else
            copyMakeBorder(img, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder);
        temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
    }

    Size sz = img.size();
    Mat prevLevel = pyramid.getMatRef(0);
    Mat thisLevel = prevLevel;

    for (int level = 0; level <= maxLevel; ++level)
    {
        if (level != 0)
        {
            Mat& temp = pyramid.getMatRef(level * pyrstep);

            if (!temp.empty())
                temp.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
            if (temp.type() != img.type() || temp.cols != winSize.width * 2 + sz.width || temp.rows != winSize.height * 2 + sz.height)
                temp.create(sz.height + winSize.height * 2, sz.width + winSize.width * 2, img.type());

            thisLevel = temp(Rect(winSize.width, winSize.height, sz.width, sz.height));
            pyrDown(prevLevel, thisLevel, sz);

            if (pyrBorder != BORDER_TRANSPARENT)
                copyMakeBorder(thisLevel, temp, winSize.height, winSize.height, winSize.width, winSize.width, pyrBorder | BORDER_ISOLATED);
            temp.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
        }

        if (withDerivatives)
        {
            Mat& deriv = pyramid.getMatRef(level * pyrstep + 1);

            if (!deriv.empty())
                deriv.adjustROI(winSize.height, winSize.height, winSize.width, winSize.width);
            if (deriv.type() != derivType || deriv.cols != winSize.width * 2 + sz.width || deriv.rows != winSize.height * 2 + sz.height)
                deriv.create(sz.height + winSize.height * 2, sz.width + winSize.width * 2, derivType);

            Mat derivI = deriv(Rect(winSize.width, winSize.height, sz.width, sz.height));
            calcSharrDeriv(thisLevel, derivI);

            if (derivBorder != BORDER_TRANSPARENT)
                copyMakeBorder(derivI, deriv, winSize.height, winSize.height, winSize.width, winSize.width, derivBorder | BORDER_ISOLATED);
            deriv.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
        }

        sz = Size(static_cast<int>((sz.width + 1) / levelScale[0]),
            static_cast<int>((sz.height + 1) / levelScale[1]));
        if (sz.width <= winSize.width || sz.height <= winSize.height)
        {
            pyramid.create(1, (level + 1) * pyrstep, 0 /*type*/, -1, true);//check this
            return level;
        }

        prevLevel = thisLevel;
    }

    return maxLevel;
}
int CImageBuffer::buildPyramid(cv::Size winSize, int maxLevel, float levelScale[2])
{
    if (m_Overwrite == false)
        return m_maxLevel;
    m_maxLevel = buildOpticalFlowPyramidScale(m_Image, m_ImagePyramid, winSize, maxLevel, false, 4, 0, true, levelScale);
    return m_maxLevel;
}

void calcLocalOpticalFlowCore(
    Ptr<CImageBuffer>  prevPyramids[2],
    Ptr<CImageBuffer> currPyramids[2],
    InputArray _prevPts,
    InputOutputArray _nextPts,
    const RLOFOpticalFlowParameter & param)
{

    bool useAdditionalRGB = param.supportRegionType == SR_CROSS;
    int iWinSize = param.largeWinSize;
    int winSizes[2] = { iWinSize, iWinSize };
    if (param.supportRegionType != SR_FIXED)
    {
        winSizes[0] = param.smallWinSize;
    }
    //cv::Size winSize(iWinSize, iWinSize);
    cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, param.maxIteration, 0.01);
    std::vector<float> rlofNorm = get_norm(param.normSigma0, param.normSigma1);
    CV_Assert(winSizes[0] <= winSizes[1]);

    Mat prevPtsMat = _prevPts.getMat();
    const int derivDepth = DataType<detail::deriv_type>::depth;

    CV_Assert(param.maxLevel >= 0 && iWinSize > 2);

    int level = 0,  npoints;
    CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);

    if (!(param.useInitialFlow))
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    Mat nextPtsMat = _nextPts.getMat();
    CV_Assert(nextPtsMat.checkVector(2, CV_32F, true) == npoints);

    const Point2f* prevPts = (const Point2f*)prevPtsMat.data;
    Point2f* nextPts = (Point2f*)nextPtsMat.data;
    std::vector<uchar> status(npoints);
    std::vector<float> err(npoints);
    std::vector<Point2f> gainPts(npoints);

    float levelScale[2] = { 2.f,2.f };

    int maxLevel = prevPyramids[0]->buildPyramid(cv::Size(iWinSize, iWinSize), param.maxLevel, levelScale);

    maxLevel = currPyramids[0]->buildPyramid(cv::Size(iWinSize, iWinSize), maxLevel, levelScale);
    if (useAdditionalRGB)
    {
        prevPyramids[1]->buildPyramid(cv::Size(iWinSize, iWinSize), maxLevel, levelScale);
        currPyramids[1]->buildPyramid(cv::Size(iWinSize, iWinSize), maxLevel, levelScale);
    }

    if ((criteria.type & TermCriteria::COUNT) == 0)
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if ((criteria.type & TermCriteria::EPS) == 0)
        criteria.epsilon = 0.001;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;

    // dI/dx ~ Ix, dI/dy ~ Iy
    Mat derivIBuf;
    derivIBuf.create(prevPyramids[0]->m_ImagePyramid[0].rows + iWinSize * 2, prevPyramids[0]->m_ImagePyramid[0].cols + iWinSize * 2, CV_MAKETYPE(derivDepth, prevPyramids[0]->m_ImagePyramid[0].channels() * 2));

    for (level = maxLevel; level >= 0; level--)
    {
        Mat derivI;
        Size imgSize = prevPyramids[0]->getImage(level).size();
        Mat _derivI(imgSize.height + iWinSize * 2, imgSize.width + iWinSize * 2, derivIBuf.type(), derivIBuf.data);
        derivI = _derivI(Rect(iWinSize, iWinSize, imgSize.width, imgSize.height));
        calcSharrDeriv(prevPyramids[0]->getImage(level), derivI);
        copyMakeBorder(derivI, _derivI, iWinSize, iWinSize, iWinSize, iWinSize, BORDER_CONSTANT | BORDER_ISOLATED);

        cv::Mat tRGBPrevPyr;
        cv::Mat tRGBNextPyr;
        if (useAdditionalRGB)
        {
            tRGBPrevPyr = prevPyramids[1]->getImage(level);
            tRGBNextPyr = prevPyramids[1]->getImage(level);

            prevPyramids[1]->m_Overwrite = false;
            currPyramids[1]->m_Overwrite = false;
        }

        cv::Mat prevImage = prevPyramids[0]->getImage(level);
        cv::Mat currImage = currPyramids[0]->getImage(level);

        // apply plk like tracker
        if (isrobust(param) == false)
        {
            if (param.useIlluminationModel)
            {
                cv::parallel_for_(cv::Range(0, npoints),
                    plk::radial::TrackerInvoker(
                        prevImage, derivI, currImage, tRGBPrevPyr, tRGBNextPyr,
                        prevPts, nextPts, &status[0], &err[0], &gainPts[0],
                        level, maxLevel, winSizes,
                        param.maxIteration,
                        param.useInitialFlow,
                        param.supportRegionType,
                        param.minEigenValue,
                        param.crossSegmentationThreshold));
            }
            else
            {
                if (param.solverType == SolverType::ST_STANDART)
                {
                    cv::parallel_for_(cv::Range(0, npoints),
                        plk::ica::TrackerInvoker(
                            prevImage, derivI, currImage, tRGBPrevPyr, tRGBNextPyr,
                            prevPts, nextPts, &status[0], &err[0],
                            level, maxLevel, winSizes,
                            param.maxIteration,
                            param.useInitialFlow,
                            param.supportRegionType,
                            param.crossSegmentationThreshold,
                            param.minEigenValue));
                }
                else
                {
                    cv::parallel_for_(cv::Range(0, npoints),
                        beplk::ica::TrackerInvoker(prevImage, derivI, currImage, tRGBPrevPyr, tRGBNextPyr,
                            prevPts, nextPts, &status[0], &err[0],
                            level, maxLevel, winSizes,
                            param.maxIteration,
                            param.useInitialFlow,
                            param.supportRegionType,
                            param.crossSegmentationThreshold,
                            param.minEigenValue));
                }
            }
        }
        // for robust models
        else
        {
            if (param.useIlluminationModel)
            {
                if (param.solverType == SolverType::ST_STANDART)
                {
                    cv::parallel_for_(cv::Range(0, npoints),
                        rlof::radial::TrackerInvoker(
                            prevImage, derivI, currImage, tRGBPrevPyr, tRGBNextPyr,
                            prevPts, nextPts, &status[0], &err[0], &gainPts[0],
                            level, maxLevel, winSizes,
                            param.maxIteration,
                            param.useInitialFlow,
                            param.supportRegionType,
                            rlofNorm,
                            param.minEigenValue,
                            param.crossSegmentationThreshold));
                }
                else
                {
                    cv::parallel_for_(cv::Range(0, npoints),
                        berlof::radial::TrackerInvoker(prevImage, derivI, currImage, tRGBPrevPyr, tRGBNextPyr,
                            prevPts, nextPts, &status[0], &err[0], &gainPts[0],
                            level, maxLevel, winSizes,
                            param.maxIteration,
                            param.useInitialFlow,
                            param.supportRegionType,
                            param.crossSegmentationThreshold,
                            rlofNorm,
                            param.minEigenValue));
                }
            }
            else
            {

                if (param.solverType == SolverType::ST_STANDART)
                {
                    cv::parallel_for_(cv::Range(0, npoints),
                        rlof::ica::TrackerInvoker(
                            prevImage, derivI, currImage, tRGBPrevPyr, tRGBNextPyr,
                            prevPts, nextPts, &status[0], &err[0],
                            level, maxLevel, winSizes,
                            param.maxIteration,
                            param.useInitialFlow,
                            param.supportRegionType,
                            rlofNorm,
                            param.minEigenValue,
                            param.crossSegmentationThreshold));
                }
                else
                {
                    cv::parallel_for_(cv::Range(0, npoints),
                        berlof::ica::TrackerInvoker(prevImage, derivI, currImage, tRGBPrevPyr, tRGBNextPyr,
                            prevPts, nextPts, &status[0], &err[0],
                            level, maxLevel, winSizes,
                            param.maxIteration,
                            param.useInitialFlow,
                            param.supportRegionType,
                            param.crossSegmentationThreshold,
                            rlofNorm,
                            param.minEigenValue));
                }

            }
        }

        prevPyramids[0]->m_Overwrite = true;
        currPyramids[0]->m_Overwrite = true;
    }
}

void preprocess(Ptr<CImageBuffer> prevPyramids[2],
    Ptr<CImageBuffer> currPyramids[2],
    const std::vector<cv::Point2f> & prevPoints,
    std::vector<cv::Point2f> & currPoints,
    const RLOFOpticalFlowParameter & param)
{
    cv::Mat mask, homography;
    if (param.useGlobalMotionPrior == false)
        return;

    currPoints.resize(prevPoints.size());

    RLOFOpticalFlowParameter gmeTrackerParam = param;
    gmeTrackerParam.useGlobalMotionPrior = false;
    gmeTrackerParam.largeWinSize = 17;
    // use none robust tracker for global motion estimation since it is faster
    gmeTrackerParam.normSigma0 = std::numeric_limits<float>::max();
    gmeTrackerParam.maxIteration = MAX(15, param.maxIteration);
    gmeTrackerParam.minEigenValue = 0.000001f;

    std::vector<cv::Point2f> gmPrevPoints, gmCurrPoints;

    // Initialize point grid
    int stepr = prevPyramids[0]->m_Image.rows / 30;
    int stepc = prevPyramids[0]->m_Image.rows / 40;
    for (int r = stepr / 2; r < prevPyramids[0]->m_Image.rows; r += stepr)
    {
        for (int c = stepc / 2; c < prevPyramids[0]->m_Image.cols; c += stepc)
        {
            gmPrevPoints.push_back(cv::Point2f(static_cast<float>(c), static_cast<float>(r)));
        }
    }

    // perform motion estimation
    calcLocalOpticalFlowCore(prevPyramids, currPyramids, gmPrevPoints, gmCurrPoints, gmeTrackerParam);

    cv::Mat prevPointsMat(static_cast<int>(gmPrevPoints.size()), 1, CV_32FC2);
    cv::Mat currPointsMat(static_cast<int>(gmPrevPoints.size()), 1, CV_32FC2);
    cv::Mat distMat(static_cast<int>(gmPrevPoints.size()), 1, CV_32FC1);

    // Forward backward confidence to estimate optimal ransac reprojection error
    int noPoints = 0;
    for (unsigned int n = 0; n < gmPrevPoints.size(); n++)
    {
        cv::Point2f flow = gmCurrPoints[n] - gmPrevPoints[n];
        prevPointsMat.at<cv::Point2f>(noPoints) = gmPrevPoints[n];
        currPointsMat.at<cv::Point2f>(noPoints) = gmCurrPoints[n];
        distMat.at<float>(noPoints) = flow.x * flow.x + flow.y* flow.y;
        if (isnan(distMat.at<float>(noPoints)) == false)
            noPoints++;
    }

    float medianDist = (param.globalMotionRansacThreshold == 0) ? 1.f :
        quickselect<float>(distMat, static_cast<int>(noPoints  * static_cast<float>(param.globalMotionRansacThreshold) / 100.f));
    medianDist = sqrt(medianDist);

    if (noPoints < 8)
        return;

    cv::findHomography(prevPointsMat(cv::Rect(0, 0, 1, noPoints)), currPointsMat(cv::Rect(0, 0, 1, noPoints)), cv::RANSAC, medianDist, mask).convertTo(homography, CV_32FC1);

    if (homography.empty())
        return;

    cv::perspectiveTransform(prevPoints, currPoints, homography);
}

void calcLocalOpticalFlow(
    const Mat prevImage,
    const Mat currImage,
    Ptr<CImageBuffer>  prevPyramids[2],
    Ptr<CImageBuffer>  currPyramids[2],
    const std::vector<Point2f> & prevPoints,
    std::vector<Point2f> & currPoints,
    const RLOFOpticalFlowParameter & param)
{
    if (prevImage.empty() == false && currImage.empty()== false)
    {
        prevPyramids[0]->m_Overwrite = true;
        currPyramids[0]->m_Overwrite = true;
        prevPyramids[1]->m_Overwrite = true;
        currPyramids[1]->m_Overwrite = true;
        if (prevImage.type() == CV_8UC3)
        {
            prevPyramids[0]->setGrayFromRGB(prevImage);
            currPyramids[0]->setGrayFromRGB(currImage);
            prevPyramids[1]->setImage(prevImage);
            currPyramids[1]->setImage(currImage);

            if (param.supportRegionType == SR_CROSS)
            {
                prevPyramids[1]->setBlurFromRGB(prevImage);
                currPyramids[1]->setBlurFromRGB(currImage);
            }
        }
        else
        {
            prevPyramids[0]->setImage(prevImage);
            currPyramids[0]->setImage(currImage);
        }
    }
    preprocess(prevPyramids, currPyramids, prevPoints, currPoints, param);
    RLOFOpticalFlowParameter internParam = param;
    if (param.useGlobalMotionPrior == true)
        internParam.useInitialFlow = true;
    calcLocalOpticalFlowCore(prevPyramids, currPyramids, prevPoints, currPoints, internParam);
}

}} // namespace
