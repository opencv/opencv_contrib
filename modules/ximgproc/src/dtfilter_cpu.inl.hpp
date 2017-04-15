/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *  
 *  
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *  
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *  
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *  
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *  
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#ifndef __OPENCV_DTFILTER_INL_HPP__
#define __OPENCV_DTFILTER_INL_HPP__
#include "precomp.hpp"
#include "edgeaware_filters_common.hpp"
#include <limits>

namespace cv
{
namespace ximgproc
{

using namespace cv::ximgproc::intrinsics;

#define NC_USE_INTEGRAL_SRC
//#undef NC_USE_INTEGRAL_SRC

template<typename GuideVec>
DTFilterCPU DTFilterCPU::create_(const Mat& guide, double sigmaSpatial, double sigmaColor, int mode, int numIters)
{
    DTFilterCPU dtf;
    dtf.init_<GuideVec>(guide, sigmaSpatial, sigmaColor, mode, numIters);
    return dtf;
}

template<typename GuideVec>
DTFilterCPU* DTFilterCPU::create_p_(const Mat& guide, double sigmaSpatial, double sigmaColor, int mode, int numIters)
{
    DTFilterCPU* dtf = new DTFilterCPU();
    dtf->init_<GuideVec>(guide, sigmaSpatial, sigmaColor, mode, numIters);
    return dtf;
}

template<typename GuideVec>
void DTFilterCPU::init_(Mat& guide, double sigmaSpatial_, double sigmaColor_, int mode_, int numIters_)
{
    CV_Assert(guide.type() == cv::DataType<GuideVec>::type);

    this->release();

    h = guide.rows;
    w = guide.cols;

    sigmaSpatial = std::max(1.0f, (float)sigmaSpatial_);
    sigmaColor   = std::max(0.01f, (float)sigmaColor_);

    mode = mode_;
    numIters = std::max(1, numIters_);

    if (mode == DTF_NC)
    {
        {
            ComputeIDTHor_ParBody<GuideVec> horBody(*this, guide, idistHor);
            parallel_for_(horBody.getRange(), horBody);
        }
        {
            Mat guideT = guide.t();
            ComputeIDTHor_ParBody<GuideVec> horBody(*this, guideT, idistVert);
            parallel_for_(horBody.getRange(), horBody);
        }
    }
    else if (mode == DTF_IC)
    {
        {
            ComputeDTandIDTHor_ParBody<GuideVec> horBody(*this, guide, distHor, idistHor);
            parallel_for_(horBody.getRange(), horBody);
        }
        {
            Mat guideT = guide.t();
            ComputeDTandIDTHor_ParBody<GuideVec> horBody(*this, guideT, distVert, idistVert);
            parallel_for_(horBody.getRange(), horBody);
        }
    }
    else if (mode == DTF_RF)
    {
        ComputeA0DTHor_ParBody<GuideVec> horBody(*this, guide);
        ComputeA0DTVert_ParBody<GuideVec> vertBody(*this, guide);
        parallel_for_(horBody.getRange(), horBody);
        parallel_for_(vertBody.getRange(), vertBody);
    }
    else
    {
        CV_Error(Error::StsBadFlag, "Incorrect DT filter mode");
    }
}

template <typename SrcVec>
void DTFilterCPU::filter_(const Mat& src, Mat& dst, int dDepth)
{
    typedef typename DataType<Vec<WorkType, SrcVec::channels> >::vec_type WorkVec;
    CV_Assert( src.type() == SrcVec::type );
    if ( src.cols != w || src.rows != h )
    {
        CV_Error(Error::StsBadSize, "Size of filtering image must be equal to size of guide image");
    }

    if (singleFilterCall)
    {
        CV_Assert(numFilterCalls == 0);
    }
    numFilterCalls++;

    Mat res;
    if (dDepth == -1) dDepth = src.depth();
    
    //small optimization to avoid extra copying of data
    bool useDstAsRes = (dDepth == WorkVec::depth && (mode == DTF_NC || mode == DTF_RF));
    if (useDstAsRes)
    {
        dst.create(h, w, WorkVec::type);
        res = dst;
    }

    if (mode == DTF_NC)
    {
        Mat resT(src.cols, src.rows, WorkVec::type);
        src.convertTo(res, WorkVec::type);

        FilterNC_horPass<WorkVec> horParBody(res, idistHor, resT);
        FilterNC_horPass<WorkVec> vertParBody(resT, idistVert, res);

        for (int iter = 1; iter <= numIters; iter++)
        {
            horParBody.radius = vertParBody.radius = getIterRadius(iter);

            parallel_for_(Range(0, res.rows), horParBody);
            parallel_for_(Range(0, resT.rows), vertParBody);
        }
    }
    else if (mode == DTF_IC)
    {
        Mat resT;
        prepareSrcImg_IC<WorkVec>(src, res, resT);

        FilterIC_horPass<WorkVec> horParBody(res, idistHor, distHor, resT);
        FilterIC_horPass<WorkVec> vertParBody(resT, idistVert, distVert, res);

        for (int iter = 1; iter <= numIters; iter++)
        {
            horParBody.radius = vertParBody.radius = getIterRadius(iter);

            parallel_for_(Range(0, res.rows), horParBody);
            parallel_for_(Range(0, resT.rows), vertParBody);
        }
    }
    else if (mode == DTF_RF)
    {
        src.convertTo(res, WorkVec::type);

        for (int iter = 1; iter <= numIters; iter++)
        {
            if (!singleFilterCall && iter == 2)
            {
                a0distHor.copyTo(adistHor);
                a0distVert.copyTo(adistVert);
            }

            bool useA0DT = (singleFilterCall || iter == 1);
            Mat& a0dHor  = (useA0DT) ? a0distHor : adistHor;
            Mat& a0dVert = (useA0DT) ? a0distVert : adistVert;
            
            FilterRF_horPass<WorkVec> horParBody(res, a0dHor, iter);
            FilterRF_vertPass<WorkVec> vertParBody(res, a0dVert, iter);
            parallel_for_(horParBody.getRange(), horParBody);
            parallel_for_(vertParBody.getRange(), vertParBody);
        }
    }

    if (!useDstAsRes)
    {
        res.convertTo(dst, dDepth);
    }
}

template<typename SrcVec, typename SrcWorkVec>
void DTFilterCPU::integrateRow(const SrcVec *src, SrcWorkVec *dst, int cols)
{
    SrcWorkVec sum = SrcWorkVec::all(0);
    dst[0] = sum;

    for (int j = 0; j < cols; j++)
    {
        sum += SrcWorkVec(src[j]);
        dst[j + 1] = sum;
    }
}


template<typename SrcVec, typename SrcWorkVec>
void DTFilterCPU::integrateSparseRow(const SrcVec *src, const float *dist, SrcWorkVec *dst, int cols)
{
    SrcWorkVec sum = SrcWorkVec::all(0);
    dst[0] = sum;

    for (int j = 0; j < cols-1; j++)
    {
        sum += dist[j] * 0.5f * (SrcWorkVec(src[j]) + SrcWorkVec(src[j+1]));
        dst[j + 1] = sum;
    }
}

template<typename WorkVec>
void DTFilterCPU::prepareSrcImg_IC(const Mat& src, Mat& dst, Mat& dstT)
{
    Mat dstOut(src.rows, src.cols + 2, WorkVec::type);
    Mat dstOutT(src.cols, src.rows + 2, WorkVec::type);

    dst = dstOut(Range::all(), Range(1, src.cols+1));
    dstT = dstOutT(Range::all(), Range(1, src.rows+1));

    src.convertTo(dst, WorkVec::type);

    WorkVec *line;
    int ri = dstOut.cols - 1;
    for (int i = 0; i < src.rows; i++)
    {
        line        = dstOut.ptr<WorkVec>(i);
        line[0]     = line[1];
        line[ri]    = line[ri - 1];
    }

    WorkVec *topLine = dst.ptr<WorkVec>(0);
    WorkVec *bottomLine = dst.ptr<WorkVec>(dst.rows - 1);
    ri = dstOutT.cols - 1;
    for (int i = 0; i < src.cols; i++)
    {
        line        = dstOutT.ptr<WorkVec>(i);
        line[0]     = topLine[i];
        line[ri]    = bottomLine[i]; 
    }
}


template <typename WorkVec>
DTFilterCPU::FilterNC_horPass<WorkVec>::FilterNC_horPass(Mat& src_, Mat& idist_, Mat& dst_)
: src(src_), idist(idist_), dst(dst_), radius(1.0f)
{
    CV_DbgAssert(src.type() == WorkVec::type && dst.type() == WorkVec::type && dst.rows == src.cols && dst.cols == src.rows);
}

template <typename WorkVec>
void DTFilterCPU::FilterNC_horPass<WorkVec>::operator()(const Range& range) const
{
    #ifdef NC_USE_INTEGRAL_SRC
    std::vector<WorkVec> isrcBuf(src.cols + 1);
    WorkVec *isrcLine = &isrcBuf[0];
    #endif

    for (int i = range.start; i < range.end; i++)
    {
        const WorkVec   *srcLine    = src.ptr<WorkVec>(i);
        IDistType       *idistLine  = idist.ptr<IDistType>(i);
        int leftBound = 0, rightBound = 0;
        WorkVec sum;

        #ifdef NC_USE_INTEGRAL_SRC
        integrateRow(srcLine, isrcLine, src.cols);
        #else
        sum = srcLine[0];
        #endif

        for (int j = 0; j < src.cols; j++)
        {
            IDistType curVal = idistLine[j];
            #ifdef NC_USE_INTEGRAL_SRC
            leftBound  = getLeftBound(idistLine, leftBound, curVal - radius);
            rightBound = getRightBound(idistLine, rightBound, curVal + radius);
            sum = (isrcLine[rightBound + 1] - isrcLine[leftBound]);
            #else
            while (idistLine[leftBound] < curVal - radius)
            {
                sum -= srcLine[leftBound];
                leftBound++;
            }

            while (idistLine[rightBound + 1] < curVal + radius)
            {
                rightBound++;
                sum += srcLine[rightBound];
            }
            #endif

            dst.at<WorkVec>(j, i) = sum / (float)(rightBound + 1 - leftBound);
        }
    }
}

template <typename WorkVec>
DTFilterCPU::FilterIC_horPass<WorkVec>::FilterIC_horPass(Mat& src_, Mat& idist_, Mat& dist_, Mat& dst_)
: src(src_), idist(idist_), dist(dist_), dst(dst_), radius(1.0f)
{
    CV_DbgAssert(src.type() == WorkVec::type && dst.type() == WorkVec::type && dst.rows == src.cols && dst.cols == src.rows);

    #ifdef CV_GET_NUM_THREAD_WORKS_PROPERLY
    isrcBuf.create(cv::getNumThreads(), src.cols + 1, WorkVec::type);
    #else
    isrcBuf.create(src.rows, src.cols + 1, WorkVec::type);
    #endif
}

template <typename WorkVec>
void DTFilterCPU::FilterIC_horPass<WorkVec>::operator()(const Range& range) const
{
    #ifdef CV_GET_NUM_THREAD_WORKS_PROPERLY
    WorkVec *isrcLine = const_cast<WorkVec*>( isrcBuf.ptr<WorkVec>(cv::getThreadNum()) );
    #else
    WorkVec *isrcLine = const_cast<WorkVec*>( isrcBuf.ptr<WorkVec>(range.start) );
    #endif

    for (int i = range.start; i < range.end; i++)
    {
        WorkVec   *srcLine      = src.ptr<WorkVec>(i);
        DistType  *distLine     = dist.ptr<DistType>(i);
        IDistType *idistLine    = idist.ptr<IDistType>(i);

        integrateSparseRow(srcLine, distLine, isrcLine, src.cols);

        int leftBound = 0, rightBound = 0;
        WorkVec sumL, sumR, sumC;

        srcLine[-1] = srcLine[0];
        srcLine[src.cols] = srcLine[src.cols - 1];

        for (int j = 0; j < src.cols; j++)
        {
            IDistType curVal = idistLine[j];
            IDistType valueLeft = curVal - radius;
            IDistType valueRight = curVal + radius;

            leftBound = getLeftBound(idistLine, leftBound, valueLeft);
            rightBound = getRightBound(idistLine, rightBound, valueRight);

            float areaL = idistLine[leftBound] - valueLeft;
            float areaR = valueRight - idistLine[rightBound];
            float dl = areaL / distLine[leftBound - 1];
            float dr = areaR / distLine[rightBound];

            sumL = 0.5f*areaL*(dl*srcLine[leftBound - 1] + (2.0f - dl)*srcLine[leftBound]);
            sumR = 0.5f*areaR*((2.0f - dr)*srcLine[rightBound] + dr*srcLine[rightBound + 1]);
            sumC = isrcLine[rightBound] - isrcLine[leftBound];

            dst.at<WorkVec>(j, i) = (sumL + sumC + sumR) / (2.0f * radius);
        }
    }
}


template <typename WorkVec>
DTFilterCPU::FilterRF_horPass<WorkVec>::FilterRF_horPass(Mat& res_, Mat& alphaD_, int iteration_)
: res(res_), alphaD(alphaD_), iteration(iteration_)
{
    CV_DbgAssert(res.type() == WorkVec::type);
    CV_DbgAssert(res.type() == WorkVec::type && res.size() == res.size());
}


template <typename WorkVec>
void DTFilterCPU::FilterRF_horPass<WorkVec>::operator()(const Range& range) const
{
    for (int i = range.start; i < range.end; i++)
    {
        WorkVec     *dstLine = res.ptr<WorkVec>(i);
        DistType    *adLine  = alphaD.ptr<DistType>(i);
        int j;

        if (iteration > 1)
        {
            for (j = res.cols - 2; j >= 0; j--)
                adLine[j] *= adLine[j];
        }

        for (j = 1; j < res.cols; j++)
        {
            dstLine[j] += adLine[j-1] * (dstLine[j-1] - dstLine[j]);
        }

        for (j = res.cols - 2; j >= 0; j--)
        {
            dstLine[j] += adLine[j] * (dstLine[j+1] - dstLine[j]);
        }
    }
}


template <typename WorkVec>
DTFilterCPU::FilterRF_vertPass<WorkVec>::FilterRF_vertPass(Mat& res_, Mat& alphaD_, int iteration_)
: res(res_), alphaD(alphaD_), iteration(iteration_)
{
    CV_DbgAssert(res.type() == WorkVec::type);
    CV_DbgAssert(res.type() == WorkVec::type && res.size() == res.size());
}


template <typename WorkVec>
void DTFilterCPU::FilterRF_vertPass<WorkVec>::operator()(const Range& range) const
{
    #ifdef CV_GET_NUM_THREAD_WORKS_PROPERLY
    Range rcols = getWorkRangeByThread(res.cols, range);
    #else
    Range rcols = range;
    #endif

    for (int i = 1; i < res.rows; i++)
    {
        WorkVec     *curRow  = res.ptr<WorkVec>(i);
        WorkVec     *prevRow = res.ptr<WorkVec>(i - 1);
        DistType    *adRow   = alphaD.ptr<DistType>(i - 1);

        if (iteration > 1)
        {
            for (int j = rcols.start; j < rcols.end; j++)
                adRow[j] *= adRow[j];
        }

        for (int j = rcols.start; j < rcols.end; j++)
        {
            curRow[j] += adRow[j] * (prevRow[j] - curRow[j]);
        }
    }

    for (int i = res.rows - 2; i >= 0; i--)
    {
        WorkVec     *prevRow = res.ptr<WorkVec>(i + 1);
        WorkVec     *curRow  = res.ptr<WorkVec>(i);
        DistType    *adRow   = alphaD.ptr<DistType>(i);

        for (int j = rcols.start; j < rcols.end; j++)
        {
            curRow[j] += adRow[j] * (prevRow[j] - curRow[j]);
        }
    }
}

template <typename GuideVec>
DTFilterCPU::ComputeIDTHor_ParBody<GuideVec>::ComputeIDTHor_ParBody(DTFilterCPU& dtf_, Mat& guide_, Mat& dst_)
: dtf(dtf_), guide(guide_), dst(dst_)
{
    dst.create(guide.rows, guide.cols + 1, IDistVec::type);
}

template <typename GuideVec>
void DTFilterCPU::ComputeIDTHor_ParBody<GuideVec>::operator()(const Range& range) const
{
    for (int i = range.start; i < range.end; i++)
    {
        const GuideVec *guideLine   = guide.ptr<GuideVec>(i);
        IDistType *idistLine        = dst.ptr<IDistType>(i);

        IDistType curDist   = (IDistType)0;
        idistLine[0]        = (IDistType)0;

        for (int j = 1; j < guide.cols; j++)
        {
            curDist += dtf.getTransformedDistance(guideLine[j-1], guideLine[j]);
            idistLine[j] = curDist;
        }
        idistLine[guide.cols] = std::numeric_limits<IDistType>::max();
    }
}

template <typename GuideVec>
DTFilterCPU::ComputeDTandIDTHor_ParBody<GuideVec>::ComputeDTandIDTHor_ParBody(DTFilterCPU& dtf_, Mat& guide_, Mat& dist_, Mat& idist_)
: dtf(dtf_), guide(guide_), dist(dist_), idist(idist_)
{
    dist = getWExtendedMat(guide.rows, guide.cols, IDistVec::type, 1, 1);
    idist = getWExtendedMat(guide.rows, guide.cols + 1, IDistVec::type);
    maxRadius = dtf.getIterRadius(1);
}

template <typename GuideVec>
void DTFilterCPU::ComputeDTandIDTHor_ParBody<GuideVec>::operator()(const Range& range) const
{
    for (int i = range.start; i < range.end; i++)
    {
        const GuideVec  *guideLine  = guide.ptr<GuideVec>(i);
              DistType  *distLine   = dist.ptr<DistType>(i);
              IDistType *idistLine  = idist.ptr<IDistType>(i);

        DistType  curDist;
        IDistType curIDist = (IDistType)0;
        int j;
                
        distLine[-1] = maxRadius;
        //idistLine[-1] = curIDist - maxRadius;
        idistLine[0] = curIDist;
        for (j = 0; j < guide.cols-1; j++)
        {
            curDist = (DistType) dtf.getTransformedDistance(guideLine[j], guideLine[j + 1]);
            curIDist += curDist;

            distLine[j] = curDist;
            idistLine[j + 1] = curIDist;
        }
        idistLine[j + 1] = curIDist + maxRadius;
        distLine[j] = maxRadius;
    }
}

template <typename GuideVec>
DTFilterCPU::ComputeA0DTHor_ParBody<GuideVec>::ComputeA0DTHor_ParBody(DTFilterCPU& dtf_, Mat& guide_)
: dtf(dtf_), guide(guide_)
{
    dtf.a0distHor.create(guide.rows, guide.cols - 1, DistVec::type);
    lna = std::log(dtf.getIterAlpha(1));
}

template <typename GuideVec>
void DTFilterCPU::ComputeA0DTHor_ParBody<GuideVec>::operator()(const Range& range) const
{
    for (int i = range.start; i < range.end; i++)
    {
        const GuideVec  *guideLine  = guide.ptr<GuideVec>(i);
              DistType  *dstLine    = dtf.a0distHor.ptr<DistType>(i);

        for (int j = 0; j < guide.cols - 1; j++)
        {
            DistType d = (DistType)dtf.getTransformedDistance(guideLine[j], guideLine[j + 1]);
            dstLine[j] = lna*d;
        }
    }
}

template <typename GuideVec>
DTFilterCPU::ComputeA0DTHor_ParBody<GuideVec>::~ComputeA0DTHor_ParBody()
{
    cv::exp(dtf.a0distHor, dtf.a0distHor);
}

template <typename GuideVec>
DTFilterCPU::ComputeA0DTVert_ParBody<GuideVec>::ComputeA0DTVert_ParBody(DTFilterCPU& dtf_, Mat& guide_)
: dtf(dtf_), guide(guide_)
{
    dtf.a0distVert.create(guide.rows - 1, guide.cols, DistVec::type);
    lna = std::log(dtf.getIterAlpha(1));
}

template <typename GuideVec>
void DTFilterCPU::ComputeA0DTVert_ParBody<GuideVec>::operator()(const Range& range) const
{
    for (int i = range.start; i < range.end; i++)
    {
        DistType *dstLine = dtf.a0distVert.ptr<DistType>(i);
        GuideVec *guideRow1 = guide.ptr<GuideVec>(i);
        GuideVec *guideRow2 = guide.ptr<GuideVec>(i+1);

        for (int j = 0; j < guide.cols; j++)
        {
            DistType d = (DistType)dtf.getTransformedDistance(guideRow1[j], guideRow2[j]);
            dstLine[j] = lna*d;
        }
    }
}

template <typename GuideVec>
DTFilterCPU::ComputeA0DTVert_ParBody<GuideVec>::~ComputeA0DTVert_ParBody()
{
    cv::exp(dtf.a0distVert, dtf.a0distVert);
}


template<typename GuideVec, typename SrcVec>
void domainTransformFilter( const Mat_<GuideVec>& guide,
                            const Mat_<SrcVec>& source,
                            Mat& dst,
                            double sigmaSpatial, double sigmaColor,
                            int mode, int numPasses
                            )
{
    DTFilterCPU *dtf = DTFilterCPU::create_p_<GuideVec>(guide, sigmaSpatial, sigmaColor, mode, numPasses);
    dtf->filter_<SrcVec>(source, dst);
    delete dtf;
}

template<typename GuideVec, typename SrcVec>
void domainTransformFilter( const Mat& guide,
                            const Mat& source,
                            Mat& dst,
                            double sigmaSpatial, double sigmaColor,
                            int mode, int numPasses
                          )
{
    DTFilterCPU *dtf = DTFilterCPU::create_p_<GuideVec>(guide, sigmaSpatial, sigmaColor, mode, numPasses);
    dtf->filter_<SrcVec>(source, dst);
    delete dtf;
}

}
}
#endif