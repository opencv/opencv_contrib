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

#include "precomp.hpp"
#include "dtfilter_cpu.hpp"

namespace cv
{
namespace ximgproc
{

typedef Vec<uchar, 1> Vec1b;
typedef Vec<float, 1> Vec1f;

Ptr<DTFilterCPU> DTFilterCPU::create(InputArray guide, double sigmaSpatial, double sigmaColor, int mode, int numIters)
{
    Ptr<DTFilterCPU> dtf(new DTFilterCPU());
    dtf->init(guide, sigmaSpatial, sigmaColor, mode, numIters);
    return dtf;
}

Ptr<DTFilterCPU> DTFilterCPU::createRF(InputArray adistHor, InputArray adistVert, double sigmaSpatial, double sigmaColor, int numIters /*= 3*/)
{
    Mat adh = adistHor.getMat();
    Mat adv = adistVert.getMat();
    CV_Assert(adh.type() == CV_32FC1 && adv.type() == CV_32FC1 && adh.rows == adv.rows + 1 && adh.cols == adv.cols - 1);

    Ptr<DTFilterCPU> dtf(new DTFilterCPU());
    dtf->release();
    dtf->mode = DTF_RF;
    dtf->numIters = std::max(1, numIters);

    dtf->h = adh.rows;
    dtf->w = adh.cols + 1;

    dtf->sigmaSpatial = std::max(1.0f, (float)sigmaSpatial);
    dtf->sigmaColor = std::max(0.01f, (float)sigmaColor);

    dtf->a0distHor = adh;
    dtf->a0distVert = adv;

    return dtf;
}

void DTFilterCPU::init(InputArray guide_, double sigmaSpatial_, double sigmaColor_, int mode_, int numIters_)
{
    Mat guide = guide_.getMat();

    int cn = guide.channels();
    int depth = guide.depth();

    CV_Assert(cn <= 4);
    CV_Assert((depth == CV_8U || depth == CV_32F) && !guide.empty());

    #define CREATE_DTF(Vect) init_<Vect>(guide, sigmaSpatial_, sigmaColor_, mode_, numIters_);

    if (cn == 1)
    {
        if (depth == CV_8U)
            CREATE_DTF(Vec1b);
        if (depth == CV_32F)
            CREATE_DTF(Vec1f);
    }
    else if (cn == 2)
    {
        if (depth == CV_8U)
            CREATE_DTF(Vec2b);
        if (depth == CV_32F)
            CREATE_DTF(Vec2f);
    }
    else if (cn == 3)
    {
        if (depth == CV_8U)
            CREATE_DTF(Vec3b);
        if (depth == CV_32F)
            CREATE_DTF(Vec3f);
    }
    else if (cn == 4)
    {
        if (depth == CV_8U)
            CREATE_DTF(Vec4b);
        if (depth == CV_32F)
            CREATE_DTF(Vec4f);
    }

    #undef CREATE_DTF
}

void DTFilterCPU::filter(InputArray src_, OutputArray dst_, int dDepth)
{
    Mat src = src_.getMat();
    dst_.create(src.size(), src.type());
    Mat& dst = dst_.getMatRef();

    int cn = src.channels();
    int depth = src.depth();

    CV_Assert(cn <= 4 && (depth == CV_8U || depth == CV_32F));

    if (cn == 1)
    {
        if (depth == CV_8U)
            filter_<Vec1b>(src, dst, dDepth);
        if (depth == CV_32F)
            filter_<Vec1f>(src, dst, dDepth);
    }
    else if (cn == 2)
    {
        if (depth == CV_8U)
            filter_<Vec2b>(src, dst, dDepth);
        if (depth == CV_32F)
            filter_<Vec2f>(src, dst, dDepth);
    }
    else if (cn == 3)
    {
        if (depth == CV_8U)
            filter_<Vec3b>(src, dst, dDepth);
        if (depth == CV_32F)
            filter_<Vec3f>(src, dst, dDepth);
    }
    else if (cn == 4)
    {
        if (depth == CV_8U)
            filter_<Vec4b>(src, dst, dDepth);
        if (depth == CV_32F)
            filter_<Vec4f>(src, dst, dDepth);
    }
}

void DTFilterCPU::setSingleFilterCall(bool value)
{
    singleFilterCall = value;
}

void DTFilterCPU::release()
{
    if (mode == -1) return;

    idistHor.release();
    idistVert.release();

    distHor.release();
    distVert.release();

    a0distHor.release();
    a0distVert.release();

    adistHor.release();
    adistVert.release();
}

Mat DTFilterCPU::getWExtendedMat(int h, int w, int type, int brdleft /*= 0*/, int brdRight /*= 0*/, int cacheAlign /*= 0*/)
{
    int wrapperCols = w + brdleft + brdRight;
    if (cacheAlign > 0)
        wrapperCols += ((wrapperCols + cacheAlign-1) / cacheAlign)*cacheAlign;
    Mat mat(h, wrapperCols, type);
    return mat(Range::all(), Range(brdleft, w + brdleft));
}


Range DTFilterCPU::getWorkRangeByThread(const Range& itemsRange, const Range& rangeThread, int declaredNumThreads)
{
    if (declaredNumThreads <= 0)
        declaredNumThreads = cv::getNumThreads();

    int chunk   = itemsRange.size() / declaredNumThreads;
    int start   = itemsRange.start + chunk * rangeThread.start;
    int end     = itemsRange.start + ((rangeThread.end >= declaredNumThreads) ? itemsRange.size() : chunk * rangeThread.end);

    return Range(start, end);
}

Range DTFilterCPU::getWorkRangeByThread(int items, const Range& rangeThread, int declaredNumThreads)
{
    return getWorkRangeByThread(Range(0, items), rangeThread, declaredNumThreads);
}

}
}