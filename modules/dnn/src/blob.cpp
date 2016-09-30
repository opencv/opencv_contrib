/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

Blob::Blob()
{
    CV_DNN_UMAT_ONLY(state = UNINITIALIZED);
}

Blob::Blob(const BlobShape &shape, int type, int allocFlags)
{
    CV_DNN_UMAT_ONLY(state = UNINITIALIZED);
    this->create(shape, type, allocFlags);
}

Blob::Blob(InputArray data)
{
#ifndef CV_DNN_UMAT
    m = data.getMat();
#else
    if (data.isUMat())
    {
        um = data.getUMat();
        state = HEAD_AT_UMAT;
    }
    else
    {
        m = data.getMat();
        state = HEAD_AT_MAT;
    }
#endif
}

void Blob::create(const BlobShape &shape, int type, int allocFlags)
{
#ifndef CV_DNN_UMAT
    CV_Assert(allocFlags & ALLOC_MAT);
    m.create(shape.dims(), shape.ptr(), type);
#else
    CV_Assert(allocFlags & ALLOC_MAT || allocFlags & ALLOC_UMAT);

    if (allocFlags & ALLOC_MAT)
        m.create(shape.dims(), shape.ptr(), type);
    if (allocFlags & ALLOC_UMAT)
        um.create(shape.dims(), shape.ptr(), type);

    if (state == UNINITIALIZED)
    {
        if (allocFlags & ALLOC_MAT && allocFlags & ALLOC_UMAT)
            state = SYNCED;
        else if (allocFlags & ALLOC_MAT)
            state = HEAD_AT_MAT;
        else
            state = HEAD_AT_UMAT;
    }
#endif
}

void Blob::fill(InputArray in)
{
#ifdef CV_DNN_UMAT
    CV_Assert(in.isMat() || in.isUMat());
    if (in.isMat())
    {
        m = in.getMat();
        state = HEAD_AT_MAT;
    }
    else
    {
        um = in.getUMat();
        state = HEAD_AT_UMAT;
    }
#else
    CV_Assert(in.isMat());
    m = in.getMat();
#endif
}

static inline int getMatChannels(const Mat &mat)
{
    return (mat.dims <= 2) ? mat.channels() : mat.size[0];
}

static BlobShape getBlobShape(std::vector<Mat> &vmat, int requestedCn = -1)
{
    BlobShape shape(BlobShape::all(4));
    int cnSum = 0, matCn;

    CV_Assert(vmat.size() > 0);

    for (size_t i = 0; i < vmat.size(); i++)
    {
        Mat &mat = vmat[i];
        CV_Assert(!mat.empty());
        CV_Assert((mat.dims == 3 && mat.channels() == 1) || mat.dims <= 2);

        matCn = getMatChannels(mat);
        cnSum += getMatChannels(mat);

        if (i == 0)
        {
            shape[-1] = mat.cols;
            shape[-2] = mat.rows;
            shape[-3] = (requestedCn <= 0) ? matCn : requestedCn;
        }
        else
        {
            if (mat.cols != shape[-1] || mat.rows != shape[-2])
                CV_Error(Error::StsError, "Each Mat.size() must be equal");

            if (requestedCn <= 0 && matCn != shape[-3])
                CV_Error(Error::StsError, "Each Mat.chnannels() (or number of planes) must be equal");
        }
    }

    if (cnSum % shape[-3] != 0)
        CV_Error(Error::StsError, "Total number of channels in vector is not a multiple of requsted channel number");

    shape[0] = cnSum / shape[-3];
    return shape;
}

static std::vector<Mat> extractMatVector(InputArray in)
{
    if (in.isMat() || in.isUMat())
    {
        return std::vector<Mat>(1, in.getMat());
    }
    else if (in.isMatVector())
    {
        return *static_cast<const std::vector<Mat>*>(in.getObj());
    }
    else if (in.isUMatVector())
    {
        std::vector<Mat> vmat;
        in.getMatVector(vmat);
        return vmat;
    }
    else
    {
        CV_Assert(in.isMat() || in.isMatVector() || in.isUMat() || in.isUMatVector());
        return std::vector<Mat>();
    }
}

void Blob::batchFromImages(InputArray image, int dstCn)
{
    CV_Assert(dstCn == -1 || dstCn > 0);
    std::vector<Mat> inMats = extractMatVector(image);
    BlobShape dstShape = getBlobShape(inMats, dstCn);

    int dtype = CV_32F;
    this->create(dstShape, dtype, ALLOC_MAT);
    uchar *dstPtr = this->matRef().ptr();
    int elemSize = CV_ELEM_SIZE(dtype);

    std::vector<Mat> wrapBuf(dstShape[-3]);
    for (size_t i = 0; i < inMats.size(); i++)
    {
        Mat inMat = inMats[i];

        if (inMat.dims <= 2)
        {
            inMat.convertTo(inMat, dtype);

            wrapBuf.resize(0);
            for (int cn = 0; cn < inMat.channels(); cn++)
            {
                wrapBuf.push_back(Mat(inMat.rows, inMat.cols, dtype, dstPtr));
                dstPtr += elemSize * inMat.total();
            }

            cv::split(inMat, wrapBuf);
        }
        else
        {
            inMat.convertTo(Mat(inMat.dims, inMat.size, dtype, dstPtr), dtype);
            dstPtr += elemSize * inMat.total();
        }
    }
}

Blob Blob::fromImages(InputArray image, int dstCn)
{
    Blob res;
    res.batchFromImages(image, dstCn);
    return res;
}

void Blob::fill(const BlobShape &shape, int type, void *data, bool deepCopy)
{
    if (deepCopy)
    {
        create(shape, type);
        memcpy(ptr(), data, this->total() * CV_ELEM_SIZE(type));
    }
    else
    {
        m = Mat(shape.dims(), shape.ptr(), type, data);
    }
    CV_DNN_UMAT_ONLY(state = HEAD_AT_MAT);
}

void Blob::setTo(InputArray value, int allocFlags)
{
#ifdef CV_DNN_UMAT
    if (allocFlags == -1)
    {
        if (state == HEAD_AT_UMAT)
            um.setTo(value);
        else if (state == HEAD_AT_MAT)
            m.setTo(value);
        else //SYNCED or UNINITIALIZED
        {
            um.setTo(value);
            m.setTo(value);

            if (state == UNINITIALIZED)
                state = SYNCED;
        }
    }
    else if (allocFlags == ALLOC_BOTH)
    {
        m.setTo(value);
        um.setTo(value);
        state = SYNCED;
    }
    else if (allocFlags == ALLOC_MAT)
    {
        matRef().setTo(value);
    }
    else if (allocFlags == ALLOC_UMAT)
    {
        umatRef().setTo(value);
    }
    else
    {
        CV_Error(Error::StsBadArg, "allocFlags sholud be -1 or one of Blob::AllocFlag values");
    }
#else
    m.setTo(value);
#endif
}

void Blob::updateMat(bool syncData) const
{
#ifdef CV_DNN_UMAT
    if (state == UNINITIALIZED || state == SYNCED || state == HEAD_AT_MAT)
    {
        return;
    }
    else if (state == HEAD_AT_UMAT)
    {
        if (syncData)
            um.copyTo(m);
        else
            m.create(dims(), sizes(), type());
        state = SYNCED;
    }
    else
    {
        CV_Error(Error::StsInternal, "");
    }
#else
    (void)syncData;
#endif
}

void Blob::updateUMat(bool syncData) const
{
#ifdef CV_DNN_UMAT
    if (state == UNINITIALIZED || state == SYNCED || state == HEAD_AT_UMAT)
    {
        return;
    }
    else if (state == HEAD_AT_MAT)
    {
        if (syncData)
            m.copyTo(um);
        else
            um.create(dims(), sizes(), type());
    }
    else
    {
        CV_Error(Error::StsInternal, "");
    }
#else
    (void)syncData;
#endif
}

void Blob::sync() const
{
    updateMat();
    updateUMat();
}

Vec4i Blob::shape4() const
{
    return Vec4i(num(), channels(), rows(), cols());
}

//BlobShape

std::ostream &operator<< (std::ostream &stream, const BlobShape &shape)
{
    stream << "[";

    for (int i = 0; i < shape.dims() - 1; i++)
        stream << shape[i] << ", ";
    if (shape.dims() > 0)
        stream << shape[-1];

    return stream << "]";
}

BlobShape computeShapeByReshapeMask(const BlobShape &srcShape, const BlobShape &maskShape, Range srcRange /*= Range::all()*/)
{
    if (srcRange == Range::all())
        srcRange = Range(0, srcShape.dims());
    else
    {
        int sz = srcRange.size();
        srcRange.start = srcShape.canonicalAxis(srcRange.start);
        srcRange.end =  (srcRange.end == INT_MAX) ? srcShape.dims() : srcRange.start + sz;
    }

    CV_Assert(0 <= srcRange.start && srcRange.start <= srcRange.end && srcRange.end <= srcShape.dims());
    BlobShape dstShape(srcShape.dims() - srcRange.size() + maskShape.dims(), (const int*)NULL);

    std::copy(srcShape.ptr(), srcShape.ptr() + srcRange.start, dstShape.ptr());
    std::copy(srcShape.ptr() + srcRange.end, srcShape.ptr() + srcShape.dims(), dstShape.ptr() + srcRange.start + maskShape.dims());

    int inferDim = -1;
    for (int i = 0; i < maskShape.dims(); i++)
    {
        if (maskShape[i] > 0)
        {
            dstShape[srcRange.start + i] = maskShape[i];
        }
        else if (maskShape[i] == 0)
        {
            if (srcRange.start + i >= srcShape.dims())
                CV_Error(Error::StsBadArg, format("Copy dim[%d] (which has zero size) is out of the source shape bounds", srcRange.start + i));
            dstShape[srcRange.start + i] = srcShape[srcRange.start + i];
        }
        else if (maskShape[i] == -1)
        {
            if (inferDim != -1)
                CV_Error(Error::StsAssert, "Duplicate of inferred dim (which is denoted by -1)");
            inferDim = srcRange.start + i;
            dstShape[inferDim] = 1;
        }
        else
            CV_Error(Error::StsBadArg, "maskShape[i] >= -1");
    }

    if (inferDim != -1)
    {
        ptrdiff_t srcTotal = srcShape.total();
        ptrdiff_t dstTotal = dstShape.total();
        if (srcTotal % dstTotal != 0)
            CV_Error(Error::StsBackTrace, "Can't infer a dim denoted by -1");

        dstShape[inferDim] = (int)(srcTotal / dstTotal);
    }
    else
    {
        CV_Assert(srcShape.total() == dstShape.total());
    }

    return dstShape;
}

}
}
