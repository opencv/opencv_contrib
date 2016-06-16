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

namespace cv
{
namespace dnn
{

    Blob::Blob()
    {
        int zeros[4] = { 0, 0, 0, 0 };
        m = Mat(4, zeros, CV_32F, NULL);
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

    Blob::Blob(InputArray image, int dstCn)
    {
        CV_Assert(dstCn == -1 || dstCn > 0);
        std::vector<Mat> inMats = extractMatVector(image);
        BlobShape dstShape = getBlobShape(inMats, dstCn);

        m.create(dstShape.dims(), dstShape.ptr(), CV_32F);

        std::vector<Mat> wrapBuf(dstShape[-3]);
        int elemSize = (int)m.elemSize();
        uchar *ptr = this->ptr();
        for (size_t i = 0; i < inMats.size(); i++)
        {
            Mat inMat = inMats[i];

            if (inMat.dims <= 2)
            {
                inMat.convertTo(inMat, m.type());

                wrapBuf.resize(0);
                for (int cn = 0; cn < inMat.channels(); cn++)
                {
                    wrapBuf.push_back(Mat(inMat.rows, inMat.cols, m.type(), ptr));
                    ptr += elemSize * inMat.total();
                }

                cv::split(inMat, wrapBuf);
            }
            else
            {
                inMat.convertTo(Mat(inMat.dims, inMat.size, m.type(), ptr), m.type());
                ptr += elemSize * inMat.total();
            }
        }
    }

    Blob::Blob(const BlobShape &shape, int type)
    {
        this->create(shape, type);
    }

    void Blob::fill(const BlobShape &shape, int type, void *data, bool deepCopy)
    {
        CV_Assert(type == CV_32F || type == CV_64F);

        if (deepCopy)
        {
            m.create(shape.dims(), shape.ptr(), type);
            memcpy(m.data, data, m.total() * m.elemSize());
        }
        else
        {
            m = Mat(shape.dims(), shape.ptr(), type, data);
        }
    }

    void Blob::create(const BlobShape &shape, int type)
    {
        CV_Assert(type == CV_32F || type == CV_64F);
        m.create(shape.dims(), shape.ptr(), type);
    }

    inline void squeezeShape(const int srcDims, const int *srcSizes, const int dstDims, int *dstSizes)
    {
        const int m = std::min(dstDims, srcDims);

        //copy common(last) dimensions
        for (int i = 0; i < m; i++)
            dstSizes[dstDims - 1 - i] = srcSizes[srcDims - 1 - i];

        //either flatten extra dimensions
        for (int i = m; i < srcDims; i++)
            dstSizes[0] *= srcSizes[srcDims - 1 - i];

        //either fill gaps
        for (int i = m; i < dstDims; i++)
            dstSizes[dstDims - 1 - i] = 1;
    }

    Vec4i Blob::shape4() const
    {
        return Vec4i(num(), channels(), rows(), cols());
    }

    std::ostream &operator<< (std::ostream &stream, const BlobShape &shape)
    {
        stream << "[";

        for (int i = 0; i < shape.dims() - 1; i++)
            stream << shape[i] << ", ";
        if (shape.dims() > 0)
            stream << shape[-1];

        return stream << "]";
    }
}
}
