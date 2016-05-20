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

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "lrn_layer.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

namespace cv
{
namespace dnn
{
    LRNLayer::LRNLayer(LayerParams &params) : Layer(params)
    {
        String nrmType = params.get<String>("norm_region", "ACROSS_CHANNELS");
        if (nrmType == "ACROSS_CHANNELS")
            type = CHANNEL_NRM;
        else if (nrmType == "WITHIN_CHANNEL")
            type = SPATIAL_NRM;
        else
            CV_Error(Error::StsBadArg, "Unknown region type \"" + nrmType + "\"");

        size = params.get<int>("local_size", 5);
        if (size % 2 != 1 || size <= 0)
            CV_Error(Error::StsBadArg, "LRN layer supports only positive odd values for local_size");

        alpha = params.get<double>("alpha", 1);
        beta = params.get<double>("beta", 0.75);
    }

    void LRNLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(inputs.size() == 1);
        outputs.resize(1);

        Vec4i shape = inputs[0]->shape4();
        outputs[0].create(shape);

        shape[0] = 1; //maybe make shape[0] = 1 too
        bufBlob.create(shape);
    }

    void LRNLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        Blob &src = *inputs[0];
        Blob &dst = outputs[0];

        switch (type)
        {
        case CHANNEL_NRM:
            channelNoramlization(src, dst);
            break;
        case SPATIAL_NRM:
            spatialNormalization(src, dst);
            break;
        default:
            CV_Error(cv::Error::StsNotImplemented, "Unimplemented mode of LRN layer");
            break;
        }
    }

    void LRNLayer::channelNoramlization(Blob &srcBlob, Blob &dstBlob)
    {
        CV_DbgAssert(srcBlob.ptr() != dstBlob.ptr());

        int num = srcBlob.num();
        int channels = srcBlob.channels();
        int ksize = (size - 1) / 2;

        for (int n = 0; n < num; n++)
        {
            Mat accum = dstBlob.getPlane(n, channels-1); //trick for memory saving
            accum.setTo(0);

            for (int cn = 0; cn < std::min(ksize, channels); cn++)
                cv::accumulateSquare(srcBlob.getPlane(n, cn), accum);

            for (int cn = 0; cn < channels; cn++)
            {
                if (cn + ksize < channels)
                {
                    cv::accumulateSquare(srcBlob.getPlane(n, cn + ksize), accum);
                }

                if (cn - ksize - 1 >= 0)
                {
                    Mat left = srcBlob.getPlane(n, cn - ksize - 1);
                    cv::subtract(accum, left.mul(left), accum); //subtractSquare
                }

                Mat dst = dstBlob.getPlane(n, cn);
                accum.convertTo(dst, dst.type(), alpha/size, 1);
                cv::pow(dst, beta, dst);
                cv::divide(srcBlob.getPlane(n, cn), dst, dst);
            }
        }
    }

    void LRNLayer::spatialNormalization(Blob &srcBlob, Blob &dstBlob)
    {
        int num = srcBlob.num();
        int channels = srcBlob.channels();

        for (int n = 0; n < num; n++)
        {
            for (int cn = 0; cn < channels; cn++)
            {
                Mat src = srcBlob.getPlane(n, cn);
                Mat dst = dstBlob.getPlane(n, cn);
                uchar *dataDst0 = dst.data;

                cv::pow(srcBlob.getPlane(n, cn), 2, dst);
                //TODO: check border type
                cv::boxFilter(dst, dst, dst.depth(), cv::Size(size, size), cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
                dst.convertTo(dst, dst.type(), alpha/(size*size), 1);
                cv::pow(dst, beta, dst);
                cv::divide(src, dst, dst);

                CV_Assert(dataDst0 == dst.data); //debug
            }
        }
    }

}
}
