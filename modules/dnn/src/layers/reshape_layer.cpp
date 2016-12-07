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
#include "reshape_layer.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

ReshapeLayerImpl::ReshapeLayerImpl(const BlobShape &newShape_, Range applyingRange_, bool enableReordering_) :
    enableReordering(enableReordering_)
{
    newShapeDesc = newShape_;
    newShapeRange = applyingRange_;
}

void ReshapeLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    outputs.resize(inputs.size());
    outShapes.resize(inputs.size());

    for (size_t i = 0; i < inputs.size(); i++)
    {
        outShapes[i] = computeShapeByReshapeMask(inputs[i]->shape(), newShapeDesc, newShapeRange);
        outputs[i].shareFrom(*inputs[i]);
        outputs[i].reshape(outShapes[i]);
    }
}

void ReshapeLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    for (size_t i = 0; i < outputs.size(); i++)
    {
        Blob& srcBlob = *inputs[i];
        BlobShape inputShape = inputs[i]->shape();
        bool channelsReduced = inputShape.dims() > outShapes[i].dims() ||
                (inputShape.dims() == 4 && inputShape[1] > outShapes[i][1]);
        bool performReordering = enableReordering && inputShape.dims() == 4 && channelsReduced;

        if (performReordering)
        {
            Blob reordered_blob(inputShape, inputs[i]->type());

            float *dstData = reordered_blob.matRef().ptr<float>();
            const float *srcData = srcBlob.matRefConst().ptr<float>();

            int num = inputShape[0], channels = inputShape[1], height = inputShape[2], width = inputShape[3];
            int total = num*channels*height*width;
            for(int i_n = 0; i_n < num; i_n++) {
                for(int i_c = 0; i_c < channels; i_c++) {
                    for(int i_h = 0; i_h < height; i_h++) {
                        for(int i_w = 0; i_w < width; i_w++) {
                           int src_i = channels*height*width*i_n + height*width*i_c + width*i_h + i_w;
                           int dst_i = channels*height*width*i_n + i_c + channels*width*i_h + channels*i_w;

                           CV_Assert(dst_i < total);
                           CV_Assert(src_i < total);

                           dstData[dst_i] = srcData[src_i];
                        }
                    }
                }
            }

            srcBlob = reordered_blob;
        }

        outputs[i].shareFrom(srcBlob);
        outputs[i].reshape(outShapes[i]);
    }
}

Ptr<ReshapeLayer> ReshapeLayer::create(const BlobShape &newShape, Range applyingRange /*= Range::all()*/,
                                       bool enableReordering /*= false*/)
{
    return Ptr<ReshapeLayer>(new ReshapeLayerImpl(newShape, applyingRange, enableReordering));
}


}
}
