// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Batch Normalization layer.
*/

#include "max_unpooling_layer.hpp"

namespace cv
{
namespace dnn
{

MaxUnpoolLayerImpl::MaxUnpoolLayerImpl(Size outSize_):
    outSize(outSize_)
{}

void MaxUnpoolLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 2);

    BlobShape outShape = inputs[0]->shape();
    outShape[2] = outSize.height;
    outShape[3] = outSize.width;

    outputs.resize(1);
    outputs[0].create(outShape);
}

void MaxUnpoolLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 2);
    Blob& input = *inputs[0];
    Blob& indices = *inputs[1];

    CV_Assert(input.total() == indices.total());
    CV_Assert(input.num() == 1);

    for(int i_n = 0; i_n < outputs.size(); i_n++)
    {
        Blob& outBlob = outputs[i_n];
        CV_Assert(input.channels() == outBlob.channels());

        for (int i_c = 0; i_c < input.channels(); i_c++)
        {
            Mat outPlane = outBlob.getPlane(0, i_c);
            for(int i_wh = 0; i_wh < input.size2().area(); i_wh++)
            {
                int index = indices.getPlane(0, i_c).at<float>(i_wh);

                CV_Assert(index < outPlane.total());
                outPlane.at<float>(index) = input.getPlane(0, i_c).at<float>(i_wh);
            }
        }
    }
}

Ptr<MaxUnpoolLayer> MaxUnpoolLayer::create(Size unpoolSize)
{
    return Ptr<MaxUnpoolLayer>(new MaxUnpoolLayerImpl(unpoolSize));
}

}
}
