// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of padding layer, which adds paddings to input blob.
*/

#include "padding_layer.hpp"
#include <vector>

namespace cv
{
namespace dnn
{

PaddingLayer::PaddingLayer(LayerParams &params)
{
    paddingDim = params.get<int>("padding_dim");
    padding = abs(params.get<int>("padding"));
    inputDims = params.get<int>("input_dims", 0);
    index = params.get<int>("index", 0);
    paddingValue = params.get<double>("value", 0);

    if(paddingDim < 0 || padding < 0)
        CV_Error(cv::Error::StsNotImplemented, "Negative padding and dim aren't supported");
}

void PaddingLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    outputs.resize(inputs.size());
    for(int i = 0; i < inputs.size(); i++)
    {
        BlobShape shape = inputs[i]->shape();
        int dim = getPadDim(shape);
        CV_Assert(dim < shape.dims());

        shape[dim] += padding;
        outputs[i].create(shape);
    }
}

void PaddingLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    for(int i = 0; i < inputs.size(); i++)
    {
        outputs[i].matRef() = paddingValue;
        BlobShape inShape = inputs[i]->shape();
        BlobShape outShape = outputs[i].shape();
        int dim = getPadDim(inShape);

        int actualIndex = index;
        if(index == 0)
            actualIndex = inShape[dim];

        std::vector<std::pair<Range, Range> > srcDstRanges;
        srcDstRanges.push_back(std::make_pair(Range(0, actualIndex), Range(0, actualIndex)));
        srcDstRanges.push_back(std::make_pair(Range(actualIndex, inShape[dim]),
                                              Range(actualIndex + padding, outShape[dim])));

        std::vector<Range> srcRanges(inShape.dims(), Range::all()), dstRanges = srcRanges;

        for(int j = 0; j < srcDstRanges.size(); j++)
        {
            if(!srcDstRanges[j].first.empty())
            {
                srcRanges[dim] = srcDstRanges[j].first;
                dstRanges[dim] = srcDstRanges[j].second;
                Mat dst = outputs[i].matRef()(&dstRanges[0]);
                Mat src = inputs[i]->matRef()(&srcRanges[0]).clone();
                src.copyTo(dst);
            }
        }
    }
}

int PaddingLayer::getPadDim(const BlobShape& shape) const
{
    return inputDims > 0 && shape.dims() > inputDims ? paddingDim + 1 : paddingDim;
}

}
}
