// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of padding layer, which adds paddings to input blob.
*/

#include "../precomp.hpp"
#include <vector>

namespace cv
{
namespace dnn
{

class PaddingLayerImpl : public PaddingLayer
{
public:
    PaddingLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        paddingDim = params.get<int>("padding_dim");
        padding = abs(params.get<int>("padding"));
        inputDims = params.get<int>("input_dims", 0);
        index = params.get<int>("index", 0);
        paddingValue = params.get<double>("value", 0);

        if(paddingDim < 0 || padding < 0)
            CV_Error(cv::Error::StsNotImplemented, "Negative padding and dim aren't supported");
    }

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        size_t i, ninputs = inputs.size();
        outputs.resize(ninputs);

        for( i = 0; i < ninputs; i++ )
        {
            const Mat& inp = *inputs[i];
            int dims = inp.dims;
            std::vector<int> shape(inp.size.p, inp.size.p + dims);
            int dim = getPadDim(shape);
            CV_Assert(dim < dims);

            shape[dim] += padding;
            outputs[i].create(dims, &shape[0], inp.type());
        }
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        for(int i = 0; i < inputs.size(); i++)
        {
            outputs[i] = paddingValue;
            const Mat& inp = *inputs[i];
            Mat& out = outputs[i];
            int dims = inp.dims;
            std::vector<int> inShape(inp.size.p, inp.size.p + dims);
            std::vector<int> outShape(out.size.p, out.size.p + dims);
            int dim = getPadDim(inShape);

            int actualIndex = index;
            if(index == 0)
                actualIndex = inShape[dim];

            std::vector<std::pair<Range, Range> > srcDstRanges;
            srcDstRanges.push_back(std::make_pair(Range(0, actualIndex), Range(0, actualIndex)));
            srcDstRanges.push_back(std::make_pair(Range(actualIndex, inShape[dim]),
                                                  Range(actualIndex + padding, outShape[dim])));

            std::vector<Range> srcRanges(dims, Range::all()), dstRanges = srcRanges;

            for(int j = 0; j < srcDstRanges.size(); j++)
            {
                if(!srcDstRanges[j].first.empty())
                {
                    srcRanges[dim] = srcDstRanges[j].first;
                    dstRanges[dim] = srcDstRanges[j].second;
                    Mat dst = out(&dstRanges[0]);
                    Mat src = inp(&srcRanges[0]).clone();
                    src.copyTo(dst);
                }
            }
        }
    }

    int getPadDim(const std::vector<int>& shape) const
    {
        return inputDims > 0 && (int)shape.size() > inputDims ? paddingDim + 1 : paddingDim;
    }

    int paddingDim, padding, inputDims, index;
    float paddingValue;
};

Ptr<PaddingLayer> PaddingLayer::create(const LayerParams &params)
{
    return Ptr<PaddingLayer>(new PaddingLayerImpl(params));
}

}
}
