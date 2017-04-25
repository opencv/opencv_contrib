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
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

static void computeShapeByReshapeMask(const std::vector<int> &srcShape,
                                      const std::vector<int> &maskShape,
                                      Range srcRange /*= Range::all()*/,
                                      std::vector<int>& dstShape)
{
    int srcShapeSize = (int)srcShape.size();
    int maskShapeSize = (int)maskShape.size();

    if (srcRange == Range::all())
        srcRange = Range(0, srcShapeSize);
    else
    {
        int sz = srcRange.size();
        srcRange.start = srcRange.start < 0 ? srcRange.start + srcShapeSize : srcRange.start;
        srcRange.end = srcRange.end == INT_MAX ? srcShapeSize : srcRange.start + sz;
    }

    CV_Assert(0 <= srcRange.start && srcRange.start <= srcRange.end && srcRange.end <= srcShapeSize);
    int dstShapeSize = srcShapeSize - srcRange.size() + maskShapeSize;
    dstShape.resize(dstShapeSize);

    std::copy(srcShape.begin(), srcShape.begin() + srcRange.start, dstShape.begin());
    std::copy(srcShape.begin() + srcRange.end, srcShape.begin() + srcShapeSize, dstShape.begin() + srcRange.start + maskShapeSize);

    int inferDim = -1;
    for (int i = 0; i < maskShapeSize; i++)
    {
        if (maskShape[i] > 0)
        {
            dstShape[srcRange.start + i] = maskShape[i];
        }
        else if (maskShape[i] == 0)
        {
            if (srcRange.start + i >= srcShapeSize)
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

    size_t srcTotal = shapeTotal(srcShape);
    size_t dstTotal = shapeTotal(dstShape);

    if (inferDim != -1)
    {
        if (srcTotal % dstTotal != 0)
            CV_Error(Error::StsBackTrace, "Can't infer a dim denoted by -1");

        dstShape[inferDim] = (int)(srcTotal / dstTotal);
    }
    else
    {
        CV_Assert(srcTotal == dstTotal);
    }
}


class ReshapeLayerImpl : public ReshapeLayer
{
public:
    ReshapeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        int axis = params.get<int>("axis", 0);
        int numAxes = params.get<int>("num_axes", -1);
        enableReordering = params.get<bool>("reorder_dims", false);
        CV_Assert(numAxes >= -1);
        newShapeRange = (numAxes == -1) ? Range(axis, INT_MAX) : Range(axis, axis + numAxes);

        newShapeDesc.clear();
        if (params.has("dim"))
        {
            const DictValue &paramShape = params.get("dim");
            int i, dims = paramShape.size();
            newShapeDesc.resize(dims);
            for (i = 0; i < dims; i++)
                newShapeDesc[i] = paramShape.get<int>(i);
        }
    }

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        outputs.resize(inputs.size());
        outShapes.resize(inputs.size());

        for (size_t i = 0; i < inputs.size(); i++)
        {
            std::vector<int> inputShape(inputs[i]->size.p, inputs[i]->size.p + inputs[i]->dims);
            computeShapeByReshapeMask(inputShape, newShapeDesc, newShapeRange, outShapes[i]);
            outputs[i] = inputs[i]->reshape(1, outShapes[i]);
        }
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        for (size_t i = 0; i < outputs.size(); i++)
        {
            Mat srcBlob = *inputs[i];
            int dims = srcBlob.dims;
            std::vector<int> inputShape(srcBlob.size.p, srcBlob.size.p + dims);
            bool channelsReduced = dims > (int)outShapes[i].size() ||
            (dims == 4 && inputShape[1] > outShapes[i][1]);
            bool performReordering = enableReordering && dims == 4 && channelsReduced;

            if (performReordering)
            {
                Mat reordered_blob(inputShape, srcBlob.type());

                float *dstData = reordered_blob.ptr<float>();
                const float *srcData = srcBlob.ptr<float>();

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

            // TODO: we should not assign srcBlob if performReordering is true.
            outputs[i] = srcBlob.reshape(1, outShapes[i]);
        }
    }

    std::vector<std::vector<int> > outShapes;
    bool enableReordering;
};

Ptr<ReshapeLayer> ReshapeLayer::create(const LayerParams& params)
{
    return Ptr<ReshapeLayer>(new ReshapeLayerImpl(params));
}


}
}
