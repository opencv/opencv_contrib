// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Declaration of padding layer, which adds paddings to input blob.
*/

#ifndef __OPENCV_DNN_LAYERS_PADDING_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_PADDING_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{

class PaddingLayer : public Layer
{
public:
    PaddingLayer() {}
    PaddingLayer(LayerParams &params);
    void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);

private:
    int getPadDim(const BlobShape& shape) const;
    int paddingDim, padding, inputDims, index;
    float paddingValue;
};

}
}
#endif
