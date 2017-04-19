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
    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs);
    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs);
    int getPadDim(const std::vector<int>& shape) const;

    int paddingDim, padding, inputDims, index;
    float paddingValue;
};

}
}
#endif
