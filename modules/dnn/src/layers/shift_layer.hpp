// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Declaration of shift layer, which adds up const values to blob.
*/

#ifndef __OPENCV_DNN_LAYERS_SHIFT_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_SHIFT_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{

class ShiftLayerImpl;

class ShiftLayer : public Layer
{
    cv::Ptr<ShiftLayerImpl> impl;

public:
    ShiftLayer() {}
    ShiftLayer(LayerParams &params);
    void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
};

}
}
#endif
