// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Declaration of scale layer, which multiplies and shifts channels in input blob.
*/


#ifndef __OPENCV_DNN_LAYERS_SCALE_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_SCALE_LAYER_HPP__
#include <opencv2/dnn/all_layers.hpp>

namespace cv
{
namespace dnn
{

class ScaleLayerImpl : public ScaleLayer
{
public:
    ScaleLayerImpl(bool hasBias_): hasBias(hasBias_) {}
    void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);

private:
    bool hasBias;
};

}
}

#endif // __OPENCV_DNN_LAYERS_SCALE_LAYER_HPP__
