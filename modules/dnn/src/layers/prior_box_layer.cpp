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
#include "prior_box_layer.hpp"
#include <float.h>
#include <algorithm>
#include <cmath>

namespace cv
{
namespace dnn
{

PriorBoxLayerImpl::PriorBoxLayerImpl(const float minSize, const float maxSize,
                                     const std::vector<float>& aspectRatios,
                                     const std::vector<float>& variance,
                                     const bool flip, const bool clip,
                                     const size_t numPriors) :
    _minSize(minSize),
    _maxSize(maxSize),
    _aspectRatios(aspectRatios),
    _variance(variance),
    _flip(flip),
    _clip(clip),
    _numPriors(numPriors) {}

void PriorBoxLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 2);

    _layerWidth = inputs[0]->cols();
    _layerHeight = inputs[0]->rows();

    _imageWidth = inputs[1]->cols();
    _imageHeight = inputs[1]->rows();

    _stepX = static_cast<float>(_imageWidth) / _layerWidth;
    _stepY = static_cast<float>(_imageHeight) / _layerHeight;

    // Since all images in a batch has same height and width, we only need to
    // generate one set of priors which can be shared across all images.
    size_t outNum = 1;
    // 2 channels. First channel stores the mean of each prior coordinate.
    // Second channel stores the variance of each prior coordinate.
    size_t outChannels = 2;
    _outChannelSize = _layerHeight * _layerWidth * _numPriors * 4;

    outputs[0].create(BlobShape(outNum, outChannels, _outChannelSize));
    outputs[0].matRef() = 0;
}

void PriorBoxLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    (void)inputs; // to suppress unused parameter warning

    float* outputPtr = outputs[0].ptrf();

    // first prior: aspect_ratio = 1, size = min_size
    int idx = 0;
    for (size_t h = 0; h < _layerHeight; ++h)
    {
        for (size_t w = 0; w < _layerWidth; ++w)
        {
            _boxWidth = _boxHeight = _minSize;

            float center_x = (w + 0.5) * _stepX;
            float center_y = (h + 0.5) * _stepY;
            // xmin
            outputPtr[idx++] = (center_x - _boxWidth / 2.) / _imageWidth;
            // ymin
            outputPtr[idx++] = (center_y - _boxHeight / 2.) / _imageHeight;
            // xmax
            outputPtr[idx++] = (center_x + _boxWidth / 2.) / _imageWidth;
            // ymax
            outputPtr[idx++] = (center_y + _boxHeight / 2.) / _imageHeight;

            if (_maxSize > 0)
            {
                // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                _boxWidth = _boxHeight = sqrt(_minSize * _maxSize);
                // xmin
                outputPtr[idx++] = (center_x - _boxWidth / 2.) / _imageWidth;
                // ymin
                outputPtr[idx++] = (center_y - _boxHeight / 2.) / _imageHeight;
                // xmax
                outputPtr[idx++] = (center_x + _boxWidth / 2.) / _imageWidth;
                // ymax
                outputPtr[idx++] = (center_y + _boxHeight / 2.) / _imageHeight;
            }

            // rest of priors
            for (size_t r = 0; r < _aspectRatios.size(); ++r)
            {
                float ar = _aspectRatios[r];
                if (fabs(ar - 1.) < 1e-6)
                {
                    continue;
                }
                _boxWidth = _minSize * sqrt(ar);
                _boxHeight = _minSize / sqrt(ar);
                // xmin
                outputPtr[idx++] = (center_x - _boxWidth / 2.) / _imageWidth;
                // ymin
                outputPtr[idx++] = (center_y - _boxHeight / 2.) / _imageHeight;
                // xmax
                outputPtr[idx++] = (center_x + _boxWidth / 2.) / _imageWidth;
                // ymax
                outputPtr[idx++] = (center_y + _boxHeight / 2.) / _imageHeight;
            }
        }
    }
    // clip the prior's coordidate such that it is within [0, 1]
    if (_clip)
    {
        for (size_t d = 0; d < _outChannelSize; ++d)
        {
            outputPtr[d] = std::min<float>(std::max<float>(outputPtr[d], 0.), 1.);
        }
    }
    // set the variance.
    outputPtr = outputs[0].ptrf(0, 1);
    if(_variance.size() == 1)
    {
        Mat secondChannel(outputs[0].rows(), outputs[0].cols(), CV_32F, outputPtr);
        secondChannel.setTo(Scalar(_variance[0]));
    }
    else
    {
        int count = 0;
        for (size_t h = 0; h < _layerHeight; ++h)
        {
            for (size_t w = 0; w < _layerWidth; ++w)
            {
                for (size_t i = 0; i < _numPriors; ++i)
                {
                    for (int j = 0; j < 4; ++j)
                    {
                        outputPtr[count] = _variance[j];
                        ++count;
                    }
                }
            }
        }
    }
}

Ptr<PriorBoxLayer> PriorBoxLayer::create(const float minSize, const float maxSize,
                                     const std::vector<float>& aspectRatios,
                                     const std::vector<float>& variance,
                                     const bool flip, const bool clip,
                                     const size_t numPriors)
{
    return Ptr<PriorBoxLayer>(new PriorBoxLayerImpl(minSize, maxSize,
                                                    aspectRatios, variance,
                                                    flip, clip, numPriors));
}
}
}
