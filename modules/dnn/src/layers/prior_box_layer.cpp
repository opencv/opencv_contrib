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

void PriorBoxLayer::checkParameter(const LayerParams &params, const string &parameterName)
{
    if (!params.has(parameterName))
    {
        CV_Error(Error::StsBadArg, "PriorBox layer parameter does not contain " + parameterName + " index.");
    }
}

PriorBoxLayer::PriorBoxLayer(LayerParams &params) : Layer(params)
{
    checkParameter(params, "min_size");

    _minSize = params.min_size();
    CV_Assert(_minSize > 0);

    _aspectRatios.clear();
    _aspectRatios.push_back(1.);

    _flip = params.flip();

    for (int i = 0; i < params.aspect_ratio_size(); ++i)
    {
        float aspectRatio = params.aspect_ratio(i);
        bool already_exist = false;

        for (int j = 0; j < _aspectRatios.size(); ++j)
        {
            if (fabs(aspectRatio - _aspectRatios[j]) < 1e-6)
            {
                already_exist = true;
                break;
            }
        }
        if (!already_exist)
        {
            _aspectRatios.push_back(aspectRatio);
            if (_flip)
            {
                _aspectRatios.push_back(1./aspectRatio);
            }
        }
    }

    _numPriors = _aspectRatios.size();

    _maxSize = -1;
    if (params.has(max_size))
    {
        _maxSize = params.max_size();
        CV_Assert(_maxSize > _minSize);

        _numPriors += 1;
    }

    _clip = params.clip();

    int varianceSize = params.variance_size();
    if (varianceSize > 1)
    {
        // Must and only provide 4 variance.
        CV_Assert(varianceSize == 4);

        for (int i = 0; i < varianceSize; ++i)
        {
            float variance = params.variance(i);
            CV_Assert(variance > 0);
            _variance.push_back(variance);
        }
    }
    else
    {
        if (varianceSize == 1)
        {
            float variance = params.variance(0);
            CV_Assert(variance > 0);
            _variance.push_back(variance);
        }
        else
        {
            // Set default to 0.1.
            _variance.push_back(0.1);
        }
    }
}

void PriorBoxLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 2);

    _layerWidth = inputs[0]->width();
    _layerHeight = inputs[0]->height();

    _imageWidth = inputs[1]->width();
    _imageHeight = inputs[1]->height();

    _stepX = static_cast<float>(_imageWidth) / _layerWidth;
    _stepY = static_cast<float>(_imageHeight) / _layerHeight;

    // Since all images in a batch has same height and width, we only need to
    // generate one set of priors which can be shared across all images.
    size_t outNum = 1;
    // 2 channels. First channel stores the mean of each prior coordinate.
    // Second channel stores the variance of each prior coordinate.
    size_t outChannels = 2;
    size_t outHeight = _layerHeight;
    size_t outWidth = _layerWidth * _numPriors * 4;

    _outChannelSize = _layerHeight * _layerWidth * _numPriors * 4;

    outputs[0].create(BlobShape(outNum, outChannels, outHeight, outWidth));
}

void PriorBoxLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    float* outputPtr = outputs[0].ptrf();

    // first prior: aspect_ratio = 1, size = min_size
    _boxWidth = _boxHeight = _minSize;

    int idx = 0;
    for (int h = 0; h < _layerHeight; ++h)
    {
        for (int w = 0; w < _layerWidth; ++w)
        {
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
            for (int r = 0; r < _aspectRatios.size(); ++r)
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
        for (int d = 0; d < _outChannelSize; ++d)
        {
            outputPtr[d] = std::min<float>(std::max<float>(outputPtr[d], 0.), 1.);
        }
    }
    // set the variance.
    outputPtr = outputs[0].ptrf(0, 1);
    if(_variance.size() == 1)
    {
        Mat secondChannel(outputs[0].height(), outputs[0].width(), CV_32F, outputPtr);
        secondChannel.setTo(Scalar(_variance[0]));
    }
    else
    {
        int count = 0;
        for (int h = 0; h < _layerHeight; ++h)
        {
            for (int w = 0; w < _layerWidth; ++w)
            {
                for (int i = 0; i < _numPriors; ++i)
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
}
}
