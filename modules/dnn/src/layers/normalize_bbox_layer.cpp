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
#include "normalize_bbox_layer.hpp"
#include "op_blas.hpp"

#include <float.h>
#include <algorithm>

namespace cv
{
namespace dnn
{

const std::string NormalizeBBoxLayer::_layerName = std::string("NormalizeBBox");

DictValue NormalizeBBoxLayer::getParameterDict(const LayerParams &params,
                                               const std::string &parameterName)
{
    if (!params.has(parameterName))
    {
        std::string message = _layerName;
        message += " layer parameter does not contain ";
        message += parameterName;
        message += " index.";
        CV_Error(Error::StsBadArg, message);
    }

    DictValue parameter = params.get(parameterName);
    if(parameter.size() != 1)
    {
        std::string message = parameterName;
        message += " field in ";
        message += _layerName;
        message += " layer parameter is required";
        CV_Error(Error::StsBadArg, message);
    }

    return parameter;
}

template<typename T>
T NormalizeBBoxLayer::getParameter(const LayerParams &params,
                                   const std::string &parameterName,
                                   const size_t &idx)
{
    return getParameterDict(params, parameterName).get<T>(idx);
}


NormalizeBBoxLayer::NormalizeBBoxLayer(LayerParams &params) : Layer(params)
{
    _eps = getParameter<float>(params, "eps");
    _across_spatial = getParameter<bool>(params, "across_spatial");
    _channel_shared = getParameter<bool>(params, "channel_shared");
}

void NormalizeBBoxLayer::checkInputs(const std::vector<Blob*> &inputs)
{
    CV_Assert(inputs.size() > 0);
    for (size_t i = 0; i < inputs.size(); i++)
    {
        for (size_t j = 0; j < _numAxes; j++)
        {
            CV_Assert(inputs[i]->shape()[j] == inputs[0]->shape()[j]);
        }
    }
    CV_Assert(inputs[0]->dims() > 2);
}

void NormalizeBBoxLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    checkInputs(inputs);

    _num = inputs[0]->num();
    _channels = inputs[0]->shape()[1];
    _rows = inputs[0]->shape()[2];
    _cols = inputs[0]->shape()[3];

    _channelSize = _rows * _cols;
    _imageSize = _channelSize * _channels;

    _buffer = Blob(BlobShape(1, _channels, _rows, _cols));
    _buffer_channel = Blob(BlobShape(1, _channels, 1, 1));
    _buffer_spatial = Blob(BlobShape(1, 1, _rows, _cols));

    if (_across_spatial)
    {
        _norm = Blob(BlobShape(_num, 1, 1, 1));
    }
    else
    {
        _norm = Blob(BlobShape(_num, 1, _rows, _cols));
    }

    // add eps to avoid overflow
    _norm.matRef() = Scalar(_eps);

    _sumChannelMultiplier = Blob(BlobShape(1, _channels, 1, 1));
    _sumChannelMultiplier.matRef() = Scalar(1.0);

    _sumSpatialMultiplier = Blob(BlobShape(1, 1, _rows, _cols));
    _sumSpatialMultiplier.matRef() = Scalar(1.0);

    if (_channel_shared)
    {
        _scale = Blob(BlobShape(1, 1, 1, 1));
    }
    else
    {
        _scale = Blob(BlobShape(1, 1, 1, _channels));
    }

    for(size_t i = 0; i < inputs.size(); i++)
    {
        outputs[i].create(BlobShape(inputs[0]->shape()));
    }
}

void NormalizeBBoxLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    Mat zeroBuffer = Mat(_buffer.matRef().rows, _buffer.matRef().cols,
                         _buffer.matRef().type(), Scalar(0));
    Mat sumAbs;

    for (size_t j = 0; j < inputs.size(); j++)
    {
        for (size_t n = 0; n < _num; ++n)
        {
            Mat src = inputs[j]->getPlanes(n);
            Mat dst = outputs[j].getPlanes(n);

            Mat normCurrent = _norm.getPlanes(n);

            cv::sqrt(src, _buffer.matRef());

            if (_across_spatial)
            {
                absdiff(_buffer.matRef(), zeroBuffer, sumAbs);
                // add eps to avoid overflow
                sumAbs += _eps;

                pow(sumAbs, 0.5f, normCurrent);

                dst = src / normCurrent;
            }
            else
            {
                gemmCPU(_buffer.matRef(), _sumChannelMultiplier.matRef(), 1, normCurrent, GEMM_1_T & GEMM_2_T);

                // compute norm
                pow(normCurrent, 0.5f, normCurrent);

                // scale the layer
                gemmCPU(_sumChannelMultiplier.matRef(), normCurrent, 1, _buffer.matRef(), 0);

                dst = src / _buffer.matRef().at<float>(0, 0);
            }

            // scale the output
            if (_channel_shared)
            {
                dst *= _scale.matRef();
            }
            else
            {
                gemmCPU(_scale.matRef(), _sumSpatialMultiplier.matRef(), 1, _buffer.matRef(), 0);
                dst *= _buffer.matRef();
            }
        }
    }
}
}
}
