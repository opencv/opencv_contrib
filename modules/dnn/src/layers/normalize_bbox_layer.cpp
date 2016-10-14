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

bool NormalizeBBoxLayer::getParameterDict(const LayerParams &params,
                                          const std::string &parameterName,
                                          DictValue& result)
{
    if (!params.has(parameterName))
    {
        return false;
    }

    result = params.get(parameterName);
    return true;
}

template<typename T>
T NormalizeBBoxLayer::getParameter(const LayerParams &params,
                                   const std::string &parameterName,
                                   const size_t &idx,
                                   const bool required,
                                   const T& defaultValue)
{
    DictValue dictValue;
    bool success = getParameterDict(params, parameterName, dictValue);
    if(!success)
    {
        if(required)
        {
            std::string message = _layerName;
            message += " layer parameter does not contain ";
            message += parameterName;
            message += " parameter.";
            CV_Error(Error::StsBadArg, message);
        }
        else
        {
            return defaultValue;
        }
    }
    return dictValue.get<T>(idx);
}

NormalizeBBoxLayer::NormalizeBBoxLayer(LayerParams &params) : Layer(params)
{
    _eps = getParameter<float>(params, "eps", 0, false, 1e-10f);
    _across_spatial = getParameter<bool>(params, "across_spatial");
    _channel_shared = getParameter<bool>(params, "channel_shared");
}

void NormalizeBBoxLayer::checkInputs(const std::vector<Blob*> &inputs)
{
    CV_Assert(inputs.size() > 0);
    for (size_t i = 1; i < inputs.size(); i++)
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

    _buffer = Mat(_channels, _channelSize, CV_32F);

    _sumChannelMultiplier = Mat(_channels, 1, CV_32F, Scalar(1.0));
    _sumSpatialMultiplier = Mat(1, _channelSize, CV_32F, Scalar(1.0));

    _scale = blobs[0];

    for(size_t i = 0; i < inputs.size(); i++)
    {
        outputs[i].create(BlobShape(inputs[0]->shape()));
    }
}

void NormalizeBBoxLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    Mat zeroBuffer(_channels, _channelSize, CV_32F, Scalar(0));
    Mat absDiff;

    for (size_t j = 0; j < inputs.size(); j++)
    {
        for (size_t n = 0; n < _num; ++n)
        {
            Mat src = Mat(_channels, _channelSize, CV_32F, inputs[j]->ptrf(n));
            Mat dst = Mat(_channels, _channelSize, CV_32F, outputs[j].ptrf(n));

            _buffer = src.mul(src);

            if (_across_spatial)
            {
                absdiff(_buffer, zeroBuffer, absDiff);

                // add eps to avoid overflow
                double absSum = sum(absDiff)[0] + _eps;

                float norm = sqrt(absSum);
                dst = src / norm;
            }
            else
            {
                Mat norm(_channelSize, 1, _buffer.type()); // 1 x _channelSize

                // (_channels x_channelSize)T * _channels x 1 -> _channelSize x 1
                gemmCPU(_buffer, _sumChannelMultiplier, 1, norm, 0, GEMM_1_T);

                // compute norm
                pow(norm, 0.5f, norm);

                // scale the layer
                // _channels x 1 * (_channelSize x 1)T -> _channels x _channelSize
                gemmCPU(_sumChannelMultiplier, norm, 1, _buffer, 0, GEMM_2_T);

                dst = src / _buffer;
            }

            // scale the output
            if (_channel_shared)
            {
                // _scale: 1 x 1
                dst *= _scale.matRefConst().at<float>(0, 0);
            }
            else
            {
                // _scale: _channels x 1
                // _channels x 1 * 1 x _channelSize -> _channels x _channelSize
                gemmCPU(_scale.matRefConst(), _sumSpatialMultiplier, 1, _buffer, 0);

                dst = dst.mul(_buffer);
           }
        }
    }
}
}
}
