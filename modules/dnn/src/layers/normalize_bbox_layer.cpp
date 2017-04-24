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
#include "op_blas.hpp"

#include <float.h>
#include <algorithm>

namespace cv
{
namespace dnn
{

class NormalizeBBoxLayerImpl : public NormalizeBBoxLayer
{
public:
    Mat _buffer;

    Mat _sumChannelMultiplier;
    Mat _sumSpatialMultiplier;

    Mat _scale;

    float _eps;
    bool _across_spatial;
    bool _channel_shared;

    size_t _num;
    size_t _channels;
    size_t _rows;
    size_t _cols;

    size_t _channelSize;
    size_t _imageSize;

    static const size_t _numAxes = 4;
    static const std::string _layerName;

    bool getParameterDict(const LayerParams &params,
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
    T getParameter(const LayerParams &params,
                   const std::string &parameterName,
                   const size_t &idx=0,
                   const bool required=true,
                   const T& defaultValue=T())
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

    NormalizeBBoxLayerImpl(const LayerParams &params)
    {
        _eps = getParameter<float>(params, "eps", 0, false, 1e-10f);
        _across_spatial = getParameter<bool>(params, "across_spatial");
        _channel_shared = getParameter<bool>(params, "channel_shared");
        setParamsFrom(params);
    }

    void checkInputs(const std::vector<Mat*> &inputs)
    {
        CV_Assert(inputs.size() > 0);
        for (size_t i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->size == inputs[0]->size);
        }
        CV_Assert(inputs[0]->dims > 2);
    }

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        checkInputs(inputs);

        const Mat& inp0 = *inputs[0];
        CV_Assert(inp0.dims == 4 && inp0.type() == CV_32F);

        _num = inp0.size[0];
        _channels = inp0.size[1];
        _rows = inp0.size[2];
        _cols = inp0.size[3];

        _channelSize = _rows * _cols;
        _imageSize = _channelSize * _channels;

        _buffer = Mat(_channels, _channelSize, CV_32F);

        _sumChannelMultiplier = Mat(_channels, 1, CV_32F, Scalar(1.0));
        _sumSpatialMultiplier = Mat(1, _channelSize, CV_32F, Scalar(1.0));

        _scale = blobs[0];
        size_t i, ninputs = inputs.size();
        outputs.resize(ninputs);

        for(i = 0; i < ninputs; i++)
        {
            outputs[i].create(inp0.dims, inp0.size.p, inp0.type());
        }
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        Mat zeroBuffer(_channels, _channelSize, CV_32F, Scalar(0));
        Mat absDiff;

        for (size_t j = 0; j < inputs.size(); j++)
        {
            for (size_t n = 0; n < _num; ++n)
            {
                Mat src = Mat(_channels, _channelSize, CV_32F, inputs[j]->ptr<float>(n));
                Mat dst = Mat(_channels, _channelSize, CV_32F, outputs[j].ptr<float>(n));

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
                    dst *= _scale.at<float>(0, 0);
                }
                else
                {
                    // _scale: _channels x 1
                    // _channels x 1 * 1 x _channelSize -> _channels x _channelSize
                    gemmCPU(_scale, _sumSpatialMultiplier, 1, _buffer, 0);

                    dst = dst.mul(_buffer);
                }
            }
        }
    }

};

const std::string NormalizeBBoxLayerImpl::_layerName = std::string("NormalizeBBox");

Ptr<NormalizeBBoxLayer> NormalizeBBoxLayer::create(const LayerParams &params)
{
    return Ptr<NormalizeBBoxLayer>(new NormalizeBBoxLayerImpl(params));
}

}
}
