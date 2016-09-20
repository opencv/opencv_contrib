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
#include "flatten_layer.hpp"
#include <float.h>
#include <algorithm>

namespace cv
{
namespace dnn
{

FlattenLayer::FlattenLayer(LayerParams &params) : Layer(params)
{
    _startAxis = params.get<int>("axis", 1);
    _endAxis = params.get<int>("end_axis", -1);
}

void FlattenLayer::checkInputs(const std::vector<Blob*> &inputs)
{
    CV_Assert(inputs.size() > 0);
    for (size_t i = 1; i < inputs.size(); i++)
    {
        for (size_t j = 0; j < _numAxes; j++)
        {
            CV_Assert(inputs[i]->shape()[j] == inputs[0]->shape()[j]);
        }
    }
}

void FlattenLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    checkInputs(inputs);

    _numAxes = inputs[0]->dims();
    _endAxis = inputs[0]->canonicalAxis(_endAxis);
    CV_Assert(_startAxis >= 0);
    CV_Assert(_endAxis >= _startAxis && _endAxis < (int)_numAxes);

    size_t flattenedDimensionSize = 1;
    for (int i = _startAxis; i <= _endAxis; i++)
    {
        flattenedDimensionSize *= inputs[0]->size(i);
    }

    std::vector<int> outputShapeVec;
    for (int i = 0; i < _startAxis; i++)
    {
        outputShapeVec.push_back(inputs[0]->size(i));
    }
    outputShapeVec.push_back(flattenedDimensionSize);
    for (size_t i = _endAxis + 1; i < _numAxes; i++)
    {
        outputShapeVec.push_back(inputs[0]->size(i));
    }
    CV_Assert(outputShapeVec.size() <= 4);

    resultShape = BlobShape(outputShapeVec);

    for (size_t i = 0; i < inputs.size(); i++)
    {
        //in-place
        outputs[i].shareFrom(*inputs[i]);
        outputs[i].reshape(resultShape);
    }
}

void FlattenLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    for (size_t j = 0; j < inputs.size(); j++)
    {
        outputs[j].shareFrom(*inputs[j]);
        outputs[j].reshape(resultShape);
    }
}
}
}
