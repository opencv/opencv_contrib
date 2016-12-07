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
#include "permute_layer.hpp"
#include <float.h>
#include <algorithm>

namespace cv
{
namespace dnn
{
void PermuteLayer::checkCurrentOrder(int currentOrder)
{
    if(currentOrder < 0 || currentOrder > 3)
    {
        CV_Error(
            Error::StsBadArg,
            "Orders of dimensions in Permute layer parameter"
            "must be in [0...3] interval");
    }

    if(std::find(_order.begin(), _order.end(), currentOrder) != _order.end())
    {
        CV_Error(Error::StsBadArg,
                 "Permute layer parameter contains duplicated orders.");
    }
}

void PermuteLayer::checkNeedForPermutation()
{
    _needsPermute = false;
    for (size_t i = 0; i < _numAxes; ++i)
    {
        if (_order[i] != i)
        {
            _needsPermute = true;
            break;
        }
    }
}

PermuteLayer::PermuteLayer(LayerParams &params) : Layer(params)
{
    if (!params.has("order"))
    {
        _needsPermute = false;
        return;
    }

    DictValue paramOrder = params.get("order");
    if(paramOrder.size() > 4)
    {
        CV_Error(
            Error::StsBadArg,
            "Too many (> 4) orders of dimensions in Permute layer");
    }

    _numAxes = paramOrder.size();

    for (size_t i = 0; i < _numAxes; i++)
    {
        int currentOrder = paramOrder.get<int>(i);
        checkCurrentOrder(currentOrder);
        _order.push_back(currentOrder);
    }

    checkNeedForPermutation();
}

void PermuteLayer::computeStrides()
{
    _oldStride.resize(_numAxes);
    _newStride.resize(_numAxes);

    _oldStride[_numAxes - 1] = 1;
    _newStride[_numAxes - 1] = 1;

    for(int i = _numAxes - 2; i >= 0; i--)
    {
        _oldStride[i] = _oldStride[i + 1] * _oldDimensionSize[i + 1];
        _newStride[i] = _newStride[i + 1] * _newDimensionSize[i + 1];
    }

    _count = _oldStride[0] * _oldDimensionSize[0];
}

void PermuteLayer::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    if(!_needsPermute)
    {
        return;
    }

    CV_Assert(inputs.size() > 0);
    CV_Assert((int)_numAxes == inputs[0]->shape().dims());

    outputs.resize(inputs.size());

    _oldDimensionSize = inputs[0]->shape();
    for (size_t i = 0; i < _numAxes; i++)
    {
        _newDimensionSize[i] = _oldDimensionSize[_order[i]];
    }

    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(inputs[i]->rows() == _oldDimensionSize[2] && inputs[i]->cols() == _oldDimensionSize[3]);
        outputs[i].create(BlobShape(_newDimensionSize));
    }

    computeStrides();
}

void PermuteLayer::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    if(!_needsPermute)
    {
        for (size_t j = 0; j < inputs.size(); j++)
        {
            outputs[j].matRef() = inputs[j]->matRef();
        }
        return;
    }

    for (size_t k = 0; k < inputs.size(); k++)
    {
        float *srcData = inputs[k]->ptrf();
        float *dstData = outputs[k].ptrf();

        for (size_t i = 0; i < _count; ++i)
        {
            int oldPosition = 0;
            int newPosition = i;

            for (size_t j = 0; j < _numAxes; ++j)
            {
                oldPosition += (newPosition / _newStride[j]) * _oldStride[_order[j]];
                newPosition %= _newStride[j];
            }
            dstData[i] = srcData[oldPosition];
        }
    }
}
}
}
