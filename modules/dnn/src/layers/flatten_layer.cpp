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
#include <float.h>
#include <algorithm>

namespace cv
{
namespace dnn
{

class FlattenLayerImpl : public FlattenLayer
{
public:
    FlattenLayerImpl(const LayerParams &params)
    {
        _startAxis = params.get<int>("axis", 1);
        _endAxis = params.get<int>("end_axis", -1);
        setParamsFrom(params);
    }

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        size_t i, ninputs = inputs.size();
        CV_Assert(ninputs > 0);
        const Mat& inp0 = *inputs[0];

        for (i = 1; i < ninputs; i++)
        {
            CV_Assert(inputs[i]->size == inp0.size);
        }

        _numAxes = inp0.dims;
        _endAxis = _endAxis < 0 ? _endAxis + _numAxes : _endAxis;
        CV_Assert(_startAxis >= 0);
        CV_Assert(_endAxis >= _startAxis && _endAxis < (int)_numAxes);

        size_t flattenedDimensionSize = inp0.total(_startAxis, _endAxis+1);

        resultShape.clear();
        for (int j = 0; j < _startAxis; j++)
        {
            resultShape.push_back(inp0.size[j]);
        }
        resultShape.push_back(flattenedDimensionSize);
        for (int j = _endAxis + 1; j < _numAxes; j++)
        {
            resultShape.push_back(inp0.size[j]);
        }
        CV_Assert(resultShape.size() <= 4);

        for (i = 0; i < ninputs; i++)
        {
            //in-place
            outputs[i] = inputs[i]->reshape(1, (int)resultShape.size(), &resultShape[0]);
        }
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            outputs[i] = inputs[i]->reshape(1, (int)resultShape.size(), &resultShape[0]);
        }
    }

    int _startAxis;
    int _endAxis;
    size_t _numAxes;

    std::vector<int> resultShape;
};

Ptr<FlattenLayer> FlattenLayer::create(const LayerParams& params)
{
    return Ptr<FlattenLayer>(new FlattenLayerImpl(params));
}

}
}
