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

#ifndef __OPENCV_DNN_LAYERS_ELEMENTWISE_LAYERS_HPP__
#define __OPENCV_DNN_LAYERS_ELEMENTWISE_LAYERS_HPP__
#include "../precomp.hpp"
#include "layers_common.hpp"
#include <cmath>

namespace cv
{
namespace dnn
{

using std::abs;
using std::exp;
using std::tanh;
using std::pow;

    template<typename Func>
    class ElementWiseLayer : public Layer
    {
        Func func;
    public:

        ElementWiseLayer(LayerParams &_params) : func(_params) {}

        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            outputs.resize(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++)
                outputs[i].shareFrom(*inputs[i]); //no data copy
        }

        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
        {
            for (size_t i = 0; i < inputs.size(); i++)
            {
                CV_Assert(inputs[i]->ptr() == outputs[i].ptr() && inputs[i]->type() == outputs[i].type());

                size_t size = outputs[i].total();

                if (outputs[i].type() == CV_32F)
                {
                    float *data = outputs[i].ptrf();
                    for (size_t j = 0; j < size; j++)
                        data[j] = func(data[j]);
                }
                else if (outputs[i].type() == CV_64F)
                {
                    double *data = outputs[i].ptr<double>();
                    for (size_t j = 0; j < size; j++)
                        data[j] = func(data[j]);
                }
                else
                {
                    CV_Error(Error::StsNotImplemented, "Only CV_32F and CV_64F blobs are supported");
                }
            }
        }
    };


    struct ReLUFunctor
    {
        float negative_slope;

        ReLUFunctor(LayerParams &params)
        {
            if (params.has("negative_slope"))
                negative_slope = params.get<float>("negative_slope");
            else
                negative_slope = 0.f;
        }

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return (x >= (TFloat)0) ? x : negative_slope * x;
        }
    };

    struct TanHFunctor
    {
        TanHFunctor(LayerParams&) {}

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return tanh(x);
        }
    };

    struct SigmoidFunctor
    {
        SigmoidFunctor(LayerParams&) {}

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return (TFloat)1 / ((TFloat)1 + exp(-x));
        }
    };

    struct AbsValFunctor
    {
        AbsValFunctor(LayerParams&) {}

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return abs(x);
        }
    };

    struct PowerFunctor
    {
        float power, scale, shift;

        PowerFunctor(LayerParams &params)
        {
            power = params.get<float>("power", 1.0f);
            scale = params.get<float>("scale", 1.0f);
            shift = params.get<float>("shift", 0.0f);
        }

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return pow((TFloat)shift + (TFloat)scale * x, (TFloat)power);
        }
    };

    struct BNLLFunctor
    {
        BNLLFunctor(LayerParams&) {}

        template<typename TFloat>
        inline TFloat operator()(TFloat x)
        {
            return log((TFloat)1 + exp(-abs(x)));
        }
    };
}
}
#endif
