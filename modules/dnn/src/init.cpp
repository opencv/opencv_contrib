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

#include "precomp.hpp"

#include "layers/concat_layer.hpp"
#include "layers/convolution_layer.hpp"
#include "layers/blank_layer.hpp"
#include "layers/elementwise_layers.hpp"
#include "layers/fully_connected_layer.hpp"
#include "layers/lrn_layer.hpp"
#include "layers/mvn_layer.hpp"
#include "layers/pooling_layer.hpp"
#include "layers/reshape_layer.hpp"
#include "layers/slice_layer.hpp"
#include "layers/softmax_layer.hpp"
#include "layers/split_layer.hpp"

namespace cv
{
namespace dnn
{

struct AutoInitializer
{
    bool status;

    AutoInitializer() : status(false)
    {
        cv::dnn::initModule();
    }
};

static AutoInitializer init;

void initModule()
{
    if (init.status)
        return;

    REG_RUNTIME_LAYER_CLASS(Slice, SliceLayer)
    REG_RUNTIME_LAYER_CLASS(Softmax, SoftMaxLayer)
    REG_RUNTIME_LAYER_CLASS(Split, SplitLayer)
    REG_RUNTIME_LAYER_CLASS(Reshape, ReshapeLayer)
    REG_STATIC_LAYER_FUNC(Flatten, createFlattenLayer)
    REG_RUNTIME_LAYER_CLASS(Pooling, PoolingLayer)
    REG_RUNTIME_LAYER_CLASS(MVN, MVNLayer)
    REG_RUNTIME_LAYER_CLASS(LRN, LRNLayer)
    REG_RUNTIME_LAYER_CLASS(InnerProduct, FullyConnectedLayer)

    REG_RUNTIME_LAYER_CLASS(ReLU, ElementWiseLayer<ReLUFunctor>)
    REG_RUNTIME_LAYER_CLASS(TanH, ElementWiseLayer<TanHFunctor>)
    REG_RUNTIME_LAYER_CLASS(BNLL, ElementWiseLayer<BNLLFunctor>)
    REG_RUNTIME_LAYER_CLASS(Power, ElementWiseLayer<PowerFunctor>)
    REG_RUNTIME_LAYER_CLASS(AbsVal, ElementWiseLayer<AbsValFunctor>)
    REG_RUNTIME_LAYER_CLASS(Sigmoid, ElementWiseLayer<SigmoidFunctor>)
    REG_RUNTIME_LAYER_CLASS(Dropout, BlankLayer)

    REG_RUNTIME_LAYER_CLASS(Convolution, ConvolutionLayer)
    REG_RUNTIME_LAYER_CLASS(Deconvolution, DeConvolutionLayer)
    REG_RUNTIME_LAYER_CLASS(Concat, ConcatLayer)

    init.status = true;
}

}
}
