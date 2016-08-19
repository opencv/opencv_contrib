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
#include "eltwise_layer.hpp"

namespace cv
{
namespace dnn
{
    EltwiseLayerImpl::EltwiseLayerImpl(EltwiseOp op_, const std::vector<int> &coeffs_)
    {
        op = op_;
        coeffs = coeffs_;
    }

    void EltwiseLayerImpl::allocate(const std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        CV_Assert(2 <= inputs.size());
        CV_Assert(coeffs.size() == 0 || coeffs.size() == inputs.size());
        CV_Assert(op == SUM || coeffs.size() == 0);

        const BlobShape &shape0 = inputs[0]->shape();
        for (size_t i = 1; i < inputs.size(); ++i)
        {
            CV_Assert(shape0 == inputs[i]->shape());
        }
        outputs.resize(1);
        outputs[0].create(shape0);
    }

    void EltwiseLayerImpl::forward(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
    {
        switch (op)
        {
        case SUM:
            {
                CV_Assert(coeffs.size() == 0 || coeffs.size() == inputs.size());
                Mat& output = outputs[0].matRef();
                output.setTo(0.);
                if (0 < coeffs.size())
                {
                    for (size_t i = 0; i < inputs.size(); i++)
                    {
                        output += inputs[i]->matRefConst() * coeffs[i];
                    }
                }
                else
                {
                    for (size_t i = 0; i < inputs.size(); i++)
                    {
                        output += inputs[i]->matRefConst();
                    }
                }
            }
            break;
        case PROD:
            {
                Mat& output = outputs[0].matRef();
                output.setTo(1.);
                for (size_t i = 0; i < inputs.size(); i++)
                {
                    output = output.mul(inputs[i]->matRefConst());
                }
            }
            break;
        case MAX:
            {
                Mat& output = outputs[0].matRef();
                cv::max(inputs[0]->matRefConst(), inputs[1]->matRefConst(), output);
                for (size_t i = 2; i < inputs.size(); i++)
                {
                    cv::max(output, inputs[i]->matRefConst(), output);
                }
            }
            break;
        default:
            CV_Assert(0);
            break;
        };
    }

    Ptr<EltwiseLayer> EltwiseLayer::create(EltwiseOp op, const std::vector<int> &coeffs)
    {
        return Ptr<EltwiseLayer>(new EltwiseLayerImpl(op, coeffs));
    }
}
}
