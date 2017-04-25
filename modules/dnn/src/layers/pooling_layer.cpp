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
using std::max;
using std::min;

namespace cv
{
namespace dnn
{

//TODO: add ceil_mode param
class PoolingLayerImpl : public PoolingLayer
{
public:
    PoolingLayerImpl(const LayerParams& params)
    {
        type = PoolingLayer::MAX;

        if (params.has("pool"))
        {
            String pool = params.get<String>("pool").toLowerCase();
            if (pool == "max")
                type = PoolingLayer::MAX;
            else if (pool == "ave")
                type = PoolingLayer::AVE;
            else if (pool == "stochastic")
                type = PoolingLayer::STOCHASTIC;
            else
                CV_Error(Error::StsBadArg, "Unknown pooling type \"" + pool + "\"");
        }

        getPoolingKernelParams(params, kernel.height, kernel.width, globalPooling,
                               pad.height, pad.width, stride.height, stride.width, padMode);
        setParamsFrom(params);
    }

    void allocate(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        CV_Assert(inputs.size() == 1);

        inp = Size(inputs[0]->size[3], inputs[0]->size[2]);

        if(globalPooling)
        {
            kernel = inp;
        }

        computeOutputShape(inp);

        outputs.resize(type == MAX ? 2 * inputs.size() : inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            const Mat& inp_i = *inputs[i];
            CV_Assert(inp_i.size[2] == inp.height && inp_i.size[3] == inp.width);
            int outsz[] = { inp_i.size[0], inp_i.size[1], out.height, out.width };

            if (type == MAX)
            {
                outputs[2 * i].create(4, outsz, CV_32F);
                outputs[2 * i + 1].create(4, outsz, CV_32F);
            }
            else
            {
                outputs[i].create(4, outsz, CV_32F);
            }
        }
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        for (size_t ii = 0; ii < inputs.size(); ii++)
        {
            switch (type)
            {
                case MAX:
                    maxPooling(*inputs[ii], outputs[2 * ii], outputs[2 * ii + 1]);
                    break;
                case AVE:
                    avePooling(*inputs[ii], outputs[ii]);
                    break;
                default:
                    CV_Error(Error::StsNotImplemented, "Not implemented");
                    break;
            }
        }
    }

    void maxPooling(Mat &src, Mat &dst, Mat &mask)
    {
        CV_DbgAssert(dst.size[2] == out.height && dst.size[3] == out.width);

        for (int n = 0; n < src.size[0]; ++n)
        {
            for (int c = 0; c < src.size[1]; ++c)
            {
                const float *srcData = src.ptr<float>(n, c);
                float *dstData = dst.ptr<float>(n, c);
                float *dstMaskData = mask.ptr<float>(n, c);

                for (int ph = 0; ph < out.height; ++ph)
                {
                    for (int pw = 0; pw < out.width; ++pw)
                    {
                        int hstart = ph * stride.height - pad.height;
                        int wstart = pw * stride.width - pad.width;
                        int hend = min(hstart + kernel.height, inp.height);
                        int wend = min(wstart + kernel.width, inp.width);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        const int poolIndex = ph * out.width + pw;
                        float max_val = -FLT_MAX;
                        int max_index = -1;

                        for (int h = hstart; h < hend; ++h)
                            for (int w = wstart; w < wend; ++w)
                            {
                                const int index = h * inp.width + w;
                                if (srcData[index] > max_val)
                                {
                                    max_val = srcData[index];
                                    max_index = index;
                                }
                            }

                        dstData[poolIndex] = max_val;
                        dstMaskData[poolIndex] = max_index;
                    }
                }
            }
        }
    }

    void avePooling(Mat &src, Mat &dst)
    {
        for (int n = 0; n < src.size[0]; ++n)
        {
            for (int c = 0; c < src.size[1]; ++c)
            {
                const float *srcData = src.ptr<float>(n, c);
                float *dstData = dst.ptr<float>(n, c);

                for (int ph = 0; ph < out.height; ++ph)
                {
                    for (int pw = 0; pw < out.width; ++pw)
                    {
                        int hstart = ph * stride.height - pad.height;
                        int wstart = pw * stride.width - pad.width;
                        int hend = min(hstart + kernel.height, inp.height + pad.height);
                        int wend = min(wstart + kernel.width, inp.width + pad.width);
                        int poolSize = (hend - hstart) * (wend - wstart);
                        hstart = max(hstart, 0);
                        wstart = max(wstart, 0);
                        hend = min(hend, inp.height);
                        wend = min(wend, inp.width);

                        dstData[ph * out.width + pw] = 0.f;

                        for (int h = hstart; h < hend; ++h)
                            for (int w = wstart; w < wend; ++w)
                                dstData[ph * out.width + pw] += srcData[h * inp.width + w];

                        dstData[ph * out.width + pw] /= poolSize;
                    }
                }
            }
        }
    }

    void computeOutputShape(Size inpSz)
    {
        if (padMode.empty()) {
            //Yeah, something strange Caffe scheme-)
            out.height = static_cast<int>(ceil(static_cast<float>(inpSz.height + 2 * pad.height -
                                                                  kernel.height) / stride.height)) + 1;
            out.width = static_cast<int>(ceil(static_cast<float>(inpSz.width + 2 * pad.width -
                                                                 kernel.width) / stride.width)) + 1;

            if (pad.height || pad.width)
            {
                // If we have padding, ensure that the last pooling starts strictly
                // inside the image (instead of at the padding); otherwise clip the last.
                if ((out.height - 1) * stride.height >= inpSz.height + pad.height)
                    --out.height;
                if ((out.width - 1) * stride.width >= inpSz.width + pad.width)
                    --out.width;
                CV_Assert((out.height - 1) * stride.height < inpSz.height + pad.height);
                CV_Assert((out.width - 1) * stride.width < inpSz.width + pad.width);
            }
        }
        else
        {
            getConvPoolOutParams(inpSz.height, inpSz.width, kernel, stride, pad,
                                 padMode, out.height, out.width);
        }
    }

    Size inp, out;
};

Ptr<PoolingLayer> PoolingLayer::create(const LayerParams& params)
{
    return Ptr<PoolingLayer>(new PoolingLayerImpl(params));
}

}
}
