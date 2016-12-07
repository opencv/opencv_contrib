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
#include "pooling_layer.hpp"
#include "opencl_kernels_dnn.hpp"
#include <float.h>
#include <algorithm>
#include <opencv2/core/ocl.hpp>
using std::max;
using std::min;

namespace cv
{
namespace dnn
{
//TODO: add ceil_mode param

PoolingLayerImpl::PoolingLayerImpl()
{
    globalPooling = false;
}

PoolingLayerImpl::PoolingLayerImpl(int type_, Size kernel_, Size stride_, Size pad_, const String &padMode_)
{
    globalPooling = false;
    type = type_;
    kernel = kernel_;
    pad = pad_;
    stride = stride_;
    padMode = padMode_;
}

void PoolingLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() > 0);

    inp = inputs[0]->size2();

    if(globalPooling)
    {
        kernel = inp;
    }

    computeOutputShape(inp);

    useOpenCL = ocl::useOpenCL();

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(inputs[i]->rows() == inp.height && inputs[i]->cols() == inp.width);
        outputs[i].create(BlobShape(inputs[i]->num(), inputs[i]->channels(), out.height, out.width));
    }
}

void PoolingLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    for (size_t ii = 0; ii < inputs.size(); ii++)
    {
        switch (type)
        {
        case MAX:
            maxPooling(*inputs[ii], outputs[ii]);
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

void PoolingLayerImpl::maxPooling(Blob &src, Blob &dst)
{
    if (!useOpenCL)
        maxPooling_cpu(src, dst);
    else
    {
        CV_Assert(maxPooling_ocl(src, dst));
    }
}

bool PoolingLayerImpl::maxPooling_ocl(Blob &src, Blob &dst)
{
    return pooling_ocl("MaxPoolForward", src, dst);
}

void PoolingLayerImpl::avePooling(Blob &src, Blob &dst)
{
    if (!useOpenCL)
        avePooling_cpu(src, dst);
    else
    {
        CV_Assert(avePooling_ocl(src, dst));
    }
}

bool PoolingLayerImpl::avePooling_ocl(Blob &src, Blob &dst)
{
    return pooling_ocl("AvePoolForward", src, dst);
}

void PoolingLayerImpl::maxPooling_cpu(Blob &src, Blob &dst)
{
    CV_DbgAssert(dst.rows() == out.height && dst.cols() == out.width);

    for (int n = 0; n < src.num(); ++n)
    {
        for (int c = 0; c < src.channels(); ++c)
        {
            const float *srcData = src.ptrf(n, c);
            float *dstData = dst.ptrf(n, c);

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

                    for (int h = hstart; h < hend; ++h)
                        for (int w = wstart; w < wend; ++w)
                        {
                            const int index = h * inp.width + w;
                            if (srcData[index] > max_val)
                                max_val = srcData[index];
                        }

                    dstData[poolIndex] = max_val;
                }
            }
        }
    }
}


#ifdef HAVE_OPENCL
bool PoolingLayerImpl::pooling_ocl(const char *kname, const Blob &src, Blob &dst, Blob *mask)
{
    const UMat &srcMat = src.umatRefConst();
    UMat &dstMat = dst.umatRef();
    CV_Assert(mask == NULL && srcMat.offset == 0 && dstMat.offset == 0);

    ocl::Kernel ker(kname, ocl::dnn::pooling_oclsrc, String("-DT=") + ocl::typeToStr(src.type()));
    if (ker.empty())
        return false;

    BlobShape s = src.shape();
    size_t nthreads = dst.total();
    ker.args((int)nthreads,
             ocl::KernelArg::PtrReadOnly(srcMat), s[0], s[1], s[2], s[3],
             out.height, out.width, kernel.height, kernel.width,
             stride.height, stride.width, pad.height, pad.width,
             ocl::KernelArg::PtrWriteOnly(dstMat));

    size_t wgSize = ocl::Device::getDefault().maxWorkGroupSize();
    if (!ker.run(1, &nthreads, &wgSize, true))
        return false;

    return true;
}
#else
bool PoolingLayerImpl::pooling_ocl(const char*, const Blob&, Blob&, Blob*)
{
    return false;
}
#endif

void PoolingLayerImpl::avePooling_cpu(Blob &src, Blob &dst)
{
    for (int n = 0; n < src.num(); ++n)
    {
        for (int c = 0; c < src.channels(); ++c)
        {
            const float *srcData = src.ptrf(n, c);
            float *dstData = dst.ptrf(n, c);

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

void PoolingLayerImpl::computeOutputShape(Size inpSz)
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

Ptr<PoolingLayer> PoolingLayer::create(int type, Size kernel, Size stride, Size pad,
                                       const String& padMode)
{
    return Ptr<PoolingLayer>(new PoolingLayerImpl(type, kernel, stride, pad, padMode));
}

Ptr<PoolingLayer> PoolingLayer::createGlobal(int type)
{
    Ptr<PoolingLayer> l = PoolingLayer::create(type);
    l->globalPooling = true;
    return l;
}

}
}
