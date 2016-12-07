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
#include "softmax_layer.hpp"
#include <opencv2/core/ocl.hpp>
#include "modules/dnn/opencl_kernels_dnn.hpp"
#include <algorithm>
#include <stdlib.h>
using std::max;

namespace cv
{
namespace dnn
{

SoftMaxLayerImpl::SoftMaxLayerImpl(int axis)
{
    axisRaw = axis;
}

void SoftMaxLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    CV_Assert(inputs.size() == 1);
    axis = inputs[0]->canonicalAxis(axisRaw);

    useOpenCL = ocl::useOpenCL();

    BlobShape shape = inputs[0]->shape();
    outerSize = shape.total(0, axis);
    channels = shape[axis];
    innerSize = shape.total(axis + 1);

    int allocFlag = (useOpenCL) ? Blob::ALLOC_UMAT : Blob::ALLOC_MAT;
    shape[axis] = 1;
    buf.create(shape, inputs[0]->type(), allocFlag);

    outputs.resize(1);
    outputs[0].create(inputs[0]->shape(), inputs[0]->type(), allocFlag);
}

void SoftMaxLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    Blob &src = *inputs[0];
    Blob &dst = outputs[0];

    if (!useOpenCL)
        forward_cpu(src, dst);
    else
    {
        CV_Assert(forward_ocl(src, dst));
    }
}

#ifdef HAVE_OPENCL
bool SoftMaxLayerImpl::forward_ocl(Blob &src, Blob &dst)
{
    const UMat &srcMat = src.umatRefConst();
    UMat &dstMat = dst.umatRef();
    srcMat.copyTo(dstMat);
    UMat &bufMat = buf.umatRef();
    CV_Assert(dstMat.offset == 0);

    String buildOpts = String("-DT=") + ocl::typeToStr(src.type());
    ocl::Kernel kmax, ksub, ksum, kdiv;

    if (!kmax.create("kernel_channel_max", ocl::dnn::softmax_oclsrc, buildOpts))
        return false;

    if (!ksub.create("kernel_channel_subtract", ocl::dnn::softmax_oclsrc, buildOpts))
        return false;

    if (!ksum.create("kernel_channel_sum", ocl::dnn::softmax_oclsrc, buildOpts))
        return false;

    if (!kdiv.create("kernel_channel_div", ocl::dnn::softmax_oclsrc, buildOpts))
        return false;

    size_t wgSize = ocl::Device::getDefault().maxWorkGroupSize();
    size_t bufSize = buf.total();
    size_t totalSize = src.total();

    kmax.args((int)outerSize, (int)channels, (int)innerSize,
              ocl::KernelArg::PtrReadOnly(dstMat), ocl::KernelArg::PtrReadWrite(bufMat));
    if (!kmax.run(1, &bufSize, &wgSize, true))
        return false;

    ksub.args((int)totalSize, (int)outerSize, (int)channels, (int)innerSize,
              ocl::KernelArg::PtrReadOnly(bufMat), ocl::KernelArg::PtrReadWrite(dstMat));
    if (!ksub.run(1, &totalSize, &wgSize, true))
        return false;

    cv::exp(dstMat, dstMat);

    ksum.args((int)outerSize, (int)channels, (int)innerSize,
              ocl::KernelArg::PtrReadOnly(dstMat), ocl::KernelArg::PtrReadWrite(bufMat));
    if (!ksum.run(1, &bufSize, &wgSize, true))
        return false;

    kdiv.args((int)totalSize, (int)outerSize, (int)channels, (int)innerSize,
              ocl::KernelArg::PtrReadOnly(bufMat), ocl::KernelArg::PtrReadWrite(dstMat));
    if (!kdiv.run(1, &totalSize, &wgSize, true))
        return false;

    return true;
}
#else
bool SoftMaxLayerImpl::forward_ocl(Blob&, Blob&)
{
    return false;
}
#endif

void SoftMaxLayerImpl::forward_cpu(Blob &src, Blob &dst)
{
    CV_Assert(src.type() == CV_32F);

    float *srcPtr = src.ptrf();
    float *dstPtr = dst.ptrf();
    float *bufPtr = buf.ptrf();

    size_t outerStep = src.total(axis);
    size_t cnStep = src.total(axis + 1);

    //compute max along axis
    for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
    {
        size_t srcOffset = outerDim * outerStep;
        size_t bufOffset = outerDim * cnStep;

        memcpy(bufPtr + bufOffset, srcPtr + srcOffset, innerSize * sizeof(float));

        for (size_t cnDim = 1; cnDim < channels; cnDim++)
        {
            for (size_t i = 0; i < innerSize; i++)
                bufPtr[bufOffset + i] = std::max(bufPtr[bufOffset + i], srcPtr[srcOffset + cnDim * cnStep + i]);
        }
    }

    //subtract max
    for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
    {
        size_t srcOffset = outerDim * outerStep;
        size_t bufOffset = outerDim * cnStep;

        for (size_t cnDim = 0; cnDim < channels; cnDim++)
        {
            for (size_t i = 0; i < innerSize; i++)
                dstPtr[srcOffset + cnDim * cnStep + i] = srcPtr[srcOffset + cnDim * cnStep + i] - bufPtr[bufOffset + i];
        }
    }

    cv::exp(dst.matRef(), dst.matRef());

    for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
    {
        size_t srcOffset = outerDim * outerStep;
        size_t bufOffset = outerDim * cnStep;

        //sum exp along axis
        for (size_t i = 0; i < innerSize; i++)
            bufPtr[bufOffset + i] = 0.f;

        for (size_t cnDim = 0; cnDim < channels; cnDim++)
        {
            for (size_t i = 0; i < innerSize; i++)
                bufPtr[bufOffset + i] += dstPtr[srcOffset + cnDim * cnStep + i];
        }

        //divide by computed sum
        for (size_t cnDim = 0; cnDim < channels; cnDim++)
        {
            for (size_t i = 0; i < innerSize; i++)
                dstPtr[srcOffset + cnDim * cnStep + i] /= bufPtr[bufOffset + i];
        }
    }
}

Ptr<SoftmaxLayer> SoftmaxLayer::create(int axis)
{
    return Ptr<SoftmaxLayer>(new SoftMaxLayerImpl(axis));
}

}
}
