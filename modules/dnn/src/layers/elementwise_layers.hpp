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
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/core/ocl.hpp>
#ifdef HAVE_OPENCL
#include "modules/dnn/opencl_kernels_dnn.hpp"
#endif

namespace cv
{
namespace dnn
{

using std::abs;
using std::exp;
using std::tanh;
using std::pow;

template<typename Func>
class ElementWiseLayer : public Func::Layer
{
    bool useOpenCL;
    Func func;

    template<typename Dtype>
    class PBody : public cv::ParallelLoopBody
    {
        Func &func;
        Dtype *data;
    public:

        PBody(Mat &mat, Func &func_) :
            func(func_), data(mat.ptr<Dtype>())
        {}

        void operator()(const Range &r) const
        {
            for (int i = r.start; i < r.end; i++)
                data[i] = func(data[i]);
        }
    };

public:

    ElementWiseLayer() {}
    ElementWiseLayer(const Func &f) : func(f) {}

    void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        useOpenCL = ocl::useOpenCL();

        outputs.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            outputs[i].shareFrom(*inputs[i]); //no data copy

            //hotfix: shareFrom doesn't provide properly Mat/UMat switching
            if (useOpenCL)
                outputs[i].umatRef() = inputs[i]->umatRefConst();
            else
                outputs[i].matRef() = inputs[i]->matRefConst();
        }
    }

    void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        #ifdef HAVE_OPENCL
        if (useOpenCL)
            forwardOCL(inputs, outputs);
        else
        #endif
            forwardCPU(inputs, outputs);
    }

    #ifdef HAVE_OPENCL
    void forwardOCL(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        size_t wgSize = ocl::Device::getDefault().maxWorkGroupSize();

        for (size_t i = 0; i < inputs.size(); i++)
        {
            const UMat &src = inputs[i]->umatRefConst();
            UMat &dst = outputs[i].umatRef();
            CV_Assert(src.isContinuous() && dst.isContinuous() && !src.offset && !dst.offset);

            ocl::Kernel ker;
            CV_Assert(func.initKernel(ker, src));
            ker.set(0, (int)src.total());
            ker.set(1, ocl::KernelArg::PtrReadOnly(src));
            ker.set(2, ocl::KernelArg::PtrWriteOnly(dst));

            size_t gSize = src.total();
            CV_Assert(ker.run(1, &gSize, &wgSize, true));
        }
    }
    #endif

    void forwardCPU(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            const Mat &src = inputs[i]->matRefConst();
            Mat &dst = outputs[i].matRef();
            CV_Assert(src.ptr() == dst.ptr() && src.isContinuous());

            Range sizeRange = Range(0, dst.total());
            if (dst.type() == CV_32F)
            {
                cv::parallel_for_(sizeRange, PBody<float>(dst, func));
            }
            else if (dst.type() == CV_64F)
            {
                cv::parallel_for_(sizeRange, PBody<double>(dst, func));
            }
            else
            {
                CV_Error(Error::StsNotImplemented, "Only CV_32F and CV_64F blobs are supported");
            }
        }
    }
};

#ifdef HAVE_OPENCL
static String oclGetTMacro(const UMat &m)
{
    return String("-DT=") + ocl::typeToStr(m.type()) + String(" ");
}
#endif

struct ReLUFunctor
{
    typedef ReLULayer Layer;

    double slope;

    ReLUFunctor(double slope_)
        : slope(slope_) {}

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return (x >= (TFloat)0) ? x : (TFloat)slope * x;
    }

    #ifdef HAVE_OPENCL
    bool initKernel(ocl::Kernel &ker, const UMat &src) const
    {
        const char *buildoptSlope = (slope == 0) ? "-DRELU_NO_SLOPE" : "";
        String buildopt = oclGetTMacro(src) + buildoptSlope;

        if (!ker.create("ReLUForward", ocl::dnn::activations_oclsrc, buildopt))
            return false;

        if (slope != 0)
            ker.set(3, (float)slope);

        return true;
    }
    #endif
};

struct TanHFunctor
{
    typedef TanHLayer Layer;

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return tanh(x);
    }

    #ifdef HAVE_OPENCL
    bool initKernel(ocl::Kernel &ker, const UMat &src) const
    {
        if (!ker.create("TanHForward", ocl::dnn::activations_oclsrc, oclGetTMacro(src)))
            return false;
        return true;
    }
    #endif
};

struct SigmoidFunctor
{
    typedef SigmoidLayer Layer;

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return (TFloat)1 / ((TFloat)1 + exp(-x));
    }

    #ifdef HAVE_OPENCL
    bool initKernel(ocl::Kernel &ker, const UMat &src) const
    {
        if (!ker.create("SigmoidForward", ocl::dnn::activations_oclsrc, oclGetTMacro(src)))
            return false;
        return true;
    }
    #endif
};

struct AbsValFunctor
{
    typedef AbsLayer Layer;

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return abs(x);
    }

    #ifdef HAVE_OPENCL
    bool initKernel(ocl::Kernel &ker, const UMat &src) const
    {
        if (!ker.create("AbsValForward", ocl::dnn::activations_oclsrc, oclGetTMacro(src)))
            return false;
        return true;
    }
    #endif
};

struct BNLLFunctor
{
    typedef BNLLLayer Layer;

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return log((TFloat)1 + exp(-abs(x)));
    }

    #ifdef HAVE_OPENCL
    bool initKernel(ocl::Kernel &ker, const UMat &src) const
    {
        if (!ker.create("BNLLForward", ocl::dnn::activations_oclsrc, oclGetTMacro(src)))
            return false;
        return true;
    }
    #endif
};

struct PowerFunctor
{
    typedef PowerLayer Layer;

    double power, scale, shift;

    PowerFunctor(double power_, double scale_ = 1, double shift_ = 0)
        : power(power_), scale(scale_), shift(shift_) {}

    template<typename TFloat>
    inline TFloat operator()(TFloat x) const
    {
        return pow((TFloat)shift + (TFloat)scale * x, (TFloat)power);
    }

    #ifdef HAVE_OPENCL
    bool initKernel(ocl::Kernel &ker, const UMat &src) const
    {
        if (!ker.create("PowForward", ocl::dnn::activations_oclsrc, oclGetTMacro(src)))
            return false;

        ker.set(3, (float)power);
        ker.set(4, (float)scale);
        ker.set(5, (float)shift);

        return true;
    }
    #endif
};

}
}
#endif
