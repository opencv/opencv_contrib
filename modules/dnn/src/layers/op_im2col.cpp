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
#include <opencv2/core/ocl.hpp>
#include "opencl_kernels_dnn.hpp"
#include "op_im2col.hpp"
#include "opencl_kernels_dnn.hpp"

namespace cv
{
namespace dnn
{

#ifdef HAVE_OPENCL

bool im2col_ocl(const UMat &img,
                 int channels, int height, int width,
                 int kernel_h, int kernel_w,
                 int pad_h, int pad_w,
                 int stride_h, int stride_w,
                 int dilation_h, int dilation_w,
                 UMat &col)
{
    //TODO
    CV_Assert(dilation_h == 1 && dilation_w == 1);

    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
    int esz = img.elemSize();

    CV_Assert(img.isContinuous() && col.isContinuous());
    CV_Assert(img.total() == (size_t)channels * height * width);
    CV_Assert(col.total() == (size_t)channels_col * height_col * width_col);

    ocl::Kernel ker("im2col", ocl::dnn::im2col_oclsrc, String("-DT=") + ocl::typeToStr(img.type()));
    if (ker.empty())
        return false;

    ker.args(ocl::KernelArg::PtrReadOnly(img), (int)img.offset/esz,
             channels, height, width,
             kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
             height_col, width_col,
             ocl::KernelArg::PtrWriteOnly(col), (int)col.offset/esz
             );

    size_t localSize = ocl::Device::getDefault().maxWorkGroupSize();
    size_t globalSize = (size_t)channels * height_col * width_col;
    return ker.run(1, &globalSize, &localSize, true);
}

bool col2im_ocl(const UMat &col,
                int channels, int height, int width,
                int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                UMat &img)
{
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;
    int esz = img.elemSize();

    CV_Assert(img.isContinuous() && col.isContinuous());
    CV_Assert(img.total() == (size_t)channels * height * width);
    CV_Assert(col.total() == (size_t)channels_col * height_col * width_col);

    ocl::Kernel ker("col2im", ocl::dnn::col2im_oclsrc, String("-DT=") + ocl::typeToStr(col.type()));
    if (ker.empty())
        return false;

    ker.args((int)img.total(),
             ocl::KernelArg::PtrReadOnly(col), (int)col.offset/esz,
             height, width, channels,
             kernel_h, kernel_w,
             pad_h, pad_w,
             stride_h, stride_w,
             height_col, width_col,
             ocl::KernelArg::PtrWriteOnly(img), (int)img.offset/esz);

    size_t localSize = ocl::Device::getDefault().maxWorkGroupSize();
    size_t globalSize = img.total();
    return ker.run(1, &globalSize, &localSize, true);
}

#endif
}
}

namespace cv
{
namespace dnn
{

#ifdef HAVE_OPENCL
void im2col_ocl(UMat &img,
                int channels, int height, int width,
                int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                int height_out, int width_out,
                UMat &col)
{
    int h_out = height_out;
    int w_out = width_out;

    CV_Assert(img.isContinuous() && col.isContinuous());
    CV_Assert(img.total() == (size_t)channels * height * width);
    CV_Assert(col.total() == (size_t)channels * kernel_h * kernel_w * h_out * w_out);

    ocl::Kernel im2col_ker("im2col", ocl::dnn::im2col_oclsrc);
    CV_Assert(!im2col_ker.empty());

    im2col_ker.args(ocl::KernelArg::PtrReadOnly(img), (int)img.offset,
             channels, height, width,
             kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
             h_out, w_out,
             ocl::KernelArg::PtrWriteOnly(col), (int)col.offset
        );

    size_t localSize = ocl::Device::getDefault().maxWorkGroupSize();
    size_t globalSize = (size_t)channels * h_out * w_out;

    CV_Assert(im2col_ker.run(1, &globalSize, &localSize, true));
}
#endif // HAVE_OPENCL

}
}
