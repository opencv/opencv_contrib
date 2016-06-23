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

#ifndef __OPENCV_DNN_LAYERS_IM2COL_HPP__
#define __OPENCV_DNN_LAYERS_IM2COL_HPP__
#include <opencv2/core.hpp>
#include <iostream>

namespace cv
{
namespace dnn
{

template <typename Dtype>
class im2col_CpuPBody : public cv::ParallelLoopBody
{
    const Dtype* data_im;
    int channels, height, width;
    int kernel_h, kernel_w;
    int pad_h, pad_w;
    int stride_h, stride_w;
    Dtype* data_col;
    int height_col, width_col, channels_col;

public:

    im2col_CpuPBody(const Dtype* data_im_,
                     int channels_, int height_, int width_,
                     int kernel_h_, int kernel_w_,
                     int pad_h_, int pad_w_,
                     int stride_h_, int stride_w_,
                     Dtype* data_col_) :
        data_im(data_im_),
        channels(channels_), height(height_), width(width_),
        kernel_h(kernel_h_), kernel_w(kernel_w_),
        pad_h(pad_h_), pad_w(pad_w_),
        stride_h(stride_h_), stride_w(stride_w_),
        data_col(data_col_)
    {
        height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        channels_col = channels * kernel_h * kernel_w;
    }

    static void run(const Dtype* data_im,
                    int channels, int height, int width,
                    int kernel_h, int kernel_w,
                    int pad_h, int pad_w,
                    int stride_h, int stride_w,
                    Dtype* data_col)
    {
        im2col_CpuPBody<Dtype> pb(data_im, channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, data_col);
        cv::parallel_for_(Range(0, pb.channels_col), pb);
    }

    virtual void operator ()(const Range &r) const
    {
        for (int c = r.start; c < r.end; ++c) {
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / kernel_h / kernel_w;
            for (int h = 0; h < height_col; ++h) {
                for (int w = 0; w < width_col; ++w) {
                    int h_pad = h * stride_h - pad_h + h_offset;
                    int w_pad = w * stride_w - pad_w + w_offset;
                    if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                        data_col[(c * height_col + h) * width_col + w] =
                        data_im[(c_im * height + h_pad) * width + w_pad];
                    else
                        data_col[(c * height_col + h) * width_col + w] = 0;
                }
            }
        }
    }
};

template <typename Dtype>
void col2im_cpu(const Dtype* data_col,
                int channels, int height, int width,
                int patch_h, int patch_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                Dtype* data_im)
{
    memset(data_im, 0, height * width * channels * sizeof(Dtype));

    int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
    int channels_col = channels * patch_h * patch_w;

    for (int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % patch_w;
        int h_offset = (c / patch_w) % patch_h;
        int c_im = c / patch_h / patch_w;

        for (int h = 0; h < height_col; ++h)
        {
            for (int w = 0; w < width_col; ++w)
            {
                int h_pad = h * stride_h - pad_h + h_offset;
                int w_pad = w * stride_w - pad_w + w_offset;

                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    data_im[(c_im * height + h_pad) * width + w_pad] +=
                    data_col[(c * height_col + h) * width_col + w];
            }
        }
    }
}

#ifdef HAVE_OPENCL
void im2col_ocl(UMat &img,
                int channels, int height, int width,
                int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                UMat &col);
#endif

}
}

#endif
