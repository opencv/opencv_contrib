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
#include <cstdlib>

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
    int dilation_h, dilation_w;
    Dtype* data_col;
    int height_col, width_col, channels_col;

    im2col_CpuPBody() {}
public:

    static void run(const Dtype* data_im,
                    int channels, int height, int width,
                    int kernel_h, int kernel_w,
                    int pad_h, int pad_w,
                    int stride_h, int stride_w,
                    int dilation_h, int dilation_w,
                    int height_col, int width_col,
                    Dtype* data_col)
    {
        im2col_CpuPBody<Dtype> t;

        t.data_im = data_im;
        t.data_col = data_col;
        t.channels = channels; t.height = height; t.width = width;
        t.kernel_h = kernel_h; t.kernel_w = kernel_w;
        t.pad_h = pad_h; t.pad_w = pad_w;
        t.stride_h = stride_h; t.stride_w = stride_w;
        t.dilation_h = dilation_h; t.dilation_w = dilation_w;

        t.height_col = height_col;
        t.width_col = width_col;
        t.channels_col = channels * kernel_h * kernel_w;

        cv::parallel_for_(Range(0, t.channels_col), t);
    }

    virtual void operator ()(const Range &r) const
    {
        for (int c = r.start; c < r.end; ++c)
        {
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / kernel_h / kernel_w;
            for (int h = 0; h < height_col; ++h)
            {
                for (int w = 0; w < width_col; ++w)
                {
                    int h_pad = h * stride_h - pad_h + h_offset * dilation_h;
                    int w_pad = w * stride_w - pad_w + w_offset * dilation_w;
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
class col2im_CpuPBody : public cv::ParallelLoopBody
{
    const Dtype* data_col;
    int channels, height, width;
    int kernel_h, kernel_w;
    int pad_h, pad_w;
    int stride_h, stride_w;
    Dtype* data_im;
    int height_col, width_col;

    col2im_CpuPBody() {}

public:

    static void run(const Dtype* data_col,
                    int channels, int height, int width,
                    int kernel_h, int kernel_w,
                    int pad_h, int pad_w,
                    int stride_h, int stride_w,
                    Dtype* data_im)
    {
        //TODO: single-threaded version switch

        col2im_CpuPBody t;
        t.data_col = data_col;
        t.data_im = data_im;
        t.channels = channels; t.height = height; t.width = width;
        t.kernel_h = kernel_h; t.kernel_w = kernel_w;
        t.pad_h = pad_h; t.pad_w = pad_w;
        t.stride_h = stride_h; t.stride_w = stride_w;
        t.height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        t.width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
        int img_total = channels * height * width;

        cv::parallel_for_(Range(0, img_total), t);
    }

    virtual void operator ()(const Range &r) const
    {
        for (int index = r.start; index < r.end; index++)
        {
            Dtype val = 0;
            int w = index % width + pad_w;
            int h = (index / width) % height + pad_h;
            int c = index / (width * height);

            // compute the start and end of the output
            int w_col_start = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
            int w_col_end = std::min(w / stride_w + 1, width_col);
            int h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
            int h_col_end = std::min(h / stride_h + 1, height_col);

            // equivalent implementation
            int offset =
            (c * kernel_h * kernel_w + h * kernel_w + w) * height_col * width_col;
            int coeff_h_col = (1 - stride_h * kernel_w * height_col) * width_col;
            int coeff_w_col = (1 - stride_w * height_col * width_col);
            for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
              for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
              }
            }
            data_im[index] = val;
        }
    }
};

//single-threaded version
template <typename Dtype>
void col2im_cpu(const Dtype* data_col,
                int channels, int height, int width,
                int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                int dilation_h, int dilation_w,
                Dtype* data_im)
{
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;

    std::memset(data_im, 0, height * width * channels * sizeof(Dtype));

    for (int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / kernel_h / kernel_w;

        for (int h = 0; h < height_col; ++h)
        {
            for (int w = 0; w < width_col; ++w)
            {
                int h_pad = h * stride_h - pad_h + h_offset * dilation_h;
                int w_pad = w * stride_w - pad_w + w_offset * dilation_w;

                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    data_im[(c_im * height + h_pad) * width + w_pad] +=
                        data_col[(c * height_col + h) * width_col + w];
            }
        }
    }
}

#ifdef HAVE_OPENCL
bool im2col_ocl(const UMat &img,
                int channels, int height, int width,
                int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                int dilation_h, int dilation_w,
                UMat &col);

bool col2im_ocl(const UMat &col,
                int channels, int height, int width,
                int kernel_h, int kernel_w,
                int pad_h, int pad_w,
                int stride_h, int stride_w,
                UMat &img);
#endif

}
}

#endif
