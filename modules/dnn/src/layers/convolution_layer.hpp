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

#ifndef __OPENCV_DNN_LAYERS_CONVOLUTION_LAYER_HPP__
#define __OPENCV_DNN_LAYERS_CONVOLUTION_LAYER_HPP__
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{
    //TODO: simultaneously convolution and bias addition for cache optimization
    class ConvolutionLayer : public Layer
    {
    protected:
        bool bias;
        int numOutput, group;
        int padH, padW;
        int kerH, kerW;
        int strideH, strideW;

        int inpH, inpW, inpCn;
        int outH, outW, outCn;
        int topH, topW, topCn; //switched between inp/out on deconv/conv
        int inpGroupCn, outGroupCn;
        int ksize;

        bool useOpenCL;
        Mat colMat, biasOnesMat;

        inline bool is1x1() const;
        virtual void computeInpOutShape(const Blob &inpBlob);
        void im2col(Blob &inpBlob, int imNum, int cnGroup);

    public:
        ConvolutionLayer() {}
        ConvolutionLayer(LayerParams &params);
        void allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };

    class DeConvolutionLayer : public ConvolutionLayer
    {
    protected:
        void computeInpOutShape(const Blob &inpBlob);
        void col2im(Mat &dstMat);

    public:
        DeConvolutionLayer(LayerParams &params);
        void forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs);
    };
}
}
#endif
