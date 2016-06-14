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

#ifndef __OPENCV_DNN_DNN_ALL_LAYERS_HPP__
#define __OPENCV_DNN_DNN_ALL_LAYERS_HPP__
#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn
{

    //! LSTM recurrent layer
    class LSTMLayer : public Layer
    {
    public:
        CV_EXPORTS_W static Ptr<LSTMLayer> create();

        /** Set trained weights for LSTM layer.
        LSTM behavior on each step is defined by current input, previous output, previous cell state and learned weights.
        Let x_t be current input, h_t be current output, c_t be current state.

        Current output and current cell state is computed as follows:
        h_t = o_t (*) tanh(c_t),
        c_t = f_t (*) c_{t-1} + i_t (*) g_t,
        where (*) is per-element multiply operation and i_t, f_t, o_t, g_t is internal gates that are computed using learned wights.

        Gates are computed as follows:
        i_t = sigmoid(W_xi*x_t + W_hi*h_{t-1} + b_i)
        f_t = sigmoid(W_xf*x_t + W_hf*h_{t-1} + b_f)
        o_t = sigmoid(W_xo*x_t + W_ho*h_{t-1} + b_o)
        g_t = tanh   (W_xg*x_t + W_hg*h_{t-1} + b_g)
        where W_x?, W_h? and b_? are learned weights represented as matrices: W_x? \in R^{N_c x N_x}, W_h? \in R^{N_c x N_h}, b_? \in \R^{N_c}.

        For simplicity and performance purposes we use W_x = [W_xi; W_xf; W_xo, W_xg] (i.e. W_x is vertical contacentaion of W_x?), W_x \in R^{4N_c x N_x}.
        The same for W_h = [W_hi; W_hf; W_ho, W_hg], W_h \in R^{4N_c x N_h}
        and for b = [b_i; b_f, b_o, b_g], b \in R^{4N_c}.

        @param Wh is matrix defining how previous output is transformed to internal gates (i.e. according to abovemtioned notation is W_h)
        @param Wx is matrix defining how current input is transformed to internal gates (i.e. according to abovemtioned notation is W_x)
        @param Wb is bias vector (i.e. according to abovemtioned notation is b)
        */
        virtual void setWeights(const Blob &Wh, const Blob &Wx, const Blob &bias) = 0;

        /** In common cas it use three inputs (x_t, h_{t-1} and c_{t-1}) to compute compute two outputs: h_t and c_t.

        @param input could contain three inputs: x_t, h_{t-1} and c_{t-1}.
        The first x_t input is required.
        The second and third inputs are optional: if they weren't set than layer will use internal h_{t-1} and c_{t-1} from previous calls,
        but at the first call they will be filled by zeros.
        Size of the last dimension of x_t must be N_x, (N_h for h_{t-1} and N_c for c_{t-1}).
        Sizes of remainder dimensions could be any, but thay must be consistent among x_t, h_{t-1} and c_{t-1}.

        @param output computed outputs: h_t and c_t.
        */
        CV_EXPORTS_W void forward(std::vector<Blob*> &input, std::vector<Blob> &output);
    };

    //! Classical recurrent layer
    class RNNLayer : public Layer
    {
    public:

        CV_EXPORTS_W static Ptr<RNNLayer> create();

        /** Setups learned weights.

        Recurrent-layer behavior on each step is defined by current input x_t, previous state h_t and learned weights as follows:
        h_t = tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h),
        o_t = tanh(W_{ho} h_t + b_o),

        @param Whh is W_hh matrix
        @param Wxh is W_xh matrix
        @param bh  is b_h vector
        @param Who is W_xo matrix
        @param bo  is b_o vector
        */
        CV_EXPORTS_W virtual void setWeights(const Blob &Whh, const Blob &Wxh, const Blob &bh, const Blob &Who, const Blob &bo) = 0;

        /** Accepts two inputs x_t and h_{t-1} and compute two outputs o_t and h_t.

        @param input could contain inputs x_t and h_{t-1}.  x_t is required whereas h_{t-1} is optional.
        If the second input h_{t-1} isn't specified a layer will use internal h_{t-1} from the previous calls, at the first call h_{t-1} will be filled by zeros.

        @param output should contain outputs o_t and h_t
        */
        void forward(std::vector<Blob*> &input, std::vector<Blob> &output);
    };
}
}
#endif