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
//! @addtogroup dnn
//! @{

/** @defgroup dnnLayerList Partial List of Implemented Layers
  @{
  This subsection of dnn module contains information about bult-in layers and their descriptions.

  Classes listed here, in fact, provides C++ API for creating intances of bult-in layers.
  In addition to this way of layers instantiation, there is a more common factory API (see @ref dnnLayerFactory), it allows to create layers dynamically (by name) and register new ones.
  You can use both API, but factory API is less convinient for native C++ programming and basically designed for use inside importers (see @ref Importer, @ref createCaffeImporter(), @ref createTorchImporter()).

  Bult-in layers partially reproduce functionality of corresponding Caffe and Torch7 layers.
  In partuclar, the following layers and Caffe @ref Importer were tested to reproduce <a href="http://caffe.berkeleyvision.org/tutorial/layers.html">Caffe</a> functionality:
  - Convolution
  - Deconvolution
  - Pooling
  - InnerProduct
  - TanH, ReLU, Sigmoid, BNLL, Power, AbsVal
  - Softmax
  - Reshape, Flatten, Slice, Split
  - LRN
  - MVN
  - Dropout (since it does nothing on forward pass -))
*/

    //! LSTM recurrent layer
    class CV_EXPORTS_W LSTMLayer : public Layer
    {
    public:
        /** Creates instance of LSTM layer */
        static Ptr<LSTMLayer> create();

        /** Set trained weights for LSTM layer.
        LSTM behavior on each step is defined by current input, previous output, previous cell state and learned weights.

        Let @f$x_t@f$ be current input, @f$h_t@f$ be current output, @f$c_t@f$ be current state.
        Than current output and current cell state is computed as follows:
        @f{eqnarray*}{
        h_t &= o_t \odot tanh(c_t),               \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t, \\
        @f}
        where @f$\odot@f$ is per-element multiply operation and @f$i_t, f_t, o_t, g_t@f$ is internal gates that are computed using learned wights.

        Gates are computed as follows:
        @f{eqnarray*}{
        i_t &= sigmoid&(W_{xi} x_t + W_{hi} h_{t-1} + b_i), \\
        f_t &= sigmoid&(W_{xf} x_t + W_{hf} h_{t-1} + b_f), \\
        o_t &= sigmoid&(W_{xo} x_t + W_{ho} h_{t-1} + b_o), \\
        g_t &= tanh   &(W_{xg} x_t + W_{hg} h_{t-1} + b_g), \\
        @f}
        where @f$W_{x?}@f$, @f$W_{h?}@f$ and @f$b_{?}@f$ are learned weights represented as matrices:
        @f$W_{x?} \in R^{N_c \times N_x}@f$, @f$W_h? \in R^{N_c \times N_h}@f$, @f$b_? \in R^{N_c}@f$.

        For simplicity and performance purposes we use @f$ W_x = [W_{xi}; W_{xf}; W_{xo}, W_{xg}] @f$
        (i.e. @f$W_x@f$ is vertical contacentaion of @f$ W_{x?} @f$), @f$ W_x \in R^{4N_c x N_x} @f$.
        The same for @f$ W_h = [W_{hi}; W_{hf}; W_{ho}, W_{hg}], W_h \in R^{4N_c x N_h} @f$
        and for @f$ b = [b_i; b_f, b_o, b_g]@f$, @f$b \in R^{4N_c} @f$.

        @param Wh is matrix defining how previous output is transformed to internal gates (i.e. according to abovemtioned notation is @f$ W_h @f$)
        @param Wx is matrix defining how current input is transformed to internal gates (i.e. according to abovemtioned notation is @f$ W_x @f$)
        @param b  is bias vector (i.e. according to abovemtioned notation is @f$ b @f$)
        */
        virtual void setWeights(const Blob &Wh, const Blob &Wx, const Blob &b) = 0;

        /** In common case it uses three inputs (@f$x_t@f$, @f$h_{t-1}@f$ and @f$c_{t-1}@f$) to compute compute two outputs (@f$h_t@f$ and @f$c_t@f$).

        @param input could contain three inputs: @f$x_t@f$, @f$h_{t-1}@f$ and @f$c_{t-1}@f$.
        @param output contains computed outputs: @f$h_t@f$ and @f$c_t@f$.

        The first input @f$x_t@f$ is required.
        The second and third inputs are optional: if they weren't set than layer will use internal @f$h_{t-1}@f$ and @f$c_{t-1}@f$ from previous calls,
        but at the first call they will be filled by zeros.
        Size of the last dimension of @f$x_t@f$ must be @f$N_x@f$, (@f$N_h@f$ for @f$h_{t-1}@f$ and @f$N_c@f$ for @f$c_{t-1}@f$).
        Sizes of remainder dimensions could be any, but thay must be consistent among @f$x_t@f$, @f$h_{t-1}@f$ and @f$c_{t-1}@f$.
        */
        void forward(std::vector<Blob*> &input, std::vector<Blob> &output);
    };

    //! Classical recurrent layer
    class CV_EXPORTS_W RNNLayer : public Layer
    {
    public:
        /** Creates instance of RNNLayer */
        static Ptr<RNNLayer> create();

        /** Setups learned weights.

        Recurrent-layer behavior on each step is defined by current input x_t, previous state h_t and learned weights as follows:
        @f{eqnarray*}{
        h_t &= tanh&(W_{hh} h_{t-1} + W_{xh} x_t + b_h),  \\
        o_t &= tanh&(W_{ho} h_t + b_o),
        @f}

        @param Whh is @f$ W_{hh} @f$ matrix
        @param Wxh is @f$ W_{xh} @f$ matrix
        @param bh  is @f$ b_{h}  @f$ vector
        @param Who is @f$ W_{xo} @f$ matrix
        @param bo  is @f$ b_{o}  @f$ vector
        */
        virtual void setWeights(const Blob &Whh, const Blob &Wxh, const Blob &bh, const Blob &Who, const Blob &bo) = 0;

        /** Accepts two inputs @f$x_t@f$ and @f$h_{t-1}@f$ and compute two outputs @f$o_t@f$ and @f$h_t@f$.

        @param input could contain inputs @f$x_t@f$ and @f$h_{t-1}@f$.
        @param output should contain outputs @f$o_t@f$ and @f$h_t@f$.

        The first input @f$x_t@f$ is required whereas @f$h_{t-1}@f$ is optional.
        If the second input @f$h_{t-1}@f$ isn't specified a layer will use internal @f$h_{t-1}@f$ from the previous calls, at the first call @f$h_{t-1}@f$ will be filled by zeros.

        */
        void forward(std::vector<Blob*> &input, std::vector<Blob> &output);
    };

//! @}
//! @}

}
}
#endif
