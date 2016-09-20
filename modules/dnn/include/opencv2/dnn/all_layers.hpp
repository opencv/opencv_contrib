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
        static CV_WRAP Ptr<LSTMLayer> create();

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
        @f$W_{x?} \in R^{N_h \times N_x}@f$, @f$W_{h?} \in R^{N_h \times N_h}@f$, @f$b_? \in R^{N_h}@f$.

        For simplicity and performance purposes we use @f$ W_x = [W_{xi}; W_{xf}; W_{xo}, W_{xg}] @f$
        (i.e. @f$W_x@f$ is vertical contacentaion of @f$ W_{x?} @f$), @f$ W_x \in R^{4N_h \times N_x} @f$.
        The same for @f$ W_h = [W_{hi}; W_{hf}; W_{ho}, W_{hg}], W_h \in R^{4N_h \times N_h} @f$
        and for @f$ b = [b_i; b_f, b_o, b_g]@f$, @f$b \in R^{4N_h} @f$.

        @param Wh is matrix defining how previous output is transformed to internal gates (i.e. according to abovemtioned notation is @f$ W_h @f$)
        @param Wx is matrix defining how current input is transformed to internal gates (i.e. according to abovemtioned notation is @f$ W_x @f$)
        @param b  is bias vector (i.e. according to abovemtioned notation is @f$ b @f$)
        */
        CV_WRAP virtual void setWeights(const Blob &Wh, const Blob &Wx, const Blob &b) = 0;

        /** @brief Specifies shape of output blob which will be [[`T`], `N`] + @p outTailShape.
          * @details If this parameter is empty or unset then @p outTailShape = [`Wh`.size(0)] will be used,
          * where `Wh` is parameter from setWeights().
          */
        CV_WRAP virtual void setOutShape(const BlobShape &outTailShape = BlobShape::empty()) = 0;

        /** @brief Set @f$ h_{t-1} @f$ value that will be used in next forward() calls.
          * @details By-default @f$ h_{t-1} @f$ is inited by zeros and updated after each forward() call.
          */
        CV_WRAP virtual void setH(const Blob &H) = 0;
        /** @brief Returns current @f$ h_{t-1} @f$ value (deep copy). */
        CV_WRAP virtual Blob getH() const = 0;

        /** @brief Set @f$ c_{t-1} @f$ value that will be used in next forward() calls.
          * @details By-default @f$ c_{t-1} @f$ is inited by zeros and updated after each forward() call.
          */
        CV_WRAP virtual void setC(const Blob &C) = 0;
        /** @brief Returns current @f$ c_{t-1} @f$ value (deep copy). */
        CV_WRAP virtual Blob getC() const = 0;

        /** @brief Specifies either interpet first dimension of input blob as timestamp dimenion either as sample.
          *
          * If flag is set to true then shape of input blob will be interpeted as [`T`, `N`, `[data dims]`] where `T` specifies number of timpestamps, `N` is number of independent streams.
          * In this case each forward() call will iterate through `T` timestamps and update layer's state `T` times.
          *
          * If flag is set to false then shape of input blob will be interpeted as [`N`, `[data dims]`].
          * In this case each forward() call will make one iteration and produce one timestamp with shape [`N`, `[out dims]`].
          */
        CV_WRAP virtual void setUseTimstampsDim(bool use = true) = 0;

        /** @brief If this flag is set to true then layer will produce @f$ c_t @f$ as second output.
         * @details Shape of the second output is the same as first output.
         */
        CV_WRAP virtual void setProduceCellOutput(bool produce = false) = 0;

        /** In common case it use single input with @f$x_t@f$ values to compute output(s) @f$h_t@f$ (and @f$c_t@f$).
         * @param input should contain packed values @f$x_t@f$
         * @param output contains computed outputs: @f$h_t@f$ (and @f$c_t@f$ if setProduceCellOutput() flag was set to true).
         *
         * If setUseTimstampsDim() is set to true then @p input[0] should has at least two dimensions with the following shape: [`T`, `N`, `[data dims]`],
         * where `T` specifies number of timpestamps, `N` is number of independent streams (i.e. @f$ x_{t_0 + t}^{stream} @f$ is stored inside @p input[0][t, stream, ...]).
         *
         * If setUseTimstampsDim() is set to fase then @p input[0] should contain single timestamp, its shape should has form [`N`, `[data dims]`] with at least one dimension.
         * (i.e. @f$ x_{t}^{stream} @f$ is stored inside @p input[0][stream, ...]).
        */
        void forward(std::vector<Blob*> &input, std::vector<Blob> &output);

        int inputNameToIndex(String inputName);

        int outputNameToIndex(String outputName);
    };

    //! Classical recurrent layer
    class CV_EXPORTS_W RNNLayer : public Layer
    {
    public:
        /** Creates instance of RNNLayer */
        static CV_WRAP Ptr<RNNLayer> create();

        /** Setups learned weights.

        Recurrent-layer behavior on each step is defined by current input @f$ x_t @f$, previous state @f$ h_t @f$ and learned weights as follows:
        @f{eqnarray*}{
        h_t &= tanh&(W_{hh} h_{t-1} + W_{xh} x_t + b_h),  \\
        o_t &= tanh&(W_{ho} h_t + b_o),
        @f}

        @param Wxh is @f$ W_{xh} @f$ matrix
        @param bh  is @f$ b_{h}  @f$ vector
        @param Whh is @f$ W_{hh} @f$ matrix
        @param Who is @f$ W_{xo} @f$ matrix
        @param bo  is @f$ b_{o}  @f$ vector
        */
        CV_WRAP virtual void setWeights(const Blob &Wxh, const Blob &bh, const Blob &Whh, const Blob &Who, const Blob &bo) = 0;

        /** @brief If this flag is set to true then layer will produce @f$ h_t @f$ as second output.
         * @details Shape of the second output is the same as first output.
         */
        CV_WRAP virtual void setProduceHiddenOutput(bool produce = false) = 0;

        /** Accepts two inputs @f$x_t@f$ and @f$h_{t-1}@f$ and compute two outputs @f$o_t@f$ and @f$h_t@f$.

        @param input should contain packed input @f$x_t@f$.
        @param output should contain output @f$o_t@f$ (and @f$h_t@f$ if setProduceHiddenOutput() is set to true).

        @p input[0] should have shape [`T`, `N`, `data_dims`] where `T` and `N` is number of timestamps and number of independent samples of @f$x_t@f$ respectively.

        @p output[0] will have shape [`T`, `N`, @f$N_o@f$], where @f$N_o@f$ is number of rows in @f$ W_{xo} @f$ matrix.

        If setProduceHiddenOutput() is set to true then @p output[1] will contain a Blob with shape [`T`, `N`, @f$N_h@f$], where @f$N_h@f$ is number of rows in @f$ W_{hh} @f$ matrix.
        */
        void forward(std::vector<Blob*> &input, std::vector<Blob> &output);
    };

    class CV_EXPORTS_W BaseConvolutionLayer : public Layer
    {
    public:

        CV_PROP_RW Size kernel, stride, pad, dilation;
    };

    class CV_EXPORTS_W ConvolutionLayer : public BaseConvolutionLayer
    {
    public:

        static CV_WRAP Ptr<BaseConvolutionLayer> create(Size kernel = Size(3, 3), Size stride = Size(1, 1), Size pad = Size(0, 0), Size dilation = Size(1, 1));
    };

    class CV_EXPORTS_W DeconvolutionLayer : public BaseConvolutionLayer
    {
    public:

        static CV_WRAP Ptr<BaseConvolutionLayer> create(Size kernel = Size(3, 3), Size stride = Size(1, 1), Size pad = Size(0, 0), Size dilation = Size(1, 1));
    };

    class CV_EXPORTS_W LRNLayer : public Layer
    {
    public:

        enum Type
        {
            CHANNEL_NRM,
            SPATIAL_NRM
        };
        CV_PROP_RW int type;

        CV_PROP_RW int size;
        CV_PROP_RW double alpha, beta;

        static CV_WRAP Ptr<LRNLayer> create(int type = LRNLayer::CHANNEL_NRM, int size = 5, double alpha = 1, double beta = 0.75);
    };

    class CV_EXPORTS_W PoolingLayer : public Layer
    {
    public:

        enum Type
        {
            MAX,
            AVE,
            STOCHASTIC
        };

        CV_PROP_RW int type;
        CV_PROP_RW Size kernel, stride, pad;
        CV_PROP_RW bool globalPooling;

        static CV_WRAP Ptr<PoolingLayer> create(int type = PoolingLayer::MAX, Size kernel = Size(2, 2), Size stride = Size(1, 1), Size pad = Size(0, 0));
        static CV_WRAP Ptr<PoolingLayer> createGlobal(int type = PoolingLayer::MAX);
    };

    class CV_EXPORTS_W SoftmaxLayer : public Layer
    {
    public:

        static CV_WRAP Ptr<SoftmaxLayer> create(int axis = 1);
    };

    class CV_EXPORTS_W InnerProductLayer : public Layer
    {
    public:
        CV_PROP_RW int axis;

        static CV_WRAP Ptr<InnerProductLayer> create(int axis = 1);
    };

    class CV_EXPORTS_W MVNLayer : public Layer
    {
    public:
        CV_PROP_RW double eps;
        CV_PROP_RW bool normVariance, acrossChannels;

        static CV_WRAP Ptr<MVNLayer> create(bool normVariance = true, bool acrossChannels = false, double eps = 1e-9);
    };

    /* Reshaping */

    class CV_EXPORTS_W ReshapeLayer : public Layer
    {
    public:
        CV_PROP_RW BlobShape newShapeDesc;
        CV_PROP_RW Range newShapeRange;

        static CV_WRAP Ptr<ReshapeLayer> create(const BlobShape &newShape, Range applyingRange = Range::all());
    };

    class CV_EXPORTS_W ConcatLayer : public Layer
    {
    public:
        int axis;

        static CV_WRAP Ptr<ConcatLayer> create(int axis = 1);
    };

    class CV_EXPORTS_W SplitLayer : public Layer
    {
    public:
        int outputsCount; //!< Number of copies that will be produced (is ignored when negative).

        static CV_WRAP Ptr<SplitLayer> create(int outputsCount = -1);
    };

    class CV_EXPORTS_W SliceLayer : public Layer
    {
    public:
        CV_PROP_RW int axis;
        CV_PROP std::vector<int> sliceIndices;

        static CV_WRAP Ptr<SliceLayer> create(int axis);
        static CV_WRAP Ptr<SliceLayer> create(int axis, const std::vector<int> &sliceIndices);
    };

    /* Activations */

    class CV_EXPORTS_W ReLULayer : public Layer
    {
    public:
        CV_PROP_RW double negativeSlope;

        static CV_WRAP Ptr<ReLULayer> create(double negativeSlope = 0);
    };

    class CV_EXPORTS_W TanHLayer : public Layer
    {
    public:
        static CV_WRAP Ptr<TanHLayer> create();
    };

    class CV_EXPORTS_W SigmoidLayer : public Layer
    {
    public:
        static CV_WRAP Ptr<SigmoidLayer> create();
    };

    class CV_EXPORTS_W BNLLLayer : public Layer
    {
    public:
        static CV_WRAP Ptr<BNLLLayer> create();
    };

    class CV_EXPORTS_W AbsLayer : public Layer
    {
    public:
        static CV_WRAP Ptr<AbsLayer> create();
    };

    class CV_EXPORTS_W PowerLayer : public Layer
    {
    public:
        CV_PROP_RW double power, scale, shift;

        static CV_WRAP Ptr<PowerLayer> create(double power = 1, double scale = 1, double shift = 0);
    };

    /* Layers using in semantic segmentation */

    class CV_EXPORTS_W CropLayer : public Layer
    {
    public:
        CV_PROP int startAxis;
        CV_PROP std::vector<int> offset;

        static Ptr<CropLayer> create(int start_axis, const std::vector<int> &offset);
    };

    class CV_EXPORTS_W EltwiseLayer : public Layer
    {
    public:
        enum EltwiseOp
        {
            PROD = 0,
            SUM = 1,
            MAX = 2,
        };

        static Ptr<EltwiseLayer> create(EltwiseOp op, const std::vector<int> &coeffs);
    };

//! @}
//! @}

}
}
#endif
