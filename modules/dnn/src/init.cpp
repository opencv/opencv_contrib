#include "precomp.hpp"

#include "layers/concat_layer.hpp"
#include "layers/convolution_layer.hpp"
#include "layers/blank_layer.hpp"
#include "layers/elementwise_layers.hpp"
#include "layers/fully_connected_layer.hpp"
#include "layers/lrn_layer.hpp"
#include "layers/mvn_layer.hpp"
#include "layers/pooling_layer.hpp"
#include "layers/reshape_layer.hpp"
#include "layers/slice_layer.hpp"
#include "layers/softmax_layer.hpp"
#include "layers/split_layer.hpp"

namespace cv
{
namespace dnn
{

struct AutoInitializer
{
    bool status;

    AutoInitializer() : status(false)
    {
        cv::dnn::initModule();
    }
};

static AutoInitializer init;

void initModule()
{
    if (init.status)
        return;

    REG_RUNTIME_LAYER_CLASS(Slice, SliceLayer)
    REG_RUNTIME_LAYER_CLASS(Softmax, SoftMaxLayer)
    REG_RUNTIME_LAYER_CLASS(Split, SplitLayer)
    REG_RUNTIME_LAYER_CLASS(Reshape, ReshapeLayer)
    REG_STATIC_LAYER_FUNC(Flatten, createFlattenLayer)
    REG_RUNTIME_LAYER_CLASS(Pooling, PoolingLayer)
    REG_RUNTIME_LAYER_CLASS(MVN, MVNLayer)
    REG_RUNTIME_LAYER_CLASS(LRN, LRNLayer)
    REG_RUNTIME_LAYER_CLASS(InnerProduct, FullyConnectedLayer)

    REG_RUNTIME_LAYER_CLASS(ReLU, ElementWiseLayer<ReLUFunctor>)
    REG_RUNTIME_LAYER_CLASS(TanH, ElementWiseLayer<TanHFunctor>)
    REG_RUNTIME_LAYER_CLASS(BNLL, ElementWiseLayer<BNLLFunctor>)
    REG_RUNTIME_LAYER_CLASS(Power, ElementWiseLayer<PowerFunctor>)
    REG_RUNTIME_LAYER_CLASS(AbsVal, ElementWiseLayer<AbsValFunctor>)
    REG_RUNTIME_LAYER_CLASS(Sigmoid, ElementWiseLayer<SigmoidFunctor>)
    REG_RUNTIME_LAYER_CLASS(Dropout, BlankLayer)

    REG_RUNTIME_LAYER_CLASS(Convolution, ConvolutionLayer)
    REG_RUNTIME_LAYER_CLASS(Deconvolution, DeConvolutionLayer)
    REG_RUNTIME_LAYER_CLASS(Concat, ConcatLayer)

    init.status = true;
}

}
}
