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

#include "test_precomp.hpp"
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "npy_blob.hpp"
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/ts/ocl_test.hpp>

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static String _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/layers/") + filename;
}


enum RunLayerMode
{
    ALLOC_ONLY = 1,
    FORWARD_ONLY = 2,
    ALLOC_AND_FORWARD = ALLOC_ONLY | FORWARD_ONLY
};

typedef Ptr<std::vector<Blob*> > PtrToVecPtrBlob;

PtrToVecPtrBlob
runLayer(Ptr<Layer> layer, std::vector<Blob> &inpBlobs, std::vector<Blob> &outBlobs, int mode = ALLOC_AND_FORWARD)
{
    PtrToVecPtrBlob inpPtrs(new std::vector<Blob*>());
    inpPtrs->reserve(inpBlobs.size());
    for (size_t i = 0; i < inpBlobs.size(); i++)
        inpPtrs->push_back(&inpBlobs[i]);

    if (mode & ALLOC_ONLY) layer->allocate(*inpPtrs, outBlobs);
    if (mode & FORWARD_ONLY) layer->forward(*inpPtrs, outBlobs);

    return inpPtrs;
}


void testLayerUsingCaffeModels(String basename, bool useCaffeModel = false, bool useCommonInputBlob = true)
{
    String prototxt = _tf(basename + ".prototxt");
    String caffemodel = _tf(basename + ".caffemodel");

    String inpfile = (useCommonInputBlob) ? _tf("blob.npy") : _tf(basename + ".input.npy");
    String outfile = _tf(basename + ".npy");

    cv::setNumThreads(cv::getNumberOfCPUs());

    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(prototxt, (useCaffeModel) ? caffemodel : String());
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    Blob inp = blobFromNPY(inpfile);
    Blob ref = blobFromNPY(outfile);

    net.setBlob(".input", inp);
    net.forward();
    Blob out = net.getBlob("output");

    normAssert(ref, out);
}

TEST(Layer_Test_Softmax, Accuracy)
{
     OCL_OFF(testLayerUsingCaffeModels("layer_softmax"));
}
OCL_TEST(Layer_Test_Softmax, Accuracy)
{
     OCL_ON(testLayerUsingCaffeModels("layer_softmax"));
     OCL_OFF();
}

TEST(Layer_Test_LRN_spatial, Accuracy)
{
     OCL_OFF(testLayerUsingCaffeModels("layer_lrn_spatial"));
}
OCL_TEST(Layer_Test_LRN_spatial, Accuracy)
{
     OCL_ON(testLayerUsingCaffeModels("layer_lrn_spatial"));
     OCL_OFF();
}

TEST(Layer_Test_LRN_channels, Accuracy)
{
     OCL_OFF(testLayerUsingCaffeModels("layer_lrn_channels"));
}
OCL_TEST(Layer_Test_LRN_channels, Accuracy)
{
    OCL_ON(testLayerUsingCaffeModels("layer_lrn_channels"));
    OCL_OFF();
}

TEST(Layer_Test_Convolution, Accuracy)
{
     OCL_OFF(testLayerUsingCaffeModels("layer_convolution", true));
}
OCL_TEST(Layer_Test_Convolution, Accuracy)
{
     OCL_ON(testLayerUsingCaffeModels("layer_convolution", true));
     OCL_OFF();
}

TEST(Layer_Test_DeConvolution, Accuracy)
{
     OCL_OFF(testLayerUsingCaffeModels("layer_deconvolution", true, false));
}
OCL_TEST(Layer_Test_DeConvolution, Accuracy)
{
     OCL_ON(testLayerUsingCaffeModels("layer_deconvolution", true, false););
     OCL_OFF();
}

TEST(Layer_Test_InnerProduct, Accuracy)
{
     OCL_OFF(testLayerUsingCaffeModels("layer_inner_product", true));
}
OCL_TEST(Layer_Test_InnerProduct, Accuracy)
{
    OCL_ON(testLayerUsingCaffeModels("layer_inner_product", true));
    OCL_OFF();
}

TEST(Layer_Test_Pooling_max, Accuracy)
{
     OCL_OFF(testLayerUsingCaffeModels("layer_pooling_max"));
     OCL_ON();
}
OCL_TEST(Layer_Test_Pooling_max, Accuracy)
{
     OCL_ON(testLayerUsingCaffeModels("layer_pooling_max"));
     OCL_OFF();
}

TEST(Layer_Test_Pooling_ave, Accuracy)
{
     OCL_OFF(testLayerUsingCaffeModels("layer_pooling_ave"));
     OCL_ON();
}
OCL_TEST(Layer_Test_Pooling_ave, Accuracy)
{
     OCL_ON(testLayerUsingCaffeModels("layer_pooling_ave"));
     OCL_OFF();
}

TEST(Layer_Test_MVN, Accuracy)
{
     OCL_OFF(testLayerUsingCaffeModels("layer_mvn"));
}

TEST(Layer_Test_Reshape, squeeze)
{
    LayerParams params;
    params.set("axis", 2);
    params.set("num_axes", 1);

    Blob inp(BlobShape(4, 3, 1, 2));
    std::vector<Blob*> inpVec(1, &inp);
    std::vector<Blob> outVec;

    Ptr<Layer> rl = LayerFactory::createLayerInstance("Reshape", params);
    rl->allocate(inpVec, outVec);
    rl->forward(inpVec, outVec);

    EXPECT_EQ(outVec[0].shape(), BlobShape(4, 3, 2));
}

//template<typename XMat>
//static void test_Layer_Concat()
//{
//    Matx21f a(1.f, 1.f), b(2.f, 2.f), c(3.f, 3.f);
//    std::vector<Blob> res(1), src = { Blob(XMat(a)), Blob(XMat(b)), Blob(XMat(c)) };
//    Blob ref(XMat(Matx23f(1.f, 2.f, 3.f, 1.f, 2.f, 3.f)));
//
//    runLayer(ConcatLayer::create(1), src, res);
//    normAssert(ref, res[0]);
//}
//TEST(Layer_Concat, Accuracy)
//{
//    OCL_OFF(test_Layer_Concat<Mat>());
//}
//OCL_TEST(Layer_Concat, Accuracy)
//{
//    OCL_ON(test_Layer_Concat<Mat>());
//    OCL_OFF();
//}

template<typename XMat>
void test_Reshape_Split_Slice_layers()
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("reshape_and_slice_routines.prototxt"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    Blob input(BlobShape(6, 12));
    RNG rng(0);
    rng.fill(input.getRef<XMat>(), RNG::UNIFORM, -1, 1);

    net.setBlob(".input", input);
    net.forward();
    Blob output = net.getBlob("output");

    normAssert(input, output);
}
TEST(Layer_Test_Reshape_Split_Slice, Accuracy)
{
    OCL_OFF(test_Reshape_Split_Slice_layers<Mat>());
}
OCL_TEST(Layer_Test_Reshape_Split_Slice, Accuracy)
{
    OCL_ON(test_Reshape_Split_Slice_layers<UMat>());
    OCL_OFF();
}

class Layer_LSTM_Test : public ::testing::Test
{
public:
    int numInp, numOut;
    Blob Wh, Wx, b;
    Ptr<LSTMLayer> layer;
    std::vector<Blob> inputs, outputs;

    Layer_LSTM_Test() {}

    void init(const BlobShape &inpShape_, const BlobShape &outShape_)
    {
        numInp = inpShape_.total();
        numOut = outShape_.total();

        Wh = Blob(BlobShape(4 * numOut, numOut));
        Wx = Blob(BlobShape(4 * numOut, numInp));
        b  = Blob(BlobShape(4 * numOut, 1));

        layer = LSTMLayer::create();
        layer->setWeights(Wh, Wx, b);
        layer->setOutShape(outShape_);
    }
};

TEST_F(Layer_LSTM_Test, get_set_test)
{
    BlobShape TN(4);
    BlobShape inpShape(5, 3, 2), inpResShape = TN + inpShape;
    BlobShape outShape(3, 1, 2), outResShape = TN + outShape;

    init(inpShape, outShape);
    layer->setProduceCellOutput(true);
    layer->setUseTimstampsDim(false);
    layer->setOutShape(outShape);

    layer->setC(Blob(outResShape));
    layer->setH(Blob(outResShape));

    inputs.push_back(Blob(inpResShape));
    runLayer(layer, inputs, outputs);

    EXPECT_EQ(2u, outputs.size());
    EXPECT_EQ(outResShape, outputs[0].shape());
    EXPECT_EQ(outResShape, outputs[1].shape());

    EXPECT_EQ(outResShape, layer->getC().shape());
    EXPECT_EQ(outResShape, layer->getH().shape());

    EXPECT_EQ(0, layer->inputNameToIndex("x"));
    EXPECT_EQ(0, layer->outputNameToIndex("h"));
    EXPECT_EQ(1, layer->outputNameToIndex("c"));
}

TEST(Layer_LSTM_Test_Accuracy_with_, CaffeRecurrent)
{
    Ptr<LSTMLayer> layer = LSTMLayer::create();

    Blob Wx = blobFromNPY(_tf("lstm.prototxt.w_0.npy"));
    Blob Wh = blobFromNPY(_tf("lstm.prototxt.w_2.npy"));
    Blob b  = blobFromNPY(_tf("lstm.prototxt.w_1.npy"));
    layer->setWeights(Wh, Wx, b);

    Blob inp = blobFromNPY(_tf("recurrent.input.npy"));
    std::vector<Blob> inputs(1, inp), outputs;
    runLayer(layer, inputs, outputs);

    Blob h_t_reference = blobFromNPY(_tf("lstm.prototxt.h_1.npy"));
    normAssert(h_t_reference, outputs[0]);
}

TEST(Layer_RNN_Test_Accuracy_with_, CaffeRecurrent)
{
    Ptr<RNNLayer> layer = RNNLayer::create();

    layer->setWeights(
                blobFromNPY(_tf("rnn.prototxt.w_0.npy")),
                blobFromNPY(_tf("rnn.prototxt.w_1.npy")),
                blobFromNPY(_tf("rnn.prototxt.w_2.npy")),
                blobFromNPY(_tf("rnn.prototxt.w_3.npy")),
                blobFromNPY(_tf("rnn.prototxt.w_4.npy")) );

    std::vector<Blob> output, input(1, blobFromNPY(_tf("recurrent.input.npy")));
    runLayer(layer, input, output);

    Blob h_ref = blobFromNPY(_tf("rnn.prototxt.h_1.npy"));
    normAssert(h_ref, output[0]);
}


class Layer_RNN_Test : public ::testing::Test
{
public:
    int nX, nH, nO, nT, nS;
    Blob Whh, Wxh, bh, Who, bo;
    Ptr<RNNLayer> layer;

    std::vector<Blob> inputs, outputs;

    Layer_RNN_Test()
    {
        nT = 3;
        nS = 5;
        nX = 31;
        nH = 64;
        nO = 100;

        Whh = Blob(BlobShape(nH, nH));
        Wxh = Blob(BlobShape(nH, nX));
        bh  = Blob(BlobShape(nH, 1));
        Who = Blob(BlobShape(nO, nH));
        bo  = Blob(BlobShape(nO, 1));

        layer = RNNLayer::create();
        layer->setProduceHiddenOutput(true);
        layer->setWeights(Wxh, bh, Whh, Who, bo);
    }
};

TEST_F(Layer_RNN_Test, get_set_test)
{
    inputs.push_back(Blob(BlobShape(nT, nS, 1, nX)));
    runLayer(layer, inputs, outputs);

    EXPECT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0].shape(), BlobShape(nT, nS, nO));
    EXPECT_EQ(outputs[1].shape(), BlobShape(nT, nS, nH));
}

}
