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

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static String _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/layers/") + filename;
}

static void testLayer(String basename, bool useCaffeModel = false, bool useCommonInputBlob = true)
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
     testLayer("layer_softmax");
}

TEST(Layer_Test_LRN_spatial, Accuracy)
{
     testLayer("layer_lrn_spatial");
}

TEST(Layer_Test_LRN_channels, Accuracy)
{
     testLayer("layer_lrn_channels");
}

TEST(Layer_Test_Convolution, Accuracy)
{
     testLayer("layer_convolution", true);
}

//TODO: move this test into separate file
TEST(Layer_Test_Convolution, AccuracyOCL)
{
    if (cv::ocl::haveOpenCL())
    {
        cv::ocl::setUseOpenCL(true);
        testLayer("layer_convolution", true);
        cv::ocl::setUseOpenCL(false);
    }
}

TEST(Layer_Test_InnerProduct, Accuracy)
{
     testLayer("layer_inner_product", true);
}

TEST(Layer_Test_Pooling_max, Accuracy)
{
     testLayer("layer_pooling_max");
}

TEST(Layer_Test_Pooling_ave, Accuracy)
{
     testLayer("layer_pooling_ave");
}

TEST(Layer_Test_DeConvolution, Accuracy)
{
     testLayer("layer_deconvolution", true, false);
}

TEST(Layer_Test_MVN, Accuracy)
{
     testLayer("layer_mvn");
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

    EXPECT_EQ(outVec[0].shape(), BlobShape(Vec3i(4, 3, 2)));
}

TEST(Layer_Test_Reshape_Split_Slice, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("reshape_and_slice_routines.prototxt"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    Blob input(BlobShape(Vec2i(6, 12)));
    RNG rng(0);
    rng.fill(input.matRef(), RNG::UNIFORM, -1, 1);

    net.setBlob(".input", input);
    net.forward();
    Blob output = net.getBlob("output");

    normAssert(input, output);
}

enum RunLayerMode
{
    ALLOC_ONLY          = 1,
    FORWARD_ONLY        = 2,
    ALLOC_AND_FORWARD   = ALLOC_ONLY | FORWARD_ONLY
};

typedef Ptr<std::vector<Blob*> > PtrToVecPtrBlob;

PtrToVecPtrBlob
runLayer(Ptr<Layer> layer, std::vector<Blob> &inpBlobs, std::vector<Blob> &outBlobs, int mode=ALLOC_AND_FORWARD)
{
    PtrToVecPtrBlob inpPtrs( new std::vector<Blob*>() );
    inpPtrs->reserve(inpBlobs.size());
    for (size_t i = 0; i < inpBlobs.size(); i++)
        inpPtrs->push_back(&inpBlobs[i]);

    if (mode & ALLOC_ONLY) layer->allocate(*inpPtrs, outBlobs);
    if (mode & FORWARD_ONLY) layer->forward(*inpPtrs, outBlobs);

    return inpPtrs;
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

    EXPECT_EQ(2, outputs.size());
    EXPECT_EQ(outResShape, outputs[0].shape());
    EXPECT_EQ(outResShape, outputs[1].shape());

    EXPECT_EQ(outResShape, layer->getC().shape());
    EXPECT_EQ(outResShape, layer->getH().shape());

    EXPECT_EQ(0, layer->inputNameToIndex("x"));
    EXPECT_EQ(0, layer->outputNameToIndex("h"));
    EXPECT_EQ(1, layer->outputNameToIndex("c"));
}

TEST(Layer_LSTM_Test_Accuracy_Reference_with_, CaffeRecurrent)
{
    Ptr<LSTMLayer> layer = LSTMLayer::create();

    Blob Wx = blobFromNPY(_tf("lstm.prototxt.w_0.npy"));
    Blob Wh = blobFromNPY(_tf("lstm.prototxt.w_2.npy"));
    Blob b  = blobFromNPY(_tf("lstm.prototxt.w_1.npy"));
    layer->setWeights(Wh, Wx, b);

    Blob inp = blobFromNPY(_tf("blob.npy"));
    std::vector<Blob> inputs(1, inp), outputs;
    runLayer(layer, inputs, outputs);

    Blob &h_t_gathered = outputs[0];
    Blob h_t_reference = blobFromNPY(_tf("lstm.prototxt.h_1.npy"));

    normAssert(h_t_reference, h_t_gathered);
}


class Layer_RNN_Test : public ::testing::Test
{
public:
    int Nx, Nh, No;
    Blob Whh, Wxh, bh, Who, bo;
    Ptr<RNNLayer> layer;

    std::vector<Blob> inputs, outputs;
    std::vector<Blob*> inputsPtr;

    Layer_RNN_Test(int _Nx = 31, int _Nh = 64, int _No = 100)
    {
        Nx = _Nx;
        Nh = _Nh;
        No = _No;

        Whh = Blob(BlobShape(Nh, Nh));
        Wxh = Blob(BlobShape(Nh, Nx));
        bh  = Blob(BlobShape(Nh, 1));
        Who = Blob(BlobShape(No, Nh));
        bo  = Blob(BlobShape(No, 1));

        layer = RNNLayer::create();
        layer->setWeights(Whh, Wxh, bh, Who, bo);
    }

    void allocateAndForward()
    {
        inputsPtr.clear();
        for (size_t i = 0; i < inputs.size(); i++)
            inputsPtr.push_back(&inputs[i]);

        layer->allocate(inputsPtr, outputs);
        layer->forward(inputsPtr, outputs);
    }
};

TEST_F(Layer_RNN_Test, BasicTest_1)
{
    inputs.push_back(Blob(BlobShape(1, 2, 3, Nx)));
    allocateAndForward();

    EXPECT_EQ(outputs.size(), 2);
    EXPECT_EQ(outputs[0].shape(), BlobShape(1, 2, 3, No));
    EXPECT_EQ(outputs[1].shape(), BlobShape(1, 2, 3, Nh));
}

TEST_F(Layer_RNN_Test, BasicTest_2)
{
    inputs.push_back(Blob(BlobShape(1, 2, 3, Nx)));
    inputs.push_back(Blob(BlobShape(1, 2, 3, Nh)));
    allocateAndForward();

    EXPECT_EQ(outputs.size(), 2);
    EXPECT_EQ(outputs[0].shape(), BlobShape(1, 2, 3, No));
    EXPECT_EQ(outputs[1].shape(), BlobShape(1, 2, 3, Nh));
}

}
