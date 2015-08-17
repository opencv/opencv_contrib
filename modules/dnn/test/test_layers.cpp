#include "test_precomp.hpp"
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "npy_blob.hpp"

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static String _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/layers/") + filename;
}

static void testLayer(String basename, bool useCaffeModel = false)
{
    Blob inp = blobFromNPY(_tf("blob.npy"));
    Blob ref = blobFromNPY(_tf(basename + ".npy"));

    String prototxt = basename + ".prototxt";
    String caffemodel = basename + ".caffemodel";

    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf(prototxt), (useCaffeModel) ? _tf(caffemodel) : String());
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

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
     testLayer("layer_deconvolution", true);
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

    Ptr<Layer> rl = LayerRegister::createLayerInstance("Reshape", params);
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
    rng.fill(input.getMatRef(), RNG::UNIFORM, -1, 1);

    net.setBlob(".input", input);
    net.forward();
    Blob output = net.getBlob("output");

    normAssert(input, output);
}

}
