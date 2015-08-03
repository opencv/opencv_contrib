#include "test_precomp.hpp"
#include <iostream>
#include "npy_blob.hpp"

namespace cvtest
{

using namespace std;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

static std::string getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

template<typename TStr>
static std::string getTestFile(TStr filename)
{
    return (getOpenCVExtraDir() + "/dnn/layers/") + filename;
}

template<typename T, int n>
bool isEqual(const cv::Vec<T, n> &l, const cv::Vec<T, n> &r)
{
    for (int i = 0; i < n; i++)
    {
        if (l[i] != r[i])
            return false;
    }
    return true;
}

static void testLayer(String proto, String caffemodel = String())
{
    Blob inp = blobFromNPY(getTestFile("blob.npy"));
    Blob ref = blobFromNPY(getTestFile(proto + ".caffe.npy"));

    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile(proto), caffemodel);
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    net.setBlob(".input", inp);
    net.forward();
    Blob out = net.getBlob("output");

    EXPECT_EQ(ref.shape(), out.shape());

    Mat &mRef = ref.getMatRef();
    Mat &mOut = out.getMatRef();

    double normL1 = cvtest::norm(mRef, mOut, NORM_L1) / ref.total();
    EXPECT_LE(normL1, 0.0001);

    double normInf = cvtest::norm(mRef, mOut, NORM_INF);
    EXPECT_LE(normInf, 0.0001);
}

TEST(Layer_Softmax_Test, Accuracy)
{
     testLayer("softmax.prototxt");
}

TEST(Layer_LRN_spatial_Test, Accuracy)
{
     testLayer("lrn_spatial.prototxt");
}

TEST(Layer_LRN_channels_Test, Accuracy)
{
     testLayer("lrn_channels.prototxt");
}

TEST(Layer_Reshape_Split_Slice_Test, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile("reshape_and_slice_routines.prototxt"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    BlobShape shape = BlobShape(Vec2i(6, 12));

    Mat1f inputMat(shape[0], shape[1]);
    RNG rng(0);
    rng.fill(inputMat, RNG::UNIFORM, -1, 1);

    Blob input(inputMat);
    input.reshape(shape);
    net.setBlob(".input", input);
    net.forward();
    Blob output = net.getBlob("output");

    input.fill(shape, CV_32F, inputMat.data);
    normAssert(input, output);
}

}
