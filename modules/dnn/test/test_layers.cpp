#include "test_precomp.hpp"
#include <iostream>
#include "cnpy.h"

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

Blob loadNpyBlob(String name)
{
    cnpy::NpyArray npyBlob = cnpy::npy_load(getTestFile(name));

    Blob blob;
    blob.fill((int)npyBlob.shape.size(), (int*)&npyBlob.shape[0], CV_32F, npyBlob.data);

    npyBlob.destruct();
    return blob;
}

static void testLayer(String proto, String caffemodel = String())
{
    Blob inp = loadNpyBlob("blob.npy");
    Blob ref = loadNpyBlob(proto + ".caffe.npy");

    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile(proto), caffemodel);
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    net.setBlob("input", inp);
    net.forward();
    Blob out = net.getBlob("output");

    EXPECT_TRUE(isEqual(ref.shape(), out.shape()));

    Mat &mRef = ref.getMatRef();
    Mat &mOut = out.getMatRef();
    size_t N = ref.total();

    double normL1 = cvtest::norm(mRef, mOut, NORM_L1)/N;
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

}
