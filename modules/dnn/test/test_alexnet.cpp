#include "test_precomp.hpp"
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
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

inline void normAssert(InputArray ref, InputArray get, const char *comment = "")
{
    double normL1 = cvtest::norm(ref, get, NORM_L1)/ ref.getMat().total();
    EXPECT_LE(normL1, 0.0001) << comment;

    double normInf = cvtest::norm(ref, get, NORM_INF);
    EXPECT_LE(normInf, 0.001) << comment;
}

inline void normAssert(Blob ref, Blob test, const char *comment = "")
{
    normAssert(ref.getMatRef(), test.getMatRef(), comment);
}

TEST(Reproducibility_AlexNet, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile("bvlc_alexnet.prototxt"), getTestFile("bvlc_alexnet.caffemodel"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    std::vector<Mat> inpMats(2);
    inpMats[0] = imread(getTestFile("alexnet_0.png"));
    inpMats[1] = imread(getTestFile("alexnet_1.png"));
    ASSERT_TRUE(!inpMats[0].empty() && !inpMats[1].empty());

    inpMats[0].convertTo(inpMats[0], CV_32F);
    Blob inp(inpMats[0]);

    net.setBlob("data", inp);

    net.forward("conv1");
    normAssert(blobFromNPY(getTestFile("alexnet_conv1.npy")), net.getBlob("conv1"), "conv1");
    //saveBlobToNPY(convBlob, getTestFile("alexnet_conv1_my.npy"));

    net.forward("relu1");
    normAssert(blobFromNPY(getTestFile("alexnet_relu1.npy")), net.getBlob("relu1"), "relu1");

    net.forward("norm1");
    normAssert(blobFromNPY(getTestFile("alexnet_norm1.npy")), net.getBlob("norm1"), "norm1");

    net.forward();
    Blob out = net.getBlob("prob");
    Blob ref = blobFromNPY(getTestFile("alexnet.npy"));
    std::cout << out.shape() << " vs " << ref.shape() << std::endl;
    Mat mOut(1, 1000, CV_32F, ref.rawPtr());
    Mat mRef(1, 1000, CV_32F, ref.rawPtr());
    normAssert(mOut, mRef);
}

}
