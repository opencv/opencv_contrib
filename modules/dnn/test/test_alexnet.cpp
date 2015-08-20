#if defined(ENABLE_CAFFE_MODEL_TESTS) && defined(ENABLE_CAFFE_ALEXNET_TEST) //AlexNet is disabled now
#include "test_precomp.hpp"
#include "npy_blob.hpp"

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Reproducibility_AlexNet, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("bvlc_alexnet.prototxt"), _tf("bvlc_alexnet.caffemodel"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    std::vector<Mat> inpMats;
    inpMats.push_back( imread(_tf("alexnet_0.png")) );
    inpMats.push_back( imread(_tf("alexnet_1.png")) );
    ASSERT_TRUE(!inpMats[0].empty() && !inpMats[1].empty());

    net.setBlob(".data", Blob(inpMats));
    net.forward();

    Blob out = net.getBlob("prob");
    Blob ref = blobFromNPY(_tf("alexnet.npy"));
    normAssert(ref, out, "prob");
}

}
#endif