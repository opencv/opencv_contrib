#if defined(ENABLE_CAFFE_MODEL_TESTS)
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

TEST(Reproducibility_GoogLeNet, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("bvlc_googlenet.prototxt"), _tf("bvlc_googlenet.caffemodel"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    std::vector<Mat> inpMats;
    inpMats.push_back( imread(_tf("googlenet_0.jpg")) );
    inpMats.push_back( imread(_tf("googlenet_1.jpg")) );
    ASSERT_TRUE(!inpMats[0].empty() && !inpMats[1].empty());

    net.setBlob(".data", Blob(inpMats));
    net.forward();

    Blob out = net.getBlob("prob");
    Blob ref = blobFromNPY(_tf("googlenet_prob.npy"));
    normAssert(out, ref);
}

}
#endif