#include "test_precomp.hpp"
#include "npy_blob.hpp"

namespace cvtest
{

using namespace std;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

template<typename TString>
static std::string getTestFile(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Reproducibility_AlexNet, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile("bvlc_alexnet.prototxt"), getTestFile("bvlc_alexnet.caffemodel"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    std::vector<Mat> inpMats;
    inpMats.push_back( imread(getTestFile("alexnet_0.png")) );
    inpMats.push_back( imread(getTestFile("alexnet_1.png")) );
    ASSERT_TRUE(!inpMats[0].empty() && !inpMats[1].empty());

    net.setBlob(".data", Blob(inpMats));
    net.forward();

    Blob out = net.getBlob("prob");
    Blob ref = blobFromNPY(getTestFile("alexnet.npy"));
    normAssert(ref, out, "prob");
}

}
