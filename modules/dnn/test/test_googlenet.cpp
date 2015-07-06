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

TEST(Reproducibility_GoogLeNet, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile("bvlc_googlenet.prototxt"), getTestFile("bvlc_googlenet.caffemodel"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    std::vector<Mat> inpMats;
    inpMats.push_back( imread(getTestFile("googlenet_0.png")) );
    inpMats.push_back( imread(getTestFile("googlenet_1.png")) );
    ASSERT_TRUE(!inpMats[0].empty() && !inpMats[1].empty());

    Blob inp(inpMats);
    net.setBlob("data", inp);
    net.forward();

    Blob out = net.getBlob("prob");
    Blob ref = blobFromNPY(getTestFile("googlenet.npy"));
    normAssert(out, ref);
}

}
