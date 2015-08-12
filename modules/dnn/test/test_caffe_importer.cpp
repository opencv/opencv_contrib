#include "test_precomp.hpp"

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(ReadCaffe_GTSRB, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("gtsrb.prototxt"), "");
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }
}

TEST(ReadCaffe_GoogLeNet, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("bvlc_googlenet.prototxt"), "");
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }
}

}
