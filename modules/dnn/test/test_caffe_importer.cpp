#include "test_precomp.hpp"

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

TEST(ReadCaffe_GTSRB, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile("gtsrb.prototxt"), "");
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }
}

TEST(ReadCaffe_GoogLeNet, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(getTestFile("bvlc_googlenet.prototxt"), "");
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }
}

}
