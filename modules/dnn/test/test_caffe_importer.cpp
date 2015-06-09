#include "test_precomp.hpp"

namespace cvtest
{

using namespace std;
using namespace std::tr1;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

static std::string getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

static std::string getTestFile(const char *filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(ReadCaffePrototxt_gtsrb, Accuracy)
{
    Ptr<Importer> importer = createCaffeImporter(getTestFile("gtsrb.prototxt"), getTestFile("gtsrb_iter_36000.caffemodel") );
    Net net;
    importer->populateNet(net);
}

TEST(ReadCaffePrototxt_GoogleNet, Accuracy)
{
    Ptr<Importer> importer = createCaffeImporter(getOpenCVExtraDir() + "/dnn/googlenet_deploy.prototxt", "");
    Net net;
    importer->populateNet(net);
}

}