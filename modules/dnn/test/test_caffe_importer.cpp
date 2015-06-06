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

TEST(ReadCaffePrototxt_gtsrb, Accuracy)
{
    Ptr<Importer> importer = createCaffeImporter(getOpenCVExtraDir() + "/dnn/gtsrb.prototxt", "");
    Ptr<NetConfiguration> config = NetConfiguration::create();
    importer->populateNetConfiguration(config);
}

TEST(ReadCaffePrototxt_GoogleNet, Accuracy)
{
    Ptr<Importer> importer = createCaffeImporter(getOpenCVExtraDir() + "/dnn/googlenet_deploy.prototxt", "");
    Ptr<NetConfiguration> config = NetConfiguration::create();
    importer->populateNetConfiguration(config);
}

}