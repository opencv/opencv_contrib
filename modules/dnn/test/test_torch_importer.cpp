#if 1 || defined(ENABLE_TORCH_IMPORTER) && ENABLE_TORCH_IMPORTER
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

TEST(Torch_Importer, simple_read)
{
    Net net;
    Ptr<Importer> importer;

    //ASSERT_NO_THROW( importer = createTorchImporter("/home/vitaliy/th/conv1.txt", false) );
    ASSERT_NO_THROW( importer = createTorchImporter("L:\\home\\vitaliy\\th\\conv1.txt", false) );
    ASSERT_TRUE( importer != NULL );
    importer->populateNet(net);
    //ASSERT_NO_THROW( importer->populateNet(net) );
}

}
#endif
