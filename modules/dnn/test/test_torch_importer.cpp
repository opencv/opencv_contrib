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
    //return (getOpenCVExtraDir() + "/dnn/") + filename;
    return String("/home/vitaliy/th/") + filename;
}

TEST(Torch_Importer, simple_read)
{
    Net net;
    Ptr<Importer> importer;

    ASSERT_NO_THROW( importer = createTorchImporter(getTestFile("conv1.txt"), false) );
    ASSERT_TRUE( importer != NULL );
    ASSERT_NO_THROW( importer->populateNet(net) );
}

static Blob convertBlob(const Blob &inp, int type)
{
    Mat tmpMat;
    inp.getMatRef().convertTo(tmpMat, type);

    Blob res;
    res.create(inp.shape(), type);
    res.fill(inp.shape(), type, (void*)tmpMat.data);
    return res;
}

TEST(Torch_Importer, run_convolution)
{
    Net net;
    Ptr<Importer> importer = createTorchImporter(getTestFile("run_conv_net.txt"), false);
    ASSERT_TRUE(importer != NULL);
    importer->populateNet(net);

    Blob inp = convertBlob( readTorchMat(getTestFile("run_conv_input.txt"), false), CV_32F );
    Blob outRef = convertBlob( readTorchMat(getTestFile("run_conv_output.txt"), false), CV_32F );

    net.setBlob(".0", inp);
    net.forward();
    Blob out = net.getBlob("l1_Convolution");

    normAssert(outRef, out);
}

}
#endif
