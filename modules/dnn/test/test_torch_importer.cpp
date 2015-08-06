#if 1 || defined(ENABLE_TORCH_IMPORTER) && ENABLE_TORCH_IMPORTER
#if 1 || defined(ENABLE_TORCH_TESTS) && ENABLE_TORCH_TESTS
#include "test_precomp.hpp"

namespace cvtest
{

using namespace std;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

template<typename TStr>
static std::string _tf(TStr filename)
{
    return (getOpenCVExtraDir() + "/dnn/torch/") + filename;
}

TEST(Torch_Importer, simple_read)
{
    Net net;
    Ptr<Importer> importer;

    ASSERT_NO_THROW( importer = createTorchImporter(_tf("net_simple_net.txt"), false) );
    ASSERT_TRUE( importer != NULL );
    importer->populateNet(net);
}

static void runTorchNet(String prefix, String outLayerName, bool isBinary)
{
    String suffix = (isBinary) ? ".dat" : ".txt";

    Net net;
    Ptr<Importer> importer;
    ASSERT_NO_THROW( importer = createTorchImporter(_tf(prefix + "_net" + suffix), isBinary) );
    ASSERT_TRUE(importer != NULL);
    //ASSERT_NO_THROW( importer->populateNet(net) );
    importer->populateNet(net);

    Blob inp, outRef;
    ASSERT_NO_THROW( inp = readTorchMat(_tf(prefix + "_input" + suffix), isBinary) );
    ASSERT_NO_THROW( outRef = readTorchMat(_tf(prefix + "_output" + suffix), isBinary) );

    net.setBlob(".0", inp);
    net.forward();
    Blob out = net.getBlob(outLayerName);

    std::cout << "inp " << inp.shape() << "\n";
    std::cout << "out " << out.shape() << "\n";
    std::cout << "ref " << outRef.shape() << "\n";

    normAssert(outRef, out);
}

TEST(Torch_Importer, run_convolution)
{
    runTorchNet("net_conv", "l1_Convolution", false);
}

TEST(Torch_Importer, run_pool_max)
{
    runTorchNet("net_pool_max", "l1_Pooling", false);
}

TEST(Torch_Importer, run_pool_ave)
{
    //TODO: fix
    //runTorchNet("net_pool_ave", "l1_Pooling", false);
}

TEST(Torch_Importer, run_reshape)
{
    runTorchNet("net_reshape", "l1_Reshape", false);
    runTorchNet("net_reshape_batch", "l1_Reshape", false);
}

TEST(Torch_Importer, run_linear)
{
    runTorchNet("net_linear_2d", "l1_InnerProduct", false);
}

TEST(Torch_Importer, run_paralel)
{
    //TODO: fix and add Reshape
    //runTorchNet("net_parallel", "l2_torchMerge", false);
}

TEST(Torch_Importer, run_concat)
{
    runTorchNet("net_concat", "l2_torchMerge", false);
}

}
#endif
#endif
