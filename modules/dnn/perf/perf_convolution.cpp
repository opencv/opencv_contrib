#include "perf_precomp.hpp"

namespace cvtest
{

using std::tr1::tuple;
using std::tr1::get;
using std::tr1::make_tuple;
using std::make_pair;
using namespace perf;
using namespace testing;
using namespace cv;
using namespace cv::dnn;

enum {STRIDE_OFF = 1, STRIDE_ON = 2};
CV_ENUM(StrideSize, STRIDE_OFF, STRIDE_ON);

enum {GROUP_OFF = 1, GROUP_2 = 2};
CV_ENUM(GroupSize, GROUP_OFF, GROUP_2);

//Squared Size
#define SSZ(n) cv::Size(n, n)

typedef std::pair<BlobShape, int> InpShapeNumOut;
typedef tuple<Size, InpShapeNumOut, GroupSize, StrideSize> ConvParam; //kernel_size, inp shape, groups, stride
typedef TestBaseWithParam<ConvParam> ConvolutionPerfTest;

PERF_TEST_P( ConvolutionPerfTest, perf, Combine(
    Values(Size(1, 1), Size(3, 3), Size(5, 5), Size(11, 11)),
    Values(make_pair(BlobShape(1,   4, 224, 224),  64),
           make_pair(BlobShape(1,  64, 112, 122), 128),
           make_pair(BlobShape(1, 256,  28,  28), 512)),
    GroupSize::all(),
    StrideSize::all())
)
{
    RNG rng(0);

    ConvParam params = GetParam();
    int ksz     = get<0>(params).width;
    BlobShape inpShape = get<1>(params).first;
    int outCn   = get<1>(params).second;
    int groups  = get<2>(params);
    int stride  = (ksz >= 11) ? 4 : (int)get<3>(params);

    int inpCn = inpShape[1];
    Blob wgtBlob(BlobShape(outCn, inpCn/groups, ksz, ksz)), biasBlob(BlobShape(outCn, 1, 1, 1));
    Blob inpBlob(inpShape);
    rng.fill(biasBlob.matRef(), RNG::UNIFORM, -1, +1);
    rng.fill(wgtBlob.matRef(), RNG::UNIFORM, -1, +1);
    rng.fill(inpBlob.matRef(), RNG::UNIFORM, -1, +1);

    LayerParams lp;
    lp.set("num_output", outCn);
    lp.set("group", groups);
    lp.set("stride", stride);
    lp.set("kernel_size", ksz);
    lp.blobs.reserve(2);
    lp.blobs.push_back(wgtBlob);
    lp.blobs.push_back(biasBlob);

    std::vector<Blob*> inpBlobs(1, &inpBlob);
    std::vector<Blob> outBlobs;

    cv::setNumThreads(cv::getNumberOfCPUs());

    Ptr<Layer> layer = cv::dnn::LayerFactory::createLayerInstance("Convolution", lp);
    layer->allocate(inpBlobs, outBlobs);

    declare.in(inpBlob.matRef(), wgtBlob.matRef(), WARMUP_RNG).out(outBlobs[0].matRef()).tbb_threads(cv::getNumThreads());

    TEST_CYCLE_N(10)
    {
        layer->forward(inpBlobs, outBlobs);
    }

    SANITY_CHECK_NOTHING();
}

}