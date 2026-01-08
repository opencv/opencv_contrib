#include "test_precomp.hpp"
#include "opencv2/cann_call.hpp"

namespace opencv_test
{
namespace
{

TEST(ASCENDC_KERNEL, THRESHOLD)
{
    cv::cann::setDevice(DEVICE_ID);
    Mat cpuRet, npuRet;
    AscendMat npuImg, npuTmpMat;

    // opencv do not support CV_8S, CV_32S, CV_16F
    // ascend do not support CV_16U, CV_64F
    uint8_t dtypes[] = {CV_8U, CV_16S, CV_32F};

    for (uint i = 0; i <= 4; i++)
    {
        for (uint j = 0; j < sizeof(dtypes) / sizeof(dtypes[0]); j++)
        {
            double thresh = 90.5;
            double maxVal = 85.2;

            Mat img = randomMat(10, 10, CV_MAKETYPE(dtypes[j], 3), 0.0f, 128.0f);
            npuImg.upload(img);
            npuTmpMat.create(npuImg.rows, npuImg.cols, npuImg.type());

            cv::threshold(img, cpuRet, thresh, maxVal, i);
            ThresholdOpencvTilingData tiling;
            tiling.maxVal = maxVal;
            tiling.thresh = thresh;
            size_t totalBytes = img.rows * img.cols * img.channels();
            // AscendMat memory will be align to 32B, it's safe to set totalLengh a little bigger.
            tiling.totalLength = ((totalBytes + 32) & ~31);
            tiling.threshType = i;
            tiling.dtype = dtypes[j];
            kernel_launch(aclrtlaunch_threshold_opencv, AscendStream::Null(), tiling,
                          npuImg.data.get(), npuTmpMat.data.get());

            npuTmpMat.download(npuRet);
            EXPECT_MAT_NEAR(cpuRet, npuRet, 10.0f);
        }
    }

    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
