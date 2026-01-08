// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

namespace opencv_test { namespace {

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Moments

CV_ENUM(MaxMomentsOrder, MomentsOrder::FIRST_ORDER_MOMENTS, MomentsOrder::SECOND_ORDER_MOMENTS, MomentsOrder::THIRD_ORDER_MOMENTS)

PARAM_TEST_CASE(Moments, cv::cuda::DeviceInfo, cv::Size, bool, MatDepth, MatDepth, UseRoi, MaxMomentsOrder)
{
    DeviceInfo devInfo;
    Size size;
    bool isBinary;
    float pcWidth = 0.6f;
    int momentsType;
    int imgType;
    bool useRoi;
    MomentsOrder order;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        isBinary = GET_PARAM(2);
        momentsType = GET_PARAM(3);
        imgType = GET_PARAM(4);
        useRoi = GET_PARAM(5);
        order = static_cast<MomentsOrder>(static_cast<int>(GET_PARAM(6)));
        cv::cuda::setDevice(devInfo.deviceID());
    }

    static void drawCircle(cv::Mat& dst, const cv::Vec3i& circle, bool fill)
    {
        dst.setTo(Scalar::all(0));
        cv::circle(dst, Point2i(circle[0], circle[1]), circle[2], Scalar::all(255), fill ? -1 : 1, cv::LINE_AA);
    }
};

bool Equal(const double m0, const double m1, const double absPcErr) {
    if (absPcErr == 0) return m0 == m1;
    if (m0 == 0) {
        if (m1 < absPcErr) return true;
        else return false;
    }
    const double pcDiff = abs(m0 - m1) / m1;
    return pcDiff < absPcErr;
}

void CheckMoments(const cv::Moments m0, const cv::Moments m1, const MomentsOrder order, const int momentsType) {
    double absPcErr = momentsType == CV_64F ? 0 : 5e-7;
    ASSERT_TRUE(Equal(m0.m00, m1.m00, absPcErr)) << "m0.m00: " << m0.m00 << ", m1.m00: " << m1.m00 << ", absPcErr: " << absPcErr;
    ASSERT_TRUE(Equal(m0.m10, m1.m10, absPcErr)) << "m0.m10: " << m0.m10 << ", m1.m10: " << m1.m10 << ", absPcErr: " << absPcErr;
    ASSERT_TRUE(Equal(m0.m01, m1.m01, absPcErr)) << "m0.m01: " << m0.m01 << ", m1.m01: " << m1.m01 << ", absPcErr: " << absPcErr;
    if (static_cast<int>(order) >= static_cast<int>(MomentsOrder::SECOND_ORDER_MOMENTS)) {
        ASSERT_TRUE(Equal(m0.m20, m1.m20, absPcErr)) << "m0.m20: " << m0.m20 << ", m1.m20: " << m1.m20 << ", absPcErr: " << absPcErr;
        ASSERT_TRUE(Equal(m0.m11, m1.m11, absPcErr)) << "m0.m11: " << m0.m11 << ", m1.m11: " << m1.m11 << ", absPcErr: " << absPcErr;
        ASSERT_TRUE(Equal(m0.m02, m1.m02, absPcErr)) << "m0.m02: " << m0.m02 << ", m1.m02: " << m1.m02 << ", absPcErr: " << absPcErr;
    }
    if (static_cast<int>(order) >= static_cast<int>(MomentsOrder::THIRD_ORDER_MOMENTS)) {
        ASSERT_TRUE(Equal(m0.m30, m1.m30, absPcErr)) << "m0.m30: " << m0.m30 << ", m1.m30: " << m1.m30 << ", absPcErr: " << absPcErr;
        ASSERT_TRUE(Equal(m0.m21, m1.m21, absPcErr)) << "m0.m21: " << m0.m21 << ", m1.m21: " << m1.m21 << ", absPcErr: " << absPcErr;
        ASSERT_TRUE(Equal(m0.m12, m1.m12, absPcErr)) << "m0.m12: " << m0.m12 << ", m1.m12: " << m1.m12 << ", absPcErr: " << absPcErr;
        ASSERT_TRUE(Equal(m0.m03, m1.m03, absPcErr)) << "m0.m03: " << m0.m03 << ", m1.m03: " << m1.m03 << ", absPcErr: " << absPcErr;
    }
}

CUDA_TEST_P(Moments, Accuracy)
{
    Mat imgHost(size, imgType);
    const Rect roi = useRoi ? Rect(1, 0, imgHost.cols - 2, imgHost.rows) : Rect(0, 0, imgHost.cols, imgHost.rows);
    const Vec3i circle(size.width / 2, size.height / 2, static_cast<int>(static_cast<float>(size.width/2) * pcWidth));
    drawCircle(imgHost, circle, true);
    const GpuMat imgDevice(imgHost);
    const int nMoments = numMoments(order);
    setBufferPoolUsage(true);
    setBufferPoolConfig(getDevice(), nMoments * ((momentsType == CV_64F) ? sizeof(double) : sizeof(float)), 1);
    const cv::Moments moments = cuda::moments(imgDevice(roi), isBinary, order, momentsType);
    Mat imgHostFloat; imgHost(roi).convertTo(imgHostFloat, CV_32F);
    const cv::Moments momentsGs = cv::moments(imgHostFloat, isBinary);
    CheckMoments(momentsGs, moments, order, momentsType);
}

CUDA_TEST_P(Moments, Async)
{
    Stream stream;
    const int nMoments = numMoments(order);
    GpuMat momentsDevice(1, nMoments, momentsType);
    Mat imgHost(size, imgType);
    const Rect roi = useRoi ? Rect(1, 0, imgHost.cols - 2, imgHost.rows) : Rect(0, 0, imgHost.cols, imgHost.rows);
    const Vec3i circle(size.width / 2, size.height / 2, static_cast<int>(static_cast<float>(size.width/2) * pcWidth));
    drawCircle(imgHost, circle, true);
    const GpuMat imgDevice(imgHost);
    cuda::spatialMoments(imgDevice(roi), momentsDevice, isBinary, order, momentsType, stream);
    HostMem momentsHost(1, nMoments, momentsType);
    momentsDevice.download(momentsHost, stream);
    stream.waitForCompletion();
    const cv::Moments moments  = convertSpatialMoments(momentsHost.createMatHeader(), order, momentsType);
    Mat imgHostAdjustedType = imgHost(roi);
    if (imgType != CV_8U && imgType != CV_32F)
        imgHost(roi).convertTo(imgHostAdjustedType, CV_32F);
    const cv::Moments momentsGs = cv::moments(imgHostAdjustedType, isBinary);
    CheckMoments(momentsGs, moments, order, momentsType);
}

#define SIZES DIFFERENT_SIZES
#define GRAYSCALE_BINARY testing::Bool()
#define MOMENTS_TYPE testing::Values(MatDepth(CV_32F), MatDepth(CV_64F))
#define IMG_TYPE ALL_DEPTH
#define USE_ROI WHOLE_SUBMAT
#define MOMENTS_ORDER testing::Values(MaxMomentsOrder(MomentsOrder::FIRST_ORDER_MOMENTS), MaxMomentsOrder(MomentsOrder::SECOND_ORDER_MOMENTS), MaxMomentsOrder(MomentsOrder::THIRD_ORDER_MOMENTS))
INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, Moments, testing::Combine(ALL_DEVICES, SIZES, GRAYSCALE_BINARY, MOMENTS_TYPE, IMG_TYPE, USE_ROI, MOMENTS_ORDER));
}} // namespace

#endif // HAVE_CUDA
