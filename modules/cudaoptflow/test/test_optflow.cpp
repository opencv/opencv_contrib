/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

namespace opencv_test { namespace {

//////////////////////////////////////////////////////
// BroxOpticalFlow

//#define BROX_DUMP

struct BroxOpticalFlow : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(BroxOpticalFlow, Regression)
{
    cv::Mat frame0 = readImageType("opticalflow/frame0.png", CV_32FC1);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImageType("opticalflow/frame1.png", CV_32FC1);
    ASSERT_FALSE(frame1.empty());

    cv::Ptr<cv::cuda::BroxOpticalFlow> brox =
            cv::cuda::BroxOpticalFlow::create(0.197 /*alpha*/, 50.0 /*gamma*/, 0.8 /*scale_factor*/,
                                              10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

    cv::cuda::GpuMat flow;
    brox->calc(loadMat(frame0), loadMat(frame1), flow);

    cv::cuda::GpuMat flows[2];
    cv::cuda::split(flow, flows);

    cv::cuda::GpuMat u = flows[0];
    cv::cuda::GpuMat v = flows[1];

    std::string fname(cvtest::TS::ptr()->get_data_path());
    if (devInfo.majorVersion() >= 2)
        fname += "opticalflow/brox_optical_flow_cc20.bin";
    else
        fname += "opticalflow/brox_optical_flow.bin";

#ifndef BROX_DUMP
    std::ifstream f(fname.c_str(), std::ios_base::binary);

    int rows, cols;

    f.read((char*) &rows, sizeof(rows));
    f.read((char*) &cols, sizeof(cols));

    cv::Mat u_gold(rows, cols, CV_32FC1);

    for (int i = 0; i < u_gold.rows; ++i)
        f.read(u_gold.ptr<char>(i), u_gold.cols * sizeof(float));

    cv::Mat v_gold(rows, cols, CV_32FC1);

    for (int i = 0; i < v_gold.rows; ++i)
        f.read(v_gold.ptr<char>(i), v_gold.cols * sizeof(float));

    EXPECT_MAT_SIMILAR(u_gold, u, 1e-3);
    EXPECT_MAT_SIMILAR(v_gold, v, 1e-3);
#else
    std::ofstream f(fname.c_str(), std::ios_base::binary);

    f.write((char*) &u.rows, sizeof(u.rows));
    f.write((char*) &u.cols, sizeof(u.cols));

    cv::Mat h_u(u);
    cv::Mat h_v(v);

    for (int i = 0; i < u.rows; ++i)
        f.write(h_u.ptr<char>(i), u.cols * sizeof(float));

    for (int i = 0; i < v.rows; ++i)
        f.write(h_v.ptr<char>(i), v.cols * sizeof(float));
#endif
}

CUDA_TEST_P(BroxOpticalFlow, OpticalFlowNan)
{
    cv::Mat frame0 = readImageType("opticalflow/frame0.png", CV_32FC1);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImageType("opticalflow/frame1.png", CV_32FC1);
    ASSERT_FALSE(frame1.empty());

    cv::Mat r_frame0, r_frame1;
    cv::resize(frame0, r_frame0, cv::Size(1380,1000));
    cv::resize(frame1, r_frame1, cv::Size(1380,1000));

    cv::Ptr<cv::cuda::BroxOpticalFlow> brox =
            cv::cuda::BroxOpticalFlow::create(0.197 /*alpha*/, 50.0 /*gamma*/, 0.8 /*scale_factor*/,
                                              10 /*inner_iterations*/, 77 /*outer_iterations*/, 10 /*solver_iterations*/);

    cv::cuda::GpuMat flow;
    brox->calc(loadMat(frame0), loadMat(frame1), flow);

    cv::cuda::GpuMat flows[2];
    cv::cuda::split(flow, flows);

    cv::cuda::GpuMat u = flows[0];
    cv::cuda::GpuMat v = flows[1];

    cv::Mat h_u, h_v;
    u.download(h_u);
    v.download(h_v);

    EXPECT_TRUE(cv::checkRange(h_u));
    EXPECT_TRUE(cv::checkRange(h_v));
};

INSTANTIATE_TEST_CASE_P(CUDA_OptFlow, BroxOpticalFlow, ALL_DEVICES);

//////////////////////////////////////////////////////
// PyrLKOpticalFlow

namespace
{
    IMPLEMENT_PARAM_CLASS(Chan, int)
    IMPLEMENT_PARAM_CLASS(DataType, int)
}

PARAM_TEST_CASE(PyrLKOpticalFlow, cv::cuda::DeviceInfo, Chan, DataType)
{
    cv::cuda::DeviceInfo devInfo;
    int channels;
    int dataType;
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        channels = GET_PARAM(1);
        dataType = GET_PARAM(2);
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(PyrLKOpticalFlow, Sparse)
{
    cv::Mat frame0 = readImage("opticalflow/frame0.png", channels == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("opticalflow/frame1.png", channels == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    ASSERT_FALSE(frame1.empty());

    cv::Mat gray_frame;
    if (channels == 1)
        gray_frame = frame0;
    else
        cv::cvtColor(frame0, gray_frame, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(gray_frame, pts, 1000, 0.01, 0.0);

    cv::cuda::GpuMat d_pts;
    cv::Mat pts_mat(1, (int) pts.size(), CV_32FC2, (void*) &pts[0]);
    d_pts.upload(pts_mat);

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> pyrLK =
            cv::cuda::SparsePyrLKOpticalFlow::create();

    std::vector<cv::Point2f> nextPts_gold;
    std::vector<unsigned char> status_gold;
    cv::calcOpticalFlowPyrLK(frame0, frame1, pts, nextPts_gold, status_gold, cv::noArray());


    cv::cuda::GpuMat d_nextPts;
    cv::cuda::GpuMat d_status;
    cv::Mat converted0, converted1;
    if(channels == 4)
    {
        cv::cvtColor(frame0, frame0, cv::COLOR_BGR2BGRA);
        cv::cvtColor(frame1, frame1, cv::COLOR_BGR2BGRA);
    }
    frame0.convertTo(converted0, dataType);
    frame1.convertTo(converted1, dataType);

    pyrLK->calc(loadMat(converted0), loadMat(converted1), d_pts, d_nextPts, d_status);

    std::vector<cv::Point2f> nextPts(d_nextPts.cols);
    cv::Mat nextPts_mat(1, d_nextPts.cols, CV_32FC2, (void*)&nextPts[0]);
    d_nextPts.download(nextPts_mat);

    std::vector<unsigned char> status(d_status.cols);
    cv::Mat status_mat(1, d_status.cols, CV_8UC1, (void*)&status[0]);
    d_status.download(status_mat);

    ASSERT_EQ(nextPts_gold.size(), nextPts.size());
    ASSERT_EQ(status_gold.size(), status.size());

    size_t mistmatch = 0;
    for (size_t i = 0; i < nextPts.size(); ++i)
    {
        cv::Point2i a = nextPts[i];
        cv::Point2i b = nextPts_gold[i];

        if (status[i] != status_gold[i])
        {
            ++mistmatch;
            continue;
        }

        if (status[i])
        {
            bool eq = std::abs(a.x - b.x) <= 1 && std::abs(a.y - b.y) <= 1;

            if (!eq)
                ++mistmatch;
        }
    }

    double bad_ratio = static_cast<double>(mistmatch) / nextPts.size();

    ASSERT_LE(bad_ratio, 0.01);


}

INSTANTIATE_TEST_CASE_P(CUDA_OptFlow, PyrLKOpticalFlow, testing::Combine(
    ALL_DEVICES,
    testing::Values(Chan(1), Chan(3), Chan(4)),
    testing::Values(DataType(CV_8U), DataType(CV_16U), DataType(CV_32S), DataType(CV_32F))));



//////////////////////////////////////////////////////
// FarnebackOpticalFlow

namespace
{
    IMPLEMENT_PARAM_CLASS(PyrScale, double)
    IMPLEMENT_PARAM_CLASS(PolyN, int)
    CV_FLAGS(FarnebackOptFlowFlags, 0, OPTFLOW_FARNEBACK_GAUSSIAN)
    IMPLEMENT_PARAM_CLASS(UseInitFlow, bool)
}

PARAM_TEST_CASE(FarnebackOpticalFlow, cv::cuda::DeviceInfo, PyrScale, PolyN, FarnebackOptFlowFlags, UseInitFlow)
{
    cv::cuda::DeviceInfo devInfo;
    double pyrScale;
    int polyN;
    int flags;
    bool useInitFlow;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        pyrScale = GET_PARAM(1);
        polyN = GET_PARAM(2);
        flags = GET_PARAM(3);
        useInitFlow = GET_PARAM(4);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(FarnebackOpticalFlow, Accuracy)
{
    cv::Mat frame0 = readImage("opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("opticalflow/rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    double polySigma = polyN <= 5 ? 1.1 : 1.5;

    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn =
            cv::cuda::FarnebackOpticalFlow::create();
    farn->setPyrScale(pyrScale);
    farn->setPolyN(polyN);
    farn->setPolySigma(polySigma);
    farn->setFlags(flags);

    cv::cuda::GpuMat d_flow;
    farn->calc(loadMat(frame0), loadMat(frame1), d_flow);

    cv::Mat flow;
    if (useInitFlow)
    {
        d_flow.download(flow);

        farn->setFlags(farn->getFlags() | cv::OPTFLOW_USE_INITIAL_FLOW);
        farn->calc(loadMat(frame0), loadMat(frame1), d_flow);
    }

    cv::calcOpticalFlowFarneback(
        frame0, frame1, flow, farn->getPyrScale(), farn->getNumLevels(), farn->getWinSize(),
        farn->getNumIters(), farn->getPolyN(), farn->getPolySigma(), farn->getFlags());

    // Relax test limit when the flag is set
    if (farn->getFlags() & cv::OPTFLOW_FARNEBACK_GAUSSIAN)
    {
        EXPECT_MAT_SIMILAR(flow, d_flow, 2e-2);
    }
    else
    {
        EXPECT_MAT_SIMILAR(flow, d_flow, 1e-4);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_OptFlow, FarnebackOpticalFlow, testing::Combine(
    ALL_DEVICES,
    testing::Values(PyrScale(0.3), PyrScale(0.5), PyrScale(0.8)),
    testing::Values(PolyN(5), PolyN(7)),
    testing::Values(FarnebackOptFlowFlags(0), FarnebackOptFlowFlags(cv::OPTFLOW_FARNEBACK_GAUSSIAN)),
    testing::Values(UseInitFlow(false), UseInitFlow(true))));


PARAM_TEST_CASE(FarnebackOpticalFlowAsync, cv::cuda::DeviceInfo, PyrScale, PolyN, FarnebackOptFlowFlags)
{
    cv::cuda::DeviceInfo devInfo;
    double pyrScale;
    int polyN;
    int flags;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        pyrScale = GET_PARAM(1);
        polyN = GET_PARAM(2);
        flags = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(FarnebackOpticalFlowAsync, Accuracy)
{
    cv::Mat frame0Mat = readImage("opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0Mat.empty());

    cv::Mat frame1Mat = readImage("opticalflow/rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1Mat.empty());

    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();
    farn->setPyrScale(pyrScale);
    farn->setPolyN(polyN);
    farn->setPolySigma(1.1);
    farn->setFlags(flags);

    Stream sourceStream;
    HostMem dummyHost(4000, 4000, CV_8UC3), frame0(frame0Mat), frame1(frame1Mat);
    GpuMat d_flow, dummyDevice(dummyHost.size(), dummyHost.type()), frame0Device(frame0.size(), frame0.type()), frame1Device(frame1.size(), frame1.type());

    // initialize and warm up CUDA kernels to ensure this doesn't occur during the test
    farn->calc(loadMat(frame0Mat), loadMat(frame1Mat), d_flow);
    d_flow.setTo(0);

    frame0Device.upload(frame0, sourceStream);
    // place extra work in sourceStream to test internal stream synchronization by delaying the upload of frame1 that stream, see https://github.com/opencv/opencv/issues/24540
    dummyDevice.upload(dummyHost, sourceStream);
    frame1Device.upload(frame1, sourceStream);
    farn->calc(frame0Device, frame1Device, d_flow, sourceStream);

    Mat flow;
    cv::calcOpticalFlowFarneback(
        frame0, frame1, flow, farn->getPyrScale(), farn->getNumLevels(), farn->getWinSize(),
        farn->getNumIters(), farn->getPolyN(), farn->getPolySigma(), farn->getFlags());
        EXPECT_MAT_SIMILAR(flow, d_flow, 1e-4);
}

INSTANTIATE_TEST_CASE_P(CUDA_OptFlow, FarnebackOpticalFlowAsync, testing::Combine(
    ALL_DEVICES,
    testing::Values(PyrScale(0.3)),
    testing::Values(PolyN(5)),
    testing::Values(FarnebackOptFlowFlags(0))));

//////////////////////////////////////////////////////
// OpticalFlowDual_TVL1

namespace
{
    IMPLEMENT_PARAM_CLASS(Gamma, double)
}

PARAM_TEST_CASE(OpticalFlowDual_TVL1, cv::cuda::DeviceInfo, Gamma)
{
    cv::cuda::DeviceInfo devInfo;
    double gamma;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        gamma = GET_PARAM(1);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(OpticalFlowDual_TVL1, Accuracy)
{
    cv::Mat frame0 = readImage("opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("opticalflow/rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> d_alg =
            cv::cuda::OpticalFlowDual_TVL1::create();
    d_alg->setNumIterations(10);
    d_alg->setGamma(gamma);

    cv::cuda::GpuMat d_flow;
    d_alg->calc(loadMat(frame0), loadMat(frame1), d_flow);

    cv::Ptr<cv::optflow::DualTVL1OpticalFlow> alg = cv::optflow::createOptFlow_DualTVL1();
    alg->setMedianFiltering(1);
    alg->setInnerIterations(1);
    alg->setOuterIterations(d_alg->getNumIterations());
    alg->setGamma(gamma);

    cv::Mat flow;
    alg->calc(frame0, frame1, flow);

    EXPECT_MAT_SIMILAR(flow, d_flow, 4e-3);
}

class TVL1AsyncParallelLoopBody : public cv::ParallelLoopBody
{
public:
    TVL1AsyncParallelLoopBody(const cv::cuda::GpuMat& d_img1_, const cv::cuda::GpuMat& d_img2_, cv::cuda::GpuMat* d_flow_, int iterations_, double gamma_)
        : d_img1(d_img1_), d_img2(d_img2_), d_flow(d_flow_), iterations(iterations_), gamma(gamma_) {}
    ~TVL1AsyncParallelLoopBody() {}
    void operator()(const cv::Range& r) const
    {
        for (int i = r.start; i < r.end; i++) {
            cv::cuda::Stream stream;
            cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> d_alg = cv::cuda::OpticalFlowDual_TVL1::create();
            d_alg->setNumIterations(iterations);
            d_alg->setGamma(gamma);
            d_alg->calc(d_img1, d_img2, d_flow[i], stream);
            stream.waitForCompletion();
        }
    }
protected:
    const cv::cuda::GpuMat& d_img1;
    const cv::cuda::GpuMat& d_img2;
    cv::cuda::GpuMat* d_flow;
    int iterations;
    double gamma;
};

#define NUM_STREAMS 16

CUDA_TEST_P(OpticalFlowDual_TVL1, Async)
{
    if (!supportFeature(devInfo, cv::cuda::FEATURE_SET_COMPUTE_30))
    {
        throw SkipTestException("CUDA device doesn't support texture objects");
    }
    else
    {
        cv::Mat frame0 = readImage("opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame0.empty());

        cv::Mat frame1 = readImage("opticalflow/rubberwhale2.png", cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(frame1.empty());

        const int iterations = 10;

        // Synchronous call
        cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> d_alg =
                cv::cuda::OpticalFlowDual_TVL1::create();
        d_alg->setNumIterations(iterations);
        d_alg->setGamma(gamma);

        cv::cuda::GpuMat d_flow_gold;
        d_alg->calc(loadMat(frame0), loadMat(frame1), d_flow_gold);

        // Asynchronous call
        cv::cuda::GpuMat d_flow[NUM_STREAMS];
        cv::parallel_for_(cv::Range(0, NUM_STREAMS), TVL1AsyncParallelLoopBody(loadMat(frame0), loadMat(frame1), d_flow, iterations, gamma));

        // Compare the results of synchronous call and asynchronous call
        for (int i = 0; i < NUM_STREAMS; i++)
            EXPECT_MAT_NEAR(d_flow_gold, d_flow[i], 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_OptFlow, OpticalFlowDual_TVL1, testing::Combine(
    ALL_DEVICES,
    testing::Values(Gamma(0.0), Gamma(1.0))));


//////////////////////////////////////////////////////
// NvidiaOpticalFlow_1_0

struct NvidiaOpticalFlow_1_0 : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(NvidiaOpticalFlow_1_0, Regression)
{
    cv::Mat frame0 = readImage("opticalflow/frame0.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("opticalflow/frame1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    cv::Ptr<cv::cuda::NvidiaOpticalFlow_1_0> d_nvof;
    try
    {
        d_nvof = cv::cuda::NvidiaOpticalFlow_1_0::create(frame0.size(),
            cv::cuda::NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_SLOW);
    }
    catch (const cv::Exception& e)
    {
        if (e.code == Error::StsBadFunc || e.code == Error::StsBadArg || e.code == Error::StsNullPtr)
            throw SkipTestException("Current configuration is not supported");
        throw;
    }
    const int gridSize = d_nvof->getGridSize();

    Mat flow, upsampledFlow;
    d_nvof->calc(loadMat(frame0), loadMat(frame1), flow);
    d_nvof->upSampler(flow, frame0.size(), gridSize, upsampledFlow);

    std::string fname(cvtest::TS::ptr()->get_data_path());
    fname += "opticalflow/nvofGolden.flo";
    cv::Mat golden = cv::readOpticalFlow(fname.c_str());
    ASSERT_FALSE(golden.empty());

    EXPECT_MAT_SIMILAR(golden, upsampledFlow, 1e-10);
    d_nvof->collectGarbage();
}

CUDA_TEST_P(NvidiaOpticalFlow_1_0, OpticalFlowNan)
{
    cv::Mat frame0 = readImage("opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("opticalflow/rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    cv::Mat r_frame0, r_frame1;

    cv::Ptr<cv::cuda::NvidiaOpticalFlow_1_0> d_nvof;
    try
    {
        d_nvof = cv::cuda::NvidiaOpticalFlow_1_0::create(frame0.size(),
            cv::cuda::NvidiaOpticalFlow_1_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_SLOW);
    }
    catch (const cv::Exception& e)
    {
        if (e.code == Error::StsBadFunc || e.code == Error::StsBadArg || e.code == Error::StsNullPtr)
            throw SkipTestException("Current configuration is not supported");
        throw;
    }

    Mat flow, flowx, flowy;
    d_nvof->calc(loadMat(frame0), loadMat(frame1), flow);

    Mat planes[] = { flowx, flowy };
    split(flow, planes);
    flowx = planes[0]; flowy = planes[1];

    EXPECT_TRUE(cv::checkRange(flowx));
    EXPECT_TRUE(cv::checkRange(flowy));
    d_nvof->collectGarbage();
};

INSTANTIATE_TEST_CASE_P(CUDA_OptFlow, NvidiaOpticalFlow_1_0, ALL_DEVICES);

//////////////////////////////////////////////////////
// NvidiaOpticalFlow_2_0

struct NvidiaOpticalFlow_2_0 : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;

    virtual void SetUp()
    {
        devInfo = GetParam();

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(NvidiaOpticalFlow_2_0, Regression)
{
    cv::Mat frame0 = readImage("opticalflow/frame0.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("opticalflow/frame1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> d_nvof;
    try
    {
        d_nvof = cv::cuda::NvidiaOpticalFlow_2_0::create(frame0.size(),
            cv::cuda::NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_SLOW);
    }
    catch (const cv::Exception& e)
    {
        if (e.code == Error::StsBadFunc || e.code == Error::StsBadArg || e.code == Error::StsNullPtr)
            throw SkipTestException("Current configuration is not supported");
        throw;
    }

    Mat flow, upsampledFlow;
    d_nvof->calc(loadMat(frame0), loadMat(frame1), flow);
    d_nvof->convertToFloat(flow, upsampledFlow);

    std::string fname(cvtest::TS::ptr()->get_data_path());
    fname += "opticalflow/nvofGolden_2.flo";
    cv::Mat golden = cv::readOpticalFlow(fname.c_str());
    ASSERT_FALSE(golden.empty());

    EXPECT_MAT_SIMILAR(golden, upsampledFlow, 1e-10);
}

CUDA_TEST_P(NvidiaOpticalFlow_2_0, OpticalFlowNan)
{
    cv::Mat frame0 = readImage("opticalflow/rubberwhale1.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame0.empty());

    cv::Mat frame1 = readImage("opticalflow/rubberwhale2.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(frame1.empty());

    cv::Mat r_frame0, r_frame1;

    cv::Ptr<cv::cuda::NvidiaOpticalFlow_2_0> d_nvof;
    try
    {
        d_nvof = cv::cuda::NvidiaOpticalFlow_2_0::create(frame0.size(),
            cv::cuda::NvidiaOpticalFlow_2_0::NVIDIA_OF_PERF_LEVEL::NV_OF_PERF_LEVEL_SLOW);
    }
    catch (const cv::Exception& e)
    {
        if (e.code == Error::StsBadFunc || e.code == Error::StsBadArg || e.code == Error::StsNullPtr)
            throw SkipTestException("Current configuration is not supported");
        throw;
    }

    Mat flow, flowx, flowy;
    d_nvof->calc(loadMat(frame0), loadMat(frame1), flow);

    Mat planes[] = { flowx, flowy };
    split(flow, planes);
    flowx = planes[0]; flowy = planes[1];

    EXPECT_TRUE(cv::checkRange(flowx));
    EXPECT_TRUE(cv::checkRange(flowy));
};

INSTANTIATE_TEST_CASE_P(CUDA_OptFlow, NvidiaOpticalFlow_2_0, ALL_DEVICES);

}} // namespace
#endif // HAVE_CUDA
