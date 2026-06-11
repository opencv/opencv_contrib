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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HistEven

typedef tuple<Size, int> hist_size_to_roi_offset_params_t;
const hist_size_to_roi_offset_params_t hist_size_to_roi_offset_params[] =
{
    // uchar reads only
    hist_size_to_roi_offset_params_t(Size(1,32), 0),
    hist_size_to_roi_offset_params_t(Size(2,32), 0),
    hist_size_to_roi_offset_params_t(Size(2,32), 1),
    hist_size_to_roi_offset_params_t(Size(3,32), 0),
    hist_size_to_roi_offset_params_t(Size(3,32), 1),
    hist_size_to_roi_offset_params_t(Size(3,32), 2),
    hist_size_to_roi_offset_params_t(Size(4,32), 0),
    hist_size_to_roi_offset_params_t(Size(4,32), 1),
    hist_size_to_roi_offset_params_t(Size(4,32), 2),
    hist_size_to_roi_offset_params_t(Size(4,32), 3),
    // uchar and int reads
    hist_size_to_roi_offset_params_t(Size(129,32), 0),
    hist_size_to_roi_offset_params_t(Size(129,32), 1),
    hist_size_to_roi_offset_params_t(Size(129,32), 2),
    hist_size_to_roi_offset_params_t(Size(129,32), 3),
    // int reads only
    hist_size_to_roi_offset_params_t(Size(128,32), 0)
};

PARAM_TEST_CASE(HistEven, cv::cuda::DeviceInfo, hist_size_to_roi_offset_params_t)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int roiOffsetX;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = get<0>(GET_PARAM(1));
        roiOffsetX = get<1>(GET_PARAM(1));

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(HistEven, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);
    const Rect roi = Rect(roiOffsetX, 0, src.cols - roiOffsetX, src.rows);
    int hbins = 30;
    float hranges[] = {50.0f, 200.0f};

    cv::cuda::GpuMat hist;
    cv::cuda::GpuMat srcDevice = loadMat(src);
    cv::cuda::histEven(srcDevice(roi), hist, hbins, (int)hranges[0], (int)hranges[1]);

    cv::Mat hist_gold;

    int histSize[] = {hbins};
    const float* ranges[] = {hranges};
    int channels[] = {0};
    Mat srcRoi = src(roi);
    cv::calcHist(&srcRoi, 1, channels, cv::Mat(), hist_gold, 1, histSize, ranges);

    hist_gold = hist_gold.t();
    hist_gold.convertTo(hist_gold, CV_32S);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, HistEven, testing::Combine(
    ALL_DEVICES, testing::ValuesIn(hist_size_to_roi_offset_params)));

namespace
{
    // Reference range histogram over half-open bins [levels[i], levels[i+1]),
    // the last boundary exclusive; matches NPP and the HIP range kernel.
    template <typename T, typename L>
    void rangeHistGold(const cv::Mat& src, int channel, const std::vector<L>& levels, cv::Mat& out)
    {
        const int bins = (int)levels.size() - 1;
        out = cv::Mat::zeros(1, bins, CV_32S);
        int* h = out.ptr<int>();
        const int cn = src.channels();
        for (int y = 0; y < src.rows; ++y)
        {
            const T* row = src.ptr<T>(y);
            for (int x = 0; x < src.cols; ++x)
            {
                const L v = (L)row[x * cn + channel];
                if (v < levels[0] || v >= levels[bins])
                    continue;
                int lo = 0, hi = bins;
                while (hi - lo > 1)
                {
                    const int mid = (lo + hi) >> 1;
                    if (v < levels[mid]) hi = mid; else lo = mid;
                }
                h[lo]++;
            }
        }
    }

    std::vector<int> evenLevels32s(int nLevels, int lower, int upper)
    {
        std::vector<int> lv(nLevels);
        const double range = (double)(upper - lower);
        for (int i = 0; i < nLevels; ++i)
            lv[i] = lower + cvRound(range * i / (nLevels - 1));
        return lv;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HistEven16 (16U / 16S single channel)

PARAM_TEST_CASE(HistEven16, cv::cuda::DeviceInfo, MatDepth)
{
    cv::cuda::DeviceInfo devInfo;
    int depth;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        depth = GET_PARAM(1);
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(HistEven16, Accuracy)
{
    const cv::Size size(257, 130);
    cv::Mat src = randomMat(size, CV_MAKETYPE(depth, 1), 0, 4096);

    const int hbins = 50;
    const int lower = 100;
    const int upper = 3500;

    cv::cuda::GpuMat hist;
    cv::cuda::histEven(loadMat(src), hist, hbins, lower, upper);

    // NPP's histEven snaps bin boundaries to the integer evenLevels layout, so
    // the reference is a range histogram over those boundaries, not the float
    // bin edges cv::calcHist would use.
    std::vector<int> levels = evenLevels32s(hbins + 1, lower, upper);
    cv::Mat gold;
    if (depth == CV_16U) rangeHistGold<ushort, int>(src, 0, levels, gold);
    else                 rangeHistGold<short,  int>(src, 0, levels, gold);

    EXPECT_MAT_NEAR(gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, HistEven16, testing::Combine(
    ALL_DEVICES, testing::Values(MatDepth(CV_16U), MatDepth(CV_16S))));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HistEven4 (4-channel)

PARAM_TEST_CASE(HistEven4, cv::cuda::DeviceInfo, MatDepth)
{
    cv::cuda::DeviceInfo devInfo;
    int depth;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        depth = GET_PARAM(1);
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(HistEven4, Accuracy)
{
    const cv::Size size(200, 150);
    cv::Mat src = randomMat(size, CV_MAKETYPE(depth, 4), 0, depth == CV_8U ? 256 : 4096);

    int histSize[4] = {20, 30, 40, 10};
    int lower[4] = {0, 50, 100, 10};
    int upper[4] = {depth == CV_8U ? 256 : 4000, 3000, 3500, 2000};

    cv::cuda::GpuMat hist[4];
    cv::cuda::histEven(loadMat(src), hist, histSize, lower, upper);

    for (int c = 0; c < 4; ++c)
    {
        std::vector<int> levels = evenLevels32s(histSize[c] + 1, lower[c], upper[c]);
        cv::Mat gold;
        if (depth == CV_8U)       rangeHistGold<uchar,  int>(src, c, levels, gold);
        else if (depth == CV_16U) rangeHistGold<ushort, int>(src, c, levels, gold);
        else                      rangeHistGold<short,  int>(src, c, levels, gold);
        EXPECT_MAT_NEAR(gold, hist[c], 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, HistEven4, testing::Combine(
    ALL_DEVICES, testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S))));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HistRange (single channel, all depths)

PARAM_TEST_CASE(HistRange, cv::cuda::DeviceInfo, MatDepth)
{
    cv::cuda::DeviceInfo devInfo;
    int depth;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        depth = GET_PARAM(1);
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(HistRange, Accuracy)
{
    const cv::Size size(220, 160);
    const bool isF = depth == CV_32F;
    cv::Mat src = randomMat(size, CV_MAKETYPE(depth, 1), 0, isF ? 1.0 : (depth == CV_8U ? 256 : 4000));

    cv::cuda::GpuMat hist;
    cv::Mat gold;
    if (isF)
    {
        std::vector<float> lv = {0.0f, 0.1f, 0.25f, 0.4f, 0.55f, 0.7f, 0.85f, 1.0f};
        cv::Mat levels(1, (int)lv.size(), CV_32FC1, lv.data());
        cv::cuda::histRange(loadMat(src), hist, loadMat(levels));
        rangeHistGold<float, float>(src, 0, lv, gold);
    }
    else
    {
        std::vector<int> lv = {0, 30, 70, 130, 200, 280, 400, 600, 900, 1500, 4000};
        cv::Mat levels(1, (int)lv.size(), CV_32SC1, lv.data());
        cv::cuda::histRange(loadMat(src), hist, loadMat(levels));
        if (depth == CV_8U)
        {
            std::vector<int> lv8 = {0, 30, 70, 130, 200, 255};
            cv::Mat levels8(1, (int)lv8.size(), CV_32SC1, lv8.data());
            cv::cuda::histRange(loadMat(src), hist, loadMat(levels8));
            rangeHistGold<uchar, int>(src, 0, lv8, gold);
        }
        else if (depth == CV_16U)
            rangeHistGold<ushort, int>(src, 0, lv, gold);
        else
            rangeHistGold<short, int>(src, 0, lv, gold);
    }

    EXPECT_MAT_NEAR(gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, HistRange, testing::Combine(
    ALL_DEVICES, testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F))));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HistRange4 (4-channel)

PARAM_TEST_CASE(HistRange4, cv::cuda::DeviceInfo, MatDepth)
{
    cv::cuda::DeviceInfo devInfo;
    int depth;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        depth = GET_PARAM(1);
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(HistRange4, Accuracy)
{
    const cv::Size size(180, 140);
    const bool isF = depth == CV_32F;
    cv::Mat src = randomMat(size, CV_MAKETYPE(depth, 4), 0, isF ? 1.0 : (depth == CV_8U ? 256 : 4000));

    cv::cuda::GpuMat hist[4];
    cv::Mat levels[4];
    cv::cuda::GpuMat dLevels[4];

    std::vector<float> lvF = {0.0f, 0.2f, 0.45f, 0.7f, 1.0f};
    std::vector<int> lvI = {0, 40, 100, 220, 400, 700, 1200, 4000};
    std::vector<int> lvI8 = {0, 40, 100, 180, 255};

    for (int c = 0; c < 4; ++c)
    {
        if (isF)
            levels[c] = cv::Mat(1, (int)lvF.size(), CV_32FC1, lvF.data()).clone();
        else if (depth == CV_8U)
            levels[c] = cv::Mat(1, (int)lvI8.size(), CV_32SC1, lvI8.data()).clone();
        else
            levels[c] = cv::Mat(1, (int)lvI.size(), CV_32SC1, lvI.data()).clone();
        dLevels[c] = loadMat(levels[c]);
    }

    cv::cuda::histRange(loadMat(src), hist, dLevels);

    for (int c = 0; c < 4; ++c)
    {
        cv::Mat gold;
        if (isF)                  rangeHistGold<float,  float>(src, c, lvF, gold);
        else if (depth == CV_8U)  rangeHistGold<uchar,  int>(src, c, lvI8, gold);
        else if (depth == CV_16U) rangeHistGold<ushort, int>(src, c, lvI, gold);
        else                      rangeHistGold<short,  int>(src, c, lvI, gold);
        EXPECT_MAT_NEAR(gold, hist[c], 0.0);
    }
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, HistRange4, testing::Combine(
    ALL_DEVICES, testing::Values(MatDepth(CV_8U), MatDepth(CV_16U), MatDepth(CV_16S), MatDepth(CV_32F))));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CalcHist

PARAM_TEST_CASE(CalcHist, cv::cuda::DeviceInfo, hist_size_to_roi_offset_params_t)
{
    cv::cuda::DeviceInfo devInfo;

    cv::Size size;

    int roiOffsetX;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = get<0>(GET_PARAM(1));
        roiOffsetX = get<1>(GET_PARAM(1));

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CalcHist, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);
    const Rect roi = Rect(roiOffsetX, 0, src.cols - roiOffsetX, src.rows);
    cv::cuda::GpuMat hist;
    GpuMat srcDevice = loadMat(src);
    cv::cuda::calcHist(srcDevice(roi), hist);

    cv::Mat hist_gold;

    const int hbins = 256;
    const float hranges[] = {0.0f, 256.0f};
    const int histSize[] = {hbins};
    const float* ranges[] = {hranges};
    const int channels[] = {0};

    const Mat srcRoi = src(roi);
    cv::calcHist(&srcRoi, 1, channels, cv::Mat(), hist_gold, 1, histSize, ranges);
    hist_gold = hist_gold.reshape(1, 1);
    hist_gold.convertTo(hist_gold, CV_32S);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, CalcHist, testing::Combine(
    ALL_DEVICES, testing::ValuesIn(hist_size_to_roi_offset_params)));

PARAM_TEST_CASE(CalcHistWithMask, cv::cuda::DeviceInfo, hist_size_to_roi_offset_params_t)
{
    cv::cuda::DeviceInfo devInfo;

    cv::Size size;

    int roiOffsetX;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = get<0>(GET_PARAM(1));
        roiOffsetX = get<1>(GET_PARAM(1));

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CalcHistWithMask, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);
    const Rect roi = Rect(roiOffsetX, 0, src.cols - roiOffsetX, src.rows);
    cv::Mat mask = randomMat(size, CV_8UC1);
    cv::Mat(mask, cv::Rect(0, 0, size.width / 2, size.height / 2)).setTo(0);

    cv::cuda::GpuMat hist;
    GpuMat srcDevice = loadMat(src);
    GpuMat maskDevice = loadMat(mask);
    cv::cuda::calcHist(srcDevice(roi), maskDevice(roi), hist);

    cv::Mat hist_gold;

    const int hbins = 256;
    const float hranges[] = {0.0f, 256.0f};
    const int histSize[] = {hbins};
    const float* ranges[] = {hranges};
    const int channels[] = {0};

    const Mat srcRoi = src(roi);
    cv::calcHist(&srcRoi, 1, channels, mask(roi), hist_gold, 1, histSize, ranges);
    hist_gold = hist_gold.reshape(1, 1);
    hist_gold.convertTo(hist_gold, CV_32S);

    EXPECT_MAT_NEAR(hist_gold, hist, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, CalcHistWithMask, testing::Combine(
    ALL_DEVICES, testing::ValuesIn(hist_size_to_roi_offset_params)));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// EqualizeHist

PARAM_TEST_CASE(EqualizeHist, cv::cuda::DeviceInfo, cv::Size)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(EqualizeHist, Async)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::cuda::Stream stream;

    cv::cuda::GpuMat dst;
    cv::cuda::equalizeHist(loadMat(src), dst, stream);

    stream.waitForCompletion();

    cv::Mat dst_gold;
    cv::equalizeHist(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

CUDA_TEST_P(EqualizeHist, Accuracy)
{
    cv::Mat src = randomMat(size, CV_8UC1);

    cv::cuda::GpuMat dst;
    cv::cuda::equalizeHist(loadMat(src), dst);

    cv::Mat dst_gold;
    cv::equalizeHist(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, EqualizeHist, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES));

TEST(EqualizeHistIssue, Issue18035)
{
    std::vector<std::string> imgPaths;
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/3MP.png");
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/5MP.png");
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/airplane.png");
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/baboon.png");
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/box.png");
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/box_in_scene.png");
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/fruits.png");
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/fruits_ecc.png");
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/graffiti.png");
    imgPaths.push_back(std::string(cvtest::TS::ptr()->get_data_path()) + "../cv/shared/lena.png");

    for (size_t i = 0; i < imgPaths.size(); ++i)
    {
        std::string imgPath = imgPaths[i];
        cv::Mat src = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
        src = src / 30;

        cv::cuda::GpuMat d_src, dst;
        d_src.upload(src);
        cv::cuda::equalizeHist(d_src, dst);

        cv::Mat dst_gold;
        cv::equalizeHist(src, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
}

PARAM_TEST_CASE(EqualizeHistExtreme, cv::cuda::DeviceInfo, cv::Size, int)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    int val;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        val = GET_PARAM(2);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(EqualizeHistExtreme, Case1)
{
    cv::Mat src(size, CV_8UC1, val);

    cv::cuda::GpuMat dst;
    cv::cuda::equalizeHist(loadMat(src), dst);

    cv::Mat dst_gold;
    cv::equalizeHist(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

CUDA_TEST_P(EqualizeHistExtreme, Case2)
{
    cv::Mat src = randomMat(size, CV_8UC1, val);

    cv::cuda::GpuMat dst;
    cv::cuda::equalizeHist(loadMat(src), dst);

    cv::Mat dst_gold;
    cv::equalizeHist(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, EqualizeHistExtreme, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Range(0, 256)));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// CLAHE

namespace
{
    IMPLEMENT_PARAM_CLASS(ClipLimit, double)
}

PARAM_TEST_CASE(CLAHE, cv::cuda::DeviceInfo, cv::Size, ClipLimit, MatType)
{
    cv::cuda::DeviceInfo devInfo;
    cv::Size size;
    double clipLimit;
    int type;

    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        size = GET_PARAM(1);
        clipLimit = GET_PARAM(2);
        type = GET_PARAM(3);

        cv::cuda::setDevice(devInfo.deviceID());
    }
};

CUDA_TEST_P(CLAHE, Accuracy)
{
    cv::Mat src;
    if (type == CV_8UC1)
        src = randomMat(size, type);
    else if (type == CV_16UC1)
        src = randomMat(size, type, 0, 65535);

    cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE(clipLimit);
    cv::cuda::GpuMat dst;
    clahe->apply(loadMat(src), dst);

    cv::Ptr<cv::CLAHE> clahe_gold = cv::createCLAHE(clipLimit);
    cv::Mat dst_gold;
    clahe_gold->apply(src, dst_gold);

    ASSERT_MAT_NEAR(dst_gold, dst, 1.0);
}

INSTANTIATE_TEST_CASE_P(CUDA_ImgProc, CLAHE, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    testing::Values(0.0, 5.0, 10.0, 20.0, 40.0),
    testing::Values(MatType(CV_8UC1), MatType(CV_16UC1))));


}} // namespace
#endif // HAVE_CUDA
