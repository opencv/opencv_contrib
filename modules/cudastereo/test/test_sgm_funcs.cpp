// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: The "adaskit Team" at Fixstars Corporation

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

#ifdef _WIN32
#define popcnt64 __popcnt64
#else
#define popcnt64 __builtin_popcountll
#endif

#include "opencv2/core/cuda.hpp"

namespace cv { namespace cuda { namespace device {
namespace stereosgm
{

namespace census_transform
{
void censusTransform(const GpuMat& src, GpuMat& dest, cv::cuda::Stream& stream);
}

namespace path_aggregation
{
namespace horizontal
{
template <unsigned int MAX_DISPARITY>
void aggregateLeft2RightPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template <unsigned int MAX_DISPARITY>
void aggregateRight2LeftPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);
}

namespace vertical
{
template <unsigned int MAX_DISPARITY>
void aggregateUp2DownPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template <unsigned int MAX_DISPARITY>
void aggregateDown2UpPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);
}

namespace oblique
{
template <unsigned int MAX_DISPARITY>
void aggregateUpleft2DownrightPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template <unsigned int MAX_DISPARITY>
void aggregateUpright2DownleftPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template <unsigned int MAX_DISPARITY>
void aggregateDownright2UpleftPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template <unsigned int MAX_DISPARITY>
void aggregateDownleft2UprightPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);
}
} // namespace path_aggregation

namespace winner_takes_all
{
template <size_t MAX_DISPARITY>
void winnerTakesAll(const GpuMat& src, GpuMat& left, GpuMat& right, float uniqueness, bool subpixel, int mode, cv::cuda::Stream& stream);
}

} // namespace stereosgm
}}} // namespace cv { namespace cuda { namespace device {

namespace opencv_test { namespace {

    void census_transform(const cv::Mat& src, cv::Mat& dst)
    {
        const int hor = 9 / 2, ver = 7 / 2;
        dst.create(src.size(), CV_32SC1);
        dst = 0;
        for (int y = ver; y < static_cast<int>(src.rows) - ver; ++y) {
            for (int x = hor; x < static_cast<int>(src.cols) - hor; ++x) {
                int32_t value = 0;
                for (int dy = -ver; dy <= 0; ++dy) {
                    for (int dx = -hor; dx <= (dy == 0 ? -1 : hor); ++dx) {
                        const auto a = src.at<uint8_t>(y + dy, x + dx);
                        const auto b = src.at<uint8_t>(y - dy, x - dx);
                        value <<= 1;
                        if (a > b) { value |= 1; }
                    }
                }
                dst.at<int32_t>(y, x) = value;
            }
        }
    }

    PARAM_TEST_CASE(StereoSGM_CensusTransformImage, cv::cuda::DeviceInfo, std::string, UseRoi)
    {
        cv::cuda::DeviceInfo devInfo;
        std::string path;
        bool useRoi;

        virtual void SetUp()
        {
            devInfo = GET_PARAM(0);
            path = GET_PARAM(1);
            useRoi = GET_PARAM(2);

            cv::cuda::setDevice(devInfo.deviceID());
        }
    };

    CUDA_TEST_P(StereoSGM_CensusTransformImage, Image)
    {
        cv::Mat image = readImage(path, cv::IMREAD_GRAYSCALE);
        cv::Mat dst_gold;
        census_transform(image, dst_gold);

        cv::cuda::GpuMat g_dst;
        g_dst.create(image.size(), CV_32SC1);
        cv::cuda::device::stereosgm::census_transform::censusTransform(loadMat(image, useRoi), g_dst, cv::cuda::Stream::Null());

        cv::Mat dst;
        g_dst.download(dst);

        EXPECT_MAT_NEAR(dst_gold, dst, 0);
    }

    INSTANTIATE_TEST_CASE_P(CUDA_StereoSGM_funcs, StereoSGM_CensusTransformImage, testing::Combine(
        ALL_DEVICES,
        testing::Values("stereobm/aloe-L.png", "stereobm/aloe-R.png"),
        WHOLE_SUBMAT));

    PARAM_TEST_CASE(StereoSGM_CensusTransformRandom, cv::cuda::DeviceInfo, cv::Size, UseRoi)
    {
        cv::cuda::DeviceInfo devInfo;
        cv::Size size;
        bool useRoi;

        virtual void SetUp()
        {
            devInfo = GET_PARAM(0);
            size = GET_PARAM(1);
            useRoi = GET_PARAM(2);

            cv::cuda::setDevice(devInfo.deviceID());
        }
    };

    CUDA_TEST_P(StereoSGM_CensusTransformRandom, Random)
    {
        cv::Mat image = randomMat(size, CV_8UC1);
        cv::Mat dst_gold;
        census_transform(image, dst_gold);

        cv::cuda::GpuMat g_dst;
        g_dst.create(image.size(), CV_32SC1);
        cv::cuda::device::stereosgm::census_transform::censusTransform(loadMat(image, useRoi), g_dst, cv::cuda::Stream::Null());

        cv::Mat dst;
        g_dst.download(dst);

        EXPECT_MAT_NEAR(dst_gold, dst, 0);
    }

    INSTANTIATE_TEST_CASE_P(CUDA_StereoSGM_funcs, StereoSGM_CensusTransformRandom, testing::Combine(
        ALL_DEVICES,
        DIFFERENT_SIZES,
        WHOLE_SUBMAT));

    static void path_aggregation(
        const cv::Mat& left,
        const cv::Mat& right,
        cv::Mat& dst,
        int max_disparity, int min_disparity, int p1, int p2,
        int dx, int dy)
    {
        const int width = left.cols;
        const int height = left.rows;
        dst.create(cv::Size(width * height * max_disparity, 1), CV_8UC1);
        std::vector<int> before(max_disparity);
        for (int i = (dy < 0 ? height - 1 : 0); 0 <= i && i < height; i += (dy < 0 ? -1 : 1)) {
            for (int j = (dx < 0 ? width - 1 : 0); 0 <= j && j < width; j += (dx < 0 ? -1 : 1)) {
                const int i2 = i - dy, j2 = j - dx;
                const bool inside = (0 <= i2 && i2 < height && 0 <= j2 && j2 < width);
                for (int k = 0; k < max_disparity; ++k) {
                    before[k] = inside ? dst.at<uint8_t>(0, k + (j2 + i2 * width) * max_disparity) : 0;
                }
                const int min_cost = *min_element(before.begin(), before.end());
                for (int k = 0; k < max_disparity; ++k) {
                    const auto l = left.at<int32_t>(i, j);
                    const auto r = (k + min_disparity > j ? 0 : right.at<int32_t>(i, j - k - min_disparity));
                    int cost = std::min(before[k] - min_cost, p2);
                    if (k > 0) {
                        cost = std::min(cost, before[k - 1] - min_cost + p1);
                    }
                    if (k + 1 < max_disparity) {
                        cost = std::min(cost, before[k + 1] - min_cost + p1);
                    }
                    cost += static_cast<int>(popcnt64(l ^ r));
                    dst.at<uint8_t>(0, k + (j + i * width) * max_disparity) = static_cast<uint8_t>(cost);
                }
            }
        }
    }

    static constexpr size_t DISPARITY = 128;
    static constexpr int P1 = 10;
    static constexpr int P2 = 120;

    PARAM_TEST_CASE(StereoSGM_PathAggregation, cv::cuda::DeviceInfo, cv::Size, UseRoi, int)
    {
        cv::cuda::DeviceInfo devInfo;
        cv::Size size;
        bool useRoi;
        int minDisp;

        virtual void SetUp()
        {
            devInfo = GET_PARAM(0);
            size = GET_PARAM(1);
            useRoi = GET_PARAM(2);
            minDisp = GET_PARAM(3);

            cv::cuda::setDevice(devInfo.deviceID());
        }

        template<typename T>
        void test_path_aggregation(T func, int dx, int dy)
        {
            cv::Mat left_image = randomMat(size, CV_32SC1, 0.0, static_cast<double>(std::numeric_limits<int32_t>::max()));
            cv::Mat right_image = randomMat(size, CV_32SC1, 0.0, static_cast<double>(std::numeric_limits<int32_t>::max()));
            cv::Mat dst_gold;
            path_aggregation(left_image, right_image, dst_gold, DISPARITY, minDisp, P1, P2, dx, dy);

            cv::cuda::GpuMat g_dst;
            g_dst.create(cv::Size(left_image.cols * left_image.rows * DISPARITY, 1), CV_8UC1);
            func(loadMat(left_image, useRoi), loadMat(right_image, useRoi), g_dst, P1, P2, minDisp, cv::cuda::Stream::Null());

            cv::Mat dst;
            g_dst.download(dst);

            EXPECT_MAT_NEAR(dst_gold, dst, 0);
        }
    };

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomLeft2Right)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::path_aggregation::horizontal::aggregateLeft2RightPath<DISPARITY>, 1, 0);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomRight2Left)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::path_aggregation::horizontal::aggregateRight2LeftPath<DISPARITY>, -1, 0);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomUp2Down)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::path_aggregation::vertical::aggregateUp2DownPath<DISPARITY>, 0, 1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomDown2Up)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::path_aggregation::vertical::aggregateDown2UpPath<DISPARITY>, 0, -1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomUpLeft2DownRight)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::path_aggregation::oblique::aggregateUpleft2DownrightPath<DISPARITY>, 1, 1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomUpRight2DownLeft)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::path_aggregation::oblique::aggregateUpright2DownleftPath<DISPARITY>, -1, 1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomDownRight2UpLeft)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::path_aggregation::oblique::aggregateDownright2UpleftPath<DISPARITY>, -1, -1);
    }

    CUDA_TEST_P(StereoSGM_PathAggregation, RandomDownLeft2UpRight)
    {
        test_path_aggregation(cv::cuda::device::stereosgm::path_aggregation::oblique::aggregateDownleft2UprightPath<DISPARITY>, 1, -1);
    }

    INSTANTIATE_TEST_CASE_P(CUDA_StereoSGM_funcs, StereoSGM_PathAggregation, testing::Combine(
        ALL_DEVICES,
        DIFFERENT_SIZES,
        WHOLE_SUBMAT,
        testing::Values(0, 1, 10)));


    void winner_takes_all_left(
        const cv::Mat& src,
        cv::Mat& dst,
        int width, int height, int disparity, int num_paths,
        float uniqueness, bool subpixel)
    {
        dst.create(cv::Size(width, height), CV_16UC1);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                std::vector<std::pair<int, int>> v;
                for (int k = 0; k < disparity; ++k) {
                    int cost_sum = 0;
                    for (int p = 0; p < num_paths; ++p) {
                        cost_sum += static_cast<int>(src.at<uint8_t>(0,
                            p * disparity * width * height +
                                i * disparity * width +
                                j * disparity +
                                k));
                    }
                    v.emplace_back(cost_sum, static_cast<int>(k));
                }
                const auto ite = std::min_element(v.begin(), v.end());
                assert(ite != v.end());
                const auto best = *ite;
                const int best_cost = best.first;
                int best_disp = best.second;
                int ans = best_disp;
                if (subpixel) {
                    ans <<= StereoMatcher::DISP_SHIFT;
                    if (0 < best_disp && best_disp < static_cast<int>(disparity) - 1) {
                        const int left = v[best_disp - 1].first;
                        const int right = v[best_disp + 1].first;
                        const int numer = left - right;
                        const int denom = left - 2 * best_cost + right;
                        ans += ((numer << StereoMatcher::DISP_SHIFT) + denom) / (2 * denom);
                    }
                }
                for (const auto& p : v) {
                    const int cost = p.first;
                    const int disp = p.second;
                    if (cost * uniqueness < best_cost && abs(disp - best_disp) > 1) {
                        ans = -1;
                        break;
                    }
                }

                dst.at<uint16_t>(i, j) = static_cast<uint16_t>(ans);
            }
        }
    }

    PARAM_TEST_CASE(StereoSGM_WinnerTakesAll, cv::cuda::DeviceInfo, cv::Size, bool, int)
    {
        cv::cuda::DeviceInfo devInfo;
        cv::Size size;
        bool subpixel;
        int mode;

        virtual void SetUp()
        {
            devInfo = GET_PARAM(0);
            size = GET_PARAM(1);
            subpixel = GET_PARAM(2);
            mode = GET_PARAM(3);

            cv::cuda::setDevice(devInfo.deviceID());
        }
    };

    CUDA_TEST_P(StereoSGM_WinnerTakesAll, RandomLeft)
    {
        int num_paths = mode == cv::cuda::StereoSGM::MODE_HH4 ? 4 : 8;
        cv::Mat aggregated = randomMat(cv::Size(size.width * size.height * DISPARITY * num_paths, 1), CV_8UC1, 0.0, 32.0);
        cv::Mat dst_gold;
        winner_takes_all_left(aggregated, dst_gold, size.width, size.height, DISPARITY, num_paths, 0.95f, subpixel);

        cv::cuda::GpuMat g_src, g_dst, g_dst_right;
        g_src.upload(aggregated);
        g_dst.create(size, CV_16UC1);
        g_dst_right.create(size, CV_16UC1);
        cv::cuda::device::stereosgm::winner_takes_all::winnerTakesAll<DISPARITY>(g_src, g_dst, g_dst_right, 0.95f, subpixel, mode, cv::cuda::Stream::Null());

        cv::Mat dst;
        g_dst.download(dst);

        EXPECT_MAT_NEAR(dst_gold, dst, 0);
    }

    INSTANTIATE_TEST_CASE_P(CUDA_StereoSGM_funcs, StereoSGM_WinnerTakesAll, testing::Combine(
        ALL_DEVICES,
        DIFFERENT_SIZES,
        testing::Values(false, true),
        testing::Values(cv::cuda::StereoSGM::MODE_HH4, cv::cuda::StereoSGM::MODE_HH)));

}} // namespace
#endif // HAVE_CUDA
