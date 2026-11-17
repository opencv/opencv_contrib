// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include "resize_reference.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
namespace opencv_test
{
namespace
{

namespace resize_ref = cv_triton_resize;

// The evaluator shares runtime planning; the gold checks below independently
// verify small cases using direct formulas, as in imgproc/test/test_imgwarp_strict.cpp.
static Mat evaluateReference(const Mat &src, Size size, double fy, double fx, int interpolation)
{
    Mat dst(size, src.type());
    resize_ref::ImageRef input{src.data,       src.rows,    src.cols,
                               src.channels(), src.step[0], src.depth() == CV_32F};
    resize_ref::ImageRef output{dst.data,       dst.rows,    dst.cols,
                                dst.channels(), dst.step[0], dst.depth() == CV_32F};
    resize_ref::resize(input, output, fy, fx, interpolation);
    return dst;
}

TEST(Triton_ResizeReference, NearestCoordinatesAndScaleFactors)
{
    struct Case
    {
        Size src, dst;
        double fx, fy;
    };
    const Case cases[] = {{Size(19, 13), Size(7, 5), 0, 0},     {Size(7, 5), Size(19, 13), 0, 0},
                          {Size(19, 13), Size(), 0.37, 0.61},   {Size(7, 5), Size(), 1.61, 2.37},
                          {Size(1, 17), Size(7, 9), 0, 0},      {Size(17, 1), Size(9, 7), 0, 0},
                          {Size(19, 13), Size(11, 9), 0.2, 3.0}};
    for (int depth : {CV_8U, CV_32F})
        for (int cn : {1, 3, 4})
            for (const auto &c : cases)
            {
                SCOPED_TRACE(cv::format("depth=%d cn=%d src=%dx%d dst=%dx%d fx=%g fy=%g", depth, cn,
                                        c.src.width, c.src.height, c.dst.width, c.dst.height, c.fx,
                                        c.fy));
                Mat parent(c.src + Size(5, 3), CV_MAKETYPE(depth, cn));
                Mat src = parent(Rect(Point(1, 1), c.src));
                for (int y = 0; y < src.rows; ++y)
                    for (int e = 0; e < src.cols * cn; ++e)
                        if (depth == CV_8U)
                            src.ptr<uchar>(y)[e] = static_cast<uchar>((y * 59 + e * 17) & 255);
                        else
                            src.ptr<float>(y)[e] = static_cast<float>(y * 59 + e * 17) / 8.f;
                const bool explicitSize = c.dst.width > 0 && c.dst.height > 0;
                const Size expectedSize =
                    explicitSize ? c.dst
                                 : Size(cvRound(c.src.width * c.fx), cvRound(c.src.height * c.fy));
                const double fx = explicitSize ? double(expectedSize.width) / c.src.width : c.fx;
                const double fy = explicitSize ? double(expectedSize.height) / c.src.height : c.fy;
                const resize_ref::Size2 plan = resize_ref::resolve_size(
                    src.rows, src.cols, c.dst.width, c.dst.height, c.fx, c.fy);
                ASSERT_EQ(expectedSize.width, plan.dst_w);
                ASSERT_EQ(expectedSize.height, plan.dst_h);
                ASSERT_EQ(fx, plan.inv_x);
                ASSERT_EQ(fy, plan.inv_y);
                const Mat actual =
                    evaluateReference(src, expectedSize, plan.inv_y, plan.inv_x, INTER_NEAREST);
                Mat expected(expectedSize, src.type());
                for (int y = 0; y < expected.rows; ++y)
                {
                    const int sy =
                        std::min(src.rows - 1, static_cast<int>(std::floor(y * (1.0 / fy))));
                    for (int x = 0; x < expected.cols; ++x)
                    {
                        const int sx =
                            std::min(src.cols - 1, static_cast<int>(std::floor(x * (1.0 / fx))));
                        std::memcpy(expected.ptr(y) + x * expected.elemSize(),
                                    src.ptr(sy) + sx * src.elemSize(), src.elemSize());
                    }
                }
                EXPECT_EQ(0, cv::norm(actual, expected, NORM_INF));
            }
}

TEST(Triton_ResizeReference, Area2x2Rounding)
{
    for (int depth : {CV_8U, CV_32F})
        for (int cn : {1, 3, 4})
            for (int width : {1, 3, 5, 7, 9})
                for (bool useScales : {false, true})
                {
                    SCOPED_TRACE(cv::format("depth=%d cn=%d width=%d scales=%d", depth, cn, width,
                                            useScales));
                    const Size outputSize(width, 5), inputSize(width * 2, 10);
                    Mat parent(inputSize + Size(5, 3), CV_MAKETYPE(depth, cn));
                    Mat src = parent(Rect(Point(1, 1), inputSize));
                    for (int y = 0; y < src.rows; ++y)
                        for (int e = 0; e < src.cols * cn; ++e)
                            if (depth == CV_8U)
                                src.ptr<uchar>(y)[e] = static_cast<uchar>(
                                    (y * 37 + e * e * 11 + e * 3 + (e + y) % 5) & 255);
                            else
                                // Small binary fractions make the sum exact regardless of
                                // the vector-prefix or scalar-tail addition order.
                                src.ptr<float>(y)[e] =
                                    static_cast<float>(((y * 37 + e * 13) % 1024) - 512) / 8.f;
                    const Size requested = useScales ? Size() : outputSize;
                    const resize_ref::Size2 plan = resize_ref::resolve_size(
                        src.rows, src.cols, requested.width, requested.height, useScales ? 0.5 : 0,
                        useScales ? 0.5 : 0);
                    ASSERT_EQ(outputSize.width, plan.dst_w);
                    ASSERT_EQ(outputSize.height, plan.dst_h);
                    ASSERT_EQ(0.5, plan.inv_x);
                    ASSERT_EQ(0.5, plan.inv_y);
                    const Mat actual =
                        evaluateReference(src, outputSize, plan.inv_y, plan.inv_x, INTER_AREA);
                    Mat expected(outputSize, src.type());
                    for (int y = 0; y < expected.rows; ++y)
                        for (int x = 0; x < expected.cols; ++x)
                            for (int c = 0; c < cn; ++c)
                            {
                                const int e = 2 * x * cn + c;
                                if (depth == CV_8U)
                                {
                                    const uchar *a = src.ptr<uchar>(2 * y);
                                    const uchar *b = src.ptr<uchar>(2 * y + 1);
                                    const int sum = a[e] + a[e + cn] + b[e] + b[e + cn];
                                    expected.ptr<uchar>(y)[x * cn + c] =
                                        static_cast<uchar>((sum + 2) / 4);
                                }
                                else
                                {
                                    const float *a = src.ptr<float>(2 * y);
                                    const float *b = src.ptr<float>(2 * y + 1);
                                    expected.ptr<float>(y)[x * cn + c] =
                                        (a[e] + a[e + cn] + b[e] + b[e + cn]) * 0.25f;
                                }
                            }
                    EXPECT_EQ(0, cv::norm(actual, expected, NORM_INF));
                }
}

// Independent 3x3 sums, with source and destination having unrelated parent pitches.
TEST(Triton_ResizeReference, Area3x3IndependentPitchesAndGuards)
{
    for (int depth : {CV_8U, CV_32F})
        for (int channels : {1, 3, 4})
        {
            const int type = CV_MAKETYPE(depth, channels);
            Mat sourceParent(11, 17, type, Scalar::all(9));
            Mat source = sourceParent(Rect(1, 1, 12, 9));
            for (int y = 0; y < source.rows; ++y)
                for (int x = 0; x < source.cols * channels; ++x)
                    if (depth == CV_8U)
                        source.ptr<uchar>(y)[x] = static_cast<uchar>(9 * ((y * 13 + x * 7) % 16));
                    else
                        source.ptr<float>(y)[x] =
                            static_cast<float>(9 * ((y * 13 + x * 7) % 16)) / 8.f;
            Mat destinationParent(7, 11, type, Scalar::all(77));
            const Mat before = destinationParent.clone();
            Mat destination = destinationParent(Rect(2, 1, 4, 3));
            resize_ref::ImageRef input{source.data, source.rows, source.cols,
                                       channels,    source.step, depth == CV_32F};
            resize_ref::ImageRef output{destination.data, destination.rows, destination.cols,
                                        channels,         destination.step, depth == CV_32F};
            resize_ref::resize(input, output, 1.0 / 3, 1.0 / 3, INTER_AREA);
            for (int y = 0; y < 3; ++y)
                for (int x = 0; x < 4; ++x)
                    for (int c = 0; c < channels; ++c)
                    {
                        double sum = 0;
                        for (int yy = 0; yy < 3; ++yy)
                            for (int xx = 0; xx < 3; ++xx)
                                sum +=
                                    depth == CV_8U
                                        ? source.ptr<uchar>(3 * y + yy)[(3 * x + xx) * channels + c]
                                        : source.ptr<float>(3 * y +
                                                            yy)[(3 * x + xx) * channels + c];
                        if (depth == CV_8U)
                            EXPECT_EQ(saturate_cast<uchar>(sum / 9),
                                      destination.ptr<uchar>(y)[x * channels + c]);
                        else
                            EXPECT_NEAR(sum / 9, destination.ptr<float>(y)[x * channels + c],
                                        8.419e-7);
                    }
            for (int y = 0; y < destinationParent.rows; ++y)
                for (int x = 0; x < destinationParent.cols; ++x)
                    if (!Rect(2, 1, 4, 3).contains(Point(x, y)))
                        EXPECT_EQ(0, std::memcmp(before.ptr(y) + x * before.elemSize(),
                                                 destinationParent.ptr(y) +
                                                     x * destinationParent.elemSize(),
                                                 before.elemSize()));
        }
}

// Direct convolution and pixel-coverage gold values are independent of the
// production route and coefficient tables. The test-only evaluator above still
// shares those tables with the runtime; these small cases check that boundary.
static Mat makeGoldInput(Size size, int depth, int channels, int quantum)
{
    Mat parent(size + Size(5, 3), CV_MAKETYPE(depth, channels));
    Mat src = parent(Rect(Point(1, 1), size));
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
            for (int c = 0; c < channels; ++c)
            {
                const int value = quantum * ((x * x + 3 * y + x * y + 2 * c) % (255 / quantum + 1));
                if (depth == CV_8U)
                    src.ptr<uchar>(y)[x * channels + c] = static_cast<uchar>(value);
                else
                    src.ptr<float>(y)[x * channels + c] = value / 128.f - 1.f;
            }
    return src;
}

static double goldSample(const Mat &src, int y, int x, int channel)
{
    y = std::max(0, std::min(src.rows - 1, y));
    x = std::max(0, std::min(src.cols - 1, x));
    const int element = x * src.channels() + channel;
    return src.depth() == CV_8U ? src.ptr<uchar>(y)[element] : src.ptr<float>(y)[element];
}

static double cubicWeight(double distance)
{
    const double x = std::abs(distance);
    // Keys' cubic convolution kernel, a = -3/4. This expression does not
    // construct or quantize the runtime's four-coefficient interpolation table.
    if (x <= 1.0)
        return 1.25 * x * x * x - 2.25 * x * x + 1.0;
    if (x < 2.0)
        return -0.75 * x * x * x + 3.75 * x * x - 6.0 * x + 3.0;
    return 0.0;
}

static Mat convolutionGold(const Mat &src, Size size, int interpolation)
{
    Mat expected(size, src.type());
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x)
        {
            // An exact 2x enlargement gives quarter-pixel phases, so the
            // coefficients and products in these cases are binary fractions.
            const double sy = (y + 0.5) * 0.5 - 0.5;
            const double sx = (x + 0.5) * 0.5 - 0.5;
            const int iy = static_cast<int>(std::floor(sy));
            const int ix = static_cast<int>(std::floor(sx));
            const int first = interpolation == INTER_CUBIC ? -1 : 0;
            const int last = interpolation == INTER_CUBIC ? 2 : 1;
            for (int c = 0; c < src.channels(); ++c)
            {
                double value = 0.0;
                for (int dy = first; dy <= last; ++dy)
                    for (int dx = first; dx <= last; ++dx)
                    {
                        const double wy = interpolation == INTER_CUBIC
                                              ? cubicWeight(sy - iy - dy)
                                              : std::max(0.0, 1.0 - std::abs(sy - iy - dy));
                        const double wx = interpolation == INTER_CUBIC
                                              ? cubicWeight(sx - ix - dx)
                                              : std::max(0.0, 1.0 - std::abs(sx - ix - dx));
                        value += goldSample(src, iy + dy, ix + dx, c) * wx * wy;
                    }
                const int element = x * src.channels() + c;
                if (src.depth() == CV_8U)
                {
                    // Generic uint8 cubic has a round-to-even vector prefix
                    // and a half-up scalar tail, including a partial final row.
                    const bool cubicTail = interpolation == INTER_CUBIC &&
                                           element >= (size.width * src.channels() / 8) * 8;
                    const double rounded =
                        cubicTail ? std::floor(value + 0.5) : std::nearbyint(value);
                    expected.ptr<uchar>(y)[element] =
                        static_cast<uchar>(std::max(0.0, std::min(255.0, rounded)));
                }
                else
                    expected.ptr<float>(y)[element] = static_cast<float>(value);
            }
        }
    return expected;
}

TEST(Triton_ResizeReference, LinearPixelCentersAndReplicatedEdges)
{
    for (int depth : {CV_8U, CV_32F})
        for (int channels : {1, 3, 4})
            for (Size inputSize : {Size(5, 3), Size(1, 3), Size(5, 1)})
            {
                SCOPED_TRACE(cv::format("depth=%d cn=%d size=%dx%d", depth, channels,
                                        inputSize.width, inputSize.height));
                const Mat src = makeGoldInput(inputSize, depth, channels, 16);
                const Size outputSize(inputSize.width * 2, inputSize.height * 2);
                const Mat expected = convolutionGold(src, outputSize, INTER_LINEAR);
                const Mat actual = evaluateReference(src, outputSize, 2.0, 2.0, INTER_LINEAR);
                EXPECT_LE(cv::norm(actual, expected, NORM_INF), depth == CV_8U ? 0.0 : 4.948e-7);
            }
}

TEST(Triton_ResizeReference, CubicConvolutionAndReplicatedEdges)
{
    for (int depth : {CV_8U, CV_32F})
        for (int channels : {1, 3, 4})
            for (Size inputSize : {Size(5, 3), Size(1, 3), Size(5, 1)})
            {
                SCOPED_TRACE(cv::format("depth=%d cn=%d size=%dx%d", depth, channels,
                                        inputSize.width, inputSize.height));
                const Mat src = makeGoldInput(inputSize, depth, channels, 16);
                const Size outputSize(inputSize.width * 2, inputSize.height * 2);
                const Mat expected = convolutionGold(src, outputSize, INTER_CUBIC);
                const Mat actual = evaluateReference(src, outputSize, 2.0, 2.0, INTER_CUBIC);
                EXPECT_LE(cv::norm(actual, expected, NORM_INF), depth == CV_8U ? 0.0 : 1.401e-6);
            }
}

TEST(Triton_ResizeReference, FractionalAreaPixelCoverage)
{
    const Size inputSizes[] = {Size(5, 3), Size(7, 5)};
    const Size outputSizes[] = {Size(3, 2), Size(3, 3)};
    for (int depth : {CV_8U, CV_32F})
        for (int channels : {1, 3, 4})
            for (int geometry = 0; geometry < 2; ++geometry)
            {
                const Size inputSize = inputSizes[geometry], outputSize = outputSizes[geometry];
                SCOPED_TRACE(cv::format("depth=%d cn=%d src=%dx%d dst=%dx%d", depth, channels,
                                        inputSize.width, inputSize.height, outputSize.width,
                                        outputSize.height));
                const Mat src = makeGoldInput(inputSize, depth, channels, inputSize.area());
                const double scaleX = double(src.cols) / outputSize.width;
                const double scaleY = double(src.rows) / outputSize.height;
                Mat expected(outputSize, src.type());
                for (int y = 0; y < outputSize.height; ++y)
                    for (int x = 0; x < outputSize.width; ++x)
                        for (int c = 0; c < channels; ++c)
                        {
                            double value = 0.0;
                            for (int sy = 0; sy < src.rows; ++sy)
                                for (int sx = 0; sx < src.cols; ++sx)
                                {
                                    const double overlapX =
                                        std::max(0.0, std::min(double(sx + 1), (x + 1) * scaleX) -
                                                          std::max(double(sx), x * scaleX));
                                    const double overlapY =
                                        std::max(0.0, std::min(double(sy + 1), (y + 1) * scaleY) -
                                                          std::max(double(sy), y * scaleY));
                                    value += goldSample(src, sy, sx, c) * overlapX * overlapY /
                                             (scaleX * scaleY);
                                }
                            if (depth == CV_8U)
                                expected.ptr<uchar>(y)[x * channels + c] =
                                    saturate_cast<uchar>(value);
                            else
                                expected.ptr<float>(y)[x * channels + c] =
                                    static_cast<float>(value);
                        }
                const Mat actual =
                    evaluateReference(src, outputSize, 1.0 / scaleY, 1.0 / scaleX, INTER_AREA);
                EXPECT_LE(cv::norm(actual, expected, NORM_INF), depth == CV_8U ? 0.0 : 8.419e-7);
            }
}

}
} // namespace opencv_test
