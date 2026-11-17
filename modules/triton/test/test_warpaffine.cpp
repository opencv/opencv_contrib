// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Public GpuMat accuracy tests against the linked CPU implementation.
// NEAREST/LINEAR/AREA use its padded vector-path reference; CUBIC uses a natural-
// width call. Fused versus separate arithmetic is identified from CPU outputs
// per depth, channel count and cv::useOptimized() state. Hardware FMA support
// alone does not identify the compiled path. These tests do not promise matching
// output from arbitrary CPU builds or natural destination widths.
// CV_8U is exact, CV_16U permits one level, and CV_32F uses a per-element bound.
// Coverage includes interpolation/borders, affine geometry, degenerate axes,
// pitched ROIs, output reuse, guard bands and unsupported requests.
#include "test_helpers.hpp"
#include <cuda.h>
#include <cmath>
namespace opencv_test
{
namespace
{

using namespace triton_test;

namespace wa = cv_triton_warp;

// ---------------------------------------------------------------- helpers

static Mat toMat6(const double m[6])
{
    Mat M(2, 3, CV_64F);
    for (int i = 0; i < 6; ++i)
        M.at<double>(i / 3, i % 3) = m[i];
    return M;
}

/** Six doubles -> a 2x3 CV_64F matrix. A plain function so no temporary array is addressed. */
static Mat M6(double a, double b, double c, double d, double e, double f)
{
    const double m[6] = {a, b, c, d, e, f};
    return toMat6(m);
}

/** warpAffine through the module, then download. */
static Mat gpuWarp(const Mat &src, const Mat &M, Size dsize, int flags, int border,
                   const Scalar &bv = Scalar(), cuda::Stream *stream = nullptr)
{
    cuda::GpuMat gsrc(src), gdst;
    if (stream)
        cv::triton::warpAffine(gsrc, gdst, M, dsize, flags, border, bv, *stream);
    else
        cv::triton::warpAffine(gsrc, gdst, M, dsize, flags, border, bv);
    if (stream)
        stream->waitForCompletion();
    else
        cuda::Stream::Null().waitForCompletion();
    Mat out;
    gdst.download(out);
    return out;
}

/** Padded vector-path reference for NEAREST/LINEAR/AREA; natural width for CUBIC.
 *
 * The CPU SIMD body uses v_fma(M0, x, float(y*M1 + M2)); the scalar tail may
 * use separate operations. A rounding difference at a half-pixel tie can
 * select an adjacent pixel for NEAREST. Padding the reference width to a
 * multiple of 128 and cropping keeps compared columns in the vector body.
 * The width-sensitivity and padding-stability tests below check this assumption
 * against the linked implementation. CUBIC instead computes coordinates from
 * absolute x in genericWarp/bicubicVec without this body/tail distinction.
 */
static Mat cpuWarp(const Mat &src, const Mat &M, Size dsize, int flags, int border,
                   const Scalar &bv = Scalar())
{
    return warpReference(src, M, dsize, flags, border, bv);
}

/** The user-visible result at exactly `dsize`, for the one test that documents the divergence. */
static Mat cpuWarpNatural(const Mat &src, const Mat &M, Size dsize, int flags, int border,
                          const Scalar &bv = Scalar())
{
    Mat out;
    cv::warpAffine(src, out, M, dsize, flags, border, bv);
    return out;
}

/** Per-element float bound with a hard absolute cap of 1e-4.
 * RecordsAccuracy also reports absolute error, error/bound and reference ULP
 * spacing; a tolerance pass alone does not imply identical outputs.
 */
static double f32Tol(double ref_value)
{
    return warpF32Tolerance(ref_value);
}

/** Comparison statistics, including exact differences within the allowed bound. */
struct WarpStats
{
    long long elements = 0;
    long long outside = 0;       // exceeds the accuracy bound
    long long not_identical = 0; // bitwise different, inside or outside
    double max_abs = 0;
    double max_norm = 0; // |diff| / bound, per element
    double max_ulp = 0;  // |diff| / spacing(ref), per element
    void merge(const WarpStats &o)
    {
        elements += o.elements;
        outside += o.outside;
        not_identical += o.not_identical;
        max_abs = std::max(max_abs, o.max_abs);
        max_norm = std::max(max_norm, o.max_norm);
        max_ulp = std::max(max_ulp, o.max_ulp);
    }
};

/** Reference float spacing at `v`, used to express error in ULPs. */
static double spacing_at(float v)
{
    const float a = std::fabs(v);
    const float next = std::nextafterf(a, std::numeric_limits<float>::infinity());
    const double sp = double(next) - double(a);
    return sp > 0 ? sp : double(std::numeric_limits<float>::denorm_min());
}

/** Compare CV_8U exactly, CV_16U within 1 LSB, and CV_32F within
 *  min(5e-5*max(|ref|,1), 1e-4). */
static ::testing::AssertionResult meetsWarpContract(const Mat &got, const Mat &ref_,
                                                    WarpStats *stats = nullptr)
{
    if (got.size() != ref_.size() || got.type() != ref_.type())
        return ::testing::AssertionFailure() << "size/type " << got.size() << "/" << got.type()
                                             << " vs " << ref_.size() << "/" << ref_.type();
    const int W = got.cols * got.channels();
    WarpStats st;
    for (int y = 0; y < got.rows; ++y)
    {
        if (got.depth() == CV_8U)
        {
            const uchar *a = got.ptr<uchar>(y);
            const uchar *b = ref_.ptr<uchar>(y);
            for (int x = 0; x < W; ++x)
            {
                ++st.elements;
                const double d = std::abs(double(a[x]) - b[x]);
                if (d != 0)
                    ++st.not_identical;
                st.max_abs = std::max(st.max_abs, d);
                st.max_norm =
                    std::max(st.max_norm, (d == 0 ? 0.0 : std::numeric_limits<double>::infinity()));
                if (d != 0.0)
                    ++st.outside;
            }
        }
        else if (got.depth() == CV_16U)
        {
            const ushort *a = got.ptr<ushort>(y);
            const ushort *b = ref_.ptr<ushort>(y);
            for (int x = 0; x < W; ++x)
            {
                ++st.elements;
                const double d = std::abs(double(a[x]) - double(b[x]));
                if (d != 0)
                    ++st.not_identical;
                st.max_abs = std::max(st.max_abs, d);
                st.max_norm = std::max(st.max_norm, d / 1.0);
                if (d > 1.0)
                    ++st.outside;
            }
        }
        else
        {
            const float *a = got.ptr<float>(y);
            const float *b = ref_.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                ++st.elements;
                const double d = std::abs(double(a[x]) - b[x]);
                const double bound = f32Tol(b[x]);
                if (d != 0)
                    ++st.not_identical;
                st.max_abs = std::max(st.max_abs, d);
                st.max_norm = std::max(st.max_norm, d / bound);
                st.max_ulp = std::max(st.max_ulp, d / spacing_at(b[x]));
                if (!(d <= bound))
                    ++st.outside;
            }
        }
    }
    if (stats)
        stats->merge(st);
    if (st.outside)
        return ::testing::AssertionFailure()
               << st.outside << " of " << st.elements
               << " elements outside the accuracy bound; max |diff| " << st.max_abs
               << ", max normalised " << st.max_norm << ", max " << st.max_ulp << " ULP";
    return ::testing::AssertionSuccess();
}

TEST(Triton_WarpAffine, AccuracyPolicyLimits)
{
    Mat ref8(1, 1, CV_8U, Scalar(20)), got8(1, 1, CV_8U, Scalar(21));
    EXPECT_FALSE(meetsWarpContract(got8, ref8));
    EXPECT_TRUE(meetsWarpContract(ref8, ref8));
    Mat ref16(1, 1, CV_16U, Scalar(20)), got16(1, 1, CV_16U, Scalar(21));
    EXPECT_TRUE(meetsWarpContract(got16, ref16));
    Mat ref32(1, 1, CV_32F, Scalar(100)), got32(1, 1, CV_32F, Scalar(100.0002));
    EXPECT_FALSE(meetsWarpContract(got32, ref32));
    EXPECT_LE(f32Tol(1e6), 1e-4);
}

struct WaCase
{
    const char *name;
    double M[6];
    int interp;
    int border;
    bool inverse;
    bool dsize_from_src;
    Size dsize;
};

static const WaCase kGeoms[] = {
    {"identity", {1, 0, 0, 0, 1, 0}, INTER_LINEAR, BORDER_CONSTANT, false, true, Size()},
    {"translate_int", {1, 0, 3, 0, 1, -2}, INTER_LINEAR, BORDER_REPLICATE, false, true, Size()},
    {"translate_sub",
     {1, 0, 0.37, 0, 1, -0.62},
     INTER_LINEAR,
     BORDER_REPLICATE,
     false,
     true,
     Size()},
    {"translate_half",
     {1, 0, 0.5, 0, 1, 0.5},
     INTER_NEAREST,
     BORDER_REPLICATE,
     false,
     true,
     Size()},
    {"rotate30",
     {0.8660254, -0.5, 0, 0.5, 0.8660254, 0},
     INTER_LINEAR,
     BORDER_CONSTANT,
     false,
     true,
     Size()},
    {"full_affine",
     {0.8, -0.3, 12.5, 0.3, 0.8, -7.25},
     INTER_LINEAR,
     BORDER_CONSTANT,
     false,
     true,
     Size()},
    {"scale_aniso", {0.5, 0, 0, 0, 0.75, 0}, INTER_LINEAR, BORDER_REFLECT_101, false, true, Size()},
    {"shear", {1, 0.35, 0, 0, 1, 0}, INTER_NEAREST, BORDER_WRAP, false, true, Size()},
    {"negative_scale", {-1, 0, 40, 0, -1, 40}, INTER_LINEAR, BORDER_CONSTANT, false, true, Size()},
    {"strong_shear", {1, 1.9, -20, 0.4, 1, -10}, INTER_LINEAR, BORDER_REFLECT, false, true, Size()},
    {"inverse_map",
     {0.9, 0.43589, -14, -0.43589, 0.9, 10},
     INTER_LINEAR,
     BORDER_CONSTANT,
     true,
     true,
     Size()},
    {"bigger_dst",
     {1, 0, -20, 0, 1, -15},
     INTER_LINEAR,
     BORDER_CONSTANT,
     false,
     false,
     Size(120, 90)},
    {"far_outside", {1, 0, -600, 0, 1, -400}, INTER_NEAREST, BORDER_CONSTANT, false, true, Size()},
};

// Three geometries for the CUBIC tests: identity (weights collapse to a delta), a rotation
// with every tap fractional, and a strong shear that reaches outside the source.
static const double kCubicGeoms[3][6] = {
    {1, 0, 0, 0, 1, 0},
    {0.9659258, -0.2588190, 6.4, 0.2588190, 0.9659258, -3.1},
    {1, 0.35, -0.75, 0, 1, 2.25},
};

// The linked CPU can produce different samples in the shared region of two
// destination widths because its SIMD body and scalar tail round differently.
// This fixture checks an adjacent-pixel change for NEAREST and a one-level
// LINEAR difference. Changes to CPU dispatch require revisiting the reference.
TEST(Triton_WarpAffine, CpuRegionIsNotWidthInvariant)
{
    if (!ready())
        throw SkipTestException("no Triton backend");

    const Mat M = M6(1, 1.9, -20, 0.4, 1, -10);
    const Size sz(53, 37);
    const int widths[] = {40, 41, 48, 52};

    double worst_nearest = 0, worst_linear = 0;
    for (uint64 seed = 0; seed < 24; ++seed)
    {
        const Mat host = randomImage(sz, CV_8UC1, 400 + seed);
        for (int w : widths)
        {
            const Mat n64 = cpuWarpNatural(host, M, Size(128, 37), INTER_NEAREST, BORDER_CONSTANT);
            const Mat l64 = cpuWarpNatural(host, M, Size(128, 37), INTER_LINEAR, BORDER_CONSTANT);
            const Mat nw = cpuWarpNatural(host, M, Size(w, 37), INTER_NEAREST, BORDER_CONSTANT);
            const Mat lw = cpuWarpNatural(host, M, Size(w, 37), INTER_LINEAR, BORDER_CONSTANT);
            worst_nearest =
                std::max(worst_nearest, cvtest::norm(nw, n64(Rect(0, 0, w, 37)), NORM_INF));
            worst_linear =
                std::max(worst_linear, cvtest::norm(lw, l64(Rect(0, 0, w, 37)), NORM_INF));
        }
    }
    SCOPED_TRACE(cv::format("worst over 24 seeds x 4 widths: NEAREST %g, LINEAR %g", worst_nearest,
                            worst_linear));
    // NEAREST: an adjacent-pixel selection, so no small-error bound is available.
    EXPECT_GT(worst_nearest, 1.0)
        << "NEAREST agreed across destination widths; the reference assumes an "
           "ulp can select an adjacent source pixel, and the reference choice should be revisited";
    // One-level LINEAR difference for this fixture, rather than a general bound.
    EXPECT_LE(worst_linear, 1.0)
        << "LINEAR differed across widths by more than the one-level fixture bound";

    // The reference is stable: widening the destination does not change the crop.
    const Mat host = randomImage(sz, CV_8UC1, 101);
    const Mat v128 = cpuWarp(host, M, Size(40, 37), INTER_LINEAR, BORDER_CONSTANT);
    const Mat v256 =
        cpuWarpNatural(host, M, Size(256, 37), INTER_LINEAR, BORDER_CONSTANT)(Rect(0, 0, 40, 37));
    EXPECT_EQ(0.0, cvtest::norm(v128, v256, NORM_INF))
        << "the vector-path reference depends on how far the destination is padded";

    // The module follows the selected vector-path reference.
    EXPECT_TRUE(
        meetsWarpContract(gpuWarp(host, M, Size(40, 37), INTER_LINEAR, BORDER_CONSTANT), v128));
    EXPECT_TRUE(meetsWarpContract(gpuWarp(host, M, Size(40, 37), INTER_NEAREST, BORDER_CONSTANT),
                                  cpuWarp(host, M, Size(40, 37), INTER_NEAREST, BORDER_CONSTANT)));
}

// CUBIC uses the natural-width CPU reference. Its absolute-x coordinates have
// no SIMD-body/scalar-tail split. Detect width dependence if that path changes.
TEST(Triton_WarpAffine, CpuCubicIsWidthInvariant)
{
    const Mat M = M6(1, 1.9, -20, 0.4, 1, -10);
    const Size sz(53, 37);
    const int widths[] = {40, 41, 48, 64, 100};
    int differing = 0;
    double worst = 0;
    for (int depth : {CV_8U, CV_16U})
        for (int interp : {INTER_CUBIC, INTER_NEAREST})
        {
            for (uint64 seed = 0; seed < 24; ++seed)
            {
                const Mat host = randomImage(sz, CV_MAKETYPE(depth, 1), 300 + seed);
                const Mat wide = cpuWarpNatural(host, M, Size(256, 37), interp, BORDER_CONSTANT);
                for (int w : widths)
                {
                    const Mat narrow =
                        cpuWarpNatural(host, M, Size(w, 37), interp, BORDER_CONSTANT);
                    const double d = cvtest::norm(narrow, wide(Rect(0, 0, w, 37)), NORM_INF);
                    if (interp == INTER_CUBIC)
                    {
                        SCOPED_TRACE(cv::format("CUBIC depth=%d seed=%llu width=%d diff=%g", depth,
                                                (unsigned long long)seed, w, d));
                        if (d != 0)
                            ++differing;
                        worst = std::max(worst, d);
                    }
                }
            }
        }
    EXPECT_EQ(0, differing) << "INTER_CUBIC differed across destination widths (worst " << worst
                            << "); the natural-width call is used as the CUBIC "
                               "reference, so this needs the reference rule revisited";
}

// CUBIC through the module against the natural-width CPU call, at widths that are NOT
// multiples of 128: a padding rule leaking into CUBIC would show up here as a mismatch.
TEST(Triton_WarpAffine, CubicMatchesNaturalWidth)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Size sz(59, 45);
    uint64 seed = 401;
    int checked = 0;
    for (int depth : {CV_8U, CV_16U, CV_32F})
        for (int cn : {1, 3, 4})
        {
            const int type = CV_MAKETYPE(depth, cn);
            const Mat host = randomImage(sz, type, seed++);
            for (const Size ds : {Size(97, 61), Size(64, 48), Size(33, 27), Size(150, 44)})
                for (const double *g : {kCubicGeoms[0], kCubicGeoms[1], kCubicGeoms[2]})
                {
                    SCOPED_TRACE(
                        cv::format("depth=%d cn=%d %dx%d", depth, cn, ds.width, ds.height));
                    const Mat M = toMat6(g);
                    const Mat got = gpuWarp(host, M, ds, INTER_CUBIC, BORDER_CONSTANT);
                    const Mat ref_ = cpuWarpNatural(host, M, ds, INTER_CUBIC, BORDER_CONSTANT);
                    EXPECT_TRUE(meetsWarpContract(got, ref_));
                    ++checked;
                }
        }
    EXPECT_EQ(3 * 3 * 4 * 3, checked);
}

TEST(Triton_WarpAffine, MatchesCpu)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Size sz(53, 37); // odd, not a multiple of the 128-wide tile
    uint64 seed = 11;
    for (int depth : {CV_8U, CV_16U, CV_32F})
        for (int cn : {1, 3, 4})
            for (int interp : {INTER_NEAREST, INTER_LINEAR, INTER_CUBIC})
                for (const WaCase &c : kGeoms)
                {
                    SCOPED_TRACE(
                        cv::format("%s depth=%d cn=%d interp=%d", c.name, depth, cn, interp));
                    const int type = CV_MAKETYPE(depth, cn);
                    const Mat host = randomImage(sz, type, seed++);
                    Mat M = toMat6(c.M);
                    int flags = interp | (c.inverse ? WARP_INVERSE_MAP : 0);
                    const Size ds = c.dsize_from_src ? sz : c.dsize;
                    const Mat got = gpuWarp(host, M, ds, flags, c.border);
                    const Mat ref_ = cpuWarp(host, M, ds, flags, c.border);
                    EXPECT_TRUE(meetsWarpContract(got, ref_));
                }
}

// Record identical-element counts separately from tolerance checks, alongside
// absolute error, normalized error and reference ULP distance for each depth.
TEST(Triton_WarpAffine, RecordsAccuracy)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Size sz(53, 37);
    WarpStats cu, cu16, cf;
    uint64 seed = 11;
    for (int depth : {CV_8U, CV_16U, CV_32F})
        for (int cn : {1, 3, 4})
            for (int interp : {INTER_NEAREST, INTER_LINEAR, INTER_CUBIC})
                for (const WaCase &c : kGeoms)
                {
                    const int type = CV_MAKETYPE(depth, cn);
                    const Mat host = randomImage(sz, type, seed++);
                    const Mat M = toMat6(c.M);
                    const int flags = interp | (c.inverse ? WARP_INVERSE_MAP : 0);
                    const Size ds = c.dsize_from_src ? sz : c.dsize;
                    const Mat got = gpuWarp(host, M, ds, flags, c.border);
                    const Mat ref_ = cpuWarp(host, M, ds, flags, c.border);
                    WarpStats st;
                    EXPECT_TRUE(meetsWarpContract(got, ref_, &st))
                        << cv::format("%s depth=%d cn=%d interp=%d", c.name, depth, cn, interp);
                    (depth == CV_8U ? cu : (depth == CV_16U ? cu16 : cf)).merge(st);
                }
    std::cout << cv::format(
        "warpAffine kernel vs the selected CPU reference "
        "(vector path for NEAREST/LINEAR, natural width for CUBIC):\n"
        "  CV_8U  elements %lld  not-identical %lld (%.4f%%)  max|d| %.0f  max norm %.3f\n"
        "  CV_16U elements %lld  not-identical %lld (%.4f%%)  max|d| %.0f  max norm %.3f\n"
        "  CV_32F elements %lld  not-identical %lld (%.4f%%)  max|d| %.3e  max norm %.3f  max %.2f ULP\n",
        cu.elements, cu.not_identical,
        cu.elements ? 100.0 * cu.not_identical / double(cu.elements) : 0.0, cu.max_abs, cu.max_norm,
        cu16.elements, cu16.not_identical,
        cu16.elements ? 100.0 * cu16.not_identical / double(cu16.elements) : 0.0, cu16.max_abs,
        cu16.max_norm, cf.elements, cf.not_identical,
        cf.elements ? 100.0 * cf.not_identical / double(cf.elements) : 0.0, cf.max_abs, cf.max_norm,
        cf.max_ulp);
    // Check both the allowed bounds and exact integer results for this fixture.
    EXPECT_EQ(0, cu.outside);
    EXPECT_EQ(0, cu16.outside);
    EXPECT_EQ(0, cf.outside);
    EXPECT_EQ(0, cu.not_identical) << "uint8 was expected to match the selected reference exactly";
    EXPECT_EQ(0, cu16.not_identical)
        << "uint16 was expected to match the selected reference exactly";
}

// Check per-channel Scalar values and fractional CV_32F border components.
TEST(Triton_WarpAffine, BorderValueIsAScalar)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat M = M6(1, 0, -1000, 0, 1, -1000); // everything outside
    const Size sz(9, 7);

    // uint8: a single scalar fills channel 0 only; a triple fills all three.
    {
        const Mat host = randomImage(sz, CV_8UC3, 1);
        for (const Scalar bv : {Scalar(77), Scalar(10, 20, 30), Scalar(10, 20, 30, 40)})
        {
            SCOPED_TRACE(cv::format("u8 borderValue=(%g,%g,%g,%g)", bv[0], bv[1], bv[2], bv[3]));
            const Mat got = gpuWarp(host, M, sz, INTER_NEAREST, BORDER_CONSTANT, bv);
            const Mat ref_ = cpuWarp(host, M, sz, INTER_NEAREST, BORDER_CONSTANT, bv);
            EXPECT_EQ(0.0, cvtest::norm(got, ref_, NORM_INF));
            EXPECT_EQ(static_cast<int>(bv[0]), int(got.ptr<uchar>(0)[0]));
            EXPECT_EQ(static_cast<int>(bv[1]), int(got.ptr<uchar>(0)[1]));
            EXPECT_EQ(static_cast<int>(bv[2]), int(got.ptr<uchar>(0)[2]));
        }
    }
    // float32: a non-integer border must survive exactly, which an integer key cannot carry.
    {
        const Mat host = randomImage(sz, CV_32FC1, 2);
        for (const Scalar bv : {Scalar(0.5), Scalar(-1.75), Scalar(3.25)})
        {
            SCOPED_TRACE(cv::format("f32 borderValue=%g", bv[0]));
            const Mat got = gpuWarp(host, M, sz, INTER_LINEAR, BORDER_CONSTANT, bv);
            EXPECT_NEAR(bv[0], double(got.ptr<float>(0)[0]), 1e-6);
            const Mat ref_ = cpuWarp(host, M, sz, INTER_LINEAR, BORDER_CONSTANT, bv);
            EXPECT_TRUE(meetsWarpContract(got, ref_));
        }
        const Mat host3 = randomImage(sz, CV_32FC3, 3);
        const Scalar bv(0.5, 1.25, 2.75);
        const Mat got = gpuWarp(host3, M, sz, INTER_LINEAR, BORDER_CONSTANT, bv);
        EXPECT_NEAR(bv[0], double(got.ptr<float>(0)[0]), 1e-6);
        EXPECT_NEAR(bv[1], double(got.ptr<float>(0)[1]), 1e-6);
        EXPECT_NEAR(bv[2], double(got.ptr<float>(0)[2]), 1e-6);
    }
}

// NEAREST rounds half to even on exact .5 coordinates; LINEAR must be exact on integer
// coordinates and behave at the tie and near-tie boundaries a float32 cannot resolve.
TEST(Triton_WarpAffine, SubPixelCoordinates)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Size sz(64, 48);
    const Mat host8 = randomImage(sz, CV_8UC1, 21);
    const Mat hostf = randomImage(sz, CV_32FC1, 22);
    const double shifts[] = {0.0, 0.5, 0.25, 0.4999999, 0.5000001, 0.75, -0.5, -0.25};
    for (double dx : shifts)
        for (double dy : shifts)
        {
            SCOPED_TRACE(cv::format("shift=(%.7f, %.7f)", dx, dy));
            const Mat M = M6(1, 0, dx, 0, 1, dy);
            EXPECT_EQ(0.0, cvtest::norm(gpuWarp(host8, M, sz, INTER_NEAREST, BORDER_REPLICATE),
                                        cpuWarp(host8, M, sz, INTER_NEAREST, BORDER_REPLICATE),
                                        NORM_INF));
            EXPECT_TRUE(meetsWarpContract(gpuWarp(hostf, M, sz, INTER_LINEAR, BORDER_REPLICATE),
                                          cpuWarp(hostf, M, sz, INTER_LINEAR, BORDER_REPLICATE)));
        }
    // Integer coordinates with LINEAR: the interpolated value must equal the source value
    // exactly, which catches a weight that does not sum to one.
    // (norm(A) alone would be the image's own maximum, not a difference -- compare to the input)
    const Mat id = M6(1, 0, 0, 0, 1, 0);
    EXPECT_EQ(0.0,
              cvtest::norm(gpuWarp(hostf, id, sz, INTER_LINEAR, BORDER_CONSTANT), hostf, NORM_INF));
}

// A singular matrix is a legal warpAffine input (D = 0), not an error: the result is a
// constant image at the border value.
// Hardware FMA capability must not be confused with this linked CPU build's
// selected arithmetic. Exercise both useOptimized states without changing the
// frozen vector reference, over every dtype/channel dispatch at known ties.
TEST(Triton_WarpAffine, CoordinateFusionFollowsTheCpu)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    struct RestoreOptimized
    {
        bool previous = cv::useOptimized();
        ~RestoreOptimized()
        {
            cv::setUseOptimized(previous);
        }
    } restore;
    const Mat M = M6(1, 1.9, -20, .4, 1, -10);
    for (bool optimized : {true, false, true})
    {
        cv::setUseOptimized(optimized);
        for (int depth : {CV_8U, CV_16U, CV_32F})
            for (int cn : {1, 3, 4})
                for (int mode : {INTER_NEAREST, INTER_LINEAR, INTER_CUBIC})
                {
                    SCOPED_TRACE(cv::format("optimized=%d depth=%d cn=%d mode=%d hardwareFMA=%d",
                                            optimized, depth, cn, mode,
                                            cv::checkHardwareSupport(CV_CPU_FMA3)));
                    const Mat src = randomImage(Size(53, 37), CV_MAKETYPE(depth, cn), 11);
                    const Mat got = gpuWarp(src, M, Size(53, 37), mode, BORDER_CONSTANT);
                    const Mat ref = cpuWarp(src, M, Size(53, 37), mode, BORDER_CONSTANT);
                    EXPECT_TRUE(meetsWarpContract(got, ref));
                }
    }
}

TEST(Triton_WarpAffine, SingularMatrix)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(16, 16), CV_8UC1, 31);
    const Mat M = M6(1, 2, 5, 2, 4, 7); // rank 1
    const Mat got = gpuWarp(host, M, Size(16, 16), INTER_LINEAR, BORDER_CONSTANT, Scalar(9));
    EXPECT_EQ(0.0,
              cvtest::norm(got,
                           cpuWarp(host, M, Size(16, 16), INTER_LINEAR, BORDER_CONSTANT, Scalar(9)),
                           NORM_INF));
    // With WARP_INVERSE_MAP the matrix is used as given, singular or not.
    const Mat goti =
        gpuWarp(host, M, Size(16, 16), INTER_NEAREST | WARP_INVERSE_MAP, BORDER_REPLICATE);
    EXPECT_EQ(0.0, cvtest::norm(goti,
                                cpuWarp(host, M, Size(16, 16), INTER_NEAREST | WARP_INVERSE_MAP,
                                        BORDER_REPLICATE),
                                NORM_INF));
}

// Coordinates far outside int16/int32 ranges: the index arithmetic must stay in a wide type,
// and the modulo forms must still land in range rather than wrap into a valid-looking pixel.
TEST(Triton_WarpAffine, ExtremeCoordinates)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(32, 32), CV_8UC3, 41);
    const double ts[] = {1e3, -1e3, 1e5, -1e5};
    for (double t : ts)
        for (int border : {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP})
        {
            SCOPED_TRACE(cv::format("t=%g border=%d", t, border));
            const Mat M = M6(1, 0, t, 0, 1, -t);
            EXPECT_EQ(0.0, cvtest::norm(gpuWarp(host, M, Size(32, 32), INTER_NEAREST, border),
                                        cpuWarp(host, M, Size(32, 32), INTER_NEAREST, border),
                                        NORM_INF));
        }
}

// A 1-pixel axis is sampled with REPLICATE whatever the requested mode, except for CONSTANT.
TEST(Triton_WarpAffine, DegenerateAxes)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat row = randomImage(Size(64, 1), CV_8UC3, 51);
    const Mat col = randomImage(Size(1, 64), CV_8UC3, 52);
    const Mat M = M6(1, 0, 0.3, 0, 1, 0.2);
    for (int border : {BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_WRAP})
    {
        SCOPED_TRACE(cv::format("row/col border=%d", border));
        EXPECT_EQ(0.0, cvtest::norm(gpuWarp(row, M, row.size(), INTER_LINEAR, border),
                                    cpuWarp(row, M, row.size(), INTER_LINEAR, border), NORM_INF));
        EXPECT_EQ(0.0, cvtest::norm(gpuWarp(col, M, col.size(), INTER_LINEAR, border),
                                    cpuWarp(col, M, col.size(), INTER_LINEAR, border), NORM_INF));
    }
}

// ---------------------------------------------------------------- layout and resources

// ROI source and destination, a padded stride, and guard bands: nothing outside the
// destination ROI is written, and the result equals the CPU warp of the source ROI.
TEST(Triton_WarpAffine, RoiAndPaddedLayout)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Size sz(41, 29);
    const Mat host = randomImage(sz, CV_8UC3, 61);
    const Mat M = M6(0.9, -0.2, 1.5, 0.2, 0.9, -1.0);

    cuda::GpuMat srcParent;
    cuda::GpuMat srcRoi = roiOf(srcParent, sz, CV_8UC3, Point(3, 2), 6, 5, 0.0);
    srcRoi.upload(host);
    const Mat ref_ = cpuWarp(host, M, sz, INTER_LINEAR, BORDER_REPLICATE);

    const double guard = 250.0;
    cuda::GpuMat dstParent;
    cuda::GpuMat dstRoi = roiOf(dstParent, sz, CV_8UC3, Point(5, 4), 9, 7, guard);
    cv::triton::warpAffine(srcRoi, dstRoi, M, sz, INTER_LINEAR, BORDER_REPLICATE);
    cuda::Stream::Null().waitForCompletion();
    Mat got;
    dstRoi.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, ref_, NORM_INF));
    EXPECT_TRUE(guardsIntact(dstParent, Rect(5, 4, sz.width, sz.height), guard))
        << "the destination guard band was written";
}

// Output reuse: a second call into the same GpuMat must not reallocate it. An off-centre
// warp of a smaller destination exercises the case where the reused buffer is larger than
// the new result, which is where a "recreate always" bug hides.
TEST(Triton_WarpAffine, ReusesOutput)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(97, 131), CV_8UC1, 71);
    const Mat M = M6(0.7, -0.3, 5.0, 0.3, 0.7, -4.0);
    cuda::GpuMat src(host), dst;
    cv::triton::warpAffine(src, dst, M, Size(97, 131), INTER_LINEAR, BORDER_CONSTANT);
    cuda::Stream::Null().waitForCompletion();
    const void *before = dst.cudaPtr();
    cv::triton::warpAffine(src, dst, M, Size(97, 131), INTER_LINEAR, BORDER_CONSTANT);
    cuda::Stream::Null().waitForCompletion();
    EXPECT_EQ(before, dst.cudaPtr()) << "the output was reallocated on the second call";
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, cpuWarp(host, M, Size(97, 131), INTER_LINEAR, BORDER_CONSTANT),
                                NORM_INF));
}

// A non-default stream: the work is ordered on it, and a CPU read after
// waitForCompletion sees the result.
TEST(Triton_WarpAffine, NonDefaultStream)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(73, 51), CV_32FC3, 81);
    const Mat M = M6(1.1, 0, -2.0, 0, 1.1, -1.0);
    cuda::Stream s;
    const Mat got = gpuWarp(host, M, Size(73, 51), INTER_LINEAR, BORDER_REFLECT_101, Scalar(), &s);
    EXPECT_TRUE(
        meetsWarpContract(got, cpuWarp(host, M, Size(73, 51), INTER_LINEAR, BORDER_REFLECT_101)));
    EXPECT_STREQ("linear", td::runtimeStats().last_route.c_str());
}

TEST(Triton_WarpAffine, AreaIsLinear)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Size size(37, 53);
    const Mat M = M6(0.9, -0.43589, 4, 0.43589, 0.9, -2);
    for (int depth : {CV_8U, CV_16U, CV_32F})
        for (int cn : {1, 3, 4})
            for (int border : {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
                               BORDER_REFLECT_101, BORDER_WRAP, BORDER_TRANSPARENT})
                for (int inverse : {0, int(WARP_INVERSE_MAP)})
                {
                    SCOPED_TRACE(cv::format("depth=%d cn=%d border=%d inverse=%d", depth, cn,
                                            border, inverse));
                    const Mat host = randomImage(size, CV_MAKETYPE(depth, cn), 810);
                    const Mat initial = randomImage(size, host.type(), 811);
                    cuda::GpuMat src(host), linear(initial), area(initial);
                    cv::triton::warpAffine(src, linear, M, size, INTER_LINEAR | inverse, border);
                    cv::triton::warpAffine(src, area, M, size, INTER_AREA | inverse, border);
                    Mat gotLinear, gotArea;
                    linear.download(gotLinear);
                    area.download(gotArea);
                    EXPECT_EQ(0.0, cvtest::norm(gotArea, gotLinear, NORM_INF));

                    // Prefill both CPU targets too: TRANSPARENT reads their previous values.
                    Mat cpuLinear = initial.clone(), cpuArea = initial.clone();
                    cv::warpAffine(host, cpuLinear, M, size, INTER_LINEAR | inverse, border);
                    cv::warpAffine(host, cpuArea, M, size, INTER_AREA | inverse, border);
                    EXPECT_EQ(0.0, cvtest::norm(cpuArea, cpuLinear, NORM_INF));
                }
}

TEST(Triton_WarpAffine, TransparentMatchesCpu)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    for (int depth : {CV_8U, CV_16U, CV_32F})
        for (int cn : {1, 3, 4})
            for (const Size &size : {Size(128, 48), Size(37, 53), Size(1, 1), Size(7, 9)})
                for (int geometry = 0; geometry < 3; ++geometry)
                    for (int interp : {INTER_NEAREST, INTER_LINEAR, INTER_AREA})
                    {
                        SCOPED_TRACE(cv::format("depth=%d cn=%d size=%dx%d geometry=%d interp=%d",
                                                depth, cn, size.width, size.height, geometry,
                                                interp));
                        const int type = CV_MAKETYPE(depth, cn);
                        const Rect roi(3, 2, size.width, size.height);
                        const Mat backing =
                            randomImage(Size(size.width + 8, size.height + 6), type, 820);
                        const Mat host = backing(roi);
                        const Mat initial = randomImage(size, type, 821);
                        Mat M = M6(1, 0, 0.5, 0, 1, -0.5);
                        if (geometry == 1)
                            M = M6(0.9, 0.43589, -14, -0.43589, 0.9, 10);
                        if (geometry == 2)
                            M = M6(1, 0, size.width + 4, 0, 1, size.height + 4);

                        // Use the same selected vector-path reference as cpuWarp, but preserve the initial
                        // destination. Padding an empty target would read uninitialised values at the border.
                        const int wideWidth = ((size.width + 255) / 128) * 128;
                        Mat wide = randomImage(Size(wideWidth, size.height), type, 822);
                        initial.copyTo(wide(Rect(0, 0, size.width, size.height)));
                        cv::warpAffine(host, wide, M, wide.size(), interp | WARP_INVERSE_MAP,
                                       BORDER_TRANSPARENT);
                        const Mat reference = wide(Rect(0, 0, size.width, size.height));

                        cuda::GpuMat srcBacking(backing), src = srcBacking(roi);
                        const Mat guard = randomImage(backing.size(), type, 823);
                        Mat expectedBacking = guard.clone();
                        reference.copyTo(expectedBacking(roi));
                        cuda::GpuMat dstBacking(guard), dst = dstBacking(roi);
                        dst.upload(initial);
                        cv::triton::warpAffine(src, dst, M, size, interp | WARP_INVERSE_MAP,
                                               BORDER_TRANSPARENT);
                        Mat got, whole;
                        dst.download(got);
                        dstBacking.download(whole);
                        EXPECT_TRUE(meetsWarpContract(got, reference));
                        EXPECT_TRUE(meetsWarpContract(whole, expectedBacking));
                        // Guard bands must remain byte-identical, even for floating-point output.
                        guard(roi).copyTo(whole(roi));
                        EXPECT_EQ(0.0, cvtest::norm(whole, guard, NORM_INF));
                        if (geometry == 2)
                        {
                            EXPECT_EQ(0.0, cvtest::norm(got, initial, NORM_INF))
                                << "fully outside must preserve dst";
                        }
                    }
}

// ---------------------------------------------------------------- rejection surface

// These requests carry address identities only; normalize never dereferences them.
// Keep validation coverage runnable on a host without a CUDA device.
TEST(Triton_WarpAffine, PureOutputAdmission)
{
    // Source/options resolution must need neither a destination allocation nor
    // a readable source pointer or CPU fusion calibration.
    wa::Request request;
    request.src.type = cv_triton_runtime::ElemType::F32;
    request.src.channels = 3;
    request.inverse_map = true;
    EXPECT_TRUE(wa::validate_source(request).ok());
    EXPECT_TRUE(wa::validate_output(request, 37, 53).ok());
    EXPECT_EQ(cv_triton_runtime::Reason::EmptyOrSize, wa::validate_output(request, 0, 53).reason);
    request.border[0] = 1e100;
    EXPECT_EQ(cv_triton_runtime::Reason::IndexRange, wa::validate_source(request).reason);
    request.border[0] = 0;
    request.M[2] = 2147483648.0;
    EXPECT_TRUE(wa::validate_source(request).ok());
    EXPECT_EQ(cv_triton_runtime::Reason::IndexRange, wa::validate_output(request, 37, 53).reason);
    request.interpolation = INTER_CUBIC;
    EXPECT_TRUE(wa::validate_output(request, 37, 53).ok());
    request.M[0] = 1e38;
    request.M[2] = 0;
    EXPECT_EQ(cv_triton_runtime::Reason::IndexRange, wa::validate_output(request, 37, 53).reason);
    request.M[0] = 1;
    request.interpolation = INTER_AREA;
    request.border_mode = BORDER_TRANSPARENT;
    EXPECT_TRUE(wa::validate_output(request, 1, 1).ok());
    request.interpolation = INTER_CUBIC;
    EXPECT_EQ(cv_triton_runtime::Reason::Interpolation, wa::validate_source(request).reason);
}

TEST(Triton_WarpAffine, HostSamplingBoundaries)
{
    wa::Request request;
    request.src.ptr = 4096;
    request.src.rows = 37;
    request.src.cols = 53;
    request.src.channels = 1;
    request.src.step = 53;
    request.dst = request.src;
    request.dst.ptr = 16384;
    request.inverse_map = true;
    wa::Normalized normalized;
    for (int interp : {INTER_NEAREST, INTER_LINEAR, INTER_AREA})
    {
        request.interpolation = interp;
        request.M[2] = 2147483648.0;
        EXPECT_EQ(cv_triton_runtime::Reason::IndexRange, wa::normalize(request, normalized).reason);
    }
    request.interpolation = INTER_CUBIC;
    EXPECT_TRUE(wa::normalize(request, normalized).ok()); // CUBIC clamps before int32 conversion
    request.M[0] = 1e38;
    request.M[2] = 0;
    EXPECT_EQ(cv_triton_runtime::Reason::IndexRange, wa::normalize(request, normalized).reason);
    request.M[0] = 1;
    for (double invalid :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        request.M[2] = invalid;
        EXPECT_EQ(cv_triton_runtime::Reason::IndexRange, wa::normalize(request, normalized).reason);
        request.M[2] = 0;
        request.border[0] = invalid;
        EXPECT_EQ(cv_triton_runtime::Reason::IndexRange, wa::normalize(request, normalized).reason);
    }
    request.border[0] = 1e100;
    request.src.type = request.dst.type = cv_triton_runtime::ElemType::F32;
    request.src.step = request.dst.step = 53 * sizeof(float);
    EXPECT_EQ(cv_triton_runtime::Reason::IndexRange, wa::normalize(request, normalized).reason);
}

TEST(Triton_WarpAffine, HostAreaTransparentNormalization)
{
    wa::Request request;
    request.src.ptr = 4096;
    request.src.rows = 1;
    request.src.cols = 1;
    request.src.channels = 1;
    request.src.step = 1;
    request.dst = request.src;
    request.dst.ptr = 16384;
    request.border_mode = BORDER_TRANSPARENT;
    request.interpolation = INTER_AREA;
    wa::Normalized area, linear;
    ASSERT_TRUE(wa::normalize(request, area).ok());
    request.interpolation = INTER_LINEAR;
    ASSERT_TRUE(wa::normalize(request, linear).ok());
    EXPECT_EQ(wa::Mode::Linear, area.mode);
    EXPECT_EQ(wa::constexprs_text(linear), wa::constexprs_text(area));
    EXPECT_EQ(BORDER_TRANSPARENT, area.border_x);
    EXPECT_EQ(BORDER_TRANSPARENT, area.border_y);
    EXPECT_NE(std::string::npos, wa::constexprs_text(area).find("TRANSPARENT=1"));
    request.interpolation = INTER_CUBIC;
    EXPECT_EQ(cv_triton_runtime::Reason::Interpolation, wa::normalize(request, area).reason);
}

TEST(Triton_WarpAffine, RejectsWithoutFallback)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const Mat host = randomImage(Size(32, 32), CV_8UC1, 91);
    const Mat M = M6(1, 0, 0, 0, 1, 0);
    const double before = double(td::runtimeStats().launches);

    // AREA aliases LINEAR; unsupported LANCZOS4 must be refused before submission.
    for (int interp : {INTER_LANCZOS4})
    {
        SCOPED_TRACE(cv::format("interp=%d", interp));
        cuda::GpuMat src(host), dst;
        EXPECT_THROW(cv::triton::warpAffine(src, dst, M, Size(32, 32), interp, BORDER_CONSTANT),
                     cv::Exception);
    }
    // Only interpolation plus WARP_INVERSE_MAP belongs to this entry's flag
    // surface. Known-but-unserved flags must not silently change the request.
    for (int flags : {INTER_LINEAR | WARP_FILL_OUTLIERS, INTER_LINEAR | 32, -1})
    {
        cuda::GpuMat src(host), dst;
        EXPECT_THROW(cv::triton::warpAffine(src, dst, M, host.size(), flags), cv::Exception);
    }
    // C2 and CV_16S need the CPU's other numerical path and are deliberately refused.
    for (int type : {CV_8UC2, CV_16UC2, CV_32FC2, CV_16SC1, CV_16SC3, CV_16SC4, CV_8UC(5)})
        for (int interp : {INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA})
        {
            SCOPED_TRACE(cv::format("refused type=%d interp=%d", type, interp));
            cuda::GpuMat src2(randomImage(Size(32, 32), type, 93)), dst2;
            EXPECT_THROW(cv::triton::warpAffine(src2, dst2, M, Size(32, 32), interp),
                         cv::Exception);
        }
    for (int depth : {CV_8U, CV_16U, CV_32F})
        for (int cn : {1, 3, 4})
        {
            const Mat initial = randomImage(Size(32, 32), CV_MAKETYPE(depth, cn), 94);
            cuda::GpuMat src2(initial), dst2(initial);
            EXPECT_THROW(cv::triton::warpAffine(src2, dst2, M, initial.size(), INTER_CUBIC,
                                                BORDER_TRANSPARENT),
                         cv::Exception);
            Mat got;
            dst2.download(got);
            EXPECT_EQ(0.0, cvtest::norm(got, initial, NORM_INF));
        }
    // Matrix shape and type are cv::warpAffine's own preconditions.
    {
        cuda::GpuMat src(host), dst;
        Mat bad33(3, 3, CV_64F, Scalar(0)), badint(2, 3, CV_32S, Scalar(0));
        EXPECT_THROW(cv::triton::warpAffine(src, dst, bad33, Size(32, 32), INTER_LINEAR),
                     cv::Exception);
        EXPECT_THROW(cv::triton::warpAffine(src, dst, badint, Size(32, 32), INTER_LINEAR),
                     cv::Exception);
    }
    // Overlapping input and output: refused, not silently producing garbage.
    {
        cuda::GpuMat buf(host);
        EXPECT_THROW(cv::triton::warpAffine(buf, buf, M, Size(32, 32), INTER_LINEAR),
                     cv::Exception);
    }
    EXPECT_EQ(before, double(td::runtimeStats().launches)) << "a refused call still launched";
}

}
} // namespace opencv_test
