// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

template <typename ElemType>
class GpuMatNDTest : public ::testing::Test
{
public:
    using MatType = Mat_<ElemType>;
    using CnType = typename Mat_<ElemType>::channel_type;
    static constexpr int cn = DataType<ElemType>::channels;
    using SizeArray = GpuMatND::SizeArray;

    static MatType RandomMat(const SizeArray& size)
    {
        const auto dims = static_cast<int>(size.size());

        MatType ret(dims, size.data());

        for (ElemType& elem : ret)
            for (int i = 0; i < cn; ++i)
                elem[i] = cv::randu<CnType>();

        return ret;
    }

    static std::vector<Range> RandomRange(const SizeArray& size)
    {
        const auto dims = static_cast<int>(size.size());

        std::vector<Range> ret;

        const auto margin = cv::randu<int>() & 0x1 + 1; // 1 or 2

        for (int s : size)
            if (s > margin * 2)
                ret.emplace_back(margin, s-margin);
            else
                ret.push_back(Range::all());

        if (dims == 1)
        {
            // Mat expects two ranges even in this case
            ret.push_back(Range::all());
        }

        return ret;
    }

    static std::vector<Range> RandomRange2D(const SizeArray& size)
    {
        const auto dims = static_cast<int>(size.size());

        std::vector<Range> ret = RandomRange(size);

        for (int i = 0; i < dims - 2; ++i)
        {
            const auto start = cv::randu<unsigned int>() % size[i];
            ret[i] = Range(static_cast<int>(start), static_cast<int>(start) + 1);
        }

        return ret;
    }

    static void doTest1(const SizeArray& size)
    {
        const MatType gold = RandomMat(size);

        MatType dst;
        GpuMatND gmat;

        // simple upload, download test for GpuMatND
        gmat.upload(gold);
        gmat.download(dst);
        EXPECT_TRUE(std::equal(gold.begin(), gold.end(), dst.begin()));
    }

    static void doTest2(const SizeArray& size)
    {
        const MatType gold = RandomMat(size);
        const std::vector<Range> ranges = RandomRange(size);
        const MatType goldSub = gold(ranges);

        MatType dst;
        GpuMatND gmat;

        // upload partial mat, download it, and compare
        gmat.upload(goldSub);
        gmat.download(dst);
        EXPECT_TRUE(std::equal(goldSub.begin(), goldSub.end(), dst.begin()));

        // upload full mat, extract partial mat from it, download it, and compare
        gmat.upload(gold);
        gmat = gmat(ranges);
        gmat.download(dst);
        EXPECT_TRUE(std::equal(goldSub.begin(), goldSub.end(), dst.begin()));
    }

    static void doTest3(const SizeArray& size)
    {
        if (std::is_same<CnType, float16_t>::value) // GpuMat::convertTo is not implemented for CV_16F
            return;

        const MatType gold = RandomMat(size);
        const std::vector<Range> ranges = RandomRange2D(size);

        MatType dst;
        GpuMatND gmat;

        // Test GpuMatND to GpuMat conversion:
        // extract a 2D-plane and set its elements in the extracted region to 1
        // compare the values of the full mat between Mat and GpuMatND

        gmat.upload(gold);
        GpuMat plane = gmat(ranges).createGpuMatHeader();
        EXPECT_TRUE(!plane.refcount); // plane points to externally allocated memory(a part of gmat)

        const GpuMat dummy = plane.clone();
        EXPECT_TRUE(dummy.refcount); // dummy is clone()-ed from plane, so it manages its memory

        // currently, plane(GpuMat) points to a sub-matrix of gmat(GpuMatND)
        // in this case, dummy and plane have same size and type,
        // so plane does not get reallocated inside convertTo,
        // so this convertTo sets a sub-matrix region of gmat to 1
        dummy.convertTo(plane, -1, 0, 1);
        EXPECT_TRUE(!plane.refcount); // plane still points to externally allocated memory(a part of gmat)

        gmat.download(dst);

        // set a sub-matrix region of gold to 1
        Mat plane_ = gold(ranges);
        const Mat dummy_ = plane_.clone();
        dummy_.convertTo(plane_, -1, 0, 1);

        EXPECT_TRUE(std::equal(gold.begin(), gold.end(), dst.begin()));
    }

    static void doTest4(const SizeArray& size)
    {
        if (std::is_same<CnType, float16_t>::value) // GpuMat::convertTo is not implemented for CV_16F
            return;

        const MatType gold = RandomMat(size);
        const std::vector<Range> ranges = RandomRange2D(size);

        MatType dst;
        GpuMatND gmat;

        // Test handling external memory
        gmat.upload(gold);
        const GpuMatND external(gmat.size, gmat.type(), gmat.getDevicePtr(), {gmat.step.begin(), gmat.step.end() - 1});

        // set a sub-matrix region of external to 2
        GpuMat plane = external(ranges).createGpuMatHeader();
        const GpuMat dummy = plane.clone();
        dummy.convertTo(plane, -1, 0, 2);
        external.download(dst);

        // set a sub-matrix region of gold to 2
        Mat plane_ = gold(ranges);
        const Mat dummy_ = plane_.clone();
        dummy_.convertTo(plane_, -1, 0, 2);

        EXPECT_TRUE(std::equal(gold.begin(), gold.end(), dst.begin()));
    }

    static void doTest5(const SizeArray& size)
    {
        if (std::is_same<CnType, float16_t>::value) // GpuMat::convertTo is not implemented for CV_16F
            return;

        const MatType gold = RandomMat(size);
        const std::vector<Range> ranges = RandomRange(size);
        MatType goldSub = gold(ranges);

        MatType dst;
        GpuMatND gmat;

        // Upload a sub-mat, set a sub-region of the sub-mat to 3, download, and compare
        gmat.upload(goldSub);
        const std::vector<Range> rangesInRanges = RandomRange2D(gmat.size);

        GpuMat plane = gmat(rangesInRanges).createGpuMatHeader();
        const GpuMat dummy = plane.clone();
        dummy.convertTo(plane, -1, 0, 3);
        gmat.download(dst);

        Mat plane_ = goldSub(rangesInRanges);
        const Mat dummy_ = plane_.clone();
        dummy_.convertTo(plane_, -1, 0, 3);

        EXPECT_TRUE(std::equal(goldSub.begin(), goldSub.end(), dst.begin()));
    }
};

using ElemTypes = ::testing::Types<
    Vec<uchar, 1>, Vec<uchar, 2>, Vec<uchar, 3>, Vec<uchar, 4>, // CV_8U
    Vec<schar, 1>, Vec<schar, 2>, Vec<schar, 3>, Vec<schar, 4>, // CV_8S
    Vec<ushort, 1>, Vec<ushort, 2>, Vec<ushort, 3>, Vec<ushort, 4>, // CV_16U
    Vec<short, 1>, Vec<short, 2>, Vec<short, 3>, Vec<short, 4>, // CV_16S
    Vec<int, 1>, Vec<int, 2>, Vec<int, 3>, Vec<int, 4>, // CV_32S
    Vec<float, 1>, Vec<float, 2>, Vec<float, 3>, Vec<float, 4>, // CV_32F
    Vec<double, 1>, Vec<double, 2>, Vec<double, 3>, Vec<double, 4>, //CV_64F
    Vec<float16_t, 1>, Vec<float16_t, 2>, Vec<float16_t, 3>, Vec<float16_t, 4> // CV_16F
>;

using SizeArray = GpuMatND::SizeArray;

#define DIFFERENT_SIZES_ND std::vector<SizeArray>{ \
    SizeArray{2, 1}, SizeArray{3, 2, 1}, SizeArray{1, 3, 2, 1}, SizeArray{2, 1, 3, 2, 1}, SizeArray{3, 2, 1, 3, 2, 1}, \
    SizeArray{1}, SizeArray{1, 1}, SizeArray{1, 1, 1}, SizeArray{1, 1, 1, 1}, \
    SizeArray{4}, SizeArray{4, 4}, SizeArray{4, 4, 4}, SizeArray{4, 4, 4, 4}, \
    SizeArray{11}, SizeArray{13, 11}, SizeArray{17, 13, 11}, SizeArray{19, 17, 13, 11}}

TYPED_TEST_CASE(GpuMatNDTest, ElemTypes);

TYPED_TEST(GpuMatNDTest, Test1)
{
    for (auto& size : DIFFERENT_SIZES_ND)
        GpuMatNDTest<TypeParam>::doTest1(size);
}

TYPED_TEST(GpuMatNDTest, Test2)
{
    for (auto& size : DIFFERENT_SIZES_ND)
        GpuMatNDTest<TypeParam>::doTest2(size);
}

TYPED_TEST(GpuMatNDTest, Test3)
{
    for (auto& size : DIFFERENT_SIZES_ND)
        GpuMatNDTest<TypeParam>::doTest3(size);
}

TYPED_TEST(GpuMatNDTest, Test4)
{
    for (auto& size : DIFFERENT_SIZES_ND)
        GpuMatNDTest<TypeParam>::doTest4(size);
}

TYPED_TEST(GpuMatNDTest, Test5)
{
    for (auto& size : DIFFERENT_SIZES_ND)
        GpuMatNDTest<TypeParam>::doTest5(size);
}

}} // namespace
