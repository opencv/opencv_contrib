// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test
{
namespace
{

class DummyAllocator : public AscendMat::Allocator
{
public:
    std::shared_ptr<uchar> allocate(size_t size) CV_OVERRIDE
    {
        CV_UNUSED(size);
        return std::shared_ptr<uchar>();
    }
    bool allocate(cv::cann::AscendMat* mat, int rows, int cols, size_t elemSize) CV_OVERRIDE
    {
        CV_UNUSED(rows);
        CV_UNUSED(cols);
        CV_UNUSED(elemSize);
        mat->data = std::shared_ptr<uchar>((uchar*)0x12345, [](void* ptr) { CV_UNUSED(ptr); });
        return true;
    }
};

TEST(AscendMat, Construct)
{
    cv::cann::setDevice(0);
    // 1 Default constructor.
    AscendMat defaultAscendMat;
    AscendMat::Allocator* defaultAllocator = AscendMat::defaultAllocator();
    ASSERT_EQ(defaultAscendMat.allocator, defaultAllocator);

    // 2 get & set allocator.
    DummyAllocator dummyAllocator;
    AscendMat::setDefaultAllocator(&dummyAllocator);
    ASSERT_EQ(defaultAscendMat.defaultAllocator(), &dummyAllocator);
    AscendMat::setDefaultAllocator(defaultAllocator);

    // 3 constructs AscendMat of the specified size and type
    AscendMat specifiedSizeAscendMat1(5, 6, CV_8UC3);
    AscendMat specifiedSizeAscendMat2(Size(300, 200), CV_64F);

    ASSERT_EQ(specifiedSizeAscendMat1.rows, 5);
    ASSERT_EQ(specifiedSizeAscendMat1.cols, 6);
    ASSERT_EQ(specifiedSizeAscendMat1.depth(), CV_8U);
    ASSERT_EQ(specifiedSizeAscendMat1.channels(), 3);

    ASSERT_EQ(specifiedSizeAscendMat2.cols, 300);
    ASSERT_EQ(specifiedSizeAscendMat2.rows, 200);
    ASSERT_EQ(specifiedSizeAscendMat2.depth(), CV_64F);
    ASSERT_EQ(specifiedSizeAscendMat2.channels(), 1);

    // 4 constructs AscendMat and fills it with the specified value s
    srand((unsigned int)(time(NULL)));
    Scalar sc(rand() % 256, rand() % 256, rand() % 256, rand() % 256);

    Mat scalarToMat(7, 8, CV_8UC3, sc);
    AscendMat scalarToAscendMat1(7, 8, CV_8UC3, sc);
    Mat scalarToMatChecker;
    scalarToAscendMat1.download(scalarToMatChecker);

    EXPECT_MAT_NEAR(scalarToMat, scalarToMatChecker, 0.0);

    AscendMat scalarToAscendMat2(Size(123, 345), CV_32S);

    ASSERT_EQ(scalarToAscendMat1.rows, 7);
    ASSERT_EQ(scalarToAscendMat1.cols, 8);
    ASSERT_EQ(scalarToAscendMat1.depth(), CV_8U);
    ASSERT_EQ(scalarToAscendMat1.channels(), 3);

    ASSERT_EQ(scalarToAscendMat2.cols, 123);
    ASSERT_EQ(scalarToAscendMat2.rows, 345);
    ASSERT_EQ(scalarToAscendMat2.depth(), CV_32S);
    ASSERT_EQ(scalarToAscendMat2.channels(), 1);

    // 6 builds AscendMat from host memory
    Scalar sc2(rand() % 256, rand() % 256, rand() % 256, rand() % 256);
    Mat randomMat(7, 8, CV_8UC3, sc2);
    InputArray arr = randomMat;

    AscendMat fromInputArray(arr, AscendStream::Null());
    Mat randomMatChecker;
    fromInputArray.download(randomMatChecker);
    EXPECT_MAT_NEAR(randomMat, randomMatChecker, 0.0);

    cv::cann::resetDevice();
}

TEST(AscendMat, Assignment)
{
    DummyAllocator dummyAllocator;
    AscendMat mat1;
    AscendMat mat2(3, 4, CV_8SC1, &dummyAllocator);
    mat1 = mat2;

    ASSERT_EQ(mat1.rows, 3);
    ASSERT_EQ(mat1.cols, 4);
    ASSERT_EQ(mat1.depth(), CV_8S);
    ASSERT_EQ(mat1.channels(), 1);
    ASSERT_EQ(mat1.data.get(), (uchar*)0x12345);
}

TEST(AscendMat, SetTo)
{
    cv::cann::setDevice(0);

    srand((unsigned int)(time(NULL)));
    Scalar sc(rand() % 256, rand() % 256, rand() % 256, rand() % 256);

    AscendMat ascendMat(2, 2, CV_8UC4);
    ascendMat.setTo(sc);
    Mat mat(2, 2, CV_8UC4, sc);
    Mat checker;
    ascendMat.download(checker);

    EXPECT_MAT_NEAR(mat, checker, 0.0);

    cv::cann::resetDevice();
}

TEST(AscendMat, ConvertTo)
{
    cv::cann::setDevice(0);

    srand((unsigned int)(time(NULL)));
    Scalar sc(rand() % 256, rand() % 256, rand() % 256, rand() % 256);

    AscendMat ascendMat(2, 2, CV_8UC4, sc);
    AscendMat convertedAscendMat;
    ascendMat.convertTo(convertedAscendMat, CV_16S);
    Mat mat(2, 2, CV_16SC4, sc);
    Mat checker;
    convertedAscendMat.download(checker);

    EXPECT_MAT_NEAR(mat, checker, 0.0);

    cv::cann::resetDevice();
}

} // namespace
} // namespace opencv_test
