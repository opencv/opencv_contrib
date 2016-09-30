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
#include <string>

//#define DUMP_RESULTS
//#define TEST_TRANSFORMS

#ifdef TEST_TRANSFORMS
#include "..\..\xphoto\src\bm3d_denoising_invoker_commons.hpp"
#include "..\..\xphoto\src\bm3d_denoising_transforms.hpp"
#include "..\..\xphoto\src\kaiser_window.hpp"
using namespace cv::xphoto;
#endif

#ifdef DUMP_RESULTS
#  define DUMP(image, path) imwrite(path, image)
#else
#  define DUMP(image, path)
#endif

#ifdef OPENCV_ENABLE_NONFREE

namespace cvtest
{
    TEST(xphoto_DenoisingBm3dGrayscale, regression_L2)
    {
        std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "cv/xphoto/bm3d_image_denoising/";
        std::string original_path = folder + "lena_noised_gaussian_sigma=10.png";
        std::string expected_path = folder + "lena_noised_denoised_bm3d_wiener_grayscale_l2_tw=4_sw=16_h=10_bm=400.png";

        cv::Mat original = cv::imread(original_path, cv::IMREAD_GRAYSCALE);
        cv::Mat expected = cv::imread(expected_path, cv::IMREAD_GRAYSCALE);

        ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
        ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

        // BM3D: two different calls doing exactly the same thing
        cv::Mat result, resultSec;
        cv::xphoto::bm3dDenoising(original, cv::Mat(), resultSec, 10, 4, 16, 2500, 400, 8, 1, 0.0f, cv::NORM_L2, cv::xphoto::BM3D_STEPALL);
        cv::xphoto::bm3dDenoising(original, result, 10, 4, 16, 2500, 400, 8, 1, 0.0f, cv::NORM_L2, cv::xphoto::BM3D_STEPALL);

        DUMP(result, expected_path + ".res.png");

        ASSERT_EQ(cvtest::norm(result, resultSec, cv::NORM_L2), 0);
        ASSERT_LT(cvtest::norm(result, expected, cv::NORM_L2), 200);
    }

    TEST(xphoto_DenoisingBm3dGrayscale, regression_L2_separate)
    {
        std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "cv/xphoto/bm3d_image_denoising/";
        std::string original_path = folder + "lena_noised_gaussian_sigma=10.png";
        std::string expected_basic_path = folder + "lena_noised_denoised_bm3d_grayscale_l2_tw=4_sw=16_h=10_bm=2500.png";
        std::string expected_path = folder + "lena_noised_denoised_bm3d_wiener_grayscale_l2_tw=4_sw=16_h=10_bm=400.png";

        cv::Mat original = cv::imread(original_path, cv::IMREAD_GRAYSCALE);
        cv::Mat expected_basic = cv::imread(expected_basic_path, cv::IMREAD_GRAYSCALE);
        cv::Mat expected = cv::imread(expected_path, cv::IMREAD_GRAYSCALE);

        ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
        ASSERT_FALSE(expected_basic.empty()) << "Could not load reference image " << expected_basic_path;
        ASSERT_FALSE(expected.empty()) << "Could not load input image " << expected_path;

        cv::Mat basic, result;

        // BM3D step 1
        cv::xphoto::bm3dDenoising(original, basic, 10, 4, 16, 2500, -1, 8, 1, 0.0f, cv::NORM_L2, cv::xphoto::BM3D_STEP1);
        ASSERT_LT(cvtest::norm(basic, expected_basic, cv::NORM_L2), 200);
        DUMP(basic, expected_basic_path + ".res.basic.png");

        // BM3D step 2
        cv::xphoto::bm3dDenoising(original, basic, result, 10, 4, 16, 2500, 400, 8, 1, 0.0f, cv::NORM_L2, cv::xphoto::BM3D_STEP2);
        ASSERT_LT(cvtest::norm(basic, expected_basic, cv::NORM_L2), 200);
        DUMP(basic, expected_basic_path + ".res.basic2.png");

        DUMP(result, expected_path + ".res.png");

        ASSERT_LT(cvtest::norm(result, expected, cv::NORM_L2), 200);
    }

    TEST(xphoto_DenoisingBm3dGrayscale, regression_L1)
    {
        std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "cv/xphoto/bm3d_image_denoising/";
        std::string original_path = folder + "lena_noised_gaussian_sigma=10.png";
        std::string expected_path = folder + "lena_noised_denoised_bm3d_grayscale_l1_tw=4_sw=16_h=10_bm=2500.png";

        cv::Mat original = cv::imread(original_path, cv::IMREAD_GRAYSCALE);
        cv::Mat expected = cv::imread(expected_path, cv::IMREAD_GRAYSCALE);

        ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
        ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

        cv::Mat result;
        cv::xphoto::bm3dDenoising(original, result, 10, 4, 16, 2500, -1, 8, 1, 0.0f, cv::NORM_L1, cv::xphoto::BM3D_STEP1);

        DUMP(result, expected_path + ".res.png");

        ASSERT_LT(cvtest::norm(result, expected, cv::NORM_L2), 200);
    }

    TEST(xphoto_DenoisingBm3dGrayscale, regression_L2_8x8)
    {
        std::string folder = std::string(cvtest::TS::ptr()->get_data_path()) + "cv/xphoto/bm3d_image_denoising/";
        std::string original_path = folder + "lena_noised_gaussian_sigma=10.png";
        std::string expected_path = folder + "lena_noised_denoised_bm3d_grayscale_l2_tw=8_sw=16_h=10_bm=2500.png";

        cv::Mat original = cv::imread(original_path, cv::IMREAD_GRAYSCALE);
        cv::Mat expected = cv::imread(expected_path, cv::IMREAD_GRAYSCALE);

        ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
        ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

        cv::Mat result;
        cv::xphoto::bm3dDenoising(original, result, 10, 8, 16, 2500, -1, 8, 1, 0.0f, cv::NORM_L2, cv::xphoto::BM3D_STEP1);

        DUMP(result, expected_path + ".res.png");

        ASSERT_LT(cvtest::norm(result, expected, cv::NORM_L2), 200);
    }

#ifdef TEST_TRANSFORMS

    TEST(xphoto_DenoisingBm3dKaiserWindow, regression_4)
    {
        float beta = 2.0f;
        int N = 4;

        cv::Mat kaiserWindow;
        calcKaiserWindow1D(kaiserWindow, N, beta);

        float kaiser4[] = {
            0.43869004f,
            0.92432547f,
            0.92432547f,
            0.43869004f
        };

        for (int i = 0; i < N; ++i)
            ASSERT_FLOAT_EQ(kaiser4[i], kaiserWindow.at<float>(i));
    }

    TEST(xphoto_DenoisingBm3dKaiserWindow, regression_8)
    {
        float beta = 2.0f;
        int N = 8;

        cv::Mat kaiserWindow;
        calcKaiserWindow1D(kaiserWindow, N, beta);

        float kaiser8[] = {
            0.43869004f,
            0.68134475f,
            0.87685609f,
            0.98582518f,
            0.98582518f,
            0.87685609f,
            0.68134463f,
            0.43869004f
        };

        for (int i = 0; i < N; ++i)
            ASSERT_FLOAT_EQ(kaiser8[i], kaiserWindow.at<float>(i));
    }

    TEST(xphoto_DenoisingBm3dTransforms, regression_2D_generic)
    {
        const int templateWindowSize = 8;
        const int templateWindowSizeSq = templateWindowSize * templateWindowSize;

        uchar src[templateWindowSizeSq];
        short dst[templateWindowSizeSq];
        short dstSec[templateWindowSizeSq];

        // Initialize array
        for (uchar i = 0; i < templateWindowSizeSq; ++i)
            src[i] = (i % 10) * 10;

        // Use tailored transforms
        HaarTransform<uchar, short>::RegisterTransforms2D(templateWindowSize);
        HaarTransform<uchar, short>::forwardTransform2D(src, dst, templateWindowSize, templateWindowSize);
        HaarTransform<uchar, short>::inverseTransform2D(dst, templateWindowSize);

        // Use generic transforms
        HaarTransform2D::ForwardTransformXxX<uchar, short, templateWindowSize>(src, dstSec, templateWindowSize, templateWindowSize);
        HaarTransform2D::InverseTransformXxX<short, templateWindowSize>(dstSec, templateWindowSize);

        for (unsigned i = 0; i < templateWindowSizeSq; ++i)
            ASSERT_EQ(dst[i], dstSec[i]);
    }

    TEST(xphoto_DenoisingBm3dTransforms, regression_2D_4x4)
    {
        const int templateWindowSize = 4;
        const int templateWindowSizeSq = templateWindowSize * templateWindowSize;

        uchar src[templateWindowSizeSq];
        short dst[templateWindowSizeSq];

        // Initialize array
        for (uchar i = 0; i < templateWindowSizeSq; ++i)
        {
            src[i] = i;
        }

        HaarTransform2D::ForwardTransform4x4(src, dst, templateWindowSize, templateWindowSize);
        HaarTransform2D::InverseTransform4x4(dst, templateWindowSize);

        for (uchar i = 0; i < templateWindowSizeSq; ++i)
            ASSERT_EQ(static_cast<short>(src[i]), dst[i]);
    }

    TEST(xphoto_DenoisingBm3dTransforms, regression_2D_8x8)
    {
        const int templateWindowSize = 8;
        const int templateWindowSizeSq = templateWindowSize * templateWindowSize;

        uchar src[templateWindowSizeSq];
        short dst[templateWindowSizeSq];

        // Initialize array
        for (uchar i = 0; i < templateWindowSizeSq; ++i)
        {
            src[i] = i;
        }

        HaarTransform2D::ForwardTransform8x8(src, dst, templateWindowSize, templateWindowSize);
        HaarTransform2D::InverseTransform8x8(dst, templateWindowSize);

        for (uchar i = 0; i < templateWindowSizeSq; ++i)
            ASSERT_EQ(static_cast<short>(src[i]), dst[i]);
    }

    template <typename T, typename DT, typename CT>
    static void Test1dTransform(
        T *thrMap,
        int groupSize,
        int templateWindowSizeSq,
        BlockMatch<T, DT, CT> *bm,
        BlockMatch<T, DT, CT> *bmOrig,
        int expectedNonZeroCount = -1)
    {
        if (expectedNonZeroCount < 0)
            expectedNonZeroCount = groupSize * templateWindowSizeSq;

        // Test group size
        short sumNonZero = 0;
        T *thrMapPtr1D = thrMap + (groupSize - 1) * templateWindowSizeSq;
        for (int n = 0; n < templateWindowSizeSq; n++)
        {
            switch (groupSize)
            {
            case 16:
                HaarTransform1D::ForwardTransform16(bm, n);
                sumNonZero += HardThreshold<16>(bm, n, thrMapPtr1D);
                HaarTransform1D::InverseTransform16(bm, n);
                break;
            case 8:
                HaarTransform1D::ForwardTransform8(bm, n);
                sumNonZero += HardThreshold<8>(bm, n, thrMapPtr1D);
                HaarTransform1D::InverseTransform8(bm, n);
                break;
            case 4:
                HaarTransform1D::ForwardTransform4(bm, n);
                sumNonZero += HardThreshold<4>(bm, n, thrMapPtr1D);
                HaarTransform1D::InverseTransform4(bm, n);
                break;
            case 2:
                HaarTransform1D::ForwardTransform2(bm, n);
                sumNonZero += HardThreshold<2>(bm, n, thrMapPtr1D);
                HaarTransform1D::InverseTransform2(bm, n);
                break;
            default:
                HaarTransform1D::ForwardTransformN(bm, n, groupSize);
                sumNonZero += HardThreshold(bm, n, thrMapPtr1D, groupSize);
                HaarTransform1D::InverseTransformN(bm, n, groupSize);
            }
        }

        // Assert transform
        if (expectedNonZeroCount == groupSize * templateWindowSizeSq)
        {
            for (int i = 0; i < groupSize; ++i)
                for (int j = 0; j < templateWindowSizeSq; ++j)
                    ASSERT_EQ(bm[i][j], bmOrig[i][j]);
        }

        // Assert shrinkage
        ASSERT_EQ(sumNonZero, expectedNonZeroCount);
    }

    TEST(xphoto_DenoisingBm3dTransforms, regression_1D_transform)
    {
        const int templateWindowSize = 4;
        const int templateWindowSizeSq = templateWindowSize * templateWindowSize;
        const int searchWindowSize = 16;
        const int searchWindowSizeSq = searchWindowSize * searchWindowSize;
        const float h = 10;
        int maxGroupSize = 64;

        // Precompute separate maps for transform and shrinkage verification
        short *thrMapTransform = NULL;
        short *thrMapShrinkage = NULL;
        HaarTransform<short, short>::calcThresholdMap3D(thrMapTransform, 0, templateWindowSize, maxGroupSize);
        HaarTransform<short, short>::calcThresholdMap3D(thrMapShrinkage, h, templateWindowSize, maxGroupSize);

        // Generate some data
        BlockMatch<short, int, short> *bm = new BlockMatch<short, int, short>[maxGroupSize];
        BlockMatch<short, int, short> *bmOrig = new BlockMatch<short, int, short>[maxGroupSize];
        for (int i = 0; i < maxGroupSize; ++i)
        {
            bm[i].init(templateWindowSizeSq);
            bmOrig[i].init(templateWindowSizeSq);
        }

        for (short i = 0; i < maxGroupSize; ++i)
        {
            for (short j = 0; j < templateWindowSizeSq; ++j)
            {
                bm[i][j] = (j + 1);
                bmOrig[i][j] = bm[i][j];
            }
        }

        // Verify transforms
        Test1dTransform<short, int, short>(thrMapTransform, 2, templateWindowSizeSq, bm, bmOrig);
        Test1dTransform<short, int, short>(thrMapTransform, 4, templateWindowSizeSq, bm, bmOrig);
        Test1dTransform<short, int, short>(thrMapTransform, 8, templateWindowSizeSq, bm, bmOrig);
        Test1dTransform<short, int, short>(thrMapTransform, 16, templateWindowSizeSq, bm, bmOrig);
        Test1dTransform<short, int, short>(thrMapTransform, 32, templateWindowSizeSq, bm, bmOrig);
        Test1dTransform<short, int, short>(thrMapTransform, 64, templateWindowSizeSq, bm, bmOrig);

        // Verify shrinkage
        Test1dTransform<short, int, short>(thrMapShrinkage, 2, templateWindowSizeSq, bm, bmOrig, 6);
        Test1dTransform<short, int, short>(thrMapShrinkage, 4, templateWindowSizeSq, bm, bmOrig, 6);
        Test1dTransform<short, int, short>(thrMapShrinkage, 8, templateWindowSizeSq, bm, bmOrig, 6);
        Test1dTransform<short, int, short>(thrMapShrinkage, 16, templateWindowSizeSq, bm, bmOrig, 6);
        Test1dTransform<short, int, short>(thrMapShrinkage, 32, templateWindowSizeSq, bm, bmOrig, 6);
        Test1dTransform<short, int, short>(thrMapShrinkage, 64, templateWindowSizeSq, bm, bmOrig, 14);
    }

    const float sqrt2 = std::sqrt(2.0f);

    TEST(xphoto_DenoisingBm3dTransforms, regression_1D_generate)
    {
        const int numberOfElements = 8;
        const int arrSize = (numberOfElements << 1) - 1;
        float *thrMap1D = NULL;
        HaarTransform<short, short>::calcThresholdMap1D(thrMap1D, numberOfElements);

        // Expected array
        const float kThrMap1D[arrSize] = {
            1.0f,  // 1 element
            sqrt2 / 2.0f,    sqrt2, // 2 elements
            0.5f,            1.0f,            sqrt2,       sqrt2,  // 4 elements
            sqrt2 / 4.0f,    sqrt2 / 2.0f,    1.0f,        1.0f,  sqrt2, sqrt2, sqrt2, sqrt2  // 8 elements
        };

        for (int j = 0; j < arrSize; ++j)
            ASSERT_EQ(thrMap1D[j], kThrMap1D[j]);

        delete[] thrMap1D;
    }

    TEST(xphoto_DenoisingBm3dTransforms, regression_2D_generate_4x4)
    {
        const int templateWindowSize = 4;
        float *thrMap2D = NULL;
        HaarTransform<short, short>::calcThresholdMap2D(thrMap2D, templateWindowSize);

        // Expected array
        const float kThrMap4x4[templateWindowSize * templateWindowSize] = {
            0.25f,           0.5f,       sqrt2 / 2.0f,    sqrt2 / 2.0f,
            0.5f,            1.0f,       sqrt2,           sqrt2,
            sqrt2 / 2.0f,    sqrt2,      2.0f,            2.0f,
            sqrt2 / 2.0f,    sqrt2,      2.0f,            2.0f
        };

        for (int j = 0; j < templateWindowSize * templateWindowSize; ++j)
            ASSERT_EQ(thrMap2D[j], kThrMap4x4[j]);

        delete[] thrMap2D;
    }

    TEST(xphoto_DenoisingBm3dTransforms, regression_2D_generate_8x8)
    {
        const int templateWindowSize = 8;
        float *thrMap2D = NULL;
        HaarTransform<short, short>::calcThresholdMap2D(thrMap2D, templateWindowSize);

        // Expected array
        const float kThrMap8x8[templateWindowSize * templateWindowSize] = {
            0.125f,       0.25f,        sqrt2 / 4.0f, sqrt2 / 4.0f, 0.5f,  0.5f,  0.5f,  0.5f,
            0.25f,        0.5f,         sqrt2 / 2.0f, sqrt2 / 2.0f, 1.0f,  1.0f,  1.0f,  1.0f,
            sqrt2 / 4.0f, sqrt2 / 2.0f, 1.0f,         1.0f,         sqrt2, sqrt2, sqrt2, sqrt2,
            sqrt2 / 4.0f, sqrt2 / 2.0f, 1.0f,         1.0f,         sqrt2, sqrt2, sqrt2, sqrt2,
            0.5f,         1.0f,         sqrt2,        sqrt2,        2.0f,  2.0f,  2.0f,  2.0f,
            0.5f,         1.0f,         sqrt2,        sqrt2,        2.0f,  2.0f,  2.0f,  2.0f,
            0.5f,         1.0f,         sqrt2,        sqrt2,        2.0f,  2.0f,  2.0f,  2.0f,
            0.5f,         1.0f,         sqrt2,        sqrt2,        2.0f,  2.0f,  2.0f,  2.0f
        };

        for (int j = 0; j < templateWindowSize * templateWindowSize; ++j)
            ASSERT_EQ(thrMap2D[j], kThrMap8x8[j]);

        delete[] thrMap2D;
    }

    TEST(xphoto_Bm3dDenoising, powerOf2)
    {
        ASSERT_EQ(8, getLargestPowerOf2SmallerThan(9));
        ASSERT_EQ(16, getLargestPowerOf2SmallerThan(21));
        ASSERT_EQ(4, getLargestPowerOf2SmallerThan(7));
        ASSERT_EQ(8, getLargestPowerOf2SmallerThan(8));
        ASSERT_EQ(4, getLargestPowerOf2SmallerThan(5));
        ASSERT_EQ(4, getLargestPowerOf2SmallerThan(4));
        ASSERT_EQ(2, getLargestPowerOf2SmallerThan(3));
        ASSERT_EQ(1, getLargestPowerOf2SmallerThan(1));
        ASSERT_EQ(0, getLargestPowerOf2SmallerThan(0));
    }

#endif  // TEST_TRANSFORMS

}

#endif  // OPENCV_ENABLE_NONFREE