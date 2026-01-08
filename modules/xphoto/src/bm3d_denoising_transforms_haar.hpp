/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective icvers.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef __OPENCV_BM3D_DENOISING_TRANSFORMS_HAAR_HPP__
#define __OPENCV_BM3D_DENOISING_TRANSFORMS_HAAR_HPP__

#include "bm3d_denoising_transforms_1D.hpp"
#include "bm3d_denoising_transforms_2D.hpp"

namespace cv
{
namespace xphoto
{

// Forward declaration
template <typename T, typename TT>
class Transform;

template <typename T, typename TT>
class HaarTransform
{
    static void calcCoefficients1D(cv::Mat &coeff1D, const int &numberOfElements)
    {
        // Generate base array and initialize with zeros
        cv::Mat baseArr = cv::Mat::zeros(numberOfElements, numberOfElements, CV_32FC1);

        // Calculate base array coefficients.
        int currentRow = 0;
        for (int i = numberOfElements; i > 0; i /= 2)
        {
            for (int k = 0, sign = -1; k < numberOfElements; ++k)
            {
                // Alternate sign every i-th element
                if (k % i == 0)
                    sign *= -1;

                // Move to the next row every 2*i-th element
                if (k != 0 && (k % (2 * i) == 0))
                    ++currentRow;

                baseArr.at<float>(currentRow, k) = sign * 1.0f / i;
            }
            ++currentRow;
        }

        // Square each elements of the base array
        float *ptr = baseArr.ptr<float>(0);
        for (unsigned i = 0; i < baseArr.total(); ++i)
            ptr[i] = ptr[i] * ptr[i];

        // Multiply baseArray with 1D vector of ones
        cv::Mat unitaryArr = cv::Mat::ones(numberOfElements, 1, CV_32FC1);
        coeff1D = baseArr * unitaryArr;
    }

    // Method to generate threshold coefficients for 1D transform depending on the number of elements.
    static void fillHaarCoefficients1D(float *thrCoeff1D, int &idx, const int &numberOfElements)
    {
        cv::Mat coeff1D;
        calcCoefficients1D(coeff1D, numberOfElements);

        // Square root the array to get standard deviation
        float *ptr = coeff1D.ptr<float>(0);
        for (unsigned i = 0; i < coeff1D.total(); ++i)
        {
            ptr[i] = std::sqrt(ptr[i]);
            thrCoeff1D[idx++] = ptr[i];
        }
    }

    // Method to generate threshold coefficients for 2D transform depending on the number of elements.
    static void fillHaarCoefficients2D(float *thrCoeff2D, const int &templateWindowSize)
    {
        cv::Mat coeff1D;
        calcCoefficients1D(coeff1D, templateWindowSize);

        // Calculate 2D array
        cv::Mat coeff1Dt;
        cv::transpose(coeff1D, coeff1Dt);
        cv::Mat coeff2D = coeff1D * coeff1Dt;

        // Square root the array to get standard deviation
        float *ptr = coeff2D.ptr<float>(0);
        for (unsigned i = 0; i < coeff2D.total(); ++i)
            thrCoeff2D[i] = std::sqrt(ptr[i]);
    }

public:
    // Method to calculate 1D threshold map based on the maximum number of elements
    // Allocates memory for the output array.
    static void calcThresholdMap1D(float *&thrMap1D, const int &numberOfElements)
    {
        CV_Assert(numberOfElements > 0);

        // Allocate memory for the array
        const int arrSize = (numberOfElements << 1) - 1;
        if (thrMap1D == NULL)
            thrMap1D = new float[arrSize];

        for (int i = 1, idx = 0; i <= numberOfElements; i *= 2)
            fillHaarCoefficients1D(thrMap1D, idx, i);
    }

    // Method to calculate 2D threshold map based on the maximum number of elements
    // Allocates memory for the output array.
    static void calcThresholdMap2D(float *&thrMap2D, const int &templateWindowSize)
    {
        // Allocate memory for the array
        if (thrMap2D == NULL)
            thrMap2D = new float[templateWindowSize * templateWindowSize];

        fillHaarCoefficients2D(thrMap2D, templateWindowSize);
    }

    // Method to calculate 3D threshold map based on the maximum number of elements.
    // Allocates memory for the output array.
    static void calcThresholdMap3D(
        TT *&outThrMap1D,
        const float &hardThr1D,
        const int &templateWindowSize,
        const int &groupSize)
    {
        const int templateWindowSizeSq = templateWindowSize * templateWindowSize;

        // Allocate memory for the output array
        if (outThrMap1D == NULL)
            outThrMap1D = new TT[templateWindowSizeSq * ((groupSize << 1) - 1)];

        // Generate 1D coefficients map
        float *thrMap1D = NULL;
        calcThresholdMap1D(thrMap1D, groupSize);

        // Generate 2D coefficients map
        float *thrMap2D = NULL;
        calcThresholdMap2D(thrMap2D, templateWindowSize);

        // Generate 3D threshold map
        TT *thrMapPtr1D = outThrMap1D;
        for (int i = 1, ii = 0; i <= groupSize; ++ii, i *= 2)
        {
            float coeff = (i == 1) ? 1.0f : std::sqrt(2.0f * std::log((float)i));
            for (int jj = 0; jj < templateWindowSizeSq; ++jj)
            {
                for (int ii1 = 0; ii1 < (1 << ii); ++ii1)
                {
                    int indexIn1D = (1 << ii) - 1 + ii1;
                    int indexIn2D = jj;
                    int thr = static_cast<int>(thrMap1D[indexIn1D] * thrMap2D[indexIn2D] * hardThr1D * coeff);

                    // Set DC component to zero
                    if (jj == 0 && ii1 == 0)
                        thr = 0;

                    *thrMapPtr1D++ = cv::saturate_cast<TT>(thr);
                }
            }
        }

        delete[] thrMap1D;
        delete[] thrMap2D;
    }

    // Method that registers 2D transform calls
    static void RegisterTransforms2D(const int &templateWindowSize)
    {
        // Check if template window size is a power of two
        if (!isPowerOf2(templateWindowSize))
            CV_Error(Error::StsBadArg, "Unsupported template size! Template size must be power of two!");

        switch (templateWindowSize)
        {
        case 2:
            forwardTransform2D = HaarTransform2D::ForwardTransform2x2<T, TT>;
            inverseTransform2D = HaarTransform2D::InverseTransform2x2<TT>;
            break;
        case 4:
            forwardTransform2D = HaarTransform2D::ForwardTransform4x4<T, TT>;
            inverseTransform2D = HaarTransform2D::InverseTransform4x4<TT>;
            break;
        case 8:
            forwardTransform2D = HaarTransform2D::ForwardTransform8x8<T, TT>;
            inverseTransform2D = HaarTransform2D::InverseTransform8x8<TT>;
            break;
        case 16:
            forwardTransform2D = HaarTransform2D::ForwardTransformXxX<T, TT, 16>;
            inverseTransform2D = HaarTransform2D::InverseTransformXxX<TT, 16>;
            break;
        case 32:
            forwardTransform2D = HaarTransform2D::ForwardTransformXxX<T, TT, 32>;
            inverseTransform2D = HaarTransform2D::InverseTransformXxX<TT, 32>;
            break;
        case 64:
            forwardTransform2D = HaarTransform2D::ForwardTransformXxX<T, TT, 64>;
            inverseTransform2D = HaarTransform2D::InverseTransformXxX<TT, 64>;
            break;
        default:
            forwardTransform2D = HaarTransform2D::ForwardTransformXxX<T, TT>;
            inverseTransform2D = HaarTransform2D::InverseTransformXxX<TT>;
        }
    }

    // 2D transform pointers
    static typename Transform<T, TT>::Forward2D forwardTransform2D;
    static typename Transform<T, TT>::Inverse2D inverseTransform2D;

    // 1D transform pointers
    static typename Transform<T, TT>::Forward1D forwardTransformN;
    static typename Transform<T, TT>::Inverse1D inverseTransformN;

    // Specialized 1D forward transform pointers
    static typename Transform<T, TT>::Forward1Ds forwardTransform2;
    static typename Transform<T, TT>::Forward1Ds forwardTransform4;
    static typename Transform<T, TT>::Forward1Ds forwardTransform8;
    static typename Transform<T, TT>::Forward1Ds forwardTransform16;

    // Specialized 1D inverse transform pointers
    static typename Transform<T, TT>::Inverse1Ds inverseTransform2;
    static typename Transform<T, TT>::Inverse1Ds inverseTransform4;
    static typename Transform<T, TT>::Inverse1Ds inverseTransform8;
    static typename Transform<T, TT>::Inverse1Ds inverseTransform16;
};

/// Explicit static members initialization

#define INITIALIZE_HAAR_TRANSFORM(type, member, value)              \
template <typename T, typename TT>                                  \
typename Transform<T, TT>::type HaarTransform<T, TT>::member = value;

// 2D transforms
INITIALIZE_HAAR_TRANSFORM(Forward2D, forwardTransform2D, NULL)
INITIALIZE_HAAR_TRANSFORM(Inverse2D, inverseTransform2D, NULL)

// 1D transforms
INITIALIZE_HAAR_TRANSFORM(Forward1D, forwardTransformN, HaarTransform1D::ForwardTransformN)
INITIALIZE_HAAR_TRANSFORM(Inverse1D, inverseTransformN, HaarTransform1D::InverseTransformN)

// Specialized 1D forward transforms
INITIALIZE_HAAR_TRANSFORM(Forward1Ds, forwardTransform2, HaarTransform1D::ForwardTransform2)
INITIALIZE_HAAR_TRANSFORM(Forward1Ds, forwardTransform4, HaarTransform1D::ForwardTransform4)
INITIALIZE_HAAR_TRANSFORM(Forward1Ds, forwardTransform8, HaarTransform1D::ForwardTransform8)
INITIALIZE_HAAR_TRANSFORM(Forward1Ds, forwardTransform16, HaarTransform1D::ForwardTransform16)

// Specialized 1D inverse transforms
INITIALIZE_HAAR_TRANSFORM(Inverse1Ds, inverseTransform2, HaarTransform1D::InverseTransform2)
INITIALIZE_HAAR_TRANSFORM(Inverse1Ds, inverseTransform4, HaarTransform1D::InverseTransform4)
INITIALIZE_HAAR_TRANSFORM(Inverse1Ds, inverseTransform8, HaarTransform1D::InverseTransform8)
INITIALIZE_HAAR_TRANSFORM(Inverse1Ds, inverseTransform16, HaarTransform1D::InverseTransform16)

}  // namespace xphoto
}  // namespace cv

#endif