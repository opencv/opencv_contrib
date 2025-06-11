/*
 * Copyright (c) 2024-2025 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#ifndef OPENCV_WARP_HPP
#define OPENCV_WARP_HPP

#include <opencv2/imgproc.hpp>
namespace cv {
namespace fastcv {

/**
 * @defgroup fastcv Module-wrapper for FastCV hardware accelerated functions
*/

//! @addtogroup fastcv
//! @{

/**
 * @brief   Transform an image using perspective transformation, same as cv::warpPerspective but not bit-exact.
 * @param _src          Input 8-bit image.
 * @param _dst          Output 8-bit image.
 * @param _M0           3x3 perspective transformation matrix.
 * @param dsize         Size of the output image.
 * @param interpolation Interpolation method. Only cv::INTER_NEAREST, cv::INTER_LINEAR and cv::INTER_AREA are supported.
 * @param borderType    Pixel extrapolation method. Only cv::BORDER_CONSTANT, cv::BORDER_REPLICATE and cv::BORDER_TRANSPARENT
 *                      are supported.
 * @param borderValue   Value used in case of a constant border.
 */
CV_EXPORTS_W void warpPerspective(InputArray _src, OutputArray _dst, InputArray _M0, Size dsize, int interpolation, int borderType,
    const Scalar&  borderValue);

/**
 * @brief Perspective warp two images using the same transformation. Bi-linear interpolation is used where applicable.
 *        For example, to warp a grayscale image and an alpha image at the same time, or warp two color channels.
 * @param _src1     First input 8-bit image. Size of buffer is src1Stride*srcHeight bytes.
 * @param _src2     Second input 8-bit image. Size of buffer is src2Stride*srcHeight bytes.
 * @param _dst1     First warped output image (correspond to src1). Size of buffer is dst1Stride*dstHeight bytes, type CV_8UC1
 * @param _dst2     Second warped output image (correspond to src2). Size of buffer is dst2Stride*dstHeight bytes, type CV_8UC1
 * @param _M0       The 3x3 perspective transformation matrix (inversed map)
 * @param dsize     The output image size
*/
CV_EXPORTS_W void warpPerspective2Plane(InputArray _src1, InputArray _src2, OutputArray _dst1, OutputArray _dst2,
    InputArray _M0, Size dsize);

/**
 * @brief Performs an affine transformation on an input image using a provided transformation matrix.
 * 
 * This function performs two types of operations based on the transformation matrix:
 * 
 * 1. Standard Affine Transformation (2x3 matrix):
 *    - Transforms the entire input image using the affine matrix
 *    - Supports both CV_8UC1 and CV_8UC3 types
 * 
 * 2. Patch Extraction with Transformation (2x2 matrix):
 *    - Extracts and transforms a patch from the input image
 *    - Only supports CV_8UC1 type
 *    - If input is a ROI: patch is extracted from ROI center in the original image
 *    - If input is full image: patch is extracted from image center
 * 
 * @param _src              Input image. Supported formats:
 *                          - CV_8UC1: 8-bit single-channel
 *                          - CV_8UC3: 8-bit three-channel - only for 2x3 matrix 
 * @param _dst              Output image. Will have the same type as src and size specified by dsize
 * @param _M                2x2/2x3 affine transformation matrix (inversed map)
 * @param dsize             Output size:
 *                          - For 2x3 matrix: Size of the output image
 *                          - For 2x2 matrix: Size of the extracted patch
 * @param interpolation     Interpolation method. Only applicable for 2x3 transformation with CV_8UC1 input.
 *                          Options:
 *                          - INTER_NEAREST: Nearest-neighbor interpolation
 *                          - INTER_LINEAR: Bilinear interpolation (default)
 *                          - INTER_AREA: Area-based interpolation
 *                          - INTER_CUBIC: Bicubic interpolation
 *                          Note: CV_8UC3 input always use bicubic interpolation internally
 * @param borderValue       Constant pixel value for border pixels. Only applicable for 2x3 transformations 
 *                          with single-channel input.
 *
 * @note                    The affine matrix follows the inverse mapping convention, applied to destination coordinates
 *                          to produce corresponding source coordinates.
 * @note                    The function uses 'FASTCV_BORDER_CONSTANT' for border handling, with the specified 'borderValue'.
*/
CV_EXPORTS_W void warpAffine(InputArray _src, OutputArray _dst, InputArray _M, Size dsize, int interpolation = INTER_LINEAR, 
                            int borderValue = 0);

//! @}

}
}

#endif