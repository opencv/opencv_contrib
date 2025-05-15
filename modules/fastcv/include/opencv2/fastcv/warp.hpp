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
 * @brief Applies an affine transformation to a grayscale image using a 2x3 matrix.
 * @param _src              Input grayscale image of type 'CV_8UC1'.
 * @param _dst              Output image after transformation.
 * @param _M                2x3 affine transformation matrix.
 * @param dsize             Size of the output image.
 * @param interpolation     Interpolation method, Supported methods include:
 *                          'INTER_NEAREST': Nearest neighbor interpolation.
 *                          'INTER_LINEAR': Bilinear interpolation.
 *                          'INTER_AREA': Area interpolation.
 * @param borderValue       Constant pixel value for border pixels.
 *
 * @note                    The affine matrix follows the inverse mapping convention, applied to destination coordinates
 *                          to produce corresponding source coordinates.
 * @note                    The function uses 'FASTCV_BORDER_CONSTANT' for border handling, with the specified 'borderValue'.
*/
CV_EXPORTS_W void warpAffine(InputArray _src, OutputArray _dst, InputArray _M, Size dsize, int interpolation, int borderValue);

/**
 * @brief Applies an affine transformation on a 3-color channel image using a 2x3 matrix with bicubic interpolation.
 *        Pixels that would be sampled from outside the source image are not modified in the target image.
 *        The left-most and right-most pixel coordinates of each scanline are written to dstBorder.
 * @param _src       Input image (3-channel RGB). Size of buffer is src.step[0] * src.rows bytes.
 *                   WARNING: data should be 128-bit aligned.
 * @param _dst       Warped output image (3-channel RGB). Size of buffer is dst.step[0] * dst.rows bytes.
 *                   WARNING: data should be 128-bit aligned.
 * @param _M         2x3 perspective transformation matrix. The matrix stored in affineMatrix is using row major ordering:
 *                   | a11, a12, a13 |
 *                   | a21, a22, a23 |
 *                   The affine matrix follows the inverse mapping convention.
 * @param dsize      The output image size.
 * @param _dstBorder Output array receiving the x-coordinates of left-most and right-most pixels for each scanline.
 *                   The format of the array is: l0,r0,l1,r1,l2,r2,... where l0 is the left-most pixel coordinate in scanline 0
 *                   and r0 is the right-most pixel coordinate in scanline 0.
 *                   The buffer must therefore be 2 * dsize.height integers in size.
 *                   NOTE: data should be 128-bit aligned.
 */
CV_EXPORTS_W void warpAffine3Channel(InputArray _src, OutputArray _dst, InputArray _M, Size dsize, OutputArray _dstBorder);

/**
 * @brief Warps a patch centered at a specified position in the input image using an affine transformation.
 * @param src          Input grayscale image of type 'CV_8UC1'.
 * @param position     Position in the image where the patch is centered.
 * @param affine       2x2 affine transformation matrix of type 'CV_32FC1'.
 * @param patch        Output image patch after transformation.
 * @param patchSize    Size of the output patch.
 *
 * @return Returns 0 if the transformation is valid.
 */
CV_EXPORTS_W void warpAffineROI(InputArray _src, const cv::Point2f& position, InputArray _affine, OutputArray _patch, Size patchSize);

//! @}

}
}

#endif