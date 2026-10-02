// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_SPARSE_TABLE_MORPHOLOGY_HPP__
#define __OPENCV_SPARSE_TABLE_MORPHOLOGY_HPP__

#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace ximgproc {
namespace stMorph {

//! @addtogroup imgproc_filter
//! @{

/**
* @struct  kernelDecompInfo
* @brief   struct to hold the results of decomposing the structuring element.
*/
struct CV_EXPORTS kernelDecompInfo
{
    //! rows of the original kernel.
    int rows;
    //! cols of the original kernel.
    int cols;
    //!
    //! set of rectangles to covers the kernel which height and width both are power of 2.
    //! point stRects[rd][cd](c,r) means a rectangle left-top (c,r), width 2^rd and height 2^cd.
    //!
    std::vector<std::vector<std::vector<Point>>> stRects;
    //!
    //! Vec2b Mat which sotres the order to calculate sparse table.
    //! The type of returned mat is Vec2b.
    //! * if path[dr][dc][0] == 1 then st[dr+1][dc] will be calculated from st[dr][dc].
    //! * if path[dr][dc][1] == 1 then st[dr][dc+1] will be calculated from st[dr][dc].
    //!
    Mat plan;
    //! anchor position of the kernel.
    Point anchor;
    //! Number of times erosion and dilation are applied.
    int iterations;
};

/**
 * @brief Decompose the structuring element.
 *
 * @param  kernel       structuring element used for subsequent morphological operations.
 * @param  anchor       position of the anchor within the element.
 *                      default value (-1, -1) means that the anchor is at the element center.
 * @param  iterations   number of times  is applied.
 */
CV_EXPORTS kernelDecompInfo decompKernel(InputArray kernel,
                                  Point anchor = Point(-1, -1), int iterations = 1);

/**
 * @brief  Erodes an image with a kernelDecompInfo using spase table method.
 *
 * @param  src          input image
 * @param  dst          output image of the same size and type as src.
 * @param  kdi          pre-computated kernelDecompInfo structure.
 * @param  borderType   pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
 * @param  borderValue  border value in case of a constant border
 */
CV_EXPORTS void erode( InputArray src, OutputArray dst, kernelDecompInfo kdi,
                     BorderTypes borderType = BORDER_CONSTANT,
                     const Scalar& borderValue = morphologyDefaultBorderValue() );

/**
 * @brief  Dilates an image with a kernelDecompInfo using spase table method.
 *
 * @param  src          input image;
 * @param  dst          output image of the same size and type as src.
 * @param  kdi          pre-computated kernelDecompInfo structure.
 * @param  borderType   pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
 * @param  borderValue  border value in case of a constant border
 */
CV_EXPORTS void dilate( InputArray src, OutputArray dst, kernelDecompInfo kdi,
                     BorderTypes borderType = BORDER_CONSTANT,
                     const Scalar& borderValue = morphologyDefaultBorderValue() );

/**
 * @brief  Performs advanced morphological transformations with a kernelDecompInfo.
 *
 * @param  src          input image;
 * @param  dst          output image of the same size and type as src.
 * @param  op           all operations supported by cv::morphologyEx (except cv::MORPH_HITMISS)
 * @param  kdi          pre-computated kernelDecompInfo structure.
 * @param  borderType   pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
 * @param  borderValue  border value in case of a constant border
 */
CV_EXPORTS void morphologyEx( InputArray src, OutputArray dst, int op, kernelDecompInfo kdi,
                                BorderTypes borderType = BORDER_CONSTANT,
                                const Scalar& borderValue = morphologyDefaultBorderValue() );

/**
 * @brief Faster implementation of cv::erode with sparse table concept.
 *
 * @param src input image; the number of channels can be arbitrary, but the depth should be one of
 * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 * @param dst output image of the same size and type as src.
 * @param kernel structuring element used for erosion; if `element=Mat()`, a `3 x 3` rectangular
 * structuring element is used. Kernel can be created using #getStructuringElement.
 * @param anchor position of the anchor within the element; default value (-1, -1) means that the
 * anchor is at the element center.
 * @param iterations number of times erosion is applied.
 * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
 * @param borderValue border value in case of a constant border
 *
 * @see cv::erode
 */
CV_EXPORTS void erode( InputArray src, OutputArray dst, InputArray kernel,
                          Point anchor = Point(-1,-1), int iterations = 1,
                          BorderTypes borderType = BORDER_CONSTANT,
                          const Scalar& borderValue = morphologyDefaultBorderValue() );

/**
 * @brief Faster implementation of cv::dilate with sparse table concept.
 *
 * @param src input image; the number of channels can be arbitrary, but the depth should be one of
 * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 * @param dst output image of the same size and type as src.
 * @param kernel structuring element used for dilation; if element=Mat(), a 3 x 3 rectangular
 * structuring element is used. Kernel can be created using #getStructuringElement
 * @param anchor position of the anchor within the element; default value (-1, -1) means that the
 * anchor is at the element center.
 * @param iterations number of times dilation is applied.
 * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not suported.
 * @param borderValue border value in case of a constant border
 *
 * @see cv::dilate
 */
CV_EXPORTS void dilate( InputArray src, OutputArray dst, InputArray kernel,
                          Point anchor = Point(-1,-1), int iterations = 1,
                          BorderTypes borderType = BORDER_CONSTANT,
                          const Scalar& borderValue = morphologyDefaultBorderValue() );

/**
 * @brief Faster implementation of cv::morphologyEx with sparse table concept.

 * @param src Source image. The number of channels can be arbitrary. The depth should be one of
 * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 * @param dst Destination image of the same size and type as source image.
 * @param op Type of a morphological operation, see #MorphTypes
 * @param kernel Structuring element. It can be created using #getStructuringElement.
 * @param anchor Anchor position with the kernel. Negative values mean that the anchor is at the
 * kernel center.
 * @param iterations Number of times erosion and dilation are applied.
 * @param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
 * @param borderValue Border value in case of a constant border. The default value has a special
 * meaning.
 * @note The number of iterations is the number of times erosion or dilatation operation will be applied.
 * For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply
 * successively: erode -> erode -> dilate -> dilate (and not erode -> dilate -> erode -> dilate).
 *
 * @see cv::morphologyEx
 */
CV_EXPORTS void morphologyEx( InputArray src, OutputArray dst,
                                int op, InputArray kernel,
                                Point anchor = Point(-1,-1), int iterations = 1,
                                BorderTypes borderType = BORDER_CONSTANT,
                                const Scalar& borderValue = morphologyDefaultBorderValue() );
//! @}

}}} // cv::ximgproc::stMorph::
#endif
