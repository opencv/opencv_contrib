// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_XIMGPROC_ACCUMULATE_HPP__
#define __OPENCV_XIMGPROC_ACCUMULATE_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv {
namespace ximgproc {

//! @addtogroup ximgproc
//! @{

/** @brief Adds an image to the accumulator image.

The function adds src or some of its elements to dst :

\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]

The function supports multi-channel images. Each channel is processed independently.

The function cv::ximgproc::accumulate can be used, for example, to collect statistics of a scene
background viewed by a still camera and for the further foreground-background segmentation.

@param src Input image of type CV_8UC(n), CV_16UC(n), CV_32FC(n) or CV_64FC(n), where n is a positive integer.
@param dst %Accumulator image with the same number of channels as input image, and a depth of CV_32F or CV_64F.
@param mask Optional operation mask.

@sa  accumulateSquare, accumulateProduct, accumulateWeighted
 */
CV_EXPORTS_W void accumulate( InputArray src, InputOutputArray dst,
                              InputArray mask = noArray() );

/** @brief Adds the square of a source image to the accumulator image.

The function adds the input image src or its selected region, raised to a power of 2, to the
accumulator dst :

\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]

The function supports multi-channel images. Each channel is processed independently.

@param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
@param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit
floating-point.
@param mask Optional operation mask.

@sa  accumulate, accumulateProduct, accumulateWeighted
 */
CV_EXPORTS_W void accumulateSquare( InputArray src, InputOutputArray dst,
                                    InputArray mask = noArray() );

/** @brief Adds the per-element product of two input images to the accumulator image.

The function adds the product of two images or their selected regions to the accumulator dst :

\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]

The function supports multi-channel images. Each channel is processed independently.

@param src1 First input image, 1- or 3-channel, 8-bit or 32-bit floating point.
@param src2 Second input image of the same type and the same size as src1 .
@param dst %Accumulator image with the same number of channels as input images, 32-bit or 64-bit
floating-point.
@param mask Optional operation mask.

@sa  accumulate, accumulateSquare, accumulateWeighted
 */
CV_EXPORTS_W void accumulateProduct( InputArray src1, InputArray src2,
                                     InputOutputArray dst, InputArray mask=noArray() );

/** @brief Updates a running average.

The function calculates the weighted sum of the input image src and the accumulator dst so that dst
becomes a running average of a frame sequence:

\f[\texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]

That is, alpha regulates the update speed (how fast the accumulator "forgets" about earlier images).
The function supports multi-channel images. Each channel is processed independently.

@param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
@param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit
floating-point.
@param alpha Weight of the input image.
@param mask Optional operation mask.

@sa  accumulate, accumulateSquare, accumulateProduct
 */
CV_EXPORTS_W void accumulateWeighted( InputArray src, InputOutputArray dst,
                                      double alpha, InputArray mask = noArray() );

//! @}

}} // namespace cv::ximgproc

#endif // __OPENCV_XIMGPROC_ACCUMULATE_HPP__
