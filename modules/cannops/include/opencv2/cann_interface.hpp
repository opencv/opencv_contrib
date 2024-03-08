// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNOPS_CANN_INTERFACE_HPP
#define OPENCV_CANNOPS_CANN_INTERFACE_HPP

#include "opencv2/cann.hpp"

namespace cv
{
namespace cann
{

/**
  @addtogroup cannops
  @{
    @defgroup cannops_ops Operations for Ascend Backend.
    @{
        @defgroup cannops_elem Per-element Operations
        @defgroup cannops_core Core Operations on Matrices
        @defgroup cannimgproc Image Processing
    @}
  @}
 */

//! @addtogroup cannops_elem
//! @{

/** @brief Computes a matrix-matrix or matrix-scalar sum.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param dtype Optional depth of the output array.
 * @param stream AscendStream for the asynchronous version.
 * @sa cv::add cuda::add
 */
CV_EXPORTS_W void add(const InputArray src1, const InputArray src2, OutputArray dst,
                      const InputArray mask = noArray(), int dtype = -1,
                      AscendStream& stream = AscendStream::Null());
// This code should not be compiled nor analyzed by doxygen. This interface only for python binding
// code generation. add(InputArray, InputArray ...) can accept Scalar as its parametr.(Scalar -> Mat
// -> InputArray)
#ifdef NEVER_DEFINED
CV_EXPORTS_W void add(const InputArray src1, const Scalar& src2, OutputArray dst,
                      const InputArray mask = noArray(), int dtype = -1,
                      AscendStream& stream = AscendStream::Null());
CV_EXPORTS_W void add(const Scalar& src1, const InputArray src2, OutputArray dst,
                      const InputArray mask = noArray(), int dtype = -1,
                      AscendStream& stream = AscendStream::Null());
#endif
// More overload functions. In order to decouple from the main opencv repository and simplify
// user calling methods, besides the traditional Input/OutputArray parameters, some
// overloaded functions for the AcendMat parameter is also provided.
/** @overload */
CV_EXPORTS_W void add(const AscendMat& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                      const AscendMat& mask = AscendMat(), int dtype = -1,
                      AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void add(const AscendMat& src1, const Scalar& src2, CV_OUT AscendMat& dst,
                      const AscendMat& mask = AscendMat(), int dtype = -1,
                      AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void add(const Scalar& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                      const AscendMat& mask = AscendMat(), int dtype = -1,
                      AscendStream& stream = AscendStream::Null());

/** @brief Computes a matrix-matrix or matrix-scalar difference.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param dtype Optional depth of the output array.
 * @param stream AscendStream for the asynchronous version.
 * @sa cv::subtract cuda::subtract
 */
CV_EXPORTS_W void subtract(const InputArray src1, const InputArray src2, OutputArray dst,
                           const InputArray mask = noArray(), int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void subtract(const InputArray src1, const Scalar& src2, OutputArray dst,
                           const InputArray mask = noArray(), int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
CV_EXPORTS_W void subtract(const Scalar& src1, const InputArray src2, OutputArray dst,
                           const InputArray mask = noArray(), int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
#endif
/** @overload */
CV_EXPORTS_W void subtract(const AscendMat& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                           const AscendMat& mask = AscendMat(), int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void subtract(const AscendMat& src1, const Scalar& src2, CV_OUT AscendMat& dst,
                           const AscendMat& mask = AscendMat(), int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void subtract(const Scalar& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                           const AscendMat& mask = AscendMat(), int dtype = -1,
                           AscendStream& stream = AscendStream::Null());

/** @brief Computes a matrix-matrix or matrix-scalar per-element product.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param scale Optional scale factor.
 * @param dtype Optional depth of the output array.
 * @param stream AscendStream for the asynchronous version.
 * @sa cv::multiply cuda::multiply
 */
CV_EXPORTS_W void multiply(const InputArray src1, const InputArray src2, OutputArray dst,
                           float scale = 1, int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void multiply(const InputArray src1, const Scalar& src2, OutputArray dst,
                           float scale = 1, int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
CV_EXPORTS_W void multiply(const Scalar& src1, const InputArray src2, OutputArray dst,
                           float scale = 1, int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
#endif
/** @overload */
CV_EXPORTS_W void multiply(const AscendMat& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                           float scale = 1, int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void multiply(const AscendMat& src1, const Scalar& src2, CV_OUT AscendMat& dst,
                           float scale = 1, int dtype = -1,
                           AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void multiply(const Scalar& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                           float scale = 1, int dtype = -1,
                           AscendStream& stream = AscendStream::Null());

/** @brief Computes a matrix-matrix or matrix-scalar division.
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar. Matrix should have the same size and type as src1 .
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param scale Optional scale factor.
 * @param dtype Optional depth of the output array.
 * @param stream AscendStream for the asynchronous version.
 * @sa cv::divide cuda::divide
 */
CV_EXPORTS_W void divide(const InputArray src1, const InputArray src2, OutputArray dst,
                         float scale = 1, int dtype = -1,
                         AscendStream& stream = AscendStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void divide(const InputArray src1, const Scalar& src2, OutputArray dst,
                         float scale = 1, int dtype = -1,
                         AscendStream& stream = AscendStream::Null());
CV_EXPORTS_W void divide(const Scalar& src1, const InputArray src2, OutputArray dst,
                         float scale = 1, int dtype = -1,
                         AscendStream& stream = AscendStream::Null());
#endif
CV_EXPORTS_W void divide(const AscendMat& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                         float scale = 1, int dtype = -1,
                         AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void divide(const AscendMat& src1, const Scalar& src2, CV_OUT AscendMat& dst,
                         float scale = 1, int dtype = -1,
                         AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void divide(const Scalar& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                         float scale = 1, int dtype = -1,
                         AscendStream& stream = AscendStream::Null());

/** @brief Performs a per-element bitwise conjunction of two matrices (or of matrix and scalar).
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AscendStream for the asynchronous version.
 * @sa cv::bitwise_and cuda::bitwise_and
 */
CV_EXPORTS_W void bitwise_and(const InputArray src1, const InputArray src2, OutputArray dst,
                              const InputArray mask = noArray(),
                              AscendStream& stream = AscendStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void bitwise_and(const InputArray src1, const Scalar& src2, OutputArray dst,
                              const InputArray mask = noArray(),
                              AscendStream& stream = AscendStream::Null());
CV_EXPORTS_W void bitwise_and(const Scalar& src1, const InputArray src2, OutputArray dst,
                              const InputArray mask = noArray(),
                              AscendStream& stream = AscendStream::Null());
#endif
CV_EXPORTS_W void bitwise_and(const AscendMat& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                              const AscendMat& mask = AscendMat(),
                              AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void bitwise_and(const AscendMat& src1, const Scalar& src2, CV_OUT AscendMat& dst,
                              const AscendMat& mask = AscendMat(),
                              AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void bitwise_and(const Scalar& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                              const AscendMat& mask = AscendMat(),
                              AscendStream& stream = AscendStream::Null());

/** @brief Performs a per-element bitwise disjunction of two matrices (or of matrix and scalar).
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AscendStream for the asynchronous version.
 * @sa cv::bitwise_or cuda::bitwise_or
 */
CV_EXPORTS_W void bitwise_or(const InputArray src1, const InputArray src2, OutputArray dst,
                             const InputArray mask = noArray(),
                             AscendStream& stream = AscendStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void bitwise_or(const InputArray src1, const Scalar& src2, OutputArray dst,
                             const InputArray mask = noArray(),
                             AscendStream& stream = AscendStream::Null());
CV_EXPORTS_W void bitwise_or(const Scalar& src1, const InputArray src2, OutputArray dst,
                             const InputArray mask = noArray(),
                             AscendStream& stream = AscendStream::Null());
#endif
CV_EXPORTS_W void bitwise_or(const AscendMat& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                             const AscendMat& mask = AscendMat(),
                             AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void bitwise_or(const AscendMat& src1, const Scalar& src2, CV_OUT AscendMat& dst,
                             const AscendMat& mask = AscendMat(),
                             AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void bitwise_or(const Scalar& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                             const AscendMat& mask = AscendMat(),
                             AscendStream& stream = AscendStream::Null());

/** @brief Performs a per-element bitwise exclusive or operation of two matrices (or of matrix and
 * scalar).
 * @param src1 First source matrix or scalar.
 * @param src2 Second source matrix or scalar.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AscendStream for the asynchronous version.
 * @sa cv::bitwise_xor cuda::bitwise_xor
 */
CV_EXPORTS_W void bitwise_xor(const InputArray src1, const InputArray src2, OutputArray dst,
                              const InputArray mask = noArray(),
                              AscendStream& stream = AscendStream::Null());
#ifdef NEVER_DEFINED
CV_EXPORTS_W void bitwise_xor(const InputArray src1, const Scalar& src2, OutputArray dst,
                              const InputArray mask = noArray(),
                              AscendStream& stream = AscendStream::Null());
CV_EXPORTS_W void bitwise_xor(const Scalar& src1, const InputArray src2, OutputArray dst,
                              const InputArray mask = noArray(),
                              AscendStream& stream = AscendStream::Null());
#endif
CV_EXPORTS_W void bitwise_xor(const AscendMat& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                              const AscendMat& mask = AscendMat(),
                              AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void bitwise_xor(const AscendMat& src1, const Scalar& src2, CV_OUT AscendMat& dst,
                              const AscendMat& mask = AscendMat(),
                              AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void bitwise_xor(const Scalar& src1, const AscendMat& src2, CV_OUT AscendMat& dst,
                              const AscendMat& mask = AscendMat(),
                              AscendStream& stream = AscendStream::Null());

/** @brief Performs a per-element bitwise inversion.
 * @param src First source matrix.
 * @param dst Destination matrix that has the same size and number of channels as the input
 * array(s). The depth is defined by dtype or src1 depth.
 * @param mask Optional operation mask, 8-bit single channel array, that specifies elements of the
 * destination array to be changed. The mask can be used only with single channel images.
 * @param stream AscendStream for the asynchronous version.
 * @sa cv::bitwise_not cuda::bitwise_not
 */
CV_EXPORTS_W void bitwise_not(const InputArray src, OutputArray dst,
                              const InputArray mask = noArray(),
                              AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void bitwise_not(const AscendMat& src, CV_OUT AscendMat& dst,
                              const AscendMat& mask = AscendMat(),
                              AscendStream& stream = AscendStream::Null());

/** @brief Computes the weighted sum of two arrays.

@param src1 First source array.
@param alpha Weight for the first array elements.
@param src2 Second source array of the same size and channel number as src1 .
@param beta Weight for the second array elements.
@param dst Destination array that has the same size and number of channels as the input arrays.
@param gamma Scalar added to each sum.
@param dtype Optional depth of the destination array. When both input arrays have the same depth,
dtype can be set to -1, which will be equivalent to src1.depth().
@param stream Stream for the asynchronous version.

The function addWeighted calculates the weighted sum of two arrays as follows:

\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)*
\texttt{beta} +  \texttt{gamma} )\f]

where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.

@sa cv::addWeighted cv::cuda::addWeighted
 */
CV_EXPORTS_W void addWeighted(const InputArray src1, double alpha, const InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype = -1,
                              AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void addWeighted(const AscendMat& src1, double alpha, const AscendMat& src2,
                              double beta, double gamma, CV_OUT AscendMat& dst, int dtype = -1,
                              AscendStream& stream = AscendStream::Null());

/** @brief Applies a fixed-level threshold to each array element.

@param src Source array (single-channel).
@param dst Destination array with the same size and type as src .
@param thresh Threshold value.
@param maxval Maximum value to use with THRESH_BINARY and THRESH_BINARY_INV threshold types.
@param type Threshold type. For details, see threshold . The THRESH_MASK, THRESH_OTSU and
THRESH_TRIANGLE threshold types are not supported.
@param stream AscendStream for the asynchronous version.

@sa cv::threshold cv::cuda::threshold
*/
CV_EXPORTS_W double threshold(const InputArray src, OutputArray dst, double thresh, double maxval,
                              int type, AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W double threshold(const AscendMat& src, CV_OUT AscendMat& dst, double thresh,
                              double maxval, int type, AscendStream& stream = AscendStream::Null());

//! @} cannops_elem

//! @addtogroup cannops_core
//! @{

/** @brief Makes a multi-channel matrix out of several single-channel matrices.

@param src Array/vector of source matrices.
@param n Number of source matrices.
@param dst Destination matrix.
@param stream AscendStream for the asynchronous version.

@sa cv::merge cv::cuda::merge
 */
CV_EXPORTS_W void merge(const AscendMat* src, size_t n, CV_OUT AscendMat& dst,
                      AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void merge(const std::vector<AscendMat>& src, CV_OUT AscendMat& dst,
                        AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void merge(const AscendMat* src, size_t n, OutputArray& dst,
                        AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void merge(const std::vector<AscendMat>& src, OutputArray& dst,
                        AscendStream& stream = AscendStream::Null());

/** @brief Copies each plane of a multi-channel matrix into an array.

@param src Source matrix.
@param dst Destination array/vector of single-channel matrices.
@param stream AscendStream for the asynchronous version.

@sa cv::split cv::cuda::split
 */
CV_EXPORTS_W void split(const AscendMat& src, AscendMat* dst,
                      AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void split(const AscendMat& src, CV_OUT std::vector<AscendMat>& dst,
                        AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void split(const InputArray src, AscendMat* dst,
                        AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void split(const InputArray src, CV_OUT std::vector<AscendMat>& dst,
                        AscendStream& stream = AscendStream::Null());

/** @brief Transposes a matrix.

@param src Source matrix.
@param dst Destination matrix.
@param stream AscendStream for the asynchronous version.

@sa cv::transpose cv::cuda::transpose
 */
CV_EXPORTS_W void transpose(InputArray src, OutputArray dst,
                            AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void transpose(const AscendMat& src, CV_OUT AscendMat& dst,
                            AscendStream& stream = AscendStream::Null());
/** @brief Flips a 2D matrix around vertical, horizontal, or both axes.

@param src Source matrix.
@param dst Destination matrix.
@param flipCode Flip mode for the source:
-   0 Flips around x-axis.
-   \> 0 Flips around y-axis.
-   \< 0 Flips around both axes.
@param stream AscendStream for the asynchronous version.

@sa cv::flip cv::cuda::flip
 */
CV_EXPORTS_W void flip(InputArray src, OutputArray dst, int flipCode,
                       AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void flip(const AscendMat& src, CV_OUT AscendMat& dst, int flipCode,
                       AscendStream& stream = AscendStream::Null());
/** @brief Rotates a 2D array in multiples of 90 degrees.
The function cv::rotate rotates the array in one of three different ways:
*   Rotate by 90 degrees clockwise (rotateCode = ROTATE_90_CLOCKWISE).
*   Rotate by 180 degrees clockwise (rotateCode = ROTATE_180).
*   Rotate by 270 degrees clockwise (rotateCode = ROTATE_90_COUNTERCLOCKWISE).
@param src input array.
@param dst output array of the same type as src.  The size is the same with ROTATE_180,
and the rows and cols are switched for ROTATE_90_CLOCKWISE and ROTATE_90_COUNTERCLOCKWISE.
@param rotateCode an enum to specify how to rotate the array; see the enum #RotateFlags
@param stream AscendStream for the asynchronous version.

@sa cv::rotate
*/
CV_EXPORTS_W void rotate(InputArray src, OutputArray dst, int rotateCode,
                         AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void rotate(const AscendMat& src, CV_OUT AscendMat& dst, int rotateMode,
                         AscendStream& stream = AscendStream::Null());

/** @brief crop a 2D array.
The function crops the matrix by given cv::Rect.
Output matrix must be of the same depth as input one, size is specified by given rect size.

@param src input array.
@param rect a rect to crop a array to
@param stream AscendStream for the asynchronous version.

@sa cv::gapi::crop
*/
CV_EXPORTS_W AscendMat crop(InputArray src, const Rect& rect,
                            AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W AscendMat crop(const AscendMat& src, const Rect& rect,
                            AscendStream& stream = AscendStream::Null());
/** @brief Resizes an image src down to or up to the specified size.
@param src    input image
@param dst    output image; it has the size dsize (when it is non-zero) or the size computed from
src.size(), fx, and fy; the type of dst is the same as of src.
@param dsize  output image size; if it equals zero, it is computed as:
     \f[𝚍𝚜𝚒𝚣𝚎 = 𝚂𝚒𝚣𝚎(𝚛𝚘𝚞𝚗𝚍(𝚏𝚡*𝚜𝚛𝚌.𝚌𝚘𝚕𝚜), 𝚛𝚘𝚞𝚗𝚍(𝚏𝚢*𝚜𝚛𝚌.𝚛𝚘𝚠𝚜))\f]
     Either dsize or both fx and fy must be non-zero.
@param fx     scale factor along the horizontal axis; when it equals 0, it is computed as
\f[(𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚠𝚒𝚍𝚝𝚑/𝚜𝚛𝚌.𝚌𝚘𝚕𝚜\f]

@param fy     scale factor along the vertical axis; when it equals 0, it is computed as
\f[(𝚍𝚘𝚞𝚋𝚕𝚎)𝚍𝚜𝚒𝚣𝚎.𝚑𝚎𝚒𝚐𝚑𝚝/𝚜𝚛𝚌.𝚛𝚘𝚠𝚜\f]
@param interpolation    interpolation method(see **cv.cann.InterpolationFlags**)
@sa cv::resize
*/

//! interpolation algorithm
enum InterpolationFlags
{
    /** nearest neighbor interpolation */
    INTER_NEAREST = 0,
    /** bilinear interpolation */
    INTER_LINEAR = 1,
    /** bicubic interpolation */
    INTER_CUBIC = 2,
    /** resampling using pixel area relation. It may be a preferred method for image decimation, as
    it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST
    method. */
    INTER_AREA = 3,
    /** mask for interpolation codes */
    INTER_MAX = 7,
};

CV_EXPORTS_W void resize(InputArray _src, OutputArray _dst, Size dsize, double inv_scale_x,
                         double inv_scale_y, int interpolation,
                         AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void resize(const AscendMat& src, CV_OUT AscendMat& dst, Size dsize, double inv_scale_x,
                         double inv_scale_y, int interpolation,
                         AscendStream& stream = AscendStream::Null());

//! @} cannops_core

//! @addtogroup cannimgproc
//! @{

/** @brief Converts an image from one color space to another.

@param src Source image with CV_8U , CV_16U , or CV_32F depth and 1, 3, or 4 channels.
@param dst Destination image.
@param code Color space conversion code. For details, see cvtColor .
@param dstCn Number of channels in the destination image. If the parameter is 0, the number of the
channels is derived automatically from src and the code .
@param stream AscendStream for the asynchronous version.

@sa cv::cvtColor cv::cuda::cvtColor
 */
CV_EXPORTS_W void cvtColor(const InputArray src, OutputArray dst, int code, int dstCn = 0,
                           AscendStream& stream = AscendStream::Null());
/** @overload */
CV_EXPORTS_W void cvtColor(const AscendMat& src, CV_OUT AscendMat& dst, int code, int dstCn = 0,
                           AscendStream& stream = AscendStream::Null());

//! @} cannimgproc

} // namespace cann
} // namespace cv

#endif // OPENCV_CANNOPS_CANN_INTERFACE_HPP
