#ifndef OPENCV_FASTCV_BLUR_HPP
#define OPENCV_FASTCV_BLUR_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace fastcv {

/**
 * @defgroup fastcv Module-wrapper for FastCV hardware accelerated functions
 */

//! @addtogroup fastcv
//! @{

/**
 * @brief Gaussian blur with sigma = 0 and square kernel size
 * @param _src Intput image with type CV_8UC1
 * @param _dst Output image with type CV_8UC1
 * @param kernel_size Filer kernel size. One of 3, 5, 11
 * @param blur_border Blur border or not
 *
 * @sa GaussianBlur
 */
CV_EXPORTS_W void gaussianBlur(cv::InputArray _src, cv::OutputArray _dst, int kernel_size = 3, bool blur_border = true);

/**
 * @brief Filter an image with non-separable kernel
 * @param _src Intput image with type CV_8UC1
 * @param _dst Output image with type CV_8UC1, CV_16SC1 or CV_32FC1
 * @param ddepth The depth of output image
 * @param _kernel Filer kernel data
 *
 * @sa Filter2D
 */
CV_EXPORTS_W void filter2D(InputArray _src, OutputArray _dst, int ddepth, InputArray _kernel);

/**
 * @brief sepFilter an image with separable kernel.The way of handling overflow is different with OpenCV, this function will
 * do right shift for the intermediate results and final result.
 * @param _src Intput image with type CV_8UC1
 * @param _dst Output image with type CV_8UC1, CV_16SC1
 * @param ddepth The depth of output image
 * @param _kernelX Filer kernel data in x direction
 * @param _kernelY Filer kernel data in Y direction (For CV_16SC1, the kernelX and kernelY should be same)
 *
 * @sa sepFilter2D
 */
CV_EXPORTS_W void sepFilter2D(InputArray _src, OutputArray _dst, int ddepth, InputArray _kernelX, InputArray _kernelY);
//! @}

} // fastcv::
} // cv::

#endif // OPENCV_FASTCV_BLUR_HPP
