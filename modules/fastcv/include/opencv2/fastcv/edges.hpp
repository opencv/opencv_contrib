#ifndef OPENCV_EDGES_HPP
#define OPENCV_EDGES_HPP
#include "opencv2/core/mat.hpp"

namespace cv {
namespace fastcv {
/**
 * @defgroup fastcv Module-wrapper for FastCV hardware accelerated functions
 */

//! @addtogroup fastcv
//! @{

/**
 * @brief Sobel filter with return dx and dy separately
 * @param _src          Input image with type CV_8UC1
 * @param _dx           X direction 1 order derivative with type CV_16SC1.
 * @param _dy           Y direction 1 order derivative with type CV_16SC1 (same size with _dx).
 * @param kernel_size   Sobel kernel size, support 3x3, 5x5, 7x7
 * @param borderType    Border type
 * @param borderValue   Border value for constant border
*/
CV_EXPORTS_W void sobel(cv::InputArray _src, cv::OutputArray _dx, cv::OutputArray _dy, int kernel_size, int borderType,
    int borderValue);

/**
 * @brief 3x3 Sobel filter without border
 * @param _src          Input image with type CV_8UC1
 * @param _dst          If _dsty is not needed, will store 8-bit result of |dx|+|dy|,
 *                      otherwise will store the result of X direction 1 order derivative
 * @param _dsty         If this param is needed, will store the result of Y direction 1 order derivative
 * @param normalization If do normalization for the result
*/
CV_EXPORTS_W void sobel3x3u8(cv::InputArray _src, cv::OutputArray _dst, cv::OutputArray _dsty = noArray(), int ddepth = CV_8U,
    bool normalization = false);

//! @}

}
}

#endif
