#ifndef OPENCV_WARP_HPP
#define OPENCV_WARP_HPP
#include "opencv2/core/mat.hpp"
#include <opencv2/imgproc.hpp>
namespace cv {
namespace fastcv {

/**
 * @defgroup fastcv Module-wrapper for FastCV hardware accelerated functions
*/

//! @addtogroup fastcv
//! @{

/**
 * @brief Perspective warp two images using the same transformation. Bi-linear interpolation is used where applicable
 * @param _src1     The first input image data, type CV_8UC1
 * @param _src2     The second input image data, type CV_8UC1
 * @param _dst1     The first output image data, type CV_8UC1
 * @param _dst2     The second output image data, type CV_8UC1
 * @param _M0       The 3x3 perspective transformation matrix (inversed map)
 * @param dsize     The output image size
*/
CV_EXPORTS_W void warpPerspective2Plane(cv::InputArray _src1, cv::InputArray _src2, cv::OutputArray _dst1,
    cv::OutputArray _dst2, InputArray _M0, Size dsize);

//! @}

}
}

#endif