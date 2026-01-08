// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#ifndef OPENCV_FASTCV_COLOR_HPP
#define OPENCV_FASTCV_COLOR_HPP

#include <opencv2/core.hpp>

namespace cv
{
namespace fastcv
{

enum ColorConversionCodes {
    // FastCV-specific color conversion codes (avoid collision with OpenCV core)
    COLOR_YUV2YUV444sp_NV12 = 156, //!< FastCV: YCbCr420PseudoPlanar to YCbCr444PseudoPlanar
    COLOR_YUV2YUV422sp_NV12 = 157, //!< FastCV: YCbCr420PseudoPlanar to YCbCr422PseudoPlanar
    COLOR_YUV422sp2YUV444sp = 158, //!< FastCV: YCbCr422PseudoPlanar to YCbCr444PseudoPlanar
    COLOR_YUV422sp2YUV_NV12 = 159, //!< FastCV: YCbCr422PseudoPlanar to YCbCr420PseudoPlanar
    COLOR_YUV444sp2YUV422sp = 160, //!< FastCV: YCbCr444PseudoPlanar to YCbCr422PseudoPlanar
    COLOR_YUV444sp2YUV_NV12 = 161, //!< FastCV: YCbCr444PseudoPlanar to YCbCr420PseudoPlanar
    COLOR_YUV2RGB565_NV12 = 162, //!< FastCV: YCbCr420PseudoPlanar to RGB565
    COLOR_YUV422sp2RGB565 = 163, //!< FastCV: YCbCr422PseudoPlanar to RGB565
    COLOR_YUV422sp2RGB = 164, //!< FastCV: YCbCr422PseudoPlanar to RGB888
    COLOR_YUV422sp2RGBA = 165, //!< FastCV: YCbCr422PseudoPlanar to RGBA8888
    COLOR_YUV444sp2RGB565 = 166, //!< FastCV: YCbCr444PseudoPlanar to RGB565
    COLOR_YUV444sp2RGB = 167, //!< FastCV: YCbCr444PseudoPlanar to RGB888
    COLOR_YUV444sp2RGBA = 168, //!< FastCV: YCbCr444PseudoPlanar to RGBA8888
    COLOR_RGB2YUV_NV12 = 169, //!< FastCV: RGB888 to YCbCr420PseudoPlanar
    COLOR_RGB5652YUV444sp = 170, //!< FastCV: RGB565 to YCbCr444PseudoPlanar
    COLOR_RGB5652YUV422sp = 171, //!< FastCV: RGB565 to YCbCr422PseudoPlanar
    COLOR_RGB5652YUV_NV12 = 172, //!< FastCV: RGB565 to YCbCr420PseudoPlanar
    COLOR_RGB2YUV444sp = 173, //!< FastCV: RGB888 to YCbCr444PseudoPlanar
    COLOR_RGB2YUV422sp = 174, //!< FastCV: RGB888 to YCbCr422PseudoPlanar
};

CV_EXPORTS_W void cvtColor(InputArray src, OutputArray dst, int code);

}}; //cv::fastcv namespace end

#endif // OPENCV_FASTCV_COLOR_HPP
