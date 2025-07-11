// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>

namespace cv
{
namespace fastcv
{

enum ColorConversionCodes {
    // FastCV-specific color conversion codes (avoid collision with OpenCV core)
    COLOR_YUV2YUV444sp_NV12 = 1, //!< FastCV: YCbCr420PseudoPlanar to YCbCr444PseudoPlanar
    COLOR_YUV2YUV422sp_NV12 = 2, //!< FastCV: YCbCr420PseudoPlanar to YCbCr422PseudoPlanar
    COLOR_YUV422sp2YUV444sp = 3, //!< FastCV: YCbCr422PseudoPlanar to YCbCr444PseudoPlanar
    COLOR_YUV422sp2YUV_NV12 = 4, //!< FastCV: YCbCr422PseudoPlanar to YCbCr420PseudoPlanar
    COLOR_YUV444sp2YUV422sp = 5, //!< FastCV: YCbCr444PseudoPlanar to YCbCr422PseudoPlanar
    COLOR_YUV444sp2YUV_NV12 = 6, //!< FastCV: YCbCr444PseudoPlanar to YCbCr420PseudoPlanar
    COLOR_YUV2RGB565_NV12 = 7, //!< FastCV: YCbCr420PseudoPlanar to RGB565
    COLOR_YUV422sp2RGB565 = 8, //!< FastCV: YCbCr422PseudoPlanar to RGB565
    COLOR_YUV422sp2RGB = 9, //!< FastCV: YCbCr422PseudoPlanar to RGB888
    COLOR_YUV422sp2RGBA = 10, //!< FastCV: YCbCr422PseudoPlanar to RGBA8888
    COLOR_YUV444sp2RGB565 = 11, //!< FastCV: YCbCr444PseudoPlanar to RGB565
    COLOR_YUV444sp2RGB = 12, //!< FastCV: YCbCr444PseudoPlanar to RGB888
    COLOR_YUV444sp2RGBA = 13, //!< FastCV: YCbCr444PseudoPlanar to RGBA8888
    COLOR_RGB2YUV_NV12 = 14, //!< FastCV: RGB888 to YCbCr420PseudoPlanar
    COLOR_RGB5652YUV444sp = 15, //!< FastCV: RGB565 to YCbCr444PseudoPlanar
    COLOR_RGB5652YUV422sp = 16, //!< FastCV: RGB565 to YCbCr422PseudoPlanar
    COLOR_RGB5652YUV_NV12 = 17, //!< FastCV: RGB565 to YCbCr420PseudoPlanar
    COLOR_RGB2YUV444sp = 18, //!< FastCV: RGB888 to YCbCr444PseudoPlanar
    COLOR_RGB2YUV422sp = 19, //!< FastCV: RGB888 to YCbCr422PseudoPlanar
};

CV_EXPORTS_W void cvtColor(InputArray src, OutputArray dst, int code);

}}; //cv::fastcv namespace end