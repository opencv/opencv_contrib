// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_MAGNITUDE_HPP_
#define _OPENCV_MAGNITUDE_HPP_
#include "../or_utils/or_types.hpp"
#include "opencv2/core.hpp"

//------------------------------------------------------
//
// Compile time GPU Settings
//
//------------------------------------------------------
#ifndef HFS_BLOCK_DIM
#define HFS_BLOCK_DIM 16
#endif

namespace cv { namespace hfs {

class Magnitude
{
    cv::Ptr<IntImage> delta_x, delta_y, mag;
    cv::Ptr<UCharImage> gray_img, nms_mag;
    Vector2i img_size;

public:
    Magnitude(int height, int width);
    ~Magnitude();

    void loadImage(const cv::Mat& inimg, cv::Ptr<UCharImage> outimg);
    void loadImage(const cv::Ptr<UCharImage> inimg, cv::Mat& outimg);

    void derrivativeXYCpu();
    void nonMaxSuppCpu();

    void derrivativeXYGpu();
    void nonMaxSuppGpu();

    void processImgCpu(const cv::Mat& bgr3u, cv::Mat& mag1u);
    void processImgGpu(const cv::Mat& bgr3u, cv::Mat& mag1u);
};

}}

#endif
