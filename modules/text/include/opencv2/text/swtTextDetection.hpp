// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef SWT_H
#define SWT_H

#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace text {
/** @brief Applies the Stroke Width Transform operator followed by filtering of connected components of similar Stroke Widths to find letter candidates and chain them by proximity and size.
    @param input_image the input image with 3 channels.
    @param dark_on_light a boolean value signifying whether the text is darker or lighter than the background, it is observed to reverse the gradient obtained from Scharr operator, and significantly affect the result. 
    */
    std::vector<cv::Rect> SWTTextDetection (const Mat& input_image, bool dark_on_light);
}    
}

#endif 
