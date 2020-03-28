// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_TEXT_SWTTEXTDETECTOR_HPP__
#define __OPENCV_TEXT_SWTTEXTDETECTOR_HPP__

#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace text {
/** @brief Applies the Stroke Width Transform operator followed by filtering of connected components of similar Stroke Widths to return letter candidates. It also chain them by proximity and size, saving the result in chainBBs.
    @param input the input image with 3 channels.
    @param result a vector of resulting bounding boxes where probability of finding text is high
    @param dark_on_light a boolean value signifying whether the text is darker or lighter than the background, it is observed to reverse the gradient obtained from Scharr operator, and significantly affect the result.
    @param draw an optional Mat of type CV_8UC3 which visualises the detected letters using bounding boxes.
    @param chainBBs an optional parameter which chains the letter candidates according to heuristics in the paper and returns all possible regions where text is likely to occur.
    */
    CV_EXPORTS_W void detectTextSWT (InputArray input, CV_OUT std::vector<cv::Rect>& result, bool dark_on_light, OutputArray& draw=noArray(), OutputArray & chainBBs =noArray());
}
}

#endif