// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_COMMON_CLVACONTEXT_HPP_
#define SRC_COMMON_CLVACONTEXT_HPP_

#include "framebuffercontext.hpp"

namespace cv {
namespace viz {
class Viz2D;
namespace detail {

class CLVAContext {
    friend class cv::viz::Viz2D;
    CLExecContext_t context_;
    FrameBufferContext& clglContext_;
    cv::UMat frameBuffer_;
    cv::UMat videoFrame_;
    cv::UMat rgbBuffer_;
    bool hasContext_ = false;
    cv::Size videoFrameSize_;
    CLExecContext_t getCLExecContext();
public:
    CLVAContext(FrameBufferContext& fbContext);
    cv::Size getVideoFrameSize();
    bool capture(std::function<void(cv::UMat&)> fn);
    void write(std::function<void(const cv::UMat&)> fn);

    /*FIXME only public till https://github.com/opencv/opencv/pull/22780 is resolved.
     * required for manual initialization of VideoCapture/VideoWriter
     */
    bool hasContext();
    void copyContext();
};
}
}
}

#endif /* SRC_COMMON_CLVACONTEXT_HPP_ */
