// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_IMGUIContext_HPP_
#define SRC_OPENCV_IMGUIContext_HPP_

#include "opencv2/v4d/detail/framebuffercontext.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

namespace cv {
namespace v4d {
namespace detail {
CV_EXPORTS class ImGuiContext {
    friend class cv::v4d::V4D;
    FrameBufferContext& mainFbContext_;
    FrameBufferContext glFbContext_;
    std::function<void(const cv::Size&)> renderCallback_;
    bool firstFrame_ = true;
public:
    CV_EXPORTS ImGuiContext(FrameBufferContext& fbContext);
    CV_EXPORTS FrameBufferContext& fbCtx();
    CV_EXPORTS void build(std::function<void(const cv::Size&)> fn);
protected:
    CV_EXPORTS void render();
};
}
}
}

#endif /* SRC_OPENCV_IMGUIContext_HPP_ */
