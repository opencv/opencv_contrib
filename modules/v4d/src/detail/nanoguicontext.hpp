// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_NANOGUICONTEXT_HPP_
#define SRC_OPENCV_NANOGUICONTEXT_HPP_

#include "framebuffercontext.hpp"


#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

namespace nanogui {
    class Screen;
}
namespace cv {
namespace v4d {
class FormHelper;
namespace detail {
/*!
 * Used to setup a nanogui context
 */
class NanoguiContext {
protected:
    nanogui::Screen* screen_;
    cv::v4d::FormHelper* form_;
    NVGcontext* context_;
private:
    FrameBufferContext& mainFbContext_;
    cv::TickMeter tick_;
    float fps_ = 0;
    bool first_ = true;
public:
    NanoguiContext(FrameBufferContext& fbContext);
    void render();
    void updateFps(bool print, bool graphical);
    void build(std::function<void(cv::v4d::FormHelper&)> fn);
    nanogui::Screen& screen();
    cv::v4d::FormHelper& form();
};
}
}
}

#endif /* SRC_OPENCV_NANOGUICONTEXT_HPP_ */
