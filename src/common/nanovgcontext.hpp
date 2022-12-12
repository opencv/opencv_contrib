#ifndef SRC_COMMON_NANOVGCONTEXT_HPP_
#define SRC_COMMON_NANOVGCONTEXT_HPP_

#define NANOGUI_USE_OPENGL
#include "clglcontext.hpp"
#include <nanogui/nanogui.h>
#include <nanogui/opengl.h>
#include "util.hpp"

namespace kb {
class GLWindow;

class NanoVGContext {
    GLWindow& window_;
    NVGcontext *context_;
    CLGLContext &fbContext_;
public:
    NanoVGContext(GLWindow& window, NVGcontext *context, CLGLContext &fbContext);
    void render(std::function<void(NVGcontext*, const cv::Size&)> fn);
private:
    void begin();
    void end();
};
} /* namespace kb */

#endif /* SRC_COMMON_NANOVGCONTEXT_HPP_ */
