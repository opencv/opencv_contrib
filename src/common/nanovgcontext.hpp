#ifndef SRC_COMMON_NANOVGCONTEXT_HPP_
#define SRC_COMMON_NANOVGCONTEXT_HPP_

#define NANOGUI_USE_OPENGL
#include "clglcontext.hpp"
#include <nanogui/nanogui.h>
#include <nanogui/opengl.h>
#include "util.hpp"

namespace kb {
class NanoVGContext {
    Viz2D& v2d_;
    NVGcontext *context_;
    CLGLContext &clglContext_;
public:
    class Scope {
        NanoVGContext& ctx_;
    public:
        Scope(NanoVGContext& ctx) : ctx_(ctx) {
            ctx_.begin();
        }

        ~Scope() {
            ctx_.end();
        }
    };
    NanoVGContext(Viz2D& v2d, NVGcontext *context, CLGLContext &fbContext);
    void render(std::function<void(NVGcontext*, const cv::Size&)> fn);
private:
    void begin();
    void end();
};
} /* namespace kb */

#endif /* SRC_COMMON_NANOVGCONTEXT_HPP_ */
