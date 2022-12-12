#ifndef SRC_COMMON_CLVACONTEXT_HPP_
#define SRC_COMMON_CLVACONTEXT_HPP_

#include "clglcontext.hpp"
#include "util.hpp"

namespace kb {
class GLWindow;

class CLVAContext {
    friend class GLWindow;
    CLExecContext_t context_;
    CLGLContext &fbContext_;
    cv::UMat frameBuffer_;
    cv::UMat videoFrame_;
    bool hasContext_ = false;
public:
    CLVAContext(CLGLContext &fbContext);
    bool capture(std::function<void(cv::UMat&)> fn);
    void write(std::function<void(const cv::UMat&)> fn);
private:
    bool hasContext();
    void copyContext();
    CLExecContext_t getCLExecContext();
};
} /* namespace kb */

#endif /* SRC_COMMON_CLVACONTEXT_HPP_ */
