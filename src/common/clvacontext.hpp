#ifndef SRC_COMMON_CLVACONTEXT_HPP_
#define SRC_COMMON_CLVACONTEXT_HPP_

#include "clglcontext.hpp"
#include "util.hpp"

namespace kb {
class Viz2D;

class CLVAContext {
    friend class Viz2D;
    CLExecContext_t context_;
    CLGLContext &fbContext_;
    cv::UMat frameBuffer_;
    cv::UMat videoFrame_;
    cv::UMat rgbBuffer_;
    bool hasContext_ = false;
    cv::Size videoFrameSize_;
public:
    CLVAContext(CLGLContext &fbContext);
    void setVideoFrameSize(const cv::Size& sz);
    bool capture(std::function<void(cv::UMat&)> fn);
    void write(std::function<void(const cv::UMat&)> fn);
private:
    bool hasContext();
    void copyContext();
    CLExecContext_t getCLExecContext();
};
} /* namespace kb */

#endif /* SRC_COMMON_CLVACONTEXT_HPP_ */
