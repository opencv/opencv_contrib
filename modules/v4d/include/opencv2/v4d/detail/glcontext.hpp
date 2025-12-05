// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_GLCONTEXT_HPP_
#define SRC_OPENCV_GLCONTEXT_HPP_

#include "opencv2/v4d/detail/framebuffercontext.hpp"
#include "opencv2/v4d/detail/gl.hpp"
struct NVGcontext;
namespace cv {
namespace v4d {
namespace detail {
/*!
 * Used to setup an OpengLG context
 */
class CV_EXPORTS GLContext : public V4DContext {
	const int32_t idx_;
    cv::Ptr<FrameBufferContext> mainFbContext_;
    cv::Ptr<FrameBufferContext> glFbContext_;
public:
     /*!
     * Creates a OpenGL Context
     * @param fbContext The framebuffer context
     */
    GLContext(const int32_t& idx, cv::Ptr<FrameBufferContext> fbContext);
    virtual ~GLContext() {};
    /*!
     * Execute function object fn inside a gl context.
     * The context takes care of setting up opengl states.
     * @param fn A function that is passed the size of the framebuffer
     * and performs drawing using opengl
     */
    virtual void execute(std::function<void()> fn) override;
    const int32_t& getIndex() const;
    cv::Ptr<FrameBufferContext> fbCtx();
};
}
}
}

#endif /* SRC_OPENCV_GLCONTEXT_HPP_ */
