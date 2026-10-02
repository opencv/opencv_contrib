// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_SINKCONTEXT_HPP_
#define SRC_OPENCV_SINKCONTEXT_HPP_

#include "framebuffercontext.hpp"

namespace cv {
namespace v4d {
class V4D;
namespace detail {

/*!
 * Provides a context for writing to a Sink
 */
class CV_EXPORTS SinkContext : public V4DContext {
    friend class cv::v4d::V4D;
    CLExecContext_t context_;
    cv::UMat sinkBuffer_;
    bool hasContext_ = false;
    cv::Ptr<FrameBufferContext> mainFbContext_;
public:
    /*!
     * Create the CLVAContext
     * @param fbContext The corresponding framebuffer context
     */
    SinkContext(cv::Ptr<FrameBufferContext> fbContext);
    virtual ~SinkContext() {};
    /*!
     * Called to capture from a function object.
     * The functor fn is passed a UMat which it writes to which in turn is captured to the framebuffer.
     * @param fn The functor that provides the data.
     * @return true if successful-
     */
    virtual void execute(std::function<void()> fn) override;
    /*!
     * Called to pass the frambuffer to a functor which consumes it (e.g. writes to a video file).
     * @param fn The functor that consumes the data,
     */

    /*FIXME only public till https://github.com/opencv/opencv/pull/22780 is resolved.
     * required for manual initialization of VideoCapture/VideoWriter
     */
    bool hasContext();
    void copyContext();
    CLExecContext_t getCLExecContext();
    cv::UMat& sinkBuffer();
};
}
}
}

#endif /* SRC_OPENCV_SINKCONTEXT_HPP_ */
