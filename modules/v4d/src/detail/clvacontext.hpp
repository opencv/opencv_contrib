// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_CLVACONTEXT_HPP_
#define SRC_OPENCV_CLVACONTEXT_HPP_

#include "framebuffercontext.hpp"

namespace cv {
namespace v4d {
class V4D;
namespace detail {

/*!
 * Provides a context for OpenCL-VAAPI sharing
 */
class CLVAContext {
    friend class cv::v4d::V4D;
    CLExecContext_t context_;
    FrameBufferContext& mainFbContext_;
    cv::UMat videoFrame_;
    cv::UMat rgbBuffer_;
    bool hasContext_ = false;
    cv::Size videoFrameSize_;
    CLExecContext_t getCLExecContext();
public:
    /*!
     * Create the CLVAContext
     * @param fbContext The corresponding framebuffer context
     */
    CLVAContext(FrameBufferContext& fbContext);
    /*!
     * Get the current video frame size
     * @return The current video frame size
     */
    cv::Size getVideoFrameSize();
    /*!
     * Called to capture from a function object.
     * The functor fn is passed a UMat which it writes to which in turn is captured to the framebuffer.
     * @param fn The functor that provides the data.
     * @return true if successful-
     */
    bool capture(std::function<void(cv::UMat&)> fn, cv::UMat& framebuffer);
    /*!
     * Called to pass the frambuffer to a functor which consumes it (e.g. writes to a video file).
     * @param fn The functor that consumes the data,
     */
    void write(std::function<void(const cv::UMat&)> fn, const cv::UMat& framebuffer);

    /*FIXME only public till https://github.com/opencv/opencv/pull/22780 is resolved.
     * required for manual initialization of VideoCapture/VideoWriter
     */
    bool hasContext();
    void copyContext();
};
}
}
}

#endif /* SRC_OPENCV_CLVACONTEXT_HPP_ */
