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
class CV_EXPORTS SourceContext : public V4DContext {
    friend class cv::v4d::V4D;
    CLExecContext_t context_;
    cv::UMat captureBuffer_;
    cv::UMat captureBufferRGB_;
    bool hasContext_ = false;
    cv::Ptr<FrameBufferContext> mainFbContext_;
    uint64_t currentSeqNr_ = 0;
public:
    /*!
     * Create the CLVAContext
     * @param fbContext The corresponding framebuffer context
     */
    SourceContext(cv::Ptr<FrameBufferContext> fbContext);
    virtual ~SourceContext() {};
    /*!
     * Called to capture from a function object.
     * The functor fn is passed a UMat which it writes to which in turn is captured to the framebuffer.
     * @param fn The functor that provides the data.
     * @return true if successful-
     */
    virtual void execute(std::function<void()> fn) override;

    uint64_t sequenceNumber();

    /*FIXME only public till https://github.com/opencv/opencv/pull/22780 is resolved.
     * required for manual initialization of VideoCapture/VideoWriter
     */
    bool hasContext();
    void copyContext();
    CLExecContext_t getCLExecContext();
    cv::UMat& sourceBuffer();
};
}
}
}

#endif /* SRC_OPENCV_CLVACONTEXT_HPP_ */
