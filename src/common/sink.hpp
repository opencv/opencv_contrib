// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_COMMON_SINK_HPP_
#define SRC_COMMON_SINK_HPP_

#include <functional>
#include <opencv2/opencv.hpp>

namespace cv {
namespace viz {

/*!
 * A Sink object represents a way to write data produced by Viz2D (e.g. a video-file).
 */
class Sink {
    bool open_ = true;
    std::function<bool(const cv::UMat&)> consumer_;
public:
    /*!
     * Consturcts the Sink object from a consumer functor.
     * @param consumer A function object that consumes a UMat frame (e.g. writes it to a video file).
     */
    Sink(std::function<bool(const cv::UMat&)> consumer);
    /*!
     * Constucts a null Sink that is never open or ready
     */
    Sink();
    /*!
     * Default destructor
     */
    virtual ~Sink();
    /*!
     * Signals if the sink is ready to consume data.
     * @return true if the sink is ready.
     */
    bool isReady();
    /*!
     * Determines if the sink is open.
     * @return true if the sink is open.
     */
    bool isOpen();
    /*!
     * The sink operator. It accepts a UMat frame to pass to the consumer
     * @param frame The frame to pass to the consumer. (e.g. VideoWriter)
     */
    void operator()(const cv::UMat& frame);
};

} /* namespace viz2d */
} /* namespace kb */

#endif /* SRC_COMMON_SINK_HPP_ */
