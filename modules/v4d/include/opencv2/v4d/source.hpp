// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_SOURCE_HPP_
#define SRC_OPENCV_V4D_SOURCE_HPP_

#include <functional>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/mat.hpp>
#include <mutex>

namespace cv {
namespace v4d {

/*!
 * A Source object represents a way to provide data to V4D by using
 * a generator functor.
 */
class CV_EXPORTS Source {
    bool open_ = true;
    std::function<bool(cv::UMat&)> generator_;
    uint64_t count_ = 0;
    float fps_;
    bool threadSafe_ = false;
    std::mutex mtx_;
public:
    /*!
     * Constructs the Source object from a generator functor.
     * @param generator A function object that accepts a reference to a UMat frame
     * that it manipulates. This is ultimatively used to provide video data to #cv::viz::V4D
     * @param fps The fps the Source object provides data with.
     */
    CV_EXPORTS Source(std::function<bool(cv::UMat&)> generator, float fps);
    /*!
     * Constructs a null Source that is never open or ready.
     */
    CV_EXPORTS Source();
    /*!
     * Default destructor.
     */
    CV_EXPORTS virtual ~Source();
    /*!
     * Signals if the source is ready to provide data.
     * @return true if the source is ready.
     */
    CV_EXPORTS bool isReady();
    CV_EXPORTS bool isThreadSafe();
    CV_EXPORTS void setThreadSafe(bool ts);


    /*!
     * Determines if the source is open.
     * @return true if the source is open.
     */
    CV_EXPORTS bool isOpen();
    /*!
     * Returns the fps the underlying generator provides data with.
     * @return The fps of the Source object.
     */
    CV_EXPORTS float fps();
    /*!
     * The source operator. It returns the frame count and the frame generated
     * (e.g. by VideoCapture)in a pair.
     * @return A pair containing the frame count and the frame generated.
     */
    CV_EXPORTS std::pair<uint64_t, cv::UMat> operator()();
};

} /* namespace v4d */
} /* namespace kb */

#endif /* SRC_OPENCV_V4D_SOURCE_HPP_ */
