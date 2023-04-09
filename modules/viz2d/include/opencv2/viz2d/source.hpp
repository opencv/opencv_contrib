// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_COMMON_SOURCE_HPP_
#define SRC_COMMON_SOURCE_HPP_

#include <functional>
#include <opencv2/opencv.hpp>

namespace cv {
namespace viz {

/*!
 * A Source object represents a way to provide data to Viz2D by using
 * a generator functor.
 */
CV_EXPORTS class Source {
    bool open_ = true;
    std::function<bool(cv::OutputArray&)> generator_;
    cv::UMat frame_;
    uint64_t count_ = 0;
    float fps_;
public:
    /*!
     * Constructs the Source object from a generator functor.
     * @param generator A function object that accepts a reference to a UMat frame
     * that it manipulates. This is ultimatively used to provide video data to #Viz2D
     * @param fps The fps the Source object provides data with.
     */
    Source(std::function<bool(cv::OutputArray&)> generator, float fps);
    /*!
     * Constructs a null Source that is never open or ready.
     */
    Source();
    /*!
     * Default destructor.
     */
    virtual ~Source();
    /*!
     * Signals if the source is ready to provide data.
     * @return true if the source is ready.
     */
    bool isReady();
    /*!
     * Determines if the source is open.
     * @return true if the source is open.
     */
    bool isOpen();
    /*!
     * Returns the fps the underlying generator provides data with.
     * @return The fps of the Source object.
     */
    float fps();
    /*!
     * The source operator. It returns the frame count and the frame generated
     * (e.g. by VideoCapture)in a pair.
     * @return A pair containing the frame count and the frame generated.
     */
    std::pair<uint64_t, cv::InputOutputArray&> operator()();
};

} /* namespace viz2d */
} /* namespace kb */

#endif /* SRC_COMMON_SOURCE_HPP_ */
