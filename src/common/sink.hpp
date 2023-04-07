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

class Sink {
    bool open_ = true;
    std::function<bool(const cv::UMat&)> consumer_;
public:
    Sink(std::function<bool(const cv::UMat&)> consumer);
    Sink();
    virtual ~Sink();
    bool isReady();
    bool isOpen();
    void operator()(const cv::UMat& frame);
};

} /* namespace viz2d */
} /* namespace kb */

#endif /* SRC_COMMON_SINK_HPP_ */
