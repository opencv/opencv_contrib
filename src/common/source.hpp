#ifndef SRC_COMMON_SOURCE_HPP_
#define SRC_COMMON_SOURCE_HPP_

#include <functional>
#include <opencv2/opencv.hpp>

namespace kb {
namespace viz2d {

class Source {
    bool open_ = true;
    std::function<bool(cv::UMat&)> generator_;
    cv::UMat frame_;
    uint64_t count_ = 0;
    float fps_;
public:
    Source(std::function<bool(cv::UMat&)> generator, float fps);
    Source();
    virtual ~Source();
    bool isReady();
    bool isOpen();
    float fps();
    std::pair<uint64_t, cv::UMat&> operator()();
};

} /* namespace viz2d */
} /* namespace kb */

#endif /* SRC_COMMON_SOURCE_HPP_ */
