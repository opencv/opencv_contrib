#include "sink.hpp"

namespace cv {
namespace viz {

Sink::Sink(std::function<bool(const cv::UMat&)> consumer) : consumer_(consumer) {
}

Sink::Sink() {

}
Sink::~Sink() {
}

bool Sink::isReady() {
    if(consumer_)
        return true;
    else
        return false;
}

bool Sink::isOpen() {
    return open_;
}

void Sink::operator()(const cv::UMat& frame) {
    open_ = consumer_(frame);
}
} /* namespace viz2d */
} /* namespace kb */
