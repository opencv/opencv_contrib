#include "source.hpp"

namespace kb {
namespace viz2d {

Source::Source(std::function<bool(cv::UMat&)> generator, float fps) : generator_(generator), fps_(fps) {
}

Source::Source() : fps_(0) {
}

Source::~Source() {
}

bool Source::isReady() {
    if(generator_)
        return true;
    else
        return false;
}

bool Source::isOpen() {
    return open_;
}

float Source::fps() {
    return fps_;
}

std::pair<uint64_t, cv::UMat&> Source::operator()() {
    open_ = generator_(frame_);
    return {count_++, frame_};
}
} /* namespace viz2d */
} /* namespace kb */
