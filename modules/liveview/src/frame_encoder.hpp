#ifndef OPENCV_LIVEVIEW_FRAME_ENCODER_HPP
#define OPENCV_LIVEVIEW_FRAME_ENCODER_HPP

#include "opencv2/core.hpp"

namespace cv {
namespace liveview {

class FrameEncoder
{
public:
    explicit FrameEncoder(int jpegQuality = 80);

    void validate(InputArray frame) const;
    std::vector<uchar> encodeJpeg(InputArray frame) const;

private:
    int jpegQuality_;
};

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_FRAME_ENCODER_HPP
