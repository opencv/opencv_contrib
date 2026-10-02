#ifndef OPENCV_LIVEVIEW_ENCODED_FRAME_HPP
#define OPENCV_LIVEVIEW_ENCODED_FRAME_HPP

#include "opencv2/liveview.hpp"

namespace cv {
namespace liveview {

enum class VideoCodec
{
    Unknown = 0,
    H264 = 1,
    VP8 = 2
};

struct EncodedFrame
{
    String channel;
    std::vector<uchar> data;
    int64 sequence = 0;
    int64 ptsUsec = 0;
    int64 dtsUsec = 0;
    bool keyframe = false;
    VideoCodec codec = VideoCodec::Unknown;
};

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_ENCODED_FRAME_HPP
