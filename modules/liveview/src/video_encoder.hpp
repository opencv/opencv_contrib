#ifndef OPENCV_LIVEVIEW_VIDEO_ENCODER_HPP
#define OPENCV_LIVEVIEW_VIDEO_ENCODER_HPP

#include "encoded_frame.hpp"

namespace cv {
namespace liveview {

struct VideoEncoderParams
{
    int width = 0;
    int height = 0;
    int fps = 30;
    int bitrate = 2000000;
    int gop = 30;
    VideoCodec preferredCodec = VideoCodec::H264;
    String encoderName;
};

class VideoEncoder
{
public:
    virtual ~VideoEncoder() {}
    virtual void open(const VideoEncoderParams& params) = 0;
    virtual void close() = 0;
    virtual bool isOpened() const = 0;
    virtual void encode(InputArray frame, int64 sequence, int64 ptsUsec) = 0;
    virtual bool tryPop(EncodedFrame& out) = 0;
};

String videoCodecName(VideoCodec codec);
void validateVideoEncoderParams(const VideoEncoderParams& params);
bool haveVideoEncoderBackend();
std::vector<String> availableVideoEncoders(VideoCodec codec);
Ptr<VideoEncoder> createVideoEncoder();

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_VIDEO_ENCODER_HPP
