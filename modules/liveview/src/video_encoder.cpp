#include "video_encoder.hpp"

namespace cv {
namespace liveview {

String videoCodecName(VideoCodec codec)
{
    if (codec == VideoCodec::H264)
        return "H264";
    if (codec == VideoCodec::VP8)
        return "VP8";
    return "Unknown";
}

void validateVideoEncoderParams(const VideoEncoderParams& params)
{
    if (params.width <= 0 || params.height <= 0)
        CV_Error(Error::StsBadArg, "LiveView video encoder dimensions must be positive");
    if (params.fps <= 0)
        CV_Error(Error::StsBadArg, "LiveView video encoder FPS must be positive");
    if (params.bitrate <= 0)
        CV_Error(Error::StsBadArg, "LiveView video encoder bitrate must be positive");
    if (params.gop <= 0)
        CV_Error(Error::StsBadArg, "LiveView video encoder GOP must be positive");
    if (params.preferredCodec != VideoCodec::H264 && params.preferredCodec != VideoCodec::VP8)
        CV_Error(Error::StsBadArg, "LiveView video encoder supports H.264 and VP8");
}

#if !defined(HAVE_LIVEVIEW_VIDEO_ENCODER_FFMPEG)
bool haveVideoEncoderBackend()
{
    return false;
}

std::vector<String> availableVideoEncoders(VideoCodec)
{
    return std::vector<String>();
}

Ptr<VideoEncoder> createVideoEncoder()
{
    CV_Error(Error::StsNotImplemented, "LiveView was built without a video encoder backend");
}
#endif

} // namespace liveview
} // namespace cv
