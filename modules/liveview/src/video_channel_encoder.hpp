#ifndef OPENCV_LIVEVIEW_VIDEO_CHANNEL_ENCODER_HPP
#define OPENCV_LIVEVIEW_VIDEO_CHANNEL_ENCODER_HPP

#include "channel_store.hpp"
#include "video_encoder.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

namespace cv {
namespace liveview {

class VideoChannelEncoder
{
public:
    VideoChannelEncoder(ChannelStore& store, const String& channel, const VideoEncoderParams& params);
    ~VideoChannelEncoder();

    void start();
    void stop();
    bool isRunning() const;
    bool getLatest(EncodedFrame& out) const;
    bool waitForEncoded(int64 afterSequence, int timeoutMs, EncodedFrame& out) const;

private:
    void workerLoop();
    void publishEncoded(const EncodedFrame& frame);

    ChannelStore& store_;
    String channel_;
    VideoEncoderParams params_;
    bool stopping_;
    bool running_;
    Ptr<VideoEncoder> encoder_;
    std::thread thread_;
    mutable std::mutex mutex_;
    mutable std::condition_variable updated_;
    EncodedFrame latest_;
};

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_VIDEO_CHANNEL_ENCODER_HPP
