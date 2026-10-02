#include "video_channel_encoder.hpp"
#include "route_matcher.hpp"

#include <chrono>

namespace cv {
namespace liveview {

VideoChannelEncoder::VideoChannelEncoder(ChannelStore& store, const String& channel, const VideoEncoderParams& params)
    : store_(store), channel_(channel), params_(params), stopping_(false), running_(false)
{
    if (!isValidChannelName(channel))
        CV_Error(Error::StsBadArg, "Invalid LiveView channel name");
    validateVideoEncoderParams(params);
}

VideoChannelEncoder::~VideoChannelEncoder()
{
    stop();
}

void VideoChannelEncoder::start()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_)
        return;
    encoder_ = createVideoEncoder();
    encoder_->open(params_);
    stopping_ = false;
    running_ = true;
    thread_ = std::thread(&VideoChannelEncoder::workerLoop, this);
}

void VideoChannelEncoder::stop()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ && !thread_.joinable())
            return;
        stopping_ = true;
    }
    updated_.notify_all();
    if (thread_.joinable())
        thread_.join();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (encoder_)
            encoder_->close();
        encoder_.release();
        running_ = false;
        stopping_ = false;
    }
}

bool VideoChannelEncoder::isRunning() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

bool VideoChannelEncoder::getLatest(EncodedFrame& out) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_.data.empty())
        return false;
    out = latest_;
    return true;
}

bool VideoChannelEncoder::waitForEncoded(int64 afterSequence, int timeoutMs, EncodedFrame& out) const
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (latest_.sequence <= afterSequence)
        updated_.wait_for(lock, std::chrono::milliseconds(timeoutMs));
    if (latest_.sequence <= afterSequence || latest_.data.empty())
        return false;
    out = latest_;
    return true;
}

void VideoChannelEncoder::workerLoop()
{
    int64 lastRawSequence = 0;
    for (;;)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_)
                break;
        }

        RawFrameSnapshot raw;
        if (!store_.waitForRawFrame(channel_, lastRawSequence, 50, raw))
            continue;
        lastRawSequence = raw.info.sequence;

        Ptr<VideoEncoder> encoder;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            encoder = encoder_;
        }
        if (encoder.empty())
            continue;

        try
        {
            encoder->encode(raw.frame, raw.info.sequence, raw.info.sequence * 1000000 / params_.fps);
            EncodedFrame encoded;
            while (encoder->tryPop(encoded))
            {
                encoded.channel = channel_;
                publishEncoded(encoded);
            }
        }
        catch (const Exception&)
        {
            continue;
        }
    }
}

void VideoChannelEncoder::publishEncoded(const EncodedFrame& frame)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = frame;
    }
    updated_.notify_all();
}

} // namespace liveview
} // namespace cv
