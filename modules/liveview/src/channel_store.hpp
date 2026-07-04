#ifndef OPENCV_LIVEVIEW_CHANNEL_STORE_HPP
#define OPENCV_LIVEVIEW_CHANNEL_STORE_HPP

#include "opencv2/liveview.hpp"

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace cv {
namespace liveview {

struct ChannelSnapshot
{
    ChannelInfo info;
    std::vector<uchar> jpeg;
};

struct RawFrameSnapshot
{
    ChannelInfo info;
    Mat frame;
};

class ChannelStore
{
public:
    explicit ChannelStore(int jpegQuality = 80);
    ~ChannelStore();

    void publish(const String& name, InputArray frame);
    std::vector<ChannelInfo> channels() const;
    bool getRawFrame(const String& name, RawFrameSnapshot& out) const;
    bool waitForRawFrame(const String& name, int64 afterSequence, int timeoutMs, RawFrameSnapshot& out) const;
    bool getSnapshot(const String& name, ChannelSnapshot& out) const;
    bool waitForJpeg(const String& name, int64 afterSequence, int timeoutMs, ChannelSnapshot& out) const;

private:
    struct ChannelState;

    std::shared_ptr<ChannelState> getOrCreateLocked(const String& name);
    std::shared_ptr<ChannelState> find(const String& name) const;
    void encoderLoop();
    std::shared_ptr<ChannelState> nextDirty();

    int jpegQuality_;
    bool stopping_;
    std::condition_variable dirty_;
    std::thread encoderThread_;
    mutable std::mutex mutex_;
    std::map<String, std::shared_ptr<ChannelState> > channels_;
};

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_CHANNEL_STORE_HPP
