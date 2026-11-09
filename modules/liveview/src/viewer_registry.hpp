#ifndef OPENCV_LIVEVIEW_VIEWER_REGISTRY_HPP
#define OPENCV_LIVEVIEW_VIEWER_REGISTRY_HPP

#include "opencv2/core.hpp"

#include <map>
#include <mutex>

namespace cv {
namespace liveview {

class ViewerRegistry
{
public:
    ViewerRegistry()
        : totalViewers_(0), everConnected_(false),
          lastConnectedTick_(0), lastDisconnectedTick_(0), lastActivityTick_(0)
    {
    }

    void addViewer(const String& channel)
    {
        const int64 tick = getTickCount();
        std::lock_guard<std::mutex> lock(mutex_);
        channelViewers_[channel]++;
        totalViewers_++;
        everConnected_ = true;
        lastConnectedTick_ = tick;
        lastActivityTick_ = tick;
    }

    void removeViewer(const String& channel)
    {
        const int64 tick = getTickCount();
        std::lock_guard<std::mutex> lock(mutex_);
        std::map<String, int>::iterator it = channelViewers_.find(channel);
        if (it != channelViewers_.end())
        {
            if (it->second > 0)
                it->second--;
            if (it->second <= 0)
                channelViewers_.erase(it);
        }
        if (totalViewers_ > 0)
            totalViewers_--;
        lastDisconnectedTick_ = tick;
        lastActivityTick_ = tick;
    }

    void recordActivity()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        lastActivityTick_ = getTickCount();
    }

    int viewerCount(const String& channel = String()) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (channel.empty())
            return totalViewers_;
        std::map<String, int>::const_iterator it = channelViewers_.find(channel);
        return it == channelViewers_.end() ? 0 : it->second;
    }

    bool hasEverConnected() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return everConnected_;
    }

    int64 lastViewerConnectedTick() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return lastConnectedTick_;
    }

    int64 lastViewerDisconnectedTick() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return lastDisconnectedTick_;
    }

    int64 lastViewerActivityTick() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return lastActivityTick_;
    }

private:
    mutable std::mutex mutex_;
    std::map<String, int> channelViewers_;
    int totalViewers_;
    bool everConnected_;
    int64 lastConnectedTick_;
    int64 lastDisconnectedTick_;
    int64 lastActivityTick_;
};

class ScopedViewer
{
public:
    ScopedViewer(ViewerRegistry& registry, const String& channel)
        : registry_(&registry), channel_(channel)
    {
        registry_->addViewer(channel_);
    }

    ~ScopedViewer()
    {
        if (registry_)
            registry_->removeViewer(channel_);
    }

private:
    ScopedViewer(const ScopedViewer&);
    ScopedViewer& operator=(const ScopedViewer&);

    ViewerRegistry* registry_;
    String channel_;
};

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_VIEWER_REGISTRY_HPP
