#ifndef OPENCV_LIVEVIEW_WEBRTC_MANAGER_HPP
#define OPENCV_LIVEVIEW_WEBRTC_MANAGER_HPP

#include "channel_store.hpp"
#include "video_channel_encoder.hpp"
#include "webrtc_session.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace cv {
namespace liveview {

enum class WebRtcSignalStatus
{
    Ok = 0,
    BadRequest = 1,
    NotFound = 2,
    NotSupported = 3,
    Error = 4
};

struct WebRtcSignalResult
{
    WebRtcSignalStatus status = WebRtcSignalStatus::Error;
    String body;
};

class WebRtcManager
{
public:
    explicit WebRtcManager(ChannelStore& store);
    ~WebRtcManager();

    bool isAvailable() const;
    String viewerHtml(const String& baseUrl, const String& channel) const;
    WebRtcSignalResult createOfferAnswer(const String& channel, const String& requestBody);
    WebRtcSignalResult addCandidate(const String& sessionId, const String& requestBody);
    void stop();

private:
    struct ChannelRuntime;
    struct SessionRuntime;

    std::shared_ptr<ChannelRuntime> getOrCreateChannelLocked(const String& channel, const ChannelInfo& info);
    void sessionPump(const std::shared_ptr<SessionRuntime>& session);

    ChannelStore& store_;
    mutable std::mutex mutex_;
    int nextSessionId_;
    std::atomic<bool> stopping_;
    std::map<String, std::shared_ptr<ChannelRuntime> > channels_;
    std::map<String, std::shared_ptr<SessionRuntime> > sessions_;
};

String extractJsonStringField(const String& json, const String& key);
bool extractJsonIntField(const String& json, const String& key, int& out);

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_WEBRTC_MANAGER_HPP
