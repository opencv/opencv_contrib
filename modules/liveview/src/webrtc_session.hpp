#ifndef OPENCV_LIVEVIEW_WEBRTC_SESSION_HPP
#define OPENCV_LIVEVIEW_WEBRTC_SESSION_HPP

#include "encoded_frame.hpp"

#include "opencv2/core.hpp"

namespace cv {
namespace liveview {

struct WebRtcOffer
{
    String type;
    String sdp;
};

struct WebRtcAnswer
{
    String type;
    String sdp;
};

struct WebRtcIceCandidate
{
    String candidate;
    int sdpMLineIndex = 0;
};

class WebRtcSession
{
public:
    virtual ~WebRtcSession() {}
    virtual bool open(VideoCodec codec, int fps) = 0;
    virtual bool createAnswer(const WebRtcOffer& offer, WebRtcAnswer& answer) = 0;
    virtual bool addRemoteCandidate(const WebRtcIceCandidate& candidate) = 0;
    virtual bool pushEncoded(const EncodedFrame& frame) = 0;
    virtual void close() = 0;
};

bool haveWebRtcBackend();
Ptr<WebRtcSession> createWebRtcSession();

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_WEBRTC_SESSION_HPP
