#include "webrtc_session.hpp"

namespace cv {
namespace liveview {

#if !defined(HAVE_LIVEVIEW_WEBRTC_GSTREAMER)
bool haveWebRtcBackend()
{
    return false;
}

Ptr<WebRtcSession> createWebRtcSession()
{
    CV_Error(Error::StsNotImplemented, "LiveView was built without a WebRTC backend");
}
#endif

} // namespace liveview
} // namespace cv
