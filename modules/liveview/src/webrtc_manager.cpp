#include "webrtc_manager.hpp"

#include "route_matcher.hpp"

#include <chrono>
#include <cstdlib>
#include <sstream>

namespace cv {
namespace liveview {
namespace {

static String statusJson(const String& status, const String& detail)
{
    return "{\"status\":\"" + jsonEscape(status) + "\",\"detail\":\"" + jsonEscape(detail) + "\"}";
}

static String answerJson(const String& sessionId, const WebRtcAnswer& answer)
{
    return "{\"session\":\"" + jsonEscape(sessionId) + "\",\"type\":\"" + jsonEscape(answer.type) +
           "\",\"sdp\":\"" + jsonEscape(answer.sdp) + "\"}";
}

static VideoEncoderParams paramsForChannel(const ChannelInfo& info)
{
    VideoEncoderParams params;
    params.width = info.width;
    params.height = info.height;
    params.fps = 30;
    params.bitrate = 2000000;
    params.gop = 30;
    params.preferredCodec = VideoCodec::VP8;
    return params;
}

static bool containsVideoMediaSection(const String& sdp)
{
    if (sdp.find("m=video ") == 0)
        return true;
    return sdp.find("\nm=video ") != String::npos || sdp.find("\rm=video ") != String::npos;
}

} // namespace

struct WebRtcManager::ChannelRuntime
{
    explicit ChannelRuntime(ChannelStore& store, const String& channel, const VideoEncoderParams& params)
        : encoder(store, channel, params), codec(params.preferredCodec), refs(0)
    {
    }

    VideoChannelEncoder encoder;
    VideoCodec codec;
    int refs;
};

struct WebRtcManager::SessionRuntime
{
    String id;
    String channel;
    std::shared_ptr<ChannelRuntime> channelRuntime;
    Ptr<WebRtcSession> session;
    std::unique_ptr<ScopedViewer> viewer;
    std::atomic<bool> stopping;
    std::thread thread;

    SessionRuntime()
        : stopping(false)
    {
    }
};

String extractJsonStringField(const String& json, const String& key)
{
    const String pattern = "\"" + key + "\"";
    size_t p = json.find(pattern);
    if (p == String::npos)
        return String();
    p = json.find(':', p + pattern.size());
    if (p == String::npos)
        return String();
    p = json.find('"', p + 1);
    if (p == String::npos)
        return String();

    String out;
    bool escaped = false;
    for (size_t i = p + 1; i < json.size(); ++i)
    {
        const char c = json[i];
        if (escaped)
        {
            switch (c)
            {
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            default: out += c; break;
            }
            escaped = false;
            continue;
        }
        if (c == '\\')
        {
            escaped = true;
            continue;
        }
        if (c == '"')
            return out;
        out += c;
    }
    return String();
}

bool extractJsonIntField(const String& json, const String& key, int& out)
{
    const String pattern = "\"" + key + "\"";
    size_t p = json.find(pattern);
    if (p == String::npos)
        return false;
    p = json.find(':', p + pattern.size());
    if (p == String::npos)
        return false;
    ++p;
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t'))
        ++p;
    char* end = NULL;
    const long value = std::strtol(json.c_str() + p, &end, 10);
    if (end == json.c_str() + p)
        return false;
    out = static_cast<int>(value);
    return true;
}

WebRtcManager::WebRtcManager(ChannelStore& store, ViewerRegistry& viewers)
    : store_(store), viewers_(viewers), nextSessionId_(1), stopping_(false)
{
}

WebRtcManager::~WebRtcManager()
{
    stop();
}

bool WebRtcManager::isAvailable() const
{
    return haveWebRtcBackend();
}

String WebRtcManager::viewerHtml(const String& baseUrl, const String& channel) const
{
    std::ostringstream os;
    os << "<!doctype html><html><head><meta charset='utf-8'>";
    os << "<title>OpenCV LiveView WebRTC</title>";
    os << "<style>body{font-family:sans-serif;margin:24px;background:#111;color:#eee}";
    os << "video{max-width:100%;background:#000}code{background:#222;padding:2px 4px}</style>";
    os << "</head><body><h1>OpenCV LiveView</h1>";
    os << "<p><code>" << htmlEscape(channel) << "</code></p>";
    os << "<video id='v' autoplay playsinline muted controls></video>";
    os << "<p id='s'>connecting</p>";
    os << "<script>";
    os << "const channel=" << "\"" << jsonEscape(channel) << "\";";
    os << "let sessionId=null;";
    os << "const status=document.getElementById('s');";
    os << "const pc=new RTCPeerConnection({iceServers:[]});";
    os << "pc.addTransceiver('video',{direction:'recvonly'});";
    os << "pc.ontrack=e=>{document.getElementById('v').srcObject=e.streams[0];status.textContent='connected';};";
    os << "pc.onconnectionstatechange=()=>{status.textContent=pc.connectionState;};";
    os << "async function waitIce(){if(pc.iceGatheringState==='complete')return;";
    os << "await new Promise(r=>{pc.onicegatheringstatechange=()=>{";
    os << "if(pc.iceGatheringState==='complete')r();};setTimeout(r,1500);});}";
    os << "(async()=>{const offer=await pc.createOffer();await pc.setLocalDescription(offer);await waitIce();";
    os << "const res=await fetch('/webrtc/'+channel+'/offer',{method:'POST',";
    os << "headers:{'Content-Type':'application/json'},";
    os << "body:JSON.stringify({type:pc.localDescription.type,sdp:pc.localDescription.sdp})});";
    os << "if(!res.ok){status.textContent='signaling failed';return;}";
    os << "const ans=await res.json();sessionId=ans.session;";
    os << "await pc.setRemoteDescription({type:ans.type,sdp:ans.sdp});})();";
    os << "function closeSession(){if(!sessionId)return;";
    os << "const url='/webrtc/session/'+sessionId+'/close';";
    os << "if(navigator.sendBeacon)navigator.sendBeacon(url,'');";
    os << "else fetch(url,{method:'POST',keepalive:true}).catch(()=>{});";
    os << "sessionId=null;}";
    os << "window.addEventListener('pagehide',closeSession);";
    os << "window.addEventListener('beforeunload',closeSession);";
    os << "</script></body></html>";
    (void)baseUrl;
    return os.str();
}

std::shared_ptr<WebRtcManager::ChannelRuntime>
WebRtcManager::getOrCreateChannelLocked(const String& channel, const ChannelInfo& info)
{
    std::map<String, std::shared_ptr<ChannelRuntime> >::iterator it = channels_.find(channel);
    if (it != channels_.end())
        return it->second;

    std::shared_ptr<ChannelRuntime> runtime(new ChannelRuntime(store_, channel, paramsForChannel(info)));
    runtime->encoder.start();
    channels_[channel] = runtime;
    return runtime;
}

WebRtcSignalResult WebRtcManager::createOfferAnswer(const String& channel, const String& requestBody)
{
    WebRtcSignalResult result;
    if (!isValidChannelName(channel))
    {
        result.status = WebRtcSignalStatus::BadRequest;
        result.body = statusJson("error", "invalid channel");
        return result;
    }
    if (!haveWebRtcBackend())
    {
        result.status = WebRtcSignalStatus::NotSupported;
        result.body = statusJson("error", "WebRTC backend is not available");
        return result;
    }

    RawFrameSnapshot raw;
    if (!store_.getRawFrame(channel, raw))
    {
        result.status = WebRtcSignalStatus::NotFound;
        result.body = statusJson("error", "channel has no published frame");
        return result;
    }

    WebRtcOffer offer;
    offer.type = extractJsonStringField(requestBody, "type");
    offer.sdp = extractJsonStringField(requestBody, "sdp");
    if (offer.type != "offer" || offer.sdp.empty())
    {
        result.status = WebRtcSignalStatus::BadRequest;
        result.body = statusJson("error", "expected JSON offer with type and sdp");
        return result;
    }
    if (!containsVideoMediaSection(offer.sdp))
    {
        result.status = WebRtcSignalStatus::BadRequest;
        result.body = statusJson("error", "expected WebRTC offer SDP with a video media section");
        return result;
    }

    std::shared_ptr<SessionRuntime> runtime(new SessionRuntime);
    runtime->channel = channel;
    runtime->session = createWebRtcSession();

    try
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = false;
            runtime->id = cv::format("s%d", nextSessionId_++);
            runtime->channelRuntime = getOrCreateChannelLocked(channel, raw.info);
            runtime->channelRuntime->refs++;
        }

        if (!runtime->session->open(runtime->channelRuntime->codec, 30))
            CV_Error(Error::StsError, "failed to open WebRTC session");

        WebRtcAnswer answer;
        if (!runtime->session->createAnswer(offer, answer))
            CV_Error(Error::StsError, "failed to create WebRTC answer");

        runtime->viewer.reset(new ScopedViewer(viewers_, channel));
        runtime->thread = std::thread(&WebRtcManager::sessionPump, this, runtime);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            sessions_[runtime->id] = runtime;
        }

        result.status = WebRtcSignalStatus::Ok;
        result.body = answerJson(runtime->id, answer);
        return result;
    }
    catch (const Exception& e)
    {
        if (runtime->session)
            runtime->session->close();
        std::shared_ptr<ChannelRuntime> unusedChannel;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (runtime->channelRuntime)
            {
                runtime->channelRuntime->refs--;
                if (runtime->channelRuntime->refs <= 0)
                {
                    std::map<String, std::shared_ptr<ChannelRuntime> >::iterator it = channels_.find(runtime->channel);
                    if (it != channels_.end() && it->second == runtime->channelRuntime)
                    {
                        unusedChannel = it->second;
                        channels_.erase(it);
                    }
                }
            }
        }
        if (unusedChannel)
            unusedChannel->encoder.stop();
        result.status = WebRtcSignalStatus::BadRequest;
        result.body = statusJson("error", e.what());
        return result;
    }
}

WebRtcSignalResult WebRtcManager::addCandidate(const String& sessionId, const String& requestBody)
{
    WebRtcSignalResult result;
    std::shared_ptr<SessionRuntime> runtime;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::map<String, std::shared_ptr<SessionRuntime> >::iterator it = sessions_.find(sessionId);
        if (it != sessions_.end())
            runtime = it->second;
    }
    if (!runtime)
    {
        result.status = WebRtcSignalStatus::NotFound;
        result.body = statusJson("error", "unknown WebRTC session");
        return result;
    }

    WebRtcIceCandidate candidate;
    candidate.candidate = extractJsonStringField(requestBody, "candidate");
    extractJsonIntField(requestBody, "sdpMLineIndex", candidate.sdpMLineIndex);
    if (!runtime->session->addRemoteCandidate(candidate))
    {
        result.status = WebRtcSignalStatus::BadRequest;
        result.body = statusJson("error", "invalid ICE candidate");
        return result;
    }

    result.status = WebRtcSignalStatus::Ok;
    result.body = statusJson("ok", "candidate accepted");
    return result;
}

void WebRtcManager::releaseSession(const std::shared_ptr<SessionRuntime>& session)
{
    if (!session)
        return;

    session->stopping = true;
    if (session->thread.joinable())
        session->thread.join();
    if (session->session)
        session->session->close();
    session->viewer.reset();

    std::shared_ptr<ChannelRuntime> unusedChannel;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (session->channelRuntime)
        {
            session->channelRuntime->refs--;
            if (session->channelRuntime->refs <= 0)
            {
                std::map<String, std::shared_ptr<ChannelRuntime> >::iterator it = channels_.find(session->channel);
                if (it != channels_.end() && it->second == session->channelRuntime)
                {
                    unusedChannel = it->second;
                    channels_.erase(it);
                }
            }
            session->channelRuntime.reset();
        }
    }
    if (unusedChannel)
        unusedChannel->encoder.stop();
}

WebRtcSignalResult WebRtcManager::closeSession(const String& sessionId)
{
    WebRtcSignalResult result;
    std::shared_ptr<SessionRuntime> runtime;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::map<String, std::shared_ptr<SessionRuntime> >::iterator it = sessions_.find(sessionId);
        if (it != sessions_.end())
        {
            runtime = it->second;
            sessions_.erase(it);
        }
    }
    if (!runtime)
    {
        result.status = WebRtcSignalStatus::NotFound;
        result.body = statusJson("error", "unknown WebRTC session");
        return result;
    }

    releaseSession(runtime);
    result.status = WebRtcSignalStatus::Ok;
    result.body = statusJson("ok", "session closed");
    return result;
}

void WebRtcManager::sessionPump(const std::shared_ptr<SessionRuntime>& session)
{
    int64 sequence = 0;
    while (!stopping_.load() && !session->stopping.load())
    {
        EncodedFrame frame;
        if (!session->channelRuntime->encoder.waitForEncoded(sequence, 100, frame))
            continue;
        sequence = frame.sequence;
        if (!session->session->pushEncoded(frame))
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void WebRtcManager::stop()
{
    std::map<String, std::shared_ptr<SessionRuntime> > sessions;
    std::map<String, std::shared_ptr<ChannelRuntime> > channels;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_.load() && sessions_.empty() && channels_.empty())
            return;
        stopping_ = true;
        sessions.swap(sessions_);
        channels.swap(channels_);
    }

    for (std::map<String, std::shared_ptr<SessionRuntime> >::iterator it = sessions.begin(); it != sessions.end(); ++it)
        it->second->stopping = true;
    for (std::map<String, std::shared_ptr<SessionRuntime> >::iterator it = sessions.begin(); it != sessions.end(); ++it)
    {
        if (it->second->thread.joinable())
            it->second->thread.join();
        if (it->second->session)
            it->second->session->close();
        it->second->viewer.reset();
    }
    for (std::map<String, std::shared_ptr<ChannelRuntime> >::iterator it = channels.begin(); it != channels.end(); ++it)
        it->second->encoder.stop();
}

} // namespace liveview
} // namespace cv
