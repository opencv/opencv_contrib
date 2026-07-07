#include "opencv2/liveview.hpp"

#include "channel_store.hpp"
#include "mjpeg_writer.hpp"
#include "route_matcher.hpp"
#include "web_backend.hpp"
#include "webrtc_manager.hpp"

#include <sstream>

namespace cv {
namespace liveview {

struct Server::Impl
{
    Impl()
        : host("127.0.0.1"), port(0), enableWebRTC(false), store(80),
          backend(createWebBackend()), webrtc(store)
    {
    }

    String host;
    int port;
    bool enableWebRTC;
    ChannelStore store;
    Ptr<WebBackend> backend;
    WebRtcManager webrtc;

    void start()
    {
        if (enableWebRTC && !webrtc.isAvailable())
            CV_Error(Error::StsNotImplemented, "LiveView WebRTC transport is not available in this build/runtime");
        backend->start(host, port, [this](const WebRequest& request, WebResponse& response) {
            handle(request, response);
        });
    }

    void stop()
    {
        webrtc.stop();
        backend->stop();
    }

    bool isRunning() const
    {
        return backend->isRunning();
    }

    String baseUrl() const
    {
        std::ostringstream os;
        os << "http://" << backend->host() << ":" << backend->port();
        return os.str();
    }

    String channelUrl(const String& name, Transport transport) const
    {
        if (!isValidChannelName(name))
            CV_Error(Error::StsBadArg, "Invalid LiveView channel name");
        const Transport resolved = (transport == Transport::Auto) ?
            (enableWebRTC ? Transport::WebRTC : Transport::Mjpeg) : transport;
        if (resolved == Transport::Snapshot)
            return baseUrl() + "/frame/" + name + ".jpg";
        if (resolved == Transport::Mjpeg)
            return baseUrl() + "/stream/" + name + ".mjpeg";
        if (resolved == Transport::WebRTC)
        {
            if (!enableWebRTC || !webrtc.isAvailable())
                CV_Error(Error::StsNotImplemented, "Requested LiveView WebRTC transport is not available");
            return baseUrl() + "/webrtc/" + name;
        }
        CV_Error(Error::StsBadArg, "Unknown LiveView transport");
    }

    String channelsJson() const
    {
        const std::vector<ChannelInfo> infos = store.channels();
        std::ostringstream os;
        os << "{\"channels\":[";
        for (size_t i = 0; i < infos.size(); ++i)
        {
            if (i)
                os << ",";
            os << "{";
            os << "\"name\":\"" << jsonEscape(infos[i].name) << "\",";
            os << "\"width\":" << infos[i].width << ",";
            os << "\"height\":" << infos[i].height << ",";
            os << "\"type\":\"" << typeToString(infos[i].type) << "\",";
            os << "\"sequence\":" << infos[i].sequence;
            os << "}";
        }
        os << "]}";
        return os.str();
    }

    String indexHtml() const
    {
        const std::vector<ChannelInfo> infos = store.channels();
        std::ostringstream os;
        os << "<!doctype html><html><head><meta charset='utf-8'>";
        os << "<title>OpenCV LiveView</title>";
        os << "<style>body{font-family:sans-serif;margin:24px}li{margin:8px 0}";
        os << "code{background:#eee;padding:2px 4px}</style>";
        os << "</head><body><h1>OpenCV LiveView</h1>";
        if (infos.empty())
        {
            os << "<p>No channels have been published yet.</p>";
        }
        else
        {
            os << "<ul>";
            for (size_t i = 0; i < infos.size(); ++i)
            {
                const String n = htmlEscape(infos[i].name);
                os << "<li><code>" << n << "</code> ";
                os << infos[i].width << "x" << infos[i].height << " ";
                os << htmlEscape(typeToString(infos[i].type)) << " ";
                os << "<a href='/frame/" << n << ".jpg'>snapshot</a> ";
                os << "<a href='/stream/" << n << ".mjpeg'>stream</a> ";
                if (enableWebRTC && webrtc.isAvailable())
                    os << "<a href='/webrtc/" << n << "'>webrtc</a>";
                os << "</li>";
            }
            os << "</ul>";
        }
        os << "</body></html>";
        return os.str();
    }

    void sendBody(WebResponse& response, int status, const String& body, const String& contentType) const
    {
        response.setStatus(status);
        response.setHeader("Content-Type", contentType);
        response.setHeader("Cache-Control", "no-store");
        response.writeString(body);
    }

    void handle(const WebRequest& request, WebResponse& response)
    {
        const RouteMatch route = matchRoute(request.method, request.path);
        if (request.method != "GET" && route.kind != RouteKind::WebRtcOffer && route.kind != RouteKind::WebRtcCandidate)
        {
            sendBody(response, 405, "405 Method Not Allowed\n", "text/plain; charset=utf-8");
            return;
        }

        if (route.kind == RouteKind::Index)
        {
            sendBody(response, 200, indexHtml(), "text/html; charset=utf-8");
            return;
        }
        if (route.kind == RouteKind::ChannelsJson)
        {
            sendBody(response, 200, channelsJson(), "application/json; charset=utf-8");
            return;
        }
        if (route.kind == RouteKind::Health)
        {
            sendBody(response, 200, "ok\n", "text/plain; charset=utf-8");
            return;
        }
        if (route.kind == RouteKind::WebRtcViewer)
        {
            if (!enableWebRTC || !webrtc.isAvailable())
            {
                sendBody(response, 404, "404 Not Found\n", "text/plain; charset=utf-8");
                return;
            }
            sendBody(response, 200, webrtc.viewerHtml(baseUrl(), route.channel), "text/html; charset=utf-8");
            return;
        }
        if (route.kind == RouteKind::WebRtcOffer)
        {
            if (!enableWebRTC || !webrtc.isAvailable())
            {
                sendBody(response, 404, "404 Not Found\n", "text/plain; charset=utf-8");
                return;
            }
            WebRtcSignalResult result = webrtc.createOfferAnswer(route.channel, request.body);
            sendWebRtcSignal(response, result);
            return;
        }
        if (route.kind == RouteKind::WebRtcCandidate)
        {
            if (!enableWebRTC || !webrtc.isAvailable())
            {
                sendBody(response, 404, "404 Not Found\n", "text/plain; charset=utf-8");
                return;
            }
            WebRtcSignalResult result = webrtc.addCandidate(route.channel, request.body);
            sendWebRtcSignal(response, result);
            return;
        }
        if (route.kind == RouteKind::Snapshot)
        {
            ChannelSnapshot snapshot;
            if (!store.getSnapshot(route.channel, snapshot) &&
                !store.waitForJpeg(route.channel, 0, 1000, snapshot))
            {
                sendBody(response, 404, "404 Not Found\n", "text/plain; charset=utf-8");
                return;
            }
            response.setStatus(200);
            response.setHeader("Content-Type", "image/jpeg");
            response.setHeader("Cache-Control", "no-store");
            response.write(&snapshot.jpeg[0], snapshot.jpeg.size());
            return;
        }
        if (route.kind == RouteKind::Mjpeg)
        {
            ChannelSnapshot snapshot;
            if (!store.getSnapshot(route.channel, snapshot) &&
                !store.waitForJpeg(route.channel, 0, 1000, snapshot))
            {
                sendBody(response, 404, "404 Not Found\n", "text/plain; charset=utf-8");
                return;
            }
            const String boundary = "opencv-liveview-frame";
            response.setStatus(200);
            response.setHeader("Content-Type", "multipart/x-mixed-replace; boundary=" + boundary);
            response.setHeader("Cache-Control", "no-store");
            const String part = mjpegPartHeader(boundary, snapshot.jpeg.size());
            response.writeString(part);
            response.write(&snapshot.jpeg[0], snapshot.jpeg.size());
            response.writeString("\r\n");
            return;
        }

        sendBody(response, 404, "404 Not Found\n", "text/plain; charset=utf-8");
    }

    void sendWebRtcSignal(WebResponse& response, const WebRtcSignalResult& result) const
    {
        int status = 500;
        if (result.status == WebRtcSignalStatus::Ok)
            status = 200;
        else if (result.status == WebRtcSignalStatus::BadRequest)
            status = 400;
        else if (result.status == WebRtcSignalStatus::NotFound)
            status = 404;
        else if (result.status == WebRtcSignalStatus::NotSupported)
            status = 501;
        sendBody(response, status, result.body + "\n", "application/json; charset=utf-8");
    }
};

Server::Server()
    : impl_(makePtr<Impl>())
{
}

Server::Server(const String& host, int port, bool enableWebRTC)
    : impl_(makePtr<Impl>())
{
    impl_->host = host;
    impl_->port = port;
    impl_->enableWebRTC = enableWebRTC;
}

Server::~Server()
{
    stop();
}

void Server::start()
{
    if (impl_)
        impl_->start();
}

void Server::stop()
{
    if (impl_)
        impl_->stop();
}

bool Server::isRunning() const
{
    return impl_ && impl_->isRunning();
}

String Server::url() const
{
    if (!impl_ || !impl_->isRunning())
        return String();
    return impl_->baseUrl() + "/";
}

String Server::channelUrl(const String& name, Transport transport) const
{
    if (!impl_ || !impl_->isRunning())
        return String();
    return impl_->channelUrl(name, transport);
}

void Server::publish(const String& name, InputArray frame)
{
    if (!impl_)
        CV_Error(Error::StsError, "LiveView server implementation is not initialized");
    impl_->store.publish(name, frame);
}

std::vector<ChannelInfo> Server::channels() const
{
    return impl_ ? impl_->store.channels() : std::vector<ChannelInfo>();
}

Ptr<Server> createServer(const String& host, int port, bool enableWebRTC)
{
    return makePtr<Server>(host, port, enableWebRTC);
}

} // namespace liveview
} // namespace cv
