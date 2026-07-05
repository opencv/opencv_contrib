#include "opencv2/liveview.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ts.hpp"

#include "../src/channel_store.hpp"
#include "../src/frame_encoder.hpp"
#include "../src/mjpeg_writer.hpp"
#include "../src/route_matcher.hpp"
#include "../src/video_channel_encoder.hpp"
#include "../src/video_encoder.hpp"
#include "../src/web_backend.hpp"
#include "../src/webrtc_manager.hpp"
#include "../src/webrtc_session.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace opencv_test {
namespace {

struct HttpResponse
{
    int status;
    std::string raw;
    std::string body;
};

static int parsePort(const cv::String& url)
{
    const size_t colon = url.rfind(':');
    CV_Assert(colon != cv::String::npos);
    size_t end = url.find('/', colon);
    if (end == cv::String::npos)
        end = url.size();
    return std::atoi(url.substr(colon + 1, end - colon - 1).c_str());
}

static int parseStatus(const std::string& raw)
{
    if (raw.size() < 12 || raw.find("HTTP/1.1 ") != 0)
        return 0;
    return std::atoi(raw.substr(9, 3).c_str());
}

static HttpResponse makeResponse(const std::string& raw)
{
    HttpResponse response;
    response.status = parseStatus(raw);
    response.raw = raw;
    const size_t bodyPos = raw.find("\r\n\r\n");
    response.body = bodyPos == std::string::npos ? std::string() : raw.substr(bodyPos + 4);
    return response;
}

#if defined(HAVE_LIVEVIEW_VIDEO_ENCODER_FFMPEG)
static bool startsWithAnnexBStartCode(const std::vector<uchar>& bytes)
{
    if (bytes.size() < 4)
        return false;
    if (bytes[0] == 0 && bytes[1] == 0 && bytes[2] == 1)
        return true;
    return bytes[0] == 0 && bytes[1] == 0 && bytes[2] == 0 && bytes[3] == 1;
}

static cv::Mat syntheticFrame(int rows, int cols, int index)
{
    cv::Mat frame(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y)
    {
        cv::Vec3b* row = frame.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cols; ++x)
        {
            row[x][0] = static_cast<uchar>((x * 3 + index * 7) & 255);
            row[x][1] = static_cast<uchar>((y * 5 + index * 11) & 255);
            row[x][2] = static_cast<uchar>(((x + y) * 2 + index * 13) & 255);
        }
    }
    return frame;
}
#endif

static cv::liveview::VideoEncoderParams testVideoParams(cv::liveview::VideoCodec codec)
{
    cv::liveview::VideoEncoderParams params;
    params.width = 160;
    params.height = 120;
    params.fps = 30;
    params.bitrate = 250000;
    params.gop = 5;
    params.preferredCodec = codec;
    return params;
}

#ifndef _WIN32
static std::string httpRequest(int port, const std::string& method, const std::string& path,
                               size_t maxBytes = 1024 * 1024, int timeoutMs = 2000,
                               const std::string& body = std::string())
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    CV_Assert(fd >= 0);

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    CV_Assert(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
    {
        ::close(fd);
        CV_Error(cv::Error::StsError, "failed to connect to LiveView test server");
    }

    std::string request = method + " " + path + " HTTP/1.1\r\n"
                          "Host: 127.0.0.1\r\n"
                          "Connection: close\r\n";
    if (method != "GET")
    {
        request += "Content-Type: application/json\r\n";
        request += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    request += "\r\n";
    request += body;
    const char* out = request.data();
    size_t remaining = request.size();
    while (remaining > 0)
    {
        const ssize_t n = ::send(fd, out, remaining, 0);
        if (n <= 0)
        {
            ::close(fd);
            CV_Error(cv::Error::StsError, "failed to send LiveView test request");
        }
        out += n;
        remaining -= static_cast<size_t>(n);
    }

    std::string raw;
    char buffer[2048];
    while (raw.size() < maxBytes)
    {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        timeval tv;
        tv.tv_sec = timeoutMs / 1000;
        tv.tv_usec = (timeoutMs % 1000) * 1000;
        const int ready = ::select(fd + 1, &rfds, NULL, NULL, &tv);
        if (ready <= 0)
            break;
        const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0)
            break;
        raw.append(buffer, static_cast<size_t>(n));
    }
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    return raw;
}
#endif

TEST(LiveView, CreateServer)
{
    cv::Ptr<cv::liveview::Server> server = cv::liveview::createServer();
    ASSERT_FALSE(server.empty());
    EXPECT_FALSE(server->isRunning());
    EXPECT_TRUE(server->url().empty());
}

TEST(LiveView, FrameEncoderAcceptsOnlyWebSafe8BitFrames)
{
    cv::liveview::FrameEncoder encoder(85);

    const cv::Mat gray(8, 12, CV_8UC1, cv::Scalar(42));
    EXPECT_FALSE(cv::imdecode(encoder.encodeJpeg(gray), cv::IMREAD_UNCHANGED).empty());

    const cv::Mat bgr(8, 12, CV_8UC3, cv::Scalar(10, 20, 30));
    EXPECT_EQ(3, cv::imdecode(encoder.encodeJpeg(bgr), cv::IMREAD_UNCHANGED).channels());

    const cv::Mat bgra(8, 12, CV_8UC4, cv::Scalar(10, 20, 30, 255));
    EXPECT_EQ(3, cv::imdecode(encoder.encodeJpeg(bgra), cv::IMREAD_UNCHANGED).channels());

    EXPECT_THROW(cv::liveview::FrameEncoder(0), cv::Exception);
    EXPECT_THROW(encoder.validate(cv::Mat()), cv::Exception);
    EXPECT_THROW(encoder.validate(cv::Mat(4, 4, CV_32FC1, cv::Scalar(1))), cv::Exception);
}

TEST(LiveView, ChannelStoreUsesBackgroundEncoderAndTracksSnapshots)
{
    cv::liveview::ChannelStore store;
    EXPECT_TRUE(cv::liveview::isValidChannelName("camera_1.left"));
    EXPECT_FALSE(cv::liveview::isValidChannelName(""));
    EXPECT_FALSE(cv::liveview::isValidChannelName("../bad"));
    EXPECT_FALSE(cv::liveview::isValidChannelName("bad/name"));

    const cv::Mat frame(6, 7, CV_8UC3, cv::Scalar(1, 2, 3));
    store.publish("camera_1.left", frame);
    store.publish("camera_1.left", frame);

    std::vector<cv::liveview::ChannelInfo> channels = store.channels();
    ASSERT_EQ(1u, channels.size());
    EXPECT_EQ("camera_1.left", channels[0].name);
    EXPECT_EQ(7, channels[0].width);
    EXPECT_EQ(6, channels[0].height);
    EXPECT_EQ(CV_8UC3, channels[0].type);
    EXPECT_EQ(2, channels[0].sequence);

    cv::liveview::ChannelSnapshot snapshot;
    ASSERT_TRUE(store.waitForJpeg("camera_1.left", 0, 2000, snapshot));
    EXPECT_GE(snapshot.info.sequence, 1);
    EXPECT_FALSE(snapshot.jpeg.empty());
    EXPECT_FALSE(cv::imdecode(snapshot.jpeg, cv::IMREAD_COLOR).empty());

    EXPECT_THROW(store.publish("bad/name", frame), cv::Exception);
    EXPECT_FALSE(store.getSnapshot("missing", snapshot));

    cv::liveview::RawFrameSnapshot raw;
    ASSERT_TRUE(store.getRawFrame("camera_1.left", raw));
    EXPECT_EQ(2, raw.info.sequence);
    EXPECT_EQ(7, raw.frame.cols);
    EXPECT_EQ(6, raw.frame.rows);
    EXPECT_EQ(CV_8UC3, raw.frame.type());
}

TEST(LiveView, ChannelStoreWaitsForNewEncodedSequences)
{
    cv::liveview::ChannelStore store;
    store.publish("depth", cv::Mat(3, 5, CV_8UC1, cv::Scalar(10)));

    cv::liveview::ChannelSnapshot snapshot;
    ASSERT_TRUE(store.waitForJpeg("depth", 0, 1000, snapshot));
    EXPECT_EQ(1, snapshot.info.sequence);
    EXPECT_FALSE(store.waitForJpeg("depth", 1, 10, snapshot));

    std::thread publisher([&store]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        store.publish("depth", cv::Mat(3, 5, CV_8UC1, cv::Scalar(20)));
    });
    EXPECT_TRUE(store.waitForJpeg("depth", 1, 1000, snapshot));
    publisher.join();
    EXPECT_EQ(2, snapshot.info.sequence);

    cv::liveview::RawFrameSnapshot raw;
    ASSERT_TRUE(store.getRawFrame("depth", raw));
    EXPECT_EQ(2, raw.info.sequence);
    EXPECT_FALSE(store.waitForRawFrame("depth", 2, 10, raw));
}

TEST(LiveView, VideoEncoderParamsValidation)
{
    cv::liveview::VideoEncoderParams params = testVideoParams(cv::liveview::VideoCodec::H264);
    EXPECT_NO_THROW(cv::liveview::validateVideoEncoderParams(params));
    EXPECT_EQ("H264", cv::liveview::videoCodecName(cv::liveview::VideoCodec::H264));
    EXPECT_EQ("VP8", cv::liveview::videoCodecName(cv::liveview::VideoCodec::VP8));
    EXPECT_EQ("Unknown", cv::liveview::videoCodecName(cv::liveview::VideoCodec::Unknown));

    params.width = 0;
    EXPECT_THROW(cv::liveview::validateVideoEncoderParams(params), cv::Exception);
    params = testVideoParams(cv::liveview::VideoCodec::H264);
    params.height = -1;
    EXPECT_THROW(cv::liveview::validateVideoEncoderParams(params), cv::Exception);
    params = testVideoParams(cv::liveview::VideoCodec::H264);
    params.fps = 0;
    EXPECT_THROW(cv::liveview::validateVideoEncoderParams(params), cv::Exception);
    params = testVideoParams(cv::liveview::VideoCodec::H264);
    params.bitrate = 0;
    EXPECT_THROW(cv::liveview::validateVideoEncoderParams(params), cv::Exception);
    params = testVideoParams(cv::liveview::VideoCodec::H264);
    params.gop = 0;
    EXPECT_THROW(cv::liveview::validateVideoEncoderParams(params), cv::Exception);
    params = testVideoParams(cv::liveview::VideoCodec::Unknown);
    EXPECT_THROW(cv::liveview::validateVideoEncoderParams(params), cv::Exception);
}

TEST(LiveView, VideoEncoderBuildFlagBehavior)
{
#if defined(HAVE_LIVEVIEW_VIDEO_ENCODER_FFMPEG)
    EXPECT_TRUE(cv::liveview::haveVideoEncoderBackend());
    EXPECT_FALSE(cv::liveview::createVideoEncoder().empty());
#else
    EXPECT_FALSE(cv::liveview::haveVideoEncoderBackend());
    EXPECT_TRUE(cv::liveview::availableVideoEncoders(cv::liveview::VideoCodec::H264).empty());
    EXPECT_THROW(cv::liveview::createVideoEncoder(), cv::Exception);
#endif
}

TEST(LiveView, WebRtcBuildFlagBehavior)
{
#if defined(HAVE_LIVEVIEW_WEBRTC_GSTREAMER)
    EXPECT_TRUE(cv::liveview::haveWebRtcBackend());
    EXPECT_FALSE(cv::liveview::createWebRtcSession().empty());
#else
    EXPECT_FALSE(cv::liveview::haveWebRtcBackend());
    EXPECT_THROW(cv::liveview::createWebRtcSession(), cv::Exception);
#endif
}

TEST(LiveView, FfmpegVideoEncoderProducesAnnexBH264AccessUnits)
{
#if !defined(HAVE_LIVEVIEW_VIDEO_ENCODER_FFMPEG)
    throw SkipTestException("LiveView FFmpeg encoder support is disabled");
#else
    if (cv::liveview::availableVideoEncoders(cv::liveview::VideoCodec::H264).empty())
        throw SkipTestException("No FFmpeg H.264 encoder is available");

    cv::Ptr<cv::liveview::VideoEncoder> encoder = cv::liveview::createVideoEncoder();
    cv::liveview::VideoEncoderParams params = testVideoParams(cv::liveview::VideoCodec::H264);
    encoder->open(params);
    ASSERT_TRUE(encoder->isOpened());

    bool produced = false;
    bool keyframe = false;
    int64 lastSequence = 0;
    for (int i = 1; i <= 10; ++i)
    {
        encoder->encode(syntheticFrame(params.height, params.width, i), i, i * 33333);
        cv::liveview::EncodedFrame frame;
        while (encoder->tryPop(frame))
        {
            produced = true;
            keyframe = keyframe || frame.keyframe;
            EXPECT_EQ(cv::liveview::VideoCodec::H264, frame.codec);
            EXPECT_FALSE(frame.data.empty());
            EXPECT_TRUE(startsWithAnnexBStartCode(frame.data));
            EXPECT_GE(frame.sequence, lastSequence);
            lastSequence = frame.sequence;
        }
    }
    EXPECT_TRUE(produced);
    EXPECT_TRUE(keyframe);
    EXPECT_THROW(encoder->encode(cv::Mat(params.height, params.width, CV_32FC1, cv::Scalar(1)), 11, 11 * 33333), cv::Exception);
    EXPECT_THROW(encoder->encode(cv::Mat(params.height + 2, params.width, CV_8UC3, cv::Scalar(1, 2, 3)), 12, 12 * 33333), cv::Exception);
    encoder->close();
    EXPECT_FALSE(encoder->isOpened());
    EXPECT_NO_THROW(encoder->open(params));
    encoder->close();
#endif
}

TEST(LiveView, FfmpegVideoEncoderProducesVP8WhenAvailable)
{
#if !defined(HAVE_LIVEVIEW_VIDEO_ENCODER_FFMPEG)
    throw SkipTestException("LiveView FFmpeg encoder support is disabled");
#else
    if (cv::liveview::availableVideoEncoders(cv::liveview::VideoCodec::VP8).empty())
        throw SkipTestException("No FFmpeg VP8 encoder is available");

    cv::Ptr<cv::liveview::VideoEncoder> encoder = cv::liveview::createVideoEncoder();
    cv::liveview::VideoEncoderParams params = testVideoParams(cv::liveview::VideoCodec::VP8);
    encoder->open(params);
    bool produced = false;
    for (int i = 1; i <= 4; ++i)
    {
        encoder->encode(syntheticFrame(params.height, params.width, i), i, i * 33333);
        cv::liveview::EncodedFrame frame;
        while (encoder->tryPop(frame))
        {
            produced = true;
            EXPECT_EQ(cv::liveview::VideoCodec::VP8, frame.codec);
            EXPECT_FALSE(frame.data.empty());
        }
    }
    EXPECT_TRUE(produced);
#endif
}

TEST(LiveView, FfmpegVideoEncoderFailsCleanlyForUnavailableExplicitEncoder)
{
#if !defined(HAVE_LIVEVIEW_VIDEO_ENCODER_FFMPEG)
    throw SkipTestException("LiveView FFmpeg encoder support is disabled");
#else
    cv::Ptr<cv::liveview::VideoEncoder> encoder = cv::liveview::createVideoEncoder();
    cv::liveview::VideoEncoderParams params = testVideoParams(cv::liveview::VideoCodec::H264);
    params.encoderName = "definitely_missing_liveview_encoder";
    EXPECT_THROW(encoder->open(params), cv::Exception);
    params.encoderName = "libvpx";
    EXPECT_THROW(encoder->open(params), cv::Exception);
#endif
}

TEST(LiveView, VideoChannelEncoderConsumesLatestChannelFrames)
{
#if !defined(HAVE_LIVEVIEW_VIDEO_ENCODER_FFMPEG)
    throw SkipTestException("LiveView FFmpeg encoder support is disabled");
#else
    if (cv::liveview::availableVideoEncoders(cv::liveview::VideoCodec::H264).empty())
        throw SkipTestException("No FFmpeg H.264 encoder is available");

    cv::liveview::ChannelStore store;
    cv::liveview::VideoEncoderParams params = testVideoParams(cv::liveview::VideoCodec::H264);
    cv::liveview::VideoChannelEncoder encoder(store, "camera", params);
    encoder.start();
    EXPECT_TRUE(encoder.isRunning());

    cv::liveview::EncodedFrame encoded;
    bool gotFirst = false;
    for (int i = 1; i <= 60 && !gotFirst; ++i)
    {
        store.publish("camera", syntheticFrame(params.height, params.width, i));
        gotFirst = encoder.waitForEncoded(0, 50, encoded);
    }
    ASSERT_TRUE(gotFirst);
    EXPECT_EQ("camera", encoded.channel);
    EXPECT_EQ(cv::liveview::VideoCodec::H264, encoded.codec);
    EXPECT_FALSE(encoded.data.empty());
    EXPECT_GT(encoded.sequence, 0);

    const int64 first = encoded.sequence;
    bool gotNext = false;
    for (int i = 61; i <= 120 && !gotNext; ++i)
    {
        store.publish("camera", syntheticFrame(params.height, params.width, i));
        gotNext = encoder.waitForEncoded(first, 50, encoded);
    }
    EXPECT_TRUE(gotNext);
    EXPECT_GE(encoded.sequence, first + 1);
    EXPECT_TRUE(encoder.getLatest(encoded));

    encoder.stop();
    EXPECT_FALSE(encoder.isRunning());
#endif
}

TEST(LiveView, RouteMatcherAndWritersAreDeterministic)
{
    cv::liveview::RouteMatch route = cv::liveview::matchRoute("GET", "/channels.json?compact=1");
    EXPECT_EQ(cv::liveview::RouteKind::ChannelsJson, route.kind);

    route = cv::liveview::matchRoute("GET", "/frame/camera.jpg");
    EXPECT_EQ(cv::liveview::RouteKind::Snapshot, route.kind);
    EXPECT_EQ("camera", route.channel);

    route = cv::liveview::matchRoute("GET", "/stream/camera.mjpeg");
    EXPECT_EQ(cv::liveview::RouteKind::Mjpeg, route.kind);

    route = cv::liveview::matchRoute("GET", "/webrtc/camera");
    EXPECT_EQ(cv::liveview::RouteKind::WebRtcViewer, route.kind);
    EXPECT_EQ("camera", route.channel);

    route = cv::liveview::matchRoute("POST", "/webrtc/camera/offer");
    EXPECT_EQ(cv::liveview::RouteKind::WebRtcOffer, route.kind);
    EXPECT_EQ("camera", route.channel);

    route = cv::liveview::matchRoute("POST", "/webrtc/session/s1/candidate");
    EXPECT_EQ(cv::liveview::RouteKind::WebRtcCandidate, route.kind);
    EXPECT_EQ("s1", route.channel);

    EXPECT_EQ(cv::liveview::RouteKind::Unknown, cv::liveview::matchRoute("POST", "/channels.json").kind);
    EXPECT_EQ(cv::liveview::RouteKind::Unknown, cv::liveview::matchRoute("GET", "/frame/../x.jpg").kind);

    EXPECT_EQ("Not Found", cv::liveview::httpStatusText(404));
    const cv::String mjpeg = cv::liveview::mjpegPartHeader("boundary", 123);
    EXPECT_NE(cv::String::npos, mjpeg.find("--boundary\r\n"));
    EXPECT_NE(cv::String::npos, mjpeg.find("Content-Type: image/jpeg\r\n"));
    EXPECT_NE(cv::String::npos, mjpeg.find("Content-Length: 123\r\n"));

    EXPECT_EQ("&lt;&amp;&quot;&#39;&gt;", cv::liveview::htmlEscape("<&\"'>"));
    EXPECT_EQ("line\\nquote\\\"ctrl\\u0001", cv::liveview::jsonEscape("line\nquote\"ctrl\001"));
    EXPECT_EQ("CV_8UC3", cv::liveview::typeToString(CV_8UC3));
}

TEST(LiveView, WebBackendConformance)
{
#ifdef _WIN32
    throw SkipTestException("LiveView backend conformance tests use localhost sockets");
#else
    cv::Ptr<cv::liveview::WebBackend> backend = cv::liveview::createWebBackend();
    ASSERT_FALSE(backend.empty());
    backend->start("127.0.0.1", 0, [](const cv::liveview::WebRequest& request, cv::liveview::WebResponse& response) {
        if (request.path == "/ok")
        {
            response.setStatus(200);
            response.setHeader("Content-Type", "text/plain");
            response.writeString("backend-ok\n");
            return;
        }
        if (request.path == "/echo")
        {
            response.setStatus(200);
            response.setHeader("Content-Type", "application/json");
            response.writeString(request.body);
            return;
        }
        response.setStatus(404);
        response.setHeader("Content-Type", "text/plain");
        response.writeString("missing\n");
    });
    ASSERT_TRUE(backend->isRunning());
    ASSERT_GT(backend->port(), 0);
    EXPECT_FALSE(backend->name().empty());

    HttpResponse ok = makeResponse(httpRequest(backend->port(), "GET", "/ok"));
    EXPECT_EQ(200, ok.status);
    EXPECT_EQ("backend-ok\n", ok.body);

    HttpResponse missing = makeResponse(httpRequest(backend->port(), "GET", "/missing"));
    EXPECT_EQ(404, missing.status);

    HttpResponse echo = makeResponse(httpRequest(backend->port(), "POST", "/echo", 1024, 2000, "{\"x\":1}"));
    EXPECT_EQ(200, echo.status);
    EXPECT_EQ("{\"x\":1}", echo.body);

    backend->stop();
    EXPECT_FALSE(backend->isRunning());
#endif
}

TEST(LiveView, PublicServerLifecycleAndUrls)
{
#ifdef _WIN32
    throw SkipTestException("LiveView Step 1 HTTP server is not implemented on Windows");
#else
    cv::Ptr<cv::liveview::Server> server = cv::liveview::createServer("127.0.0.1", 0);
    server->start();
    EXPECT_TRUE(server->isRunning());
    EXPECT_NE(cv::String::npos, server->url().find("http://127.0.0.1:"));
    EXPECT_EQ("/frame/camera.jpg", server->channelUrl("camera", cv::liveview::Transport::Snapshot).substr(server->url().size() - 1));
    EXPECT_EQ("/stream/camera.mjpeg", server->channelUrl("camera", cv::liveview::Transport::Mjpeg).substr(server->url().size() - 1));
    EXPECT_THROW(server->channelUrl("camera", cv::liveview::Transport::WebRTC), cv::Exception);
    EXPECT_THROW(server->channelUrl("bad/name", cv::liveview::Transport::Snapshot), cv::Exception);
    server->stop();
    EXPECT_FALSE(server->isRunning());
    EXPECT_TRUE(server->url().empty());

    cv::Ptr<cv::liveview::Server> webrtcServer = cv::liveview::createServer("127.0.0.1", 0, true);
#if defined(HAVE_LIVEVIEW_WEBRTC_GSTREAMER)
    if (!cv::liveview::haveWebRtcBackend())
        EXPECT_THROW(webrtcServer->start(), cv::Exception);
    else
    {
        webrtcServer->start();
        EXPECT_TRUE(webrtcServer->isRunning());
        EXPECT_NE(cv::String::npos, webrtcServer->channelUrl("camera", cv::liveview::Transport::WebRTC).find("/webrtc/camera"));
        EXPECT_NE(cv::String::npos, webrtcServer->channelUrl("camera", cv::liveview::Transport::Auto).find("/webrtc/camera"));
        webrtcServer->stop();
    }
#else
    EXPECT_THROW(webrtcServer->start(), cv::Exception);
#endif
#endif
}

TEST(LiveView, WebRtcRoutesAndSignalingValidation)
{
#ifdef _WIN32
    throw SkipTestException("LiveView WebRTC route tests use localhost sockets");
#else
#if !defined(HAVE_LIVEVIEW_WEBRTC_GSTREAMER)
    throw SkipTestException("LiveView WebRTC support is disabled");
#else
    if (!cv::liveview::haveWebRtcBackend())
        throw SkipTestException("LiveView WebRTC runtime elements are unavailable");

    cv::Ptr<cv::liveview::Server> server = cv::liveview::createServer("127.0.0.1", 0, true);
    server->start();
    const int port = parsePort(server->url());

    server->publish("camera", cv::Mat(120, 160, CV_8UC3, cv::Scalar(10, 120, 240)));

    HttpResponse index = makeResponse(httpRequest(port, "GET", "/"));
    EXPECT_EQ(200, index.status);
    EXPECT_NE(std::string::npos, index.body.find("/webrtc/camera"));

    HttpResponse viewer = makeResponse(httpRequest(port, "GET", "/webrtc/camera"));
    EXPECT_EQ(200, viewer.status);
    EXPECT_NE(std::string::npos, viewer.body.find("RTCPeerConnection"));
    EXPECT_NE(std::string::npos, viewer.body.find("/webrtc/'+channel+'/offer"));

    HttpResponse missing = makeResponse(httpRequest(port, "POST", "/webrtc/missing/offer", 4096, 2000,
                                                    "{\"type\":\"offer\",\"sdp\":\"v=0\\r\\n\"}"));
    EXPECT_EQ(404, missing.status);

    HttpResponse bad = makeResponse(httpRequest(port, "POST", "/webrtc/camera/offer", 4096, 2000,
                                                "{\"type\":\"answer\",\"sdp\":\"\"}"));
    EXPECT_EQ(400, bad.status);
    EXPECT_NE(std::string::npos, bad.body.find("expected JSON offer"));

    HttpResponse badSdp = makeResponse(httpRequest(port, "POST", "/webrtc/camera/offer", 4096, 2000,
                                                   "{\"type\":\"offer\",\"sdp\":\"v=0\\r\\n\"}"));
    EXPECT_EQ(400, badSdp.status);
    EXPECT_NE(std::string::npos, badSdp.body.find("video media section"));

    HttpResponse candidate = makeResponse(httpRequest(port, "POST", "/webrtc/session/missing/candidate", 4096, 2000,
                                                      "{\"candidate\":\"\",\"sdpMLineIndex\":0}"));
    EXPECT_EQ(404, candidate.status);

    server->stop();
#endif
#endif
}

TEST(LiveView, HttpRoutesExposePublishedChannels)
{
#ifdef _WIN32
    throw SkipTestException("LiveView Step 1 HTTP server is not implemented on Windows");
#else
    cv::Ptr<cv::liveview::Server> server = cv::liveview::createServer("127.0.0.1", 0);
    server->start();
    const int port = parsePort(server->url());

    HttpResponse health = makeResponse(httpRequest(port, "GET", "/healthz"));
    EXPECT_EQ(200, health.status);
    EXPECT_EQ("ok\n", health.body);

    HttpResponse emptyJson = makeResponse(httpRequest(port, "GET", "/channels.json"));
    EXPECT_EQ(200, emptyJson.status);
    EXPECT_NE(std::string::npos, emptyJson.body.find("\"channels\":[]"));

    server->publish("camera", cv::Mat(24, 32, CV_8UC3, cv::Scalar(10, 120, 240)));

    HttpResponse index = makeResponse(httpRequest(port, "GET", "/"));
    EXPECT_EQ(200, index.status);
    EXPECT_NE(std::string::npos, index.body.find("OpenCV LiveView"));
    EXPECT_NE(std::string::npos, index.body.find("camera"));

    HttpResponse json = makeResponse(httpRequest(port, "GET", "/channels.json"));
    EXPECT_EQ(200, json.status);
    EXPECT_NE(std::string::npos, json.body.find("\"name\":\"camera\""));
    EXPECT_NE(std::string::npos, json.body.find("\"width\":32"));
    EXPECT_NE(std::string::npos, json.body.find("\"height\":24"));
    EXPECT_NE(std::string::npos, json.body.find("\"sequence\":1"));

    HttpResponse jpeg = makeResponse(httpRequest(port, "GET", "/frame/camera.jpg"));
    EXPECT_EQ(200, jpeg.status);
    std::vector<uchar> bytes(jpeg.body.begin(), jpeg.body.end());
    EXPECT_FALSE(cv::imdecode(bytes, cv::IMREAD_COLOR).empty());

    EXPECT_EQ(404, makeResponse(httpRequest(port, "GET", "/frame/missing.jpg")).status);
    EXPECT_EQ(405, makeResponse(httpRequest(port, "POST", "/channels.json")).status);
    EXPECT_EQ(404, makeResponse(httpRequest(port, "GET", "/unknown")).status);

    server->stop();
#endif
}

TEST(LiveView, MjpegRouteStreamsMultipartFrames)
{
#ifdef _WIN32
    throw SkipTestException("LiveView Step 1 HTTP server is not implemented on Windows");
#else
    cv::Ptr<cv::liveview::Server> server = cv::liveview::createServer("127.0.0.1", 0);
    server->start();
    const int port = parsePort(server->url());
    server->publish("camera", cv::Mat(12, 16, CV_8UC3, cv::Scalar(20, 30, 40)));

    const std::string raw = httpRequest(port, "GET", "/stream/camera.mjpeg", 8192, 2000);
    EXPECT_NE(std::string::npos, raw.find("HTTP/1.1 200 OK\r\n"));
    EXPECT_NE(std::string::npos, raw.find("multipart/x-mixed-replace; boundary=opencv-liveview-frame"));
    EXPECT_NE(std::string::npos, raw.find("--opencv-liveview-frame\r\n"));
    EXPECT_NE(std::string::npos, raw.find("Content-Type: image/jpeg\r\n"));

    EXPECT_EQ(404, makeResponse(httpRequest(port, "GET", "/stream/missing.mjpeg")).status);

    server->stop();
#endif
}

} // namespace
} // namespace opencv_test
