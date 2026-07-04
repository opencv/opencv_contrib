#ifndef OPENCV_LIVEVIEW_HPP
#define OPENCV_LIVEVIEW_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace liveview {

//! Transport preference for channel URLs.
enum class Transport
{
    Auto = 0,
    Snapshot = 1,
    Mjpeg = 2,
    WebRTC = 3
};

//! Basic published channel metadata.
struct CV_EXPORTS_W_SIMPLE ChannelInfo
{
    CV_PROP_RW String name;
    CV_PROP_RW int width = 0;
    CV_PROP_RW int height = 0;
    CV_PROP_RW int type = 0;
    CV_PROP_RW int64 sequence = 0;
};

//! LiveView server facade.
class CV_EXPORTS_W Server
{
public:
    CV_WRAP Server();
    CV_WRAP explicit Server(const String& host, int port = 0, bool enableWebRTC = false);
    ~Server();

    CV_WRAP void start();
    CV_WRAP void stop();
    CV_WRAP bool isRunning() const;

    CV_WRAP String url() const;
    CV_WRAP String channelUrl(const String& name, Transport transport = Transport::Auto) const;

    CV_WRAP void publish(const String& name, InputArray frame);
    CV_WRAP std::vector<ChannelInfo> channels() const;

private:
    struct Impl;
    Ptr<Impl> impl_;
};

CV_EXPORTS_W Ptr<Server> createServer(const String& host = "127.0.0.1",
                                      int port = 0,
                                      bool enableWebRTC = false);

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_HPP
