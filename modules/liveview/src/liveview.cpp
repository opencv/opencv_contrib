#include "opencv2/liveview.hpp"

namespace cv {
namespace liveview {

struct Server::Impl
{
    String host = "127.0.0.1";
    int port = 0;
    bool enableWebRTC = false;
    bool running = false;
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
    CV_Error(Error::StsNotImplemented, "cv::liveview::Server::start is not implemented in the skeleton module");
}

void Server::stop()
{
    if (impl_)
        impl_->running = false;
}

bool Server::isRunning() const
{
    return impl_ && impl_->running;
}

String Server::url() const
{
    CV_Error(Error::StsNotImplemented, "cv::liveview::Server::url is not implemented in the skeleton module");
}

String Server::channelUrl(const String&, Transport) const
{
    CV_Error(Error::StsNotImplemented, "cv::liveview::Server::channelUrl is not implemented in the skeleton module");
}

void Server::publish(const String&, InputArray)
{
    CV_Error(Error::StsNotImplemented, "cv::liveview::Server::publish is not implemented in the skeleton module");
}

std::vector<ChannelInfo> Server::channels() const
{
    return std::vector<ChannelInfo>();
}

Ptr<Server> createServer(const String& host, int port, bool enableWebRTC)
{
    return makePtr<Server>(host, port, enableWebRTC);
}

} // namespace liveview
} // namespace cv
