#ifndef OPENCV_LIVEVIEW_WEB_BACKEND_HPP
#define OPENCV_LIVEVIEW_WEB_BACKEND_HPP

#include "opencv2/core.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace cv {
namespace liveview {

struct WebRequest
{
    String method;
    String path;
    String body;
};

class WebResponse
{
public:
    virtual ~WebResponse() {}
    virtual void setStatus(int status) = 0;
    virtual void setHeader(const String& key, const String& value) = 0;
    virtual bool startStream() = 0;
    virtual bool write(const void* data, size_t size) = 0;
    bool writeString(const String& text) { return write(text.data(), text.size()); }
};

typedef std::function<void(const WebRequest&, WebResponse&)> WebHandler;

class WebBackend
{
public:
    virtual ~WebBackend() {}
    virtual void start(const String& host, int port, const WebHandler& handler) = 0;
    virtual void stop() = 0;
    virtual bool isRunning() const = 0;
    virtual String host() const = 0;
    virtual int port() const = 0;
    virtual String name() const = 0;
};

Ptr<WebBackend> createWebBackend();
Ptr<WebBackend> createMongooseWebBackend();
Ptr<WebBackend> createCivetWebBackend();
Ptr<WebBackend> createBoostWebBackend();

} // namespace liveview
} // namespace cv

#endif // OPENCV_LIVEVIEW_WEB_BACKEND_HPP
