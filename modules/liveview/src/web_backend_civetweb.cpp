#if defined(HAVE_LIVEVIEW_HTTP_CIVETWEB)

#include "web_backend.hpp"

#include "mjpeg_writer.hpp"

#include <civetweb.h>

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace cv {
namespace liveview {
namespace {

class CivetResponse : public WebResponse
{
public:
    explicit CivetResponse(mg_connection* conn) : conn_(conn) {}
    void setStatus(int status) CV_OVERRIDE { status_ = status; }
    void setHeader(const String& key, const String& value) CV_OVERRIDE
    {
        headers_ += key + ": " + value + "\r\n";
    }
    bool startStream() CV_OVERRIDE
    {
        if (sent_)
            return true;
        mg_printf(conn_, "HTTP/1.1 %d %s\r\n%sConnection: close\r\n\r\n",
                  status_, httpStatusText(status_).c_str(), headers_.c_str());
        sent_ = true;
        return true;
    }
    bool write(const void* data, size_t size) CV_OVERRIDE
    {
        if (sent_)
            return mg_write(conn_, data, size) == static_cast<int>(size);
        body_.append(static_cast<const char*>(data), size);
        return true;
    }
    void send()
    {
        if (sent_)
            return;
        mg_printf(conn_, "HTTP/1.1 %d %s\r\n%sContent-Length: %lu\r\nConnection: close\r\n\r\n",
                  status_, httpStatusText(status_).c_str(), headers_.c_str(),
                  static_cast<unsigned long>(body_.size()));
        if (!body_.empty())
            mg_write(conn_, body_.data(), body_.size());
        mg_close_connection(conn_);
    }

private:
    mg_connection* conn_;
    int status_ = 200;
    String headers_;
    std::string body_;
    bool sent_ = false;
};

class CivetWebBackend : public WebBackend
{
public:
    CivetWebBackend() : ctx_(NULL), boundPort_(0), running_(false) {}
    ~CivetWebBackend() CV_OVERRIDE { stop(); }

    void start(const String& host, int port, const WebHandler& handler) CV_OVERRIDE
    {
        if (running_)
            return;
        host_ = host.empty() ? "127.0.0.1" : host;
        handler_ = handler;
        std::ostringstream ports;
        ports << host_ << ":" << port;
        const char* options[] = {
            "listening_ports", ports.str().c_str(),
            "num_threads", "4",
            NULL
        };
        ctx_ = mg_start(NULL, this, options);
        if (!ctx_)
            CV_Error(Error::StsError, "LiveView failed to start CivetWeb backend");
        mg_set_request_handler(ctx_, "**", &CivetWebBackend::handle, this);
        mg_server_port portsOut[8];
        std::memset(portsOut, 0, sizeof(portsOut));
        int count = mg_get_server_ports(ctx_, 8, portsOut);
        boundPort_ = count > 0 ? portsOut[0].port : port;
        running_ = true;
    }

    void stop() CV_OVERRIDE
    {
        if (!running_)
            return;
        running_ = false;
        mg_stop(ctx_);
        ctx_ = NULL;
        boundPort_ = 0;
    }

    bool isRunning() const CV_OVERRIDE { return running_; }
    String host() const CV_OVERRIDE { return host_; }
    int port() const CV_OVERRIDE { return boundPort_; }
    String name() const CV_OVERRIDE { return "CIVETWEB"; }

private:
    static int handle(mg_connection* conn, void* cbdata)
    {
        CivetWebBackend* self = static_cast<CivetWebBackend*>(cbdata);
        const mg_request_info* info = mg_get_request_info(conn);
        WebRequest request;
        request.method = info && info->request_method ? info->request_method : "";
        request.path = info && info->request_uri ? info->request_uri : "";
        if (info && info->query_string)
            request.path += "?" + String(info->query_string);
        if (info && info->content_length > 0)
        {
            std::string body;
            body.resize(static_cast<size_t>(info->content_length));
            size_t offset = 0;
            while (offset < body.size())
            {
                const int n = mg_read(conn, &body[0] + offset, body.size() - offset);
                if (n <= 0)
                    break;
                offset += static_cast<size_t>(n);
            }
            body.resize(offset);
            request.body = body.c_str();
        }
        CivetResponse response(conn);
        if (self->handler_)
            self->handler_(request, response);
        response.send();
        return 1;
    }

    String host_;
    mg_context* ctx_;
    WebHandler handler_;
    int boundPort_;
    bool running_;
};

} // namespace

Ptr<WebBackend> createCivetWebBackend()
{
    return makePtr<CivetWebBackend>();
}

} // namespace liveview
} // namespace cv

#endif // HAVE_LIVEVIEW_HTTP_CIVETWEB
