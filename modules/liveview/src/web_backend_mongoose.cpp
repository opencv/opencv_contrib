#if defined(HAVE_LIVEVIEW_HTTP_MONGOOSE)

#include "web_backend.hpp"

#include "mjpeg_writer.hpp"

#include "mongoose.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

namespace cv {
namespace liveview {
namespace {

static String mgToString(const mg_str& s)
{
    return String(s.buf ? s.buf : "", s.len);
}

class MongooseResponse : public WebResponse
{
public:
    explicit MongooseResponse(mg_connection* c) : conn_(c) {}

    void setStatus(int status) CV_OVERRIDE { status_ = status; }
    void setHeader(const String& key, const String& value) CV_OVERRIDE
    {
        headers_ += key + ": " + value + "\r\n";
    }
    bool startStream() CV_OVERRIDE
    {
        if (sent_)
            return true;
        std::ostringstream os;
        os << "HTTP/1.1 " << status_ << " " << httpStatusText(status_) << "\r\n";
        os << headers_;
        os << "Connection: close\r\n\r\n";
        const String header = os.str();
        sent_ = rawSend(header.data(), header.size());
        return true;
    }
    bool write(const void* data, size_t size) CV_OVERRIDE
    {
        if (sent_)
            return rawSend(data, size);
        body_.insert(body_.end(), static_cast<const char*>(data), static_cast<const char*>(data) + size);
        return true;
    }
    void sendBuffered()
    {
        if (sent_)
            return;
        mg_printf(conn_, "HTTP/1.1 %d %s\r\n%sContent-Length: %lu\r\nConnection: close\r\n\r\n",
                  status_, httpStatusText(status_).c_str(), headers_.c_str(),
                  static_cast<unsigned long>(body_.size()));
        if (!body_.empty())
            mg_send(conn_, &body_[0], body_.size());
        conn_->is_draining = 1;
        sent_ = true;
    }

private:
    bool rawSend(const void* data, size_t size)
    {
        const char* ptr = static_cast<const char*>(data);
        size_t sent = 0;
        while (sent < size)
        {
            const long n = mg_io_send(conn_, ptr + sent, size - sent);
            if (n <= 0)
                return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    mg_connection* conn_;
    int status_ = 200;
    String headers_;
    std::vector<char> body_;
    bool sent_ = false;
};

class MongooseWebBackend : public WebBackend
{
public:
    MongooseWebBackend() : boundPort_(0), running_(false) {}
    ~MongooseWebBackend() CV_OVERRIDE { stop(); }

    void start(const String& host, int port, const WebHandler& handler) CV_OVERRIDE
    {
        if (running_)
            return;
        host_ = host.empty() ? "127.0.0.1" : host;
        handler_ = handler;
        mg_mgr_init(&mgr_);

        std::ostringstream url;
        url << "http://" << host_ << ":" << port;
        listener_ = mg_http_listen(&mgr_, url.str().c_str(), &MongooseWebBackend::onEvent, this);
        if (!listener_)
        {
            mg_mgr_free(&mgr_);
            CV_Error(Error::StsError, "LiveView failed to start Mongoose HTTP backend");
        }
        boundPort_ = mg_ntohs(listener_->loc.port);
        running_ = true;
        thread_ = std::thread(&MongooseWebBackend::pollLoop, this);
    }

    void stop() CV_OVERRIDE
    {
        if (!running_)
            return;
        running_ = false;
        if (thread_.joinable())
            thread_.join();
        mg_mgr_free(&mgr_);
        listener_ = NULL;
        boundPort_ = 0;
    }

    bool isRunning() const CV_OVERRIDE { return running_; }
    String host() const CV_OVERRIDE { return host_; }
    int port() const CV_OVERRIDE { return boundPort_; }
    String name() const CV_OVERRIDE { return "MONGOOSE"; }

private:
    static void onEvent(mg_connection* c, int ev, void* evData)
    {
        if (ev != MG_EV_HTTP_MSG)
            return;
        MongooseWebBackend* self = static_cast<MongooseWebBackend*>(c->fn_data);
        mg_http_message* hm = static_cast<mg_http_message*>(evData);
        WebRequest request;
        request.method = mgToString(hm->method);
        request.path = mgToString(hm->uri);
        if (hm->query.len)
            request.path += "?" + mgToString(hm->query);
        request.body = mgToString(hm->body);
        MongooseResponse response(c);
        if (self->handler_)
            self->handler_(request, response);
        response.sendBuffered();
    }

    void pollLoop()
    {
        while (running_)
            mg_mgr_poll(&mgr_, 20);
    }

    String host_;
    int boundPort_;
    mg_mgr mgr_;
    mg_connection* listener_ = NULL;
    WebHandler handler_;
    std::atomic<bool> running_;
    std::thread thread_;
};

} // namespace

Ptr<WebBackend> createMongooseWebBackend()
{
    return makePtr<MongooseWebBackend>();
}

} // namespace liveview
} // namespace cv

#endif // HAVE_LIVEVIEW_HTTP_MONGOOSE
