#if defined(HAVE_LIVEVIEW_HTTP_BOOST)

#include "web_backend.hpp"

#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

namespace cv {
namespace liveview {
namespace {

namespace asio = boost::asio;
namespace http = boost::beast::http;
using tcp = asio::ip::tcp;

class BeastResponse : public WebResponse
{
public:
    explicit BeastResponse(tcp::socket& socket) : socket_(socket) {}
    void setStatus(int status) CV_OVERRIDE { status_ = status; }
    void setHeader(const String& key, const String& value) CV_OVERRIDE
    {
        headers_.push_back(std::make_pair(key, value));
    }
    bool write(const void* data, size_t size) CV_OVERRIDE
    {
        body_.append(static_cast<const char*>(data), size);
        return true;
    }
    void send()
    {
        http::response<http::string_body> res(static_cast<http::status>(status_), 11);
        for (size_t i = 0; i < headers_.size(); ++i)
            res.set(headers_[i].first.c_str(), headers_[i].second.c_str());
        res.body() = body_;
        res.prepare_payload();
        boost::beast::error_code ec;
        http::write(socket_, res, ec);
    }

private:
    tcp::socket& socket_;
    int status_ = 200;
    std::vector<std::pair<String, String> > headers_;
    std::string body_;
};

class BoostWebBackend : public WebBackend
{
public:
    BoostWebBackend() : io_(), acceptor_(io_), running_(false), boundPort_(0) {}
    ~BoostWebBackend() CV_OVERRIDE { stop(); }

    void start(const String& host, int port, const WebHandler& handler) CV_OVERRIDE
    {
        if (running_)
            return;
        host_ = host.empty() ? "127.0.0.1" : host;
        handler_ = handler;
        tcp::endpoint endpoint(asio::ip::make_address(host_.c_str()), static_cast<unsigned short>(port));
        acceptor_.open(endpoint.protocol());
        acceptor_.set_option(asio::socket_base::reuse_address(true));
        acceptor_.bind(endpoint);
        acceptor_.listen();
        acceptor_.non_blocking(true);
        boundPort_ = acceptor_.local_endpoint().port();
        running_ = true;
        thread_ = std::thread(&BoostWebBackend::acceptLoop, this);
    }

    void stop() CV_OVERRIDE
    {
        if (!running_)
            return;
        running_ = false;
        boost::system::error_code ec;
        acceptor_.close(ec);
        io_.stop();
        if (thread_.joinable())
            thread_.join();
        joinClientThreads();
        boundPort_ = 0;
    }

    bool isRunning() const CV_OVERRIDE { return running_; }
    String host() const CV_OVERRIDE { return host_; }
    int port() const CV_OVERRIDE { return boundPort_; }
    String name() const CV_OVERRIDE { return "BOOST"; }

private:
    void acceptLoop()
    {
        while (running_)
        {
            boost::system::error_code ec;
            tcp::socket socket(io_);
            acceptor_.accept(socket, ec);
            if (!ec)
            {
                std::lock_guard<std::mutex> lock(clientsMutex_);
                clientThreads_.push_back(std::thread(&BoostWebBackend::handleClient, this, std::move(socket)));
            }
            else if (ec == asio::error::would_block || ec == asio::error::try_again)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }

    void handleClient(tcp::socket socket)
    {
        boost::beast::flat_buffer buffer;
        http::request<http::string_body> req;
        boost::beast::error_code ec;
        http::read(socket, buffer, req, ec);
        if (ec)
            return;
        WebRequest request;
        request.method = req.method_string().to_string().c_str();
        request.path = req.target().to_string().c_str();
        request.body = req.body().c_str();
        BeastResponse response(socket);
        if (handler_)
            handler_(request, response);
        response.send();
        socket.shutdown(tcp::socket::shutdown_both, ec);
    }

    void joinClientThreads()
    {
        std::vector<std::thread> threads;
        {
            std::lock_guard<std::mutex> lock(clientsMutex_);
            threads.swap(clientThreads_);
        }
        for (size_t i = 0; i < threads.size(); ++i)
        {
            if (threads[i].joinable())
                threads[i].join();
        }
    }

    String host_;
    asio::io_context io_;
    tcp::acceptor acceptor_;
    WebHandler handler_;
    std::atomic<bool> running_;
    int boundPort_;
    std::thread thread_;
    std::mutex clientsMutex_;
    std::vector<std::thread> clientThreads_;
};

} // namespace

Ptr<WebBackend> createBoostWebBackend()
{
    return makePtr<BoostWebBackend>();
}

} // namespace liveview
} // namespace cv

#endif // HAVE_LIVEVIEW_HTTP_BOOST
