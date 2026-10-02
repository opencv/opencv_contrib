#include "mjpeg_writer.hpp"

#include <sstream>

namespace cv {
namespace liveview {

String httpStatusText(int status)
{
    switch (status)
    {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Internal Server Error";
    }
}

String httpHeader(int status, const String& contentType, size_t contentLength)
{
    std::ostringstream os;
    os << "HTTP/1.1 " << status << " " << httpStatusText(status) << "\r\n";
    os << "Content-Type: " << contentType << "\r\n";
    os << "Content-Length: " << contentLength << "\r\n";
    os << "Cache-Control: no-store\r\n";
    os << "Connection: close\r\n\r\n";
    return os.str();
}

String mjpegStreamHeader(const String& boundary)
{
    std::ostringstream os;
    os << "HTTP/1.1 200 OK\r\n";
    os << "Content-Type: multipart/x-mixed-replace; boundary=" << boundary << "\r\n";
    os << "Cache-Control: no-store\r\n";
    os << "Connection: close\r\n\r\n";
    return os.str();
}

String mjpegPartHeader(const String& boundary, size_t jpegSize)
{
    std::ostringstream os;
    os << "--" << boundary << "\r\n";
    os << "Content-Type: image/jpeg\r\n";
    os << "Content-Length: " << jpegSize << "\r\n\r\n";
    return os.str();
}

} // namespace liveview
} // namespace cv
