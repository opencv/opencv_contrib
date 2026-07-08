#include "route_matcher.hpp"

#include <iomanip>
#include <sstream>

namespace cv {
namespace liveview {

bool isValidChannelName(const String& name)
{
    if (name.empty() || name.size() > 128)
        return false;
    if (name == "." || name == ".." || name.find("..") != String::npos)
        return false;
    for (size_t i = 0; i < name.size(); ++i)
    {
        const unsigned char c = static_cast<unsigned char>(name[i]);
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                        c == '_' || c == '-' || c == '.';
        if (!ok)
            return false;
    }
    return true;
}

static bool extractChannel(const String& path, const String& prefix, const String& suffix, String& name)
{
    if (path.find(prefix) != 0)
        return false;
    if (path.size() <= prefix.size() + suffix.size())
        return false;
    if (!suffix.empty() && path.rfind(suffix) != path.size() - suffix.size())
        return false;
    name = path.substr(prefix.size(), path.size() - prefix.size() - suffix.size());
    return isValidChannelName(name);
}

RouteMatch matchRoute(const String& method, const String& path)
{
    RouteMatch out;
    out.kind = RouteKind::Unknown;
    const size_t query = path.find('?');
    const String cleanPath = query == String::npos ? path : path.substr(0, query);
    if (method == "GET")
    {
        if (cleanPath == "/")
            out.kind = RouteKind::Index;
        else if (cleanPath == "/healthz")
            out.kind = RouteKind::Health;
        else if (cleanPath == "/channels.json")
            out.kind = RouteKind::ChannelsJson;
        else if (extractChannel(cleanPath, "/frame/", ".jpg", out.channel))
            out.kind = RouteKind::Snapshot;
        else if (extractChannel(cleanPath, "/stream/", ".mjpeg", out.channel))
            out.kind = RouteKind::Mjpeg;
        else if (extractChannel(cleanPath, "/webrtc/", "", out.channel))
            out.kind = RouteKind::WebRtcViewer;
    }
    else if (method == "POST")
    {
        if (extractChannel(cleanPath, "/webrtc/", "/offer", out.channel))
            out.kind = RouteKind::WebRtcOffer;
        else if (extractChannel(cleanPath, "/webrtc/session/", "/candidate", out.channel))
            out.kind = RouteKind::WebRtcCandidate;
        else if (extractChannel(cleanPath, "/webrtc/session/", "/close", out.channel))
            out.kind = RouteKind::WebRtcClose;
    }
    return out;
}

String typeToString(int type)
{
    const int depth = CV_MAT_DEPTH(type);
    const int channels = CV_MAT_CN(type);
    const char* depthName = "CV_USRTYPE";
    switch (depth)
    {
    case CV_8U: depthName = "CV_8U"; break;
    case CV_8S: depthName = "CV_8S"; break;
    case CV_16U: depthName = "CV_16U"; break;
    case CV_16S: depthName = "CV_16S"; break;
    case CV_32S: depthName = "CV_32S"; break;
    case CV_32F: depthName = "CV_32F"; break;
    case CV_64F: depthName = "CV_64F"; break;
    }
    std::ostringstream os;
    os << depthName << "C" << channels;
    return os.str();
}

String jsonEscape(const String& value)
{
    std::ostringstream os;
    for (size_t i = 0; i < value.size(); ++i)
    {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        switch (c)
        {
        case '\\': os << "\\\\"; break;
        case '"': os << "\\\""; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default:
            if (c < 0x20)
                os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
            else
                os << value[i];
        }
    }
    return os.str();
}

String htmlEscape(const String& value)
{
    std::ostringstream os;
    for (size_t i = 0; i < value.size(); ++i)
    {
        switch (value[i])
        {
        case '&': os << "&amp;"; break;
        case '<': os << "&lt;"; break;
        case '>': os << "&gt;"; break;
        case '"': os << "&quot;"; break;
        case '\'': os << "&#39;"; break;
        default: os << value[i];
        }
    }
    return os.str();
}

} // namespace liveview
} // namespace cv
